
import os
import sys
import numpy as np
import pickle

import mne
import scipy.io
import pandas as pd
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def get_eeg_data(subject, dirs, n_runs=8):

    eeg_path = os.path.join(dirs['eeg'], subject)
    behav_path = os.path.join(dirs['behav'], subject)

    eeg = []
    labels = []
    runs = []

    for run in range(n_runs):

        mat_file = 'run{:02d}_eeg.mat'.format(run+1)
        mat_path = os.path.join(behav_path, mat_file)
        mat_data = scipy.io.loadmat(mat_path, simplify_cells=True)

        trial_df = pd.DataFrame(mat_data['results']['trial'])
        image_nr = trial_df['image_nr'].values

        eeg_filename = '{}_b{}.vhdr'.format(subject.lower(), run+1)
        eeg_file = os.path.join(eeg_path, eeg_filename)

        # Load EEG
        raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
        raw.filter(l_freq=0.1, h_freq=40)
        raw.resample(sfreq = 50)
        events, event_id = mne.events_from_annotations(raw)
        raw.pick_types(eeg=True)

        img_events = events[np.isin(events[:, 2],[4,8])]

        exp_mask = image_nr < 65
        try:
            events_oi = img_events[exp_mask]
        except IndexError:
            print('Error: no events found. Handle it')
            continue

        images_oi = image_nr[exp_mask]

        epochs = mne.Epochs(
            raw, events_oi, tmin=-0.1, tmax=1,
            baseline=(None, 0), preload=True, reject_by_annotation=False, reject=None  # Change accordingly for other ERPs
        )

        eeg_ = epochs.get_data()

        if eeg_.shape[0] != len(events_oi):
            print('Error: epochs were dropped. Handle it')
            break

        eeg.append(eeg_)
        labels.append(images_oi)
        runs.append(run)
        channels = channels if 'channels' in locals() else epochs.ch_names
        times = times if 'times' in locals() else epochs.times

    eeg = np.concatenate(eeg, axis=0)
    labels = np.concatenate(labels, axis=0)

    #univariate normalization
    n_trials, n_chans, n_time = eeg.shape
    bl_mean = np.array([[np.mean(eeg[x,y,:100]) for y in range(n_chans)] for x in range(n_trials)])
    bl_std = np.array([[np.std(eeg[x,y,:100]) for y in range(n_chans)] for x in range(n_trials)])
    eeg = np.array([[(eeg[x,y,:] - bl_mean[x,y]) / bl_std[x,y] for y in range(n_chans)] for x in range(n_trials)])

    labels_unique, labels_count = np.unique(labels, return_counts=True)
    n_imgs = len(labels_unique)
    n_reps = max(labels_count)
    _, n_chans, n_times = eeg.shape

    data = np.full((n_imgs, n_reps, n_chans, n_times), np.nan)
    for i,img in enumerate(labels_unique):
        im_mask = labels == img
        data[i] = eeg[im_mask]

    out_dict = {
                'eeg': data,
                'img': labels_unique,
                'chans': channels,
                'times': times,
                'subject': subject, 
                'runs' : runs
    }

    return(out_dict)



def decode_cat(data, subject, n_perms=50, n_pstrials=4, n_test=1):

    categories = {
        "faces": range(1, 17),  
        "animals": range(17, 33),  
        "places": range(33, 49),  
        "objects": range(49, 65)  
    }

    n_imgs, n_trials, n_chans, n_time = data['eeg'].shape
    eeg = np.mean(data['eeg'], axis=1)
    n_cat = len(categories)
    imgs_per_cat = int(n_imgs / n_cat) 

    eeg_cat = np.full((n_cat, imgs_per_cat, n_chans, n_time), np.nan)

    for i, (category, indices) in enumerate(categories.items()):
        mask = np.isin(data['img'], indices)
        eeg_cat[i] = eeg[mask]

    DA = np.full((n_cat, n_cat, n_time), np.nan)  
    for p in tqdm(range(n_perms)):

        pstrials = eeg_cat[:,np.random.permutation(imgs_per_cat)] 
        pstrials = np.reshape(pstrials, (n_cat, n_pstrials,-1, n_chans, n_time)) 
        pstrials = np.mean(pstrials, axis=2) 

        n_folds = int(n_pstrials / n_test) 
        ps_ixs = np.arange(n_pstrials)  

        for cv in range(n_folds):

            test_ix = np.arange(n_test) + (cv * n_test) 
            train_ix = np.delete(ps_ixs.copy(), test_ix)   

            ps_train = pstrials[:,train_ix,:,:] 
            ps_test = pstrials[:,test_ix,:,:]  

            for cA in range(n_cat):
                for cB in range(cA+1, n_cat):
                    for t in range(n_time):

                        x_train = np.array((ps_train[cA,:,:,t], ps_train[cB,:,:,t]))    
                        x_train = np.reshape(x_train, (len(train_ix)*2, n_chans))  

                        x_test = np.array((ps_test[cA,:,:,t], ps_test[cB,:,:,t]))   
                        x_test = np.reshape(x_test, (len(test_ix)*2, n_chans))  

                        y_train = np.array([1]*len(train_ix) + [2]*len(train_ix))   
                        y_test = np.array([1]*len(test_ix) + [2]*len(test_ix))  

                        classifier = LinearSVC(penalty = 'l2',
                                                loss = 'hinge',
                                                C = .5,
                                                fit_intercept = True,
                                                max_iter = 10000)    # initialize decoder

                        classifier.fit(x_train, y_train)    # train decoder

                        y_pred = classifier.predict(x_test) # predict labels in the test set
                        acc = accuracy_score(y_test, y_pred)    # assess accuracy of predictions

                        DA[cA,cB,t] = np.nansum(np.array((DA[cA,cB,t], acc)))   # add accuracy to results array

    DA = DA / (n_perms * n_folds)   # divide result array in the total number of iterations
    
    out_dict = {
                'DA': DA,
                'subject': subject,
                'n_perms': n_perms,
                'n_folds': n_folds,
                'n_test': n_test, 
                'categories': categories,
                'times': data['times']
                }   

    return(out_dict)


def load_data(file):
    print('loading file: ' + file)
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return(data)


def dump_data(data, filename):
    print('writing file: ' + filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def plot_results(dfile):
    import matplotlib.pyplot as plt
    #%matplotlib qt

    data = load_data(dfile)

    plt.plot(np.nanmean(data['DA'], axis=(0,1)))

def get_default_dirs():
    
    base_dir = Path(__file__).resolve().parent
    return {
        'eeg': base_dir / 'data' / 'EEG_data',
        'behav': base_dir / 'data' / 'behav_data',
        'out': base_dir / 'results'
    }
if __name__ == '__main__':
    dirs = get_default_dirs()

    participants = ['K01', 'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10']
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <participant_index>")
        sys.exit(1)

    subject = participants[int(sys.argv[1])]

    print(f'>>Processing subject {subject}')

    data = get_eeg_data(subject, dirs)
    out_dict = decode_cat(data, subject)

    out_file = os.path.join(dirs['out'], f"dec_cat_{subject}.pkl")
    dump_data(out_dict, out_file)
