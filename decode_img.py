
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

def decode_img(data, subject, n_perms=10, n_pstrials=4, n_test=1):

    n_imgs, n_trials, n_chans, n_time = data.shape

    DA = np.full((n_imgs, n_imgs, n_time), np.nan)  # array to contain the resulting decoding accuracy
    for p in tqdm(range(n_perms)):

        pstrials = data[:,np.random.permutation(n_trials)]  #shuffle trials
        pstrials = np.reshape(pstrials, (n_imgs,n_pstrials,-1,n_chans,n_time))    # split to make sub average
        pstrials = np.mean(pstrials, axis=2) # average in pseudotrials

        n_folds = int(n_pstrials / n_test) # number of folds for crossvalidation
        ps_ixs = np.arange(n_pstrials)  # array with pseudotrials indices

        for cv in range(n_folds):

            test_ix = np.arange(n_test) + (cv * n_test) # index of test set
            train_ix = np.delete(ps_ixs.copy(), test_ix)    # index of train set

            ps_train = pstrials[:,train_ix,:,:] # subset train set
            ps_test = pstrials[:,test_ix,:,:]   # subset test set

            for cA in range(n_imgs):
                for cB in range(cA+1, n_imgs):
                    for t in range(n_time):

                        x_train = np.array((ps_train[cA,:,:,t], ps_train[cB,:,:,t]))    # concatenate conditions in train set
                        x_train = np.reshape(x_train, (len(train_ix)*2, n_chans))   # format

                        x_test = np.array((ps_test[cA,:,:,t], ps_test[cB,:,:,t]))   # concatenate conditions in train set
                        x_test = np.reshape(x_test, (len(test_ix)*2, n_chans))  # format

                        y_train = np.array([1]*len(train_ix) + [2]*len(train_ix))   # set labels for train set
                        y_test = np.array([1]*len(test_ix) + [2]*len(test_ix))  # set labels for test set

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
                'n_test': n_test
                }   # output dictionary

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

if __name__ == '__main__':

    if sys.platform == 'darwin':
        dirs = {
            "out": r"C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\results",
            "eeg": r"C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\EEG_data",
            "behav": r"C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\behav_data"
        }

    elif sys.platform == 'win32':
        dirs = {
            'out': r"C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\results",
            'eeg': r"C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\EEG_data",
            'behav': r"C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\behav_data"
        }
    else: 
        dirs = {
            'out': r'C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\results',
            'eeg': r'C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\EEG_data',
            'behav': r'C:\Users\victo\Desktop\MRes\25MResButler\250129pilot\behav_data'
        }

    participants = ['K01', 'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10']
    subject = participants[int(sys.argv[1])]

    print(f'>>Processing subject {subject}')

    data = get_eeg_data(subject, dirs)
    out_dict = decode_img(data['eeg'], subject)

    out_file = os.path.join(dirs['out'], f"dec_img_{subject}.pkl")
    dump_data(out_dict, out_file)
