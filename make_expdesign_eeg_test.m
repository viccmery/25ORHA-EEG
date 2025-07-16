function design = make_expdesign_eeg_test(subject_id , overwrite)

%--------------------------------------------------------------------------
% Input:
%   subject_id: subject identifier
%   overwrite (optional): 1 or 0 (default: 0)
%
% Example call: make_expdesign_eeg('ORHA_999')
%--------------------------------------------------------------------------

if ~exist('overwrite','var')
    overwrite = 0;
end

%% Settings

design.subject_id              = subject_id;
design.type                    = 'eeg';

% experiment
design.n_runs                  = 1;
design.n_reps_per_run          = 1;
design.n_cat                   = 16; % number of categories in the stimulus set
design.n_trials_exp            = design.n_cat * design.n_reps_per_run; % number of trials for each run
design.n_trials_catch          = round((design.n_cat * design.n_reps_per_run)*0.25); % catch trials where response is required


while mod(design.n_trials_catch,4) >0 
    design.n_trials_catch = design.n_trials_catch-1; 
end 

design.catch_positions_base    = [4 5 5 6];
design.final_wait              = 0;   % time to wait after last trial
design.initial_wait            = 1;   % time until first trial starts

% trial
design.trial_duration          = 1.2; 
design.jitters                 = [300 400]; 
design.image_duration          = 0.2; 


rng('shuffle') % sets random numbers to be truly random 
%% for now, we are going to run this without saving it in advance

 % if same across runs, replicate
if length(design.n_trials_exp) == 1
    design.n_trials_exp = repmat(design.n_trials_exp,1,design.n_runs);
end
if length(design.n_trials_catch) == 1
    design.n_trials_catch = repmat(design.n_trials_catch,1,design.n_runs);
end

design.n_trials_total = design.n_trials_exp + design.n_trials_catch;
% get image details
image_details = readtable('image_paths_eeg.csv','ReadVariableNames',1,'Delimiter',',');

for i_run=1:design.n_runs
% randomize per block
while 1 % make sure no repeats of same trial type
    image_order = [];
    for i_rep = 1:design.n_reps_per_run
        image_order = [image_order randperm(design.n_cat)];
    end
    if all(diff(image_order)~=0) % no repeats
        break
    end
end

design.trial_order = image_order;


% now get experimental images of current session
image_details_exp = image_details(strcmp(image_details.condition,'exp'),:);
image_details_catch = image_details(strcmp(image_details.condition,'catch'),:);

% now shuffle rows within each condition
image_details_exp = image_details_exp(randperm(size(image_details_exp,1)),:);
image_details_catch = image_details_catch(randperm(size(image_details_catch,1)),:);

% inter-trial-interval jittered
jitters = (randi(design.jitters, design.n_trials_total(i_run), 1)/1000);
jitters = round(jitters,2);

design.catch_positions = repmat(design.catch_positions_base,1,design.n_trials_catch(i_run)/length(design.catch_positions_base));
design.catch_positions = design.catch_positions(randperm(length(design.catch_positions))); % shuffle order
design.catch_positions_absolute = cumsum(design.catch_positions);

for i = 1:design.n_trials_catch(i_run)
   design.trial_order(design.catch_positions_absolute(i):end+1) = [0 design.trial_order(design.catch_positions_absolute(i):end)]; 
end
   
% set trial type order
trial_type = cell(design.n_trials_total(i_run),1);
trial_type(design.trial_order ~=0) = {'exp'};
trial_type(design.trial_order ==0 ) = {'catch'};
  
cumulative_time = design.initial_wait; % start off with adding the initial wait
    
    for i_trial = 1:design.n_trials_total(i_run)
        
        % get current trial type
        switch trial_type{i_trial}
            case 'exp'
                curr_trial = image_details_exp(find(image_details_exp.category_nr == design.trial_order(i_trial)),:);
                jitters(i_trial) = jitters(i_trial); 
            case 'catch'
                catchnumber = randperm(5,1);
                curr_trial = image_details_catch(catchnumber,:);
                jitters(i_trial) = jitters(i_trial)+.6; 
            otherwise
                error('trial type not defined.')
        end
        
        design.run(i_run,1).trial(i_trial,1).trial_type         = curr_trial.condition{1};       % exp, catch
        design.run(i_run,1).trial(i_trial,1).image_nr           = curr_trial.image_nr;           % image number of current image
        design.run(i_run,1).trial(i_trial,1).category_nr        = curr_trial.category_nr;        % category number of current image
        design.run(i_run,1).trial(i_trial,1).image_path         = curr_trial.image_path{1};
        design.run(i_run,1).trial(i_trial,1).onset              = cumulative_time;
        design.run(i_run,1).trial(i_trial,1).iti                = jitters(i_trial);
        cumulative_time = cumulative_time + design.image_duration + jitters(i_trial);
        
    end
    
    design.run(i_run,1).duration = cumulative_time + design.final_wait;
    
    disp(['Run ' num2str(i_run) ' completed.']);
end     



%% Now format for table

cnt = 0;
str = struct;
for i_run = 1:design.n_runs
    for i_trial = 1:design.n_trials_total(i_run)
        cnt = cnt+1;
        str(cnt,1).subject_id         = design.subject_id;
        str(cnt,1).type               = design.type;
        str(cnt,1).run_nr             = i_run;
        str(cnt,1).run_duration       = design.run(i_run).duration;
        str(cnt,1).trial_onset        = design.run(i_run).trial(i_trial).onset;
        str(cnt,1).trial_iti          = design.run(i_run,1).trial(i_trial,1).iti;    
        str(cnt,1).trial_nr           = i_trial;
        str(cnt,1).trial_type         = design.run(i_run).trial(i_trial).trial_type;
        str(cnt,1).image_nr           = design.run(i_run).trial(i_trial).image_nr;
        str(cnt,1).category_nr        = design.run(i_run).trial(i_trial).category_nr;
        str(cnt,1).image_path         = design.run(i_run).trial(i_trial).image_path; 
    end
end

design_table = struct2table(str);

%% Save design and design_table as mat file and table as csv

savepath = fullfile(pwd,'design',lower(design.type),subject_id);
if ~isdir(savepath), mkdir(savepath), end


fn = sprintf('%s_%s','design_eeg',subject_id);

savename = fullfile(savepath,[fn '.mat']);
if exist(savename,'file') && ~overwrite
    error('File %s already exists. Set overwrite = 1 to overwrite it or delete it manually.',savename)
end
save(savename,'design','design_table')

writetable(design_table,fullfile(savepath,[fn '.csv']))