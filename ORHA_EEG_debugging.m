function ORHA_EEG_debugging(subject_id, debug_mode)

%--------------------------------------------------------------------------
% This is the script to test/debug on your own computer (not connected to
% any EEG system)  
% EEG experiment for Object Representations in Healthy Aging (ORHA) project
%
% Original version: Johannes Singer
% Current version: Marleen Haupt
%
% Last update: 03.05.2024 
%
% Input:
%       subject_id: subject identifier (string in '')
%       debug_mode: 1 = debugging mode
%
% Command to enter in command window: ORHA_EEG('ORHA_999',1)
%
% If you want to run this code on a computer that is connected to an
% EEG system, please do not forget to search for all send_triggerIO64
% appearances and uncomment them (remove the %) to avoid error messages.
%--------------------------------------------------------------------------

%% Initial setup

% avoid PTB welcome screen
Screen('Preference', 'VisualDebugLevel', 1);

% clean up routine for button press devices
KbQueueRelease;

% textrenderer workaround 
Screen('Preference','TextRenderer', 0)

% set debug mode to zero if only participant id was given as input
if ~exist('debug_mode','var'), debug_mode = 0; end

% set experiment variables
exp                = load_settings;
exp.root_dir       = pwd;
exp.subject_id     = 'subject_id'; % subject ID
exp.modality       = 'eeg';
exp.debug_mode     = debug_mode;

%-------------------------------------------------------------------------
% Image size related settings
%
% Important: these settings have to be adapted to your setup
% 
% see: visualanglecalculation.m
%--------------------------------------------------------------------------

exp.pix_per_deg = 75.7893; %adapt this based on the visualanglecalculation script
exp.font_size = 20; %adapt this based on the visual needs of your participants

exp.image.size_pixels = round(exp.image.size_dva * exp.pix_per_deg);
exp.fixation.size_pixels = round(exp.fixation.size_dva * exp.pix_per_deg);

%-------------------------------------------------------------------------
% Trigger related settings
%
% Important: these settings have to be adapted to your setup
%
% see: help document
%-------------------------------------------------------------------------

% add path to the functions sending triggers to EEG
% the functions under ./IOport work for a general input/output port on a
% Windows 64bit operating system
addpath(fullfile(exp.root_dir,'IOport'))

% specify port address (see help document for more information)
address=hex2dec('3FE0'); 

% set trigger delay time (if required)
exp.trigger_delay_time = 0.01367;

% set initial condition number
condition=99;

%% Prompt keyboard to command window to prevent writing into experiment function
commandwindow
    
%% open PTB screen

disp('Opening Psychtoolbox')
if exp.debug_mode == 0 
    Screen('Preference', 'SkipSyncTests', 0); % run syn test
    [exp.on_screen, exp.screen_rect] = Screen('OpenWindow',exp.screen_num); % open screen
else
    Screen('Preference', 'SkipSyncTests', 1); % skip synctest in debug mode
    [exp.on_screen , exp.screen_rect] = Screen('OpenWindow', exp.screen_num, [],[1 1 500 500]); %open small screen
end

% Retreive the maximum priority number and set max priority
topPriorityLevel = MaxPriority(exp.on_screen);
Priority(topPriorityLevel);
%fprintf('Top priority level: %2.4fs\n',topPriorityLevel);

% Activate for alpha blending
Screen('BlendFunction', exp.on_screen, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
exp.window.width = exp.screen_rect(3); % width of screen (in pixels)
exp.window.height = exp.screen_rect(4); % height of screen (in pixels)
HideCursor % hide mouse cursor

disp('done')

% fill screen with background color
Screen('FillRect', exp.on_screen, exp.window.bg_color);

%% Go through all runs, for saving results treat each one as a separate experiment

% define the result directory
exp.results_dir = fullfile('results', subject_id, exp.modality);
if ~isdir(exp.results_dir)
    mkdir(exp.results_dir)
end

% define a backup result directory (if required)
exp.results_dir_backup = fullfile('results', subject_id, exp.modality, 'backup');
if ~isdir(exp.results_dir_backup)
    mkdir(exp.results_dir_backup)
end

% define a directory where the diary/log files are stored
exp.results_dir_diary = fullfile('results', subject_id, exp.modality, 'diary');
if ~isdir(exp.results_dir_diary)
    mkdir(exp.results_dir_diary)
end

%-------------------------------------------------------------------------
% Load condition order and event timing for all runs
%
% Important: the design file has to be created before executing this code
%-------------------------------------------------------------------------
[design, design_table] = load_experimental_design(exp);

for run_number = 1:8
 
 % start diary function for logging all information that appears in Matlab
 % command window
 % if you do not run diary and Matlab crashes/closes, you will not have
 % access to that information
 diary on
 diary(fullfile(exp.results_dir_diary,datestr(now,'yyyy-mm-dd_HH-MM-SS.txt')));
 
 % set filename for results
 resultsname = fullfile(exp.results_dir, [sprintf('run%02i_',run_number),exp.modality, '.mat']);
 exptime = datestr(now,'yyyy-mm-dd_HH-MM-SS');
 results.exptime = exptime;
 % backup just in case something is overwritten by accident
 resultsname_backup = fullfile(exp.results_dir_backup, [sprintf('run%02i_',run_number),exp.modality, exptime, '.mat']);
 
 exp.run_number = run_number; % run number
 results.run_number = run_number;
 
 trials = design.run(run_number).trial; % get current run trials
 
 disp(['Run duration is: ' num2str(design.run(run_number).duration) ' seconds']);
 
 %-------------------------------------------------------------------------
 % load run pictures and turn into textures
 %-------------------------------------------------------------------------
 disp('Loading images and converting to textures')
 [image_textures, fixation_texture] = load_images_to_textures(exp,trials);
 
 % between runs, draw wait screen
 draw_wait_screen(exp,run_number);
 
 % prepare recording of button presses
 KbQueueCreate; % ACTIVATED
 KbQueueStart; % ACTIVATED
 ListenChar; 
 
 %-------------------------------------------------------------------------
 % instruction screen
 %-------------------------------------------------------------------------
 disp(['Current run: ' num2str(run_number)]);
 draw_instruction_screen(exp,run_number,design.n_runs); % draw instruction on off-screen buffer
 Screen('Flip', exp.on_screen);
 
 %-------------------------------------------------------------------------
 % wait for trigger to begin run and set time 0 
 %-------------------------------------------------------------------------
 
 [~, keyCode] = KbQueueCheck; 
 disp('waiting for button press...');
 while sum(keyCode(exp.response_keys)) == 0
    [~,keyCode] = KbQueueCheck; 
	WaitSecs('Yieldsecs', 0.001); 
 end
 
 %log beginning time of experiment
 results.run_start = GetSecs; % 
 
 %send_triggerIO64(address, 100+run_number); % mark beginning of run in EEG
 %-------------------------------------------------------------------------
 % start counting time 
 %-------------------------------------------------------------------------

 draw_fixation_cross(exp,exp.colors(1,:)) %indicate color in position 2 and texture in position 3 (if required);
 Screen('Flip', exp.on_screen); % start time of the run
 
 %-------------------------------------------------------------------------
 % run through all trials
 %-------------------------------------------------------------------------
 
 ListenChar(2); % prevent key presses from appearing on screen
 
 for i_trial = 1:design.n_trials_total(run_number) % go through all trials

	 % reset button presses
	 all_key_id   = [];
	 all_key_time = [];
     all_TR_time = []; 
	 
	 %-------------------------------------------------------------------------
	 % draw image
	 %-------------------------------------------------------------------------
	 fprintf('*******************************************\n'); % new trial
	 fprintf('********** RUN %02i  TRIAL %04i  ************\n',run_number,i_trial); % new trial
	 fprintf('*******************************************\n'); % new trial
	 
	 fprintf('Picture number shown: %05i, i.e. image name: %s\n',trials(i_trial).image_nr, image_textures(i_trial).name);
     
     Screen('DrawTexture', exp.on_screen, image_textures(i_trial).tex);

     if exp.fixation.on_image
        draw_fixation_cross(exp,exp.colors(1,:)) %draw white fixation cross on image
     end
     
	 %calculate when to switch on the image
	 switch_time = results.run_start + trials(i_trial).onset;   % beginning of trial
	 
	 % Check for response before image on (excluding 2ms)
	 [key_id, key_time] = wait_and_get_keys(switch_time-0.002,[exp.quit_key exp.response_keys exp.scanner_trigger]);
         if any(key_id==exp.quit_key)
             disp('Quit key pressed.')
             cleanup(exp);
             return
         end
         
         if any(key_id == exp.scanner_trigger)
            all_TR_time = [all_TR_time key_time(key_id == exp.scanner_trigger)]; 
            key_time = key_time(key_id ~= exp.scanner_trigger);
            key_id = key_id(key_id ~= exp.scanner_trigger);
         end 
         
         if ismember(exp.response_keys, key_id)
             key_time = key_time(key_id ~= exp.scanner_trigger);
             key_id = key_id(key_id ~= exp.scanner_trigger);
         end
	 
	 % IMAGE ON
	 [imtime(1), est_onstime(1),fliptime(1)] = Screen('Flip', exp.on_screen,switch_time-0.001);  % switch image on
	 WaitSecs(exp.trigger_delay_time); % Wait Xms before sending the trigger
   	 %send_triggerIO64(address, trials(i_trial).image_nr);  % send trigger
   	 %1-192 for stimuli and 193-197 for catch stimuli
	 
	 fprintf('Image on at: %2.4fs\n',imtime(1)-results.run_start);
     fprintf('Estimated onset time: %2.4fs\n',est_onstime(1)-results.run_start);
     fprintf('Estimated time for flip execution: %2.4fs\n',imtime(1)-fliptime(1));

	 
	 % draw fixation cross and calculate when to turn image off
	 draw_fixation_cross(exp,exp.colors(1,:)) %draw white fixation cross
	 %calculate switch time to turn image off
	 switch_time = imtime(1) + design.image_duration;
	 
     % Collect all button presses before image off 
     [key_id, key_time] = wait_and_get_keys(switch_time-0.002, [exp.response_keys exp.quit_key exp.scanner_trigger]);
     
     if any(key_id==exp.quit_key)
        disp('Quit key pressed.')
        cleanup(exp);
        return
     end
         
	 if any(key_id == exp.scanner_trigger)
        all_TR_time = [all_TR_time key_time(key_id == exp.scanner_trigger)];
        key_time = key_time(key_id ~= exp.scanner_trigger);
        key_id = key_id(key_id ~= exp.scanner_trigger); 
     end 
         
	 if ismember(exp.response_keys, key_id)
        key_time = key_time(key_id ~= exp.scanner_trigger);
        key_id = key_id(key_id ~= exp.scanner_trigger);
     end
     
     all_key_id   = [all_key_id key_id];
     all_key_time = [all_key_time key_time];

	 
	 % IMAGE OFF
	 imtime(2) = Screen('Flip', exp.on_screen, switch_time-0.001); % switch image off
	 if exp.verbose == 1
		 fprintf('Image off at: %2.4fs\n',imtime(2)-results.run_start);
	 end
	 
     if i_trial == design.n_trials_total(run_number)
        switch_time = results.run_start + trials(i_trial).onset;
     else
        switch_time = results.run_start + trials(i_trial+1).onset; % end of trial
     end
	 
	 % Collect all button presses after image off (excluding last 50 ms of trial)
     [key_id, key_time] = wait_and_get_keys(switch_time-0.05, [exp.response_keys exp.quit_key exp.scanner_trigger]);
     if any(key_id==exp.quit_key)
             disp('Quit key pressed.')
             cleanup(exp);
             return
     end
         
         if any(key_id == exp.scanner_trigger)
             all_TR_time = [all_TR_time key_time(key_id == exp.scanner_trigger)];
             key_time = key_time(key_id ~= exp.scanner_trigger); 
             key_id = key_id(key_id ~= exp.scanner_trigger);
         end
         
         if ismember(exp.response_keys, key_id)
             key_time = key_time(key_id ~= exp.scanner_trigger);
             key_id = key_id(key_id ~= exp.scanner_trigger);
             %send_triggerIO64(address, 99);
         end

     all_key_id   = [all_key_id key_id];
	 all_key_time = [all_key_time key_time];

	 
	 %-------------------------------------------------------------------------
	 % log responses
	 %-------------------------------------------------------------------------
	 
	 % copy from design into results
	 if i_trial == 1, fieldn = fieldnames(trials(i_trial)); n_fieldn = length(fieldn); end
	 for i_fieldn = 1:n_fieldn, results.trial(i_trial).(fieldn{i_fieldn}) = trials(i_trial).(fieldn{i_fieldn}); end
	 results.trial(i_trial).image_on  = imtime(1)-results.run_start;
	 results.trial(i_trial).image_off = imtime(2)-results.run_start;
	 
	 results.trial(i_trial).responded      = false;
	 results.trial(i_trial).key_id         = NaN;
	 results.trial(i_trial).key_time       = key_time;
	 results.trial(i_trial).RT             = NaN;
     results.trial(i_trial).scanner_trigger= NaN;

	 button_pressed = ~isempty(all_key_id(all_key_id ~= exp.scanner_trigger));
	 
	 try % add a try-catch around to prevent any odd things from throwing an error
		 if button_pressed
			 results.trial(i_trial).responded  = true;
			 results.trial(i_trial).key_id     = all_key_id(all_key_id ~= exp.scanner_trigger); 
			 results.trial(i_trial).key_time   = all_key_time;
			 results.trial(i_trial).RT         = all_key_time - results.run_start - results.trial(i_trial).image_on;

         end
         
         if ~isempty(all_TR_time)
             results.trial(i_trial).scanner_trigger = all_TR_time(1) - results.run_start;
         end
	 catch
		 results.trial(i_trial).key_id = 'something weird happened.';
		 disp('something weird happened with recording button presses');
		 disp(['all_key_id = ' num2str(all_key_id)])
	 end
	 
	 if strcmp(trials(i_trial).trial_type,'catch')
		 if button_pressed
			 disp('BUTTON PRESSED, HIT.');
		 else
			 disp('BUTTON NOT PRESSED, MISS.');
		 end
	 else
		 if button_pressed
			 disp('BUTTON PRESSED, FALSE ALARM.')
		 end
		 % there should be a lot of correct rejections, so let's ignore those.
	 end
	 
	 % save results only into backup (if we notice early enough, the
	 % results file is not overwritten yet)
	 save(resultsname_backup,'results','exp','design')
	 
 end %end trial loop
 
 WaitSecs(design.final_wait);
 %send_triggerIO64(address, 199); %end trigger
 
 save(resultsname,'results','exp','design')
 save(resultsname_backup,'results','exp','design')

KbQueueStop; % ACTIVATED
 
%end diary logging 
diary off 

end %end run loop

cleanup(exp); % end Psychtoolbox, stop eyetracking, close port
get_performance(results); % get performance for last run again on screen
clear all;

end

%-------------------------------------------------------------------------
% HELPER FUNCTIONS
%-------------------------------------------------------------------------

function get_performance(results)
% get performance (add try catch around just in case)
try
    catch_trials = strcmp({results.trial.trial_type},'catch');
    noncatch_trials = ~strcmp({results.trial.trial_type},'catch');
    hit_rate = 100*mean([results.trial(catch_trials).responded]);
    fa_rate = 100*mean([results.trial(noncatch_trials).responded]);
    fprintf('Hit rate: %.2f%%\n',hit_rate)
    fprintf('False Alarm rate (incl. long latency hits): %.2f%%\n',fa_rate)
catch
    disp('could not calculate performance online.')
end
end

function draw_fixation_cross(exp, color, fixation_texture)


rect = exp.fixation.size_pixels * [0 0 1 1];
[x, y] = RectCenter(exp.screen_rect);
rect_centered = CenterRectOnPointd(rect, x, y);

if nargin > 2
Screen('DrawTexture', exp.on_screen, fixation_texture.tex, [], rect_centered,[],[],exp.fixation_cross_alpha) %last input for globalAlpha -maybe adjust a bit more in the lab 

elseif nargin == 2
    
    xCoords = [-exp.fixation.size_pixels/2 exp.fixation.size_pixels/2 0 0]; yCoords = [0 0 -exp.fixation.size_pixels/2 exp.fixation.size_pixels/2]; allCoords = [xCoords; yCoords]; 

    Screen('DrawLines', exp.on_screen, allCoords,exp.fixation.line_width, color, [x y], 2); %draw white cross
end 
end


function [image_textures,fixation_texture] = load_images_to_textures(exp,trials)

% we get one texture for each trial (for simplicity)
    
n_trials = length(trials);

image_textures(n_trials,1) = struct('tex',[]);
    
for i_trial = 1:n_trials
    [im, ~, alpha] = imread(trials(i_trial).image_path);
    
    if ~isempty(alpha)
        im(:,:,4) = alpha;
    end 
    % keep in resizing option in case we need it
    im = imresize(im,[exp.image.size_pixels exp.image.size_pixels],'Method','lanczos3');
    
    image_textures(i_trial).tex = Screen('MakeTexture', exp.on_screen, im); % make texture
    [~,image_textures(i_trial).name] = fileparts(trials(i_trial).image_path);
end

[im,~,alpha] = imread('fixation.png');
im(:,:,4) = alpha;
im = imresize(im,[exp.fixation.size_pixels exp.fixation.size_pixels]);
% rather than resizing the image, we are going to load in the texture and present it in the correct size
fixation_texture.tex = Screen('MakeTexture', exp.on_screen, im);

end

function draw_instruction_screen(exp,run_number, runs)

% prepare settings for showing the red fixation cross
rect = exp.fixation.size_pixels * [0 0 1 1];
[x, y] = RectCenter(exp.screen_rect);
xCoords = [-exp.fixation.size_pixels/2 exp.fixation.size_pixels/2 0 0]; yCoords = [0 0 -exp.fixation.size_pixels/2 exp.fixation.size_pixels/2]; allCoords = [xCoords; yCoords];

% prepare text for instructions
text2 = ['You are now in block ' num2str(run_number) ' of ' num2str(runs) ' blocks in total.'];
text3 = 'Please fixate on the cross in the center of the screen throughout the block.';
text4 = 'Whenever you see the image of a smurf, please blink and press the space bar.';
text5 = 'Please press the space bar when you want to start the next block.';


if run_number == 1 % different text for first run
    text2 = '';
    text3 = '';
    text5 = 'Please press the space bar when you want to start the first block.';    
end 


Screen('TextSize', exp.on_screen, exp.font_size); % increase text size for visibility
DrawFormattedText(exp.on_screen, text2, 'center', (exp.window.height)*0.3, [255 255 255]);
DrawFormattedText(exp.on_screen, text3, 'center', (exp.window.height)*0.35,  [255 255 255]);
DrawFormattedText(exp.on_screen, text4, 'center', (exp.window.height)*0.4, [255 255 255]);
DrawFormattedText(exp.on_screen, text5, 'center', (exp.window.height)*0.6,  [255 255 255]);
end


function draw_wait_screen(exp, run_number)

if run_number == 1 % instructions for the beginning of the experiment        
	text1 = 'We will start the experiment in a moment.';
	text2 = 'Please fixate on the center of the screen.';      
else % instructions for rest of experiment
	text1= 'Well done!'; 
	text2 = 'You can take a short break now. Let us know if you need anything.';
	text3 = 'Please press the space bar when you are ready to continue.';
end 


Screen('TextSize', exp.on_screen, exp.font_size); % increase text size for visibility 
DrawFormattedText(exp.on_screen, text1,'center',(exp.window.height)*0.35,  [255 255 255]);
DrawFormattedText(exp.on_screen, text2,'center',(exp.window.height)*0.55,  [255 255 255]);
DrawFormattedText(exp.on_screen, text3,'center',(exp.window.height)*0.75,  [255 255 255]);

Screen('Flip', exp.on_screen);


% wait for pause key
waitforkey(inf,exp.pause_key)

end


function cleanup(exp)

KbQueueRelease; % release keyboard queue
sca; % end Psychtoolbox
close_port(exp); % close port
ListenChar(0); % re-enable key presses to screen

end

function [key_id, key_time] = wait_and_get_keys(until,key_codes)

% each button may only be pressed once per call

key_id = []; key_time = [];
while GetSecs < until
    [pressed, keymap] = KbQueueCheck; % ACTIVATED
    keys = keymap(key_codes);
    if pressed && any(keys)
        code = key_codes(find(keys,1,'first'));
        secs = keys(find(keys,1,'first')); % ACTIVATED
        if any(key_id == code), continue, end
        key_id    = [key_id code];
        key_time  = [key_time secs];
    end
    WaitSecs('Yieldsecs', 0.001); % ACTIVATED
end


end

function waitforkey(duration,key_codes)

t0 = GetSecs;

while GetSecs < t0+duration
    [pressed,~,keymap] = KbCheckM;
    keys = keymap(key_codes);
    if pressed && any(keys)
        break
    end
end

end


function close_port(exp)

clear io64
disp('Port cleared.')

IOPort('CloseAll')
disp('Port closed.')

end
