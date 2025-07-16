function exp = load_settings

%% experimental variables
disp('setting up experimental variables')

KbName('UnifyKeyNames');

exp.name                    = 'ORHA';       % name of experiment
exp.verbose                 = 1;            % whether onsets of all sub-components of each trial are displayed in the command line
exp.screen_num              = 0;            % screen number where experiment should be displayed
exp.response_keys           = [KbName('b') KbName('space')]; %specify all the keys that should evoke a reponse
exp.quit_key                = KbName('escape');  % key to quit experiment
exp.pause_key               = KbName('space'); % key to press to end pause
exp.scanner_trigger         = KbName('q');

%% stimulus-specific settings

exp.window.bg_color           = [128 128 128]; %window background color - here grey
exp.image.size_dva            = 10;
exp.fixation.size_dva         = 1.5; 
exp.fixation.line_width       = 4; % line width of fixation cross
exp.fixation.on_image         = 1; % whether fixation should appear on top of image
exp.colors                    = [255 255 255; 255 0 0]; % colors for fixation cross - here: white and red
exp.fixation_cross_alpha  = 0.2; % transparency level of fixation cross - not used when displaying the fixation cross with DrawLines

