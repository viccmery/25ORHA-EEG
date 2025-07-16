%% create stimuli path table

modality = 'eeg';

% photo cell array

fp = './stim_eeg';
fntmp = [dir(fullfile(fp,'*.jpg')); dir(fullfile(fp,'*.png'))];
fn = {fntmp.name}';
fn = natsortfiles(fn);
fullfn = cellfun(@(x) fullfile(fp,x),fn,'uniformoutput',0);
fullfn(:,2) = {'exp'};
fullfn(:,3) = num2cell(1:length(fullfn));
fullfn(:,4) = num2cell(1:length(fullfn));

% catch items

fp_mem = './stim_eeg/catch';
fntmp_mem = [dir(fullfile(fp_mem,'*.jpg')); dir(fullfile(fp_mem,'*.png'))];
fn_mem = {fntmp_mem.name}';
fn_mem = natsortfiles(fn_mem);
fullfn_mem = cellfun(@(x) fullfile(fp_mem,x),fn_mem,'uniformoutput',0);
fullfn_mem(:,2) = {'catch'};
%fullfn_mem(end,2) = {'catch'};
fullfn_mem(:,3) = num2cell(192+1:192+length(fullfn_mem));
fullfn_mem(:,4) = num2cell(192+1:192+length(fullfn_mem));

% combine

fullfn_final = vertcat(fullfn, fullfn_mem);

% make table out of cells 

stim_table = cell2table(fullfn_final);

% name the variables w

stim_table.Properties.VariableNames{1} = 'image_path';
stim_table.Properties.VariableNames{2} = 'condition';
stim_table.Properties.VariableNames{3} = 'image_nr';
stim_table.Properties.VariableNames{4} = 'category_nr';

% save the table 
if strcmpi(modality, 'eeg') savename = 'image_paths_eeg.csv'; 
elseif strcmpi(modality,'meg') savename = 'image_paths_meg.csv';
end 
writetable(stim_table, savename)
