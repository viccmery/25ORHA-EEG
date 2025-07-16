function [design, design_table] = load_experimental_design(exp)

design_fname = fullfile('design',exp.modality, exp.subject_id,sprintf('design_%s_%s.mat',...
lower(exp.modality), exp.subject_id));
x = load(design_fname);
fprintf('Loaded design')

design = x.design;
design_table = x.design_table;
        