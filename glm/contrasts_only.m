%% estimate_contrasts_only.m
% This script loops over all subjects and tasks, loads each subject's SPM.mat,
% defines contrasts using the same logic as the original GLM script,
% and runs only the contrast estimation step (matlabbatch{3}).

clear; clc;

% ------------------------------------------------------------------------
% USER-DEFINED PATHS
% ------------------------------------------------------------------------
derivativesDir = '/data/projects/chess/data/BIDS/derivatives';
glmRoot = fullfile(derivativesDir, ...
    'fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked', ...
    'MNI', 'fmriprep-SPM-MNI', 'GLM');

% ------------------------------------------------------------------------
% CONTRAST DEFINITIONS (match those used in the original GLM script)
% ------------------------------------------------------------------------
selectedTasks(1).name = 'exp';
selectedTasks(1).contrasts = {'Check > No-Check', 'All > Rest'};
selectedTasks(1).weights(1) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', -1);
selectedTasks(1).weights(2) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', 1);

% ------------------------------------------------------------------------
% INITIALIZE SPM
% ------------------------------------------------------------------------
spm('defaults', 'FMRI');
spm_jobman('initcfg');

% ------------------------------------------------------------------------
% FIND SUBJECT DIRECTORIES
% ------------------------------------------------------------------------
dirInfo = dir(fullfile(glmRoot, 'sub-*'));
subjects = dirInfo([dirInfo.isdir]);

% ------------------------------------------------------------------------
% MAIN LOOP: FOR EACH SUBJECT & TASK, RUN CONTRAST ESTIMATION ONLY
% ------------------------------------------------------------------------
for iSub = 1:numel(subjects)
    subName = subjects(iSub).name;
    subjDir = fullfile(glmRoot, subName);
    SPMfile = fullfile(subjDir, 'exp', 'SPM.mat');

    if ~exist(SPMfile, 'file')
        warning('[WARN] SPM.mat not found for %s at %s', subName, SPMfile);
        continue;
    end

    fprintf('[INFO] Loading SPM model for %s...\n', subName);
    load(SPMfile, 'SPM');  % load SPM struct

    % Loop over each defined task (e.g. 'exp')
    for t = 1:numel(selectedTasks)
        contrasts = selectedTasks(t).contrasts;
        weightsStruct = selectedTasks(t).weights;

        % Prepare batch for contrasts step
        clear matlabbatch;
        matlabbatch{1}.spm.stats.con.spmmat = {SPMfile};

        for k = 1:numel(contrasts)
            % Compute design matrix weights via adjust_contrasts helper
            w = adjust_contrasts(SPMfile, weightsStruct(k));
            matlabbatch{1}.spm.stats.con.consess{k}.tcon.weights = w;
            matlabbatch{1}.spm.stats.con.consess{k}.tcon.name    = contrasts{k};
            matlabbatch{1}.spm.stats.con.consess{k}.tcon.sessrep = 'none';
        end

        % Run only contrast estimation step
        fprintf('[INFO] Estimating contrasts for %s - task %s...\n', subName, selectedTasks(t).name);
        spm_jobman('run', matlabbatch);
        fprintf('[INFO] Completed contrasts for %s - task %s.\n', subName, selectedTasks(t).name);
    end
end

fprintf('\n[INFO] All subjects and tasks processed. Contrast images estimated.\n');
