%% s05_group_searchlight_tfce.m
% Perform group-level TFCE cluster statistics on MVPA searchlight maps.
%
% This script recursively searches `rootDir` for NIfTI files whose names
% contain `filterStr`. Subject identifiers are parsed from the file paths
% and divided into experts and novices, using the lists defined in
% `chess-mvpa/modules/__init__.py`.  For each group a one-sample
% Threshold-Free Cluster Enhancement (TFCE) test against chance is
% computed, and a two-sample test compares experts against novices,
% all using `cosmo_montecarlo_cluster_stat`.
%
% Unthresholded z-maps as well as maps thresholded at |z|>1.96 (p<=0.05
% two-tailed) are saved under `results/mvpa-second-level`, preserving the
% input folder structure.
%
% Requirements: CoSMoMVPA must be installed and on the MATLAB path.
%
% Example usage:
%   Set `rootDir` and `filterStr` below and simply run the script.
%
%% Configuration
rootDir   = '/path/to/searchlight/results';
filterStr = 'checkmate'; % text to match in filenames
niter     = 1000;        % number of permutations for TFCE

%% Determine expert and novice subject IDs from modules/__init__.py
initFile = fullfile('modules', '__init__.py');
txt      = fileread(initFile);
expTok   = regexp(txt, 'EXPERT_SUBJECTS\s*=\s*\(([^\)]*)\)', 'tokens', 'once');
novTok   = regexp(txt, 'NONEXPERT_SUBJECTS\s*=\s*\(([^\)]*)\)', 'tokens', 'once');

EXPERT_SUBJECTS  = regexp(expTok{1},  '(\d+)', 'match');
NONEXPERT_SUBJECTS = regexp(novTok{1}, '(\d+)', 'match');

%% Output directory
addpath(fullfile(pwd, 'matlab_helpers'));
resultsRoot = fullfile('results', 'mvpa-second-level');
if ~exist(resultsRoot, 'dir'); mkdir(resultsRoot); end
saveCurrentScript(resultsRoot);

%% Locate NIfTI files (including .nii.gz)
patternNii = fullfile(rootDir, '**', ['*' filterStr '*.nii']);
patternGz  = fullfile(rootDir, '**', ['*' filterStr '*.nii.gz']);
files = [dir(patternNii); dir(patternGz)];
if isempty(files)
    error('No NIfTI files matching %s found in %s', filterStr, rootDir);
end

% Determine chance level based on the first filename
h0_mean = inferChanceFromName(files(1).name);

%% Build datasets for each group
expert_ds_cell  = {}; expert_ids  = {};
novice_ds_cell  = {}; novice_ids  = {};
for k=1:numel(files)
    fpath = fullfile(files(k).folder, files(k).name);
    tok = regexp(fpath, 'sub-(\d+)', 'tokens');
    if isempty(tok); warning('Could not find subject ID in %s', fpath); continue; end
    sid = tok{1}{1};
    ds = cosmo_fmri_dataset(fpath);
    ds.sa.targets = 1; % all samples belong to the same condition
    if ismember(sid, EXPERT_SUBJECTS)
        ds.sa.chunks = numel(expert_ds_cell)+1;
        expert_ds_cell{end+1} = ds; %#ok<*AGROW>
        expert_ids{end+1} = sid;
    elseif ismember(sid, NONEXPERT_SUBJECTS)
        ds.sa.chunks = numel(novice_ds_cell)+1;
        novice_ds_cell{end+1} = ds;
        novice_ids{end+1} = sid;
    else
        warning('Subject %s not listed as expert or novice', sid);
    end
end

%% Stack datasets
if ~isempty(expert_ds_cell)
    expert_ds = cosmo_stack(expert_ds_cell);
    expert_ds.sa.targets = ones(numel(expert_ds_cell),1);
    expert_ds.sa.labels  = expert_ids';
else
    error('No expert datasets found');
end

if ~isempty(novice_ds_cell)
    novice_ds = cosmo_stack(novice_ds_cell);
    novice_ds.sa.targets = ones(numel(novice_ds_cell),1);
    novice_ds.sa.labels  = novice_ids';
else
    error('No novice datasets found');
end

%% Compute TFCE statistics for each group
cluster_nbrhood = cosmo_cluster_neighborhood(expert_ds);
expert_stat = cosmo_montecarlo_cluster_stat(expert_ds, cluster_nbrhood, ...
                    'niter', niter, 'h0_mean', h0_mean);
novice_stat = cosmo_montecarlo_cluster_stat(novice_ds, cluster_nbrhood, ...
                    'niter', niter, 'h0_mean', h0_mean);

% Combine groups for experts vs novices comparison
diff_ds = cosmo_stack({expert_ds, novice_ds});
diff_ds.sa.targets = [ones(numel(expert_ds_cell),1); 2*ones(numel(novice_ds_cell),1)];
diff_ds.sa.chunks  = (1:(numel(expert_ds_cell)+numel(novice_ds_cell)))';
diff_stat = cosmo_montecarlo_cluster_stat(diff_ds, cluster_nbrhood, ...
                    'niter', niter);

%% Determine relative output path based on the first input file
templatePath = fullfile(files(1).folder, files(1).name);
relPath = templatePath(numel(rootDir)+2:end); % keep subject subfolder
[outRel, baseName, ext] = fileparts(relPath);
if strcmp(ext, '.gz')
    [~, baseName] = fileparts(baseName); % remove .nii from .nii.gz
end

expDir = fullfile(resultsRoot, 'experts', outRel);
novDir = fullfile(resultsRoot, 'novices', outRel);
diffDir = fullfile(resultsRoot, 'experts_vs_novices', outRel);
if ~exist(expDir, 'dir'); mkdir(expDir); end
if ~exist(novDir, 'dir'); mkdir(novDir); end
if ~exist(diffDir, 'dir'); mkdir(diffDir); end

%% Save unthresholded maps
cosmo_map2fmri(expert_stat, fullfile(expDir, [baseName '_exp_tfce_z.nii']));
cosmo_map2fmri(novice_stat, fullfile(novDir, [baseName '_nov_tfce_z.nii']));
cosmo_map2fmri(diff_stat,   fullfile(diffDir, [baseName '_exp_vs_nov_tfce_z.nii']));

%% Threshold maps at p<=0.05 and save
z_thr = norminv(1 - 0.05/2);
expert_thr = expert_stat;
expert_thr.samples(abs(expert_thr.samples) < z_thr) = 0;
novice_thr = novice_stat;
novice_thr.samples(abs(novice_thr.samples) < z_thr) = 0;
cosmo_map2fmri(expert_thr, fullfile(expDir, [baseName '_exp_tfce_z_thr_p05.nii']));
cosmo_map2fmri(novice_thr, fullfile(novDir, [baseName '_nov_tfce_z_thr_p05.nii']));
diff_thr = diff_stat;
diff_thr.samples(abs(diff_thr.samples) < z_thr) = 0;
cosmo_map2fmri(diff_thr, fullfile(diffDir, [baseName '_exp_vs_nov_tfce_z_thr_p05.nii']));

%% -------------------------------------------------------------------------
function chance = inferChanceFromName(fname)
% Infer chance level from filename text according to dataset type
fname = lower(fname);
if contains(fname, 'decoding')
    if contains(fname, 'visual')
        c = 20;
    elseif contains(fname, 'check')
        c = 2;
    elseif contains(fname, 'strategy')
        c = 5;
    else
        error('Cannot determine number of classes from filename: %s', fname);
    end
    chance = 1 / c;
else
    chance = 0;
end
end
