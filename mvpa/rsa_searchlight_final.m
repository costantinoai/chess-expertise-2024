%% ========================================================================
%  LIGHTWEIGHT SEARCHLIGHT SCRIPT FOR FMRI (RSA + OPTIONAL SVM) USING CoSMoMVPA
%
%  This script performs a whole‐brain searchlight representational similarity
%  analysis (RSA), and optionally a searchlight decoding (SVM), on fMRI data
%  (SPM‐based GLM outputs) using CoSMoMVPA. We focus on a single binary regressor
%  (“checkmate”) for decoding if requested, and three model RDMs for RSA:
%    1) checkmate         – binary indicator: checkmate (C) vs. no‐check (NC)
%    2) visual similarity – visual‐stimulus label (ignoring “(nomate)”)
%    3) strategy         – concatenated category codes
%
%  To run only RSA, set svmFlag = false. To include searchlight decoding,
%  set svmFlag = true (performs classification on "checkmate").
%
%  For each subject:
%    1) Load preprocessed SPM‐derived beta images into a CoSMoMVPA dataset
%    2) Remove any voxels with NaNs or zero variance
%    3) Parse labels to extract stimulus IDs and regressors
%    4) If svmFlag: run searchlight decoding on "checkmate"
%    5) Compute three 40×40 model RDMs (binary “similarity”)
%    6) Assign ds_full.sa.targets = stimuliVec, average across runs to get ds_avg
%    7) Run searchlight RSA for each model RDM
%    8) Save resulting NIfTIs under rsaSearchlightRoot/sub‐XX/
%
%  PARALLELIZATION:
%    – Uses CoSMoMVPA’s built‐in parallel support via opt.nproc = feature('numcores')
%
%  DEPENDENCIES:
%    – CoSMoMVPA on MATLAB path
%    – SPM (for cosmo_fmri_dataset and cosmo_map2fmri)
%
%  OUTPUTS:
%    <rsaSearchlightRoot>/sub‐XX/
%       • sub‐XX_searchlight_decoding_checkmate.nii.gz     [if svmFlag==true]
%       • sub‐XX_searchlight_checkmate.nii.gz
%       • sub‐XX_searchlight_visualSimilarity.nii.gz
%       • sub‐XX_searchlight_strategy.nii.gz
% ========================================================================

clear; clc;

%% --------------------- PATH AND SUBJECT SETUP ----------------------------

% Define root output directory for RSA results
rsaSearchlightRoot = fullfile('/data/projects/chess/data/BIDS/derivatives/mvpa_searchlight_removeuseless_smooth4');
if ~exist(rsaSearchlightRoot, 'dir')
    mkdir(rsaSearchlightRoot);
end

% Define input directory containing GLM results (one folder per subject)
glmPath = fullfile('/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM');
subDirs = findSubjectsFolders(glmPath, '*');

fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmPath);

%% ========================= MAIN SUBJECT LOOP ==============================

for subIdx = 1:numel(subDirs)
    % Get subject folder and name (e.g., 'sub-01')
    subFolder = subDirs(subIdx).name;
    subName   = subFolder;
    fprintf('[INFO] Processing subject: %s\n', subName);

    % Define path to subject’s GLM directory (SPM outputs)
    spmSubjDir = fullfile(glmPath, subName, 'exp');
    if ~exist(fullfile(spmSubjDir, 'SPM.mat'), 'file')
        error('[WARN] No SPM.mat found for %s. Skipping.\n', subName);
    end

    %% 1) LOAD AND VALIDATE FMRI DATASET ------------------------------------

    % Load dataset from SPM model (beta estimates)
    ds_full = cosmo_fmri_dataset(fullfile(spmSubjDir, 'SPM.mat'));
    ds_full = cosmo_remove_useless_data(ds_full);

    % Validate structure
    cosmo_check_dataset(ds_full);
    if isempty(ds_full.samples)
        error('[WARN] Dataset for %s is empty or invalid. Skipping.', subName);
    end

    %% 2) PARSE LABEL VECTORS FOR REGRESSORS --------------------------------
    % Create stimulus-specific regressors from condition labels:
    % - stimuliVec:       Unique integer per stimulus (1–40)
    % - checkmateVec:     Binary 1/2 depending on checkmate condition
    % - visualStimVec:    Integer label per visual configuration
    % - categoriesVec:    Strategy category (combined piece-category)

    [stimuliVec, checkmateVec, visualStimVec, categoriesVec] = ...
        parseLabelRegressors(ds_full.sa.labels);

    %% 3) COMPUTE MODEL RDMs (40×40) ----------------------------------------

    % RDMs based on binary similarity: 0 = same, 1 = different
    modelRDMs = struct();
    modelRDMs.checkmate      = computeRDM(checkmateVec(1:40),   'similarity');
    modelRDMs.visualStimuli  = computeRDM(visualStimVec(1:40),  'similarity');
    modelRDMs.categories     = computeRDM(categoriesVec(1:40),  'similarity');

    %% 4) AVERAGE PATTERNS ACROSS RUNS PER STIMULUS ------------------------

    % Assign targets (i.e., stimulus IDs)
    ds_full.sa.targets = stimuliVec(:);

    % Average voxel patterns across occurrences of each stimulus
    fprintf('[INFO] Averaging across runs to create one pattern per stimulus...\n');
    ds_avg = cosmo_fx(ds_full, @(x) mean(x, 1), 'targets');
    cosmo_check_dataset(ds_avg);

    % Output: ds_avg.samples has size [40 × nVoxels]
    % Each row = mean activation pattern for one unique stimulus

    %% 5) DEFINE SPHERICAL SEARCHLIGHT NEIGHBORHOOD ------------------------

    % IMPORTANT: Do this before removing any voxels, or neighborhood geometry breaks
    radius_rsa = 3;  % in voxel units
    nh_rsa = cosmo_spherical_neighborhood(ds_avg, 'radius', radius_rsa);

    % Report average number of voxels per sphere
    avgVoxCount = mean(cellfun(@numel, nh_rsa.neighbors));
    fprintf('[INFO] Defined RSA searchlight (radius=%d), avg voxels/sphere=%.1f\n\n', ...
        radius_rsa, avgVoxCount);


    %% 7) RUN RSA SEARCHLIGHT FOR EACH MODEL RDM ---------------------------

    % Map model field names to output-friendly suffixes
    suffixMap = containers.Map( ...
        {'checkmate', 'visualStimuli', 'categories'}, ...
        {'checkmate', 'visualSimilarity', 'strategy'} ...
    );

    % Set parallel processing options
    nCores = feature('numcores');
    optRSA = struct('nproc', nCores, 'progress', true);

    % Loop over each RDM and run searchlight RSA
    for key = {'checkmate', 'visualStimuli', 'categories'}
        regName = key{1};
        fprintf('[INFO] Running RSA for model RDM: %s\n', regName);

        % Define RSA measure arguments
        rsa_args = struct();
        rsa_args.target_dsm  = modelRDMs.(regName);  % Model RDM
        rsa_args.center_data = true;                 % Mean-center patterns before correlation

        % Run RSA searchlight (correlation between local neural RDM and model RDM)
        sl_rsa = cosmo_searchlight(ds_avg, nh_rsa, @cosmo_target_dsm_corr_measure, rsa_args, optRSA);

        % Define output path
        outDir_rsa = fullfile(rsaSearchlightRoot, subName);
        if ~exist(outDir_rsa, 'dir'), mkdir(outDir_rsa); end

        % Save NIfTI result
        suffix   = suffixMap(regName);
        outFile  = fullfile(outDir_rsa, sprintf('%s_searchlight_%s.nii.gz', subName, suffix));
        cosmo_map2fmri(sl_rsa, outFile);

        fprintf('[INFO]   Saved RSA map: %s\n\n', outFile);
    end

    fprintf('[INFO] Completed subject: %s\n============================================\n\n', subName);
end

fprintf('[INFO] All subjects done. RSA (and optional SVM) searchlight complete.\n');

%% ========================================================================
%  HELPER FUNCTIONS
% ========================================================================

function subDirs = findSubjectsFolders(glmRoot, pattern)
    % Get all directories matching the pattern
    d = dir(fullfile(glmRoot, pattern));
    
    % Filter: is a directory and name starts with "sub"
    isDir = [d.isdir] & startsWith({d.name}, 'sub');
    
    % Return only matching directories
    subDirs = d(isDir);
end


function [stimuliVec, checkmateVec, visualStimVec, categoriesVec] = parseLabelRegressors(labels)
% parseLabelRegressors Parses label strings to extract four vectors:
%   • stimuliVec    – integer code per unique stimulus (lowercase)
%   • checkmateVec  – numeric: 2 if 'C...', 1 if 'NC...'
%   • visualStimVec – integer code per unique stimulus ignoring "(nomate)"
%   • categoriesVec – integer code per concatenated category (strategy)
n = numel(labels);
stimuliLabels  = cell(n,1);
checkmateVec   = zeros(n,1);
categoriesCode = zeros(n,1);
rawVisual      = cell(n,1);

for i = 1:n
    lbl = labels{i};

    % (1) Checkmate: token '(C|NC)\d+'
    tok = regexp(lbl, '(?<=\s)(C|NC)\d+', 'match', 'once');
    if ~isempty(tok) && tok(1)=='C'
        checkmateVec(i) = 2;
    else
        checkmateVec(i) = 1;
    end

    % (2) Category code: e.g. 'C3' → use 'C3' as unique key
    catTok = regexp(lbl, '(?<=\s)(C|NC)(\d+)', 'tokens', 'once');
    concat = [catTok{1}, catTok{2}];
    categoriesCode(i) = str2double(catTok{2}) + (strcmpi(catTok{1},'C')*100);

    % (3) Stimulus string: substring between '_' and '*'
    stimTok = regexp(lbl, '(?<=_).*?(?=\*)', 'match', 'once');
    stimuliLabels{i} = lower(stimTok);

    % (4) Raw visual stimulus (remove '(nomate)')
    rawVisual{i} = erase(lower(stimTok), '(nomate)');
end

% Map stimuliLabels → stimuliVec (1:40)
[~, ~, ic]    = unique(stimuliLabels, 'stable');
stimuliVec    = ic;

% Map rawVisual → visualStimVec (1:numUnique)
[~, ~, iv]    = unique(rawVisual, 'stable');
visualStimVec = iv;

% Remap categoriesCode → dense integers
uniqueCats    = unique(categoriesCode, 'stable');
catMap        = containers.Map(num2cell(uniqueCats), 1:numel(uniqueCats));
categoriesVec = arrayfun(@(x) catMap(x), categoriesCode);
end

function RDM = computeRDM(vec, metric)
% computeRDM Creates an N×N dissimilarity matrix from vec
vec = vec(:);
switch lower(metric)
    case 'similarity'
        RDM = double(bsxfun(@ne, vec, vec'));
    otherwise
        error('Unsupported RDM metric: %s', metric);
end
end
