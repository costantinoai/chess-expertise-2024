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

% Toggle decoding (SVM) on/off
svmFlag = false;  % set to false to run only RSA

%% --------------------- PATH AND SUBJECT SETUP ----------------------------

rsaSearchlightRoot =  fullfile('/home/eik-tb/Desktop/mvpa_searchlight/group-results');
if ~exist(rsaSearchlightRoot, 'dir')
    mkdir(rsaSearchlightRoot);
end
glmPath =fullfile('/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM');

subDirs = findSubjectsFolders(glmPath, '*');
fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmPath);

%% ========================= MAIN SUBJECT LOOP ==============================
for subIdx = 1:numel(subDirs)
    subFolder = subDirs(subIdx).name;   % e.g., 'sub-01'
    subName   = subFolder;
    fprintf('[INFO] Processing subject: %s\n', subName);

    spmSubjDir = fullfile(glmPath, subName, 'exp');
    if ~exist(fullfile(spmSubjDir,'SPM.mat'),'file')
        warning('[WARN] No SPM.mat found for %s. Skipping.\n', subName);
        continue;
    end

%% 1) LOAD AND CLEAN FULL DATASET --------------------------------------
ds_full = cosmo_fmri_dataset(fullfile(spmSubjDir, 'SPM.mat'));
ds_full = cosmo_remove_useless_data(ds_full);  % drop NaNs/zero‐variance voxels
cosmo_check_dataset(ds_full);
if isempty(ds_full.samples)
    warning('[WARN] Empty/full‐invalid dataset for %s. Skipping.', subName);
    return;  % or continue if inside a larger loop
end

%% 2) PARSE LABELS FOR REGRESSORS ---------------------------------------
% Extract four vectors (length = nSamples):
%   • stimuliVec    – integer 1:40 per unique stimulus string
%   • checkmateVec  – 2 if label starts with 'C', 1 if 'NC'
%   • visualStimVec – integer per unique stimulus ignoring '(nomate)'
%   • categoriesVec – integer per concatenated category code
[stimuliVec, checkmateVec, visualStimVec, categoriesVec] = ...
    parseLabelRegressors(ds_full.sa.labels);

%% 3) OPTIONAL: SEARCHLIGHT DECODING FOR EACH REGRESSOR -----------------
if svmFlag
    % Put the target vectors in a cell array along with their names
    targetVectors = {checkmateVec, visualStimVec, categoriesVec};
    targetNames = {'checkmate', 'visualStim', 'categories'};

    % Set radius for spherical neighborhood (voxels)
    radius_svm = 3;

    % Get number of CPU cores for parallel processing
    nCores = feature('numcores') - 2;
    opt = struct('nproc', nCores, 'progress', true);

    % Turn on detailed warnings
    cosmo_warning('on');
    cosmo_warning('verbose', true);

    for t = 1:numel(targetVectors)
        fprintf('[INFO] Running decoding searchlight for "%s"...\n', targetNames{t});
        
        ds_clf = ds_full;
        ds_clf.sa.targets = targetVectors{t}(:);

        % Build spherical neighborhood on ds_clf
        nh_svm = cosmo_spherical_neighborhood(ds_clf, 'radius', radius_svm);

        % Create balanced n-fold partitions
        partitions = cosmo_nfold_partitioner(ds_clf);
        partitions = cosmo_balance_partitions(partitions, ds_clf, 'nmin', 1);

        % Prepare measure arguments for classification
        measureArgs = struct();
        measureArgs.classifier    = @cosmo_classify_svm;
        measureArgs.partitions    = partitions;

        % Run searchlight crossvalidation
        sl_clf = cosmo_searchlight(ds_clf, nh_svm, @cosmo_crossvalidation_measure, measureArgs, opt);

        % Save decoding map
        outDir_clf = fullfile(rsaSearchlightRoot, subName);
        if ~exist(outDir_clf, 'dir')
            mkdir(outDir_clf);
        end

        outFile_clf = fullfile(outDir_clf, sprintf('%s_searchlight_decoding_%s.nii.gz', subName, targetNames{t}));
        cosmo_map2fmri(sl_clf, outFile_clf);
        fprintf('[INFO]   Saved decoding map: %s\n\n', outFile_clf);
    end
end



    %% 4) BUILD MODEL RDMs (40×40) FOR RSA ----------------------------------
    % Using binary "similarity": 0 if same label, 1 if different
    modelRDMs = struct();
    modelRDMs.checkmate      = computeRDM(checkmateVec(1:40),   'similarity');
    modelRDMs.visualStimuli  = computeRDM(visualStimVec(1:40),  'similarity');
    modelRDMs.categories     = computeRDM(categoriesVec(1:40),  'similarity');

    %% 5) AVERAGE ACROSS RUNS TO GET ONE PATTERN PER STIMULUS ---------------
    ds_full.sa.targets = stimuliVec(:);
    fprintf('[INFO] Averaging across runs to create one pattern per stimulus...\n');
    ds_avg = cosmo_fx(ds_full, @(x) mean(x,1), 'targets');
    cosmo_check_dataset(ds_avg);
    % Now ds_avg.samples is [40 × nVoxels], ds_avg.sa.targets = 1:40

    %% 6) DEFINE RSA SEARCHLIGHT NEIGHBORHOOD ------------------------------
    radius_rsa   = 3;  % voxels
    nh_rsa       = cosmo_spherical_neighborhood(ds_avg, 'radius', radius_rsa);
    avgVoxCount  = mean(cellfun(@numel, nh_rsa.neighbors));
    fprintf('[INFO] Defined RSA searchlight (radius=%d), avg voxels/sphere=%.1f\n\n', ...
            radius_rsa, avgVoxCount);

    %% 7) RUN RSA SEARCHLIGHT FOR EACH MODEL RDM ----------------------------
    suffixMap = containers.Map( ...
        {'checkmate', 'visualStimuli', 'categories'}, ...
        {'checkmate', 'visualSimilarity', 'strategy'} ...
    );

    nCores = feature('numcores');
    optRSA = struct('nproc', nCores, 'progress', true);

    % for key = {'checkmate','visualStimuli','categories'}
    for key = {'visualStimuli'}
        regName = key{1};
        fprintf('[INFO] Running RSA for model RDM: %s\n', regName);

        rsa_args = struct();
        rsa_args.target_dsm  = modelRDMs.(regName);
        rsa_args.center_data = true;

        sl_rsa = cosmo_searchlight(ds_avg, nh_rsa, @cosmo_target_dsm_corr_measure, rsa_args, optRSA);

        outDir_rsa = fullfile(rsaSearchlightRoot, subName);
        if ~exist(outDir_rsa, 'dir'), mkdir(outDir_rsa); end

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
    d      = dir(fullfile(glmRoot, pattern));
    isDir  = [d.isdir] & ~ismember({d.name}, {'.','..'});
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
