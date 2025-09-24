%% ========================================================================
%
%  This script combines two analyses (SVM-based MVPA and RSA) on the
%  same fMRI dataset using CoSMoMVPA. The pipeline:
%    1) Loads subject data (SPM-based GLM results)
%    2) Prepares the dataset for MVPA/RSA
%    3) Performs SVM-based cross-validation classification
%    4) Performs RSA correlation analysis
%
%  Outputs:
%    - Tables summarizing classification accuracy (per ROI, per target)
%    - Tables summarizing RSA correlations (per ROI, per regressor RDM)
%
%
%  Dependencies:
%     - CoSMoMVPA
%     - SPM
%     - MATLAB
%     - Functions in this file or in your path:
%         * findSubjectsFolders
%         * gunzipNiftiFile (if needed)
%         * prepareDataset
%         * prepareHalfDataset
%         * createTargetRDMs
% ========================================================================

clear; clc;  
% Clears the workspace (removes variables) and command window 
% for a clean start.

%% ====================== PATH AND SUBJECT DEFINITIONS =====================
% In this section, we define the key directory paths, retrieve subject folders,
% and prepare output directories. We also define ROI paths and load them.

% Path to main 'derivatives' folder (contains preprocessed data, SPM outputs, etc.)
derivativesDir = '/data/projects/chess/data/BIDS/derivatives';

% SPM root folder within derivativesDir containing fmriprep + SPM outputs
spmRoot = fullfile(derivativesDir, 'fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked');

roisRoots = {
    '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/bilalic_sphere_rois'
    '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortices_bilateral'
    '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_regions_bilateral'
};

% Folder containing the ROIs. Typically a folder with one or more .nii files
for iRoi = 1:numel(roisRoots)

    roisRoot = roisRoots{iRoi};

    % Extract the last folder name from roisRoot. E.g., roisRoot might end in
    % ".../bilalic_sphere_rois", so roiFolderName = 'bilalic_sphere_rois'
    [~, roiFolderName] = fileparts(roisRoot);
    
    % Create a timestamp string (e.g., "20250402_102030")
    ts = createTimestamp;
    
    % Build the main output root folder:
    % <derivativesDir>/mvpa/<timestamp>_<roiFolderName>
    % This ensures our output is time-stamped and named after the ROI set.
    outRoot = fullfile(derivativesDir, 'mvpa-lda', strcat(ts, '_', roiFolderName));
    
    % Make sure the outRoot directory exists
    mkdir(outRoot);
    
    % We create two subfolders under outRoot: one for SVM results, one for RSA correlations
    outRootSVM     = fullfile(outRoot, 'svm');
    outRootRSACorr = fullfile(outRoot, 'rsa_corr');
    
    % Path to GLM results. We assume each subject's SPM outputs live under this directory.
    glmPath = fullfile(spmRoot, 'MNI', 'fmriprep-SPM-MNI', 'GLM');
    
    % A wildcard pattern or specific numeric array for selecting subjects.
    % E.g., '*' means "all available subjects".
    selectedSubjectsList = '*';
    
    % Use findSubjectsFolders to detect subject directories in glmPath 
    % that match the pattern in selectedSubjectsList.
    subDirs = findSubjectsFolders(glmPath, selectedSubjectsList);
    
    %% ============================ MAIN LOOP ==================================
    % We will iterate over each subject folder, prepare the data, 
    % and run SVM and RSA analyses in sequence. 
    % Additional logic for skipping or continuing is included if needed.
    
    % First, we store a copy of this script in the outRoot for reproducibility.
    % (assuming saveCurrentScript is a custom function that saves the .m file 
    % to the outRoot directory)
    saveCurrentScript(outRoot);
    
    % Prepare the ROI file pattern to look for NIfTI files within roisRoot
    roiFilePattern = fullfile(roisRoot, '*.nii');
    % retrieve the ROI file info using a helper function (e.g. prepareRoiFile).
    % This should return a struct with folder/name fields for the ROI .nii
    roiPathStruct  = prepareRoiFile(roiFilePattern);
    
    % If no ROI file is found, raise an error (the analysis cannot proceed)
    if isempty(roiPathStruct)
        error('[WARN] No ROI file found. Skipping...');
    end
    
    % Construct the full path to the .nii ROI file
    roiPath = fullfile(roiPathStruct.folder, roiPathStruct.name);
    
    % Also look for a TSV file in the same directory as the ROI .nii
    tsvFiles = dir(fullfile(roiPathStruct.folder, '*.tsv'));
    
    % If no TSV is found, raise an error
    if isempty(tsvFiles)
        error('[WARN] No TSV file found in folder: %s', roiPathStruct.folder);
    end
    
    % Use the first TSV found (if multiple, adapt as needed)
    tsvFilePath = fullfile(tsvFiles(1).folder, tsvFiles(1).name);
    
    % Load the TSV into a table (e.g., containing region IDs, region names, etc.)
    roiTSV = readtable(tsvFilePath, 'FileType', 'text', 'Delimiter', '\t');
    
    % 2) Load ROI
    % Convert the ROI .nii into a CoSMoMVPA dataset 
    % (roi_ds.samples typically stores the voxel label IDs)
    roi_ds  = cosmo_fmri_dataset(roiPath);
    
    % Extract just the numeric array from roi_ds
    roi_data = roi_ds.samples;
    
    % Get unique nonzero labels from roi_data. 
    % These typically correspond to different ROIs.
    unique_labels_in_data = unique(roi_data(:));
    unique_labels_in_data = unique_labels_in_data(unique_labels_in_data ~= 0);
    
    % Extract region IDs and region names from the TSV
    region_ids   = roiTSV.region_id;
    region_names = roiTSV.region_name;
    
    % Sanity check: 
    % Ensure that the set of numeric ROI labels matches the region IDs from the TSV.
    assert(isequal(sort(unique_labels_in_data), sort(region_ids)), ...
        'Mismatch between ROI data labels and TSV region IDs');
    
    % Now we start looping over each subject folder we found in subDirs.
    %for subIdx = 1:length(subDirs)

    for subIdx = 1:length(subDirs)
        
        % Extract subject folder name (e.g., 'sub-01')
        subFolder = subDirs(subIdx).name;
        
        % subName is the same as the folder name, but you can rename if needed
        subName = subFolder;
        
        % Example: if subName = 'sub-01', we parse out the numeric portion "01"
        % This is optional if you need to manipulate subject IDs further
        subIDPart = strsplit(subName, '-');
        subID     = str2double(subIDPart{2});
        
        % Print progress for the user
        fprintf('[INFO] Processing subject folder: %s\n', subName);
    
        % Build path to the subject-specific SPM directory:
        spmSubjDir = fullfile(glmPath, subName, 'exp');
        
        % Create subject-specific output subfolders for SVM and RSA correlation
        outDirSVM      = fullfile(outRootSVM, subName);
        outDirRSACorr  = fullfile(outRootRSACorr, subName);
    
        % Ensure these output directories exist or are newly created
        if ~exist(outDirSVM, 'dir'), mkdir(outDirSVM); end
        if ~exist(outDirRSACorr, 'dir'), mkdir(outDirRSACorr); end
    
        % Prepare the CoSMoMVPA dataset from the subject's SPM results
        % (Assume prepareDataset is your custom function that loads in beta 
        %  images and organizes ds.samples, ds.sa, ds.fa, etc.)
        ds      = prepareDataset(spmSubjDir);
        % Similarly, ds_half is a subset (e.g., only checkmate trials)
        ds_half = prepareHalfDataset(spmSubjDir);
    
        % If the dataset is invalid or empty, skip this subject
        if isempty(ds) || isempty(ds.samples)
            error('[WARN] Invalid dataset for %s. Skipping...', subName);
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 0) CREATE REGRESSORS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This function produces the structure "regressors" with fields:
        %  regressors.<regName>.targets
        %  regressors.<regName>.unique_targets
        %  regressors.<regName>.rdm
        %  regressors.<regName>.dataset_name    (='ds' or ='ds_half')
        regressors = createTargetRDMs(ds, ds_half);

        % This below is in case we want to run analysis only on a subset of
        % regressors, by keeping only what we need
        % regressors = struct('check_n_half', regressors.check_n_half, 'stimuli_half', regressors.stimuli_half);
    
        fprintf('\n[INFO] Starting combined MVPA (SVM) + RSA correlation analysis...\n');
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 1) GATHER REGRESSORS AND ROIs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (A) Extract all regressor names from 'regressors' structure
        regressorNames = fieldnames(regressors);
        nRegressors    = numel(regressorNames);
    
        % (B) Gather information about ROIs (region_ids, region_names, etc.)
        nItems = numel(region_ids);
    
        fprintf('[INFO] Found %d regressors; will run analyses on %d ROIs.\n',...
            nRegressors, nItems);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 2) PREPARE RESULT STORAGE: SVM
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % We'll create a matrix for classification accuracies: rows=regressors, columns=ROIs
        results_svm_matrix = zeros(nRegressors, nItems);
    
        % Convert that matrix into a table and set the column names to region_names
        results_svm_table  = array2table(results_svm_matrix, 'VariableNames', region_names);
    
        % Also insert a column at the start listing regressor names
        results_svm_table = addvars(results_svm_table, regressorNames(:), ...
            'Before', 1, 'NewVariableNames','target');
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 3) PREPARE RESULT STORAGE: RSA CORRELATION
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Similarly, we'll create a matrix for RSA correlation values
        results_rsa_matrix = zeros(nRegressors, nItems);
    
        % Convert that matrix into a table, labeling columns by ROI
        results_rsa_table  = array2table(results_rsa_matrix, 'VariableNames', region_names);
    
        % Insert a column listing regressor names
        results_rsa_table = addvars(results_rsa_table, regressorNames(:), ...
            'Before', 1, 'NewVariableNames','target');
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 4) DEFINE MEASURES: SVM & RSA
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % (A) SVM classifier
        classifier = @cosmo_classify_svm;
    
        % (B) RSA measure
        rsa_measure = @cosmo_target_dsm_corr_measure;
        rsa_args    = struct('center_data', true);
        % Note: We will set rsa_args.target_dsm = regressors.(...).rdm later
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 5) NESTED LOOP: OVER REGRESSORS, THEN OVER ROIs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for rIdx = 1:nRegressors
    
            % (a) Obtain this regressor's name and its structure
            thisRegName = regressorNames{rIdx};
            thisReg     = regressors.(thisRegName);
    
            % (b) Determine which dataset to slice
            %     If .dataset_name=='ds', use the full ds
            %     If .dataset_name=='ds_half', use ds_half
            switch lower(thisReg.dataset_name)
                case 'ds'
                    dsLocal = ds;
                case 'ds_half'
                    dsLocal = ds_half;
                otherwise
                    error('Unknown dataset_name: %s', thisReg.dataset_name);
            end
    
            fprintf('\n[INFO] Regressor "%s" -> Using dataset "%s"\n',...
                thisRegName, thisReg.dataset_name);
    
            % (c) Loop over each ROI
            for colIdx = 1:nItems
    
                % Gather the ROI info
                this_region_id   = region_ids(colIdx);
                this_region_name = region_names{colIdx};
    
                fprintf('[INFO]   ROI #%d (%s)...\n', this_region_id, this_region_name);
    
                % (i) Build mask to select the voxels belonging to this ROI
                mask = ismember(roi_data, this_region_id);
                if ~any(mask)
                    fprintf('[WARN]   No voxels found for ROI %s. Skipping.\n', this_region_name);
                    continue;
                end
    
                % (ii) Slice dsLocal by the ROI mask
                ds_slice = cosmo_slice(dsLocal, mask, 2);
                ds_slice = cosmo_remove_useless_data(ds_slice);
                cosmo_check_dataset(ds_slice);
    
                % Ensure we have enough features
                if isempty(ds_slice.samples) || size(ds_slice.samples,2) < 6
                    fprintf('[WARN]   ROI %s has too few features. Skipping.\n', this_region_name);
                    continue;
                end
    
                % ================== SVM ANALYSIS ====================
                % (iii) Assign the classifier's target labels from the regressor's .targets
                ds_slice.sa.targets = thisReg.targets;
    
                % Check if we have at least two classes
                if numel(unique(ds_slice.sa.targets)) < 2
                    fprintf('[WARN]   Only one class in target "%s". Skipping SVM.\n', thisRegName);
                else
                    % (iv) Create n-fold partitions, balancing them
                    partitions = cosmo_nfold_partitioner(ds_slice);
                    partitions = cosmo_balance_partitions(partitions, ds_slice, 'nmin', 1);
    
                    % (v) Run cross-validation
                    [~, accuracy] = cosmo_crossvalidate(ds_slice, classifier, partitions);
    
                    % (vi) Store the SVM accuracy for this regressor & ROI
                    results_svm_matrix(rIdx, colIdx) = accuracy;
                end
    
                % ================== RSA CORRELATION ====================
                % For RSA, we want to average the data over "stimuli" or "stimuli_half".
                % (i) Build a separate slice for RSA
                ds_slice_rsa = ds_slice;
    
                % (ii) Depending on which dataset we are using ('ds' or 'ds_half'),
                %      we assign the correct "stimuli-based" target.
                %      That is either regressors.stimuli or regressors.stimuli_half
                %      (both created by createTargetRDMs).
    
                if strcmpi(thisReg.dataset_name, 'ds')
                    % For the full dataset, use the "stimuli" regressor
                    ds_slice_rsa.sa.targets = regressors.stimuli.targets;
                    % ^ This presumes you have a 'stimuli' field in regressors.
                else
                    % For ds_half, use the "stimuli_half" regressor
                    ds_slice_rsa.sa.targets = regressors.stimuli_half.targets;
                    % ^ This presumes you have a 'stimuli_half' field in regressors.
                end
    
                % (iii) Average the dataset by these stimulus-based targets
                %       This collapses data across repeated presentations of
                %       each stimulus, giving one sample per unique stimulus.
                ds_slice_rsa_averaged = cosmo_fx(ds_slice_rsa, @(x) mean(x,1), 'targets');
    
                % (iv) Prepare the correlation measure arguments
                rsa_args.target_dsm = thisReg.rdm;
                % ^ We want to correlate the regressor's RDM with the neural RDM
                %   derived from ds_slice_rsa_averaged.
    
                % (v) Run the correlation measure
                rsa_result = rsa_measure(ds_slice_rsa_averaged, rsa_args);
    
                if ~isempty(rsa_result.samples)
                    % If successful, we get a single correlation value
                    results_rsa_matrix(rIdx, colIdx) = rsa_result.samples;
                else
                    error('[WARN]   RSA measure returned empty for "%s" / ROI %s.\n',...
                        thisRegName, this_region_name);
                end

            end % ROI loop
    
        end % Regressor loop

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 6) STORE RESULTS INTO TABLES
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Fill in the numeric data (starting from column 2 onwards)
        results_svm_table{:,2:end} = results_svm_matrix;
        results_rsa_table{:,2:end} = results_rsa_matrix;
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 7) WRITE TABLES TO DISK
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~exist(outDirSVM,'dir'), mkdir(outDirSVM); end
        outFileSVM = fullfile(outDirSVM, 'mvpa_cv.tsv');
        writetable(results_svm_table, outFileSVM, 'FileType','text','Delimiter','\t');
        fprintf('\n[INFO] SVM results table saved: %s\n', outFileSVM);
    
        if ~exist(outDirRSACorr,'dir'), mkdir(outDirRSACorr); end
        outFileRSACorr = fullfile(outDirRSACorr, 'rsa_corr.tsv');
        writetable(results_rsa_table, outFileRSACorr, 'FileType','text','Delimiter','\t');
        fprintf('[INFO] Correlation-based RSA table saved: %s\n', outFileRSACorr);
    
        % ========================================================
    
        % Done with this subject
        fprintf('[INFO] Completed analyses for subject: %s\n\n', subName);
    end % ROIs folder loop

    clear ds dsLocal ds_half ds_slice ds_slice_rsa ds_slice_rsa_averaged regressors

end % END. All ROIs folders have been processed


%% ========================================================================
%   FUNCTION DEFINITIONS
% ========================================================================

function ds_checkmate = prepareHalfDataset(spmSubjDir)
% prepareDataset
%
% Loads and prepares an fMRI dataset (SPM-based) for MVPA or RSA using
% CoSMoMVPA.
%
% Usage:
%   ds = prepareDataset(spmSubjDir)
%
% Inputs:
%   spmSubjDir (string)
%       Directory containing the 'SPM.mat' file for the subject.
%
% Outputs:
%   ds (CoSMoMVPA dataset struct)
%       - .samples:   [nSamples x nFeatures] data matrix
%       - .sa:        Sample attributes (e.g., labels, targets)
%       - .fa:        Feature attributes (e.g., voxel indices)
%       - .a:         Additional dataset meta-info
%
% Example:
%   >> ds = prepareDataset('/path/to/sub-xx/exp')
%   >> cosmo_check_dataset(ds)

% Load dataset from the SPM.mat file
ds = cosmo_fmri_dataset(fullfile(spmSubjDir, 'SPM.mat'));

% Identify the items in sa.labels containing " NC"
mask = ~contains(ds.sa.labels, ' NC');

% Slice the dataset to exclude those items
ds_checkmate = cosmo_slice(ds, mask);

% Check dataset integrity
cosmo_check_dataset(ds_checkmate);

% Warn if too few features
if size(ds_checkmate.samples, 2) < 6
    warning('[WARN] Only %d features found. Analysis might be unreliable.', size(ds.samples, 2));
end
end

function ds = prepareDataset(spmSubjDir)
% prepareDataset
%
% Loads and prepares an fMRI dataset (SPM-based) for MVPA or RSA using
% CoSMoMVPA.
%
% Usage:
%   ds = prepareDataset(spmSubjDir)
%
% Inputs:
%   spmSubjDir (string)
%       Directory containing the 'SPM.mat' file for the subject.
%
% Outputs:
%   ds (CoSMoMVPA dataset struct)
%       - .samples:   [nSamples x nFeatures] data matrix
%       - .sa:        Sample attributes (e.g., labels, targets)
%       - .fa:        Feature attributes (e.g., voxel indices)
%       - .a:         Additional dataset meta-info
%
% Example:
%   >> ds = prepareDataset('/path/to/sub-xx/exp')
%   >> cosmo_check_dataset(ds)

% Load dataset from the SPM.mat file
ds = cosmo_fmri_dataset(fullfile(spmSubjDir, 'SPM.mat'));

% Check dataset integrity
cosmo_check_dataset(ds);

% Warn if too few features
if size(ds.samples, 2) < 6
    warning('[WARN] Only %d features found. Analysis might be unreliable.', size(ds.samples, 2));
end
end

function regressors = createTargetRDMs(ds, ds_half)
% createTargetRDMs - Computes targets and RDMs for full dataset (ds)
% and half dataset (ds_half), then stores them all in a single structure:
%
%   regressors.<regressorName>.targets
%   regressors.<regressorName>.unique_targets
%   regressors.<regressorName>.rdm
%   regressors.<regressorName>.dataset_name
%
% USAGE:
%   regressors = createTargetRDMs(ds, ds_half);
%
% EXAMPLE:
%   R = createTargetRDMs(myFullDs, myHalfDs);
%   imagesc(R.checkmate.rdm), colorbar;

%% ------------------------------------------------------------------------
% A. Prepare a single output structure
% -------------------------------------------------------------------------
regressors = struct();

%% ------------------------------------------------------------------------
% B. Parse label-based categorical regressors from ds (full dataset)
% -------------------------------------------------------------------------
[checkmate_full, categories_full, stimuli_full, visualStimuli_full] = ...
    parseLabelRegressors(ds);

% Numeric vectors for ds (40 items):
total_pieces_full = [16;17;18;16;19;18;24;26;15;22;19;20;13;23;26;17;...
    20;28;17;13;17;17;19;18;20;18;24;26;16;23;20;20;...
    13;23;26;18;21;28;17;13];

legal_moves_full  = [38;40;41;47;50;36;39;41;38;47;34;35;41;43;46;30;...
    45;48;33;35;37;39;40;42;47;36;39;41;37;47;34;35;...
    41;42;41;29;41;47;32;34];

%% ------------------------------------------------------------------------
% C. Add regressors for the FULL dataset (ds)
%    We'll replicate the base vectors to match all runs,
%    but the RDM is computed only from chunk==1 observations.
% -------------------------------------------------------------------------
addRegressor('checkmate',     checkmate_full,    'similarity', ds, 'ds');
addRegressor('categories',    categories_full,   'similarity', ds, 'ds');
addRegressor('stimuli',       stimuli_full,      'similarity', ds, 'ds');
addRegressor('visualStimuli', visualStimuli_full,'similarity', ds, 'ds');

addRegressor('total_pieces', repeatForAllFolds(total_pieces_full, ds), 'subtraction', ds, 'ds');
addRegressor('legal_moves',  repeatForAllFolds(legal_moves_full,  ds),  'subtraction', ds, 'ds');

%% ------------------------------------------------------------------------
% D. Parse label-based categorical regressors from ds_half (20 items)
% -------------------------------------------------------------------------
[~, categories_half, stimuli_half, ~] = ...
    parseLabelRegressors(ds_half);

% Hard-coded numeric/categorical vectors for the 20 items in ds_half:
total_pieces_half_base = [16;17;18;16;19;18;24;26;15;22;19;20;13;23;26;17;20;28;17;13];
legal_moves_half_base  = [38;40;41;47;50;36;39;41;38;47;34;35;41;43;46;30;45;48;33;35];
check_n_half_base = [3; 3; 4; 4; 4; 4; 4; 4; 4; 4; 4; 4; 4; 4; 4; 3; 4; 1; 1; 1];

side_half_base = [1;1;0;0;1;1;0;0;0;0;1;1;0;1;0;0;1;1;0;1];
motif_half_base = categorical(["deflection/pulling"; "defence removal"; ...
    "deflection/pulling"; "defence removal"; "deflection/pulling"; ...
    "other (overextension)"; "defence removal"; "other (overextension)"; ...
    "deflection/pulling"; "deflection/pulling"; "deflection/pulling"; ...
    "defence removal"; "deflection/pulling"; "defence removal"; ...
    "defence removal"; "other (straightforward checkmate)"; ...
    "deflection/pulling"; "other (straightforward checkmate)"; ...
    "other (straightforward checkmate)"; "other (straightforward checkmate)"]);

% first_piece_half_base = categorical(["rook"; "queen"; "queen"; "rook"; "queen"; "rook"; ...
%     "queen"; "knight"; "rook"; "queen"; "knight"; "bishop"; "knight"; "queen"; ...
%     "queen"; "rook"; "rook"; "queen"; "bishop"; "knight"]);
%
% checkmate_piece_half_base = categorical(["queen"; "queen"; "rook"; "queen"; "rook"; "queen"; ...
%     "rook"; "knight"; "knight"; "knight"; "knight"; "queen"; "rook"; "bishop"; ...
%     "bishop"; "bishop"; "queen"; "queen"; "bishop"; "knight"]);

%% ------------------------------------------------------------------------
% E. Add regressors for the HALF dataset (ds_half)
% -------------------------------------------------------------------------
addRegressor('categories_half',    categories_half,    'similarity', ds_half, 'ds_half');
addRegressor('stimuli_half',       stimuli_half,       'similarity', ds_half, 'ds_half');

addRegressor('total_pieces_half', repeatForAllFolds(total_pieces_half_base, ds_half), 'subtraction', ds_half, 'ds_half');
addRegressor('legal_moves_half',  repeatForAllFolds(legal_moves_half_base,  ds_half),  'subtraction', ds_half, 'ds_half');
addRegressor('check_n_half',  repeatForAllFolds(check_n_half_base,  ds_half),  'subtraction', ds_half, 'ds_half');

addRegressor('side_half',            repeatForAllFolds(side_half_base, ds_half),            'similarity', ds_half, 'ds_half');
addRegressor('motif_half',           repeatForAllFolds(double(motif_half_base), ds_half),   'similarity', ds_half, 'ds_half');
% addRegressor('first_piece_half',     repeatForAllFolds(double(first_piece_half_base), ds_half), 'similarity', ds_half, 'ds_half');
% addRegressor('checkmate_piece_half', repeatForAllFolds(double(checkmate_piece_half_base), ds_half), 'similarity', ds_half, 'ds_half');

fprintf('[INFO] All regressors successfully created.\n');

%% ========================================================================
%                       NESTED FUNCTIONS
%% ========================================================================

    function expandedVec = repeatForAllFolds(baseVec, dsLocal)
        % repeatForAllFolds - Repeats the given baseVec across the number
        % of runs (folds) present in dsLocal.sa.chunks.
        %
        % Example: If baseVec has 20 elements, and dsLocal has 2 runs,
        % expandedVec will have 40 elements (the 20 repeated twice).

        nRunsLocal = max(dsLocal.sa.chunks);
        expandedVec = repmat(baseVec(:), nRunsLocal, 1);
    end

    function [checkmateVec, categoriesVec, stimVec, visStimVec] = parseLabelRegressors(d)
        % parseLabelRegressors - Given a CoSMoMVPA dataset d, parse the
        % label strings to extract:
        %   1) checkmate    : numeric indicator (2 if 'C', 1 if 'NC')
        %   2) categories   : numeric codes
        %   3) stimuli      : numeric codes for unique lowercased strings
        %   4) visStimuli   : same as 'stimuli' but ignoring '(nomate)'

        labels = d.sa.labels(:);

        % (1) "C" or "NC" from label, e.g. "NC5"
        checkmateLabels = regexp(labels, '(?<=\s)(C|NC)\d+', 'match', 'once');
        % This yields 2 if 'C...' else 1
        checkmateVec = cellfun(@(x) strcmpi(x(1), 'C') + 1, checkmateLabels);

        % (2) "concatenated category" pattern, e.g. "NC5"
        catTokens  = regexp(labels, '(?<=\s)(C|NC)(\d+)', 'tokens', 'once');
        concatCats = cellfun(@(x) [x{1}, x{2}], catTokens, 'UniformOutput', false);
        uniqueCats = unique(concatCats, 'stable');
        catMap     = containers.Map(uniqueCats, 1:numel(uniqueCats));
        categoriesVec = cellfun(@(x) catMap(x), concatCats);

        % (3) Stimuli from substring after "_" and before "*"
        stimuliLabels = regexp(labels, '(?<=_).*?(?=\*)', 'match', 'once');
        lowerStim     = lower(stimuliLabels);
        [uStim, ~]    = unique(lowerStim, 'stable');
        stimMap       = containers.Map(uStim, 1:numel(uStim));
        stimVec       = cellfun(@(x) stimMap(x), lowerStim);

        % (4) Remove '(nomate)' from stimuli for a "visual" version
        cleanStim  = erase(lowerStim, '(nomate)');
        [uVis, ~]  = unique(cleanStim, 'stable');
        visStimMap = containers.Map(uVis, 1:numel(uVis));
        visStimVec = cellfun(@(x) visStimMap(x), cleanStim);
    end

    function addRegressor(name, fullVec, distMetric, dsLocal, dsLabel)
        % addRegressor - Adds a named regressor to the 'regressors' struct.
        %
        %  Storing:
        %    regressors.(name).targets         = repeated vector
        %    regressors.(name).unique_targets  = unique of that vector
        %    regressors.(name).rdm             = RDM of chunk==1 subset
        %    regressors.(name).dataset_name    = 'ds' or 'ds_half'
        %
        %  The RDM is computed using the chunk==1 subset from dsLocal:
        %    RDM(i,j) = 0/1 for 'similarity', or abs difference for 'subtraction'.

        % Store repeated vector (across all chunks)
        regressors.(name).targets = fullVec(:);

        % Also store the unique values of that vector
        regressors.(name).unique_targets = unique(fullVec(:));

        % Keep track of which dataset
        regressors.(name).dataset_name = dsLabel;

        % RDM is computed only on the chunk==1 subset
        foldMask   = (dsLocal.sa.chunks == 1);
        fold1_vec  = fullVec(foldMask);

        % Compute RDM
        RDM = computeRDM(fold1_vec(:), distMetric);

        regressors.(name).rdm = RDM;
    end

    function RDM = computeRDM(vec, metric)
        % computeRDM - Creates a representational dissimilarity matrix from
        % a vector using the given metric.
        %
        %   'similarity'  => RDM(i,j) = 0 if vec(i)==vec(j), else 1
        %   'subtraction' => RDM(i,j) = |vec(i) - vec(j)|

        vec = vec(:);
        switch lower(metric)
            case 'similarity'
                RDM = double(bsxfun(@ne, vec, vec'));
            case 'subtraction'
                RDM = abs(bsxfun(@minus, vec, vec'));
            otherwise
                error('Unknown metric: %s', metric);
        end
    end

end
