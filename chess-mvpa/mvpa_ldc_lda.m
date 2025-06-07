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
    %'/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/bilalic_sphere_rois'
    '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_regions_bilateral',
    '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortex_bilateral'
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
    parfor subIdx = 1:length(subDirs)
        
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
        classifier = @cosmo_classify_lda;
       
        % (B) RSA measure
        rsa_measure = @cosmo_crossnobis_dist_measure;
        rsa_args    = struct('center_data', false);
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
                    opt = struct();
                    opt.max_feature_count = 6000; % or any value > 5400
                    [~, accuracy] = cosmo_crossvalidate(ds_slice, classifier, partitions,  opt);
    
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
                % ds_slice_rsa_averaged = cosmo_fx(ds_slice_rsa, @(x) mean(x,1), 'targets');
    
                % (iv) Prepare the correlation measure arguments
                rsa_args.target_dsm = thisReg.rdm;
                % ^ We want to correlate the regressor's RDM with the neural RDM
                %   derived from ds_slice_rsa_averaged.
    
                % (v) Run the correlation measure
                rsa_result = rsa_measure(ds_slice_rsa, rsa_args);
    
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

first_piece_half_base = categorical(["rook"; "queen"; "queen"; "rook"; "queen"; "rook"; ...
    "queen"; "knight"; "rook"; "queen"; "knight"; "bishop"; "knight"; "queen"; ...
    "queen"; "rook"; "rook"; "queen"; "bishop"; "knight"]);

checkmate_piece_half_base = categorical(["queen"; "queen"; "rook"; "queen"; "rook"; "queen"; ...
    "rook"; "knight"; "knight"; "knight"; "knight"; "queen"; "rook"; "bishop"; ...
    "bishop"; "bishop"; "queen"; "queen"; "bishop"; "knight"]);

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
addRegressor('first_piece_half',     repeatForAllFolds(double(first_piece_half_base), ds_half), 'similarity', ds_half, 'ds_half');
addRegressor('checkmate_piece_half', repeatForAllFolds(double(checkmate_piece_half_base), ds_half), 'similarity', ds_half, 'ds_half');

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
% function ds_sa = cosmo_cv_dsm_measure(ds, varargin)
% % measure correlation with target dissimilarity matrix
% %
% % ds_sa = cosmo_target_dsm_corr_measure(dataset, args)
% %
% % Inputs:
% %   ds             dataset struct with field .samples PxQ for P samples and
% %                  Q features
% %   args           struct with fields:
% %     .target_dsm  (optional) Either:
% %                  - target dissimilarity matrix of size PxP. It should
% %                    have zeros on the diagonal and be symmetric.
% %                  - target dissimilarity vector of size Nx1, with
% %                    N=P*(P-1)/2 the number of pairs of samples in ds.
% %                  This option is mutually exclusive with the 'glm_dsm'
% %                  option.
% %     .metric      (optional) distance metric used in pdist to compute
% %                  pair-wise distances between samples in ds. It accepts
% %                  any metric supported by pdist (default: 'correlation')
% %     .type        (optional) type of correlation between target_dsm and
% %                  ds, one of 'Pearson' (default), 'Spearman', or
% %                  'Kendall'.
% %     .regress_dsm (optional) target dissimilarity matrix or vector (as
% %                  .target_dsm), or a cell with matrices or vectors, that
% %                  should be regressed out. If this option is provided then
% %                  the output is the partial correlation between the
% %                  pairwise distances between samples in ds and target_dsm,
% %                  after controlling for the effect of the matrix
% %                  (or matrices) in regress_dsm. (Using this option yields
% %                  similar behaviour as the Matlab function
% %                  'partial_corr')
% %     .glm_dsm     (optional) cell with model dissimilarity matrices or
% %                  vectors (as .target_dsm) for using a general linear
% %                  model to get regression coefficients for each element in
% %                  .glm_dsm. Both the input data and the dissimilarity
% %                  matrices are z-scored before estimating the regression
% %                  coefficients.
% %                  This option is required when 'target_dsm' is not
% %                  provided; it cannot cannot used together with
% %                  .target_dsm or regress_dsm.
% %                  When using this option, the 'type' option is ignored.
% %                  For this option, the output has as many rows (samples)
% %                  as there are elements (dissimilarity matrices) in
% %                  .glm_dsm.
% %     .center_data If set to true, then the mean of each feature (column in
% %                  ds.samples) is subtracted from each column prior to
% %                  computing the pairwise distances for all samples in ds.
% %                  Default: false
% %
% % Output:
% %    ds_sa         Dataset struct with fields:
% %      .samples    Scalar correlation value between the pair-wise
% %                  distances of the samples in ds and target_dsm; or
% %                  (when 'glm_dsms' is supplied) a column vector with
% %                  normalized beta coefficients. These values
% %                  are untransformed (e.g. there is no Fisher transform).
% %      .sa         Struct with field:
% %        .labels   {'rho'}; or (when 'glm_dsm' is supplied) a cell
% %                  {'beta1','beta2',...}.
% %
% % Examples:
% %     % generate synthetic dataset with 6 classes (conditions),
% %     % one sample per class
% %     ds=cosmo_synthetic_dataset('ntargets',6,'nchunks',1);
% %     %
% %     % create target dissimilarity matrix to test whether
% %     % - class 1 and 2 are similar (and different from classes 3-6)
% %     % - class 3 and 4 are similar (and different from classes 1,2,5,6)
% %     % - class 5 and 6 are similar (and different from classes 1-4)
% %     target_dsm=1-kron(eye(3),ones(2));
% %     %
% %     % show the target dissimilarity matrix
% %     cosmo_disp(target_dsm);
% %     > [ 0         0         1         1         1         1
% %     >   0         0         1         1         1         1
% %     >   1         1         0         0         1         1
% %     >   1         1         0         0         1         1
% %     >   1         1         1         1         0         0
% %     >   1         1         1         1         0         0 ]
% %     %
% %     % compute similarity between pairw-wise similarity of the
% %     % patterns in the dataset and the target dissimilarity matrix
% %     dcm_ds=cosmo_target_dsm_corr_measure(ds,'target_dsm',target_dsm);
% %     %
% %     % Pearson correlation is about 0.56
% %     cosmo_disp(dcm_ds)
% %     > .samples
% %     >   [ 0.562 ]
% %     > .sa
% %     >   .labels
% %     >     { 'rho' }
% %     >   .metric
% %     >     { 'correlation' }
% %     >   .type
% %     >     { 'Pearson' }
% %
% % Notes:
% %   - for group analysis, correlations can be fisher-transformed
% %     through:
% %       dcm_ds.samples=atanh(dcm_ds.samples)
% %   - it is recommended to set the 'center_data' to true when using
% %     the default 'correlation' metric, as this removes a main effect
% %     common to all samples; but note that this option is disabled by
% %     default due to historical reasons.
% %
% % #   For CoSMoMVPA's copyright information and license terms,   #
% % #   see the COPYING file distributed with CoSMoMVPA.           #
% 
% % process input arguments
% params=cosmo_structjoin('type','Pearson',... % set default
%     'metric','correlation',...
%     'center_data',false,...
%     varargin);
% 
% check_input(ds);
% check_params(params);
% 
% % - compute the pair-wise distance between all dataset samples using
% %   cosmo_pdist
% 
% samples=ds.samples;
% if params.center_data
%     samples=bsxfun(@minus,samples,mean(samples,1));
% end
% 
% % --------------------------------------------------------------------
% 
% 
% % MODIFIED BY MORITZ: use partitions (for cross RSA or leave-one-out RSA)
% if isfield(params,'partitions')
%     ds_corr = cosmo_correlation_measure(ds,'output','correlation','partitions',params.partitions); % fisher transformation by default
%     corrMat = real(cosmo_unflatten(ds_corr,1));
%     if isfield(params,'RSAz')
%         zMat=logical(tril(ones(8),0));  
%         ds_pdist = 1 - tanh(corrMat(zMat))'; % transform back to orig correlation values, convert to dissimilairty values        
%     else
%         corrMat(logical(eye(size(corrMat)))) = 0;
%         ds_pdist = 1 - tanh(squareform(corrMat)); % transform back to orig correlation values, convert to dissimilairty values
%     end
% else
%     ds_pdist = cosmo_pdist(ds.samples, params.metric)';
% end
% 
% 
% % --------------------------------------------------------------------
% 
% 
% 
% has_model_dsms=isfield(params,'glm_dsm');
% 
% if has_model_dsms
%     ds_sa=linear_regression_dsm(ds_pdist, params);
% else
%     ds_sa=correlation_dsm(ds_pdist,params);
% end
% 
% check_output(ds,ds_sa);
% end
% function check_output(input_ds,output_ds_sa)
% if any(isnan(output_ds_sa.samples))
%     if any(isnan(input_ds.samples(:)))
%         msg=['Input dataset has NaN values, which results in '...
%             'NaN values in the output. Consider masking the '...
%             'dataset to remove NaN values'];
%     elseif any(var(input_ds.samples)==0)
%         msg=['Input dataset has constant or infinite features, ',...
%             'which results in NaN values in the output. '...
%             'Consider masking the dataset to remove constant '...
%             'or non-finite features, for example using '...
%             'cosmo_remove_useless_data'];
%     else
%         msg=['Output has NaN values, even though the input does '...
%             'not. This can be due to the presence of constant '...
%             'features and/or non-finite values in the input, '...
%             'and/or target similarity structures with constant '...
%             'and/of non-finite data. When in doubt, please '...
%             'contact the CoSMoMVPA developers'];
%     end
%     cosmo_warning(msg);
% end
% 
% end

function ds_sa = cosmo_crossnobis_dist_measure(ds, varargin)
% COSMO_CROSSNOBIS_DIST_MEASURE
% Computes a cross-validated Mahalanobis (cross-nobis) RDM from multivariate
% activity patterns, then correlates it with a user-specified target DSM.
%
% This function is intended for Representational Similarity Analysis (RSA).
%
% INPUT:
%   ds         : CoSMoMVPA dataset with the following required fields:
%       ds.samples      [P × Q]  - data matrix (patterns × features)
%       ds.sa.targets   [P × 1]  - condition labels (e.g., stimuli)
%       ds.sa.chunks    [P × 1]  - run labels (used for cross-validation)
%
%   varargin   : Optional parameters (name-value pairs)
%       'target_dsm'      - reference DSM (vector or square matrix) [required]
%       'type'            - correlation type for RSA
%                            ('Pearson' | 'Spearman' | 'Kendall')
%                            [default: 'Spearman']
%
% OUTPUT:
%   ds_sa : Struct with RSA result:
%       ds_sa.samples    - [1 × 1] scalar correlation (rho)
%       ds_sa.sa.labels  - label name for the correlation
%       ds_sa.sa.metric  - distance metric used (here: 'crossnobis')
%       ds_sa.sa.type    - correlation type used

% -------------------------------------------------------------------------
% Parse and validate parameters
params = cosmo_structjoin('type','Spearman', varargin{:});
assert(isfield(params,'target_dsm'), 'Missing required input: ''target_dsm''');

% -------------------------------------------------------------------------
% Compute the cross-validated Mahalanobis RDM from data
RDM = compute_crossnobis_rdm(ds);              % [nStim × nStim] matrix

% -------------------------------------------------------------------------
% Compute correlation between empirical RDM and target DSM
rdm_vec = cosmo_squareform(RDM, 'tovector');
target_vec = cosmo_squareform(params.target_dsm, 'tovector');

rho = cosmo_corr(rdm_vec(:), target_vec(:), params.type);

% -------------------------------------------------------------------------
% Package RSA result into a CoSMo-compatible output structure
ds_sa             = struct();
ds_sa.samples     = rho;                     % scalar correlation
ds_sa.sa.labels   = {'rho'};                % output label
ds_sa.sa.metric   = {'crossnobis'};         % distance metric used
ds_sa.sa.type     = {params.type};          % RSA correlation type
end

function RDM = compute_crossnobis_rdm(ds)
% COMPUTE_CROSSNOBIS_RDM
% -------------------------------------------------------------------------
% Cross-validated Mahalanobis (Crossnobis) RDM Computation
%
% This code block computes a cross-validated Representational Dissimilarity
% Matrix (RDM) using the crossnobis distance metric. It is intended for
% Representational Similarity Analysis (RSA) on multivariate neural or
% model activation patterns.
%
% -------------------------------------------------------------------------
% INPUT:
%   ds : CoSMoMVPA dataset with the following required fields:
%        - ds.samples      [P × Q] matrix
%            Rows are pattern observations (e.g., trials, conditions).
%            Columns are features (e.g., voxels, channels, units).
%
%        - ds.sa.targets   [P × 1] vector
%            Condition/stimulus labels for each sample.
%
%        - ds.sa.chunks    [P × 1] vector
%            Run/session labels used for cross-validation folds.
%            Each unique chunk forms a test set in leave-one-out CV.
%
% -------------------------------------------------------------------------
% OUTPUT (within the calling function):
%   RDM : [nStim × nStim] symmetric dissimilarity matrix
%         Contains average cross-validated Mahalanobis distances between
%         all pairs of stimulus conditions.
%
% -------------------------------------------------------------------------
% METHOD DESCRIPTION:
% This implementation follows the standard protocol for computing the
% crossnobis (cross-validated Mahalanobis) distance:
%
% 1. Leave-one-run-out cross-validation is applied using the chunk labels.
% 2. For each fold:
%    a. Compute mean activation vectors per stimulus in train and test sets.
%    b. Estimate noise covariance from residuals in the training set:
%         residual = pattern - stimulus mean
%       Use Ledoit–Wolf shrinkage for stability.
%    c. Invert the noise covariance to obtain a precision matrix.
%    d. For each pair of stimuli (i, j), compute:
%
%         d_ij = Δ_testᵀ * Σ⁻¹ * Δ_train
%
%       where Δ is the difference vector between stimulus means.
%    e. Accumulate distances across folds.
%
% 3. Average across folds and symmetrize the result:
%         RDM(i,j) = mean distance across folds
%         RDM(j,i) = RDM(i,j)
%
% -------------------------------------------------------------------------
% NOTES:
% - The crossnobis distance is unbiased under the null (mean zero), unlike
%   Euclidean or correlation distances.
% - Values may be negative — this is expected due to cross-validation.
% - The use of Ledoit–Wolf shrinkage ensures well-conditioned covariance
%   estimates even when p ≫ n.
% - Output RDM is suitable for correlation-based RSA against theoretical
%   or behavioral DSMs.
%
% -------------------------------------------------------------------------
% REFERENCE:
%   Walther et al. (2016). Reliability of dissimilarity measures for
%   multi-voxel pattern analysis. NeuroImage.
%
%   Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional
%   covariance matrices. J. Multivariate Analysis.
%
% -------------------------------------------------------------------------
%
% INPUT:
%   ds : CoSMoMVPA dataset with .samples, .sa.targets, .sa.chunks
%
% OUTPUT:
%   RDM : [nStim × nStim] crossnobis rdm (symmetric)

% -------------------------------------------------------------------------
% Get stimulus conditions and feature dimensionality
targets     = unique(ds.sa.targets);           % unique conditions
nStim       = numel(targets);                  % number of unique stimuli
nFeat       = size(ds.samples,2);              % number of features (voxels, etc.)

% Partition data using leave-one-run-out cross-validation
partitions  = cosmo_nfold_partitioner(ds);     % chunk-wise CV splits
nFolds      = numel(partitions);               % number of folds = #chunks

% Initialize RDM
RDM = zeros(nStim);                            % accumulator

% -------------------------------------------------------------------------
% Iterate over folds and compute cross-validated distances
for f = 1:nFolds
    % Training and test split for fold f
    tr_idx = partitions.train_indices{f};
    te_idx = partitions.test_indices{f};

    ds_tr = cosmo_slice(ds, tr_idx);           % training data
    ds_te = cosmo_slice(ds, te_idx);           % test data

    % Compute mean activation patterns per stimulus for train/test
    M_tr = compute_means(ds_tr, targets);      % [nStim × nFeat]
    M_te = compute_means(ds_te, targets);      % [nStim × nFeat]

    % ---------------------------------------------------------------------
    % Estimate noise covariance from residuals (training data only)
    % Residuals = pattern - class mean for that pattern
    [~, tr_lab] = ismember(ds_tr.sa.targets, targets); % indices in 1:nStim
    resid       = ds_tr.samples - M_tr(tr_lab,:);      % [nTr × nFeat]
    Sigma       = ledoit_wolf_cov(resid);              % shrinkage covariance
    Pinv        = pinv(Sigma);                         % precision matrix (inverse)

    % ---------------------------------------------------------------------
    % Compute pairwise cross-validated distances
    for i = 1:nStim-1
        for j = i+1:nStim
            delta_tr = M_tr(i,:) - M_tr(j,:);          % train difference
            delta_te = M_te(i,:) - M_te(j,:);          % test difference

            % Compute crossnobis distance (symmetric dot product)
            d = delta_te * Pinv * delta_tr';           % cross-validated Mahalanobis

            RDM(i,j) = RDM(i,j) + d;                   % accumulate across folds
        end
    end
end

% -------------------------------------------------------------------------
% Average distances across folds
RDM = RDM / nFolds;

% Symmetrize by copying upper triangle to lower
RDM = RDM + RDM';   % symmetric: RDM(i,j) = RDM(j,i)

end


% -------------------------------------------------------------------------
function M = compute_means(ds, target_levels)
% Returns an [nStim × nFeat] matrix of stimulus means.
nStim = numel(target_levels);
nFeat = size(ds.samples,2);
M     = zeros(nStim, nFeat);
for k = 1:nStim
    idx = ds.sa.targets == target_levels(k);
    M(k,:) = mean(ds.samples(idx,:), 1);
end
end

% -------------------------------------------------------------------------
function Sigma = ledoit_wolf_cov(X)
% LEDOIT_WOLF_COV  Computes a shrinkage estimator of the covariance matrix.
%
%   Sigma = ledoit_wolf_cov(X)
%
%   This function implements the Ledoit–Wolf shrinkage estimator for the
%   covariance matrix, which provides a regularized, better-conditioned
%   estimate of the true covariance matrix, especially in high-dimensional
%   settings (p >> n).
%
%   INPUT:
%       X : [n × p] data matrix
%           Each row is an observation, each column is a variable (feature).
%           The function assumes that the noise structure is represented by
%           X (e.g., residuals from condition means).
%
%   OUTPUT:
%       Sigma : [p × p] shrinkage covariance matrix estimate
%           The estimate is a convex combination of:
%               - The empirical sample covariance matrix S
%               - A shrinkage target T = mu * I (scaled identity)
%
%   The formula is:
%       Sigma = (1 - delta) * S + delta * T
%   where delta ∈ [0,1] is the optimal shrinkage intensity.
%
%   Reference:
%       Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional
%       covariance matrices. Journal of Multivariate Analysis.

% -------------------------------------------------------------------------
% Step 1: Dimensions
[n, p] = size(X);  % n = number of samples, p = number of variables/features

% -------------------------------------------------------------------------
% Step 2: Center each column (feature-wise mean subtraction)
%         This ensures that the sample covariance is unbiased.
X = bsxfun(@minus, X, mean(X,1));  % zero-mean per feature (column-wise)

% -------------------------------------------------------------------------
% Step 3: Compute the empirical (sample) covariance matrix
%         Formula: S = (1/n) * (X' * X)
S = (X' * X) / n;  % [p × p] covariance matrix

% -------------------------------------------------------------------------
% Step 4: Define the shrinkage target: mu * I
%         The target is a scaled identity matrix with the same average
%         variance as the diagonal of the sample covariance matrix.
mu = trace(S) / p;         % average variance across features
T  = mu * eye(p);          % target matrix (scaled identity)

% -------------------------------------------------------------------------
% Step 5: Estimate φ (phi)
%         φ measures the variance of the individual covariance estimates.
%         It is the average squared Frobenius norm of deviations:
%             φ = (1/n) ∑₁ⁿ ‖xᵢxᵢᵗ - S‖²_F
%         This reflects how much individual outer products deviate from S.
phi = 0;
for i = 1:n
    xi   = X(i,:)';                      % [p × 1] column vector for sample i
    Si   = xi * xi';                     % [p × p] outer product: xᵢxᵢᵗ
    diff = Si - S;                       % deviation from sample covariance
    phi  = phi + norm(diff, 'fro')^2;   % squared Frobenius norm
end
phi = phi / n;  % average over all samples

% -------------------------------------------------------------------------
% Step 6: Compute ρ (rho)
%         ρ is the squared Frobenius distance between the sample covariance
%         and the shrinkage target:
%             ρ = ‖S - T‖²_F
rho = norm(S - T, 'fro')^2;

% -------------------------------------------------------------------------
% Step 7: Compute the optimal shrinkage intensity delta
%         Ensures delta ∈ [0,1] to produce a valid convex combination
delta = max(0, min(1, phi / rho));

% -------------------------------------------------------------------------
% Step 8: Compute the final shrinkage estimator
%         Convex combination of S and T
Sigma = (1 - delta) * S + delta * T;

end


