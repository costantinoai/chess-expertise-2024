% Function to perform RSA analysis on fMRI data using CoSMoMVPA
% This script performs a representational similarity analysis (RSA) for each subject and region of interest (ROI),
% comparing the resulting RDM with a provided RDM (confoundRDM). Results are saved in specified output folders.

% Define paths
derivativesDir = '/data/projects/chess/data/BIDS/derivatives';
spmRoot = fullfile(derivativesDir, 'fmriprep-SPM');
roisRoot = fullfile(derivativesDir, 'rois-HCP');
outRoot = fullfile(derivativesDir, 'mvpa', 'svm_hpc');
glmPath = fullfile(spmRoot, 'MNI', 'fmriprep-SPM-MNI-checknocheck', 'GLM');
% selectedSubjectsList = '*'; % Can be modified to specify subjects
selectedSubjectsList = [37, 38, 39, 40];        % Must be list of integers or '*'

% Find subject folders based on selected subjects list
subDirs = findSubjectsFolders(glmPath, selectedSubjectsList);

% Process each subject
for subDirIdx = 1:length(subDirs)
    subPath = subDirs(subDirIdx);
    subName = subPath.name;
    subID = strsplit(subName, '-');
    selectedSub = str2double(subID{2});
    spmSubjDir = fullfile(glmPath, subName, 'exp');
    outDir = fullfile(outRoot, subName);
    mkdir(outDir);
    sprintf("Processing %s", subName)

    % Path to multi-label ROI mask
    roiFilePattern = fullfile(roisRoot, subName, strcat(subName, '_HCPMMP1_volume_MNI.nii'));
    roiPathStruct = prepareRoiFile(roiFilePattern);
    colorTablePath = fullfile(roisRoot, subName, 'label','lh_HCPMMP1_color_table.txt');


    if isempty(roiPathStruct)
        continue; % Skip if no ROI file is found
    end

    roiPath = fullfile(roiPathStruct.folder, roiPathStruct.name);

    % Get dataset
    ds = prepareDataset(spmSubjDir);

    % Get RDM to compare to brain RDM
    [targets_, null] = createTargetRDMs(ds);
    targets = {targets_.checkmate, targets_.visualStimuli, targets_.categories};

    % Perform RSA for each ROI in the atlas, now expecting two outputs
    results = performMVPACrossValidationForSubject(ds, targets, roiPath, outDir, colorTablePath);
    % cosmo_plot_slices(cosmo_slice(results,1));
    % cosmo_plot_slices(cosmo_slice(results,2));
    % cosmo_plot_slices(cosmo_slice(results,3));

end


%% FUNCTIONS
function ds = prepareDataset(spmSubjDir)
% Prepares the CoSMoMVPA dataset for RSA analysis.
%
% This function loads fMRI data, prepares it for RSA by removing
% useless data, and checks the dataset's integrity.
%
% Parameters:
%   spmSubjDir: Directory containing the subject's SPM.mat file.
%   roiPath: Path to the subject's multi-label ROI NIFTI file.
%
% Returns:
%   ds: A CoSMoMVPA dataset ready for RSA analysis.

% Load dataset with mask
ds = cosmo_fmri_dataset(fullfile(spmSubjDir, 'SPM.mat'));

% Remove features with no variance and other useless data
% ds = cosmo_remove_useless_data(ds);

% Check dataset integrity
cosmo_check_dataset(ds);

% Warn if there are too few features for RSA
if size(ds.samples, 2) < 6
    warning('Less than 6 features found for this dataset after cleaning and masking. Skipping..');
end
end

function roiPathStruct = prepareRoiFile(roiFilePattern)
% Check and prepare ROI file, handling .nii and .nii.gz cases
roiPathStruct = dir(roiFilePattern);

% If multiple or no NIFTI files found, throw an error
if size(roiPathStruct,1) > 1
    error('Multiple ROI files found at %s.', roiFilePattern)

    % If no .nii files found, check for .nii.gz files and decompress
elseif isempty(roiPathStruct)
    warning('No ROI file found at %s. Checking for nii.gz file... ', roiFilePattern)
    roiFilePattern = fullfile(strcat(roiFilePattern, '.gz'));
    roiPathStruct = dir(roiFilePattern);
    % If no .nii.gz files found
    if isempty(roiPathStruct)
        warning('No ROI file found at %s. SKIPPING!', roiFilePattern)
        % If multiple files are found for this run
    elseif size(roiPathStruct,1) > 1
        error('Multiple NII.GZ files found.')
        % If only one file is found for this run (expected)
    else
        roiFilePatternOld = fullfile(roiPathStruct.folder, roiPathStruct.name);
        gunzippedNii = gunzipNiftiFile(roiFilePatternOld, roiPathStruct.folder);
        roiPathStruct = dir(gunzippedNii{1});
    end
end
end

function [labels, names] = loadColorTable(colorTablePath)
    % Load FreeSurfer color table, modify indices by adding offsets, and adjust ROI names.
    % Concatenates the original labels with a 2000 offset and their corresponding names with the first letter replaced by "R".
    %
    % :param colorTablePath: Path to the color table file.
    % :return: Concatenated labels and modified names.

    % Open the file
    fid = fopen(colorTablePath);
    % Read the data, skipping the header
    data = textscan(fid, '%d %s %*[^\n]', 'HeaderLines', 1);
    fclose(fid);
    
    % Original labels with a 1000 offset
    original_labels = data{1} + 1000;
    original_names = data{2};
    
    % Labels with a 2000 offset
    modified_labels = data{1} + 2000;
    % Modify names: replace the first letter with "R"
    modified_names = cellfun(@(name) ['R' name(2:end)], original_names, 'UniformOutput', false);
    
    % Concatenate the original and modified labels
    labels = [original_labels; modified_labels];
    % Concatenate the original and modified names
    names = [original_names; modified_names];
end


function results_table = performMVPACrossValidationForSubject(ds, targets, roiPath, outDir, colorTablePath)
% Performs Multivariate Pattern Analysis (MVPA) using a Support Vector Machine (SVM) classifier
% with cross-validation for a single subject across different ROIs defined by labels in a given ROI file.
%
% :param ds: Dataset structure loaded into CoSMoMVPA.
% :param targets: A cell array of target labels for each classification task.
% :param roiPath: Path to the subject's multi-label ROI NIFTI file.
% :param outDir: Output directory to save the results.

fprintf('Starting MVPA analysis...\n');

% Load the ROI voxel data
fprintf('Loading ROI data from: %s\n', roiPath);
roi_nii = cosmo_fmri_dataset(roiPath);
roi_data = roi_nii.samples;

% Load FreeSurfer color table and extract ROI names and indices
[ct_labels, ct_names] = loadColorTable(colorTablePath);

% Exclude ROIs not present in the color table
unique_labels = intersect(unique(roi_data), ct_labels);

% Initialize the results table with appropriate column names
roi_names = ct_names(ismember(ct_labels, unique_labels));
results_table = array2table(zeros(size(targets, 2), numel(roi_names)), 'VariableNames', strrep(roi_names, ' ', '_'));

% Analysis per label
for label_idx = 1:numel(unique_labels)
    label = unique_labels(label_idx);
    fprintf('Analyzing ROI %d...\n', label);

    % Find voxels belonging to the current label
    mask = roi_data == label;

    for targetIdx = 1:numel(targets)

        fprintf('Processing target %d for ROI %d...\n', targetIdx, label);

        % Apply mask and preprocess dataset for the current ROI
        ds_masked = cosmo_slice(ds, mask, 2); % Slice dataset to include only voxels from current ROI
        ds_masked = cosmo_remove_useless_data(ds_masked); % Remove features with no variance and other useless data
        ds_masked.sa.targets = targets{targetIdx}; % Filter targets

        % Ensure that there are enough samples for each class
        if numel(unique(ds_masked.sa.targets)) < 2
            warning('Not enough classes found for ROI %d and target %d. Skipping...', label, targets{targetIdx});
            continue;
        end

        % Check dataset integrity
        cosmo_check_dataset(ds_masked);

        % Warn if there are too few features for MVPA
        if size(ds_masked.samples, 2) < 6
            warning('Less than 6 features found for this dataset after cleaning and masking. Skipping..');
            continue;
        end
        
        % Classifier setup
        classifier = @cosmo_classify_svm;

        % Setup for cross-validation
        partitions = cosmo_nfold_partitioner(ds_masked);
        partitions=cosmo_balance_partitions(partitions, ds_masked);
        cosmo_check_partitions(partitions, ds_masked);

        % Run the classifier with cross-validation
        [pred, accuracy] = cosmo_crossvalidate(ds_masked, classifier, partitions);

        % Store classification accuracy in the correct row (targetIdx) and column (label)
        column_name = ct_names{ct_labels == label};
        chance = 1/length(unique(targets{targetIdx}));
        results_table{targetIdx, column_name} = accuracy;


        fprintf('Accuracy - chance for target %d in ROI %d: %.2f%%\n', targetIdx, label, (accuracy - chance) * 100);

    end
end

% Save the table to 'outdir' with a specified name
outFilePath = fullfile(outDir, 'mvpa_crossval_accuracy_check-vis-strategy.tsv');
fprintf('Saving results to %s\n', outFilePath);
writetable(results_table, outFilePath, 'FileType', 'text', 'Delimiter', '\t');
end

function [targets, targetRDMs] = createTargetRDMs(ds)
% Creates target Representational Dissimilarity Matrices (RDMs) based on different labelings.
%
% This function processes labels from a dataset to infer class labels for three separate
% comparisons: CheckTarget (Check vs No-Check), CategoryTarget (C1, C2, ..., NC5),
% and StimuliTargets, where each unique stimulus has a unique label over runs. It constructs
% target RDMs for each of these label sets.
%
% Parameters:
%   ds: The dataset structure with sample attributes .sa.labels containing the labels.
%
% Returns:
%   targetRDMs: A struct containing the target RDMs for checkmate, categories, and stimuli.

fprintf('STEP: importing SPM conditions to infer class labels and runs\n');

% Get labels from the dataset
labels = ds.sa.labels;

% Extract checkmate targets, categories, and stimulus labels
[checkmateLabels, categoryMatches, stimuliLabels] = cellfun(@(x) deal(regexp(x, '(?<=\s)(C|NC)\d+', 'match', 'once'), ...
    regexp(x, '(?<=\s)(C|NC)(\d+)', 'tokens', 'once'), ...
    regexp(x, '(?<=_).*?(?=\*)', 'match', 'once')), labels, 'UniformOutput', false);

% Convert checkmate labels 'C' and 'NC' to 1 and 2
checkmateTargets = cellfun(@(x) strcmp(x(1), 'C') + 1, checkmateLabels);

% Concatenate the strings of cells 1 and 2 for each entry in categoryMatches
concatCategories = cellfun(@(x) [x{1}, x{2}], categoryMatches, 'UniformOutput', false);

% Map concatenated labels to target integers based on their order of appearance
[uniqueCategories, ~] = unique(concatCategories, 'stable');
categoryDict = containers.Map(uniqueCategories, 1:length(uniqueCategories));
categories = cellfun(@(x) categoryDict(x), concatCategories);

% Map stimuli labels to target integers based on their order of appearance
% Convert stimuli labels to lower case before finding unique labels
lowerStimuliLabels = lower(stimuliLabels);
[uniqueStimuli, ~] = unique(lowerStimuliLabels, 'stable');
stimuliDict = containers.Map(uniqueStimuli, 1:length(uniqueStimuli));
stimuliTargets = cellfun(@(x) stimuliDict(x), lowerStimuliLabels);

% Clean '(NoMate)' from stimuli labels and map to target integers
cleanedStimuliLabels = cellstr(erase(string(lowerStimuliLabels), '(nomate)'));
[uniqueVisStimuli, ~] = unique(cleanedStimuliLabels, 'stable');
stimuliVisDict = containers.Map(uniqueVisStimuli, 1:length(uniqueVisStimuli));
stimuliVisTargets = cellfun(@(x) stimuliVisDict(x), cleanedStimuliLabels);

% Create target RDMs
targetRDMs.checkmate = squareform(pdist(checkmateTargets(:), 'hamming'));
targetRDMs.categories = squareform(pdist(categories(:), 'hamming'));
targetRDMs.stimuli = squareform(pdist(stimuliTargets(:), 'hamming'));
targetRDMs.visualStimuli = squareform(pdist(stimuliVisTargets(:), 'hamming'));

targets.checkmate = checkmateTargets;
targets.categories = categories;
targets.stimuli = stimuliTargets;
targets.visualStimuli = stimuliVisTargets;

fprintf('DONE: Target RDMs created\n');
end
