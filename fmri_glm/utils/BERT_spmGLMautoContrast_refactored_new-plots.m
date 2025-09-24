% GLM Analysis Script for fMRI Data using SPM. The script takes fRMIprep
% BIDS structure, decompress the images, smooth if necessary, defines and
% runs a GLM on each defined task, and applies contrasts.

% Assumed Shape of Input Data: - The fMRI data should be preprocessed using
% fMRIprep and organized in BIDS format. - The event files for each run
% should be in TSV format following BIDS specifications. - The confound
% files for each run should be in JSON and TSV format following BIDS
% specifications.

% Script Steps:
% 1. Retrieve the subject folders based on the selectedSubjects parameter.
% 2. Iterate over each subject and perform the GLM analysis steps.
% 3. For each subject, retrieve the event files, confound files, and preprocessed fMRI data files for the selected task and runs.
% 4. Set the SPM parameters for the model specification and estimation.
% 5. Define the contrasts of interest for the analysis.
% 6. Perform the GLM analysis using the event files, confound files, and fMRI data files for each run.
% 7. Save the results of the analysis in the specified output folder.

% Parameters:
% - selectedSubjects: List of subject IDs to include in the analysis. Can be a list of integers or '*' to include all subjects.
% - selectedRuns: List of run numbers to include in the analysis. Can be an integer or '*' to include all runs.
% - selectedTask: Structure with information about the task(s) to analyze.
%       The structures must have at least one item (i.e., one task), and
%       each item must have the following four fields: name, contrasts,
%       weights, and smoothBool.
%           - name: String. The name of the task
%           - contrasts: Cell array of strings. The name(s) of the contrast(s)
%           - weights: Cell array of ints. The weights for the contrast(s).
%           - smoothBool: Boolean. Whether to smooth the data before the GLM.
%
%       e.g., % Define tasks, weights and constrasts as a structure
%       selectedTasks(1).name = 'loc1';
%       selectedTasks(1).contrasts = {'Faces > Objects', 'Objects > Scrambled', 'Scenes > Objects'};
%       selectedTasks(1).weights = {[1 -1 0 0], [0 1 0 -1], [0 -1 1 0]};
%       selectedTasks(1).smoothBool = true; % Whether to smooth the images before GLM

% Paths:
% - fmriprepRoot: Path to the root folder of the fMRIprep output.
% - BIDSRoot: Path to the root folder of the BIDS dataset.
% - outRoot: Path to the output root folder for the analysis results.

% Preprocessing Options (uses fMRIprep confounds table):
% - pipeline: Denoising pipeline strategy for SPM confound regression. It should be a cell array of strings specifying the strategies to use.
%
% Author: Andrea Ivan Costantino
% Date: 5 July 2023

clc
clear

%% PARAMETERS

% This script processes fMRI data by creating events files from an Excel file,
% and then running a General Linear Model (GLM) analysis using SPM.
% The script allows for optional removal of the first TR (Time Repetition) and adjusts the events accordingly.

% Set to true if you want to remove the first TR and adjust the events accordingly.
removeFirstTR = false;

% Paths to data directories
niftiSpace = 'MNIPediatricAsym_cohort-1_res-2'; % Specify the space of the NIfTI files (e.g., T1w, MNI)

% Define the root paths for BIDS, derivatives, and fmriprep outputs
BIDSRoot = '/home/eik-tb/Desktop/bert-fmri/BIDS';
derivativesPath = fullfile(BIDSRoot, 'derivatives');
fmriprepRoot = fullfile(derivativesPath, 'fmriprep');
outputRoot = fullfile(derivativesPath, 'fmriprep-SPM');
temporaryDir = fullfile(derivativesPath, 'pre-SPM');

% Path to the input Excel file containing event information
inputExcelPath = '/home/eik-tb/Desktop/bert-fmri/MergeConditionNew.xlsx';

% Experimental conditions used in the study
conditionsList = {'Face', 'Number', 'Word', 'Fixation'};

% Parameters for the fMRI runs
if removeFirstTR == true
    numberOfTRs = 34; % Total number of TRs in each run
else
    numberOfTRs = 35; % Total number of TRs in each run
end

TRDuration = 2; % Duration of each TR in seconds

% Subjects and runs to process
selectedSubjectsList = [19, 20]; % List of subject IDs to process (use '*' for all subjects)
selectedRuns = '*'; % Runs to process (use '*' for all runs)

% Define tasks, contrasts, and weights for the GLM analysis
selectedTasks(1).name = 'exp';
selectedTasks(1).contrasts = {'Faces > Fixation'};
selectedTasks(1).weights(1) = struct('Face', 1, 'Fixation', -1);
selectedTasks(1).smoothImages = true; % Whether to smooth the images before GLM (univariate --> smooth; MVPA --> no smooth)

% Define tasks, weights and constrasts as a structure
% selectedTasks(2).name = 'loc1';
% selectedTasks(2).contrasts = {'Faces > Objects', 'Objects > Scrambled', 'Scenes > Objects'};
% selectedTasks(2).weights(1) = struct('Faces', 1, 'Objects', -1, 'Scrambled', 0, 'Scenes', 0);
% selectedTasks(2).weights(2) = struct('Faces', 0, 'Objects', 1, 'Scrambled', -1, 'Scenes', 0);
% selectedTasks(2).weights(3) = struct('Faces', 0, 'Objects', -1, 'Scrambled', 0, 'Scenes', 1);
% selectedTasks(2).smoothBool = true; % Whether to smooth the images before GLM

% Denoising pipeline strategy for SPM confound regression Must be a cell
% array of strings {}. Possible strategies:
%     HMP - Head motion parameters (6,12,24)
%     GS - Brain mask global signal (1,2,4)
%     CSF_WM - CSF and WM masks global signal (2,4,8)
%     FD - Raw framewise displacement
%     Null - returns a blank df

% denoisingPipelines = {
%     {'Null'}, ...
%     {'GS-1'}, ...
%     {'HMP-6'}, ...
%     {'FD'}, ...
%     {'GS-1', 'HMP-6'}, ...
%     {'GS-1', 'FD'}, ...
%     {'HMP-6', 'FD'}, ...
%     {'GS-1', 'HMP-6', 'FD'} ...
%     };

denoisingPipelines = {
    {'GS-1', 'FD'}, ...
    };

% Thresholds for statistical analysis
thresholds = {
    0.001, ...
    0.01, ...
    0.05 ...
    };

% Get the list of subject folders to process
subjectPaths = findSubjectsFolders(fmriprepRoot, selectedSubjectsList);

%% Load and Prepare Data

% Set import options for the Excel file
% We need to ensure that the 'run' column is read as a string
excelImportOptions = detectImportOptions(inputExcelPath);
excelImportOptions = setvartype(excelImportOptions, 'run', 'string'); % Treat 'run' column as a string

% Read data from the Excel file into a table
eventDataTable = readtable(inputExcelPath, excelImportOptions);

% Extract unique subject identifiers from the data
subjectIDs = unique(eventDataTable.subject);
subjectIDs(subjectIDs == "") = []; % Remove empty entries

% Calculate the total run duration in seconds
runEndTime = numberOfTRs * TRDuration;

%% Create BIDS events.tsv Files from Excel Data
% WARNING: this will overwrwite the
% BIDS/sub-xx/func/sub-xx_run-x_events.tsv files if they  exist!

% Loop over each subject to create events files
for subjectIndex = 1:length(selectedSubjectsList)
    % Get the current subject ID in the format 'sub-XX'
    subjectID = sprintf('sub-%02d', selectedSubjectsList(subjectIndex));

    % Call the function to create events files for the subject
    createEventsFilesForSubject(subjectID, eventDataTable, BIDSRoot, runEndTime, removeFirstTR, TRDuration);
end

%% RUN THE GLM FOR EACH SUBJECT, TASK, AND DENOISING PIPELINE

for subjectPathIndex = 1:length(subjectPaths)

    % Loop over each task
    for taskIndex = 1:length(selectedTasks)

        % Get task information
        selectedTask = selectedTasks(taskIndex).name;        % Current task name
        contrasts = selectedTasks(taskIndex).contrasts;      % Contrasts for the current task
        smoothImages = selectedTasks(taskIndex).smoothImages; % Whether to smooth images before GLM

        % Extract subject information
        subjectPath = subjectPaths(subjectPathIndex);
        subjectName = subjectPath.name;
        subjectIDParts = strsplit(subjectName, '-');
        subjectIDNumber = str2double(subjectIDParts{2});
        subjectID = sprintf('sub-%02d', subjectIDNumber);

        % Check if the subject has the specified task; if not, skip
        funcDir = fullfile(subjectPath.folder, subjectPath.name, 'func');
        funcFiles = dir(funcDir);
        fileNames = {funcFiles.name};
        hasTask = any(contains(fileNames, ['task-', selectedTask]));

        if ~hasTask
            warning('Task %s not found for %s in %s. Skipping...', selectedTask, subjectName, funcDir);
            continue;
        end

        % Iterate over denoising pipelines
        for pipelineIndex = 1:length(denoisingPipelines)

            clearvars matlabbatch % Clear previous matlabbatch variable

            % Get the current denoising pipeline
            pipeline = denoisingPipelines{pipelineIndex};

            % Create a string representation of the pipeline for naming
            pipelineStr = strjoin(pipeline, '_');

            % Set output path with pipeline included
            outputPath = fullfile([outputRoot, '_', pipelineStr], subjectName);

            % Print status update
            fprintf('############################### \n# Processing %s - %s #\n############################### \n', subjectName, selectedTask);

            % Set paths for fMRI data and BIDS data
            funcPathSubject = fullfile(fmriprepRoot, subjectName, 'func');
            bidsPathSubject = fullfile(BIDSRoot, subjectName, 'func');

            %% Find and Load Events and Confounds from fmriprep Folder

            % Get events and confounds files for the subject and task
            if ismember('*', selectedRuns)
                eventsTsvFiles = dir(fullfile(bidsPathSubject, sprintf('%s_task-%s_run-*_events.tsv', subjectName, selectedTask)));
                jsonConfoundsFiles = dir(fullfile(funcPathSubject, sprintf('%s_task-%s_run-*_desc-confounds_timeseries.json', subjectName, selectedTask)));
                tsvConfoundsFiles = dir(fullfile(funcPathSubject, sprintf('%s_task-%s_run-*_desc-confounds_timeseries.tsv', subjectName, selectedTask)));
            else
                eventsTsvFiles = arrayfun(@(x) dir(fullfile(bidsPathSubject, sprintf('%s_task-%s_run-%01d_events.tsv', subjectName, selectedTask, x))), selectedRuns, 'UniformOutput', true);
                jsonConfoundsFiles = arrayfun(@(x) dir(fullfile(funcPathSubject, sprintf('%s_task-%s_run-%01d_desc-confounds_timeseries.json', subjectName, selectedTask, x))), selectedRuns, 'UniformOutput', true);
                tsvConfoundsFiles = arrayfun(@(x) dir(fullfile(funcPathSubject, sprintf('%s_task-%s_run-%01d_desc-confounds_timeseries.tsv', subjectName, selectedTask, x))), selectedRuns, 'UniformOutput', true);
            end

            % Sort files by name
            eventsTsvFiles = table2struct(sortrows(struct2table(eventsTsvFiles), 'name'));
            jsonConfoundsFiles = table2struct(sortrows(struct2table(jsonConfoundsFiles), 'name'));
            tsvConfoundsFiles = table2struct(sortrows(struct2table(tsvConfoundsFiles), 'name'));

            % Check that the number of events and confounds files match
            assert(numel(eventsTsvFiles) == numel(jsonConfoundsFiles) && numel(jsonConfoundsFiles) == numel(tsvConfoundsFiles), ...
                'Mismatch in number of events and confounds files for %s', funcPathSubject);

            %% SPM Model Parameters (Non-Run Dependent)
            % Define general model parameters for SPM
            % Specify the output directory for the SPM analysis
            matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(outputPath);

            % Set the units of time for the onsets and durations ('secs' for seconds)
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';

            % Define the repetition time (TR) in seconds, which is the time between scans
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TRDuration;

            % Set the number of slices acquired in the fMRI acquisition
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 52;

            % Specify the reference slice for timing correction (usually middle slice)
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 27;

            % Specify any factorial designs (empty here as we are not using factorial designs)
            matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});

            % Set the basis function for modeling the hemodynamic response function (HRF)
            % [0 0] indicates that no temporal or dispersion derivatives are used
            matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];

            % Specify whether to allow session-specific regressors for parametric modulation (1 means yes)
            matlabbatch{1}.spm.stats.fmri_spec.volt = 1;

            % Set global normalization (set to 'None' to avoid scaling data globally)
            matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

            % Define the threshold for implicit masking (exclude voxels with intensity < 0.8)
            matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;

            % Specify an explicit mask file if desired (empty here means no explicit mask is used)
            matlabbatch{1}.spm.stats.fmri_spec.mask = {''};

            % Set the autocorrelation model for serial correlation correction ('AR(1)' is autoregressive model of order 1)
            matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

            % Model estimation parameters
            % Define the dependency for the SPM.mat file to be generated in the first batch
            matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', ...
                substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));

            % Set whether to write residuals as output (0 means no residuals will be written)
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;

            % Specify the estimation method (Classical means maximum likelihood estimation)
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;


            %% SPM Run Settings (Events, Confounds, Images)

            % Iterate over each run
            for runIndex = 1:numel(eventsTsvFiles)

                % Extract events for the current run from the events TSV file
                eventsStruct = eventsBIDS2SPM(fullfile(eventsTsvFiles(runIndex).folder, eventsTsvFiles(runIndex).name));

                % Extract the exact run identifier from the file name
                selectedRun = findRunSubstring(eventsTsvFiles(runIndex).name);

                % Display a message indicating the task and run being processed
                fprintf('## Processing Task %s - %s\n', selectedTask, selectedRun);

                % Select the corresponding confounds JSON files for the current run
                jsonRows = filterRowsBySubstring(jsonConfoundsFiles, selectedRun);

                % Select the corresponding confounds TSV files for the current run
                confoundsRows = filterRowsBySubstring(tsvConfoundsFiles, selectedRun);

                % Ensure there is exactly one JSON confounds file for this run
                if length(jsonRows) ~= 1
                    error('Unexpected number of JSON confounds files for run %s.', selectedRun);
                else
                    jsonRow = jsonRows{1}; % Extract the single JSON row
                end

                % Ensure there is exactly one TSV confounds file for this run
                if length(confoundsRows) ~= 1
                    error('Unexpected number of TSV confounds files for run %s.', selectedRun);
                else
                    confoundsRow = confoundsRows{1}; % Extract the single TSV row
                end

                % Extract confound regressors based on the specified denoising pipeline
                confoundsArray = fMRIprepConfounds2SPM(fullfile(confoundsRow.folder, confoundsRow.name), pipeline);

                % Define the expected NIfTI file name pattern for the current run
                spaceString = getSpaceString(niftiSpace); % Get the space string (e.g., MNI)
                filePattern = sprintf('%s_task-%s_%s_space-%s_desc-preproc_bold', subjectName, selectedTask, selectedRun, spaceString);

                % Search for the corresponding NIfTI file
                niiFileStruct = dir(fullfile(fmriprepRoot, subjectName, 'func', [filePattern, '.nii']));

                % Handle scenarios where the NIfTI file is not found or multiple files exist
                if isempty(niiFileStruct)
                    % If no NIfTI file is found, check for compressed NIfTI files (.nii.gz)
                    niiGzFileStruct = dir(fullfile(funcPathSubject, [filePattern, '.nii.gz']));

                    if isempty(niiGzFileStruct)
                        % If no compressed file is found, display a warning and skip this run
                        warning('No NIfTI file found for run %s. Skipping...', selectedRun);
                        continue;
                    elseif numel(niiGzFileStruct) > 1
                        % If multiple compressed files are found, throw an error
                        error('Multiple NIfTI.gz files found for run %s.', selectedRun);
                    else
                        % Decompress the single compressed NIfTI file
                        niiGzFilePath = fullfile(niiGzFileStruct.folder, niiGzFileStruct.name);
                        gunzippedNii = gunzipNiftiFile(niiGzFilePath, fullfile(temporaryDir, 'gunzipped', subjectName));
                        niiFileStruct = dir(gunzippedNii{1});
                    end
                elseif numel(niiFileStruct) > 1
                    % If multiple uncompressed NIfTI files are found, throw an error
                    error('Multiple NIfTI files found for run %s.', selectedRun);
                end

                % Construct the full path to the selected NIfTI file
                niiFilePath = fullfile(niiFileStruct.folder, niiFileStruct.name);

                % Smooth the NIfTI file if smoothing is required for this task
                if smoothImages
                    niiFilePath = smoothNiftiFile(niiFilePath, fullfile(temporaryDir, 'smoothed', subjectName));
                else
                    fprintf('Smoothing not applied for this task.\n');
                end

                % Prepare the list of scan files for SPM
                niiFileCell = {niiFilePath};

                % Expand the scan list to include all volumes in the NIfTI file
                allScans = spm_select('expand', niiFileCell);

                % If removing the first TR, exclude the first scan; otherwise, include all scans
                if removeFirstTR == true
                    selectedScans = allScans(2:end);
                else
                    selectedScans = allScans;
                end

                % Set the list of scans for the current session in the SPM batch
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).scans = selectedScans;

                % Set the high-pass filter for the session to avoid filtering (set to run duration + 100 seconds)
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).hpf = (TRDuration * size(selectedScans, 1)) + 100;

                % Set condition-specific information (e.g., names, onsets, durations) for the session
                for condIndex = 1:length(eventsStruct.names)
                    matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).cond(condIndex).name = eventsStruct.names{condIndex};
                    matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).cond(condIndex).onset = eventsStruct.onsets{condIndex};
                    matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).cond(condIndex).duration = eventsStruct.durations{condIndex};
                end

                % Set the confound regressors for the session based on the extracted confounds
                for regIndex = 1:size(confoundsArray, 2)
                    % Assign the name of the confound regressor
                    matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).regress(regIndex).name = confoundsArray.Properties.VariableNames{regIndex};

                    % Extract the values of the confound regressor
                    confoundValues = confoundsArray{:, regIndex};

                    % If removing the first TR, exclude the first value; otherwise, include all values
                    if removeFirstTR == true
                        confoundValues = confoundValues(2:end);
                    end

                    % Assign the confound values to the batch
                    matlabbatch{1}.spm.stats.fmri_spec.sess(runIndex).regress(regIndex).val = confoundValues;
                end
            end

            %% Run SPM Batches for Model Specification and Estimation

            spm('defaults', 'fmri');
            spm_jobman('initcfg');
            fprintf('Running GLM for %s - Task: %s\n', subjectName, selectedTask);
            spm_jobman('run', matlabbatch(1:2));
            fprintf('GLM completed.\n');

            %% Load SPM.mat

            spmMatPath = fullfile(outputPath, 'SPM.mat');
            if ~exist(spmMatPath, 'file')
                error('SPM.mat file not found in %s.', outputPath);
            end

            %% Save Boxcar plot and design matrix of estimated model
            SPMstruct = load(spmMatPath);

            plotBoxcarAndHRFResponses(SPMstruct, outputPath)
            saveSPMDesignMatrix(SPMstruct, outputPath)

            %% Set Contrasts

            % Set contrasts in SPM
            matlabbatch{3}.spm.stats.con.spmmat(1) = {spmMatPath};

            for contrastIndex = 1:length(contrasts)
                % Get weights for the contrast
                weights = adjust_contrasts(spmMatPath, selectedTasks(taskIndex).weights(contrastIndex));

                matlabbatch{3}.spm.stats.con.consess{contrastIndex}.tcon.weights = weights;
                matlabbatch{3}.spm.stats.con.consess{contrastIndex}.tcon.name = contrasts{contrastIndex};
                matlabbatch{3}.spm.stats.con.consess{contrastIndex}.tcon.sessrep = 'none';
            end

            % Run the contrast batch
            fprintf('Setting contrasts...\n');
            spm_jobman('run', matlabbatch(3));
            fprintf('Contrasts set.\n');

            %% Save Contrast Plots

            % Iterate over contrasts to generate plots
            for constrastIdx = 1:length(selectedTasks(taskIndex).contrasts)

                % Iterate over thresholds to generate plots
                for thresholdIndex = 1:length(thresholds)

                    % Generate and Save Contrast Overlay Images

                    % Set crosshair coordinates (modify if needed)
                    crossCoords = [40, -52, -18];

                    % Set the index of the contrast to display (modify if needed)
                    spmContrastIndex = constrastIdx;

                    % Call the function to generate and save contrast overlay images
                    generateContrastOverlayImages(spmMatPath, outputPath, fmriprepRoot, subjectName, pipelineStr, thresholds{thresholdIndex}, spmContrastIndex, crossCoords);

                end
            end

            %% Save Script for Reproducibility

            scriptSourcePath = [mfilename('fullpath'), '.m'];
            scriptDestinationPath = fullfile(outputPath, 'spmGLMautoContrast.m');
            copyfile(scriptSourcePath, scriptDestinationPath);

        end

        %% Combine Plots for Each Threshold Level

        % Iterate over contrasts to generate plots
        for constrastIdx = 1:length(selectedTasks(taskIndex).contrasts)

            % Iterate over thresholds to generate plots
            for thresholdIndex = 1:length(thresholds)

                % Initialize a figure for combined plots
                combinedFig = figure('Visible', 'off');
                numPipelines = length(denoisingPipelines);
                numRows = 2; % Fixed number of rows
                numCols = ceil(numPipelines / numRows); % Adjust columns based on pipelines

                % Create a tiled layout
                tiledLayout = tiledlayout(numRows, numCols, 'Padding', 'none', 'TileSpacing', 'none');

                % Iterate over all denoising pipelines
                for pipelineIndex = 1:numPipelines

                    % Load the overlay image
                    pipelineStr = strjoin(denoisingPipelines{pipelineIndex}, '_');
                    contrastName = sprintf('%s_%s_%g_%s', subjectName, pipelineStr, thresholds{thresholdIndex}, selectedTasks(taskIndex).contrasts{constrastIdx});
                    overlayImgPath = fullfile([outputRoot, '_', pipelineStr], subjectName, sprintf('%s.png', contrastName));

                    if exist(overlayImgPath, 'file')
                        % Add image to combined plot
                        ax = nexttile(tiledLayout);
                        img = imread(overlayImgPath);
                        imshow(img, 'Parent', ax);
                        axis(ax, 'off'); % Hide axis
                    else
                        warning('Overlay image not found: %s', overlayImgPath);
                    end
                end

                % Save the combined figure
                combinedImgPath = fullfile(derivativesPath, sprintf('combined-pipelines_%s_%g_%s.png', subjectName, thresholds{thresholdIndex}, selectedTasks(taskIndex).contrasts{constrastIdx}));
                exportgraphics(combinedFig, combinedImgPath, 'Resolution', 300);
                fprintf('Combined plot saved as %s\n', combinedImgPath);
                close(combinedFig); % Close the figure
            end
        end
    end
end

%% Function Definitions
function createEventsFilesForSubject(subjectID, eventDataTable, BIDSRoot, runEndTime, removeFirstTR, TRDuration)
% CREATEEVENTSFILESFORSUBJECT Creates BIDS events.tsv files for a given subject
%
% Inputs:
%   subjectID       - String representing the subject ID (e.g., 'sub-01')
%   eventDataTable  - Table containing event data loaded from Excel
%   BIDSRoot        - String representing the root directory of the BIDS dataset
%   runEndTime      - Scalar, total run duration in seconds
%   removeFirstTR   - Boolean, whether to remove the first TR and adjust events
%   TRDuration      - Scalar, duration of each TR in seconds
%
% This function processes the event data for the specified subject and creates
% BIDS-compliant events.tsv files for each experiment and run.

fprintf('Processing subject: %s\n', subjectID);

% Filter data for the current subject
subjectData = eventDataTable(strcmp(eventDataTable.subject, subjectID), :);

% Check if there is data for the current subject
if isempty(subjectData)
    fprintf('No data available for subject: %s\n', subjectID);
    return;
end

% Extract unique experiments and runs for the subject
uniqueExperiments = unique(subjectData.experiment);
uniqueRuns = unique(subjectData.run);

% Loop over each experiment and run to create individual events.tsv files
for expIndex = 1:length(uniqueExperiments)
    experimentName = uniqueExperiments{expIndex}; % Current experiment

    for runIndex = 1:length(uniqueRuns)
        runName = uniqueRuns{runIndex}; % Current run

        % Filter data for the current experiment and run
        runData = subjectData(strcmp(subjectData.experiment, experimentName) & ...
            strcmp(subjectData.run, runName), :);

        % Skip if no data for this experiment and run
        if isempty(runData)
            continue;
        end

        % Adjust block onsets: if keeping the first TR, add 2000 ms to all onsets except the first; otherwise, keep as is
        if removeFirstTR == false
            runData.block_onset = runData.block_onset + (TRDuration*1000); % Add TR duration in s to all but the first row
        end

        % Initialize BIDS events table
        eventsTable = table();

        % Identify non-fixation events (e.g., 'Face', 'Number', 'Word')
        isNonFixation = ismember(runData.condition, {'Face', 'Number', 'Word'});

        % Onsets and durations for non-fixation conditions
        eventsTable.onset = runData.block_onset(isNonFixation) / 1000; % Convert to seconds
        eventsTable.duration = repmat(6, sum(isNonFixation), 1); % Fixed duration of 6 seconds
        eventsTable.trial_type = runData.condition(isNonFixation); % Use condition names
        eventsTable.stimulus = runData.pattern_split(isNonFixation); % Optional: stimulus information

        % Compute fixation events based on gaps between non-fixation events
        fixationOnsets = [];
        fixationDurations = [];
        lastEndTime = 0; % Track the end of the last event

        % Sort non-fixation events by onset time
        [sortedOnsets, sortIndices] = sort(eventsTable.onset);
        sortedDurations = eventsTable.duration(sortIndices);

        % Loop through sorted onsets to identify fixation gaps
        for idx = 1:length(sortedOnsets)
            currentOnset = sortedOnsets(idx);
            if lastEndTime < currentOnset
                % Start of fixation
                fixationOnsets(end + 1, 1) = lastEndTime;
                % Duration of fixation
                fixationDurations(end + 1, 1) = currentOnset - lastEndTime;
            end
            % Update last end time
            lastEndTime = currentOnset + sortedDurations(idx);
        end

        % Add fixation after the last event if applicable
        if lastEndTime < runEndTime
            fixationOnsets(end + 1, 1) = lastEndTime;
            fixationDurations(end + 1, 1) = runEndTime - lastEndTime;
        end

        % Create fixation events table
        fixationTable = table();
        fixationTable.onset = fixationOnsets;
        fixationTable.duration = fixationDurations;
        fixationTable.trial_type = repmat({'Fixation'}, length(fixationOnsets), 1);
        fixationTable.stimulus = repmat({'fix'}, length(fixationOnsets), 1); % No stimulus for fixation

        % Combine non-fixation and fixation events
        eventsTable = [eventsTable; fixationTable];

        % Sort the events by onset time
        eventsTable = sortrows(eventsTable, 'onset');

        % Define output directory and filename
        outputDir = fullfile(BIDSRoot, subjectID, 'func');
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end

        % Construct the output filename
        outputFileName = fullfile(outputDir, sprintf('%s_task-%s_run-%s_events.tsv', ...
            subjectID, experimentName, runName));

        % Write the events table to a TSV file
        writetable(eventsTable, outputFileName, 'FileType', 'text', 'Delimiter', '\t');
    end
end
disp('BIDS events.tsv files created successfully.');
end

function generateContrastOverlayImages(spmMatPath, outputPath, fmriprepRoot, subjectName, pipelineStr, thresholdValue, spmContrastIndex, crossCoords)
% GENERATECONTRASTOVERLAYIMAGES Generates and saves contrast overlay images for specified thresholds
%
% This function loads the SPM.mat file, sets up the xSPM structure for the specified contrast,
% and generates overlay images on the anatomical image. The overlay images are saved to the output directory.
%
% Inputs:
%   spmMatPath        - String, path to the SPM.mat file
%   outputPath        - String, output directory where images will be saved
%   fmriprepRoot      - String, root directory of fmriprep outputs
%   subjectName       - String, subject identifier (e.g., 'sub-01')
%   pipelineStr       - String, representation of the denoising pipeline
%   thresholdValue         - Numeric threshold to use for generating images
%   spmContrastIndex  - Integer, index of the contrast in SPM.xCon to use (default is 1)
%   crossCoords       - Vector [x, y, z], coordinates to set the crosshair (default is [40, -52, -18])
%
% Outputs:
%   None (overlay images are saved to the output directory)
%
% Usage:
%   generateContrastOverlayImages(spmMatPath, outputPath, fmriprepRoot, subjectName, pipelineStr, thresholds);
%
% Notes:
%   - The function assumes that SPM and SPM12 toolboxes are properly set up.
%   - The function handles any errors during the generation of xSPM and provides informative messages.
%
% Example:
%   generateContrastOverlayImages('/path/to/SPM.mat', '/output/dir', '/fmriprep/root', 'sub-01', 'GS-1_HMP-6', {0.001, 0.01}, 1, [40, -52, -18]);

if nargin < 8
    crossCoords = [40, -52, -18]; % Default crosshair coordinates
end
if nargin < 7
    spmContrastIndex = 1; % Default contrast index
end

% Load SPM.mat to access contrast data
fprintf('Loading SPM.mat from %s to process contrasts...\n', spmMatPath);
load(spmMatPath, 'SPM');

% Verify the contrast index is valid
if spmContrastIndex > numel(SPM.xCon)
    error('Invalid contrast index. Ensure the index is within the range of defined contrasts.');
end

% Get the contrast name from SPM.xCon
contrastNameSPM = SPM.xCon(spmContrastIndex).name;

% Iterate over thresholds to generate and save images
% Prepare xSPM structure for results
contrastName = sprintf('%s_%s_%g_%s', subjectName, pipelineStr, thresholdValue, contrastNameSPM);
xSPM = struct();
xSPM.swd = outputPath; % Directory where SPM.mat is saved
xSPM.title = contrastName;
xSPM.Ic = spmContrastIndex; % Contrast index
xSPM.Im = []; % Mask (empty means no mask)
xSPM.pm = []; % P-value masking
xSPM.Ex = []; % Mask exclusion
xSPM.u = thresholdValue; % Threshold (uncorrected p-value)
xSPM.k = 0; % Extent threshold (number of voxels)
xSPM.STAT = 'T'; % Use T-statistics
xSPM.thresDesc = 'none'; % No threshold description

% Generate results without GUI
[SPM, xSPM] = spm_getSPM(xSPM);

xSPM.thresDesc = 'none'; % No threshold description

% Display results
[hReg, xSPM] = spm_results_ui('setup', xSPM);

% Set crosshair coordinates
spm_results_ui('SetCoords', crossCoords);

% Overlay activations on anatomical image
sectionImgPath = fullfile(fmriprepRoot, subjectName, 'anat', [subjectName, '_space-MNIPediatricAsym_cohort-1_res-2_desc-preproc_T1w.nii']);
if exist(sectionImgPath, 'file')
    fprintf('Overlaying activations for threshold %g...\n', thresholdValue);
    spm_sections(xSPM, hReg, sectionImgPath);

    % Save the overlay image
    overlayImgPath = fullfile(outputPath, sprintf('%s.png', contrastName));
    spm_figure('GetWin', 'Graphics');
    print('-dpng', overlayImgPath);
    fprintf('Overlay saved as %s\n', overlayImgPath);
else
    warning('Anatomical image not found at %s. Skipping overlay.', sectionImgPath);
end

% Close graphics window
spm_figure('Close', 'Graphics');
end
function newFilePath = smoothNiftiFile(niiFile, outPath)
% smoothNiftiFile - Smooth a (already gunzipped) .nii file and save it into a
% 'SPM/*/smoothed' folder in the derivatives directory of a BIDS dataset
% (e.g., ./BIDS/derivatives/SPM/sub-xx/smoothed).
%
% Author: Andrea Costantino
% Date: 3/2/2023
%
% Usage:
%   outRoot = smoothNiftiFile(niiFile, outPath)
%
% Inputs:
%    niiFile - String indicating the path to the input .nii file.
%    outRoot - String indicating the output directory.
%
% Outputs:
%    newFilePath - String indicating the new directory of the output file.
%

% Extract subject and task from nii file name
[niiFolder, niiName, niiExt] = fileparts(niiFile); % isolate the name of the nii file
subAndTask = split(niiName, "_"); % split the nii file name into segments
selectedSub = subAndTask{1}; % subject is the first segment

% Infer the output path if not provided
if nargin < 2
    % get the BIDS root folder path
    splitPath = strsplit(niiFile, '/'); % split the string into folders
    idx = find(contains(splitPath, 'BIDS')); % find the index of the split that includes 'BIDS'
    bidsPath = strjoin(splitPath(1:idx), '/'); % create the full folder path
    outPath = fullfile(bidsPath, 'derivatives', 'SPM', 'smoothed', selectedSub); % build the output path
end

% Check if the output folder already exists
if exist(outPath, 'dir') == 7
    fprintf('SMOOTH: Output directory %s already exists.\n', outPath);
else
    mkdir(outPath); % create the output directory
    fprintf('SMOOTH: Created output directory %s.\n', outPath);
end

% Check if the output file already exists
smoothFileName = strcat('smooth_', [niiName, niiExt]);
newFilePath = fullfile(outPath, strrep(smoothFileName, niiExt, ['_smooth', niiExt]));

if exist(newFilePath, 'file') == 2
    fprintf('SMOOTH: Smoothed file already exists: %s\n', newFilePath);
else
    % Setup and run SPM smoothing job
    matlabbatch{1}.spm.spatial.smooth.data = {niiFile};
    matlabbatch{1}.spm.spatial.smooth.fwhm = [6 6 6];
    matlabbatch{1}.spm.spatial.smooth.dtype = 0;
    matlabbatch{1}.spm.spatial.smooth.im = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 'smooth_';

    % Initialize SPM
    spm_jobman('initcfg');
    spm('defaults','fmri');

    % Run batch job and suppress the SPM output
    fprintf('SMOOTH: smoothing file %s ...\n', [niiName, niiExt])
    spm_jobman('run', matlabbatch);

    % Move file to correct folder
    movefile(fullfile(niiFolder, smoothFileName), newFilePath);
    fprintf('SMOOTH: Created smoothed file: %s\n', newFilePath);

    % Save a copy of this function in the output folder
    if exist(fullfile(outPath, 'smoothNiftiFile.m'), 'file') ~= 2
        copyfile([mfilename('fullpath'), '.m'], fullfile(outPath, 'smoothNiftiFile.m'));
    end
end
end
function weight_vector = adjust_contrasts(spmMatPath, contrastWeights)
% ADJUST_CONTRASTS Adjust contrast weights according to the design matrix in SPM.
%
% DESCRIPTION:
% This function adjusts the specified contrast weights according to the design
% matrix in SPM, and provides a visual representation of the weights applied to
% the design matrix.
%
% INPUTS:
% spmMatPath: String
%   - Path to the SPM.mat file.
%     Example: '/path/to/SPM.mat'
%
% contrastWeights: Struct
%   - Specifies the weight of each condition in the contrast.
%     For wildcard specification, use '_WILDCARD_'. E.g., 'condition_WILDCARD_': weight
%     Example: struct('condition1', 1, 'condition2_WILDCARD_', -1)
%
% OUTPUTS:
% weight_vector: Numeric Vector
%   - A vector of weights for each regressor.
%     Example: [0, 1, -1, 0, ...]
%
% The function also generates a visual representation of the design matrix with
% the specified contrast weights.

% Load the SPM.mat
load(spmMatPath);
% Extracting regressor names from the SPM structure
regressor_names = SPM.xX.name;

% Generate weight vector based on SPM's design matrix and specified weights for the single contrast
weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names);

% % Plotting for visual verification
% figure;
%
% % Display the design matrix
% imagesc(SPM.xX.X);  % Display the design matrix
% colormap('gray');   % Set base colormap to gray for design matrix
% hold on;
%
% % Create a color overlay based on the weights
% for i = 1:length(weight_vector)
%     x = [i-0.5, i+0.5, i+0.5, i-0.5];
%     y = [0.5, 0.5, length(SPM.xX.X) + 0.5, length(SPM.xX.X) + 0.5];
%     if weight_vector(i) > 0
%         % Green for positive weights
%         color = [0, weight_vector(i), 0];  % Green intensity based on weight value
%         patch(x, y, color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);  % Reduced transparency
%     elseif weight_vector(i) < 0
%         % Red for negative weights
%         color = [abs(weight_vector(i)), 0, 0];  % Red intensity based on absolute weight value
%         patch(x, y, color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);  % Reduced transparency
%     end
% end
%
% % Annotate with regressor names
% xticks(1:length(regressor_names));
% xticklabels('');  % Initially empty, to be replaced by colored text objects
% xtickangle(45);  % Angle the text so it doesn't overlap
% set(gca, 'TickLabelInterpreter', 'none');  % Ensure special characters in regressor names display correctly
%
% % Color code the regressor names using text objects
% for i = 1:length(regressor_names)
%     if weight_vector(i) > 0
%         textColor = [0, 0.6, 0];
%     elseif weight_vector(i) < 0
%         textColor = [0.6, 0, 0];
%     else
%         textColor = [0, 0, 0];
%     end
%     text(i, length(SPM.xX.X) + 5, regressor_names{i}, 'Color', textColor, 'Rotation', 45, 'Interpreter', 'none', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
% end
%
% title('Design Matrix with Contrast Weights');
% xlabel('');
% ylabel('Scans');
%
% % Add legends
% legend({'Positive Weights', 'Negative Weights'}, 'Location', 'northoutside');
%
% % Optional: Add a dual color colorbar to represent positive and negative weight intensities
% colorbar('Ticks', [-1, 0, 1], 'TickLabels', {'-Max Weight', '0', '+Max Weight'}, 'Direction', 'reverse');
%
% hold off;
end

function weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names)
% GENERATE_WEIGHT_VECTOR_FROM_SPM Generates a weight vector from the SPM design matrix.
%
% This function constructs a weight vector based on the design matrix in SPM
% and the user-specified contrast weights. It's equipped to handle wildcard matches
% in condition names for flexibility in defining contrasts.
%
% USAGE:
%   weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names)
%
% INPUTS:
%   contrastWeights : struct
%       A struct specifying the weight of each condition in the contrast.
%       Fields of the struct are condition names and the associated values are the contrast weights.
%       Use '_WILDCARD_' in the condition name to denote a wildcard match.
%       Example:
%           contrastWeights = struct('Faces', 1, 'Objects_WILDCARD_', -1);
%
%   regressor_names : cell array of strings
%       Names of the regressors extracted from the SPM.mat structure.
%       Typically includes task conditions and confound regressors.
%       Example:
%           {'Sn(1) Faces*bf(1)', 'Sn(1) Objects*bf(1)', 'Sn(1) trans_x', ...}
%
% OUTPUTS:
%   weight_vector : numeric vector
%       A vector of weights for each regressor in the order they appear in the regressor_names.
%       Example:
%           [1, -1, 0, ...]
%
% NOTE:
%   This function assumes that task-related regressors in the SPM design matrix end with "*bf(1)".
%   Confound regressors (e.g., motion parameters) do not have this suffix.

% Initialize a weight vector of zeros
weight_vector = zeros(1, length(regressor_names));

% Extract field names from the contrastWeights structure
fields = fieldnames(contrastWeights);

% Iterate over the field names to match with regressor names
for i = 1:length(fields)
    field = fields{i};

    % If the field contains a wildcard, handle it
    if contains(field, '_WILDCARD_')
        % Convert the wildcard pattern to a regular expression pattern
        pattern = ['Sn\(.\) ' strrep(field, '_WILDCARD_', '.*')];

        % Find indices of matching regressors using the regular expression pattern
        idx = find(~cellfun('isempty', regexp(regressor_names, pattern)));

        % Assign the weight from contrastWeights to the matching regressors
        weight_vector(idx) = contrastWeights.(field);
    else
        % No need to extract the condition name, just append *bf(1) to match the SPM regressor pattern
        pattern = ['Sn\(.\) ' field];

        idx = find(~cellfun('isempty', regexp(regressor_names, pattern)));

        % Assign the weight from contrastWeights to the regressor
        if ~isempty(idx)
            weight_vector(idx) = contrastWeights.(field);
        end
    end
end
end

function new_df = eventsBIDS2SPM(tsv_file)
% eventsBIDS2SPM - Convert BIDS event files to SPM format
% This function reads a BIDS event file and converts it to the format required by SPM.
% It extracts the unique trial types and their onsets and durations and stores them in a
% Matlab structure.
%
% Author: Andrea Costantino
% Date: 23/1/2023
%
% Usage:
%   mat_dict = eventsBIDS2SPM(tsv_file, run_id)
%
% Inputs:
%   tsv_file - string, path to the tsv file containing the events
%
% Outputs:
%   mat_dict - struct, a Matlab structure containing the events in the format
%              required by SPM. The structure contains three fields:
%                - 'names': cell array of string, the names of the trial types
%                - 'onsets': cell array of double, onset times of the trials
%                - 'durations': cell array of double, duration of the trials
%
% This function reads a BIDS event file and converts it to the format required by SPM.
% It extracts the unique trial types and their onsets and durations and stores them in a
% Matlab structure

% read the tsv file
df = readtable(tsv_file,'FileType','text');
% Select unique trial type name
unique_names = unique(df.trial_type);
% Make new table in a form that SPM can read
new_df = table('Size',[length(unique_names),3],'VariableTypes',{'cellstr', 'cellstr', 'cellstr'},'VariableNames',{'names', 'onsets', 'durations'});
% For each trial type (i.e., condition)
for k = 1:length(unique_names)
    % Select rows belonging to that condition
    filtered = df(strcmp(df.trial_type,unique_names{k}),:);
    % Copy trial name, onset and duration to the new table
    new_df.names(k) = unique(filtered.trial_type);
    new_df.onsets(k) = {filtered.onset};
    new_df.durations(k) = {filtered.duration};
end
new_df = sortrows(new_df, 'names');
end

function confounds = fMRIprepConfounds2SPM(tsv_path, pipeline)
% fMRIprepConfounds2SPM - Extracts and formats fMRI confounds for SPM analysis
%
% This function processes confound data from fMRIprep outputs, suitable for
% Statistical Parametric Mapping (SPM) analysis. It reads a JSON file with
% confound descriptions and a TSV file with confound values, then selects and
% formats the required confounds based on the specified denoising pipeline.
%
% Usage:
%   confounds = fMRIprepConfounds2SPM(json_path, tsv_path, pipeline)
%
% Inputs:
%   json_path (string): Full path to the JSON file. This file contains metadata
%                       about the confounds, such as their names and properties.
%
%   tsv_path (string):  Full path to the TSV file. This file holds the actual
%                       confound values in a tabular format for each fMRI run.
%
%   pipeline (cell array of strings): Specifies the denoising strategies to be
%                                     applied. Each element is a string in the
%                                     format 'strategy-number'. For example,
%                                     'HMP-6' indicates using 6 head motion
%                                     parameters. Valid strategies include:
%             'HMP': Head Motion Parameters, options: 6, 12, 24
%             'GS': Global Signal, options: 1, 2, 4
%             'CSF_WM': CSF and White Matter signals, options: 2, 4, 8
%             'FD': Framewise Displacement, a raw non-binary value
%             'Null': Returns an empty table if no confounds are to be applied
%
% Outputs:
%   confounds (table): A table containing the selected confounds, formatted for
%                      use in SPM. Each column represents a different confound,
%                      and each row corresponds to a time point in the fMRI data.
%
% Author: Andrea Costantino
% Date: 23/1/2023
%
% Example:
%   confounds = fMRIprepConfounds2SPM('path/to/json', 'path/to/tsv', {'HMP-6', 'GS-4'});
%
% This example would extract and format 6 head motion parameters and the global
% signal (with raw, derivative, and squared derivative) for SPM analysis.

% Read the TSV file containing the confound values
tsv_run = readtable(tsv_path, 'FileType', 'text');

% Initialize an empty cell array to store the keys of the selected confounds
selected_keys = {};

% If 'Null' is found in the pipeline, return an empty table and exit the function
if any(strcmp(pipeline, 'Null'))
    disp('"Null" found in the pipeline. Returning an empty table.')
else
    % Process each specified strategy in the pipeline

    % Head Motion Parameters (HMP)
    if any(contains(pipeline, 'HMP'))
        % Extract and validate the specified number of head motion parameters
        idx = find(contains(pipeline, 'HMP'));
        conf_num_str = pipeline(idx(1));
        conf_num_str_split = strsplit(conf_num_str{1}, '-');
        conf_num = str2double(conf_num_str_split(2));
        if ~any([6, 12, 24] == conf_num)
            error('HMP must be 6, 12, or 24.');
        else
            % Add the appropriate head motion parameters to selected_keys
            hmp_id = floor(conf_num / 6);
            if hmp_id > 0
                selected_keys = [selected_keys, {'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'}];
            end
            if hmp_id > 1
                selected_keys = [selected_keys, {'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1', 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1'}];
            end
            if hmp_id > 2
                selected_keys = [selected_keys, {'rot_x_power2', 'rot_y_power2', 'rot_z_power2', 'trans_x_power2', 'trans_y_power2', 'trans_z_power2', 'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2', 'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2'}];
            end
        end
    end

    % Global Signal (GS)
    if any(contains(pipeline, 'GS'))
        % Extract and validate the specified level of global signal processing
        idx = find(contains(pipeline, 'GS'));
        conf_num_str = pipeline(idx(1));
        conf_num_str_split = strsplit(conf_num_str{1}, '-');
        conf_num = str2double(conf_num_str_split(2));
        if ~any([1, 2, 4] == conf_num)
            error('GS must be 1, 2, or 4.');
        else
            % Add the global signal parameters to selected_keys based on the specified level
            gs_id = conf_num;
            if gs_id > 0
                selected_keys = [selected_keys, {'global_signal'}];
            end
            if gs_id > 1
                selected_keys = [selected_keys, {'global_signal_derivative1'}];
            end
            if gs_id > 2
                selected_keys = [selected_keys, {'global_signal_derivative1_power2', 'global_signal_power2'}];
            end
        end
    end

    % CSF and WM masks global signal (CSF_WM)
    if any(contains(pipeline, 'CSF_WM'))
        % Extract and validate the specified level of CSF/WM signal processing
        idx = find(contains(pipeline, 'CSF_WM'));
        conf_num_str = pipeline(idx(1));
        conf_num_str_split = strsplit(conf_num_str{1}, '-');
        conf_num = str2double(conf_num_str_split(2));
        if ~any([2, 4, 8] == conf_num)
            error('CSF_WM must be 2, 4, or 8.');
        else
            % Add the CSF and WM parameters to selected_keys based on the specified level
            phys_id = floor(conf_num / 2);
            if phys_id > 0
                selected_keys = [selected_keys, {'white_matter', 'csf'}];
            end
            if phys_id > 1
                selected_keys = [selected_keys, {'white_matter_derivative1', 'csf_derivative1'}];
            end
            if phys_id > 2
                selected_keys = [selected_keys, {'white_matter_derivative1_power2', 'csf_derivative1_power2', 'white_matter_power2', 'csf_power2'}];
            end
        end
    end

    % MotionOutlier
    if any(contains(pipeline, 'MotionOutlier'))
        % Process motion outliers, either using pre-computed values or calculating them
        motion_outlier_keys = tsv_run.Properties.VariableNames(find(contains(tsv_run.Properties.VariableNames, {'non_steady_state_outlier', 'motion_outlier'})));
        selected_keys = [selected_keys, motion_outlier_keys];
    end

    % Framewise Displacement (FD)
    if any(contains(pipeline, 'FD'))
        % Add raw framewise displacement values to selected_keys
        % If the first row is 'n/a', replace it with 0
        fd_values = tsv_run.framewise_displacement;
        if isnan(fd_values(1))
            fd_values(1) = 0;
        end
        tsv_run.framewise_displacement = fd_values;
        selected_keys = [selected_keys, {'framewise_displacement'}];
    end
end
% Retrieve the selected confounds and convert them into a table
confounds_table = tsv_run(:, ismember(tsv_run.Properties.VariableNames, selected_keys));
confounds = fillmissing(confounds_table, 'constant', 0);

end

function gunzippedNii = gunzipNiftiFile(niiGzFile, outPath)
% gunzipNiftiFile - Decompress (gunzip) .nii.gz file and save it into a
% 'SPM/*/gunzipped' folder in the derivatives directory of a BIDS dataset
% (e.g., ./BIDS/derivatives/SPM/sub-xx/gunzipped).
%
% Author: Andrea Costantino
% Date: 3/2/2023
%
% Usage:
%   outPath = gunzipNiftiFile(niiGzFile, outPath)
%
% Inputs:
%    niiGzFile - String indicating the path to the input .nii file.
%    outPath - String indicating the root output directory.
%
% Outputs:
%    gunzippedNii - String indicating the new directory of the output file.
%

% Extract subject and task from nii file name
[~, niiGzName, niiGzExt] = fileparts(niiGzFile); % isolate the name of the nii.gz file
nameSplits = split(niiGzName, "_"); % split the nii file name into segments
selectedSub = nameSplits{1}; % subject is the first segment

% Infer the output path if not provided
if nargin < 2
    % get the BIDS root folder path
    splitPath = strsplit(niiGzFile, '/'); % split the string into folders
    idx = find(contains(splitPath, 'BIDS')); % find the index of the split that includes 'BIDS'
    bidsPath = strjoin(splitPath(1:idx), '/'); % create the full folder path
    outPath = fullfile(bidsPath, 'derivatives', 'SPM', 'gunzipped', selectedSub); % build the output path
end

% Check if the output folder already exists
if exist(outPath, 'dir') == 7
    fprintf('GUNZIP: Output directory already exists: %s.\n', outPath);
else
    mkdir(outPath); % create the output directory
    fprintf('GUNZIP: Created output directory: %s.\n', outPath);
end

% Check if the output file already exists
newFilePath = fullfile(outPath, niiGzName);

if exist(newFilePath, 'file') == 2
    fprintf('GUNZIP: Gunzipped file already exists: %s\n', newFilePath);
    gunzippedNii = {newFilePath};
else
    % gunzip them
    fprintf('GUNZIP: decompressing file %s ...\n', [niiGzName, niiGzExt])
    gunzippedNii = gunzip(niiGzFile, outPath);
    fprintf('GUNZIP: Created gunzipped file: %s\n', newFilePath);

    % Save a copy of this function in the output folder
    if exist(fullfile(outPath, 'smoothNiftiFile.m'), 'file') ~= 2
        copyfile([mfilename('fullpath'), '.m'], fullfile(outPath, 'gunzipNiftiFile.m'));
    end
end
end
function runSubstring = findRunSubstring(inputStr)
%FINDRUNSUBSTRING Extracts a 'run-xx' substring from a given string
%   This function takes an input string and searches for a substring that
%   matches the pattern 'run-xx', where 'xx' can be any one or two digit number.
%   If such a substring is found, it is returned; otherwise, an empty string
%   is returned.

% Regular expression to match 'run-' followed by one or two digits
pattern = 'run-\d{1,2}';

% Search for the pattern in the input string
matches = regexp(inputStr, pattern, 'match');

% Check if any match was found
if ~isempty(matches)
    % If a match was found, return the first match
    runSubstring = matches{1};
else
    % If no match was found, return an empty string
    runSubstring = '';
end
end

function filteredRows = filterRowsBySubstring(data, substring)
%FILTERROWSBYSUBSTRING Filters rows based on a substring in the first column
%   This function takes a cell array 'data' and a 'substring' as inputs,
%   and returns a new cell array 'filteredRows' containing only the rows
%   from 'data' where the first column includes the specified 'substring'.

% Initialize an empty cell array to store the filtered rows
filteredRows = {};

% Iterate through each row in the data
for rowIndex = 1:size(data, 1)
    % Fetch the first column of the current row
    currentEntry = data(rowIndex).name;

    % Check if the first column contains the specified substring
    if contains(currentEntry, ['_' substring '_'])
        % If it does, add the current row to the filteredRows array
        filteredRows = [filteredRows; data(rowIndex, :)];
    end
end
end

function spaceString = getSpaceString(niftiSpace)
% This function returns the space string based on the input niftiSpace
if strcmp(niftiSpace, 'T1w')
    spaceString = 'T1w';
elseif strcmp(niftiSpace, 'MNI')
    spaceString = 'MNI152NLin2009cAsym*';
else
    spaceString = niftiSpace;
end
end

function plotBoxcarAndHRFResponses(SPMstruct, outDir)
% plotBoxcarAndHRFResponses Visualize boxcar functions and convolved HRF responses per condition and session.
%
% This function generates a comprehensive visualization of the boxcar functions
% and their corresponding convolved hemodynamic response functions (HRFs) for
% each condition across all sessions, as defined in the SPM.mat structure.
%
% Usage:
%   plotBoxcarAndHRFResponses(SPM);
%   plotBoxcarAndHRFResponses(SPM, outDir);
%
% Inputs:
%   - SPM: A struct loaded from an SPM.mat file containing experimental design
%          and statistical model parameters.
%   - outDir: (Optional) A string specifying the directory to save the plot. If
%             provided, the plot is saved as a PNG file in the specified directory.
%
% Output:
%   - A figure is displayed with subplots representing each condition (row)
%     and session (column). Each subplot contains the boxcar function and the
%     convolved HRF for the corresponding condition and session.
%
% Example:
%   % Load the SPM.mat file
%   load('SPM.mat');
%
%   % Call the function to visualize
%   plotBoxcarAndHRFResponses(SPM);
%
%   % Save the plot to a directory
%   plotBoxcarAndHRFResponses(SPM, 'output_directory/');
%
% Notes:
%   - This function assumes that the SPM structure contains the following:
%     * SPM.Sess: Session-specific condition information.
%     * SPM.xY.RT: Repetition time (TR) in seconds.
%     * SPM.nscan: Number of scans per session.
%     * SPM.xX.X: Design matrix containing the convolved regressors.
%     * SPM.xX.name: Names of the columns in the design matrix.
%   - Ensure that the SPM.mat file corresponds to your specific fMRI data analysis.
%

% Get the number of sessions
SPM = SPMstruct.SPM;
num_sessions = length(SPM.Sess);

% Get the repetition time (TR)
TR = SPM.xY.RT;

% Get the number of scans per session
nscans = SPM.nscan;

% Calculate the cumulative number of scans to determine session boundaries
session_boundaries = [0 cumsum(nscans)];

% Determine the maximum number of conditions across all sessions
max_num_conditions = max(arrayfun(@(x) length(x.U), SPM.Sess));

% Create a new figure for plotting
figure;

% Adjust the figure size for better visibility
set(gcf, 'Position', [100, 100, 1400, 800]);

% Initialize subplot index
subplot_idx = 1;

% Define line styles and colors for boxcar and convolved HRF
boxcar_line_style = '-';
boxcar_line_color = [0, 0.4470, 0.7410]; % MATLAB default blue
boxcar_line_width = 1.5;

hrf_line_style = '-';
hrf_line_color = [0.8500, 0.3250, 0.0980]; % MATLAB default red
hrf_line_width = 1.5;

% Loop over each condition (regressor)
for cond_idx = 1:max_num_conditions
    % Loop over each session
    for sess_idx = 1:num_sessions
        % Create a subplot for the current condition and session
        subplot(max_num_conditions, num_sessions, subplot_idx);

        % Check if the current session has the current condition
        if length(SPM.Sess(sess_idx).U) >= cond_idx
            % Extract the condition structure
            U = SPM.Sess(sess_idx).U(cond_idx);

            % Get the condition name
            condition_name = U.name{1};

            % Get the onsets and durations of the events
            onsets = U.ons;
            durations = U.dur;

            % Get the number of scans (time points) in the current session
            num_scans = nscans(sess_idx);

            % Create the time vector for the current session
            time_vector = (0:num_scans - 1) * TR;

            % Initialize the boxcar function for the current session
            boxcar = zeros(1, num_scans);

            % Build the boxcar function based on onsets and durations
            for i = 1:length(onsets)
                onset_idx = floor(onsets(i) / TR) + 1;
                offset_idx = ceil((onsets(i) + durations(i)) / TR);
                onset_idx = max(onset_idx, 1);
                offset_idx = min(offset_idx, num_scans);
                boxcar(onset_idx:offset_idx) = 1;
            end

            % Find the rows corresponding to the current session in the design matrix
            session_row_start = session_boundaries(sess_idx) + 1;
            session_row_end = session_boundaries(sess_idx + 1);
            session_rows = session_row_start:session_row_end;

            % Find the columns in the design matrix corresponding to the current condition
            prefix = sprintf('Sn(%d) %s', sess_idx, condition_name);
            column_indices = find(strncmp(SPM.xX.name, prefix, length(prefix)));

            % Extract the convolved regressor(s) for the current condition and session
            convolved_regressor = sum(SPM.xX.X(session_rows, column_indices), 2);

            % Plot the boxcar function
            plot(time_vector, boxcar, 'LineStyle', boxcar_line_style, 'Color', boxcar_line_color, 'LineWidth', boxcar_line_width);
            hold on;

            % Plot the convolved HRF response
            plot(time_vector, convolved_regressor, 'LineStyle', hrf_line_style, 'Color', hrf_line_color, 'LineWidth', hrf_line_width);
            hold off;

            % Improve the appearance of the plot
            grid on;
            xlim([0, max(time_vector)]);
            ylim_min = min(min(boxcar), min(convolved_regressor)) - 0.1;
            ylim_max = max(max(boxcar), max(convolved_regressor)) + 0.1;
            ylim([ylim_min, ylim_max]);
            set(gca, 'FontSize', 8);

            % Add condition names as y-labels on the first column
            if sess_idx == 1
                ylabel(condition_name, 'FontSize', 10, 'Interpreter', 'none');
            else
                set(gca, 'YTick', []);
                set(gca, 'YTickLabel', []);
            end

            % Add x-labels on the bottom row
            if cond_idx == max_num_conditions
                xlabel('Time (s)', 'FontSize', 10);
            else
                set(gca, 'XTick', []);
                set(gca, 'XTickLabel', []);
            end

            % Add session titles on the first row
            if cond_idx == 1
                title(sprintf('Session %d', sess_idx), 'FontSize', 12);
            end
        else
            % If the condition is not present in the session
            axis off;
            text(0.5, 0.5, 'Not Present', 'HorizontalAlignment', 'center', 'FontSize', 12);

            % Add condition names as y-labels on the first column
            if sess_idx == 1
                ylabel(['Condition: ' num2str(cond_idx)], 'FontSize', 10, 'Interpreter', 'none');
            end

            % Add session titles on the first row
            if cond_idx == 1
                title(sprintf('Session %d', sess_idx), 'FontSize', 12);
            end
        end

        % Increment the subplot index
        subplot_idx = subplot_idx + 1;
    end
end

% Add an overall title for the figure
sgtitle('Boxcar and Convolved HRF Responses per Condition and Session', 'FontSize', 16);

% Save the plot as PNG if outDir is specified
if nargin > 1 && ~isempty(outDir)
    if ~isfolder(outDir)
        mkdir(outDir); % Create the directory if it doesn't exist
    end
    % Create a file name based on the current date and time
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    file_name = fullfile(outDir, ['BoxcarHRFResponses_' timestamp '.png']);
    saveas(gcf, file_name);
    fprintf('Figure saved to: %s\n', file_name);
end
close(gcf);

end

function saveSPMDesignMatrix(SPMstruct, outDir)
% saveSPMDesignMatrix Visualize and optionally save the design matrix from SPM.
%
% This function uses SPM's internal `spm_DesRep` function to display the design
% matrix and optionally saves the resulting figure as a PNG file.
%
% Usage:
%   saveSPMDesignMatrix(SPMstruct);
%   saveSPMDesignMatrix(SPMstruct, outDir);
%
% Inputs:
%   - SPMstruct: A struct loaded from an SPM.mat file containing experimental
%                design and statistical model parameters.
%   - outDir: (Optional) A string specifying the directory to save the figure.
%             If provided, the design matrix is saved as a PNG file in the specified
%             directory.
%
% Output:
%   - A figure is displayed showing the design matrix as produced by SPM.
%   - If `outDir` is provided, the design matrix is saved as a PNG file in the
%     specified directory.
%
% Example:
%   % Load the SPM.mat file
%   load('SPM.mat');
%
%   % Display the design matrix
%   saveSPMDesignMatrix(SPMstruct);
%
%   % Save the design matrix to a directory
%   saveSPMDesignMatrix(SPMstruct, 'output_directory/');
%
% Notes:
%   - Ensure that the SPM.mat file corresponds to your specific fMRI data analysis.
%   - This function depends on the SPM toolbox being properly set up and initialized.
%

SPM = SPMstruct.SPM;

% Check if the design matrix exists
if ~isfield(SPM, 'xX') || ~isfield(SPM.xX, 'X') || isempty(SPM.xX.X)
    error('The SPM structure does not contain a valid design matrix.');
end

% Use SPM's spm_DesRep function to display the design matrix
spm_DesRep('DesMtx', SPM.xX);

% Get the current figure handle (SPM's design matrix figure)
figHandle = gcf;

% Save the figure as a PNG if outDir is specified
if nargin > 1 && ~isempty(outDir)
    if ~isfolder(outDir)
        mkdir(outDir); % Create the directory if it doesn't exist
    end
    % Create a file name based on the current date and time
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    file_name = fullfile(outDir, ['SPMDesignMatrix_' timestamp '.png']);
    saveas(figHandle, file_name);
    fprintf('Design matrix saved to: %s\n', file_name);
end

% Close the figure after saving
close(figHandle);

end
