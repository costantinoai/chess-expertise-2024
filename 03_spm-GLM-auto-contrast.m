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

% Note: This script assumes the use of the SPM software for fMRI data analysis.

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
% IMPORTANT NOTE: this code uses filterRun1Files to remove the first run
% from the GLM! If you want to use run 1, remove this function
%
% Author: Andrea Ivan Costantino
% Date: 5 July 2023

clc
clear

%% PARAMETERS

% Path of fmriprep, BIDS and output folder
niftiSpace = 'MNI'; % T1w, MNI
fmriprepRoot = '/data/projects/chess/data/BIDS/derivatives/fmriprep';
BIDSRoot = '/data/projects/chess/data/BIDS';
outRoot = ['/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM/',niftiSpace,'/fmriprep-SPM-', niftiSpace, '-checknocheck'];
tempDir = '/media/costantino_ai/eiK-backup1/chess/temp/temp_spm';
% outRoot = '/home/eik-tb/Desktop/New Folder';

% Files to select
selectedSubjectsList = [37, 38, 39, 40];        % Must be list of integers or '*'
% selectedSubjectsList = '*';     % Must be list of integers or '*'
selectedRuns = '*';                         % Must be integer or '*'

% Define tasks, weights and constrasts as a structure
% selectedTasks(1).name = 'loc1';
% selectedTasks(1).contrasts = {'Faces > Objects', 'Objects > Scrambled', 'Scenes > Objects'};
% selectedTasks(1).weights(1) = struct('Faces', 1, 'Objects', -1, 'Scrambled', 0, 'Scenes', 0);
% selectedTasks(1).weights(2) = struct('Faces', 0, 'Objects', 1, 'Scrambled', -1, 'Scenes', 0);
% selectedTasks(1).weights(3) = struct('Faces', 0, 'Objects', -1, 'Scrambled', 0, 'Scenes', 1);
% selectedTasks(1).smoothBool = true; % Whether to smooth the images before GLM
% 
% selectedTasks(2).name = 'loc2';
% selectedTasks(2).contrasts = {'(PCC-1) Legal > Illegal', '(PCC-2) Legal > No Kings', '(TPJ) Legal > Scrambled'};
% selectedTasks(2).weights(1) = struct('legal', 1, 'illegal', -1, 'NoKings', 0, 'scrambled', 0);
% selectedTasks(2).weights(2) = struct('legal', 1, 'illegal', 0, 'NoKings', -1, 'scrambled', 0);
% selectedTasks(2).weights(3) = struct('legal', 1, 'illegal', 0, 'NoKings', 0, 'scrambled', -1);
% selectedTasks(2).smoothBool = true; % Whether to smooth the images before GLM

% Here we select the Check vs. non-check contrast, but it really doesn't matter since
% the mvpa will be run on the beta images (regressors) and not on the T maps
selectedTasks(1).name = 'exp';
selectedTasks(1).contrasts = {'Check > No-Check'};
selectedTasks(1).weights(1) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', -1);
selectedTasks(1).smoothBool = false; % Whether to smooth the images before GLM

% selectedTasks(1).name = 'exp';
% selectedTasks(1).contrasts = {'All > Rest'};
% selectedTasks(1).weights(1) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', 1);
% selectedTasks(1).smoothBool = true; % Whether to smooth the images before GLM

% Denoising pipeline strategy for SPM confound regression Must be a cell
% array of strings {}. Possible strategies:
%     HMP - Head motion parameters (6,12,24)
%     GS - Brain mask global signal (1,2,4)
%     CSF_WM - CSF and WM masks global signal (2,4,8)
%     aCompCor - aCompCor (10,50)
%     MotionOutlier - motion outliers FD > 0.5, DVARS > 1.5, non steady volumes
%     Cosine - Discrete cosine-basis regressors low frequencies -> HPF
%     FD - Raw framewise displacement
%     Null - returns a blank df
pipeline = {'HMP-6','GS-1','FD'};

% Get subjects folders from subjects list
sub_paths = findSubjectsFolders(fmriprepRoot, selectedSubjectsList);

%% RUN THE GLM FOR EACH SUBJECT AND TASK

for sub_path_idx = 1:length(sub_paths)

    % Insert new inner loop here to iterate over tasks
    for selected_task_idx = 1:length(selectedTasks)

        clearvars matlabbatch % Clear the variable matlabbatch from workspace

        %% SUBJECT AND TASK INFO

        % Get task info
        selectedTask = selectedTasks(selected_task_idx).name;       % Get current task
        contrasts = selectedTasks(selected_task_idx).contrasts;     % Get corresponding contrasts
        smoothBool = selectedTasks(selected_task_idx).smoothBool;   % Whether to smooth the images before GLM

        % Extract subject ID from subName using split and convert to number
        subPath = sub_paths(sub_path_idx);
        subName = subPath.name;
        sub_id = strsplit(subName,'-');
        selectedSub = str2double(sub_id{2});

        % Check wheter this sub has the task, otherwise skip
        fullPath = fullfile(subPath.folder, subPath.name, 'func');
        
        % Get a list of files in the directory
        files = dir(fullPath);
        
        % Check if any filenames contain the desired string (if the sub has
        % the specific task selected)
        fileNames = {files.name};
        containsTask = any(contains(fileNames, ['task-', selectedTask]));

        % If the string doesn't exist in any filenames, skip to the next iteration
        if ~containsTask
            warning(['Task ', selectedTask, ' not found for ' subName ' in ' fullPath '. Skipping..']);
            continue;
        end

        % Set output path
        outPath = fullfile(outRoot, 'GLM', subName, selectedTask);

        % Print status update
        fprintf('############################### \n# STEP: running %s - %s #\n############################### \n', subName, selectedTask)

        % Set paths for fMRI data and BIDS data
        funcPathSub = fullfile(fmriprepRoot,subName,'func');
        bidsPathSub = fullfile(BIDSRoot,subName,'func');

        %% FIND AND LOAD EVENTS AND CONFOUNDS FROM FMRIPREP FOLDER

        % If selected_runs includes all runs ('*'), get all .tsv events and
        % .json,.tsv confound files; otherwise, get specific runs.
        if ismember('*', selectedRuns)
            eventsTsvFiles = dir(fullfile(bidsPathSub, strcat(subName,'_task-',selectedTask,'_run-*_events.tsv')));
            json_confounds_files = dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask,'_run-*_desc-confounds_timeseries.json')));
            tsv_confounds_files = dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask,'_run-*_desc-confounds_timeseries.tsv')));
        else
            eventsTsvFiles = arrayfun(@(x) dir(fullfile(bidsPathSub, strcat(subName,'_task-',selectedTask, '_run-', sprintf('%01d', x), '_events.tsv'))), selected_runs, 'UniformOutput', true);
            json_confounds_files = arrayfun(@(x) dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask, '_run-', sprintf('%01d', x), '_events.json'))), selected_runs, 'UniformOutput', true);
            tsv_confounds_files = arrayfun(@(x) dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask, '_run-', sprintf('%01d', x), '_events.tsv'))), selected_runs, 'UniformOutput', true);
        end

        % Sort files by name and convert back to struct
        eventsTsvFiles = table2struct(sortrows(struct2table(filterRun1Files(eventsTsvFiles)), 'name'));
        json_confounds_files = table2struct(sortrows(struct2table(filterRun1Files(json_confounds_files)), 'name'));
        tsv_confounds_files = table2struct(sortrows(struct2table(filterRun1Files(tsv_confounds_files)), 'name'));

        % Assert that the number of events and confounds files match
        assert(numel(eventsTsvFiles) == numel(json_confounds_files) && numel(json_confounds_files) == numel(tsv_confounds_files), ...
            'Mismatch in number of TSV events, TSV confounds, and JSON confounds files in %s', funcPathSub)

        %% SPM MODEL PARAMETERS (NON RUN DEPENDENT)

        % Define the general model parameters
        matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(outPath);
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        % Model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1) =  cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        %% SPM RUN SETTINGS (E.G., EVENTS, CONFOUNDS, IMAGES)
        % Iterate over each run event file
        for runIdx = 1:numel(eventsTsvFiles)

            % Extract events
            events_struct = eventsBIDS2SPM(fullfile(eventsTsvFiles(runIdx).folder, eventsTsvFiles(runIdx).name));
            selectedRun = findRunSubstring(eventsTsvFiles(runIdx).name); % Get the exact run

            fprintf('## STEP: TASK %s - %s\n', selectedTask, selectedRun)

            % Select json and confounds tsv based on the run we selected above 
            jsonRows = filterRowsBySubstring(json_confounds_files, selectedRun); % get the row corresponding to the correct run
            confoundsRows = filterRowsBySubstring(tsv_confounds_files, selectedRun); % get the row corresponding to the correct run
            
            % Assert that jsonRows has length 1
            if length(jsonRows) ~= 1
                error('More than one JSON file found for the specified run. Please check the dataset.');
            elseif isempty(jsonRows)
                error('No JSON file found for the specified run. Please check the dataset.');
            else
                % Assign to jsonRow if exactly one row is found
                jsonRow = jsonRows{1, :}; % Adjust indexing as per your data structure
            end
            
            % Assert that confoundsRows has length 1
            if length(confoundsRows) ~= 1
                error('More than one TSV confounds file found for the specified run. Please check the dataset.');
            elseif isempty(confoundsRows)
                error('No TSV confounds file found for the specified run. Please check the dataset.');
            else
                % Assign to tsvRow if exactly one row is found
                confoundsRow = confoundsRows{1, :}; % Adjust indexing as per your data structure
            end
            
            confounds_array = fMRIprepConfounds2SPM(fullfile(jsonRow.folder, jsonRow.name),...
                fullfile(confoundsRow.folder, confoundsRow.name), pipeline);

            % Define NIFTI file name pattern and check for existing .nii files
            spaceString = getSpaceString(niftiSpace);
            filePattern = strcat(subName,'_task-', selectedTask, '_', selectedRun,'_space-',spaceString,'_desc-preproc_bold');
            niiFileStruct = dir(fullfile(fmriprepRoot,subName,'func', strcat(filePattern, '.nii')));

            % If multiple or no NIFTI files found, throw an error
            if size(niiFileStruct,1) > 1
                error('Multiple NIFTI files found for %s.', selectedRun)

                % If no .nii files found, check for .nii.gz files and decompress
            elseif isempty(niiFileStruct)
                niiGzFilePattern = fullfile(funcPathSub, strcat(filePattern, '.nii.gz'));
                niiGzFileStruct = dir(niiGzFilePattern);

                % If no .nii.gz files found
                if isempty(niiGzFileStruct)
                    warning('No NIFTI file found for run %s. SKIPPING!', selectedRun)
                    continue

                    % If multiple files are found for this run
                elseif size(niiGzFileStruct,1) > 1
                    error('Multiple NIFTI.GZ files found for this run.')

                    % If only one file is found for this run (expected)
                else
                    niiGzFileString = fullfile(niiGzFileStruct.folder, niiGzFileStruct.name);
                    gunzippedNii = gunzipNiftiFile(niiGzFileString, fullfile(tempDir, 'gunzipped', subName));
                    niiFileStruct = dir(gunzippedNii{1});
                end
            end

            % Construct full NIFTI path
            niiFileString = fullfile(niiFileStruct.folder, niiFileStruct.name);

            % Perform smoothing operation if smoothBool is true
            if smoothBool
                niiFileString = smoothNiftiFile(niiFileString, fullfile(tempDir, 'smoothed', subName));
            else
                fprintf('SMOOTH: smoothBool set to false for this task. Skipping...\n')
            end

            % Set images, events in SPM model specification for each run
            niiFileCell = {niiFileString};
            matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).scans = spm_select('expand', niiFileCell);
            % here we are setting the HPF to 100 seconds more than the run duration. this is to avoid filtering at all
            matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).hpf = (matlabbatch{1}.spm.stats.fmri_spec.timing.RT * size(matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).scans, 1)) + 100; 
            
            for cond_id=1:length(events_struct.names)
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).cond(cond_id).name = events_struct.names{cond_id};
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).cond(cond_id).onset = events_struct.onsets{cond_id};
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).cond(cond_id).duration = events_struct.durations{cond_id};
            end

            % Set confound regressors events in SPM model specification for each run
            for reg_id=1:size(confounds_array,2)
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).regress(reg_id).name = confounds_array.Properties.VariableNames{reg_id};
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).regress(reg_id).val = confounds_array{:, reg_id};
            end
        end

        %% RUN BATCHES 1 AND 2
        spm('defaults','fmri');
        spm_jobman('initcfg');
        fprintf('GLM: Running GLM for: %s - TASK:  %s\n', subName, selectedTask)
        spm_jobman('run', matlabbatch(1:2));
        fprintf('GLM: DONE!\n')

        %% FIND SPM.mat
        spmMatPath = fullfile(outPath, 'SPM.mat');
        if ~exist(spmMatPath, 'file')
            error('SPM.mat file not found in the specified output directory.');
        end

        %% MODIFY AND RUN BATCH 3 (CONTRASTS)
        matlabbatch{3}.spm.stats.con.spmmat(1) = {spmMatPath};

        for k = 1:length(contrasts)
            % Get weights using adjust_contrasts
            weights = adjust_contrasts(spmMatPath, selectedTasks(selected_task_idx).weights(k));

            matlabbatch{3}.spm.stats.con.consess{k}.tcon.weights = weights;
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.name = contrasts{k};
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.sessrep = 'none';
        end

        % Run batch 3
        fprintf('Setting contrasts..\n');
        spm_jobman('run', matlabbatch(3));
        fprintf('DONE!\n');

    end

    %% SAVE SCRIPT
    % Copy this script to output folder for replicability
    FileNameAndLocation=[mfilename('fullpath')];
    script_outdir=fullfile(outPath,'spmGLMautoContrast.m');
    currentfile=strcat(FileNameAndLocation, '.m');
    copyfile(currentfile,script_outdir);

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

function filteredFiles = filterRun1Files(files)
    % Initialize an array to hold the filenames that do not include 'run-1'
    filteredFiles = struct('name', {}, 'folder', {}, 'date', {}, 'bytes', {}, 'isdir', {}, 'datenum', {});

    % Loop through the files and filter out 'run-1'
    for k = 1:length(files)
        if isempty(strfind(files(k).name, 'run-1'))
            % Add the file to the filteredFiles struct array
            filteredFiles(end+1) = files(k);
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
        error('Invalid niftiSpace value');
    end
end
