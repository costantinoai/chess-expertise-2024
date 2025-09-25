function run_subject_glm(subPath, subName, selectedTasks, selectedRuns, ...
                         fmriprepRoot, BIDSRoot, outRoot, ...
                         tempDir, pipeline, niftiSpace, thresholds)
    % Insert new inner loop here to iterate over tasks
    fprintf('[%s] Starting GLM for %02d\n', datestr(now), subName);

    for selected_task_idx = 1:length(selectedTasks)

        % clearvars matlabbatch % Clear the variable matlabbatch from workspace

        %% SUBJECT AND TASK INFO

        % Get task info
        selectedTask = selectedTasks(selected_task_idx).name;       % Get current task
        contrasts = selectedTasks(selected_task_idx).contrasts;     % Get corresponding contrasts
        smoothBool = selectedTasks(selected_task_idx).smoothBool;   % Whether to smooth the images before GLM

        % Extract subject ID from subName using split and convert to number
        sub_id = strsplit(subName,'-');
        selectedSub = str2double(sub_id{2});

        % Check wheter this sub has the task, otherwise skip
        fullPath = fullfile(subPath, subName, 'func');
        
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
         
        % % Sort files by name and convert back to struct
        % eventsTsvFiles = table2struct(sortrows(struct2table(filterRun1Files(eventsTsvFiles)), 'name'));
        % json_confounds_files = table2struct(sortrows(struct2table(filterRun1Files(json_confounds_files)), 'name'));
        % tsv_confounds_files = table2struct(sortrows(struct2table(filterRun1Files(tsv_confounds_files)), 'name'));

        % Sort files by name and convert back to struct
        eventsTsvFiles = table2struct(sortrows(struct2table(eventsTsvFiles), 'name'));
        json_confounds_files = table2struct(sortrows(struct2table(json_confounds_files), 'name'));
        tsv_confounds_files = table2struct(sortrows(struct2table(tsv_confounds_files), 'name'));

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
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        % Set the mask path. This runs the GLM ONLY inside the mask!
        % Get the path for the subject's mask. The mask is going to be one folder
        % up from the func folder, in the anat folder, with name e.g.
        % /data/projects/chess/data/BIDS/derivatives/fmriprep/sub-01/anat/sub-01_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii
        % but with sub id corrected for the current subject
        mask_dir = fullfile(funcPathSub, '..', 'anat');
        mask_path = fullfile(mask_dir, strcat(subName, '_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii'));
        matlabbatch{1}.spm.stats.fmri_spec.mask = {mask_path};

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

        %% Save Boxcar plot and design matrix of estimated model
        SPMstruct = load(spmMatPath);

        % plotBoxcarAndHRFResponses(SPMstruct, outPath)
        % saveSPMDesignMatrix(SPMstruct, outPath)

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


        %% Save Contrast Plots

        % Iterate over contrasts to generate plots
        for constrastIdx = 1:length(selectedTasks(selected_task_idx).contrasts)

            % Iterate over thresholds to generate plots
            for thresholdIndex = 1:length(thresholds)

                % Generate and Save Contrast Overlay Images

                % Set crosshair coordinates (modify if needed)
                crossCoords = [40, -52, -18];

                % Set the index of the contrast to display (modify if needed)
                spmContrastIndex = constrastIdx;

                % Call the function to generate and save contrast overlay images
                pipelineStr = strjoin(pipeline, '_');
                generateContrastOverlayImages(spmMatPath, outPath, fmriprepRoot, subName, pipelineStr, thresholds{thresholdIndex}, spmContrastIndex, crossCoords);

            end
        end
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
sectionImgPath = fullfile(fmriprepRoot, subjectName, 'anat', [subjectName, '_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii']);
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