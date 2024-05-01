function confounds = fMRIprepConfounds2SPM(json_path, tsv_path, pipeline)
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
%             'aCompCor': CompCor, options: 10, 50
%             'MotionOutlier': Motion Outliers, options: FD > 0.5, DVARS > 1.5
%             'Cosine': Discrete Cosine Transform based regressors for HPF
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

% Open and read the JSON file, then parse it into a MATLAB structure
fid = fopen(json_path); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
json_run = jsondecode(str);

% Initialize an empty cell array to store the keys of the selected confounds
selected_keys = {};

% If 'Null' is found in the pipeline, return an empty table and exit the function
if any(strcmp(pipeline, 'Null'))
    disp('"Null" found in the pipeline. Returning an empty table.')
    return;
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

    % aCompCor
    if any(contains(pipeline, 'aCompCor'))
        % Extract and format aCompCor confounds based on the specified number
        csf_50_dict = json_run(ismember({json_run.Mask}, 'CSF') & ismember({json_run.Method}, 'aCompCor') & ~contains({json_run.key}, 'dropped'));
        wm_50_dict = json_run(ismember({json_run.Mask}, 'WM') & ismember({json_run.Method}, 'aCompCor') & ~contains({json_run.key}, 'dropped'));
        idx = find(contains(pipeline, 'aCompCor'));
        conf_num_str = pipeline{idx(1)}; 
        conf_num_str_split = strsplit(conf_num_str{1}, '-');
        conf_num = str2double(conf_num_str_split(2));
        if ~any([10, 50] == conf_num)
            error('aCompCor must be 10 or 50.');
        else
            % Select the appropriate aCompCor components and add them to selected_keys
            if conf_num == 10
                csf = sort(cell2mat(csf_50_dict.keys()));
                csf_10 = csf(1:5);
                wm = sort(cell2mat(wm_50_dict.keys()));
                wm_10 = wm(1:5);
                selected_keys = [selected_keys, csf_10, wm_10];
            elseif conf_num == 50
                csf_50 = cell2mat(csf_50_dict.keys());
                wm_50 = cell2mat(wm_50_dict.keys());
                selected_keys = [selected_keys, csf_50, wm_50];
            end
        end
    end

    % Cosine
    if any(contains(pipeline, 'Cosine'))
        % Extract cosine-based regressors for high-pass filtering
        cosine_keys = tsv_run.Properties.VariableNames(contains(tsv_run.Properties.VariableNames, 'cosine'));
        selected_keys = [selected_keys, cosine_keys];
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

    % Retrieve the selected confounds and convert them into a table
    confounds_table = tsv_run(:, ismember(tsv_run.Properties.VariableNames, selected_keys));
    confounds = fillmissing(confounds_table, 'constant', 0);
end