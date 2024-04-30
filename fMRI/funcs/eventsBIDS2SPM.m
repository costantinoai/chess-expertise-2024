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