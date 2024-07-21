% EXPBEHTOBIDS - Convert Behavioral .mat files to BIDS-compliant TSV files
%
% This script iterates through subject-specific directories, targeting 
% behavioral .mat files, then processes and exports trial-related 
% data into BIDS-compliant TSV event files.
%
% Assumptions:
% 1. Trial data is represented in a double array for each run.
% 2. 'trialList' structure houses conditions_id at column 4 and onset times at column 5.
% 3. Filename mapping for trial_ID and conditions is stored in fileList(i).name.
%
% Author: Andrea Costantino
% Modified by: Laura VH
% Last Modified: January 7, 2021
%
% Usage:
% 1. Set startSub and endSub for subject range.
% 2. Ensure directory paths are correctly set.
% 3. Run the script.

clear all;
clc;

% Define range of subjects
startSub = 41;
endSub = 44;

% Root directory for .mat files
baseInputDir = "/data/projects/chess/data/sourcedata";
% Root directory for output TSV files
outRoot = "/data/projects/chess/data/BIDS";

% Define trial duration (in seconds)
trialDur = 2.5;

% Generate list of subject IDs
subjects = arrayfun(@(x) sprintf('sub-%02d', x), startSub:endSub, 'UniformOutput', false);

fprintf('Processing data for subjects: %s to %s\n', subjects{1}, subjects{end});

% Loop through subjects
for subIndex = 1:numel(subjects)
    subID = subjects{subIndex};

    % Directories for each subject's input and output data
    srcDir = fullfile(baseInputDir, subID, 'bh');
    outDir = fullfile(outRoot, subID, 'func');
    
    % Check for source directory's existence
    if ~exist(srcDir, 'dir')
        error('Source directory %s not found!', srcDir);
    end

    % Create output directory if it doesn't exist
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % Fetch list of .mat files for current subject
    fileList = dir(fullfile(srcDir, '*exp.mat'));
    
    fprintf('Processing files for subject: %s\n', subID);

    % Loop through each .mat file for the current subject
    for i = 1:length(fileList)
        fprintf('Loading file: %s\n', fileList(i).name);
        
        % Load the .mat file
        load(fullfile(srcDir, fileList(i).name));
        
        % Initialize storage for trial data
        trialData = cell(size(trialList, 1), 3);
        
        % Loop through trialList to extract trial details
        for j = 1:length(trialList)
            % Decompose trial name for condition and stimulus details
            trialStim = strsplit(imList(trialList(j, 4)).name, '_');
            
            % Ensure expected name format
            if length(trialStim) < 3
                error('Unexpected name format in imList for index %d.', trialList(j, 4));
            end
            
            % Extract condition and stimulus info
            trialCond = trialStim{1};
            name = strtok(trialStim{3}, '.');
            
            % Compile trial type info
            imshort = strcat(trialCond, '_', name);
            
            % Store trial details: onset, duration, and type
            trialData{j, 1} = trialList(j, 5);  
            trialData{j, 2} = trialDur;         
            trialData{j, 3} = imshort;          
        end
        
        % Convert extracted trial data to table format
        trialTable = cell2table(trialData, 'VariableNames', {'onset', 'duration', 'trial_type'});
        
        % Parse filename for run details
        [~, filename] = fileparts(fileList(i).name);
        parts = strsplit(filename, '_');
        run = parts{end-1};
        
        % Construct BIDS-compliant filename for TSV output
        tsvFilename = sprintf('%s_task-exp_run-%s_events.tsv', subID, run);
        
        % Export trial table to TSV file
        outFile = fullfile(outDir, tsvFilename);
        writetable(trialTable, outFile, 'filetype', 'text', 'delimiter', '\t');
        fprintf('Created TSV file: %s\n', outFile);
    end
end

fprintf('Processing complete!\n');
