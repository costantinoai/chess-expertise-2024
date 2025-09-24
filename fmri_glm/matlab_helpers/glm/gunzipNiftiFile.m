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
        outPath = fullfile(bidsPath, 'derivatives', 'fmriprep-preSPM', 'gunzipped', selectedSub); % build the output path
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
