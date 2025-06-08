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
        evalc('spm_jobman(''run'', matlabbatch);');
        
        % Move file to correct folder
        movefile(fullfile(niiFolder, smoothFileName), newFilePath);
        fprintf('SMOOTH: Created smoothed file: %s\n', newFilePath);

        % Save a copy of this function in the output folder
        if exist(fullfile(outPath, 'smoothNiftiFile.m'), 'file') ~= 2
            copyfile([mfilename('fullpath'), '.m'], fullfile(outPath, 'smoothNiftiFile.m'));
        end
    end
end
