% TWOSAMPLE_SECONDLEVEL_MULTICONTRAST(smoothBool)
%
% Performs a two-sample t-test in SPM for multiple contrasts, comparing
% Experts vs Non-Experts. Optionally smooths the first-level contrast
% images before second-level analysis.
%
% Usage example:
%   >> TwoSample_SecondLevel_MultiContrast(true);   % with smoothing
%   >> TwoSample_SecondLevel_MultiContrast(false);  % without smoothing
%
% -------------------------------------------------------------------------
% PARAMETERS:
% 1) smoothBool:
%    - Boolean indicating whether to smooth the first-level contrast images
%      before the second-level analysis (true) or not (false).
%
% 2) EXPERT_SUBJECTS & NONEXPERT_SUBJECTS:
%    - Cell arrays of subject IDs belonging to each group.
%
% 3) rootDir:
%    - The root directory where first-level contrast images are stored.
%
% 4) contrastFiles (cell array):
%    - List of the contrast files (e.g. {'con_0001.nii','con_0002.nii','con_0003.nii'}).
%
% 5) fwhm:
%    - Smoothing kernel for Gaussian smoothing (e.g., [8 8 8]) if smoothBool = true.
%
% -------------------------------------------------------------------------
% This script creates separate output folders for each contrast, e.g.:
%   2ndLevel_ExpVsNonExp_con_0001
%   2ndLevel_ExpVsNonExp_con_0002
% Inside each folder, it runs a two-sample t-test with:
%   1) Experts group con images
%   2) Non-Experts group con images
% Then it creates two T contrasts:
%   - Experts > Non-Experts  ([1 -1])
%   - Non-Experts > Experts  ([-1 1])
%
% Written and tested with SPM12.
% -------------------------------------------------------------------------

%% 1. Check input
smoothBool = false;  % default: no smoothing if not specified
fprintf('Smoothing set to: %s\n', string(smoothBool));

%% 2. Define subject groups
EXPERT_SUBJECTS = { ...
    '03', '04', '06', '07', '08', '09', ...
    '10', '11', '12', '13', '16', '20', ...
    '22', '23', '24', '29', '30', '33', ...
    '34', '36' ...
    };

NONEXPERT_SUBJECTS = { ...
    '01', '02', '15', '17', '18', '19', ...
    '21', '25', '26', '27', '28', '32', ...
    '35', '37', '39', '40', '41', '42', ...
    '43', '44' ...
    };

%% 3. Set main parameters
rootDir    = '/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-6_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM';

% Define the contrasts you want to test across groups
contrastFiles = { ...
    'con_0001.nii', ...
    'con_0002.nii', ...
    % 'con_0003.nii', ...
    };

% If smoothing is applied, use this FWHM
fwhm = [0 0 0];  % e.g., 9mm isotropic

%% 4. Loop over each contrast
for c = 1:numel(contrastFiles)
    contrastFile = contrastFiles{c};

    % For naming the second-level folder
    [~, contrastBase, ~] = fileparts(contrastFile); % e.g. 'con_0001'
    secondLevelFolder = fullfile(rootDir, ['2ndLevel_ExpVsNonExp_smooth-6-0_' contrastBase]);

    if ~exist(secondLevelFolder, 'dir')
        mkdir(secondLevelFolder);
    end

    fprintf('\n====================================\n');
    fprintf('Processing contrast: %s\n', contrastFile);
    fprintf('Second-level folder: %s\n', secondLevelFolder);

    %% 4A. Get the contrast images for Experts
    if smoothBool
        fprintf('Smoothing contrast images for Experts...\n');
        expertConImages = smoothContrastImages(rootDir, EXPERT_SUBJECTS, contrastFile, fwhm);
    else
        fprintf('Using original (unsmoothed) contrast images for Experts...\n');
        expertConImages = getOriginalContrastPaths(rootDir, EXPERT_SUBJECTS, contrastFile);
    end

    %% 4B. Get the contrast images for Non-Experts
    if smoothBool
        fprintf('Smoothing contrast images for Non-Experts...\n');
        nonexpertConImages = smoothContrastImages(rootDir, NONEXPERT_SUBJECTS, contrastFile, fwhm);
    else
        fprintf('Using original (unsmoothed) contrast images for Non-Experts...\n');
        nonexpertConImages = getOriginalContrastPaths(rootDir, NONEXPERT_SUBJECTS, contrastFile);
    end

    %% 4C. Build and run the SPM job for a Two-Sample T-Test
    matlabbatch = {};

    % (a) Specify factorial design
    matlabbatch{1}.spm.stats.factorial_design.dir = {secondLevelFolder};
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = expertConImages;    % Experts
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = nonexpertConImages; % Non-Experts

    % Typically:
    matlabbatch{1}.spm.stats.factorial_design.des.t2.dept     = 0;  % Independence
    matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;  % Unequal variance
    matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca    = 0;  % No grand mean scaling
    matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova   = 0;  % No ANCOVA

    % Masking
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1; % No threshold
    matlabbatch{1}.spm.stats.factorial_design.masking.im         = 1; % Implicit masking
    matlabbatch{1}.spm.stats.factorial_design.masking.em         = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit     = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm    = 1;

    % (b) Model estimation
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(secondLevelFolder, 'SPM.mat')};

    % (c) Contrast specification
    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(secondLevelFolder, 'SPM.mat')};

    % Contrast 1: Experts > Non-Experts
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = ['Experts > Non-Experts: ' contrastBase];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    % Contrast 2: Non-Experts > Experts
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.name    = ['Non-Experts > Experts: ' contrastBase];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

    matlabbatch{3}.spm.stats.con.delete = 0;  % don't delete previous contrasts

    % (d) Run the job
    spm('Defaults','fMRI');
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);

    fprintf('Two-sample t-test complete for contrast: %s\n', contrastBase);
    fprintf('Results saved to: %s\n', secondLevelFolder);
end

fprintf('\nAll group (Experts vs Non-Experts) analyses are COMPLETE.\n');


%% ========================================================================
function smoothedImages = smoothContrastImages(rootDir, subjectList, contrastFile, fwhm)
% SMOOTHCONTRASTIMAGES Smooths contrast images for a list of subjects.
%
% Inputs:
%   - rootDir:       Root directory of the GLM results
%   - subjectList:   Cell array of subject IDs
%   - contrastFile:  Name of the contrast file to smooth (e.g., 'con_0001.nii')
%   - fwhm:          Smoothing kernel in mm (e.g., [8 8 8])
%
% Output:
%   - smoothedImages: Cell array of full paths to smoothed contrast images
%                     (with the prefix 's' added to the file name)
%
% Note:
%   We assume each subject's contrast image is at:
%       <rootDir>/sub-<ID>/exp/<contrastFile>
%
%   This function sets up a single spm_jobman batch with repeated smoothing
%   steps, one for each subject's contrast. The new file is typically named
%   'scon_0001.nii' or similar.

smoothedImages = {};
matlabbatch = {};
for i = 1:numel(subjectList)
    subjDir   = fullfile(rootDir, ['sub-' subjectList{i}], 'exp');
    inputFile = fullfile(subjDir, contrastFile);
    if ~exist(inputFile, 'file')
        error('File not found: %s', inputFile);
    end

    % Define path for the smoothed output file (prefix 's')
    [fileDir, fileName, fileExt] = fileparts(inputFile);
    smoothedFile = fullfile(fileDir, ['s' fileName fileExt]);

    % Add smoothing job
    matlabbatch{1}.spm.spatial.smooth.data{i,1} = inputFile; %#ok<*AGROW>
    smoothedImages{i,1} = [smoothedFile ',1'];  % store for later
end

% SPM smoothing parameters
matlabbatch{1}.spm.spatial.smooth.fwhm   = fwhm;
matlabbatch{1}.spm.spatial.smooth.dtype  = 0;   % same as input
matlabbatch{1}.spm.spatial.smooth.im     = 0;   % no implicit mask
matlabbatch{1}.spm.spatial.smooth.prefix = 's'; % prefix 's'

% Run
spm('Defaults','fMRI');
spm_jobman('initcfg');
spm_jobman('run', matlabbatch);

fprintf('Smoothing complete for %d subjects.\n', numel(subjectList));
end

%% ========================================================================
function originalImages = getOriginalContrastPaths(rootDir, subjectList, contrastFile)
% GETORIGINALCONTRASTPATHS Returns the original contrast image paths (no smoothing).
%
% Inputs:
%   - rootDir:       Root directory of the GLM results
%   - subjectList:   Cell array of subject IDs
%   - contrastFile:  Name of the contrast file (e.g., 'con_0001.nii')
%
% Output:
%   - originalImages: Cell array of full paths to the original contrast images
%                     (with volume index ",1")
%
% Note:
%   Each subject's contrast image is at:
%       <rootDir>/sub-<ID>/exp/<contrastFile>
%
%   We simply construct the paths with ",1" volume index.

originalImages = cell(numel(subjectList), 1);
for i = 1:numel(subjectList)
    subjDir = fullfile(rootDir, ['sub-' subjectList{i}], 'exp');
    thisFile = fullfile(subjDir, contrastFile);
    if ~exist(thisFile, 'file')
        error('File not found: %s', thisFile);
    end
    originalImages{i} = [thisFile ',1'];
end
end
