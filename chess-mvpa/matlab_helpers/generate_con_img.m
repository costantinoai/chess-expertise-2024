spmMatPath = '/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM/2ndLevel_Experts_con_0002/SPM.mat';
outputPath = '/home/eik-tb/Desktop/New Folder';
templatePath = '/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_space-MNI_res-02_T1w_brain_resampled.nii';

generateContrastMontageImages(spmMatPath, outputPath, 0.05, 1, templatePath, false)


function generateContrastMontageImages(spmMatPath, outputPath, thresholdValue, spmContrastIndex, templatePath, use_FWE)
% GENERATECONTRASTMONTAGEIMAGES Generates and saves contrast montage images for a given threshold
%
% This function loads the specified SPM.mat, thresholds the data for the
% given contrast index and p-value (uncorrected), and displays a slice
% montage over a template image. The montage is then saved to the
% outputPath directory.
%
% Inputs:
%   spmMatPath       - (string) Path to the SPM.mat file.
%   outputPath       - (string) Directory where images will be saved.
%   fmriprepRoot     - (string) Root directory of fMRIPrep outputs (used if 
%                              you need to fetch subject-specific T1, but 
%                              here we rely on 'templatePath' by default).
%   subjectName      - (string) Subject identifier (e.g., 'sub-01').
%   pipelineStr      - (string) Representation of your processing pipeline.
%   thresholdValue   - (numeric) P-value threshold (uncorrected).
%   spmContrastIndex - (integer) Index of the contrast in SPM.xCon to use 
%                              (default is 1).
%   templatePath     - (string) Path to the template image to use for the 
%                              montage background. If empty or not provided, 
%                              defaults to SPM's avg152T1.nii.
%   slicePositions   - (1D vector) Axial slice positions for the montage. 
%                              Defaults to a typical range for MNI (e.g., 
%                              -40:4:72).
%   orientation      - (string) Orientation for the montage. Typically 
%                              'axial', 'coronal', or 'sagittal'. 
%                              Defaults to 'axial'.
%
% Outputs:
%   None (the montage image is saved to disk as a PNG).
%
% Example usage:
%   generateContrastMontageImages('/path/to/SPM.mat', ...
%       '/output/dir', '/fmriprep/root', 'sub-01', 'GS-1_HMP-6', 0.001, ...
%       1, '', [], 'axial');
%
% Requirements:
%   - SPM12 in your MATLAB path.
%   - Valid SPM.mat with at least one defined contrast.
%
% Author: Your Name
% Date:   2025-02-07
% -------------------------------------------------------------------------

% ------------------------ Parse and Validate Inputs -----------------------


% ------------------------ Load and Validate SPM.mat ----------------------
fprintf('Loading SPM.mat from %s ...\n', spmMatPath);
load(spmMatPath, 'SPM');

% Ensure contrast index is valid
if spmContrastIndex > numel(SPM.xCon)
    error('[Error] Invalid contrast index (%d). There are only %d contrasts defined.', ...
        spmContrastIndex, numel(SPM.xCon));
end

% Fetch contrast name for labeling
contrastNameSPM = SPM.xCon(spmContrastIndex).name;

% --------------------- Prepare xSPM for Thresholding ---------------------

xSPM = struct();
xSPM.swd       = fileparts(spmMatPath);  % Directory of SPM.mat
xSPM.title     = contrastNameSPM;
xSPM.Ic        = spmContrastIndex;       % Contrast index
xSPM.Im        = [];                     % No explicit mask
xSPM.pm        = [];                     % No p-value masking
xSPM.Ex        = [];                     % No exclusive masking
if use_FWE == true
xSPM.thresDesc = 'FWE';  % request family-wise error correction
end
xSPM.u         = thresholdValue;         % Uncorrected p-value
xSPM.k         = 0;                      % Extent threshold (voxels)
xSPM.STAT      = 'T';                    % T-statistic
xSPM.thresDesc = 'none';                 % Uncorrected threshold

try
    [SPM, xSPM] = spm_getSPM(xSPM);
catch ME
    warning('[Warning] No suprathreshold voxels or other SPM error');
    return;
end

% If we reach here, xSPM contains the thresholded statistic map.

% -------------------------- Create Montage Plot --------------------------
% 1. Reset SPM graphics window
spm_figure('Close','Graphics');
Fgraph = spm_figure('GetWin','Graphics');
set(Fgraph,'Name','SPM Montage Display');

% 2. Clear any previous orthviews
spm_orthviews('Reset');


% 5. Set up a slice montage
nRows         = 1;                  % how many rows of slices
slicePositions = -40:4:72;          % axial slices from z=-40 to z=72
orientation   = 'axial';            % 'axial', 'sagittal', or 'coronal'

% 3. Display background template
hBg = spm_orthviews('Image', templatePath, 'Montage', nRows, slicePositions, orientation);

% 4. Overlay thresholded activation blobs on the template
spm_orthviews('AddBlobs', hBg, xSPM.XYZ, xSPM.Z, xSPM.M);

% 6. Redraw the orthviews
spm_orthviews('Redraw');


% -------------------------- Save Montage to File -------------------------
% Construct output filename
montageImgPath = fullfile(outputPath, sprintf('%s_montage.png', contrastNameSPM));

% Print (save) the figure in PNG format
print(Fgraph, '-dpng', '-r300', montageImgPath); % 300 DPI for good quality
fprintf('[Info] Montage saved as: %s\n', montageImgPath);

end
