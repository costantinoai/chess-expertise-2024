function roiPathStruct = prepareRoiFile(roiFilePattern)
% prepareRoiFile
%
% Ensures we have a usable multi-label ROI file (.nii). If necessary,
% checks for a .nii.gz and unzips it.
%
% Usage:
%   roiPathStruct = prepareRoiFile(roiFilePattern)
%
% Inputs:
%   roiFilePattern (string)
%       Expected path to the ROI file (without .gz)
%
% Outputs:
%   roiPathStruct (struct)
%       - Output of MATLAB's 'dir' function, pointing to the .nii file.
%       - If empty, no ROI file was found.
%
% Example:
%   >> roiFilePattern = '/path/sub-01_HCPMMP1_volume_MNI.nii'
%   >> roiInfo = prepareRoiFile(roiFilePattern)
%   >> disp(roiInfo)
%
% Steps:
%   1) Check if the .nii file exists
%   2) If not, check for .nii.gz
%   3) If found, unzip and return info
%   4) If not found, return empty

% Search for the .nii file
roiPathStruct = dir(roiFilePattern);

% If multiple or zero .nii files found
if size(roiPathStruct,1) > 1
    error('[ERROR] Multiple ROI files found for pattern: %s', roiFilePattern);
elseif isempty(roiPathStruct)
    warning('[WARN] No ROI file found for pattern: %s. Checking for .nii.gz...', roiFilePattern);

    % Check for .nii.gz
    roiFilePatternGz = [roiFilePattern '.gz'];
    roiPathStruct    = dir(roiFilePatternGz);

    if isempty(roiPathStruct)
        warning('[WARN] No .nii or .nii.gz found for ROI. Returning empty struct.');
        roiPathStruct = []; % Nothing found
        return;
    elseif size(roiPathStruct,1) > 1
        error('[ERROR] Multiple .nii.gz files found for pattern: %s', roiFilePatternGz);
    else
        % We have exactly one .nii.gz; unzip it
        gzFullPath = fullfile(roiPathStruct.folder, roiPathStruct.name);
        unzipped   = gunzipNiftiFile(gzFullPath, roiPathStruct.folder);
        roiPathStruct = dir(unzipped{1});
    end
end
end

