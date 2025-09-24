function scans = expand_fmriprep_surfaces(fmriprepRoot, subName, selectedTask, selectedRun, spaceString, varargin)
% EXPAND_FMRIPREP_SURFACES
% 
% Merges L/R anatomical surfaces into a single bilateral mesh (hemi-LR)
% and expands L/R functional GIFTI data volume-by-volume into bilateral 
% functional files. Returns a cell array of the expanded functional 
% file paths for easy SPM usage (matlabbatch).
%
% USAGE:
%   scans = expand_fmriprep_surfaces(fmriprepRoot, subName, taskName, runID, spaceString)
%   scans = expand_fmriprep_surfaces(..., outPathOverride)
%
% INPUTS:
%   fmriprepRoot   : Path to top-level fMRIPrep derivatives (BIDS-style root).
%   subName        : Subject folder name, e.g. 'sub-01'.
%   selectedTask   : Task label,   e.g. 'exp'.
%   selectedRun    : Run label,    e.g. 'run-1'.
%   spaceString    : Space label,  e.g. 'fsaverage'.
%   outPathOverride: (Optional) If provided, overrides the default output:
%                    <fmriprepRoot>/derivatives/fmriprep-surf-expanded/
%
% OUTPUTS:
%   scans : cell array of expanded bilateral functional files, one per volume.
%
% NOTES:
%   1) Expects a single anatomical L/R surface each in sub-xx/anat.
%   2) Expects a single functional L/R GIFTI each in sub-xx/func.
%   3) If the function does not find exactly one matching file for each, it errors.
%   4) The final bilateral surfaces are stored in:
%      [outPath]/sub-xx/anat/      -> for the merged .surf.gii
%      [outPath]/sub-xx/func/      -> for each volume's .func.gii
%   5) Filenames are derived from the original, with 'hemi-L' or 'hemi-R' replaced by 'hemi-LR'.

%% ----------------- Parse optional outPathOverride ------------------------
if ~isempty(varargin)
    outPath = varargin{1};
    fprintf('[INFO] Using custom output path: %s\n', outPath);
else
    outPath = fullfile(fileparts(fmriprepRoot), 'fmriprep-surf-expanded');
    fprintf('[INFO] Using default output path: %s\n', outPath);
end

%% ----------------- Prepare output subfolders (anat/func) ----------------
anatOutFolder = fullfile(outPath, subName, 'anat');
funcOutFolder = fullfile(outPath, subName, 'func');

if ~exist(anatOutFolder, 'dir')
    mkdir(anatOutFolder);
end
if ~exist(funcOutFolder, 'dir')
    mkdir(funcOutFolder);
end

%% ========================================================================
%  =                          ANATOMICAL SURFACES                         =
%  ========================================================================

%% ----------------- 1) Locate L/R anatomical surfaces --------------------
% Adjust file search pattern if your actual naming differs.
% sub-01_hemi-R_pial.surf.gii
filePatternAnat_L = fullfile(fmriprepRoot, subName, 'anat', ...
    [subName, '_hemi-L_pial.surf.gii']);
filePatternAnat_R = fullfile(fmriprepRoot, subName, 'anat', ...
    [subName, '_hemi-R_pial.surf.gii']);


anatFileStruct_L = dir(filePatternAnat_L);
anatFileStruct_R = dir(filePatternAnat_R);

% Ensure exactly one file is found for each hemisphere
if length(anatFileStruct_L) ~= 1
    error('[ERROR] Found %d matches for left anatomical surface (expected 1): %s', ...
        length(anatFileStruct_L), filePatternAnat_L);
end
if length(anatFileStruct_R) ~= 1
    error('[ERROR] Found %d matches for right anatomical surface (expected 1): %s', ...
        length(anatFileStruct_R), filePatternAnat_R);
end

anatFile_L = fullfile(anatFileStruct_L.folder, anatFileStruct_L.name);
anatFile_R = fullfile(anatFileStruct_R.folder, anatFileStruct_R.name);

fprintf('[INFO] Found left anatomical surf:  %s\n', anatFile_L);
fprintf('[INFO] Found right anatomical surf: %s\n', anatFile_R);

%% ----------------- 2) Merge L/R surfaces into bilateral -----------------
M = gifti({anatFile_L, anatFile_R});
M = spm_mesh_join(M);

% Create new name by replacing 'hemi-L' or 'hemi-R' with 'hemi-LR' in the left-file's name
[~, oldName_L, ext_L] = fileparts(anatFile_L);
newBaseName = strrep(oldName_L, 'hemi-L', 'hemi-LR');
newBaseName = strrep(newBaseName, 'hemi-R', 'hemi-LR'); % in case user calls only R file

% Final bilateral surface name
anatMergedFile = fullfile(anatOutFolder, [newBaseName, ext_L]);

% Save as external binary for consistency
save(gifti(M), anatMergedFile, 'ExternalFileBinary');
fprintf('[INFO] Saved bilateral anatomical surf -> %s\n', anatMergedFile);

%% ========================================================================
%  =                          FUNCTIONAL SURFACES                          =
%  ========================================================================

%% ----------------- 3) Locate L/R functional GIFTI files -----------------
filePatternFunc_L = fullfile(fmriprepRoot, subName, 'func', ...
    [subName, '_task-', selectedTask, '_', selectedRun, ...
     '_hemi-L_space-', spaceString, '*_bold.func.gii']);
filePatternFunc_R = fullfile(fmriprepRoot, subName, 'func', ...
    [subName, '_task-', selectedTask, '_', selectedRun, ...
     '_hemi-R_space-', spaceString, '*_bold.func.gii']);

funcFileStruct_L = dir(filePatternFunc_L);
funcFileStruct_R = dir(filePatternFunc_R);

% Ensure exactly one file is found for each hemisphere
if length(funcFileStruct_L) ~= 1
    error('[ERROR] Found %d matches for left functional file (expected 1): %s', ...
        length(funcFileStruct_L), filePatternFunc_L);
end
if length(funcFileStruct_R) ~= 1
    error('[ERROR] Found %d matches for right functional file (expected 1): %s', ...
        length(funcFileStruct_R), filePatternFunc_R);
end

funcFile_L = fullfile(funcFileStruct_L.folder, funcFileStruct_L.name);
funcFile_R = fullfile(funcFileStruct_R.folder, funcFileStruct_R.name);

fprintf('[INFO] Found left functional:  %s\n', funcFile_L);
fprintf('[INFO] Found right functional: %s\n', funcFile_R);

%% ----------------- 4) Load the GIFTI files & combine volume columns -----
gL = gifti(funcFile_L);
gR = gifti(funcFile_R);

% For each volume, we must stack the data vertically [L; R].
% The # of volumes is the number of columns in cdata
if size(gL.cdata,2) ~= size(gR.cdata,2)
    error('[ERROR] Mismatch in # of volumes between L and R. L has %d, R has %d.', ...
        size(gL.cdata,2), size(gR.cdata,2));
end
nVol = size(gL.cdata, 2);

%% ----------------- 5) Prepare new bilateral filenames & store scans -----
% We'll base the new name on the left-file's name, but replace 'hemi-L' -> 'hemi-LR'
[~, oldNameFunc, extFunc] = fileparts(funcFile_L);
newFuncBaseName = strrep(oldNameFunc, 'hemi-L', 'hemi-LR');
newFuncBaseName = strrep(newFuncBaseName, 'hemi-R', 'hemi-LR'); % Just in case
baseFile = fullfile(funcOutFolder, [newFuncBaseName, extFunc]);

%% ----------------- 6) Loop through volumes, build bilateral GIFTIs ------
% 'scans' will have one row per volume
scans = cell(nVol, 1);

for iVol = 1:nVol
    % Combine data (rows = #vertices in L + #vertices in R)
    bilateralData = [gL.cdata(:, iVol); gR.cdata(:, iVol)];
    
    % Create new GIFTI object
    gg = gifti(bilateralData);
    
    % Provide metadata: link to the new merged surface
    % (We store the 'mesh.surf.gii' or the file we just saved above.)
    gg.private.metadata = struct('name', 'SurfaceID', 'value', anatMergedFile);
    
    % Build the output functional name with volume suffix e.g. _00001
    outFile = spm_file(baseFile, 'suffix', sprintf('_%05d', iVol));
    
    % Save with external binary
    save(gg, outFile, 'ExternalFileBinary');
    fprintf('[INFO] Saved bilateral functional volume %d -> %s\n', iVol, outFile);

    % Store path in 'scans'
    % SPM often wants "filename,volumeIndex", but here we store just the file
    % so SPM knows each file is a single volume.
    scans{iVol} = outFile;
end

fprintf('[INFO] Created total of %d expanded bilateral functional files.\n', nVol);
fprintf('[INFO] Done.\n');

end
