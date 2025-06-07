
% Paths to your files
csvPath = "/data/projects/chess/data/misc/HCP-MMP1_UniqueRegionList.csv";
leftColorTablePath = "/data/projects/chess/data/BIDS/derivatives/rois-HCP/sub-01/label/lh_HCPMMP1_color_table.txt";
rightColorTablePath = "/data/projects/chess/data/BIDS/derivatives/rois-HCP/sub-01/label/rh_HCPMMP1_color_table.txt";


% TEST_ROI_MANAGER
%
% A self-contained test script that demonstrates the functionality
% of ROI.m and ROIManager.m classes in MATLAB. This script:
%
% 1) Creates mock CSV and color table files on the fly.
% 2) Instantiates the ROIManager.
% 3) Exercises the ROIManager methods (load data, list keys, unique values, filtering, etc.).
% 4) Demonstrates how to add custom attributes to an ROI and retrieve them.

% Clean up any old test files
cleanupTestFiles();

%----------------------------------------------------------------------
% 1) Create minimal CSV file with mock ROI data
%----------------------------------------------------------------------
csvPath = fullfile(pwd, 'test_rois.csv');
createTestCSV(csvPath);

%----------------------------------------------------------------------
% 2) Create left and right color table files
%----------------------------------------------------------------------
leftColorTablePath  = fullfile(pwd, 'test_left_ctab.txt');
rightColorTablePath = fullfile(pwd, 'test_right_ctab.txt');
createTestColorTable(leftColorTablePath, 'L');  % "lh_" entries
createTestColorTable(rightColorTablePath, 'R'); % "rh_" entries

%----------------------------------------------------------------------
% 3) Instantiate ROIManager and load the data
%----------------------------------------------------------------------
mgr = ROIManager(csvPath, leftColorTablePath, rightColorTablePath);

% Sanity check: how many ROIs did we load?
fprintf('Loaded %d ROI(s)\n', numel(mgr.rois));

%----------------------------------------------------------------------
% 4) Demonstrate ROIManager functionality
%----------------------------------------------------------------------

% (A) List all keys (fields) in the ROI
allKeys = mgr.listKeys();
fprintf('\nAll ROI keys:\n');
disp(allKeys);

% (B) List unique hemisphere values
uniqueHemis = mgr.listUniqueValues('hemisphere');
fprintf('\nUnique hemisphere values:\n');
disp(uniqueHemis);

% (C) List unique lobes
uniqueLobes = mgr.listUniqueValues('lobe');
fprintf('\nUnique lobe values:\n');
disp(uniqueLobes);

% (D) Filter by hemisphere = 'L'
leftROIs = mgr.getByFilter('hemisphere','L');
fprintf('\nNumber of left-hemisphere ROIs: %d\n', numel(leftROIs));

% (E) Filter by lobe containing 'occipital' (case-insensitive)
occipitalROIs = mgr.getByFilter('lobe','occipital');
fprintf('Number of occipital ROIs (any hemisphere): %d\n', numel(occipitalROIs));

% (F) Filter by exact cortexID
roisWithCortexID2 = mgr.getByFilter('cortexID',2);
fprintf('Number of ROIs with cortexID=2: %d\n', numel(roisWithCortexID2));

% (G) Filter by regionID (once assigned from color table)
% The example color tables assign region IDs 101, 102, etc. (see createTestColorTable)
roisRegionID102 = mgr.getByFilter('regionID',102);
fprintf('Number of ROIs with regionID=102: %d\n', numel(roisRegionID102));

%----------------------------------------------------------------------
% 5) Demonstrate adding custom attributes to an ROI
%----------------------------------------------------------------------
if ~isempty(mgr.rois)
    firstROI = mgr.rois(1);
    fprintf('\nOriginal first ROI details:\n');
    disp(firstROI.getDetails());

    % Add a custom attribute
    firstROI = firstROI.addAttribute('experimentDate', '2025-01-15');
    firstROI = firstROI.addAttribute('notes', 'This is an example ROI');

    fprintf('Updated first ROI details (with custom attributes):\n');
    disp(firstROI.getDetails());
end

%----------------------------------------------------------------------
% 6) Clean up temporary test files (comment out if you want to inspect them)
%----------------------------------------------------------------------
cleanupTestFiles();

fprintf('\nAll tests/demos completed successfully!\n');



%==========================================================================
%           Helper Functions for Creating Test Files
%==========================================================================

function createTestCSV(filename)
% Create a small CSV file with a few ROIs in table format.
% Fields needed by ROIManager.loadFromCSV:
%   'LR','region','regionLongName','Lobe','cortex','Cortex_ID',
%   'x-cog','y-cog','z-cog','volmm'

% We'll create a minimal table with both L and R hemispheres
LR    = {'L','L','R','R'}';
region = {'V1','MT','V1','MT'}';
regionLongName = {'Left_Visual1','Left_MiddleTemporal','Right_Visual1','Right_MiddleTemporal'}';
Lobe  = {'occipital','temporal','occipital','temporal'}';
cortex = {'primary_visual','motion_processing','primary_visual','motion_processing'}';
Cortex_ID = [1; 2; 1; 2];
x_cog = [ -10; -30; 12; 28 ];
y_cog = [ -70; -60; -68; -56 ];
z_cog = [  2;   10;   3;    8 ];
volmm = [ 500; 600; 520; 610 ];

T = table(LR, region, regionLongName, Lobe, cortex, Cortex_ID, x_cog, y_cog, z_cog, volmm);
writetable(T, filename);
fprintf('Created test CSV file: %s\n', filename);
end


function createTestColorTable(filename, hemisphere)
% Create a small color table file for the given hemisphere ('L' or 'R').
%
% Example format for color table (FreeSurfer-style):
%  region_id region_name R G B A
%
% We'll produce lines like:
%  101 lh_V1_ROI  120 120 120 0
%  102 lh_MT_ROI  140 140 140 0
% for the left hemisphere,
% or
%  201 rh_V1_ROI  220 220 220 0
%  202 rh_MT_ROI  240 240 240 0
% for the right hemisphere.
%
% We'll skip region_id=0 and ??? entries.

fid = fopen(filename, 'w');
if fid == -1
    error('Could not open file for writing: %s', filename);
end

% Write a comment line
fprintf(fid, '# Color table for hemisphere %s\n', hemisphere);

switch upper(hemisphere)
    case 'L'
        fprintf(fid, '101 lh_V1_ROI  120 120 120 0\n');
        fprintf(fid, '102 lh_MT_ROI  140 140 140 0\n');
    case 'R'
        fprintf(fid, '201 rh_V1_ROI  220 220 220 0\n');
        fprintf(fid, '202 rh_MT_ROI  240 240 240 0\n');
    otherwise
        fclose(fid);
        error('Hemisphere must be L or R');
end

fclose(fid);
fprintf('Created test color table file: %s\n', filename);
end


function cleanupTestFiles()
% Remove the test CSV and color table files if they exist
filesToDelete = {
    'test_rois.csv', ...
    'test_left_ctab.txt', ...
    'test_right_ctab.txt'
    };

for i = 1:numel(filesToDelete)
    f = fullfile(pwd, filesToDelete{i});
    if exist(f, 'file') == 2
        delete(f);
    end
end
end
