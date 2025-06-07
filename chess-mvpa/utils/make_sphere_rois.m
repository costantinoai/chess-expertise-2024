% MATLAB Script for Generating Spherical ROIs
% ================================================
% This script creates spherical ROIs with specified MNI coordinates, radii, and a reference image.
% Dependencies: MarsBaR, SPM, hop_roi_sphere function.
%
% Author: Andrea Costantino
% Date: October 2024

% Define output paths and options
outRoot = '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/results/test2';
opt = struct();
opt.space = 'MNI152NLin2009cAsym_res-2'; % Coordinate space
opt.dir.output = fullfile(outRoot, 'rois'); % Output directory for ROIs

% Define the reference image path (e.g., T1-weighted image)
m.referencePath = '/data/projects/chess/data/BIDS/derivatives/fmriprep/sub-01/func/sub-01_task-exp_run-2_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz';

% Define ROI parameters
m.radii = [5, 10]; % Radii for the spherical ROIs in mm
m.roisToCreate = struct(...
    'area', {'FFA', 'LOC', 'PPA', 'TPJ', 'PCC1', ...
             'CoS_PPA1', 'pMTL_OTJ', 'OTJ', 'pMTG', 'SMG1', ...
             'CoS_PPA2', 'RSC_PCC', 'SMG2', 'pMTG_OTJ', 'Caudatus'}, ...
    'coordsL', {[-38, -58, -14], [-44, -77, -12], [-30, -50, -10], [-56, -47, 33], [2, -30, 34], ...
                [33, 39, 12], [-47, -69, 8], [-47, -69, 8], [-60, -54, -3], [-60, -36, 36], ...
                [-32, -43, -11], [-10, -75, 16], [-63, -31, 33], [-35, -80, 25], [-15, 13, 11]}, ...
    'coordsR', {[40, -55, -12], [44, -78, -13], [30, -54, -12], [56, -47, 33], [], ...
                [30, 42, 9], [48, -69, 15], [55, -69, 14], [58, -52, 1], [63, -27, 42], ...
                [18, -52, 5], [38, -36, -13], [], [51, -69, 16], [11, 18, 10]} ...
);

% Display ROI parameters for verification
fprintf('Preparing to create ROIs with the following parameters:\n');
for i = 1:length(m.roisToCreate)
    currROI = m.roisToCreate(i);
    fprintf('  ROI: %s\n', currROI.area);
    fprintf('    MNI coordinates (left): [%s]\n', num2str(currROI.coordsL));
    fprintf('    MNI coordinates (right): [%s]\n', num2str(currROI.coordsR));
end

% Generate the ROIs
hop_roi_sphere(opt, m);

%% Helper function to create and save spherical ROIs
function hop_roi_sphere(opt, m)
    % Generates spherical ROIs and saves them as NIfTI files.
    % Parameters:
    %   opt - Contains output directory and coordinate space information.
    %   m - Contains reference image path, radii, and ROI coordinates.

    for radius = m.radii
        fprintf('\nCreating ROIs with radius: %d mm\n', radius);

        % Set output folder for current radius
        outputFolder = fullfile(opt.dir.output, sprintf('radius_%dmm', radius));
        if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

        % Load reference space using SPM functions
        refSPM = spm_vol(m.referencePath);
        referenceSpace = mars_space(refSPM);

        % Iterate through each ROI definition
        for i = 1:length(m.roisToCreate)
            currROI = m.roisToCreate(i);
            fprintf('\nProcessing ROI: %s\n', currROI.area);

            % Left hemisphere ROI
            if ~isempty(currROI.coordsL)
                create_and_save_roi(currROI.coordsL, 'L', currROI.area, radius, opt, referenceSpace, outputFolder);
            end

            % Right hemisphere ROI
            if ~isempty(currROI.coordsR)
                create_and_save_roi(currROI.coordsR, 'R', currROI.area, radius, opt, referenceSpace, outputFolder);
            end

            % Bilateral ROI
            if ~isempty(currROI.coordsL) && ~isempty(currROI.coordsR)
                create_and_save_roi_bilateral(currROI.coordsL, currROI.coordsR, currROI.area, radius, opt, referenceSpace, outputFolder);
            end
        end
    end
end

function create_and_save_roi(coords, hemi, area, radius, opt, referenceSpace, outputFolder)
    % Creates and saves a sphere ROI for given coordinates.
    % Parameters:
    %   coords - MNI coordinates for the ROI center.
    %   hemi - 'L' or 'R' indicating hemisphere.
    %   area - Label for the ROI.
    %   radius - Sphere radius in mm.
    %   opt - Options struct with output and space information.
    %   referenceSpace - MarsBaR space object for the brain reference.
    %   outputFolder - Directory to save the NIfTI file.

    roiLabel = sprintf('hemi-%s_label-%s', hemi, area);
    fprintf('  Creating %s ROI with radius %d mm\n', roiLabel, radius);

    % Create sphere ROI with MarsBaR
    sphereROI = maroi_sphere(struct('centre', coords, 'radius', radius, 'label', roiLabel, 'reference', referenceSpace));

    % Save ROI as NIfTI file
    filename = fullfile(outputFolder, sprintf('hemi-%s_space-%s_radius-%dmm_label-%s.nii', hemi, opt.space, radius, area));
    save_as_image(sphereROI, filename);
    fprintf('  Saved ROI: %s\n', filename);
end

function create_and_save_roi_bilateral(coordsL, coordsR, area, radius, opt, referenceSpace, outputFolder)
    % Creates and saves a bilateral ROI by combining left and right coordinates.
    % Parameters are identical to create_and_save_roi, but it combines two sets of coordinates.

    fprintf('  Creating bilateral ROI with radius %d mm for area %s\n', radius, area);

    % Create left and right hemisphere ROIs
    leftROI = maroi_sphere(struct('centre', coordsL, 'radius', radius, 'label', sprintf('hemi-L_label-%s', area)));
    rightROI = maroi_sphere(struct('centre', coordsR, 'radius', radius, 'label', sprintf('hemi-R_label-%s', area)));

    % Resample and combine hemispheres
    resampledLeftROI = maroi_matrix(leftROI, referenceSpace);
    resampledRightROI = maroi_matrix(rightROI, referenceSpace);
    combinedData = (struct(resampledLeftROI).dat > 0.5) | (struct(resampledRightROI).dat > 0.5);
    bilateralROI = maroi_matrix(struct('dat', combinedData, 'mat', referenceSpace.mat, 'label', sprintf('hemi-B_label-%s', area)));

    % Save as bilateral NIfTI file
    filename = fullfile(outputFolder, sprintf('hemi-B_space-%s_radius-%dmm_label-%s.nii', opt.space, radius, area));
    save_as_image(bilateralROI, filename);
    fprintf('  Saved bilateral ROI: %s\n', filename);
end