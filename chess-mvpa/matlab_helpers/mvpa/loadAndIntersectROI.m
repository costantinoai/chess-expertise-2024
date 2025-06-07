function [roi_data, unique_labels_in_data] = loadAndIntersectROI(roiPath, colorTablePath)
% LOADANDINTERSECTROI Loads ROI data from a NIfTI file and color table,
% then returns the numeric 'roi_data' and the intersection of unique ROI
% labels with the color table IDs.

% Load ROI dataset
roi_ds  = cosmo_fmri_dataset(roiPath);
roi_data = roi_ds.samples;

% Load color table
[ct_labels, ~] = loadColorTable(colorTablePath);

% Intersect unique ROI labels with those in the color table
unique_labels_in_data = intersect(unique(roi_data), ct_labels);
end

