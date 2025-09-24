function results_table = performRSACorrelation(...
    ds, roiPath, regressorRDMs, outDir, colorTablePath, mgr, hierarchicalLevel)
% performRSACorrelation
%
% Runs correlation-based RSA at one of three hierarchical levels:
%   - 'region'  => uses ROI.regionName
%   - 'cortex'  => uses ROI.cortexName (hemisphere-specific, e.g. 'L_primary_visual_cortex')
%   - 'lobe'    => uses ROI.lobeName   (hemisphere-specific, e.g. 'R_occipital_lobe')
%
% CoSMoMVPA's cosmo_target_dsm_corr_measure is applied once per unique
% hierarchical name (e.g., 'L_primary_visual_cortex'), which is automatically
% formed in the ROI constructor using hemisphere + string transformations.
%
% Inputs:
%   ds (struct)            - CoSMoMVPA dataset
%   roiPath (string)       - Path to multi-label ROI NIfTI
%   regressorRDMs (struct) - Struct with fields for each regressor (RDM)
%   outDir (string)        - Output directory
%   colorTablePath (string)- Path to the color table file
%   mgr (ROIManager)       - ROI manager instance
%   hierarchicalLevel (char)- 'region', 'cortex', or 'lobe'
%
% Outputs:
%   results_table (table)
%       Rows = regressors
%       Columns = unique hierarchical names (already hemisphere-specific
%                if 'cortex' or 'lobe')
%
% Example:
%   >> regressorRDMs = struct('Checkmate',rdm_check,'Visual',rdm_vis);
%   >> tblCorr = performRSACorrelation(dsAvg, 'roi.nii', regressorRDMs,...
%       'outDir', 'colorTable.txt', mgr, 'cortex');
%   Columns might be 'L_primary_visual_cortex','R_primary_visual_cortex',...
%   etc.
%
%

% 1) Logging & Validation
fprintf('[INFO] Starting correlation-based RSA...\n');
regressorNames = fieldnames(regressorRDMs);
fprintf('[INFO] Found %d regressors.\n', numel(regressorNames));

valid_levels = {'region','cortex','lobe'};
if ~ismember(hierarchicalLevel, valid_levels)
    error('[ERROR] hierarchicalLevel must be one of: %s', ...
        strjoin(valid_levels, ', '));
end

% 2) Load ROI data & color table
[roi_data, unique_labels_in_data] = loadAndIntersectROI(roiPath, colorTablePath);
fprintf('[INFO] Found %d unique ROI labels in data.\n', numel(unique_labels_in_data));

% 3) Gather unique hierarchical items
unique_items = getHierarchicalItems(mgr, hierarchicalLevel);
nItems = numel(unique_items);
fprintf('[INFO] Found %d items at level=%s.\n', nItems, hierarchicalLevel);

% 4) Prepare measure
measure = @cosmo_target_dsm_corr_measure;
measure_args = struct('center_data', true);

% 5) Initialize results table
nRegressors = numel(regressorNames);
results_matrix = zeros(nRegressors, nItems);
col_names = matlab.lang.makeValidName(strrep(unique_items, ' ', '_'));
results_table = array2table(results_matrix, 'VariableNames', col_names);
results_table = addvars(results_table, regressorNames(:), ...
    'Before', 1, 'NewVariableNames','target');

% 6) Main Loop: For each hierarchical item => get regionIDs => slice => measure
for colIdx = 1:nItems
    this_item = unique_items{colIdx};
    fprintf('\n[INFO] Processing "%s"...\n', this_item);

    ds_slice = sliceDatasetByHierarchy(mgr, ds, roi_data, unique_labels_in_data, hierarchicalLevel, this_item);

   % Skip iteration if ds_slice is empty
    if isempty(ds_slice)
        fprintf('[INFO] Skipping iteration as ds_slice is empty.\n');
        continue; % Skip the current iteration
    end

    % (d) For each regressor => correlation
    for rIdx = 1:nRegressors
        measure_args.target_dsm = regressorRDMs.(regressorNames{rIdx});
        rsa_result = measure(ds_slice, measure_args);
        results_matrix(rIdx, colIdx) = rsa_result.samples;
    end
end

% 7) Store in table & Save
results_table{:,2:end} = results_matrix;

if ~exist(outDir, 'dir')
    mkdir(outDir);
end
outFile = fullfile(outDir, sprintf('rsa_corr_%s.tsv', hierarchicalLevel));
writetable(results_table, outFile, 'FileType','text','Delimiter','\t');

fprintf('\n[INFO] Correlation-based RSA table saved: %s\n', outFile);
end