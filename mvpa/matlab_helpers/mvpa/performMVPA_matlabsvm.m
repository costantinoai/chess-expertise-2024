function results_table = performMVPA_matlabsvm(...
    ds, targets, roiPath, outDir, colorTablePath, mgr, hierarchicalLevel)
% performMVPA
%
% Runs SVM classification in a cross-validation scheme, for multiple
% target label vectors, at one of three hierarchical levels:
%   - 'region' => uses ROI.regionName (e.g., 'L_V1_ROI')
%   - 'cortex' => uses ROI.cortexName (e.g., 'R_primary_visual_cortex')
%   - 'lobe'   => uses ROI.lobeName   (e.g., 'L_occipital_lobe')
%
% This function masks the dataset for each unique hierarchical name, then
% runs CoSMoMVPA n-fold SVM classification for each target vector. The
% resulting accuracies populate the rows (one per target vector) and the
% columns (one per hierarchical item).
%
% Inputs:
%   ds (struct)            - CoSMoMVPA dataset (e.g., after loading fMRI runs)
%   targets (cell array)   - Each cell is a vector of class labels
%                            (e.g., [1,2,1,2,...])
%   roiPath (string)       - Path to multi-label ROI NIfTI
%   outDir (string)        - Directory to save the output table
%   colorTablePath (string)- Path to the color table text file
%   mgr (ROIManager)       - ROI manager instance with new 'cortexName'/'lobeName'
%   hierarchicalLevel (char)- 'region','cortex','lobe'
%
% Outputs:
%   results_table (table)
%       Rows   = each entry in 'targets'
%       Columns= unique hierarchical names (regionName/cortexName/lobeName)
%
% Example:
%   >> t = {targets.checkmate, targets.categories};
%   >> tbl = performMVPA(ds, t, 'roi.nii', '/outSVM', 'colortable.txt', mgr, 'cortex')
%
%

% 1) Logging & Validation
fprintf('[INFO] Starting MVPA (SVM) analysis...\n');

% targets = rmfield(targets, 'stimuli');
nTargets = numel(fieldnames(targets));

fprintf('[INFO] Found %d target vectors.\n', nTargets);

valid_levels = {'region','cortex','lobe'};
if ~ismember(hierarchicalLevel, valid_levels)
    error('[ERROR] hierarchicalLevel must be one of: %s', ...
        strjoin(valid_levels, ', '));
end

% 2) Load & Subset ROI
[roi_data, unique_labels_in_data] = loadAndIntersectROI(roiPath, colorTablePath);
fprintf('[INFO] Found %d unique ROI labels.\n', numel(unique_labels_in_data));

% 3) Determine items from ROI manager
all_items = getHierarchicalItems(mgr, hierarchicalLevel);
left_filtered_items = all_items(startsWith(all_items, 'L_'));
all_items = left_filtered_items;
nItems = numel(all_items);
fprintf('[INFO] Found %d items at level=%s.\n', nItems, hierarchicalLevel);

% 4) Prepare results table
results_matrix  = zeros(nTargets, nItems);
column_var_names = matlab.lang.makeValidName(strrep(all_items, ' ', '_'));
results_table = array2table(results_matrix, 'VariableNames', column_var_names);

regressorNames = fieldnames(targets);
results_table = addvars(results_table, regressorNames(:), ...
    'Before', 1, 'NewVariableNames','target');

% 5) Define SVM classifier
classifier = @cosmo_classify_matlabcsvm;

% 6) Main Loop over items
for colIdx = 1:nItems
    this_item = all_items{colIdx};
    fprintf('\n[INFO] %s="%s"...\n', hierarchicalLevel, this_item);

    % Slice (select only voxels belonging to the selected mask)
    ds_slice = sliceDatasetByHierarchy(mgr, ds, roi_data, unique_labels_in_data, hierarchicalLevel, this_item);

    % Skip iteration if ds_slice is empty
    if isempty(ds_slice)
        fprintf('[INFO] Skipping iteration as ds_slice is empty.\n');
        continue; % Skip the current iteration
    end

    for tIdx = 1:nTargets
        if isempty(ds_slice)
            warning('[WARN] "%s": Too few features. Skipping Target#%d.\n',...
                this_item, tIdx);
            results_matrix(tIdx, colIdx) = NaN;
            continue;
        end

        ds_slice.sa.targets = targets.(regressorNames{tIdx});

        if numel(unique(ds_slice.sa.targets)) < 2
            warning('[WARN] Only one class in target %d. Skipping.\n', tIdx);
            results_matrix(tIdx, colIdx) = NaN;
            continue;
        end

        % Cross-validation
        partitions = cosmo_nfold_partitioner(ds_slice);
        partitions = cosmo_balance_partitions(partitions, ds_slice,'nmin',1);
        [~, accuracy] = cosmo_crossvalidate(ds_slice, classifier, partitions);

        results_matrix(tIdx, colIdx) = accuracy;
    end
end

% 7) Save
results_table{:,2:end} = results_matrix;

if ~exist(outDir,'dir'), mkdir(outDir); end
outFile = fullfile(outDir, sprintf('mvpa_cv_%s.tsv', hierarchicalLevel));
writetable(results_table, outFile, 'FileType','text','Delimiter','\t');

fprintf('\n[INFO] SVM results table saved: %s\n', outFile);
end
