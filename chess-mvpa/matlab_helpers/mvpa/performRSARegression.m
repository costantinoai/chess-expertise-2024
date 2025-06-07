function results_table = performRSARegression(...
    ds, roiPath, regressorRDMs, outDir, colorTablePath, mgr, hierarchicalLevel)
% performRSARegression
%
% Conducts a GLM-based RSA at one of three hierarchical levels:
%   - 'region' => uses ROI.regionName
%   - 'cortex' => uses ROI.cortexName
%   - 'lobe'   => uses ROI.lobeName
%
% We use CoSMoMVPA's cosmo_target_dsm_corr_measure with
% measure_args.glm_dsm = struct2cell(regressorRDMs), running a multi-regressor
% GLM for each masked ROI subset (region/cortex/lobe).
%
% Inputs:
%   ds (struct)             - CoSMoMVPA dataset
%   roiPath (string)        - Path to multi-label ROI NIfTI
%   regressorRDMs (struct)  - Struct with fields for each regressor (RDM)
%   outDir (string)         - Output directory
%   colorTablePath (string) - Path to the color table
%   mgr (ROIManager)        - ROI manager instance
%   hierarchicalLevel (char)- 'region','cortex','lobe'
%
% Outputs:
%   results_table (table)
%       Each row = one regressor
%       Each column = one hierarchical name (region/cortex/lobe).
%
% Example:
%   >> dsAvg.sa.targets = myStimuliTargets;
%   >> dsAvg = cosmo_fx(dsAvg, @(x)mean(x,1), 'targets');
%   >> regressorRDMs = struct('Checkmate',rdm_checkmate,'Visual',rdm_visual);
%   >> tblRSA = performRSARegression(dsAvg, 'roi.nii',...
%         regressorRDMs, '/outRSA','ctable.txt', mgr, 'cortex');
%
%
% 1) Logging & Basic Setup
fprintf('[INFO] Starting GLM-based RSA...\n');
regressorNames = fieldnames(regressorRDMs);
nRegressors = numel(regressorNames);
fprintf('[INFO] Found %d regressors.\n', nRegressors);

% 2) Validate hierarchicalLevel
valid_levels = {'region','cortex','lobe'};
if ~ismember(hierarchicalLevel, valid_levels)
    error('[ERROR] hierarchicalLevel must be one of: %s',...
        strjoin(valid_levels, ', '));
end

% 3) Load ROI data, color table
[roi_data, unique_labels_in_data] = loadAndIntersectROI(roiPath, colorTablePath);
fprintf('[INFO] Found %d unique labels.\n', numel(unique_labels_in_data));

% 4) Prepare measure
measure = @cosmo_target_dsm_corr_measure;
measure_args = struct('center_data', true);
measure_args.glm_dsm = struct2cell(regressorRDMs);

% 5) Identify items
all_items = getHierarchicalItems(mgr, hierarchicalLevel);
nItems = numel(all_items);

% 6) Initialize results table
results_matrix = zeros(nRegressors, nItems);
col_names = matlab.lang.makeValidName(strrep(all_items, ' ', '_'));
results_table = array2table(results_matrix, 'VariableNames', col_names);
results_table = addvars(results_table, regressorNames(:), 'Before',1,...
    'NewVariableNames','target');

% 7) Main Loop
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
    
    % (d) GLM-based measure => #Regressors x 1
    rsa_result = measure(ds_slice, measure_args);
    results_matrix(:,colIdx) = rsa_result.samples(:);
end

% 8) Save
results_table{:,2:end} = results_matrix;

if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, sprintf('rsa_glm_%s.tsv', hierarchicalLevel));
writetable(results_table, outFile, 'FileType','text','Delimiter','\t');

fprintf('\n[INFO] GLM-based RSA table saved: %s\n', outFile);
end
