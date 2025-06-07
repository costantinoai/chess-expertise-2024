function ds_slice = sliceDatasetByHierarchy(mgr, ds, roi_data, unique_labels_in_data, hierarchicalLevel, this_item)
    % Function to filter ROIs by a hierarchical level (region/cortex/lobe),
    % find their IDs, and create a mask to slice and clean the dataset.
    %
    % Parameters:
    % mgr: The manager object with the getByFilter method.
    % ds: The dataset to slice.
    % roi_data: The ROI data corresponding to the dataset.
    % unique_labels_in_data: Unique region IDs present in the data.
    % hierarchicalLevel: The level of hierarchy ('region', 'cortex', 'lobe').
    % this_item: The specific region/cortex/lobe name to process.
    %
    % Returns:
    % ds_slice: The sliced and cleaned dataset. Empty if too few features.

    % (a) Filter ROIs based on the hierarchical level
    switch hierarchicalLevel
        case 'region'
            these_rois = mgr.getByFilter('regionName', this_item);
        case 'cortex'
            these_rois = mgr.getByFilter('cortexName', this_item);
        case 'lobe'
            these_rois = mgr.getByFilter('lobeName', this_item);
        otherwise
            error('Invalid hierarchical level: %s', hierarchicalLevel);
    end

    % (b) Extract valid region IDs
    regionIDs = intersect([these_rois.regionID], unique_labels_in_data);
    if isempty(regionIDs)
        fprintf('[WARN] No valid IDs for "%s". Returning NaN.', this_item);
        ds_slice = [];
        return;
    end

    % (c) Build mask for the region IDs
    mask = ismember(roi_data, regionIDs);

    % (d) Slice dataset
    ds_slice = cosmo_slice(ds, mask, 2);
    ds_slice = cosmo_remove_useless_data(ds_slice);
    cosmo_check_dataset(ds_slice)

    % (e) Check if the resulting dataset has enough features
    if size(ds_slice.samples, 2) < 6
        fprintf('[WARN] "%s": Too few features. Skipping.', this_item);
        ds_slice = [];
        return;
    end
end