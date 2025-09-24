function all_items = getHierarchicalItems(mgr, hierarchicalLevel)
% GETHIERARCHICALITEMS Returns a list of unique items based on the
% hierarchicalLevel: 'region', 'cortex', or 'lobe'. Removes empty entries.

switch hierarchicalLevel
    case 'region'
        all_items = mgr.listUniqueValues('regionName');
    case 'cortex'
        all_items = mgr.listUniqueValues('cortexName');
    case 'lobe'
        all_items = mgr.listUniqueValues('lobeName');
    otherwise
        error('[ERROR] Unknown hierarchicalLevel: %s', hierarchicalLevel);
end

% Filter out empty strings (some ROIs might lack a cortex or lobe)
all_items = all_items(~strcmp(all_items, ''));
end