classdef ROIManager < handle
    % ROIManager  Manages a collection of ROI objects
    %
    %   - Loads ROI data from CSV
    %   - Loads region mappings from color table files
    %   - Provides filter methods and hierarchical queries

    properties
        rois            % Array of ROI objects
        regionMapping   % containers.Map: region_id -> region_name
    end

    methods
        function obj = ROIManager(csvPath, leftColorTablePath, rightColorTablePath)
            % Constructor: initializes and loads data from files

            % Initialize empty for property defaults
            obj.rois = ROI.empty();

            % For regionMapping, use a containers.Map to associate
            % integer keys (region IDs) with string values (region names).
            obj.regionMapping = containers.Map('KeyType', 'int32', 'ValueType', 'char');

            % Load ROI base data from the CSV file
            obj.loadFromCSV(csvPath);

            % Load region mappings for left hemisphere and assign region IDs
            obj.loadFromColorTable(leftColorTablePath, 'L');

            % Load region mappings for right hemisphere and assign region IDs
            obj.loadFromColorTable(rightColorTablePath, 'R');
        end

        function loadFromCSV(obj, csvPath)
            % Load ROIs from a CSV file using readtable
            opts = detectImportOptions(csvPath);
            T = readtable(csvPath, opts);

            % Iterate over rows in the table and build ROI objects
            for i = 1:height(T)
                hemisphere = string(T.LR(i));   % e.g., 'L' or 'R'
                region     = string(T.region(i));
                regionLongName = strrep(string(T.regionLongName(i)), '_', ' ');
                lobe       = string(T.Lobe(i));
                cortex     = strrep(string(T.cortex(i)), '_', ' ');  % remove underscores
                cortexID   = T.Cortex_ID(i);
                xCOG       = T.x_cog(i);
                yCOG       = T.y_cog(i);
                zCOG       = T.z_cog(i);
                volmm      = T.volmm(i);

                % Format region name as {hemisphere}_{region}_ROI, e.g., 'L_V1_ROI'
                formattedName = sprintf('%s_%s_ROI', hemisphere, region);

                % regionID is left as [] for now; will be assigned later
                newROI = ROI(...
                    formattedName, ...
                    regionLongName, ...
                    hemisphere, ...
                    region, ...
                    lobe, ...
                    cortex, ...
                    cortexID, ...
                    [], ...    % regionID (unassigned yet)
                    xCOG, ...
                    yCOG, ...
                    zCOG, ...
                    volmm ...
                    );
                obj.addROI(newROI);
            end
        end

        function loadFromColorTable(obj, colorTablePath, hemisphere)
            % Load region mappings (ID -> name) from a FreeSurfer-style color table file
            % Example line format:  <region_id> <region_name> R G B A

            fid = fopen(colorTablePath, 'r');
            if fid == -1
                error('Could not open color table file: %s', colorTablePath);
            end

            while ~feof(fid)
                line = fgetl(fid);

                % Skip empty or comment lines
                if ischar(line) && ~isempty(line) && ~startsWith(line, '#')
                    parts = strsplit(strtrim(line));
                    if numel(parts) >= 2
                        % regionID = str2double(parts{1}) + (1000 * strcmp(hemisphere, 'L')) + (2000 * strcmp(hemisphere, 'R'));
                        regionID = str2double(parts{1}) + (1000 * strcmp(hemisphere, 'R'));
                        regionName = parts{2};

                        % Skip if regionID==0 or regionName=='???'
                        if strcmp(regionName, '???')
                            continue;
                        end

                        % Update the regionMapping dictionary
                        obj.regionMapping(regionID) = regionName;
                    end
                end
            end
            fclose(fid);

            % After loading the color table, assign region IDs to matching ROIs
            obj.assignRegionIDs(hemisphere);
        end

        function assignRegionIDs(obj, hemisphere)
            % Assign region IDs to ROIs for a specific hemisphere by matching ROI names
            % e.g., 'L_V1_ROI' with color-table entry 'l_V1_ROI'
            % We'll do this by looking for regionMapping keys that start with 'l_' or 'r_'
            hemiLower = lower(hemisphere);

            % Build a name->ID map for the relevant hemisphere, e.g. only 'l_' or 'r_' entries
            nameToID = containers.Map('KeyType', 'char', 'ValueType', 'int32');

            allKeys = obj.regionMapping.keys();
            for k = 1:numel(allKeys)
                rid = allKeys{k};  % regionID
                regionName = obj.regionMapping(rid);

                % Check if region name starts with e.g., 'l_' or 'r_'
                if startsWith(lower(regionName), hemiLower)

                    % Insert into nameToID, with lower() for case-insensitive match
                    nameToID(lower(regionName)) = rid;

                end
            end

            % Now loop through the rois array and assign
            for i = 1:numel(obj.rois)
                if lower(obj.rois(i).hemisphere) == hemiLower
                    roiKey = lower(obj.rois(i).regionName);  % e.g., 'l_v1_roi'

                    if isKey(nameToID, roiKey)
                        obj.rois(i).regionID = nameToID(roiKey);
                    end
                end
            end
        end

        function addROI(obj, roi)
            % Add an ROI object to the internal list
            obj.rois(end+1) = roi;
        end

        function values = listUniqueValues(obj, propertyName)
            % List all unique values for a specific ROI property across all ROIs.
            % propertyName should match one of the ROI property names exactly.

            % Gather the property values
            allValues = arrayfun(@(r) r.(propertyName), obj.rois, 'UniformOutput', false);

            % Convert to string or numeric as appropriate, then unique
            %   (If numeric, cell2mat might be needed; if string, cellfun, etc.)
            %   As an example, let's just use string() to unify:
            strValues = string(allValues);
            uniqueStrValues = unique(strValues);

            % Convert back to numeric if it *looks* numeric
            % For simplicity, we'll just return the cell array of unique string values.
            % If you want natural sorting like 'natsort', you'd implement it with
            % e.g., a custom comparator or third-party function in MATLAB.
            values = uniqueStrValues;
        end

        function keys = listKeys(obj)
            % Return a list of attribute names by looking at the first ROI
            % and calling getDetails
            if isempty(obj.rois)
                keys = {};
                return
            end
            details = obj.rois(1).getDetails();
            keys = fieldnames(details);
        end

        function matchedROIs = getByFilter(obj, varargin)
            % Retrieve ROIs based on multiple possible hierarchical filters.
            %
            %   getByFilter('hemisphere','L','lobe','occipital',...)
            %
            %   Possible filter names (case-insensitive):
            %       - 'hemisphere' : 'L' or 'R'
            %       - 'lobe'       : substring match, case-insensitive
            %       - 'cortex'     : substring match, case-insensitive
            %       - 'region'     : substring match, case-insensitive
            %       - 'regionName' : exact or substring match, case-insensitive
            %       - 'cortexName' : exact or substring match, case-insensitive
            %       - 'lobeName'   : exact or substring match, case-insensitive
            %       - 'cortexID'   : exact match (integer)
            %       - 'regionID'   : exact match (integer)
            %
            % Returns an array of matching ROI objects, sorted by regionID.

            % Parse inputs as parameter/value pairs
            p = inputParser;
            addParameter(p, 'hemisphere', '', @ischar);
            addParameter(p, 'lobe', '', @ischar);
            addParameter(p, 'cortex', '', @ischar);
            addParameter(p, 'region', '', @ischar);
            addParameter(p, 'regionName', '', @ischar);
            addParameter(p, 'cortexName', '', @ischar);
            addParameter(p, 'lobeName', '', @ischar);
            addParameter(p, 'cortexID', [], @(x) isnumeric(x) || isempty(x));
            addParameter(p, 'regionID', [], @(x) isnumeric(x) || isempty(x));
            parse(p, varargin{:});

            h     = lower(p.Results.hemisphere);
            lo    = lower(p.Results.lobe);
            cx    = lower(p.Results.cortex);
            rg    = lower(p.Results.region);
            rgnm  = lower(p.Results.regionName);
            cxnm  = lower(p.Results.cortexName);
            lbnm  = lower(p.Results.lobeName);
            cID   = p.Results.cortexID;
            rID   = p.Results.regionID;

            % Start with all ROIs
            matchedROIs = obj.rois;

            % Filter by hemisphere if provided
            if ~isempty(h)
                matchedROIs = matchedROIs(arrayfun(@(r) strcmpi(r.hemisphere, h), matchedROIs));
            end

            % Filter by lobe substring if provided
            if ~isempty(lo)
                matchedROIs = matchedROIs(arrayfun(@(r) contains(lower(r.lobe), lo), matchedROIs));
            end

            % Filter by cortex substring if provided
            if ~isempty(cx)
                matchedROIs = matchedROIs(arrayfun(@(r) contains(lower(r.cortex), cx), matchedROIs));
            end

            % Filter by region substring if provided
            if ~isempty(rg)
                matchedROIs = matchedROIs(arrayfun(@(r) contains(lower(r.region), rg), matchedROIs));
            end

            % Filter by region name if provided
            if ~isempty(rgnm)
                matchedROIs = matchedROIs(arrayfun(@(r) contains(lower(r.regionName), rgnm), matchedROIs));
            end

            % Filter by cortex name if provided
            if ~isempty(cxnm)
                matchedROIs = matchedROIs(arrayfun(@(r) contains(lower(r.cortexName), cxnm), matchedROIs));
            end

            % Filter by lobe name if provided
            if ~isempty(lbnm)
                matchedROIs = matchedROIs(arrayfun(@(r) contains(lower(r.lobeName), lbnm), matchedROIs));
            end

            % Filter by cortexID if provided
            if ~isempty(cID)
                matchedROIs = matchedROIs(arrayfun(@(r) r.cortexID == cID, matchedROIs));
            end

            % Filter by regionID if provided
            if ~isempty(rID)
                matchedROIs = matchedROIs(arrayfun(@(r) ~isempty(r.regionID) && r.regionID == rID, matchedROIs));
            end

            % Sort by regionID (ascending); handle possible empty regionID
            regionIDs = arrayfun(@(r) ifelse(isempty(r.regionID), NaN, r.regionID), matchedROIs);
            [~, idx] = sort(regionIDs, 'ascend', 'MissingPlacement', 'last');
            matchedROIs = matchedROIs(idx);
        end
    end
end

% A small helper function (inline) similar to Python's conditional expression
function out = ifelse(cond, valTrue, valFalse)
if cond
    out = valTrue;
else
    out = valFalse;
end
end
