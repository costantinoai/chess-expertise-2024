classdef ROI
    % ROI  Represents a single Region of Interest (ROI) in MATLAB

    properties
        regionName              % e.g., 'L_V1_ROI'
        regionLongName          % e.g., 'Left Primary Visual (V1)'
        hemisphere              % 'L' or 'R'
        region                  % original region label from CSV (e.g., 'V1')
        lobe                    % lobe classification (e.g., 'occipital')
        cortex                  % cortex classification (e.g., 'primary visual')
        cortexID                % numeric ID associated with the cortex
        regionID                % numeric ID associated with the region (may be [])
        xCOG                    % x-coordinate of the center of gravity
        yCOG                    % y-coordinate of the center of gravity
        zCOG                    % z-coordinate of the center of gravity
        volmm                   % volume of the region in mm^3
        additionalAttributes    % struct for extra custom info

        % NEW PROPERTIES
        cortexName              % e.g., 'L_primary_visual_cortex'
        lobeName                % e.g., 'R_frontal_lobe'
    end

    methods
        function obj = ROI(regionName, regionLongName, hemisphere, region, ...
                lobe, cortex, cortexID, regionID, xCOG, yCOG, zCOG, volmm)
            % Constructor for the ROI object
            obj.regionName         = regionName;
            obj.regionLongName     = regionLongName;
            obj.hemisphere         = hemisphere;
            obj.region             = region;
            obj.lobe               = lobe;
            obj.cortex             = cortex;
            obj.cortexID           = cortexID;
            obj.regionID           = regionID;  % may be empty if unassigned
            obj.xCOG               = xCOG;
            obj.yCOG               = yCOG;
            obj.zCOG               = zCOG;
            obj.volmm              = volmm;
            obj.additionalAttributes = struct();  % Initialize as empty struct

            % -- Create the "cortexName" if cortex is non-empty --
            if ~isempty(obj.cortex)
                % Convert to lowercase, replace spaces with underscores
                cortex_str = lower(strrep(obj.cortex, ' ', '_'));
                % e.g., 'primary visual' => 'primary_visual'
                % then prepend hemisphere + suffix '_cortex'
                obj.cortexName = sprintf('%s_%s_cortex', obj.hemisphere, cortex_str);
            else
                obj.cortexName = '';
            end

            % -- Create the "lobeName" if lobe is non-empty --
            if ~isempty(obj.lobe)
                lobe_str = lower(strrep(obj.lobe, ' ', '_'));
                obj.lobeName = sprintf('%s_%s_lobe', obj.hemisphere, lobe_str);
            else
                obj.lobeName = '';
            end
        end

        function obj = addAttribute(obj, key, value)
            % Add (or update) a key-value pair in additionalAttributes
            obj.additionalAttributes.(key) = value;
        end

        function details = getDetails(obj)
            % Return all details of the ROI as a struct, including any additional attributes
            details = struct(...
                'Hemisphere',       obj.hemisphere, ...
                'Lobe',             obj.lobe, ...
                'Cortex',           obj.cortex, ...
                'CortexID',         obj.cortexID, ...
                'Region',           obj.region, ...
                'RegionName',       obj.regionName, ...
                'RegionLongName',   obj.regionLongName, ...
                'RegionID',         obj.regionID, ...
                'XCoG',             obj.xCOG, ...
                'YCoG',             obj.yCOG, ...
                'ZCoG',             obj.zCOG, ...
                'Volume_mm3',       obj.volmm, ...
                'cortexName',       obj.cortexName, ...
                'lobeName',         obj.lobeName ...
                );

            % Merge additionalAttributes fields into 'details'
            extraFields = fieldnames(obj.additionalAttributes);
            for i = 1:numel(extraFields)
                details.(extraFields{i}) = obj.additionalAttributes.(extraFields{i});
            end
        end
    end
end

