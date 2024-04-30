function weight_vector = adjust_contrasts(spmMatPath, contrastWeights)
% ADJUST_CONTRASTS Adjust contrast weights according to the design matrix in SPM.
%
% DESCRIPTION:
% This function adjusts the specified contrast weights according to the design
% matrix in SPM, and provides a visual representation of the weights applied to
% the design matrix.
%
% INPUTS:
% spmMatPath: String
%   - Path to the SPM.mat file.
%     Example: '/path/to/SPM.mat'
%
% contrastWeights: Struct
%   - Specifies the weight of each condition in the contrast.
%     For wildcard specification, use '_WILDCARD_'. E.g., 'condition_WILDCARD_': weight
%     Example: struct('condition1', 1, 'condition2_WILDCARD_', -1)
%
% OUTPUTS:
% weight_vector: Numeric Vector
%   - A vector of weights for each regressor.
%     Example: [0, 1, -1, 0, ...]
%
% The function also generates a visual representation of the design matrix with
% the specified contrast weights.

% Load the SPM.mat
load(spmMatPath);
% Extracting regressor names from the SPM structure
regressor_names = SPM.xX.name;

% Generate weight vector based on SPM's design matrix and specified weights for the single contrast
weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names);

% % Plotting for visual verification
% figure;
% 
% % Display the design matrix
% imagesc(SPM.xX.X);  % Display the design matrix
% colormap('gray');   % Set base colormap to gray for design matrix
% hold on;
% 
% % Create a color overlay based on the weights
% for i = 1:length(weight_vector)
%     x = [i-0.5, i+0.5, i+0.5, i-0.5];
%     y = [0.5, 0.5, length(SPM.xX.X) + 0.5, length(SPM.xX.X) + 0.5];
%     if weight_vector(i) > 0
%         % Green for positive weights
%         color = [0, weight_vector(i), 0];  % Green intensity based on weight value
%         patch(x, y, color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);  % Reduced transparency
%     elseif weight_vector(i) < 0
%         % Red for negative weights
%         color = [abs(weight_vector(i)), 0, 0];  % Red intensity based on absolute weight value
%         patch(x, y, color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);  % Reduced transparency
%     end
% end
% 
% % Annotate with regressor names
% xticks(1:length(regressor_names));
% xticklabels('');  % Initially empty, to be replaced by colored text objects
% xtickangle(45);  % Angle the text so it doesn't overlap
% set(gca, 'TickLabelInterpreter', 'none');  % Ensure special characters in regressor names display correctly
% 
% % Color code the regressor names using text objects
% for i = 1:length(regressor_names)
%     if weight_vector(i) > 0
%         textColor = [0, 0.6, 0];
%     elseif weight_vector(i) < 0
%         textColor = [0.6, 0, 0];
%     else
%         textColor = [0, 0, 0];
%     end
%     text(i, length(SPM.xX.X) + 5, regressor_names{i}, 'Color', textColor, 'Rotation', 45, 'Interpreter', 'none', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
% end
% 
% title('Design Matrix with Contrast Weights');
% xlabel('');
% ylabel('Scans');
% 
% % Add legends
% legend({'Positive Weights', 'Negative Weights'}, 'Location', 'northoutside');
% 
% % Optional: Add a dual color colorbar to represent positive and negative weight intensities
% colorbar('Ticks', [-1, 0, 1], 'TickLabels', {'-Max Weight', '0', '+Max Weight'}, 'Direction', 'reverse');
% 
% hold off;
end

function weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names)
% GENERATE_WEIGHT_VECTOR_FROM_SPM Generates a weight vector from the SPM design matrix.
%
% This function constructs a weight vector based on the design matrix in SPM
% and the user-specified contrast weights. It's equipped to handle wildcard matches
% in condition names for flexibility in defining contrasts.
%
% USAGE:
%   weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names)
%
% INPUTS:
%   contrastWeights : struct
%       A struct specifying the weight of each condition in the contrast.
%       Fields of the struct are condition names and the associated values are the contrast weights.
%       Use '_WILDCARD_' in the condition name to denote a wildcard match.
%       Example:
%           contrastWeights = struct('Faces', 1, 'Objects_WILDCARD_', -1);
%
%   regressor_names : cell array of strings
%       Names of the regressors extracted from the SPM.mat structure.
%       Typically includes task conditions and confound regressors.
%       Example:
%           {'Sn(1) Faces*bf(1)', 'Sn(1) Objects*bf(1)', 'Sn(1) trans_x', ...}
%
% OUTPUTS:
%   weight_vector : numeric vector
%       A vector of weights for each regressor in the order they appear in the regressor_names.
%       Example:
%           [1, -1, 0, ...]
%
% NOTE:
%   This function assumes that task-related regressors in the SPM design matrix end with "*bf(1)".
%   Confound regressors (e.g., motion parameters) do not have this suffix.

% Initialize a weight vector of zeros
weight_vector = zeros(1, length(regressor_names));

% Extract field names from the contrastWeights structure
fields = fieldnames(contrastWeights);

% Iterate over the field names to match with regressor names
for i = 1:length(fields)
    field = fields{i};

    % If the field contains a wildcard, handle it
    if contains(field, '_WILDCARD_')
        % Convert the wildcard pattern to a regular expression pattern
        pattern = ['Sn\(.\) ' strrep(field, '_WILDCARD_', '.*')];
        
        % Find indices of matching regressors using the regular expression pattern
        idx = find(~cellfun('isempty', regexp(regressor_names, pattern)));

        % Assign the weight from contrastWeights to the matching regressors
        weight_vector(idx) = contrastWeights.(field);
    else
        % No need to extract the condition name, just append *bf(1) to match the SPM regressor pattern
        pattern = ['Sn\(.\) ' field];

        idx = find(~cellfun('isempty', regexp(regressor_names, pattern)));

        % Assign the weight from contrastWeights to the regressor
        if ~isempty(idx)
            weight_vector(idx) = contrastWeights.(field);
        end
    end
end
end

