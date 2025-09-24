function [labels, names] = loadColorTable(colorTablePath)
% loadColorTable
%
% Reads a FreeSurfer-style color table file and applies an offset (+1000
% for left, +2000 for right) to each label. The ROI names are duplicated
% with the first letter replaced by 'R' for the "right" hemisphere.
%
% Usage:
%   [labels, names] = loadColorTable(colorTablePath)
%
% Inputs:
%   colorTablePath (string)
%       Path to the color table .txt file
%
% Outputs:
%   labels (numeric array)
%   names  (cell array of strings)
%
% Example:
%   >> [labs, nms] = loadColorTable('/path/lh_HCPMMP1_color_table.txt')
%   >> disp(labs(1:5)), disp(nms(1:5))

% Open the text file and read
fid  = fopen(colorTablePath);
data = textscan(fid, '%d %s %*[^\n]', 'HeaderLines', 1);
fclose(fid);

% Original labels offset by +1000
% original_labels = data{1} + 1000;
original_labels = data{1};
original_names  = data{2};

% Duplicate labels offset by +2000,
% with 'R' prefix on the name
% modified_labels = data{1} + 2000;
modified_labels = data{1} + 1000;
modified_names  = cellfun(@(nm) ['R' nm(2:end)], original_names, 'UniformOutput', false);

% Concatenate
% labels = [original_labels; modified_labels];
% names  = [original_names;  modified_names];

labels = [original_labels];
names  = [original_names];
end