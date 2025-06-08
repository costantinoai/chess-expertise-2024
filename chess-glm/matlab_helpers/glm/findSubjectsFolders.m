function [filteredFolderStructure] = findSubjectsFolders(fmriprepRoot, selectedSubjectsList, excludedSubjectsList)
% FINDSUBJECTSFOLDERS Locate subject folders based on a list or wildcard.
%
% USAGE:
% sub_paths = findSubjectsFolders(fmriprepRoot, selectedSubjectsList)
%
% INPUTS:
% fmriprepRoot          - The root directory where 'sub-*' folders are located.
%
% selectedSubjectsList  - Can be one of two things:
%                         1) A list of integers, each representing a subject ID.
%                            For example, [7,9] would search for folders 'sub-07' 
%                            and 'sub-09' respectively.
%                         2) A single character string '*'. In this case, the function
%                            will return all folders starting with 'sub-*'.
%
% OUTPUTS:
% sub_paths             - A structure array corresponding to the found directories.
%                         Each structure has fields: 'name', 'folder', 'date', 
%                         'bytes', 'isdir', and 'datenum'.
%
% EXAMPLES:
% 1) To fetch directories for specific subjects:
%    sub_paths = findSubjectsFolders('/path/to/fmriprepRoot', [7,9]);
%
% 2) To fetch all directories starting with 'sub-*':
%    sub_paths = findSubjectsFolders('/path/to/fmriprepRoot', '*');
%
% NOTE:
% If a subject ID from the list does not match any directory, a warning is issued.

% Start by fetching all directories with the 'sub-*' pattern.
sub_paths = dir(fullfile(fmriprepRoot, 'sub-*'));
sub_paths = sub_paths([sub_paths.isdir]); % Keep only directories.

% Check the type of selectedSubjectsList
if isnumeric(selectedSubjectsList(1))
    % Case 1: selectedSubjectsList is a list of integers.

    % Convert each integer in the list to a string of the form 'sub-XX'.
    subIDs = cellfun(@(x) sprintf('sub-%02d', x), num2cell(selectedSubjectsList), 'UniformOutput', false);

    % Filter the sub_paths to keep only those directories matching the subIDs.
    sub_paths = sub_paths(ismember({sub_paths.name}, subIDs));

    % Check and throw warnings for any missing subID.
    foundSubIDs = {sub_paths.name};
    for i = 1:length(subIDs)
        if ~ismember(subIDs{i}, foundSubIDs)
            warning(['The subID ', subIDs{i}, ' was not found in sub_paths.name.']);
        end
    end

elseif ischar(selectedSubjectsList) && strcmp(selectedSubjectsList, '*')
    % Case 2: selectedSubjectsList is '*'. 
    % No further action required as we've already selected all 'sub-*' folders.

else
    % Invalid input.
    error('Invalid format for selectedSubjects. It should be either "*" or a list of integers.');
end

% Only process exclusion if the excludedSubjectsList is provided.
if nargin == 3
    % Create a list of excluded folder names
    excludedNames = cellfun(@(x) sprintf('sub-%02d', x), num2cell(excludedSubjectsList), 'UniformOutput', false);

    % Logical array of folders to exclude
    excludeMask = arrayfun(@(x) ismember(x.name, excludedNames), sub_paths);

    % Filtered structure
    filteredFolderStructure = sub_paths(~excludeMask);
else
    % If no excludedSubjectsList is provided, just return the sub_paths.
    filteredFolderStructure = sub_paths;
end
end
