function saveCurrentScript(outputPath)
% saveCurrentScript
%
% Saves the currently executing script to a specified path. If the provided
% outputPath is a directory, the script retains its current name. If
% outputPath includes a filename, it saves the script using that name.
%
% Usage:
%   saveCurrentScript(outputPath)
%
% Inputs:
%   outputPath (string) - Path to a directory or a full path (including
%                         filename) where the script should be saved.
%
% Example:
%   >> saveCurrentScript('C:\Users\username\Documents');
%   >> saveCurrentScript('C:\Users\username\Documents\backup_script.m');
%
% Notes:
%   - This function must be called from within a script. It will not work
%     if called from the Command Window or an anonymous function.

% Get the currently executing script's full path
stackInfo = dbstack('-completenames');

% Ensure the function is run from within a script
if length(stackInfo) < 2
    error('This function must be called from within a script.');
end

currentScriptPath = stackInfo(2).file; % Path of the current script
[~, currentScriptName, ext] = fileparts(currentScriptPath); % Extract script name

% If the outputPath is a directory, append the current script's name
if isfolder(outputPath)
    outputPath = fullfile(outputPath, [currentScriptName, ext]);
end

% Validate the provided directory exists
[outputDir, ~, ~] = fileparts(outputPath);
if ~exist(outputDir, 'dir')
    error('The specified directory does not exist: %s', outputDir);
end

% Copy the script to the specified output path
copyfile(currentScriptPath, outputPath);

% Confirm success
fprintf('[INFO] Script "%s" saved to "%s".\n', currentScriptPath, outputPath);
end
