function timestamp = createTimestamp()
% createTimestamp
%
% Generates a timestamp string in the format 'yyyymmdd-hhmmss'.
%
% Usage:
%   timestamp = createTimestamp();
%
% Output:
%   timestamp (string) - A string representing the current date and time.
%
% Example:
%   >> ts = createTimestamp();
%   >> disp(ts);  % Example: '20250115-143005'

    % Get the current date and time
    currentTime = datetime('now', 'Format', 'yyyyMMdd-HHmmss');
    
    % Convert to string format
    timestamp = char(currentTime);
end
