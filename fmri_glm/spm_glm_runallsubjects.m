% GLM Analysis Script for fMRI Data using SPM. The script takes fRMIprep
% BIDS structure, decompress the images, smooth if necessary, defines and
% runs a GLM on each defined task, and applies contrasts.

% Assumed Shape of Input Data: - The fMRI data should be preprocessed using
% fMRIprep and organized in BIDS format. - The event files for each run
% should be in TSV format following BIDS specifications. - The confound
% files for each run should be in JSON and TSV format following BIDS
% specifications.

% Script Steps:
% 1. Retrieve the subject folders based on the selectedSubjects parameter.
% 2. Iterate over each subject and perform the GLM analysis steps.
% 3. For each subject, retrieve the event files, confound files, and preprocessed fMRI data files for the selected task and runs.
% 4. Set the SPM parameters for the model specification and estimation.
% 5. Define the contrasts of interest for the analysis.
% 6. Perform the GLM analysis using the event files, confound files, and fMRI data files for each run.
% 7. Save the results of the analysis in the specified output folder.

% Note: This script assumes the use of the SPM software for fMRI data analysis.

% Parameters:
% - selectedSubjects: List of subject IDs to include in the analysis. Can be a list of integers or '*' to include all subjects.
% - selectedRuns: List of run numbers to include in the analysis. Can be an integer or '*' to include all runs.
% - selectedTask: Structure with information about the task(s) to analyze.
%       The structures must have at least one item (i.e., one task), and
%       each item must have the following four fields: name, contrasts,
%       weights, and smoothBool.
%           - name: String. The name of the task
%           - contrasts: Cell array of strings. The name(s) of the contrast(s)
%           - weights: Cell array of ints. The weights for the contrast(s).
%           - smoothBool: Boolean. Whether to smooth the data before the GLM.
%
%       e.g., % Define tasks, weights and constrasts as a structure
%       selectedTasks(1).name = 'loc1';
%       selectedTasks(1).contrasts = {'Faces > Objects', 'Objects > Scrambled', 'Scenes > Objects'};
%       selectedTasks(1).weights = {[1 -1 0 0], [0 1 0 -1], [0 -1 1 0]};
%       selectedTasks(1).smoothBool = true; % Whether to smooth the images before GLM

% Paths:
% - fmriprepRoot: Path to the root folder of the fMRIprep output.
% - BIDSRoot: Path to the root folder of the BIDS dataset.
% - outRoot: Path to the output root folder for the analysis results.

% Preprocessing Options (uses fMRIprep confounds table):
% - pipeline: Denoising pipeline strategy for SPM confound regression. It should be a cell array of strings specifying the strategies to use.
%
% IMPORTANT NOTE: this code uses filterRun1Files to remove the first run
% from the GLM! If you want to use run 1, remove this function
%
% Author: Andrea Ivan Costantino
% Date: 5 July 2023

clc
clear

%% PARAMETERS

% Path of fmriprep, BIDS and output folder (edit these roots for your machine)
niftiSpace = 'MNI'; % 'T1w' or 'MNI'
dataRoot   = fullfile('data');
BIDSRoot   = fullfile(dataRoot, 'BIDS');
derivRoot  = fullfile(BIDSRoot, 'derivatives');
fmriprepRoot = fullfile(derivRoot, 'fmriprep');
outRoot    = fullfile(derivRoot, ['fmriprep-SPM_smoothed-6_GS-FD-HMP_brainmasked' filesep niftiSpace], ['fmriprep-SPM-' niftiSpace]);
tempDir    = fullfile(dataRoot, 'temp', 'fmriprep-preSPM');

% Files to select
% selectedSubjectsList = '*';        % Must be list of integers or '*'
selectedSubjectsList = '*';     % Must be list of integers or '*'
selectedRuns = '*';                         % Must be integer or '*'

% Define tasks, weights and constrasts as a structure
% Here we select the Check vs. non-check contrast, and All vs. Baseline
% NOTE that we do not explicitly model the baseline condition, so
% the second contrast compares against 0 (i.e., the baseline condition)
% which also includes our rest/fixation periods
selectedTasks(1).name = 'exp';
selectedTasks(1).contrasts = {'Check > No-Check', 'All > Rest'};
selectedTasks(1).weights(1) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', -1);
selectedTasks(1).weights(2) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', 1);
selectedTasks(1).smoothBool = true; % Whether to smooth the images before GLM


% Denoising pipeline strategy for SPM confound regression Must be a cell
% array of strings {}. Possible strategies:
%     HMP - Head motion parameters (6,12,24)
%     GS - Brain mask global signal (1,2,4)
%     CSF_WM - CSF and WM masks global signal (2,4,8)
%     aCompCor - aCompCor (10,50)
%     MotionOutlier - motion outliers FD > 0.5, DVARS > 1.5, non steady volumes
%     Cosine - Discrete cosine-basis regressors low frequencies -> HPF
%     FD - Raw framewise displacement
%     Null - returns a blank df
% pipeline = {'HMP-6','GS-1','FD'};
pipeline = {'HMP-6','FD','GS-1'};

% Get subjects folders from subjects list
sub_paths = findSubjectsFolders(fmriprepRoot, selectedSubjectsList);

% Thresholds for statistical analysis
thresholds = {
    0.001, ...
    0.01, ...
    0.05 ...
    };


parfor i = 1:length(sub_paths)
    run_subject_glm(sub_paths(i).folder, sub_paths(i).name, selectedTasks, selectedRuns, ...
         fmriprepRoot, BIDSRoot, outRoot, tempDir, pipeline, niftiSpace, thresholds);
end

