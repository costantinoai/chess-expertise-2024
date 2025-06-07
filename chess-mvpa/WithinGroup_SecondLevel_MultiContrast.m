function WithinGroup_SecondLevel_MultiContrast(smoothBool)
    % WITHINGROUP_SECONDLEVEL_MULTICONTRAST(smoothBool)
    %
    % This script performs multiple within-group (1-sample t-test) analyses
    % on two groups (Experts and Non-Experts) in SPM. For each specified 
    % first-level contrast, it runs a separate group analysis. Optionally,
    % smoothing can be performed before the second-level analysis.
    %
    % USAGE EXAMPLE:
    %   >> WithinGroup_SecondLevel_MultiContrast(true);   % with smoothing
    %   >> WithinGroup_SecondLevel_MultiContrast(false);  % without smoothing
    %
    % -------------------------------------------------------------------------
    % PARAMETERS & THEIR MEANING:
    %
    % 1) smoothBool:
    %    - Boolean flag indicating whether to perform smoothing (true) or skip it (false).
    %
    % 2) EXPERT_SUBJECTS and NONEXPERT_SUBJECTS:
    %    - Cell arrays of subject IDs to include in each group.
    %
    % 3) rootDir:
    %    - The root directory where first-level contrast images are stored.
    %
    % 4) contrastFiles (cell array):
    %    - A list of the contrast files (e.g., {'con_0001.nii','con_0002.nii'}).
    %      Each entry corresponds to a first-level contrast you want to analyze.
    %
    % 5) fwhm:
    %    - The full width at half maximum (FWHM) Gaussian smoothing kernel in mm.
    %      Typically 8 or 9 mm isotropic is used for second-level smoothing.
    %
    % 6) SPM factorial design parameters (matlabbatch{1}.spm.stats.factorial_design):
    %    - .dir:          Output directory for the factorial design.
    %    - .des.t1.scans: Paths to the (smoothed or original) contrast images for the group.
    %    - .des.t1.gmsca: Grand mean scaling (usually 0 = none).
    %    - .des.t1.ancova: Covariate for overall mean (usually 0 = none).
    %    - .masking.tm.tm_none: No threshold masking (1 = use all voxels).
    %    - .masking.im:   Implicit mask (1 = on, 0 = off).
    %    - .masking.em:   Explicit mask file (if any).
    %    - .globalc.g_omit: Global calculation omitted.
    %    - .globalm.gmsca.gmsca_no: No grand mean scaling.
    %    - .globalm.glonorm: Global normalisation (1 = proportional).
    %
    % 7) Model estimation (matlabbatch{2}.spm.stats.fmri_est):
    %    - .spmmat: Path to the SPM.mat file created by factorial_design.
    %
    % 8) Contrast specification (matlabbatch{3}.spm.stats.con):
    %    - Single T contrast with weight = 1 (tests mean > 0).
    %
    % -------------------------------------------------------------------------
    %
    % Requires SPM12 on your MATLAB path.
    %
    % -------------------------------------------------------------------------
    
    %% 1. Check input argument
    if nargin < 1
        smoothBool = false; % Default: no smoothing if not specified
    end
    fprintf('Smoothing is set to: %s\n', string(smoothBool));
    
    %% 2. Define subject groups
    EXPERT_SUBJECTS = { ...
        '03', '04', '06', '07', '08', '09', ...
        '10', '11', '12', '13', '16', '20', ...
        '22', '23', '24', '29', '30', '33', ...
        '34', '36' ...
    };

    NONEXPERT_SUBJECTS = { ...
        '01', '02', '15', '17', '18', '19', ...
        '21', '25', '26', '27', '28', '32', ...
        '35', '37', '39', '40', '41', '42', ...
        '43', '44' ...
    };

    %% 3. Set up parameters
    rootDir    = '/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-6_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM';
    
    % Here you define all the contrast files you want to analyze 
    % (already computed at the subject level):
    contrastFiles = { ...
        'con_0001.nii', ...
        'con_0002.nii', ...   % Uncomment or add as needed
        % 'con_0003.nii', ...
    };
    
    % Smoothing kernel (for second-level smoothing, if applied)
    fwhm = [9 9 9];  % 9mm isotropic (you can adjust)

    %% 4. Loop over each contrast
    for c = 1:numel(contrastFiles)
        contrastFile = contrastFiles{c};

        % For organizational purposes, create suffix with the contrast's "base name"
        [~, cbase, ~] = fileparts(contrastFile);  % e.g., 'con_0001'

        % Construct second-level output folders specifically for each contrast
        secondLevelExperts    = fullfile(rootDir, ['2ndLevel_Experts_' cbase]);
        secondLevelNonExperts = fullfile(rootDir, ['2ndLevel_NonExperts_' cbase]);

        if ~exist(secondLevelExperts,    'dir'), mkdir(secondLevelExperts);    end
        if ~exist(secondLevelNonExperts, 'dir'), mkdir(secondLevelNonExperts); end

        fprintf('\n====================================================\n');
        fprintf('Processing contrast: %s\n', contrastFile);
        fprintf('Experts folder:    %s\n', secondLevelExperts);
        fprintf('Non-Experts folder:%s\n', secondLevelNonExperts);

        %% 4A. Prepare contrast images for Experts
        if smoothBool
            fprintf('Smoothing contrast images for Experts...\n');
            expertConImages = smoothContrastImages(rootDir, EXPERT_SUBJECTS, contrastFile, fwhm);
        else
            fprintf('Using original contrast images for Experts (no smoothing)...\n');
            expertConImages = getOriginalContrastPaths(rootDir, EXPERT_SUBJECTS, contrastFile);
        end

        %% 4B. Prepare contrast images for Non-Experts
        if smoothBool
            fprintf('Smoothing contrast images for Non-Experts...\n');
            nonexpertConImages = smoothContrastImages(rootDir, NONEXPERT_SUBJECTS, contrastFile, fwhm);
        else
            fprintf('Using original contrast images for Non-Experts (no smoothing)...\n');
            nonexpertConImages = getOriginalContrastPaths(rootDir, NONEXPERT_SUBJECTS, contrastFile);
        end

        %% 4C. Build and run the SPM job for Experts (1-sample t-test)
        fprintf('\n=== 1-Sample T-Test for Experts: %s ===\n', cbase);

        matlabbatch = {};

        % (a) Specify factorial design
        matlabbatch{1}.spm.stats.factorial_design.dir = {secondLevelExperts};
        matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = expertConImages;
        matlabbatch{1}.spm.stats.factorial_design.des.t1.gmsca = 0;   % No grand mean scaling
        matlabbatch{1}.spm.stats.factorial_design.des.t1.ancova = 0;  % No ANCOVA
        matlabbatch{1}.spm.stats.factorial_design.des.t1.variance = 1; % Typically 1 (unequal)
        matlabbatch{1}.spm.stats.factorial_design.des.t1.dept = 0;     % Independence
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1; % No threshold masking
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;     % Implicit masking
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};  % No explicit mask
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1; 
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

        % (b) Model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(secondLevelExperts, 'SPM.mat')};

        % (c) Contrast specification (single group mean)
        matlabbatch{3}.spm.stats.con.spmmat = {fullfile(secondLevelExperts, 'SPM.mat')};
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = ['Group Mean (Experts): ' cbase];
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;  % 1-sample t-test
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{3}.spm.stats.con.delete = 0;  % Do not delete existing contrasts

        % Run
        spm('Defaults','fMRI');
        spm_jobman('initcfg');
        spm_jobman('run', matlabbatch);

        fprintf('=> 1-sample t-test for Experts COMPLETE.\n');

        %% 4D. Build and run the SPM job for Non-Experts (1-sample t-test)
        fprintf('\n=== 1-Sample T-Test for Non-Experts: %s ===\n', cbase);

        matlabbatch = {};

        % (a) Specify factorial design
        matlabbatch{1}.spm.stats.factorial_design.dir = {secondLevelNonExperts};
        matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = nonexpertConImages;
        matlabbatch{1}.spm.stats.factorial_design.des.t1.gmsca = 0;   % No grand mean scaling
        matlabbatch{1}.spm.stats.factorial_design.des.t1.ancova = 0;  % No ANCOVA
        matlabbatch{1}.spm.stats.factorial_design.des.t1.variance = 1; % Typically 1 (unequal)
        matlabbatch{1}.spm.stats.factorial_design.des.t1.dept = 0;     % Independence
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1; % No threshold masking
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;     % Implicit masking
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};  % No explicit mask
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1; 
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

        % (b) Model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(secondLevelNonExperts, 'SPM.mat')};

        % (c) Contrast specification (single group mean)
        matlabbatch{3}.spm.stats.con.spmmat = {fullfile(secondLevelNonExperts, 'SPM.mat')};
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = ['Group Mean (Non-Experts): ' cbase];
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;  % 1-sample t-test
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{3}.spm.stats.con.delete = 0;  % Do not delete existing contrasts

        % Run
        spm('Defaults','fMRI');
        spm_jobman('initcfg');
        spm_jobman('run', matlabbatch);

        fprintf('=> 1-sample t-test for Non-Experts COMPLETE.\n');
    end

    fprintf('\nAll within-group analyses for all contrasts are COMPLETE.\n');
end


%% ========================================================================
function smoothedImages = smoothContrastImages(rootDir, subjectList, contrastFile, fwhm)
    % SMOOTHCONTRASTIMAGES Smooths contrast images for a list of subjects.
    %
    % Inputs:
    %   - rootDir:       Root directory of the GLM results
    %   - subjectList:   Cell array of subject IDs
    %   - contrastFile:  Name of the contrast file to smooth (e.g., 'con_0001.nii')
    %   - fwhm:          Smoothing kernel in mm (e.g., [8 8 8])
    %
    % Output:
    %   - smoothedImages: Cell array of full paths to smoothed contrast images 
    %                     (with the prefix 's' added to the file name)
    %
    % Each subject's contrast image is assumed to be found at:
    %   <rootDir>/sub-<subjectID>/exp/<contrastFile>
    %
    % This function calls spm_jobman('run', ...) to actually perform the smoothing.

    smoothedImages = {};
    matlabbatch = {};
    for i = 1:numel(subjectList)
        subjDir   = fullfile(rootDir, ['sub-' subjectList{i}], 'exp');
        inputFile = fullfile(subjDir, contrastFile);
        if ~exist(inputFile, 'file')
            error('File not found: %s', inputFile);
        end

        % Define path for the smoothed output file (prefix 's')
        [fileDir, fileName, fileExt] = fileparts(inputFile);
        smoothedFile = fullfile(fileDir, ['s' fileName fileExt]);

        % Add smoothing job
        matlabbatch{end+1}.spm.spatial.smooth.data = {inputFile}; %#ok<*AGROW>
        matlabbatch{end}.spm.spatial.smooth.fwhm   = fwhm; 
        matlabbatch{end}.spm.spatial.smooth.dtype  = 0;   % Same data type as input
        matlabbatch{end}.spm.spatial.smooth.im     = 0;   % No implicit mask
        matlabbatch{end}.spm.spatial.smooth.prefix = 's'; % Prefix for smoothed files

        % Store the smoothed file path (with volume index ",1")
        smoothedImages{end+1,1} = [smoothedFile ',1'];
    end

    % Run the smoothing job
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);

    fprintf('Smoothing complete for %d subjects.\n', numel(subjectList));
end


%% ========================================================================
function originalImages = getOriginalContrastPaths(rootDir, subjectList, contrastFile)
    % GETORIGINALCONTRASTPATHS Returns the original contrast image paths (no smoothing).
    %
    % Inputs:
    %   - rootDir:       Root directory of the GLM results
    %   - subjectList:   Cell array of subject IDs
    %   - contrastFile:  Name of the contrast file (e.g., 'con_0001.nii')
    %
    % Output:
    %   - originalImages: Cell array of full paths to the original contrast images 
    %                     (with volume index ",1")
    %
    % Each subject's contrast image is assumed to be found at:
    %   <rootDir>/sub-<subjectID>/exp/<contrastFile>
    %
    % This function simply constructs those paths without running smoothing.

    originalImages = cell(numel(subjectList), 1);
    for i = 1:numel(subjectList)
        subjDir  = fullfile(rootDir, ['sub-' subjectList{i}], 'exp');
        thisFile = fullfile(subjDir, contrastFile);
        if ~exist(thisFile, 'file')
            error('File not found: %s', thisFile);
        end
        originalImages{i} = [thisFile ',1'];  % Add volume index
    end
end
