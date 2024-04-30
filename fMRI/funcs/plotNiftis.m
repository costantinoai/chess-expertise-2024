imgPath1 = '/data/projects/chess/data/BIDS/derivatives/fmriprep/sub-01/anat/sub-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz';
imgPath2 = '/data/projects/chess/data/BIDS/derivatives/rois-HCP/sub-01/sub-01_HCPMMP1_volume_MNI.nii';
plotBrainSlices(imgPath1, imgPath2, 999)

function plotBrainSlices(imgPath1, imgPath2, nonBrainThreshold)
    % Plot two brain images in 20 horizontal slices for comparison.
    %
    % Parameters:
    %    imgPath1: Path to the first nifti image file
    %    imgPath2: Path to the second nifti image file
    %
    % This function loads two brain images from their paths and plots 20 horizontal
    % slices from each image to facilitate visual comparison for alignment and size.
    
    % Load the images
    img1 = niftiread(imgPath1);
    img2 = niftiread(imgPath2);

    numSlices = 20; % Number of slices to display
    xIncrement = 5; % Increment x by 5 for each slice

    % Calculate figure size to make each image at least 5 times bigger
    screenWidth = 3840; % Assuming a full HD screen, adjust if necessary
    screenHeight = 2160;
    figWidth = screenWidth * 1;
    figHeight = screenHeight * 1;
    
    % Create a large figure to accommodate all subplots
    figure('Name', 'Brain Slices Comparison', 'Position', [100, 100, figWidth, figHeight]);

    for i = 1:numSlices
        xIndex = (i - 1) * xIncrement + 1; % Calculate x index
    
        % Adjust if xIndex exceeds image dimensions
        xIndex = min(xIndex, size(img1, 2));
    
        % Plot the first image slice
        subplot(3, numSlices, i);
        imshow(squeeze(img1(:, xIndex, :)), []);
        title(['X = ', num2str(xIndex)]);
        axis off;
    
        % Plot the second image slice
        subplot(3, numSlices, i + numSlices);
        imshow(squeeze(img2(:, xIndex, :)), []);
        axis off;
    
        % Overlay the second image on the first
        subplot(3, numSlices, i + numSlices * 2);
        baseImg = squeeze(img1(:, xIndex, :));
        overlayImg = squeeze(img2(:, xIndex, :));
        imshow(baseImg, []); hold on;
        redOverlay = cat(3, overlayImg > nonBrainThreshold, zeros(size(overlayImg)), zeros(size(overlayImg)));
        h = imshow(redOverlay);
        set(h, 'AlphaData', overlayImg > nonBrainThreshold * 0.3);
        axis off;
    end
        sgtitle('Brain Slices Comparison and Overlay');
end

