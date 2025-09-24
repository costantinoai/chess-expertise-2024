import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from scipy.interpolate import griddata, splprep, splev
from matplotlib import cm
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap  # Custom colormaps

def make_brain_cmap():
    """
    Create a custom LinearSegmentedColormap blending cool blues for negative
    and pinks for positive values, centered at zero.
    Returns
    -------
    LinearSegmentedColormap
    """
    # center color from RdPu colormap at zero
    center = plt.cm.RdPu(0)[:3]
    # negative tail: interpolate from blue-teal to center
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    # positive tail: take full RdPu colormap
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    # stack and create new colormap
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

# 1. --- Data Generation ---
def generate_data(stim, features, class_sep, n_redundant):
    """
    Generate a synthetic binary classification dataset and normalize features to [0,1].

    Parameters
    ----------
    stim : int
        Number of samples to generate.
    features : int
        Total number of features.
    class_sep : float
        Separation between classes.
    n_redundant : int
        Number of redundant (linear combinations of informative) features.

    Returns
    -------
    data : ndarray, shape (stim, features)
        Normalized feature matrix.
    labels : ndarray, shape (stim,)
        Binary class labels (0 or 1).
    """
    data, labels = make_classification(
        n_samples=stim,
        n_features=features,
        n_informative=features // 2,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        n_classes=2,
        class_sep=class_sep,
        shuffle=False
    )
    # Normalize data to the range [0, 1]
    data_min, data_max = data.min(), data.max()
    data = (data - data_min) / (data_max - data_min)
    return data, labels

# 2. --- Participation Ratio ---
def compute_pr(data):
    """
    Compute the participation ratio (effective dimensionality) via PCA.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Input feature matrix.

    Returns
    -------
    pr : float
        Participation ratio: (sum of eigenvalues)^2 / sum of squared eigenvalues.
    pca : PCA object
        Fitted PCA instance for further use.
    eigvals : ndarray
        Array of PCA eigenvalues (explained variances).
    explained_ratio : ndarray
        Fraction of variance explained by each principal component.
    """
    pca = PCA()
    pca.fit(data)
    eigvals = pca.explained_variance_
    # Compute PR: square of sum divided by sum of squares
    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return pr, pca, eigvals, pca.explained_variance_ratio_

# 3. --- Clustered Heatmap ---
def plot_heatmap(data, true_labels, title="Clustered Heatmap"):
    """
    Visualize data matrix as a two-color overlay heatmap, splitting samples by class.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Input feature matrix (normalized).
    true_labels : ndarray, shape (n_samples,)
        Binary labels to separate top/bottom halves.
    title : str, optional
        Title for the plot. Default is "Clustered Heatmap".
    """
    stim, features = data.shape
    # Create mask arrays to color-code first half and second half
    row_indices = np.arange(stim)[:, None]
    mask_top = np.broadcast_to(row_indices < (stim // 2), data.shape)
    mask_bottom = ~mask_top
    top_data = np.ma.masked_where(mask_bottom, data)
    bottom_data = np.ma.masked_where(mask_top, data)

    fig, ax = plt.subplots(figsize=(10, 20))
    # Plot bottom half in red
    ax.imshow(bottom_data, cmap='Reds', aspect='equal', origin='lower', vmin=0, vmax=1.0)
    # Overlay top half in green
    ax.imshow(top_data, cmap='Greens', aspect='equal', origin='lower', vmin=0, vmax=1.0)
    [sp.set_edgecolor('gray') for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()

# 4. --- PCA Spectrum ---
def plot_pca_spectrum(eigvals, explained, pr, title="PCA Spectrum"):
    """
    Plot explained variances of principal components and highlight variance regions.

    Parameters
    ----------
    eigvals : ndarray
        PCA eigenvalues (explained variances).
    explained : ndarray
        Explained variance ratio for each component.
    pr : float
        Participation ratio to annotate plot.
    title : str, optional
        Title for the plot. Default is "PCA Spectrum".
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(eigvals) + 1), eigvals, color='lightblue', edgecolor='k')
    ax.axhline(y=np.median(eigvals), color='gray', linestyle='--', label='Noise threshold')
    cumulative = np.cumsum(explained)
    ax.fill_between(
        range(1, len(eigvals) + 1), 0, eigvals,
        where=cumulative < 0.9,
        color='green', alpha=0.3,
        label='90% variance region'
    )
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance", fontsize=12)
    ax.set_title(f'{title}\nParticipation Ratio = {pr:.2f}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

# 5. --- 3D Manifold Surface + Ribbon ---
def plot_3d_surface(embedding_3d, true_labels, add_ribbon=False, title="3D Manifold"):
    """
    Render a 3D surface fitted to the first three PCA dimensions, with optional ribbon curve.
    Removes axis ticks and background walls for a clean look.

    Parameters
    ----------
    embedding_3d : ndarray, shape (n_samples, 3)
        3D coordinates from PCA transform.
    true_labels : ndarray, shape (n_samples,)
        Binary labels for point coloring.
    add_ribbon : bool, optional
        Whether to overlay a smooth ribbon curve through sorted points.
    title : str, optional
        Title for the plot. Default is "3D Manifold".
    """
    # Generate grid over PCA dimensions
    grid_x, grid_y = np.mgrid[
        embedding_3d[:, 0].min():embedding_3d[:, 0].max():40j,
        embedding_3d[:, 1].min():embedding_3d[:, 1].max():40j
    ]
    # Interpolate z-values on grid
    grid_z = griddata(
        (embedding_3d[:, 0], embedding_3d[:, 1]),
        embedding_3d[:, 2], (grid_x, grid_y), method='cubic'
    )
    # Normalize for surface coloring
    norm_z = (grid_z - np.nanmin(grid_z)) / (np.nanmax(grid_z) - np.nanmin(grid_z))
    # surface_colors = cm.Spectral(norm_z)
    brain_cmap = make_brain_cmap()
    surface_colors = brain_cmap(norm_z)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Plot semi-transparent surface
    ax.plot_surface(
        grid_x, grid_y, grid_z,
        facecolors=surface_colors,
        rstride=1, cstride=1, linewidth=0,
        antialiased=False, shade=False, alpha=0.2
    )
    # Optionally overlay ribbon
    if add_ribbon:
        sorted_idx = np.argsort(embedding_3d[:, 0])
        x_sorted = embedding_3d[sorted_idx, 0]
        y_sorted = embedding_3d[sorted_idx, 1]
        z_sorted = embedding_3d[sorted_idx, 2]
        tck, _ = splprep([x_sorted, y_sorted, z_sorted], s=3)
        ribbon = splev(np.linspace(0, 1, 100), tck)
        ax.plot(ribbon[0], ribbon[1], ribbon[2], color='black', linewidth=2, linestyle='--')
    # Scatter points colored by label
    point_colors = np.array(['green', 'red'])[true_labels]
    ax.scatter(
        embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
        c=point_colors, s=60, edgecolor='k', linewidth=0.5
    )
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove background panes and axes lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    plt.title(title, fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()

# 6. --- Decoding Hyperplane ---
def plot_decoding_3d_with_plane(data, labels, title="Decoding: SVM Hyperplane"):
    """
    Train a linear SVM on the first three PCA dimensions and plot the decision plane.
    Removes axis ticks and background walls for clarity.

    Parameters
    ----------
    data : ndarray, shape (n_samples, 3)
        3D embedding from PCA.
    labels : ndarray, shape (n_samples,)
        Binary labels (0 or 1).
    title : str, optional
        Title for the plot. Default is "Decoding: SVM Hyperplane".
    """
    clf = SVC(kernel='linear')
    clf.fit(data, labels)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    # Create meshgrid over PCA dimensions
    x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 10)
    y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 10)
    xx, yy = np.meshgrid(x_range, y_range)
    # Compute plane z-values
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Plot decision surface
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightblue', edgecolor=None)
    # Scatter points
    point_colors = np.array(['green', 'red'])[labels]
    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=point_colors, s=60, edgecolor='k', linewidth=0.5
    )
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove background panes and axes lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    plt.title(title, fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()

# 7. --- RSA RDM Matrix ---
def plot_rsa_3d_to_rdm(data, title="RSA: RDM from Features"):
    """
    Compute representational dissimilarity matrix (RDM) and plot as heatmap.
    """
    rdm = pairwise_distances(data, metric='correlation')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(rdm, cmap='RdPu', cbar=False, ax=ax, square=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# 8. --- Wrapper ---
def plot_all(data, true_labels, label=""):
    """
    Run all visualizations: heatmap, PCA spectrum, 3D manifold, ribbon, decoding, RSA.
    """
    pr, pca, eigvals, explained = compute_pr(data)
    embedding_3d = pca.transform(data)[:, :3]

    plot_heatmap(data, true_labels, title=f"{label} Heatmap")
    plot_pca_spectrum(eigvals, explained, pr, title=f"{label} PCA Spectrum")
    plot_3d_surface(embedding_3d, true_labels, add_ribbon=False, title=f"{label} Manifold")
    plot_3d_surface(embedding_3d, true_labels, add_ribbon=True, title=f"{label} Manifold + Ribbon")
    plot_decoding_3d_with_plane(embedding_3d, true_labels, title=f"{label} Decoding")
    plot_rsa_3d_to_rdm(data, title=f"{label} RDM")

# 9. --- Execute ---
stim = 40
features = int(stim // 1.5)

# Low participation ratio: high redundancy, high class separation
lowPR_data, lowPR_labels = generate_data(stim, features, class_sep=3.0, n_redundant=stim//4)
# High participation ratio: zero redundancy, lower class separation
highPR_data, highPR_labels = generate_data(stim, features, class_sep=1.0, n_redundant=0)

# Plot pipelines
plot_all(lowPR_data, lowPR_labels, label="Low PR")
plot_all(highPR_data, highPR_labels, label="High PR")
