# Full updated script with detailed comments for clarity

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. DATA CREATION ===
# Construct a DataFrame with 40 stimuli and 3 representational variables: check, strategy, visual
data = pd.DataFrame({
    'stim_id': list(range(40)),  # Stimulus IDs
    'check': [1]*20 + [0]*20,  # Binary condition (e.g., left/right)
    'strategy': [0]*3 + [1]*4 + [2]*7 + [3]*3 + [4]*3 + [5]*3 + [6]*4 + [7]*7 + [8]*3 + [9]*3,  # Discrete strategy labels
    'visual': list(range(20)) + list(range(20))  # Repeating visual IDs
})

# === 2. BUILD RDMs (Representational Dissimilarity Matrices) ===
def build_rdm(column):
    """Builds an RDM using Hamming distance on a single column."""
    return squareform(pdist(data[[column]], metric='hamming'))

# Compute RDMs for each variable
rdm_check = build_rdm('check')
rdm_strategy = build_rdm('strategy')
rdm_visual = build_rdm('visual')

def flatten_rdm(rdm):
    """Extracts the upper triangle (excluding diagonal) from an RDM."""
    return rdm[np.triu_indices(rdm.shape[0], k=1)]

# Flatten RDMs to pairwise dissimilarity vectors
flat_check = flatten_rdm(rdm_check)
flat_strategy = flatten_rdm(rdm_strategy)
flat_visual = flatten_rdm(rdm_visual)

# Combine into a new DataFrame for correlation analysis
corr_data = pd.DataFrame({
    'check': flat_check,
    'strategy': flat_strategy,
    'visual': flat_visual
})

# === 3. CORRELATION COMPUTATION ===
# Compute all pairwise Spearman correlations
pairwise_corr = corr_data.corr(method='spearman')

# Manually compute all relevant partial correlations for each RDM as target
partial_corrs = {
    'check_strategy|visual': pg.partial_corr(corr_data, x='check', y='strategy', covar='visual', method='spearman')['r'].values[0],
    'check_visual|strategy': pg.partial_corr(corr_data, x='check', y='visual', covar='strategy', method='spearman')['r'].values[0],
    'strategy_check|visual': pg.partial_corr(corr_data, x='strategy', y='check', covar='visual', method='spearman')['r'].values[0],
    'strategy_visual|check': pg.partial_corr(corr_data, x='strategy', y='visual', covar='check', method='spearman')['r'].values[0],
    'visual_check|strategy': pg.partial_corr(corr_data, x='visual', y='check', covar='strategy', method='spearman')['r'].values[0],
    'visual_strategy|check': pg.partial_corr(corr_data, x='visual', y='strategy', covar='check', method='spearman')['r'].values[0],
}

# Build the correlation plot data properly, with correct partial correlations per target
correlation_plot_data = []

for target in ['check', 'strategy', 'visual']:
    predictors = [col for col in ['check', 'strategy', 'visual'] if col != target]

    entry = {
        'target': target,
        'pred1': predictors[0],
        'pred2': predictors[1],
        'pairwise_1': pairwise_corr.loc[target, predictors[0]],
        'partial_1': partial_corrs[f'{target}_{predictors[0]}|{predictors[1]}'],
        'pairwise_2': pairwise_corr.loc[target, predictors[1]],
        'partial_2': partial_corrs[f'{target}_{predictors[1]}|{predictors[0]}']
    }
    correlation_plot_data.append(entry)

# Convert to DataFrame
correlation_df = pd.DataFrame(correlation_plot_data)

# === 4. VARIANCE PARTITIONING ===
def partition_variance(df, target, predictors):
    """
    Compute the unique, shared, and residual variance explained in a target variable
    using nested linear regression models.

    This function performs variance partitioning to assess how much of the variance
    in a target RDM (Representational Dissimilarity Matrix) is uniquely explained
    by each of two predictor RDMs, how much is shared between them, and how much
    remains unexplained (residual).

    The method uses nested linear models:
        - The full model includes both predictors.
        - Reduced models include only one predictor at a time.
        - Unique contributions are calculated by comparing the full model to each reduced model.
        - Shared variance is the remainder of the full R² after subtracting both unique components.
        - Residual variance is 1 - R² of the full model.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the target and predictor vectors, typically
        flattened upper triangles from RDMs (excluding diagonals).
        Each column should represent a dissimilarity vector.

    target : str
        The column name in `df` representing the target RDM vector whose variance
        is to be explained.

    predictors : list of str
        A list of exactly two column names in `df` representing the predictor RDMs.
        The order of predictors matters for reporting purposes, but not for the math.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'target': str, the name of the target RDM
            - 'unique_1': float, variance uniquely explained by predictors[0]
            - 'unique_2': float, variance uniquely explained by predictors[1]
            - 'shared': float, variance jointly explained by both predictors
            - 'residual': float, unexplained variance (1 - full model R²)
            - 'pred1': str, the name of the first predictor
            - 'pred2': str, the name of the second predictor

    Notes
    -----
    - All variance components are non-negative. Negative values (due to noise or
      model overlap) are clamped to 0 using `max(..., 0)`.
    - This function assumes linear additive contributions. Nonlinear relationships
      or interactions are not modeled.
    - The predictors are assumed to be continuous or dummy-coded appropriately.
    - If predictors are strongly collinear, the shared variance may dominate,
      and unique contributions may be minimal.

    Example
    -------
    >>> partition_variance(df=corr_data, target='check', predictors=['strategy', 'visual'])
    {
        'target': 'check',
        'unique_1': 0.12,
        'unique_2': 0.03,
        'shared': 0.18,
        'residual': 0.67,
        'pred1': 'strategy',
        'pred2': 'visual'
    }
    """
    y = df[target]

    # Full model with both predictors
    full_model = LinearRegression().fit(df[predictors], y)
    r2_full = r2_score(y, full_model.predict(df[predictors]))

    # Model without predictor 1
    reduced_model_2 = LinearRegression().fit(df[[predictors[1]]], y)
    r2_reduced_2 = r2_score(y, reduced_model_2.predict(df[[predictors[1]]]))
    unique_1 = r2_full - r2_reduced_2

    # Model without predictor 2
    reduced_model_1 = LinearRegression().fit(df[[predictors[0]]], y)
    r2_reduced_1 = r2_score(y, reduced_model_1.predict(df[[predictors[0]]]))
    unique_2 = r2_full - r2_reduced_1

    # Shared and residual
    shared = r2_full - unique_1 - unique_2
    residual = 1 - r2_full

    return {
        'target': target,
        'unique_1': max(unique_1, 0),
        'unique_2': max(unique_2, 0),
        'shared': max(shared, 0),
        'residual': max(residual, 0),
        'pred1': predictors[0],
        'pred2': predictors[1]
    }


# Apply partitioning to all three RDMs
results = [
    partition_variance(corr_data, 'check', ['strategy', 'visual']),
    partition_variance(corr_data, 'strategy', ['check', 'visual']),
    partition_variance(corr_data, 'visual', ['check', 'strategy']),
]

# Convert results to DataFrame
plot_df = pd.DataFrame(results)


# Define base font size and update global rcParams
base_font_size = 26
plt.rcParams.update({
    "font.size": base_font_size,
    "axes.titlesize": base_font_size * 1.4,  # Title font size
    "axes.labelsize": base_font_size * 1.2,  # Axis label font size
    "xtick.labelsize": base_font_size,       # X tick label size
    "ytick.labelsize": base_font_size,       # Y tick label size
    "legend.fontsize": base_font_size,       # Legend font size
    "figure.figsize": (22, 22),              # Global figure size
    "font.family": "Ubuntu Condensed"
})

# Define consistent color palette
# Each RDM gets a base color: visual, strategy, check
# We then define a lighter version for the partial bar

from matplotlib.colors import to_rgba

# Assign base colors
base_colors = {
    'visual': sns.color_palette("Set2")[0],
    'strategy': sns.color_palette("Set2")[1],
    'check': sns.color_palette("Set2")[2]
}

# Utility to make a lighter version of a color
def lighten_color(color, amount=0.5):
    r, g, b, a = to_rgba(color)
    white = np.array([1, 1, 1])
    color_rgb = np.array([r, g, b])
    return tuple(color_rgb + (white - color_rgb) * amount)

# --- Figure 1: Correlation Plot ---
fig_corr, axes_corr = plt.subplots(1, 3, sharey=True, figsize=(22, 10))
fig_corr.subplots_adjust(wspace=0.3)

for i, row in correlation_df.iterrows():
    target = row['target']
    pred1 = row['pred1']
    pred2 = row['pred2']

    colors_corr = [
        base_colors[pred1],
        lighten_color(base_colors[pred1]),
        base_colors[pred2],
        lighten_color(base_colors[pred2])
    ]

    values = [row['pairwise_1'], row['partial_1'], row['pairwise_2'], row['partial_2']]
    bar = axes_corr[i].bar(range(4), values, color=colors_corr)

    axes_corr[i].set_title(f"Correlation with {target.capitalize()} RDM")
    axes_corr[i].set_xticks(range(4))
    axes_corr[i].set_xticklabels([
        pred1.capitalize(),
        f"{pred1.capitalize()}\n(partial {pred2.capitalize()})",
        pred2.capitalize(),
        f"{pred2.capitalize()}\n(partial {pred1.capitalize()})"
    ])
    axes_corr[i].set_ylim(-1, 1)
    axes_corr[i].spines['top'].set_visible(False)
    axes_corr[i].spines['right'].set_visible(False)

    if i == 0:
        axes_corr[i].set_ylabel("Spearman Correlation (ρ)")

    for b in bar:
        height = b.get_height()
        axes_corr[i].text(b.get_x() + b.get_width()/2, height + 0.05 * np.sign(height),
                          f"{height:.2f}", ha='center', va='bottom' if height > 0 else 'top')

fig_corr.suptitle("RDM Correlation Analysis", y=.96, fontsize=base_font_size * 1.6)
fig_corr.tight_layout()
plt.show()

# --- Figure 2: Variance Partitioning ---
fig_var, axes_var = plt.subplots(1, 3, sharey=True, figsize=(22, 10))
fig_var.subplots_adjust(wspace=0.3)

for i, row in plot_df.iterrows():
    target = row['target']
    pred1 = row['pred1']
    pred2 = row['pred2']

    colors_var = [
        base_colors[pred1],
        base_colors[pred2],
        sns.color_palette("tab20")[12],  # shared
        sns.color_palette("tab20")[13]   # unexplained
    ]

    values = [row['unique_1'], row['unique_2'], row['shared'], row['residual']]
    bar = axes_var[i].bar(range(4), values, color=colors_var)

    axes_var[i].set_title(f"Explaining {target.capitalize()} RDM")
    axes_var[i].set_xticks(range(4))
    axes_var[i].set_xticklabels([
        f"{pred1.capitalize()}",
        f"{pred2.capitalize()}",
        "Shared",
        "Unexplained"
    ])
    axes_var[i].set_ylim(0, 1.1)
    axes_var[i].spines['top'].set_visible(False)
    axes_var[i].spines['right'].set_visible(False)

    if i == 0:
        axes_var[i].set_ylabel("Proportion of Variance (R²)")

    for b in bar:
        height = b.get_height()
        axes_var[i].text(b.get_x() + b.get_width()/2, height + 0.02,
                         f"{height:.2f}", ha='center', va='bottom')

fig_var.suptitle("Variance Partitioning of RDMs", y=0.96, fontsize=base_font_size * 1.6)
fig_var.tight_layout()
plt.show()

# === 5. EXPORT LATEX TABLES ===

# Variance Partitioning Table
latex_var_table = plot_df.copy()
latex_var_table = latex_var_table.rename(columns={
    'target': 'Target RDM',
    'pred1': 'Predictor 1',
    'pred2': 'Predictor 2',
    'unique_1': 'Unique to P1',
    'unique_2': 'Unique to P2',
    'shared': 'Shared',
    'residual': 'Residual'
})
latex_var_table[['Unique to P1', 'Unique to P2', 'Shared', 'Residual']] = latex_var_table[
    ['Unique to P1', 'Unique to P2', 'Shared', 'Residual']
].applymap(lambda x: f"{x:.2f}")

print("=== LaTeX Table: Variance Partitioning ===\n")
print(latex_var_table.to_latex(index=False, escape=False))

# Correlation Table
latex_corr_table = correlation_df.copy()
latex_corr_table = latex_corr_table.rename(columns={
    'target': 'Target RDM',
    'pred1': 'Predictor 1',
    'pred2': 'Predictor 2',
    'pairwise_1': 'Pairwise P1',
    'partial_1': 'Partial P1 (|P2)',
    'pairwise_2': 'Pairwise P2',
    'partial_2': 'Partial P2 (|P1)',
})
latex_corr_table[['Pairwise P1', 'Partial P1 (|P2)', 'Pairwise P2', 'Partial P2 (|P1)']] = latex_corr_table[
    ['Pairwise P1', 'Partial P1 (|P2)', 'Pairwise P2', 'Partial P2 (|P1)']
].applymap(lambda x: f"{x:.2f}")

print("\n=== LaTeX Table: RDM Correlations ===\n")
print(latex_corr_table.to_latex(index=False, escape=False))
