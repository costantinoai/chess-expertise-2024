# === FULL UPDATED SCRIPT for 4 Regressors and Variance Partitioning ===

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

# ==== 1. DATA PREPARATION ====
raw_data = {
    'check_n': [3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 1, 1, 1],
    'strategy': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'legal_moves': [38, 40, 41, 47, 50, 36, 39, 41, 38, 47, 34, 35, 41, 43, 46, 30, 45, 48, 33, 35],
    'motif': [
        'deflection/pulling', 'defence removal', 'deflection/pulling',
        'defence removal', 'deflection/pulling', 'other (overextension)', 'defence removal',
        'other (overextension)', 'deflection/pulling', 'deflection/pulling', 'deflection/pulling',
        'deflection/pulling', 'deflection/pulling', 'defence removal', 'defence removal',
        'other (straightforward checkmate)', 'deflection/pulling',
        'other (straightforward checkmate)', 'other (straightforward checkmate)', 'other (straightforward checkmate)'
    ]
}
data = pd.DataFrame(raw_data)
data['motif'] = pd.Categorical(data['motif']).codes

# ==== 2. RDM CONSTRUCTION ====
distance_metrics = {
    'check_n': 'euclidean',
    'strategy': 'hamming',
    'legal_moves': 'euclidean',
    'motif': 'hamming'
}
# Mapping of internal names to prettier labels
pretty_names = {
    'check_n': 'Check #',
    'strategy': 'Strategy',
    'legal_moves': 'Legal Moves',
    'motif': 'Motif'
}
def build_rdm(column, metric):
    return squareform(pdist(data[[column]], metric=metric))

rdms = {col: build_rdm(col, metric) for col, metric in distance_metrics.items()}

def flatten_rdm(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]

flat_rdms = {key: flatten_rdm(rdm) for key, rdm in rdms.items()}
corr_data = pd.DataFrame(flat_rdms)
pairwise_corr = corr_data.corr(method='spearman')

# ==== 3. VARIANCE PARTITIONING ====
def partition_variance(df, target, predictors):
    """
    Compute the unique, shared, and residual variance explained in a target variable
    using nested linear regression models.
    """
    y = df[target]
    X = df[predictors]
    full_model = LinearRegression().fit(X, y)
    r2_full = r2_score(y, full_model.predict(X))
    unique_r2 = {}
    for pred in predictors:
        reduced = [p for p in predictors if p != pred]
        reduced_model = LinearRegression().fit(df[reduced], y)
        r2_reduced = r2_score(y, reduced_model.predict(df[reduced]))
        unique_r2[pred] = r2_full - r2_reduced
    shared = r2_full - sum(unique_r2.values())
    residual = 1 - r2_full
    return {
        'target': target,
        **{f'unique_{pred}': max(val, 0) for pred, val in unique_r2.items()},
        'shared': max(shared, 0),
        'residual': max(residual, 0),
        'predictors': predictors
    }

targets = list(corr_data.columns)
predictors = [col for col in corr_data.columns]
variance_data = [
    partition_variance(corr_data, target, [p for p in predictors if p != target])
    for target in targets
]
plot_df = pd.DataFrame(variance_data)

# ==== 4. PLOTTING ====
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

base_colors = dict(zip(corr_data.columns, sns.color_palette("Set2", n_colors=len(corr_data.columns))))
base_colors["motif"] = sns.color_palette("Set2")[6]

def lighten_color(color, amount=0.5):
    r, g, b, a = to_rgba(color)
    white = np.array([1, 1, 1])
    return tuple(np.array([r, g, b]) + (white - np.array([r, g, b])) * amount)

# --- Figure 1: Correlation Plot ---
fig_corr, axes_corr = plt.subplots(1, len(corr_data.columns), sharey=True, figsize=(35, 10))
fig_corr.subplots_adjust(wspace=0.3)

for i, target in enumerate(corr_data.columns):
    others = [col for col in corr_data.columns if col != target]
    values = [pairwise_corr.loc[target, other] for other in others]
    colors = [base_colors[other] for other in others]

    bars = axes_corr[i].bar(range(len(values)), values, color=colors)
    axes_corr[i].set_title(f"Correlations with {pretty_names.get(target, target)}")
    axes_corr[i].set_xticks(range(len(others)))
    axes_corr[i].set_xticklabels([pretty_names.get(name, name).replace(" ", "\n") for name in others])
    axes_corr[i].set_ylim(-1, 1)
    axes_corr[i].spines['top'].set_visible(False)
    axes_corr[i].spines['right'].set_visible(False)

    if i == 0:
        axes_corr[i].set_ylabel("Spearman Correlation (ρ)")

    for b, val in zip(bars, values):
        axes_corr[i].text(b.get_x() + b.get_width()/2, val + 0.05 * np.sign(val),
                          f"{val:.2f}", ha='center', va='bottom' if val > 0 else 'top')

fig_corr.suptitle("Pairwise Spearman Correlations Across RDMs", y=0.95, fontsize=base_font_size * 1.6)
fig_corr.tight_layout()
plt.show()

# --- Figure 2: Variance Partitioning ---
fig_var, axes_var = plt.subplots(1, len(plot_df), sharey=True, figsize=(35, 10))
fig_var.subplots_adjust(wspace=0.3)

for i, row in plot_df.iterrows():
    target = row['target']
    predictors = row['predictors']
    values = [row[f'unique_{p}'] for p in predictors] + [row['shared'], row['residual']]
    colors = [base_colors[p] for p in predictors] + [sns.color_palette("tab20")[12], sns.color_palette("tab20")[13]]
    xticks = [pretty_names.get(p,p).replace(" ", "\n") for p in predictors] + ["Shared", "Unexplained"]

    bars = axes_var[i].bar(range(len(values)), values, color=colors)
    axes_var[i].set_title(f"Explaining {pretty_names.get(target, target)}")
    axes_var[i].set_xticks(range(len(values)))
    axes_var[i].set_xticklabels(xticks)
    axes_var[i].set_ylim(0, 1.1)
    axes_var[i].spines['top'].set_visible(False)
    axes_var[i].spines['right'].set_visible(False)


    if i == 0:
        axes_var[i].set_ylabel("Proportion of Variance (R²)")

    for b, val in zip(bars, values):
        axes_var[i].text(b.get_x() + b.get_width()/2, val + 0.02,
                         f"{val:.2f}", ha='center', va='bottom')

fig_var.suptitle("Variance Partitioning for Each RDM Target", y=0.95, fontsize=base_font_size * 1.6)
fig_var.tight_layout()
plt.show()
