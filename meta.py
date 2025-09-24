"""
Shared meta information for analyses: plotting style, ROI names, colors.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


ROI_NAME_MAP = {
    1: "Primary Visual",
    2: "Early Visual",
    3: "Dorsal Stream Visual",
    4: "Ventral Stream Visual",
    5: "MT+ Complex",
    6: "Somatosensory and Motor",
    7: "Paracentral Lobular and Mid Cing",
    8: "Premotor",
    9: "Posterior Opercular",
    10: "Early Auditory",
    11: "Auditory Association",
    12: "Insular and Frontal Opercular",
    13: "Medial Temporal",
    14: "Lateral Temporal",
    15: "Temporo-Parieto Occipital Junction",
    16: "Superior Parietal",
    17: "Inferior Parietal",
    18: "Posterior Cing",
    19: "Anterior Cing and Medial Prefrontal",
    20: "Orbital and Polar Frontal",
    21: "Inferior Frontal",
    22: "Dorsolateral Prefrontal",
}

# 22 colors matching ROI families order above
ROI_COLORS = (
    "#a6cee3", "#a6cee3",
    "#1f78b4", "#1f78b4", "#1f78b4",
    "#b2df8a", "#b2df8a", "#b2df8a", "#b2df8a",
    "#33a02c", "#33a02c", "#33a02c",
    "#fb9a99", "#fb9a99",
    "#e31a1c", "#e31a1c", "#e31a1c", "#e31a1c",
    "#fdbf6f", "#fdbf6f", "#fdbf6f", "#fdbf6f",
)


def apply_plot_style(base_font_size: int = 24) -> None:
    """Apply consistent plotting style (fonts, sizes)."""
    sns.set_style("white", {"axes.grid": False})
    plt.rcParams.update(
        {
            "font.family": "Ubuntu Condensed",
            "font.size": base_font_size,
            "axes.titlesize": base_font_size * 1.4,
            "axes.labelsize": base_font_size * 1.2,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": base_font_size,
        }
    )

