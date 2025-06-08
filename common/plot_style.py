import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Preferred colors and palette (from neurosynth module)
COL_POS = '#216421'
COL_NEG = '#8B2E2E'
PALETTE = [COL_POS, COL_NEG]

# --- Base style settings ---
# Master font size. All other sizes are expressed as multipliers of this value.
BASE_FONT_SIZE = 22
# Default figure size for all plots.
FIG_SIZE = (12, 9)

sns.set_style('white')
sns.set_palette(sns.color_palette(PALETTE))
plt.rcParams.update({
    'font.family': 'Ubuntu Condensed',
    'figure.figsize': FIG_SIZE,
    'figure.dpi': 300,
    'font.size': BASE_FONT_SIZE,
    'axes.titlesize': BASE_FONT_SIZE * 1.2,
    'axes.labelsize': BASE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE * 0.8,
    'ytick.labelsize': BASE_FONT_SIZE * 0.8,
    'legend.fontsize': BASE_FONT_SIZE * 0.8,
    'legend.title_fontsize': BASE_FONT_SIZE * 0.8,
    'legend.frameon': False,
    'legend.loc': 'best',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Custom colormap used for brain plots (copied from neurosynth)
def make_brain_cmap():
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

BRAIN_CMAP = make_brain_cmap()

TITLE_FONT = {
    'fontfamily': 'Ubuntu Condensed',
    'fontsize': BASE_FONT_SIZE,
    'fontweight': 'bold',
    'color': 'black',
    'backgroundcolor': 'white'
}
