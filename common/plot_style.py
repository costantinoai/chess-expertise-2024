import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Preferred colors and palette (from neurosynth module)
COL_POS = '#216421'
COL_NEG = '#8B2E2E'
PALETTE = [COL_POS, COL_NEG]

# Apply global seaborn/matplotlib style
sns.set_style('white')
sns.set_palette(sns.color_palette(PALETTE))
plt.rcParams['font.family'] = 'Ubuntu Condensed'
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['axes.labelsize'] = 22

# Custom colormap used for brain plots (copied from neurosynth)
def make_brain_cmap():
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

BRAIN_CMAP = make_brain_cmap()

TITLE_FONT = {
    'fontfamily': 'Ubuntu Condensed',
    'fontsize': 22,
    'fontweight': 'bold',
    'color': 'black',
    'backgroundcolor': 'white'
}
