import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def make_brain_cmap() -> LinearSegmentedColormap:
    """Consistent diverging colormap for brain maps (neg=blue/teal, pos=pink)."""
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

