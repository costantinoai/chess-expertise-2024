from typing import Dict, List, Optional
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from typing import Union


class Observation:
    def __init__(
        self,
        id: str,
        data: np.ndarray,
        attributes: Optional[Dict[str, Union[int, float, str]]] = None,
    ):
        self.id = id
        self.data = data  # Feature vector
        self.attributes = attributes or {}

    def __repr__(self):
        return f"Observation(id={self.id}, n_features={len(self.data)})"


class Dataset:
    """
    Dataset class encapsulating a set of observations and associated metadata (attributes).

    Attributes:
        observations: list of Observation instances
        attr_types: dict mapping attribute name → 'numerical' or 'categorical'
        rdm: cosine-based RDM of observations
        mds: MDS embedding in a reduced space (e.g., 2D)
        regressor_correlations: Spearman rho between dataset RDM and RDMs from each attribute
    """

    def __init__(
        self, observations: List[Observation], attr_types: Optional[Dict[str, str]] = None
    ):
        if not observations:
            raise ValueError("Dataset must contain at least one observation.")
        self.observations = observations
        self.attr_types = attr_types or {}
        self.rdm: Optional[np.ndarray] = None
        self.mds: Optional[np.ndarray] = None
        self.regressor_correlations: Dict[str, float] = {}

    def __len__(self):
        return len(self.observations)

    def __repr__(self):
        return f"<Dataset: {len(self)} observations, {len(self.attr_types)} attributes>"

    def compute_rdm(self):
        """Compute pairwise cosine dissimilarity RDM between observations."""
        data_matrix = np.array([obs.data for obs in self.observations])
        self.rdm = squareform(pdist(data_matrix, metric="cosine"))

    def compute_mds(self, n_components: int = 2):
        """Compute MDS embedding from RDM."""
        if self.rdm is None:
            self.compute_rdm()
        mds_model = MDS(
            n_components=n_components, dissimilarity="precomputed", random_state=42
        )
        self.mds = mds_model.fit_transform(self.rdm)

    def compute_regressor_correlations(self):
        """
        Compute Spearman rho between observation RDM and attribute-based RDMs.
        Uses Hamming distance for categorical, Euclidean for numerical.
        """
        if self.rdm is None:
            self.compute_rdm()

        for attr, typ in self.attr_types.items():
            values = [obs.attributes[attr] for obs in self.observations]
            if typ == "numerical":
                attr_rdm = squareform(
                    pdist(np.array(values).reshape(-1, 1), metric="euclidean")
                )
            elif typ == "categorical":
                attr_rdm = np.array(
                    [[0 if a == b else 1 for b in values] for a in values]
                )
            else:
                raise ValueError(f"Unknown attribute type for {attr}: {typ}")
            rho, _ = spearmanr(squareform(self.rdm), squareform(attr_rdm))
            self.regressor_correlations[attr] = rho

    def has_valid_mds(self, dim: int = 2) -> bool:
        """Check if MDS has been computed and has correct dimension."""
        return self.mds is not None and self.mds.shape[1] == dim

    def get_mds(self, dim: int = 2) -> Optional[np.ndarray]:
        """Return MDS, compute if missing."""
        if self.mds is None or self.mds.shape[1] != dim:
            self.compute_mds(dim)
        return self.mds

    def set_mds(self, embedding: np.ndarray):
        """Set MDS embedding explicitly."""
        self.mds = embedding

    def summary(self):
        """
        Print summary statistics about the dataset.
        """
        print(f"Dataset with {len(self.observations)} observations")
        print(f"Available attributes: {list(self.observations[0].attributes.keys())}")
        if self.rdm is not None:
            print(f"RDM shape: {self.rdm.shape}")
        if self.mds is not None:
            print(f"MDS shape: {self.mds.shape}")
        if self.regressor_correlations:
            print("Spearman correlations:")
            for k, v in self.regressor_correlations.items():
                print(f"  {k}: r={v:.3f}")


import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial import procrustes

logger = logging.getLogger(__name__)


# ============================================================================
# PARCEL
# ============================================================================
class Parcel:
    """
    Represents a single parcel (ROI) in a brain parcellation.

    Attributes:
        name (str): Unique identifier of the parcel.
        label (int): Numerical label from the LUT/annot file.
        hemisphere (str): 'left' or 'right'.
        vertices (List[int]): List of vertex indices that belong to this parcel.
        group (Optional[str]): Optional group identifier (e.g., 'visual').
        color (Optional[tuple]): RGBA color for visualization.
        dataset (Optional[Dataset]): Optional associated dataset.
        scalar_metrics (Dict[str, float]): Optional scalar values for ROI (e.g., SVM acc).
        extra (Dict[str, Any]): Arbitrary additional metadata for the parcel.
    """

    def __init__(
        self,
        name: str,
        label: int,
        hemisphere: str,
        vertices: List[int],
        group: Optional[str] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.name = name
        self.label = label
        self.hemisphere = hemisphere
        self.vertices = vertices
        self.group = group
        self.color = color
        self.dataset: Optional[Any] = None  # Must be assigned externally
        self.scalar_metrics: Dict[str, float] = {}
        self.extra: Dict[str, Any] = {}  # Extra metadata container

    def add_scalar_metric(self, key: str, value: float):
        """Add or update a scalar metric for this parcel."""
        self.scalar_metrics[key] = value

    def __str__(self):
        return f"<Parcel: {self.name} ({self.hemisphere}, {len(self.vertices)} verts)>"

    def __len__(self):
        return len(self.vertices)

    def __repr__(self):
        return f"Parcel(name={self.name}, label={self.label}, hemi={self.hemisphere})"

# ============================================================================
# PARCELLATION
# ============================================================================
class Parcellation:
    """
    Collection of parcels, potentially hierarchically organized.

    Attributes:
        name (str): Name of the parcellation.
        parcels (Dict[str, Parcel]): Map from parcel name to Parcel.
        hierarchy (Dict[int, List[str]]): Map from level ID to list of parcel names.
        meshes (Dict[str, Any]): Holds surface meshes (e.g., {'pial': ..., 'inflated': ...}).
    """

    def __init__(self, name: str):
        self.name = name
        self.parcels: Dict[str, Parcel] = {}
        self.hierarchy: Dict[int, List[str]] = {}
        self.meshes: Dict[str, Any] = {}

    def add_parcel(self, parcel: Parcel, level: int = 0):
        """
        Add a parcel to the parcellation and specify its hierarchical level.
        """
        self.parcels[parcel.name] = parcel
        self.hierarchy.setdefault(level, []).append(parcel.name)

    def get_parcels_by_level(self, level: int) -> List[Parcel]:
        """
        Return all parcels at the specified hierarchical level.
        """
        return [self.parcels[name] for name in self.hierarchy.get(level, [])]

    def add_dataset_to_parcel(self, parcel_name: str, dataset: Any):
        """
        Assign a dataset to a parcel by name.
        """
        if parcel_name not in self.parcels:
            raise KeyError(f"Parcel '{parcel_name}' not found in parcellation.")
        self.parcels[parcel_name].dataset = dataset
        logger.info(f"Dataset added to parcel: {parcel_name}")

    def align_mds_across_parcels(self, dim: int = 2):
        """
        Align MDS embeddings (if present) across all parcels using Procrustes analysis.

        MDS must be computed beforehand. This will replace MDS in each dataset
        with the aligned version in a shared space.
        """
        logger.info(f"Aligning MDS embeddings across parcels (dim={dim})")
        mds_dict = {
            name: p.dataset.mds
            for name, p in self.parcels.items()
            if p.dataset and p.dataset.mds is not None and p.dataset.mds.shape[1] == dim
        }

        if not mds_dict:
            logger.warning("No MDS data found across parcels.")
            return

        ref_name, ref = next(iter(mds_dict.items()))
        for name, mds in mds_dict.items():
            if name == ref_name:
                continue
            _, aligned, _ = procrustes(ref, mds)
            self.parcels[name].dataset.mds = aligned
            logger.debug(f"Aligned {name} to {ref_name} reference space")

        logger.info("All MDS embeddings aligned.")

    def plot_surface(self, level: int = 0, color_by: Optional[str] = None):
        """
        Plot the surface with parcels colored by a scalar metric or color property.
        Uses `nilearn.plotting.plot_surf_roi`.

        Args:
            level: hierarchy level to plot.
            color_by: if given, uses this scalar_metric to color the parcels.
        """
        from nilearn.plotting import plot_surf_roi
        from nilearn.surface import SurfaceImage
        from nilearn.datasets import load_fsaverage
        import numpy as np

        # Ensure required mesh is available
        if "inflated" not in self.meshes:
            raise RuntimeError("Mesh not loaded: expected key 'inflated' in self.meshes.")

        # Get inflated surface and parcel definitions
        surf = self.meshes["inflated"]
        parcels = self.get_parcels_by_level(level)

        # Initialize label vectors for both hemispheres
        label_vec = {
            "left": np.full(surf.parts["left"].coordinates.shape[0], -1),
            "right": np.full(surf.parts["right"].coordinates.shape[0], -1),
        }

        # Fill in parcel labels
        for parcel in parcels:
            label_vec[parcel.hemisphere][parcel.vertices] = parcel.label

        # Construct a SurfaceImage containing both hemispheres
        surface = SurfaceImage(mesh=surf, data=label_vec)

        # Plot both hemispheres using nilearn
        fig = plot_surf_roi(
            roi_map=surface,
            bg_map=surf,
            view="lateral",
            bg_on_data=True,
            engine="plotly",
            darkness=0.6,
            title=f"{self.name} – level {level}",
        )

        logger.info(f"Opening brain surface plot (level={level}) with both hemispheres...")
        fig.figure.show()
        return fig


    def __str__(self):
        return f"<Parcellation '{self.name}' with {len(self.parcels)} parcels>"

    def __len__(self):
        return len(self.parcels)

    def __repr__(self):
        levels = ", ".join(str(k) for k in self.hierarchy)
        return f"Parcellation(name={self.name}, levels=[{levels}])"


import numpy as np
from typing import Dict, Tuple
import logging
from nilearn.datasets import load_fsaverage
from nilearn.surface import load_surf_data, load_surf_mesh

logger = logging.getLogger(__name__)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__)


import pandas as pd
from collections import OrderedDict


class GlasserParcellation(Parcellation):
    """
    Specialized Parcellation class for Glasser atlas.
    Loads parcellation from .annot + LUT, adds region metadata (cortex, group).
    """

    def __init__(self, name="Glasser"):
        super().__init__(name=name)
        self.region_metadata: Dict[str, Dict] = {}
        self.metadata_csv: Optional[str] = None

    def load_from_glasser_files(
        self,
        annot_paths: Dict[str, str],
        lut_paths: Dict[str, str],
        metadata_csv: Optional[str] = None,
        mesh_kind: str = "inflated",
    ):
        fsavg = load_fsaverage("fsaverage")
        self.meshes = {
            kind: {
                "left": load_surf_mesh(fsavg[kind].parts["left"]),
                "right": load_surf_mesh(fsavg[kind].parts["right"]),
            }
            for kind in ["pial", "inflated", "flat"]
        }

        if metadata_csv:
            import os
            self._load_metadata_csv(metadata_csv)
            # Optional path to extra communities TSV
            community_path = metadata_csv.replace(".csv", "_dseg.tsv")
            if os.path.exists(community_path):
                self._load_community_metadata(community_path)


        cortex_map = {}
        group_map = {}

        for hemi in ["left", "right"]:
            labels = load_surf_data(annot_paths[hemi])
            coords, faces = self.meshes[mesh_kind][hemi]
            lut = self._load_lut(lut_paths[hemi])

            for label_id in np.unique(labels):
                if label_id == 0 or label_id not in lut:
                    continue

                name, rgba = lut[label_id]
                vertices = np.where(labels == label_id)[0].tolist()
                meta = self.region_metadata_by_id.get(label_id, {})

                cortex = meta.get("cortex", "unknown")
                cortex_id = meta.get("cortex_id", -1)
                group = cortex_info_map.get(cortex_id, {}).get("group", "unknown")

                # Level 0: ROIs
                parcel = Parcel(
                    name=name,
                    label=label_id,
                    hemisphere=hemi,
                    vertices=vertices,
                    group=cortex,
                    color=rgba
                )
                parcel.extra = meta.copy()
                self.add_parcel(parcel, level=0)

                cortex_map.setdefault((cortex, hemi), []).extend(parcel.vertices)
                group_map.setdefault((group, hemi), []).extend(parcel.vertices)

        # Level 1: Cortices
        def average_rgba(colors):
            if not colors:
                return (0.6, 0.6, 0.6, 1.0)
            rgb = np.mean([c[:3] for c in colors], axis=0)
            return (*rgb, 0.0)  # force alpha = 1.0

        # In level 1 loop (cortex)
        for (cortex, hemi), vertices in cortex_map.items():

            children = [p for p in self.get_parcels_by_level(0)
                        if p.hemisphere == hemi and p.extra.get("cortex") == cortex]

            avg_color = average_rgba([p.color for p in children if p.color])

            parcel = Parcel(
                name=f"{cortex}_{hemi}",
                label=-1,
                hemisphere=hemi,
                vertices=sorted(set(vertices)),
                group="cortex",
                color=avg_color
            )
            parcel.extra = {"hierarchy": "cortex"}
            self.add_parcel(parcel, level=1)

        # Level 2: Cortex Groups
        # Build group_name → color mapping
        # Build group-level parcels (level 2)
        for (group, hemi), verts in group_map.items():
            name = f"{group}_{hemi}"

            # Find the correct RGBA color from cortex_info_map (by finding one cortex_id that belongs to this group)
            matching_cortex_ids = [cid for cid, info in cortex_info_map.items() if info["group"] == group]
            if matching_cortex_ids:
                # Take the first matching cortex_id
                hex_color = cortex_info_map[matching_cortex_ids[0]]["color"]
                rgba = tuple(int(hex_color[i:i+2], 16)/255. for i in (1, 3, 5)) + (0.0,)
            else:
                rgba = (0.5, 0.5, 0.5, 1.0)  # fallback color

            parcel = Parcel(
                name=name,
                label=-2,
                hemisphere=hemi,
                vertices=sorted(set(verts)),
                group="group",
                color=rgba
            )
            parcel.extra = {"hierarchy": "group", "group": group}
            self.add_parcel(parcel, level=2)



    def _load_metadata_csv(self, path: str):
        df = pd.read_csv(path)
        self.region_metadata = {}
        self.region_metadata_by_id = {}

        for _, row in df.iterrows():
            name = row["regionName"]
            region_id = int(str(row["regionIdLabel"]).split("_")[0])  # e.g., "14_L" → 14

            meta = {
                "long_name": row["regionLongName"],
                "region": row["region"],
                "lobe": row["Lobe"],
                "cortex": row["cortex"],
                "cortex_id": int(row["Cortex_ID"]),
                "coords": (row["x-cog"], row["y-cog"], row["z-cog"]),
                "volume_mm3": row["volmm"],
            }

            self.region_metadata[name] = meta
            self.region_metadata_by_id[region_id] = meta

    def _load_lut(
        self, path: str
    ) -> Dict[int, Tuple[str, Tuple[float, float, float, float]]]:
        """
        Load .annot LUT file (txt) and map label index to (name, RGBA).
        """
        lut = {}
        with open(path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 6:
                        idx = int(parts[0])
                        name = parts[1]
                        r, g, b, a = map(int, parts[2:6])
                        rgba = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
                        lut[idx] = (name, rgba)
        return lut
    def _load_community_metadata(self, tsv_path: str):
        """
        Load additional community labels (Yeo, Mesulam, Economo) and assign them
        to parcels using the index (regionIdLabel) match.
        """
        df = pd.read_csv(tsv_path, sep="\t")
        for _, row in df.iterrows():
            try:
                idx = int(row["index"])
                yeo = row.get("community_yeo", None)
                mesulam = row.get("community_mesulam", None)
                economo = row.get("community_economo", None)

                if idx in self.region_metadata_by_id:
                    self.region_metadata_by_id[idx]["community_yeo"] = yeo
                    self.region_metadata_by_id[idx]["community_mesulam"] = mesulam
                    self.region_metadata_by_id[idx]["community_economo"] = economo
            except Exception as e:
                print(f"Failed to parse row {row} in community metadata: {e}")

    def assign_rsa_stats(self, rsa_results: Dict):
        for regressor, cortex_dict in rsa_results.items():
            for cortex_name, stats in cortex_dict.items():
                for parcel in self.parcels.values():
                    # print(parcel.extra)
                    if "hierarchy" in parcel.extra.keys() and parcel.extra["hierarchy"]  == "group":
                        continue
                    try:
                        clean_name = parcel.extra["region"]
                    except:
                        clean_name = parcel.name.replace("_left", "")

                    if clean_name.lower() == cortex_name.lower():
                        if "rsa" not in parcel.extra:
                            parcel.extra["rsa"] = {}
                        parcel.extra["rsa"][regressor] = stats


cortex_info_map = OrderedDict(
    {
        # Group 1: Early Visual
        1: {"color": "#a6cee3", "group": "Early Visual"},  # Primary_Visual
        2: {"color": "#a6cee3", "group": "Early Visual"},  # Early_Visual
        # Group 2: Intermediate Visual
        3: {"color": "#1f78b4", "group": "Intermediate Visual"},
        4: {"color": "#1f78b4", "group": "Intermediate Visual"},
        5: {"color": "#1f78b4", "group": "Intermediate Visual"},
        # Group 3: Sensorimotor
        6: {"color": "#b2df8a", "group": "Sensorimotor"},
        7: {"color": "#b2df8a", "group": "Sensorimotor"},
        8: {"color": "#b2df8a", "group": "Sensorimotor"},
        9: {"color": "#b2df8a", "group": "Sensorimotor"},
        # Group 4: Auditory
        10: {"color": "#33a02c", "group": "Auditory"},
        11: {"color": "#33a02c", "group": "Auditory"},
        12: {"color": "#33a02c", "group": "Auditory"},
        # Group 5: Temporal
        13: {"color": "#fb9a99", "group": "Temporal"},
        14: {"color": "#fb9a99", "group": "Temporal"},
        # Group 6: Posterior
        15: {"color": "#e31a1c", "group": "Posterior"},
        16: {"color": "#e31a1c", "group": "Posterior"},
        17: {"color": "#e31a1c", "group": "Posterior"},
        18: {"color": "#e31a1c", "group": "Posterior"},
        # Group 7: Anterior
        19: {"color": "#fdbf6f", "group": "Anterior"},
        20: {"color": "#fdbf6f", "group": "Anterior"},
        21: {"color": "#fdbf6f", "group": "Anterior"},
        22: {"color": "#fdbf6f", "group": "Anterior"},
    }
)

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from nilearn.datasets import load_fsaverage_data
from nibabel import Nifti1Image

class BrainViewer:
    def __init__(self, parcellation, port: int = 8050):
        self.parcellation = parcellation
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.port = port
        self.sulcal_data = load_fsaverage_data("fsaverage", data_type="curvature").data.parts
        self._prepare_layout()
        self._register_callbacks()
        self.vmin, self.vmax = self._compute_global_rsa_ranges()[0]
        # self.rsa_ylim = self._compute_global_rsa_ranges()[1]
        self.rsa_ylim = (-.1,.2)
        self.stat_map_surface: Dict[str, np.ndarray] = {}  # hemi → surface values
        self.stat_map_img: Optional[Nifti1Image] = None  # cache the loaded NIfTI image

    def __repr__(self):
        return f"<BrainViewer for {self.parcellation.name}, port={self.port}>"

    def __str__(self):
        return f"BrainViewer with parcellation '{self.parcellation.name}' running on port {self.port}"

    def _sulcal_to_gray(self, sulc: np.ndarray, invert: bool = True) -> np.ndarray:
        """Convert sulcal depth to grayscale RGB."""
        norm = (sulc - np.min(sulc)) / (np.max(sulc) - np.min(sulc))
        if invert:
            norm = 1.0 - norm
        return np.stack([norm, norm, norm], axis=-1)

    from typing import Tuple

    def _compute_global_rsa_ranges(self, margin_ratio: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute global RSA value ranges:
        - color_range: Symmetric around 0, based on mean_diff across all parcels/regressors
        - bar_ylim: Y-axis limits based on lower/upper CI bounds across all parcels/regressors

        Returns:
            color_range: (vmin, vmax) for color mapping
            bar_ylim: (ymin, ymax) for barplot y-axis
        """
        mean_diffs = []
        ci_lows = []
        ci_highs = []

        for parcel in self.parcellation.parcels.values():
            rsa = parcel.extra.get("rsa", {})
            for stats in rsa.values():
                try:
                    mean = float(stats.get("mean_diff", 0.0))
                    ci_low, ci_high = stats["CI95"]

                    mean_diffs.append(mean)
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)
                except Exception:
                    continue

        # Fallback defaults
        if not mean_diffs or not ci_lows or not ci_highs:
            return (-1.0, 1.0), (-1.0, 1.0)

        # Symmetric color range centered on 0
        max_abs_mean = max(abs(min(mean_diffs)), abs(max(mean_diffs)))
        color_range = (-max_abs_mean, max_abs_mean)

        # CI range with margin
        ymin, ymax = min(ci_lows), max(ci_highs)
        margin = (ymax - ymin) * margin_ratio
        bar_ylim = (ymin - margin, ymax + margin)

        return color_range, bar_ylim


    def _prepare_layout(self):
        all_regressors = sorted({
            reg for p in self.parcellation.parcels.values()
            if "rsa" in p.extra for reg in p.extra["rsa"]
        })

        self.app.layout = html.Div([
            html.H2("Interactive Brain Viewer with ROI Analytics"),
            html.Div(style={"display": "flex"}, children=[
                html.Div(style={"width": "45%", "padding": "10px"}, children=[
                    dcc.Dropdown(
                        id="level-dropdown",
                        options=[{"label": f"Level {lvl}", "value": lvl} for lvl in sorted(self.parcellation.hierarchy)],
                        value=0,
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    dcc.Dropdown(
                        id="mesh-type-dropdown",
                        options=[
                            {"label": "Inflated", "value": "inflated"},
                            {"label": "Pial", "value": "pial"},
                            {"label": "Flat", "value": "flat"},
                        ],
                        value="inflated",
                        clearable=False,
                        style={"marginTop": "10px", "width": "100%"},
                    ),
                    dcc.Checklist(
                        id="hemi-toggle",
                        options=[{"label": "Left", "value": "left"}, {"label": "Right", "value": "right"}],
                        value=["left", "right"],
                        inline=True,
                        style={"marginTop": "10px"},
                    ),
                    dcc.Checklist(
                        id="rsa-toggle",
                        options=[{"label": "Color by RSA significance", "value": "rsa"}],
                        value=[],
                        style={"marginTop": "10px"},
                    ),
                    dcc.Dropdown(
                        id="rsa-regressor-dropdown",
                        options=[{"label": r, "value": r} for r in all_regressors],
                        value=all_regressors[0] if all_regressors else None,
                        style={"marginTop": "10px", "width": "100%"},
                    ),
                    html.Hr(),
                    dcc.Upload(
                        id='statmap-upload',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select a Statistical Map')
                        ]),
                        style={
                            'width': '100%', 'height': '60px',
                            'lineHeight': '60px', 'borderWidth': '1px',
                            'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'marginTop': '10px'
                        },
                        multiple=False
                    ),
                    html.Div([
                        html.Label("Stat Map Threshold (t > x):"),
                        dcc.Slider(
                            id='stat-threshold-slider',
                            min=0, max=10, step=0.1, value=1.96,
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={"marginTop": "10px", "display": "block"}),
                    dcc.Checklist(
                        id="statmap-toggle",
                        options=[{"label": "Show Stat Map", "value": "stat"}],
                        value=[],
                        style={"marginTop": "10px"},
                    ),
                    dcc.Graph(id="brain-3d", style={"height": "750px"}),
                ]),
                html.Div(style={"width": "55%", "padding": "10px"}, children=[
                    html.Div(id="parcel-info", style={"marginBottom": "10px"}),
                    dcc.Graph(id="rsa-effect-barplot", style={"height": "400px"}),
                ])
            ])
        ])


    def _register_callbacks(self):
        import matplotlib.cm as cm
        import nibabel as nib
        from plotly.subplots import make_subplots
        from nilearn.datasets import fetch_surf_fsaverage

        @self.app.callback(
            Output("brain-3d", "figure"),
            Input("level-dropdown", "value"),
            Input("hemi-toggle", "value"),
            Input("rsa-toggle", "value"),
            Input("rsa-regressor-dropdown", "value"),
            Input("mesh-type-dropdown", "value"),
            Input("statmap-toggle", "value"),
            State("brain-3d", "relayoutData"),
            Input("stat-threshold-slider", "value"),
        )
        def update_surface(level, hemis, rsa_toggle, rsa_regressor, mesh_type,
                           statmap_toggle, relayoutData, stat_threshold):

            coords_all, faces_all, hover_all, custom_all, rgb_all = [], [], [], [], []
            offset = 100.0
            total_offset = 0.0
            rsa_mode = ("rsa" in rsa_toggle and rsa_regressor is not None)
            stat_mode = ("stat" in statmap_toggle and self.stat_map_path is not None)
            cmap = cm.get_cmap("seismic")

            # Preserve camera view
            camera = relayoutData.get("scene.camera") if relayoutData else None

            for hemi in ["left", "right"]:
                if hemi not in hemis:
                    continue
                # Load surface file path and geometry
                fsavg = fetch_surf_fsaverage('fsaverage')
                mesh_kind = "infl" if mesh_type == "inflated" else mesh_type
                coords, faces = load_surf_mesh(fsavg[f'{mesh_kind}_{hemi}'])

                n_verts = coords.shape[0]

                # Initialize sulcal background
                sulc_rgb = self._sulcal_to_gray(self.sulcal_data[hemi])
                vertex_rgb = sulc_rgb.copy()

                if stat_mode:
                    try:
                        # Project volume onto surface (uses filename, not coords)
                        surface_image = self.stat_map_surf

                        # Thresholding
                        vals = surface_image.data.parts[hemi].copy()
                        vals[np.abs(vals) < stat_threshold] = np.nan

                        if np.any(~np.isnan(vals)):
                            vmax = np.nanmax(np.abs(vals))
                            norm = (vals + vmax) / (2 * vmax)
                            norm = np.clip(norm, 0.0, 1.0)

                            for idx, v in enumerate(norm):
                                if not np.isnan(v):
                                    vertex_rgb[idx] = cmap(v)[:3]


                    except Exception as e:
                        print(f"Failed to project stat map to {hemi}: {e}")
                        vals = np.full(n_verts, np.nan)

                # Initialize hover and custom data arrays
                hovertext = np.array([""] * n_verts, dtype=object)
                customdata = np.array([""] * n_verts, dtype=object)

                for p in self.parcellation.get_parcels_by_level(level):
                    if p.hemisphere != hemi:
                        continue
                    hovertext[p.vertices] = f"{p.name} ({p.hemisphere})"
                    customdata[p.vertices] = p.name

                    if not stat_mode:
                        if rsa_mode:
                            stats = p.extra.get("rsa", {}).get(rsa_regressor, {})
                            if stats.get("fdr_reject", False):
                                m = stats["mean_diff"]
                                nv = (m - self.vmin) / (self.vmax - self.vmin)
                                nv = np.clip(nv, 0.0, 1.0)
                                vertex_rgb[p.vertices] = cmap(nv)[:3]
                        elif p.color:
                            vertex_rgb[p.vertices] = p.color[:3]

                # Apply offset to coordinates (after projection, before stacking)
                if hemi == "right":
                    coords[:, 0] += offset

                coords_all.append(coords)
                faces_all.append(faces + int(total_offset))
                rgb_all.append(vertex_rgb)
                hover_all.append(hovertext)
                custom_all.append(customdata)
                total_offset += n_verts

            coords = np.vstack(coords_all)
            faces = np.vstack(faces_all)
            vertexcolor = np.vstack(rgb_all)
            hovertext = np.concatenate(hover_all)
            customdata = np.concatenate(custom_all)

            mesh = go.Mesh3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                vertexcolor=vertexcolor,
                hovertext=hovertext,
                customdata=customdata,
                hoverinfo="text",
                lighting=dict(ambient=1.0),
                flatshading=True,
            )

            fig = go.Figure([mesh])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=camera or dict(eye=dict(x=0, y=0, z=2)),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=750,
            )
            return fig



        from nilearn.image import load_img

        @self.app.callback(
            Output("brain-3d", "figure", allow_duplicate=True),
            Input("statmap-upload", "contents"),
            State("statmap-upload", "filename"),
            prevent_initial_call=True,
        )
        def load_stat_map(contents, filename):
            import base64, tempfile
            if contents is None:
                return dash.no_update

            # Decode base64 and save to temporary file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name

            self.stat_map_path = tmp_path
            self.stat_map_img = nib.load(tmp_path)  # Load and cache
            from nilearn.datasets import load_fsaverage
            from nilearn.surface import SurfaceImage
            fsaverage_meshes = load_fsaverage("fsaverage")
            self.stat_map_surf = SurfaceImage.from_volume(
                mesh=fsaverage_meshes["pial"],
                volume_img=self.stat_map_img,
            )

            print(f"Stat map loaded once from {self.stat_map_path}")
            return dash.no_update

        @self.app.callback(
            [Output("rsa-effect-barplot", "figure"),
             Output("parcel-info", "children")],
            Input("brain-3d", "clickData"),
            Input("level-dropdown", "value"),
        )
        def show_rsa_bars(clickData, level):
            if not clickData or "points" not in clickData:
                return go.Figure(), "Click on a parcel"

            name = clickData["points"][0].get("customdata")
            if name not in self.parcellation.parcels:
                return go.Figure(), f"No data for {name}"

            parcel = self.parcellation.parcels[name]

            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.9, 0.1],
                specs=[[{"type": "bar"}, {"type": "heatmap"}]],
                horizontal_spacing=0.05,
            )

            for reg, stats in parcel.extra.get("rsa", {}).items():
                try:
                    mean = float(stats["mean_diff"])
                    ci_low, ci_high = stats["CI95"]
                    fig.add_trace(
                        go.Bar(
                            x=[reg], y=[mean],
                            error_y=dict(
                                type='data',
                                array=[ci_high - mean],
                                arrayminus=[mean - ci_low],
                                visible=True,
                            ),
                            marker_color="indianred",
                            name=reg,
                        ),
                        row=1, col=1
                    )

                    if stats.get("fdr_reject", False):
                        fig.add_trace(
                            go.Scatter(
                                x=[reg], y=[ci_high + 0.001],
                                text=["*"], mode="text",
                                textposition="top center",
                                showlegend=False,
                                textfont=dict(size=20)
                            ),
                            row=1, col=1
                        )
                except Exception as e:
                    print(f"RSA barplot error for {reg}: {e}")

            fig.update_layout(
                title="RSA Effects (mean ± CI)",
                height=400,
                yaxis1=dict(title="Effect size", range=list(self.rsa_ylim)),
                showlegend=False,
            )

            meta_items = [
                html.Li(f"Name: {parcel.extra['long_name']}"),
                html.Li(f"Hemisphere: {parcel.hemisphere}"),
                html.Li(f"Vertices: {len(parcel.vertices)}"),
                html.Li(f"Color: {parcel.color}"),
                html.Li(f"Group: {parcel.group}"),
                html.Li(f"Has Dataset: {'Yes' if parcel.dataset else 'No'}"),
            ]

            # Add community metadata if available
            community_keys = ["community_yeo", "community_mesulam", "community_economo"]
            for key in community_keys:
                val = parcel.extra.get(key)
                if val:
                    meta_items.append(html.Li(f"{key.replace('_', ' ').title()}: {val}"))

            meta = html.Div([
                html.H4("Parcel Metadata"),
                html.Ul(meta_items)
            ], style={"padding": "10px", "fontSize": "16px"})

            return fig, meta

    def run(self):
        self.app.run(debug=True, port=self.port)
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{self.port}/")



from collections import OrderedDict

# Load Glasser
annot_paths = {
    "left": "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/lh.HCPMMP1.annot",
    "right": "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/rh.HCPMMP1.annot",
}
lut_paths = {
    "left": "/data/projects/chess/data/misc/lh_HCPMMP1_color_table.txt",
    "right": "/data/projects/chess/data/misc/rh_HCPMMP1_color_table.txt",
}

parcellation = GlasserParcellation()

parcellation.load_from_glasser_files(
    annot_paths,
    lut_paths,
    "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/data/HCP-MMP1_UniqueRegionList.csv",
)

# Load RSA results
import pickle

# with open("/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-230003_glasser_regions_bilateral/rsa_corr/20250403-153416_group/ttest_group_results.pkl", "rb") as f:
with open("/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-230003_glasser_regions_bilateral/rsa_corr/20250403-153416_group/ttest_group_results.pkl", "rb") as f:
    results = pickle.load(f)

parcellation.assign_rsa_stats(results["experts_vs_nonexperts"])

viewer = BrainViewer(parcellation)
viewer.run()
