import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
import nibabel as nib
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf, load_surf_mesh
from matplotlib import cm
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import SurfaceImage
from nilearn.datasets import load_fsaverage, load_fsaverage_data

# === 1. Load volumetric statistical image ===
stat_path = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-6_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM/2ndLevel_SMOOTH6_ExpVsNonExp_con_0002/spmT_0001.nii"
stat_img = nib.load(stat_path)

# # === 2. Fetch fsaverage mesh ===
# fsavg = fetch_surf_fsaverage('fsaverage')  # fsaverage5 is lower res

# # === 3. Load left hemisphere mesh ===
# coords_l, faces_l = load_surf_mesh(fsavg['infl_left'])  # use 'pial' for smoother surface
# surf_values_l = vol_to_surf(stat_img, fsavg['infl_left'], interpolation="linear")
from nilearn.datasets import load_fsaverage
from nilearn.surface import SurfaceImage
fsaverage_meshes = load_fsaverage("fsaverage")
surface_image = SurfaceImage.from_volume(
    mesh=fsaverage_meshes["pial"],
    volume_img=stat_img,
)

from nilearn.plotting import view_surf

fsaverage_sulcal = load_fsaverage_data(mesh="fsaverage", data_type="sulcal", mesh_type="pial")
view = view_surf(
    surf_mesh=fsaverage_meshes["inflated"],
    surf_map=surface_image,
    bg_map=fsaverage_sulcal,
    hemi="left",
    title="3D visualization in a web browser",
    threshold=0
)

view.open_in_browser()

# # Optional: Load and project right hemisphere if needed
# # coords_r, faces_r = load_surf_mesh(fsavg['pial_right'])
# # surf_values_r = vol_to_surf(stat_img, fsavg['pial_right'], interpolation="linear")

# # === 4. Use only left hemisphere for this demo ===
# coords = coords_l
# faces = faces_l
# values = surf_values_l.copy()

# # === 5. Apply threshold and normalize ===
# threshold = 0  # example threshold
# values[np.abs(values) < threshold] = np.nan

# # Normalize to [0, 1] for colormap
# vmax = np.nanmax(np.abs(values))
# norm = (values + vmax) / (2 * vmax)
# norm = np.clip(norm, 0.0, 1.0)

# # Map to RGB colors
# cmap = cm.get_cmap("seismic")
# vertex_rgb = np.array([
#     cmap(v)[:3] if not np.isnan(v) else (0, 0, 0)
#     for v in norm
# ])

# # === 6. Create interactive Plotly surface ===
# mesh = go.Mesh3d(
#     x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
#     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
#     vertexcolor=vertex_rgb,
#     lighting=dict(ambient=0.9),
#     flatshading=True,
#     hoverinfo="skip",
#     name="StatMap"
# )

# fig = go.Figure(mesh)
# fig.update_layout(
#     title="Surface Projection of Stat Map (Left Hemisphere)",
#     scene=dict(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         zaxis=dict(visible=False),
#         camera=dict(eye=dict(x=0, y=0, z=2.0))
#     ),
#     margin=dict(l=0, r=0, t=50, b=0),
#     height=800
# )

# === 7. Save and open in browser ===
out_html = "surface_projection.html"
pio.write_html(fig, file=out_html, auto_open=False)
webbrowser.open(out_html)
