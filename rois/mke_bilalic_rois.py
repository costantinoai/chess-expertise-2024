import os
import re
import numpy as np
import nibabel as nib
import pandas as pd

from nilearn.image import coord_transform

###############################################################################
# 1) User Parameters
###############################################################################
ref_img_path = "data/MNI_Glasser_HCP_v1.0.nii"  # MNI reference
target_img_path = "data/beta_0001.nii"
out_dir = "results/bilalic_sphere_rois"
radius_mm = 5  # Sphere radius in mm
os.makedirs(out_dir, exist_ok=True)

# NOTE: we do everything in high res, and then downsample

###############################################################################
# 2) Hard-coded ROI Table
###############################################################################
# Each entry is a dict with columns:
#  - ROI               (e.g. "FFA")
#  - MNI Coordinates (L)
#  - MNI Coordinates (R)
#  - Source
#  - Information
# Cleaned up to standard "x, y, z" formats as strings

roi_table = [
    # Face/Body object detection studies
    {
        "ROI": "FFA",
        "MNI Coordinates (L)": "-38, -58, -14",
        "MNI Coordinates (R)": "40, -55, -12",
        "Source": "Schobert et al. (2018)",
        "Information": "N = 15"
    },
    {
        "ROI": "LOC",
        "MNI Coordinates (L)": "-44, -77, -12",
        "MNI Coordinates (R)": "44, -78, -13",
        "Source": "Bona et al. (2014)",
        "Information": "N = 15"
    },
    {
        "ROI": "PPA",
        "MNI Coordinates (L)": "-30, -50, -10",
        "MNI Coordinates (R)": "30, -54, -12",
        "Source": "Wang et al. (2019)",
        "Information": "N = 21"
    },
    {
        "ROI": "TPJ",
        "MNI Coordinates (L)": "-56, -47, 33",
        "MNI Coordinates (R)": "56, -47, 33",
        "Source": "Kovács et al. (2014)",
        "Information": "Peak activations across 26 studies"
    },
    {
        "ROI": "PCC",
        "MNI Coordinates (L)": "2, -30, 34",
        "MNI Coordinates (R)": "",
        "Source": "Dillen et al. (2016)",
        "Information": "Peak coordinates Bai et al. (2009)"
    },

    # Chess Studies
    # {
    #     "ROI": "CoS/PPA",
    #     "MNI Coordinates (L)": "-33, 39, 12",
    #     "MNI Coordinates (R)": "30, 42, 9",
    #     "Source": "Bilalic et al. 2010",
    #     "Information": "interaction Expertise x normal/random"
    # },
    {
        "ROI": "pMTL/OTJ",
        "MNI Coordinates (L)": "-47, -69, 8",
        "MNI Coordinates (R)": "48, -69, 15",
        "Source": "Bilalic et al. 2010",
        "Information": "main effect expertise"
    },
    # {
    #     "ROI": "OTJ",
    #     "MNI Coordinates (L)": "-47, -69, 8",
    #     "MNI Coordinates (R)": "55, -69, 14",
    #     "Source": "Bilalic et al. 2011",
    #     "Information": "Expertise effect object recognition"
    # },
    {
        "ROI": "pMTG",
        "MNI Coordinates (L)": "-60, -54, -3",
        "MNI Coordinates (R)": "58, -52, 1",
        "Source": "Bilalic et al. 2011",
        "Information": "Expertise effect object recognition"
    },
    {
        "ROI": "SMG",
        "MNI Coordinates (L)": "-60, -36, 36",
        "MNI Coordinates (R)": "63, -27, 42",
        "Source": "Bilalic et al. 2011",
        "Information": "Expertise effect object function"
    },
    # {
    #     "ROI": "CoS/PPA",
    #     "MNI Coordinates (L)": "-32, -43, -11",
    #     "MNI Coordinates (R)": "18, -52, 5",
    #     "Source": "Bilalic et al. 2012",
    #     "Information": "interaction Expertise x normal/random"
    # },
    # {
    #     "ROI": "RSC/PCC",
    #     "MNI Coordinates (L)": "-10, -75, 16",
    #     "MNI Coordinates (R)": "38, -36, -13",
    #     "Source": "Bilalic et al. 2012",
    #     "Information": "interaction Expertise x normal/random"
    # },
    # {
    #     "ROI": "SMG",
    #     "MNI Coordinates (L)": "-63, -31, 33",
    #     "MNI Coordinates (R)": "",
    #     "Source": "Bilalic et al. 2012",
    #     "Information": "expertise effect"
    # },
    # {
    #     "ROI": "pMTG/OTJ",
    #     "MNI Coordinates (L)": "-35, -80, 25",
    #     "MNI Coordinates (R)": "51, -69, 16",
    #     "Source": "Bilalic et al. 2012",
    #     "Information": "expertise effect"
    # },
    {
        "ROI": "Caudatus",
        "MNI Coordinates (L)": "-15, 13, 11",
        "MNI Coordinates (R)": "11, 18, 10",
        "Source": "Wan et al 2011",
        "Information": "decision making"
    },
]

###############################################################################
# 3) Prepare Reference Image & Output Volume
###############################################################################
ref_img = nib.load(ref_img_path)
ref_data = np.zeros(ref_img.shape, dtype=np.int32)  # initialize output volume to 0
affine = ref_img.affine  # used by coord_transform

region_info_rows = []
curr_region_id = 1  # We'll label each sphere with a unique integer

###############################################################################
# 4) Helper: Create a sphere mask (voxel-level) at an MNI coordinate
###############################################################################
def sphere_mask_in_3d(center_mni, radius_mm, out_shape):
    """
    Return a 3D boolean array of shape=out_shape indicating which voxels
    are within 'radius_mm' of 'center_mni' in MNI space.
    """
    i_idx, j_idx, k_idx = np.indices(out_shape)
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()

    # Convert voxel coords -> MNI
    x_mni, y_mni, z_mni = coord_transform(i_flat, j_flat, k_flat, affine)

    # Compute distance squared
    dist_sq = ((x_mni - center_mni[0])**2 +
               (y_mni - center_mni[1])**2 +
               (z_mni - center_mni[2])**2)
    inside_sphere = dist_sq <= (radius_mm**2)
    return inside_sphere.reshape(out_shape)

###############################################################################
# 5) Populate the Output Volume
###############################################################################
# ---------------------------------------------------------------------------
# Keep track of already-labeled IDs and names to detect collisions by name
# Example: assigned_regions = {1: "FFA", 2: "LOC", ...}
# Make sure this dictionary and 'curr_region_id' are defined BEFORE this loop,
# so they persist across all ROIs.
# ---------------------------------------------------------------------------

assigned_regions = {}
curr_region_id = 1

for entry in roi_table:
    roi_name = entry["ROI"]

    # Build a union mask for bilateral ROI
    union_mask = np.zeros(ref_data.shape, dtype=bool)

    # -----------------------------------------------------------------------
    # Left side
    # -----------------------------------------------------------------------
    left_str = entry["MNI Coordinates (L)"]
    if isinstance(left_str, str) and left_str.strip() and left_str.lower() != "n/a":
        coords = [float(x) for x in re.split(r'[, ]+', left_str.strip()) if x]
        if len(coords) == 3:
            left_mask = sphere_mask_in_3d(tuple(coords), radius_mm, ref_data.shape)
            union_mask |= left_mask  # combine with union

    # -----------------------------------------------------------------------
    # Right side
    # -----------------------------------------------------------------------
    right_str = entry["MNI Coordinates (R)"]
    if isinstance(right_str, str) and right_str.strip() and right_str.lower() != "n/a":
        coords = [float(x) for x in re.split(r'[, ]+', right_str.strip()) if x]
        if len(coords) == 3:
            right_mask = sphere_mask_in_3d(tuple(coords), radius_mm, ref_data.shape)
            union_mask |= right_mask  # combine with union

    # -----------------------------------------------------------------------
    # If union_mask is empty, skip
    # -----------------------------------------------------------------------
    if not union_mask.any():
        # No valid MNI coords for this ROI
        continue

    # -----------------------------------------------------------------------
    # Collision check: any nonzero IDs in these voxels?
    # -----------------------------------------------------------------------
    overlap_voxels = ref_data[union_mask]
    collision_ids = np.unique(overlap_voxels)
    collision_ids = collision_ids[collision_ids != 0]  # ignore background
    if len(collision_ids) > 0:
        # Gather the names of regions that occupy these voxels
        collision_names = [assigned_regions[cid] for cid in collision_ids]
        raise RuntimeError(
            f"Collision detected for ROI '{roi_name}' at ID={curr_region_id}.\n"
            f"Overlaps with existing IDs: {list(collision_ids)}\n"
            f"Corresponding to regions: {collision_names}"
        )

    # -----------------------------------------------------------------------
    # Assign the current ID to all voxels in the union mask
    # -----------------------------------------------------------------------
    ref_data[union_mask] = curr_region_id

    # Record the ID → name mapping so future collisions can reference it
    assigned_regions[curr_region_id] = roi_name

    # Optionally store full region metadata for output
    region_info_rows.append({
        "region_id": curr_region_id,
        "region_name": roi_name,  # single label for bilateral
        "Source": entry["Source"],
        "Information": entry["Information"],
        "color": ""
    })

    # Increment so next ROI uses a new label
    curr_region_id += 1

###############################################################################
# 6) Save the Final Single NIfTI Volume + TSV
###############################################################################

from nilearn.image import resample_to_img

target_img = nib.load(target_img_path)

out_img = nib.Nifti1Image(ref_data, affine, ref_img.header)
out_img_resampled = resample_to_img(out_img, target_img, interpolation="nearest")

nii_fname = os.path.join(out_dir, f"bilalic_spheres_{int(radius_mm)}mm.nii")
nib.save(out_img_resampled, nii_fname)

region_info_df = pd.DataFrame(region_info_rows)
tsv_fname = os.path.join(out_dir, f"region_info_{int(radius_mm)}mm.tsv")
region_info_df.to_csv(tsv_fname, sep="\t", index=False)

print(f"Done! Created {nii_fname} with {len(region_info_rows)} labeled spheres.")
print(f"Saved region info to {tsv_fname}.")

import os
import nibabel as nib
from nilearn import plotting

# Example: sample cut coordinates along Z
z_slices = [-30, -20, -10, 0, 10, 20, 30]

# Plot ROI as an overlay on the background
display = plotting.plot_roi(
    roi_img=out_img_resampled,
    display_mode='z',         # use axial slices
    cut_coords=z_slices,      # positions (in mm) for each slice
    cmap='tab20',             # or any colormap you like
    title='Bilateral Spheres (Montage)'
)

# Optionally save figure to disk
fig_path = os.path.join(out_dir, 'roi_montage.png')
display.close()

print(f"Saved montage figure to '{fig_path}'.")
