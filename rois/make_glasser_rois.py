import os
import pandas as pd
import nibabel as nib
import numpy as np
import re
import pandas as pd

def parse_afni_value_label_dtable(head_path):
    label_map = {}
    with open(head_path, 'r') as f:
        lines = f.readlines()

    in_dtable = False
    for line in lines:
        if 'name = VALUE_LABEL_DTABLE' in line:
            in_dtable = True
            continue
        if in_dtable:
            if line.strip().startswith("'<VALUE_LABEL_DTABLE"):
                continue  # skip header line
            elif line.strip().endswith("~'"):
                break  # end of dtable
            else:
                match = re.match(r'^\s*(\d+)\s+(.*)', line)
                if match:
                    idx = int(match.group(1))
                    label = match.group(2).strip()
                    label_map[idx] = label
    return label_map

# ----------------------------------------------------------------------------
# 1) Load & Merge CSV/TSV
# ----------------------------------------------------------------------------

# File paths
file_csv = "data/HCP-MMP1_UniqueRegionList.csv"          # regionName, regionID, etc.
file_tsv = "data/atlas-Glasser_dseg.tsv"                 # label, cifti_label, etc.
atlas_path = "data/MNI_Glasser_HCP_v1.0_resampled.nii"   # volumetric Glasser atlas

df1 = pd.read_csv(file_csv)          # columns like: regionName, regionID, Cortex_ID, ...
df2 = pd.read_csv(file_tsv, sep='\t')# columns like: label, cifti_label, ...

# Helper: Convert "V1_L" -> "L_V1_ROI", "MST_R" -> "R_MST_ROI", etc.
def regionName_to_cifti(region_name):
    hemi = region_name[-1]
    roi = region_name[:-2]
    return f"{hemi}_{roi}_ROI".upper()  # force uppercase

# Apply mapping and ensure both sides are uppercase for case-insensitive join
df1["cifti_label"] = df1["regionName"].apply(regionName_to_cifti)
df2["cifti_label"] = df2["cifti_label"].str.upper()

df_merged = pd.merge(df1, df2, on="cifti_label", how="left")

# Parse label map from AFNI .HEAD
head_path = "data/hcp_rank+tlrc.HEAD"

# Parse label map from AFNI HEAD file
img_label_map = parse_afni_value_label_dtable(head_path)

df_img_labels = pd.DataFrame([
    {"img_label": k, "img_label_name": v}
    for k, v in img_label_map.items()
    if k != 0
])

df_merged["img_label"] = df_img_labels["img_label"]
df_merged["original_name"] = df_img_labels["img_label_name"]

# ----------------------------------------------------------------------------
# 2) Create a simple region-info table [region_id, region_name, color]
# ----------------------------------------------------------------------------

# Apply the function to the 'Cortex_ID' column to create the 'color' column.
df_merged["color"] = ""

# Use img_label and regionLongName instead
region_info_cols = ["img_label", "regionLongName", "color"]
df_merged_info = df_merged[region_info_cols].drop_duplicates()

# Rename columns as required
df_merged_info = df_merged_info.rename(columns={
    "img_label": "region_id",
    "regionLongName": "region_name"
})

# ----------------------------------------------------------------------------
# Output 3: Single ROI Volume (one file, each voxel = regionID)
# ----------------------------------------------------------------------------
# Ensure 'color' column exists
def assign_color(cortex_id):
    # Return the color from cortex_info_map for a given cortex_id.
    # If the cortex_id is not found, a default color (e.g., black) is returned.
    from collections import OrderedDict
    cortex_info_map = OrderedDict({
        # Group 1: Early Visual
        1: {"color": "#a6cee3", "group": "Early Visual"},      # Primary_Visual
        2: {"color": "#a6cee3", "group": "Early Visual"},      # Early_Visual

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
    })

    return cortex_info_map.get(cortex_id, {}).get("color", "#000000")

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
output_data = np.zeros_like(atlas_data, dtype=np.int32)

out_dir = "results/glasser_regions"
os.makedirs(out_dir, exist_ok=True)

# Apply the function to the 'Cortex_ID' column to create the 'color' column.
df_merged["color"] = df_merged["Cortex_ID"].apply(assign_color)
df_merged = df_merged.sort_values(by=["Cortex_ID", "regionID"]).reset_index(drop=True)
df_merged["order"] = range(len(df_merged))

df_single_info = df_merged[["regionID", "regionName", "color"]].drop_duplicates()

for _, row in df_single_info.iterrows():
    rid = row["regionID"]
    mask = (atlas_data == rid)

    if mask.sum() == 0:
        continue

    # Safety: check for collision
    if output_data[mask].sum() != 0:
        raise RuntimeError(f"Collision detected at regionID {rid}")

    output_data[mask] = rid

# Save one combined image
nib.save(nib.Nifti1Image(output_data, atlas_img.affine, atlas_img.header),
         os.path.join(out_dir, "glasser_regions.nii"))

df_single_info.rename(columns={"regionID": "region_id", "regionName": "region_name"}, inplace=True)
df_single_info.to_csv(os.path.join(out_dir, "region_info.tsv"), sep="\t", index=False)

# ----------------------------------------------------------------------------
# Output 4: Bilateral ROI Volume (one file, voxel = merged regionID)
# ----------------------------------------------------------------------------

df_merged["base_name"] = df_merged["regionName"].str[:-2]
output_data = np.zeros_like(atlas_data, dtype=np.int32)

out_dir = "results/glasser_regions_bilateral"
os.makedirs(out_dir, exist_ok=True)

bilat_info_rows = []
for base, subdf in df_merged.groupby("base_name"):
    rids = subdf["regionID"].unique()
    if len(rids) == 0:
        continue
    new_rid = min(rids)
    color = subdf["color"].values[0]
    order = subdf["order"].values[0]

    mask = np.isin(atlas_data, rids)
    if mask.sum() == 0:
        continue

    if output_data[mask].sum() != 0:
        raise RuntimeError(f"Collision detected for bilateral ROI {base}")

    output_data[mask] = new_rid
    bilat_info_rows.append({
        "region_id": new_rid,
        "region_name": base,
        "color": color,
        "order": order
    })

nib.save(nib.Nifti1Image(output_data, atlas_img.affine, atlas_img.header),
         os.path.join(out_dir, "glasser_regions_bilateral.nii"))

pd.DataFrame(bilat_info_rows).to_csv(
    os.path.join(out_dir, "region_info.tsv"),
    sep="\t", index=False
)

# ----------------------------------------------------------------------------
# Output 5: Bilateral Cortex ROI Volume (voxel = Cortex_ID)
# ----------------------------------------------------------------------------

df_merged["Cortex_ID"] = df_merged["Cortex_ID"].astype(int)
output_data = np.zeros_like(atlas_data, dtype=np.int32)

out_dir = "results/glasser_cortex_bilateral"
os.makedirs(out_dir, exist_ok=True)

cortex_info_rows = []
for cortex_id, subdf in df_merged.groupby("Cortex_ID"):
    rids = subdf["regionID"].unique()
    mask = np.isin(atlas_data, rids)
    if mask.sum() == 0:
        continue

    if output_data[mask].sum() != 0:
        raise RuntimeError(f"Collision detected for Cortex_ID {cortex_id}")

    color = subdf["color"].values[0]
    order = subdf["Cortex_ID"].values[0]

    output_data[mask] = cortex_id
    cortex_name = subdf["cortex"].unique()[0]
    cortex_info_rows.append({
        "region_id": cortex_id,
        "region_name": cortex_name,
        "color": color
    })

nib.save(nib.Nifti1Image(output_data, atlas_img.affine, atlas_img.header),
         os.path.join(out_dir, "glasser_cortex_bilateral.nii"))

pd.DataFrame(cortex_info_rows).to_csv(
    os.path.join(out_dir, "region_info.tsv"),
    sep="\t", index=False
)

print("Finished generating all ROI volumes and info tables.")
