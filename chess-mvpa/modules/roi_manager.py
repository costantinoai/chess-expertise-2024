import re
import csv
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any, Optional
from pprint import pprint
from natsort import natsorted
from dataclasses import dataclass, field

from modules import ROIS_CSV, LEFT_LUT, RIGHT_LUT, FS_PATH, LH_ANNOT, RH_ANNOT, logging

# We have 7 color codes:
#   1) #a6cee3
#   2) #1f78b4
#   3) #b2df8a
#   4) #33a02c
#   5) #fb9a99
#   6) #e31a1c
#   7) #fdbf6f
#
# The 22 regions are grouped as follows:
#   Group 1 (Early Visual):     IDs 1-2
#   Group 2 (Intermediate Vis): IDs 3-5
#   Group 3 (Sensorimotor):     IDs 6-9
#   Group 4 (Auditory):         IDs 10-12
#   Group 5 (Temporal):         IDs 13-14
#   Group 6 (Posterior):        IDs 15-18
#   Group 7 (Anterior):         IDs 19-22


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



@dataclass
class ROI:
    """
    Represents a single Region of Interest (ROI) using Python's dataclass.

    Attributes:
        region_name (str): Short name of the region (e.g., 'L_V1_ROI').
        region_long_name (str): Long descriptive name of the region (underscores removed).
        hemisphere (str): Hemisphere ('L' or 'R').
        region (str): Original region label from the CSV (e.g., 'V1').
        lobe (str): Lobe classification (e.g., 'occipital').
        cortex (str): Cortex classification (e.g., 'primary visual').
        cortex_id (int): Numeric ID associated with the cortex (from CSV).
        region_id (Optional[int]): Numeric ID associated with the region (from the color table);
            can be None initially if not yet assigned.
        x_cog (float): X-coordinate of the center of gravity.
        y_cog (float): Y-coordinate of the center of gravity.
        z_cog (float): Z-coordinate of the center of gravity.
        volmm (int): Volume of the region in mm^3.
        additional_attributes (Dict[str, Any]): A dictionary for storing extra custom info
            that might be added at runtime.
    """
    region_name: str
    region_long_name: str
    hemisphere: str
    region: str
    lobe: str
    cortex: str
    cortex_color: str
    cortex_group: str
    cortex_id: int
    region_id: Optional[int]  # Set to None initially if we haven't assigned yet
    x_cog: float
    y_cog: float
    z_cog: float
    volmm: int

    region_order: Optional[int] = None
    cortex_order: Optional[int] = None
    lobe_order: Optional[int] = None

    # A dictionary for arbitrary additional data. Initialized to empty by default.
    additional_attributes: Dict[str, Any] = field(default_factory=dict)

    def add_attribute(self, key: str, value: Any) -> None:
        """
        Add (or update) a key-value pair to the ROI's additional attributes.

        Args:
            key (str): The attribute name to add.
            value (Any): The value associated with the attribute.
        """
        self.additional_attributes[key] = value

    def get_details(self) -> Dict[str, Any]:
        """
        Return all details of the ROI as a dictionary, including any additional attributes.
        """
        # Core attributes
        base_data = {
            "Hemisphere": self.hemisphere,
            "Lobe": self.lobe,
            "Cortex": self.cortex,
            "Cortex Color": self.cortex_color,
            "Cortex Group": self.cortex_group,
            "Cortex ID": self.cortex_id,
            "Region": self.region,
            "Region Name": self.region_name,
            "Region Long Name": self.region_long_name,
            "Region ID": self.region_id,
            "X-CoG": self.x_cog,
            "Y-CoG": self.y_cog,
            "Z-CoG": self.z_cog,
            "Volume (mm^3)": self.volmm,

        }
        # Merge additional attributes into the base data
        base_data.update(self.additional_attributes)
        return base_data


class ROIManager:
    """
    Manages a collection of ROI objects, providing methods for
    loading data from CSV/color tables, interrogating attributes,
    and hierarchical navigation of the ROI set.

    Attributes:
        rois (List[ROI]): A list holding all ROI objects managed here.
        region_mapping (Dict[int, str]): A mapping of region IDs to region names
            as defined in the color table files.
    """

    def __init__(
        self, csv_path: str, left_color_table_path: str, right_color_table_path: str
    ):
        """
        Initialize the ROIManager and load data from the provided files.

        The manager requires:
            1) A CSV file specifying ROI information (names, coordinates, volumes, etc.).
            2) Two color table files (left and right hemisphere) that map
               region IDs to region names.

        Args:
            csv_path (str): Path to the CSV file containing ROI information.
            left_color_table_path (str): Path to the left hemisphere color table file.
            right_color_table_path (str): Path to the right hemisphere color table file.
        """
        # Initialize an empty list to hold ROI objects.
        self.rois: List[ROI] = []

        # Initialize a dictionary to map region IDs to region names.
        self.region_mapping: Dict[int, str] = {}

        # Load ROI base data from the CSV file.
        self.load_from_csv(csv_path)

        # Load region mappings (region_id -> region_name) from the
        # left and right hemisphere color table files.
        self.load_from_color_table(left_color_table_path, hemisphere="L")
        self.load_from_color_table(right_color_table_path, hemisphere="R")

        self.rois = self._order_by_region()

        self.MNINLinAsym_AFNI_img = None

    def _order_by_region(self):
        rois = sorted(self.rois, key=lambda x: x.region_id)
        return rois

    def _order_by_cortex(self):
        rois = sorted(self.rois, key=lambda x: x.cortex_id)
        return rois

    def get_ordered_cortices_names(self):

        rois = self._order_by_cortex()
        cortices_names = [0] + [roi.cortex for i, roi in enumerate(rois) if rois[i-1].cortex != roi.cortex]
        cortices_names = tuple(cortices_names[1:])

        return cortices_names


    def load_from_csv(self, csv_path: str):
        """
        Load ROIs from a CSV file.

        Reads each row from the given CSV file and creates an ROI object,
        then adds it to the manager's internal list.

        Args:
            csv_path (str): Path to the CSV file containing ROI info.
        """
        # Open the CSV file in read mode.
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            # Iterate over each row (dict) in the CSV.
            for row in reader:
                # The hemisphere (L or R) as read from the CSV.
                hemisphere = row["LR"]

                # Format region name as {hemisphere}_{region}_ROI.
                formatted_name = f"{hemisphere}_{row['region']}_ROI"

                # Remove underscores in the region long name for readability.
                region_long_name = row["regionLongName"].replace("_", " ")

                # Remove underscores in the cortex name for readability.
                cortex = row["cortex"].replace("_", " ")

                # Create an ROI object with placeholders for region_id (None for now).
                roi = ROI(
                    formatted_name,
                    region_long_name,
                    hemisphere,
                    row["region"],
                    row["Lobe"],
                    cortex,
                    cortex_info_map[int(row["Cortex_ID"])]["color"],
                    cortex_info_map[int(row["Cortex_ID"])]["group"],
                    int(row["Cortex_ID"]),
                    None,  # region_id will be assigned when loading color tables
                    float(row["x-cog"]),
                    float(row["y-cog"]),
                    float(row["z-cog"]),
                    int(row["volmm"]),
                )

                # Add the ROI object to the manager's list.
                self.add_roi(roi)

    def load_from_color_table(self, color_table_path: str, hemisphere: str):
        """
        Load region mappings (ID -> name) from a FreeSurfer-style color table file.

        This file typically has lines in the format:
            region_id region_name R G B A
        Comments or lines starting with '#' are skipped.
        We store the IDs and names in the manager's 'region_mapping' attribute.

        Args:
            color_table_path (str): Path to the hemisphere color table file.
            hemisphere (str): Hemisphere label, 'L' or 'R'.
        """
        # Open the color table file in read mode.
        with open(color_table_path, "r") as file:
            for line in file:
                # Ignore empty or comment lines (starting with '#').
                if line.strip() and not line.startswith("#"):
                    # Split the line by whitespace.
                    parts = line.split()

                    # Check we have enough parts to extract region_id and region_name.
                    if len(parts) >= 2:
                        region_id = int(parts[0])
                        region_name = parts[1]

                        # Skip the default/placeholder region with ID 0 or name '???'.
                        if region_id == 0 or region_name == "???":
                            continue

                        # Add to the region mapping dictionary.
                        self.region_mapping[region_id] = region_name

        # After loading, assign region IDs to the ROI objects for this hemisphere.
        self.assign_region_ids(hemisphere)

    def assign_region_ids(self, hemisphere: str) -> None:
        """
        Assign region IDs to ROIs for a specific hemisphere by matching ROI names
        (e.g., 'L_V1_ROI') to color-table names (e.g., 'lh_V1_ROI').

        This method includes a performance optimization using a name->ID dictionary
        for constant-time lookups instead of nested loops.

        Args:
            hemisphere (str): 'L' or 'R' to specify which ROIs to assign.
        """
        hemi_lower = hemisphere.lower()

        # First, filter the mapping to only those that match the hemisphere prefix, e.g. "lh_" or "rh_"
        # Then invert to { region_name.lower(): region_id } for quick lookups
        name_to_id = {}
        for rid, name in self.region_mapping.items():
            if name.lower().startswith(f"{hemi_lower}_"):
                name_to_id[name.lower()] = rid

        # Assign region_id to matching ROIs
        for roi in self.rois:
            if roi.hemisphere.lower() == hemi_lower:
                # ROI region_name might be 'L_V1_ROI'; color table might be 'lh_V1_ROI'
                # We'll ensure both are lowercase to compare
                roi_key = roi.region_name.lower()
                # If there's a match, set the region_id directly
                if roi_key in name_to_id:
                    roi.region_id = name_to_id[roi_key]

    def add_roi(self, roi: ROI):
        """
        Add an ROI object to the manager's list of ROIs.

        Args:
            roi (ROI): The ROI instance to add.
        """
        # Simply append the ROI to our internal list.
        self.rois.append(roi)

    def list_unique_values(self, key: str) -> List[Any]:
        """
        List all unique values for a specific attribute across all ROIs.

        Args:
            key (str): The name of the attribute to retrieve.

        Returns:
            List[Any]: A sorted list of unique values for that attribute.
        """
        # Use a set comprehension to collect unique values for the given attribute.
        # getattr(roi, key.lower(), None) attempts to get the attribute 'key'
        # (converted to lowercase for convenience). If not found, returns None.
        return natsorted(list({getattr(roi, key.lower(), None) for roi in self.rois}))

    def list_keys(self):
        """
        Retrieve the set of keys (attribute names) available in any ROI.

        This method uses the first ROI in the list to determine the
        available keys by calling its get_details method.

        Returns:
            List[str]: A list of attribute names/keys.
        """
        roi = self.rois[0]  # Assuming we have at least one ROI loaded.
        keys = list(roi.get_details().keys())
        return keys

    def _make_valid_name(self, name: str) -> str:
        """
        Cleans and standardizes an ROI name by:
        - Removing hemisphere prefixes ('R_' or 'L_').
        - Removing specific suffixes ('_ROI', '_cortex', '_lobe').
        - Converting to lowercase.
        - Removing non-alphanumeric characters.

        Args:
            name (str): The input name to standardize.

        Returns:
            str: The cleaned, standardized name.
        """
        # Remove hemisphere prefix if present
        if name[:2].lower() in ["r_", "l_"]:
            name = name[2:]

        # Remove specific suffixes
        for suffix in ["_ROI", "_cortex", "_lobe"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]  # Remove the suffix

        # Convert to lowercase
        lowered = name.lower()

        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^a-z0-9]', '', lowered)

        return cleaned

    def get_by_filter(
        self,
        hemisphere: Optional[str] = None,
        lobe: Optional[str] = None,
        cortex: Optional[str] = None,
        region: Optional[str] = None,
        cortex_id: Optional[int] = None,
        region_id: Optional[int] = None,
        region_name: Optional[str] = None,
        return_mask: Optional = False
    ) -> List[ROI]:
        """
        Retrieve ROIs based on multiple possible hierarchical filters.

        If multiple filters are provided, they are combined with a logical AND
        (i.e., an ROI must match all provided filters to be included).

        Args:
            hemisphere (Optional[str]): Filter by hemisphere ('L' or 'R').
            lobe (Optional[str]): Filter by lobe (substring match, case-insensitive).
            cortex (Optional[str]): Filter by cortex classification (substring match, case-insensitive).
            region (Optional[str]): Filter by region name (substring match, case-insensitive).
            cortex_id (Optional[int]): Filter by exact cortex ID.
            region_id (Optional[int]): Filter by exact region ID.
            region_name (Optional[str]): Filter by exact region name (case-insensitive).

        Returns:
            List[ROI]: A list of matching ROI objects, sorted by their region_id.
        """
        # Start with all ROIs.
        filtered_rois = self.rois

        # Filter by lobe if provided, using make_valid_name for case-insensitive comparison.
        if lobe:
            filtered_rois = [
                roi for roi in filtered_rois
                if self._make_valid_name(lobe) in self._make_valid_name(roi.lobe)
            ]

        # Filter by cortex if provided, using make_valid_name for case-insensitive comparison.
        if cortex:
            filtered_rois = [
                roi for roi in filtered_rois
                if self._make_valid_name(cortex) in self._make_valid_name(roi.cortex)
            ]

        # Filter by region if provided, using make_valid_name for case-insensitive comparison.
        if region:
            filtered_rois = [
                roi for roi in filtered_rois
                if self._make_valid_name(region) in self._make_valid_name(roi.region)
            ]

        # Filter by region_name if provided, using make_valid_name for case-insensitive comparison.
        if region_name:
            filtered_rois = [
                roi for roi in filtered_rois
                if self._make_valid_name(region_name) == self._make_valid_name(roi.region_name)
            ]

        # Filter by cortex_id if provided (exact integer match).
        if cortex_id is not None:
            if cortex_id >= 1000:
                if hemisphere is not None:
                    logging.warning(
                        "ROIManager.get_by_filter: a 'hemisphere' argument was provided when"
                        " 'cortex_id' > 1000. A label > 1000 implies the hemisphere (1000 left"
                        " h, 2000 rh), so any input argument for hemisphere will be ignored.")

                if cortex_id >= 2000:
                    hemisphere="r"
                    cortex_id -=2000
                else:
                    hemisphere="l"
                    cortex_id -=1000

            filtered_rois = [roi for roi in filtered_rois if roi.cortex_id == cortex_id]

        # Filter by region_id if provided (exact integer match).
        if region_id is not None:
            if region_id >= 1000:
                if hemisphere is not None:
                    logging.warning(
                        "ROIManager.get_by_filter: a 'hemisphere' argument was provided when"
                        " 'region_id' > 1000. A label > 1000 implies the hemisphere (1000 left"
                        " h, 2000 rh), so any input argument for hemisphere will be ignored.")

                if region_id >= 2000:
                    hemisphere="r"
                    region_id -=2000
                else:
                    hemisphere="l"
                    region_id -=1000

            filtered_rois = [roi for roi in filtered_rois if roi.region_id == region_id]


        # Filter by hemisphere if provided, using make_valid_name.
        if hemisphere:
            filtered_rois = [
                roi for roi in filtered_rois
                if self._make_valid_name(roi.hemisphere) == self._make_valid_name(hemisphere)
            ]


        # Finally, sort the results by region_id before returning.
        sorted_rois = sorted(filtered_rois, key=lambda x: x.region_id)

        if return_mask:
            mask = self._get_AFNI_mask_from_ROIS(sorted_rois)
        else:
            mask = None

        return sorted_rois, mask


    def _group_by(self, attribute):
        """
        Groups ROIs by a given attribute.

        Args:
            attribute (str): Attribute to group by (e.g., 'cortex', 'lobe').

        Returns:
            Dict[str, List[ROI]]: Mapping from attribute value to list of ROIs.
        """
        groups = OrderedDict()
        for roi in self.rois:
            key = getattr(roi, attribute)
            if key not in groups:
                groups[key] = []
            groups[key].append(roi)
        return groups

    def get_MNINLinAsym_AFNI(self, level="region", symmetrical_labels=False):
        """
        Process and remap the AFNI atlas into a new format.

        Parameters:
        - level (str): "region" (default) or "cortex". Determines the level of transformation.
        - symmetrical_labels (bool): If True, left and right hemisphere labels are unified.

        Returns:
        - nib.Nifti1Image: The transformed atlas image.
        """
        logging.info("Loading AFNI atlas image")
        from nilearn import image

        if self.MNINLinAsym_AFNI_img is None:
            atlas = self._load_MNINLinAsym_AFNI_img()
            logging.debug("AFNI atlas loaded from source")
        else:
            atlas = self.MNINLinAsym_AFNI_img
            logging.debug("Using cached AFNI atlas image")

        atlas_data = atlas.get_fdata()
        remapped_atlas = atlas_data.copy()

        # The AFNI atlas uses left hemisphere (LH) labels 1-180 and right hemisphere (RH) labels 1001-1180.
        # We shift them to: LH → 1000+ and RH → 2000+
        logging.info("Remapping region labels")
        remapped_atlas[remapped_atlas != 0] += 1000

        if symmetrical_labels:
            logging.info("Applying symmetrical label transformation")
            remapped_atlas[remapped_atlas >= 2000] -= 1000  # Shift RH to match LH labels
            remapped_atlas[remapped_atlas != 0] -= 1000      # Shift all labels back to original range

        # Initialize output atlas data
        if level == "region":
            logging.debug("Returning region-level atlas")
            new_atlas_data = remapped_atlas

        elif level == "cortex":
            logging.info("Processing cortex-level mapping")
            cortex_atlas_data = np.zeros_like(remapped_atlas)

            for region_label in np.unique(remapped_atlas):
                if int(region_label) == 0:
                    continue  # Skip background voxels

                if not symmetrical_labels:
                    # Labels already distinguish LH and RH
                    rois_list, roi_mask = manager.get_by_filter(region_id=region_label)
                    if len(rois_list) != 1:
                        logging.warning(f"Unexpected number of ROIs ({len(rois_list)}) for region {region_label}")

                    roi_obj = rois_list[0]
                    selected_cortex_id = roi_obj.cortex_id
                    selected_hemi = roi_obj.hemisphere.lower()

                    # Ensure no conflicting assignments
                    assert np.sum(cortex_atlas_data[remapped_atlas == region_label]) == 0

                    # Assign hemisphere-based cortex ID
                    base = 1000 if selected_hemi.startswith("l") else 2000
                    cortex_atlas_data[remapped_atlas == region_label] = base + selected_cortex_id
                    logging.debug(f"Assigned cortex label {base + selected_cortex_id} to region {region_label}")

                else:  # Apply symmetrical labels
                    rois_list, roi_mask = self.get_by_filter(hemisphere="l", region_id=region_label)
                    if len(rois_list) != 1:
                        logging.error(f"Unexpected number of ROIs ({len(rois_list)}) for bilateral region {region_label}")

                    roi_obj = rois_list[0]
                    selected_cortex_id = roi_obj.cortex_id

                    # Ensure no conflicting assignments
                    assert np.sum(cortex_atlas_data[remapped_atlas == region_label]) == 0

                    # Assign the same cortex ID for both hemispheres
                    cortex_atlas_data[remapped_atlas == region_label] = selected_cortex_id
                    logging.debug(f"Assigned symmetrical cortex label {selected_cortex_id} to region {region_label}")

            new_atlas_data = cortex_atlas_data
        else:
            logging.error(f"Invalid level argument: {level}")
            raise ValueError("Invalid level argument. Choose 'region' or 'cortex'.")

        unique_labels = np.unique(new_atlas_data)
        logging.info(f"Number of unique labels in output atlas: {len(unique_labels)}")

        return image.new_img_like(atlas, new_atlas_data, atlas.affine, atlas.header)

    def _load_MNINLinAsym_AFNI_img(self, path=None):

        import nibabel as nib

        if path==None:
            path = "/data/projects/chess/data/misc/afni_glasser/MNI_Glasser_HCP_v1.0_resampled.nii"

        self.MNINLinAsym_AFNI_img = nib.load(path)
        logging.info(f"Atlas loaded successfully from {path}.")

        return self.MNINLinAsym_AFNI_img

    def _get_AFNI_mask_from_ROIS(self, rois : List[ROI]):

        if self.MNINLinAsym_AFNI_img == None:
            atlas = self._load_MNINLinAsym_AFNI_img()

        atlas_data = atlas.get_fdata()

        mask = np.zeros_like(atlas_data)

        for roi in rois:

            region_id = roi.region_id
            hemi = roi.hemisphere

            # Here we need to map the index used in this AFNI atlas
            # AFNI: lh --> 1-180; rh --> 1001-1180;
            if hemi[0].lower() == "r":
                region_id += 1000

            mask[mask==region_id] = 1

        return mask





# ========================================================================
# Below is example usage / tests for the ROIManager.
# ========================================================================
if __name__ == "__main__":

    print("Running some tests...")

    # Initialize the ROIManager with the paths.
    roi_manager = ROIManager(
        csv_path=ROIS_CSV,
        left_color_table_path=LEFT_LUT,
        right_color_table_path=RIGHT_LUT,
    )

    # Test 1: Retrieve keys we can interrogate (attribute names).
    keys = roi_manager.list_keys()

    print("\n\nAvailable keys:")
    pprint(keys)

    # Test 2: Get all possible values for a given key.
    # Select one of the keys we just retrieved, e.g., the third key in the list.
    selected_key = keys[2]
    key_values = roi_manager.list_unique_values(selected_key)
    print(f"\n\nUnique values for {selected_key}:")
    pprint(key_values)

    # Test 3: Filter the manager for a specific set of ROIs based on
    # hemisphere='R', lobe containing 'occ', cortex containing 'primary visual', region containing 'v1'.
    results, roi_mask = roi_manager.get_by_filter(
        hemisphere="R",
        lobe="occ",
        cortex="primary visual",
        region="v1",
        cortex_id=None,
        region_id=None,
    )

    # If results is not empty, take the first ROI and display its details.
    if results:
        result = results[0]
        print("\n\nROI details for the first match:")
        pprint(result.get_details())
    else:
        print("\n\nNo ROIs found with the specified filters.")

    # Test 4: Retrieve ROI by cortex ID (e.g., 1).
    cortex_id_results, roi_mask = roi_manager.get_by_filter(cortex_id=3)
    print(f"\n\nROIs with Cortex ID 1 ({len(cortex_id_results)} found):")
    for roi in cortex_id_results[:5]:  # Print the first 5 for brevity
        pprint(roi.get_details())

    # Test 5: Retrieve ROI by region ID (e.g., 1).
    region_id_results, roi_mask = roi_manager.get_by_filter(region_id=1)
    if region_id_results:
        print("\n\nDetails of ROI with Region ID 1:")
        for roi in region_id_results[:5]:  # Print the first 5 for brevity
            pprint(roi.get_details())
    else:
        print("\n\nNo ROI found with Region ID 1.")

    # Quick check to make sure every ROI is in the right place:
    from modules.surf_helpers import plot_roi_selection_overlay

    # Initialize ROIManager
    manager = ROIManager(
        csv_path=ROIS_CSV,
        left_color_table_path=LEFT_LUT,
        right_color_table_path=RIGHT_LUT,
    )

    cortices = manager.list_unique_values('cortex')

    for cortex in cortices:

        # Define filters for ROI selection
        filters = {
            "hemisphere": None,
            "lobe": None,
            "cortex": cortex,  # None means no filtering for this attribute
            "region": None
        }

        # Plot selected ROIs with region_id-based overlay
        plot_roi_selection_overlay(
            freesurfer_path=FS_PATH,
            lh_annotation=LH_ANNOT,
            rh_annotation=RH_ANNOT,
            manager=manager,
            out_path=None,  # Set to a Path if you want to save the figure
            show_plot=True,  # Displays the plot inline
            **filters
        )
