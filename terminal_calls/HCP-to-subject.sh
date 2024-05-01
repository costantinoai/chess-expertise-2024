# This script processes neuroimaging data with capabilities for environment configuration, data preprocessing,
# annotation conversion, volume transformation, and statistical analysis. Specifically, it:
# - Configures multithreading environment variables.
# - Ensures necessary annotation files are present, downloading missing ones.
# - Optionally accepts a predefined subject list or generates one from FASTSURFER_DIR.
# - Converts FreeSurfer annotations to labels, transforms these into volumes, and maps volumes to T1-weighted and MNI spaces.
# - Optionally generates anatomical statistics for each region.

# Usage:
#   Execute in a Unix-like environment with necessary dependencies (e.g., FreeSurfer, ANTs) installed.
#   Ensure correct paths to FASTSURFER_DIR, ANTS_PATH, FMRIPREP_DIR, and OUTPUT_BASE_DIR.
#   Run: ./this_script_name.sh [options]
#   Example: ./this_script_name.sh -n 12 -f /path/to/fastsurfer -a /path/to/ants -m /path/to/fmriprep -o /path/to/output
#
#   Options:
#     -s <subjects>: Specify subject numbers to process (space-separated if multiple), e.g., -s "1 2 3".
#     -n <nprocs>: Set the number of processors for parallel computation. Defaults to 15 or maximum available minus one.
#                  e.g., -n 8. If not specified or exceeds available, uses the system max minus one.
#     -f <FASTSURFER_DIR>: Set the FastSurfer directory path. Default: /data/projects/chess/data/BIDS/derivatives/fastsurfer
#     -a <ANTS_PATH>: Set the path to ANTs binaries. Default: /home/eik-tb/Documents/misctools/ants-2.5.1/bin
#     -m <FMRIPREP_DIR>: Set the path to fMRIPrep directory. Default: /data/projects/chess/data/BIDS/derivatives/fmriprep
#     -o <OUTPUT_BASE_DIR>: Set the base output directory. Default: /data/projects/chess/data/BIDS/derivatives/rois-HCP

# Environmental Variables:
#   ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS and OMP_NUM_THREADS control the multithreading for ITK and OpenMP processes, respectively.

# Directories:
#   FASTSURFER_DIR: Base directory for FastSurfer outputs.
#   SUBJECTS_DIR: Subject-specific processing directory, usually same as FASTSURFER_DIR.
#   ANTS_PATH: Path to ANTs binaries.
#   FMRIPREP_DIR: fMRIPrep outputs directory.
#   OUTPUT_BASE_DIR: Directory for storing processed outputs.

# Processing Steps:
#   1. Configure environment for parallel processing.
#   2. Check for and download missing annotation files.
#   3. Use a predefined subject list or generate one.
#   4. Process each subject by converting annotations to labels, transforming labels to volumes, and mapping these volumes to T1w and MNI spaces.
#   5. Optionally, compute anatomical statistics.
#   6. Cleanup temporary files.

# Dependencies:
#   - FreeSurfer for neuroimaging analysis.
#   - ANTs for image registration and transformation.
#   - wget for downloading internet files.
#   - Unix utilities (echo, mkdir, ls, xargs, basename, rm).

# Note:
#   Assumes FreeSurfer and ANTs are accessible in the system path or specified directories, and a BIDS-like structure for input data.

# Author: costantino_ai
# Created: 11/04/2024

# Exit immediately if a command exits with a non-zero status to ensure script halts on error.
set -e

# Parse command-line options for specifying paths and parallel computation resources.
# Default values for project structure, tools paths, and number of processors.
FASTSURFER_DIR_DEFAULT="/data/projects/chess/data/BIDS/derivatives/fastsurfer"
ANTS_PATH_DEFAULT="/home/eik-tb/Documents/misctools/ants-2.5.1/bin"
FMRIPREP_DIR_DEFAULT="/data/projects/chess/data/BIDS/derivatives/fmriprep"
OUTPUT_BASE_DIR_DEFAULT="/data/projects/chess/data/BIDS/derivatives/rois-HCP"
DEFAULT_NPROCS=15  # Default number of processors

# Use getopts to capture new flags for paths and --nprocs for parallel computation.
while getopts ":s:n:f:a:m:o:" opt; do
  case ${opt} in
    s )
      s_flag=1
      IFS=' ' read -r -a subject_nums <<< "${OPTARG}"
      ;;
    n )
      nprocs="${OPTARG}"
      ;;
    f )
      FASTSURFER_DIR="${OPTARG}"
      ;;
    a )
      ANTS_PATH="${OPTARG}"
      ;;
    m )
      FMRIPREP_DIR="${OPTARG}"
      ;;
    o )
      OUTPUT_BASE_DIR="${OPTARG}"
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Set default paths if not provided by the user.
FASTSURFER_DIR="${FASTSURFER_DIR:-$FASTSURFER_DIR_DEFAULT}"
ANTS_PATH="${ANTS_PATH:-$ANTS_PATH_DEFAULT}"
FMRIPREP_DIR="${FMRIPREP_DIR:-$FMRIPREP_DIR_DEFAULT}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$OUTPUT_BASE_DIR_DEFAULT}"

# Determine the number of processors to use.
nprocs="${nprocs:-$DEFAULT_NPROCS}"

# Ensure the number of processors does not exceed the system's capabilities.
max_procs=$(nproc)
if [ "${nprocs}" -gt "${max_procs}" ]; then
    echo "Requested number of processors (${nprocs}) exceeds the system's capabilities (${max_procs}). Using maximum available (${max_procs})."
    nprocs=${max_procs}
fi

# Set parallel computation resources.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${nprocs}  # Set ITK global thread limit.
export OMP_NUM_THREADS=${nprocs}  # Set OpenMP thread limit.

# Set sub-paths
FS_AVERAGE_DIR="${FASTSURFER_DIR}/fsaverage"  # Directory for FreeSurfer average data.
SUBJECTS_DIR="${FASTSURFER_DIR}"  # Subjects directory, here same as FastSurfer dir.

# Ensure annotation files are available, download if missing.
ANNOT_DIR="${FS_AVERAGE_DIR}/label"  # Directory for annotation files.
mkdir -p "${ANNOT_DIR}"  # Create annotation directory if it doesn't exist.
# Download left hemisphere annotation if missing.
if [ ! -f "${ANNOT_DIR}/lh.HCPMMP1.annot" ]; then
    echo "Left hemisphere annotation file not found. Downloading..."
    wget -O "${ANNOT_DIR}/lh.HCPMMP1.annot" https://figshare.com/ndownloader/files/5528816
fi
# Download right hemisphere annotation if missing.
if [ ! -f "${ANNOT_DIR}/rh.HCPMMP1.annot" ]; then
    echo "Right hemisphere annotation file not found. Downloading..."
    wget -O "${ANNOT_DIR}/rh.HCPMMP1.annot" https://figshare.com/ndownloader/files/5528819
fi

# Generate a list of subjects if not manually provided.
if [ -z "${L}" ]; then
    echo "Generating subject list from ${FASTSURFER_DIR}"
    L=$(mktemp)  # Create a temporary file for the list.
	if [ "${s_flag}" ]; then
	  for num in "${subject_nums[@]}"; do
		printf "sub-%s\n" "$num" >> "${L}"  # Corrected format string to handle any number format
	  done
	else
	  ls ${FASTSURFER_DIR}/sub-* -d | xargs -n 1 basename > "${L}"  # List all subjects and save to temp file.
	fi
fi

# Ensure output directory exists.
mkdir -p "${OUTPUT_BASE_DIR}"

# Process each subject individually.
while IFS= read -r subject; do
    # Define paths specific to the current subject.
    SUBJECT_DIR="${FS_AVERAGE_DIR}/${subject}"  # Directory for subject-specific data in fsaverage.
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${subject}"  # Output directory for the subject.
    mkdir -p "${OUTPUT_DIR}"  # Ensure subject's output directory exists.

    echo "Processing ${subject}..."

    # Convert annotations to labels and prepare color lookup tables for each hemisphere.
    for hemi in lh rh; do
        echo "Converting ${hemi}.HCPMMP1 annotation to label for ${subject}..."
        mri_annotation2label --subject fsaverage --hemi ${hemi} --annotation HCPMMP1 --outdir "${OUTPUT_DIR}/label" --ctab "${OUTPUT_DIR}/label/${hemi}_HCPMMP1_color_table.txt"

        # Transform labels from fsaverage to subject space.
        mkdir -p "${OUTPUT_DIR}/label"
        for label in "${OUTPUT_DIR}/label/${hemi}"*.label; do
            label_basename=$(basename "$label")
            mri_label2label --srcsubject fsaverage --srclabel "$label" --trgsubject "${subject}" --trglabel "${OUTPUT_DIR}/label/${label_basename}" --regmethod surface --hemi ${hemi}
        done
    done

    # Combine labels into a single annotation file per hemisphere for the subject.
    for hemi in lh rh; do
        echo "Creating annotation file from labels for ${hemi} hemisphere of ${subject}..."
        mris_label2annot --s ${subject} --h ${hemi} --ctab ${OUTPUT_DIR}/label/${hemi}_HCPMMP1_color_table.txt --a ${subject}_HCPMMP1 --ldir "${OUTPUT_DIR}/label" --noverbose --no-unknown
    done

    # Convert cortical parcellations to a volume in native space.
    echo "Converting annotations to volume for ${subject}..."
    mri_aparc2aseg --s "${subject}" --annot "${subject}_HCPMMP1" --o "${OUTPUT_DIR}/${subject}_HCPMMP1_volume_fsnative.nii"
    
    # Transform the volume to T1-weighted space.
    echo "Converting volume to T1w space for ${subject}..."
    TRANSFORMATION_FILE="${FMRIPREP_DIR}/${subject}/anat/${subject}_from-fsnative_to-T1w_mode-image_xfm.txt"
    REF_FILE="${FMRIPREP_DIR}/${subject}/anat/${subject}_desc-preproc_T1w.nii.gz"
    ${ANTS_PATH}/antsApplyTransforms -d 3 -i "${OUTPUT_DIR}/${subject}_HCPMMP1_volume_fsnative.nii" -r "${REF_FILE}" -t "${TRANSFORMATION_FILE}" -o "${OUTPUT_DIR}/${subject}_HCPMMP1_volume_T1w.nii" -n "NearestNeighbor"

    # Further transform the volume to standard MNI space.
    echo "Converting volume to MNI space for ${subject}..."
    TRANSFORMATION_FILE="${FMRIPREP_DIR}/${subject}/anat/${subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    REF_FILE="${FMRIPREP_DIR}/${subject}/anat/${subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"
    ${ANTS_PATH}/antsApplyTransforms -d 3 -i "${OUTPUT_DIR}/${subject}_HCPMMP1_volume_T1w.nii" -r "${REF_FILE}" -t "${TRANSFORMATION_FILE}" -o "${OUTPUT_DIR}/${subject}_HCPMMP1_volume_MNI.nii" -n "NearestNeighbor"
    
    # Optionally generate anatomical statistics for each region.
    echo "Generating anatomical stats for ${subject}..."
    mkdir -p "${OUTPUT_DIR}/stats"
    for hemi in lh rh; do
        mris_anatomical_stats -m ${FASTSURFER_DIR}/${subject}/surf/${hemi}.white -a ${FASTSURFER_DIR}/${subject}/label/${hemi}.${subject}_HCPMMP1.annot -f ${OUTPUT_DIR}/stats/${hemi}_HCPMMP1_stats.txt ${subject} ${hemi}
    done

    echo "Processing complete for ${subject}."

done < "${L}"  # Read from the list of subjects.

# Clean up temporary files if generated by this script.
if [ -z "${OPTARG}" ]; then
    rm "${L}"
fi

echo "All processing complete."

