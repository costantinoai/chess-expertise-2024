#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:29:46 2024

@author: costantino_ai
"""

import pydicom
import numpy as np

def calculate_slice_timing(tr, slices, multiband_factor):
    """
    Calculate the slice timing order for fMRI data according to a specific interleaved pattern.
    In this pattern, acquisition starts from both ends of the slice stack towards the middle,
    with the first and the middle slice starting at time 0, and so on in pairs.

    :param tr: Repetition time (TR) in seconds.
    :param slices: Total number of slices.
    :param multiband_factor: Multiband acceleration factor (how many slices per time bin).
    :return: A list of slice timings in seconds, compatible with BIDS format.
    """
    # Time per slice is the TR divided by the number of slice groups (half the number of slices)
    time_per_slice = tr / (len(slices) / multiband_factor)
    # Initialize slice timings array with zeros
    slice_timings = np.zeros(len(slices))
    
    for i in range(len(slices) // multiband_factor):
        # Time for the current pair of slices
        current_time = i * time_per_slice
        # Set timing for a pair: one from the start and one from the middle
        slice_timings[i] = current_time
        slice_timings[i + len(slices) // multiband_factor] = current_time

    return np.array(slice_timings)

def extract_dicom_info(f):
    """
    Extracts TR and the total number of slices from a DICOM file.

    :param f: Path to the DICOM file.
    :return: Tuple containing the TR in seconds and the total number of slices.
    """
    d = pydicom.read_file(f, defer_size='1KB', stop_before_pixels=True)
    tr = float(d.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RepetitionTime) / 1000
    all_slices = d.PerFrameFunctionalGroupsSequence
    slices = [s for s in all_slices
              if s.FrameContentSequence[0].TemporalPositionIndex == 1]
    return tr, slices

# Set filename to the DICOM image file
dicom_fname = "/data/projects/chess/data/sourcedata/sub-34/dicom/DICOM/IM_0001"

# Get relevant info
tr, slices = extract_dicom_info(dicom_fname)
multiband_factor = 2

# Get the slice timing
# NOTE: this assumes an ascending order (FH), where the first time bin takes
#       the first slice and the slice at half. For instance, if you have 60 
#       slices, at time 0 you get slice 1 and 31, at time 1 slice 2-32, etc.  
#       For single-band, see https://neurostars.org/t/how-dcm2niix-handles-different-imaging-types/22697/4
slice_timings = calculate_slice_timing(tr, slices, multiband_factor)
print(slice_timings)
