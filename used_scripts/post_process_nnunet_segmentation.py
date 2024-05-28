import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, find_objects
from scipy import ndimage
from glob import glob
import os
from os.path import join, dirname, basename


def extract_largest_connected_component(mask):
    # Label connected components in the mask
    labeled_mask, num_labels = label(mask)

    # Find the sizes of all connected components
    component_sizes = np.bincount(labeled_mask.ravel())

    # Find the label of the largest connected component
    largest_component_label = np.argmax(component_sizes[1:]) + 1

    # Extract the largest connected component
    largest_component_mask = np.zeros_like(mask)
    largest_component_mask[labeled_mask == largest_component_label] = 1

    # Fill holes in the largest connected component
    filled_mask = ndimage.binary_fill_holes(largest_component_mask)
    largest_component_mask = filled_mask

    return largest_component_mask

# Example usage

mask_data_dir = "data/brain_mask"
mask_files = list(glob(mask_data_dir, "*.nii.gz"))

out_mask_data_dir = "data/largest_connected_component"
for mask_file in mask_files:
    mask_itk = sitk.ReadImage(mask_file)
    mask_arr = sitk.GetArrayFromImage(mask_itk)

    new_mask_arr = np.zeros_like(mask_arr)
    for i, mask_slice in enumerate(mask_arr):
        new_mask_arr[i] = extract_largest_connected_component(mask_slice)

    new_mask_itk = sitk.GetImageFromArray(new_mask_arr)
    new_mask_itk.CopyInformation(mask_itk)
    out_new_mask_path = join(out_mask_data_dir, basename(mask_file))
    os.makedirs(dirname(out_new_mask_path), exist_ok=True)
    sitk.WriteImage(new_mask_itk, out_new_mask_path)

# data = "LUFA.nii.gz"
# itk_img = sitk.ReadImage(data)
# mask = sitk.GetArrayFromImage(itk_img)

# new_mask = np.zeros_like(mask)
# for i, mask_slice in enumerate(mask):
#     new_mask[i] = extract_largest_connected_component(mask_slice)

# new_itk_img = sitk.GetImageFromArray(new_mask)
# new_itk_img.CopyInformation(itk_img)
# sitk.WriteImage(new_itk_img, "largest_connected_component.nii.gz")