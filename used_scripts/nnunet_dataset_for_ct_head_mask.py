# construct nnunet dataset

import os
import numpy as np
import json
from glob import glob
import SimpleITK as sitk
from shutil import copyfile


ct_data_in_dir = "data/pipeline_niigz_betneck/ct"
# brain_mask_data_in_dir = "brain_mask/labelsTr"
brain_mask_data_in_dir = "data/original_ct_betneck/labelsTr"

out_dir = "brain_mask"

brain_masks = list(glob(os.path.join(brain_mask_data_in_dir, "*.nii.gz")))


# for brain_mask in brain_masks:
#     ct_file = os.path.join(ct_data_in_dir, os.path.basename(brain_mask).replace("_brainmask.nii.gz", ".nii.gz"))
#     print(ct_file)

#     pid = os.path.basename(ct_file).split(".nii")[0][:-5]

#     if not os.path.exists(ct_file):
#         print(f"CT file not found: {ct_file}")
#         continue

#     out_ct_file = os.path.join(out_dir, "imagesTr", os.path.basename(ct_file))
#     copyfile(ct_file, out_ct_file)

#     # we transform the value in brainmask from 255 to 1
#     brain_mask_img = sitk.ReadImage(brain_mask)
#     brain_mask_arr = sitk.GetArrayFromImage(brain_mask_img).astype(np.bool_)
#     print(brain_mask_arr.sum(), brain_mask_arr.shape)
#     bg_bmarr = np.zeros(brain_mask_arr.shape)
#     bg_bmarr[brain_mask_arr] = 1
#     print(bg_bmarr.sum())
#     new_brain_mask_img = sitk.GetImageFromArray(bg_bmarr)
#     new_brain_mask_img.SetDirection(brain_mask_img.GetDirection())
#     new_brain_mask_img.SetOrigin(brain_mask_img.GetOrigin())
#     new_brain_mask_img.SetSpacing(brain_mask_img.GetSpacing())

#     out_brain_mask_file = os.path.join(out_dir, "new_labelsTr", pid + ".nii.gz")
#     sitk.WriteImage(new_brain_mask_img, out_brain_mask_file)

#     print(f"CT file: {ct_file} -> {out_ct_file}")
#     print(f"Brain mask file: {brain_mask} -> {out_brain_mask_file}")


ct_files = list(glob(os.path.join(ct_data_in_dir, "*.nii.gz")))

out_dir = "data/ct_reg2_mr_betneck/original_ct_betneck"
for ct_file in ct_files:
    ct_itk = sitk.ReadImage(ct_file)
    ct_arr = sitk.GetArrayFromImage(ct_itk)

    if ct_arr.max() < 0:
        continue

    out_ct_file = os.path.join(out_dir, os.path.basename(ct_file))

    copyfile(ct_file, out_ct_file)