#!/usr/bin/env python
# written by: Dr. Shaodong Ding 2024-05-30
# we add the -1024 padding during the inverse transformation of fake_ct to mr space (actually in the window loss version).
# mr2ct, folder, or file path

import os
from os.path import dirname, basename, join
import sys
import SimpleITK as sitk
import torch
import numpy as np

from utils.infer_funcs import do_mr_to_pct


input_mr_path = "./MAGUIXIA_0000.nii.gz"
out_pseudo_ct_path = "./valid_MAGUIXIA_0000_pseudo_ct_realistic_.nii.gz"

# transformer + L1 loss
# pretrained_model_path = "runs/mr2ct_supervise_transformer/no_weight_decay/model_best.pt"
# state_dict = torch.load(pretrained_model_path)['G']

# transformer + L1 loss + SSIM loss
pretrained_model_path = "runs/mr2ct_supervise_transformer_windowed_ssim_mseskull/2/model_best.pt"
state_dict = torch.load(pretrained_model_path)['G']

# pretrained_model_path = "runs/mr2ct_supervise_larger_patch/1/model_best.pt"
# state_dict = torch.load(pretrained_model_path)['G']

# do_mr_to_pct(
#     input_mr_file=input_mr_path, 
#     output_pct_file=out_pseudo_ct_path, 
#     saved_model=state_dict, 
#     device="cuda", prep_t1=False,
#     sliding_window_infer=False,
#     transformer_layers=3, 
#     img_size=(448, 448, 64)
# )

# ref_ct_arr = sitk.GetArrayFromImage(sitk.ReadImage("ct_MAGUIXIA_0000.nii.gz"))
# # pct_arr = sitk.GetArrayFromImage(sitk.ReadImage("valid_MAGUIXIA_0000_pseudo_ct_pad_minus1024_window_loss_transformer_headmask.nii.gz"))

# window_500_1500_mask = (ref_ct_arr > 0) & (ref_ct_arr < 100)
# error_map_img = sitk.GetImageFromArray(window_500_1500_mask.astype(np.float32))
# error_map_img.CopyInformation(sitk.ReadImage("ct_MAGUIXIA_0000.nii.gz"))
# sitk.WriteImage(error_map_img, "window_100_1500_mask.nii.gz")

# window_500_1500_error = abs(ref_ct_arr[window_500_1500_mask] - pct_arr[window_500_1500_mask])
# print("avg abs diff: ", abs(ref_ct_arr - pct_arr).mean(), "avg abs diff in window 500-1500: ", window_500_1500_error.mean())
# # save error map
# error_map = abs(ref_ct_arr - pct_arr)
# error_map_img = sitk.GetImageFromArray(window_500_1500_mask.astype(np.float32))
# error_map_img.CopyInformation(sitk.ReadImage("ct_LUFA_0000.nii.gz"))
# sitk.WriteImage(error_map_img, "train_LUFA_0000_pseudo_ct_pad_minus1024_window_loss_error_map.nii.gz")


json_path = "data/cross_validation/cross_validation_fold_0.json"
output_data_dir = "data/ct_reg2_mr_betneck/transformed_data"


import json
from shutil import copyfile
from tqdm import tqdm
with open(json_path) as f:
    data = json.load(f)

test_data = data['validation']

for case_idx, case in enumerate(tqdm(test_data)):
    mr_path = case['mr_image']
    ct_path = case['ct_image']
    mr_name = basename(mr_path).split(".nii")[0]
    ct_name = basename(ct_path).split(".nii")[0]

    out_dir = join(output_data_dir, str(case_idx))
    os.makedirs(out_dir, exist_ok=True)

    copyfile(mr_path, join(out_dir, f"mr.nii.gz"))
    copyfile(ct_path, join(out_dir, f"ct.nii.gz"))

    pct_path = join(out_dir, f"pseudo_ct.nii.gz")
    do_mr_to_pct(
        input_mr_file=mr_path, 
        output_pct_file=pct_path, 
        saved_model=state_dict, 
        device="cuda", prep_t1=False,
        sliding_window_infer=False,
        transformer_layers=3, 
        img_size=(448, 448, 64)
    )