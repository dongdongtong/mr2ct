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
import nibabel as nib
from collections import defaultdict

from tqdm import tqdm

from glob import glob

from utils.infer_funcs_t12ct import do_mr_to_pct


# input_mr_path = "./t1_MAGUIXIA_0000.nii.gz"
# out_pseudo_ct_path = "./t12ct_valid_MAGUIXIA_0000_pseudo_ct.nii.gz"

# transformer + L1 loss
# pretrained_model_path = "runs/mr2ct_supervise_transformer/no_weight_decay/model_best.pt"
# state_dict = torch.load(pretrained_model_path)['G']

# transformer + L1 loss + SSIM loss
pretrained_model_path = "runs/mr2ct_supervise_t12ct_larger_patch_transformer_windowloss/8/model_best.pt"
state_dict = torch.load(pretrained_model_path)['G']

# pretrained_model_path = "runs/mr2ct_supervise_larger_patch/1/model_best.pt"
# state_dict = torch.load(pretrained_model_path)['G']

# do_mr_to_pct(
#     input_mr_file=input_mr_path, 
#     output_pct_file=out_pseudo_ct_path, 
#     saved_model=state_dict, 
#     device="cpu", prep_t1=False,
#     sliding_window_infer=False,
#     transformer_layers=2, 
#     img_size=(576, 576, 192)
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


json_path = "data/cross_validation_t12ct/cross_validation_fold_0.json"
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data"


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
        transformer_layers=2, 
        img_size=(576, 576, 192)
    )
    
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data"
cases = list(glob(join(output_data_dir, "*")))

mae_dict = defaultdict(list)

for case_dir in tqdm(cases):
    mr_path = join(case_dir, "mr.nii.gz")
    ct_path = join(case_dir, "ct.nii.gz")
    pct_path = join(case_dir, "pseudo_ct.nii.gz")
    
    mr_arr = nib.load(mr_path).get_fdata()
    ct_arr = nib.load(ct_path).get_fdata()
    pct_arr = nib.load(pct_path).get_fdata()
    
    # global MAE
    mae_global = np.abs(ct_arr - pct_arr).mean()
    
    # MAE in the window 500-1500
    window_500_1500_mask = (ct_arr > 500) & (ct_arr < 1500)
    mae_window_500_1500 = np.abs(ct_arr[window_500_1500_mask] - pct_arr[window_500_1500_mask]).mean()
    
    # MAE in the window 0-100
    window_0_100_mask = (ct_arr > 0) & (ct_arr < 100)
    mae_window_0_100 = np.abs(ct_arr[window_0_100_mask] - pct_arr[window_0_100_mask]).mean()
    
    # MAE in the window 100-1500
    window_100_1500_mask = (ct_arr > 100) & (ct_arr < 1500)
    mae_window_100_1500 = np.abs(ct_arr[window_100_1500_mask] - pct_arr[window_100_1500_mask]).mean()
    
    # MAE in the window 100-500
    window_100_500_mask = (ct_arr > 100) & (ct_arr < 500)
    mae_window_100_500 = np.abs(ct_arr[window_100_500_mask] - pct_arr[window_100_500_mask]).mean()
    
    mae_dict['global'].append(mae_global)
    mae_dict['window_500_1500'].append(mae_window_500_1500)
    mae_dict['window_0_100'].append(mae_window_0_100)
    mae_dict['window_100_1500'].append(mae_window_100_1500)
    mae_dict['window_100_500'].append(mae_window_100_500)
    

print("global MAE: ", np.mean(mae_dict['global']))
print("window 0-100 MAE: ", np.mean(mae_dict['window_0_100']))
print("window 100-500 MAE: ", np.mean(mae_dict['window_100_500']))
print("window 100-1500 MAE: ", np.mean(mae_dict['window_100_1500']))
print("window 500-1500 MAE: ", np.mean(mae_dict['window_500_1500']))

# global MAE:  121.41088780741441
# window 0-100 MAE:  32.44071605098422
# window 100-500 MAE:  149.15146288338298
# window 100-1500 MAE:  173.42578104213732
# window 500-1500 MAE:  190.588127359214