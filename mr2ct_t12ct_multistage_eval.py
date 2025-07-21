#!/usr/bin/env python
# written by: Dr. Shaodong Ding 2024-11-5
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

import json
from shutil import copyfile

from utils.infer_funcs_t12ct import do_mr_to_pct


def eval_window_mae(output_data_dir="data/orig_dcm_data/preprocessed/transformed_data"):
    
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
        
        # MAE in the window 0-100
        window_0_100_mask = (ct_arr > 0) & (ct_arr < 100)
        mae_window_0_100 = np.abs(ct_arr[window_0_100_mask] - pct_arr[window_0_100_mask]).mean()
        
        # MAE in the window 100-1500
        window_100_1500_mask = (ct_arr > 100) & (ct_arr < 1500)
        mae_window_100_1500 = np.abs(ct_arr[window_100_1500_mask] - pct_arr[window_100_1500_mask]).mean()
        
        # MAE in the window 100-2000
        window_100_2000_mask = (ct_arr > 100) & (ct_arr < 2000)
        mae_window_100_2000 = np.abs(ct_arr[window_100_2000_mask] - pct_arr[window_100_2000_mask]).mean()
        
        # MAE in the window 500-1500
        window_500_1500_mask = (ct_arr > 500) & (ct_arr < 1500)
        mae_window_500_1500 = np.abs(ct_arr[window_500_1500_mask] - pct_arr[window_500_1500_mask]).mean()
        
        # MAE in the window 500-2000
        window_500_2000_mask = (ct_arr > 500) & (ct_arr < 2000)
        mae_window_500_2000 = np.abs(ct_arr[window_500_2000_mask] - pct_arr[window_500_2000_mask]).mean()
        
        # MAE in the window 100-500
        window_100_500_mask = (ct_arr > 100) & (ct_arr < 500)
        mae_window_100_500 = np.abs(ct_arr[window_100_500_mask] - pct_arr[window_100_500_mask]).mean()
        
        mae_dict['global'].append(mae_global)
        mae_dict['window_0_100'].append(mae_window_0_100)
        mae_dict['window_100_1500'].append(mae_window_100_1500)
        mae_dict['window_100_2000'].append(mae_window_100_2000)
        mae_dict['window_500_1500'].append(mae_window_500_1500)
        mae_dict['window_500_2000'].append(mae_window_500_2000)
        mae_dict['window_100_500'].append(mae_window_100_500)
        

    print("global MAE: ", np.mean(mae_dict['global']))
    print("window 0-100 MAE: ", np.mean(mae_dict['window_0_100']))
    print("window 100-500 MAE: ", np.mean(mae_dict['window_100_500']))
    print("window 100-1500 MAE: ", np.mean(mae_dict['window_100_1500']))
    print("window 100-2000 MAE: ", np.mean(mae_dict['window_100_2000']))
    print("window 500-1500 MAE: ", np.mean(mae_dict['window_500_1500']))
    print("window 500-2000 MAE: ", np.mean(mae_dict['window_500_2000']))
    

@torch.no_grad()
def generate_pseudo_ct(
    model_path="runs/mr2ct_supervise_t12ct_larger_patch_transformer_windowloss/8/model_best.pt", 
    output_data_dir="data/orig_dcm_data/preprocessed/transformed_data",
    transformer_layers=2,
    img_size=(576, 576, 192),
    patch_size=(576, 576, 64)
    ):
    
    state_dict = torch.load(model_path)['G']
    
    json_path = "data/cross_validation_t12ct/cross_validation_fold_0.json"

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
            sliding_window_infer=True,
            transformer_layers=transformer_layers, 
            img_size=img_size,
            patch_size=patch_size
        )



# # ============================================
# # ====  T1 to CT Patch Size 192, 192, 32 =====
# # ============================================
print("T1 to CT Patch Size 192, 192, 32")
model_path = "runs/mr2ct_supervise_t12ct/2/model_best.pt"
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data_ps_192_192_32"
generate_pseudo_ct(model_path, output_data_dir, transformer_layers=0, img_size=(576, 576, 192), patch_size=(192, 192, 32))


# ============================================
# ====  T1 to CT Patch Size 256, 256, 64 =====
# ============================================
print("T1 to CT Patch Size 256, 256, 64")
model_path = "runs/mr2ct_supervise_t12ct_larger_patch/3/model_best.pt"
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data_ps_256_256_64"
generate_pseudo_ct(model_path, output_data_dir, transformer_layers=0, img_size=(576, 576, 192), patch_size=(256, 256, 64))


# ===================================================================
# ====  T1 to CT Patch Size 256, 256, 64 + 2 transformer layers =====
# ===================================================================
print("T1 to CT Patch Size 256, 256, 64 + 2 transformer layers")
model_path = "runs/mr2ct_supervise_t12ct_larger_patch_transformer/1/model_best.pt"
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data_ps_256_256_64_2transformer"
generate_pseudo_ct(model_path, output_data_dir, transformer_layers=2, img_size=(576, 576, 192), patch_size=(256, 256, 64))


# ===================================================================
# ====  T1 to CT Patch Size 576, 576, 64 + 2 transformer layers =====
# ===================================================================
print("T1 to CT Patch Size 576, 576, 64 + 2 transformer layers")
model_path = "runs/mr2ct_supervise_t12ct_larger_patch_completehw_transformer/4/model_best.pt"
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data_ps_576_576_64_2transformer"
generate_pseudo_ct(model_path, output_data_dir, transformer_layers=2, img_size=(576, 576, 192), patch_size=(576, 576, 64))


# # ===================================================================
# # ====  T1 to CT Patch Size 576, 576, 64 + 2 transformer layers + window loss =====
# # ===================================================================
print("T1 to CT Patch Size 576, 576, 64 + 2 transformer layers + window loss")
model_path = "runs/mr2ct_supervise_t12ct_larger_patch_transformer_windowloss/8/model_best.pt"
output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data_ps_576_576_64_2transformer_windowloss"
generate_pseudo_ct(model_path, output_data_dir, transformer_layers=2, img_size=(576, 576, 192), patch_size=(576, 576, 64))



# ===================================================================
# ====  Evaluation  =====
# ===================================================================
print("Evaluation on 192, 192, 32")
eval_window_mae(output_data_dir="data/orig_dcm_data/preprocessed/transformed_data_ps_192_192_32")

print("Evaluation on 256, 256, 64")
eval_window_mae(output_data_dir="data/orig_dcm_data/preprocessed/transformed_data_ps_256_256_64")

print("Evaluation on 256, 256, 64 + 2 transformer layers")
eval_window_mae(output_data_dir="data/orig_dcm_data/preprocessed/transformed_data_ps_256_256_64_2transformer")

print("Evaluation on 576, 576, 64 + 2 transformer layers")
eval_window_mae(output_data_dir="data/orig_dcm_data/preprocessed/transformed_data_ps_576_576_64_2transformer")

print("Evaluation on 576, 576, 64 + 2 transformer layers + window loss")
eval_window_mae(output_data_dir="data/orig_dcm_data/preprocessed/transformed_data_ps_576_576_64_2transformer_windowloss")

# print("Evaluation on brain stimulation")
# eval_window_mae(output_data_dir="/data/dingsd/mr2ct/data/orig_dcm_data/preprocessed/brain_stimulation_transformed_data")

# Evaluation on 192, 192, 32
# global MAE:  519.8048246791785
# window 0-100 MAE:  523.1781756525123
# window 100-500 MAE:  794.1845074979007
# window 100-1500 MAE:  1223.6158064629496
# window 100-2000 MAE:  1246.5001971030085
# window 500-1500 MAE:  1525.9778652152404
# window 500-2000 MAE:  1549.0892161208667

# Evaluation on 256, 256, 64
# global MAE:  150.83023049780644
# window 0-100 MAE:  97.66711685078602
# window 100-500 MAE:  175.08481764272915
# window 100-1500 MAE:  199.79638052252992
# window 100-2000 MAE:  206.13606440452594
# window 500-1500 MAE:  217.15269369423527
# window 500-2000 MAE:  226.72972555485754

# Evaluation on 256, 256, 64 + 2 transformer layers
# global MAE:  148.60337476282024
# window 0-100 MAE:  84.15938590830345
# window 100-500 MAE:  159.0199896290663
# window 100-1500 MAE:  191.12944278804903
# window 100-2000 MAE:  197.35381403986884
# window 500-1500 MAE:  213.76212539584898
# window 500-2000 MAE:  222.92048262089557

# Evaluation on 576, 576, 64 + 2 transformer layers
# global MAE:  116.04050417603337
# window 0-100 MAE:  49.14212688840328
# window 100-500 MAE:  141.4296700403317
# window 100-1500 MAE:  188.34921361829151
# window 100-2000 MAE:  193.36219160501892
# window 500-1500 MAE:  221.4527431322143
# window 500-2000 MAE:  228.07928088461622

# Evaluation on 576, 576, 64 + 2 transformer layers + window loss
# global MAE:  121.41088780741441
# window 0-100 MAE:  32.44071605098422
# window 100-500 MAE:  149.15146288338298
# window 100-1500 MAE:  173.42578104213732
# window 100-2000 MAE:  176.66602712891608
# window 500-1500 MAE:  190.588127359214
# window 500-2000 MAE:  195.04084066709794


# Evaluation using the brain stimulation tools
# global MAE:  158.42221854065718
# window 0-100 MAE:  64.93241441518697
# window 100-500 MAE:  258.6890613788137
# window 100-1500 MAE:  311.9585533788625
# window 100-2000 MAE:  317.1343188549011
# window 500-1500 MAE:  349.4577268784875
# window 500-2000 MAE:  356.0866557035673




# ======================================
# === Eval using sliding window infer ==
# ======================================

# Evaluation on 192, 192, 32
# global MAE:  121.34901265040908
# window 0-100 MAE:  83.41847121553785
# window 100-500 MAE:  174.2524351626938
# window 100-1500 MAE:  237.9506083284588
# window 100-2000 MAE:  247.26964089786588
# window 500-1500 MAE:  282.6451608540935
# window 500-2000 MAE:  295.729050056064

# Evaluation on 256, 256, 64
# global MAE:  124.68292634573125
# window 0-100 MAE:  62.12045209661302
# window 100-500 MAE:  165.2185928041054
# window 100-1500 MAE:  199.62289756255552
# window 100-2000 MAE:  204.70773492945023
# window 500-1500 MAE:  223.84675692003012
# window 500-2000 MAE:  230.97067560363638

# Evaluation on 256, 256, 64 + 2 transformer layers
# global MAE:  117.43758429202225
# window 0-100 MAE:  64.2026419767237
# window 100-500 MAE:  163.88844701687464
# window 100-1500 MAE:  200.75082384982784
# window 100-2000 MAE:  206.47099375856982
# window 500-1500 MAE:  226.65085073359
# window 500-2000 MAE:  234.76114043599964

# Evaluation on 576, 576, 64 + 2 transformer layers
# global MAE:  89.99462051512681
# window 0-100 MAE:  61.15904345419932
# window 100-500 MAE:  166.0980548117468
# window 100-1500 MAE:  194.5579670075793
# window 100-2000 MAE:  198.377748206024
# window 500-1500 MAE:  214.66994193371698
# window 500-2000 MAE:  220.0100315896196

# Evaluation on 576, 576, 64 + 2 transformer layers + window loss
# global MAE:  99.85034948818274
# window 0-100 MAE:  44.38150375341068
# window 100-500 MAE:  159.04477794186369
# window 100-1500 MAE:  175.22950561134832
# window 100-2000 MAE:  177.70776393911666
# window 500-1500 MAE:  186.72264672599425
# window 500-2000 MAE:  190.25027200934687

# Evaluation using the brain stimulation tools
# global MAE:  158.42221854065718
# window 0-100 MAE:  64.93241441518697
# window 100-500 MAE:  258.6890613788137
# window 100-1500 MAE:  311.9585533788625
# window 100-2000 MAE:  317.1343188549011
# window 500-1500 MAE:  349.4577268784875
# window 500-2000 MAE:  356.0866557035673