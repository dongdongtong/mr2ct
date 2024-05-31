#!/usr/bin/env python
# written by: Dr. Shaodong Ding 2024-05-30
# mr2ct, folder, or file path

import os
from os.path import dirname, basename, join
import sys
import SimpleITK as sitk
import torch

from utils.infer_funcs import do_mr_to_pct


input_mr_path = "./MAGUIXIA_0000.nii.gz"
out_pseudo_ct_path = "./MAGUIXIA_0000_pseudo_ct_transformer.nii.gz"

# transformer + L1 loss
# pretrained_model_path = "runs/mr2ct_supervise_transformer/no_weight_decay/model_best.pt"
# state_dict = torch.load(pretrained_model_path)['G']

# transformer + L1 loss + SSIM loss
pretrained_model_path = "runs/mr2ct_supervise_transformer_ssim/2/model_best.pt"
state_dict = torch.load(pretrained_model_path)['G']

# pretrained_model_path = "runs/mr2ct_supervise_larger_patch/1/model_best.pt"
# state_dict = torch.load(pretrained_model_path)['G']

do_mr_to_pct(
    input_mr_file=input_mr_path, 
    output_pct_file=out_pseudo_ct_path, 
    saved_model=state_dict, 
    device="cuda", prep_t1=False,
    sliding_window_infer=False,
    transformer_layers=3, 
    img_size=(448, 448, 64)
)