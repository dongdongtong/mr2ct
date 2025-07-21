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


input_mr_path = "./t1_MAGUIXIA_0000.nii.gz"
out_pseudo_ct_path = "./t12ct_valid_MAGUIXIA_0000_pseudo_ct.nii.gz"

# transformer + L1 loss + SSIM loss
pretrained_model_path = "runs/mr2ct_supervise_t12ct_larger_patch_transformer_windowloss/8/model_best.pt"
state_dict = torch.load(pretrained_model_path)['G']

do_mr_to_pct(
    input_mr_file=input_mr_path, 
    output_pct_file=out_pseudo_ct_path, 
    saved_model=state_dict, 
    device="cuda", prep_t1=True,
    sliding_window_infer=True,
    transformer_layers=2, 
    img_size=(576, 576, 192)
)