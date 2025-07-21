#!/usr/bin/env python
# mr-to-pct_script.py
# Runs mr to pct as a python script
# Inputs:
#   input_mr_file       Filename of your T1w MRI including full path
#   output_pct_file     Filename to give the output pCT including full path
#
# SNY: Wed 23 Nov 08:51:11 GMT 2022

import sys, os, torch
from utils.infer_funcs import do_mr_to_pct

# set device, use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trained model to load
saved_model = torch.load("/data/dingsd/mr2ct/pretrained_weights/pretrained_net_final_20220825.pth", map_location=device)

# Do you want to prepare the t1 image? This will perform bias correction and create a head mask
# yes = True, no = False. Output will be saved to _prep.nii
prep_t1 = False

# Do you want to produce an example plot? yes = True, no = False.
plot_mrct = False

# print('T1w MR input file path: {}'.format(input_mr_file))
# print('pCT output file path: {}'.format(output_pct_file))

# Run MR to pCT
# do_mr_to_pct(input_mr_file, output_pct_file, saved_model, device, prep_t1, plot_mrct)

    
json_path = "/data/dingsd/mr2ct/data/cross_validation_t12ct/cross_validation_fold_0.json"

import json
with open(json_path) as f:
    data = json.load(f)

test_data = data['validation']

from tqdm import tqdm
from os.path import basename, dirname, join
from shutil import copyfile

output_data_dir = "/data/dingsd/mr2ct/data/orig_dcm_data/preprocessed/brain_stimulation_transformed_data"

for case_idx, case in enumerate(tqdm(test_data)):
    mr_path = join("/data/dingsd/mr2ct", case['mr_image'])
    ct_path = join("/data/dingsd/mr2ct", case['ct_image'])
    mr_name = basename(mr_path).split(".nii")[0]
    ct_name = basename(ct_path).split(".nii")[0]

    out_dir = join(output_data_dir, str(case_idx))
    os.makedirs(out_dir, exist_ok=True)

    copyfile(mr_path, join(out_dir, f"mr.nii.gz"))
    copyfile(ct_path, join(out_dir, f"ct.nii.gz"))

    pct_path = join(out_dir, f"pseudo_ct.nii.gz")
    
    # Run MR to pCT
    do_mr_to_pct(mr_path, pct_path, saved_model, device, prep_t1, plot_mrct)