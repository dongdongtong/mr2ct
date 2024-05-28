#!/bin/bash


# ===================================
# ======= Convert to nii.gz =========
# ===================================
# in_dir="/opt/dingsd/cyclegan_mr2ct/data/dicom"
# out_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz"

# /home/dingsd/anaconda3/envs/nnunet/bin/python /home/dingsd/workstation/Parkinson7T_Pipeline/process_dicom_nii_mixed_to_niigz.py --in_dir $in_dir --out_dir $out_dir


# =========================================
# ======= Remove neck slices =========
# =========================================
# in_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz"
# out_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_betneck"

# /home/dingsd/anaconda3/envs/nnunet/bin/python /home/dingsd/workstation/Parkinson7T_Pipeline/crop_neck_slices.py --in_dir $in_dir --out_dir $out_dir


# ====================================
# ======= Bet using Synthseg =========
# ====================================
# firstly, get brain mask
synthseg_workspace_dir="/home/dingsd/workstation/SynthSeg"
synthseg_python_env_path="/home/dingsd/anaconda3/envs/synthseg/bin/python"
in_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_betneck"
out_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_betneck_brainsegs"

$synthseg_python_env_path /home/dingsd/workstation/Parkinson7T_Pipeline/synthseg_predict_brainmask.py \
    --in_dir $in_dir \
    --out_dir $out_dir \
    --python_path $synthseg_python_env_path \
    --synthseg_path $synthseg_workspace_dir

# # secondly, bet
# in_img_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_betneck"
# in_mask_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_betneck_brainsegs"
# out_dir="/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_bet"
# /home/dingsd/anaconda3/envs/nnunet/bin/python /home/dingsd/workstation/Parkinson7T_Pipeline/bet_7T_using_brainmask.py --in_img_dir $in_img_dir --in_mask_dir $in_mask_dir --out_dir $out_dir