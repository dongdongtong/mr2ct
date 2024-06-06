#!/bin/bash

# Set the GPU device IDs you want to use for training
# Replace the numbers inside the parentheses with the GPU IDs you want to use (e.g., 0,1,2 for GPUs 0, 1, and 2)
gpu_devices=(0)

# Join the GPU device IDs into a comma-separated string
gpu_ids=$(IFS=,; echo "${gpu_devices[*]}")

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$gpu_ids"

# Get the length of the gpu_devices array
# gpu_devices_length=${#gpu_devices[@]}

# python3 scripts/train.py \
#     --config configs/mr2ct.yml \
#     --trainer cyclegan_mr2ct \
#     --gradient_accumulation_step 1 \
#     --amp


# python3 scripts/train_sup.py \
#     --config configs/mr2ct_supervise.yml \
#     --trainer sup_mr2ct \
#     --fold 0 \
#     --gradient_accumulation_step 1 \
#     # --load_weights \
#     # --resume \
#     # --resume_path runs/mr2ct_supervise/shuffleunet_patch_size_256_256_32_no_amp/model_best.pt
#     # --amp

# python3 scripts/train_sup.py \
#     --config configs/mr2ct_supervise_larger_patch.yml \
#     --trainer sup_mr2ct \
#     --fold 0 \
#     --gradient_accumulation_step 1 \
#     --load_weights \
#     --resume \
#     --resume_path runs/mr2ct_supervise/shuffleunet_patch_size_256_256_32_no_amp_last_channel_384/model_best.pt
#     # --amp


# python3 scripts/train_sup.py \
#     --config configs/mr2ct_supervise_transformer.yml \
#     --trainer sup_mr2ct_no_patch \
#     --fold 0 \
#     --gradient_accumulation_step 1 \
#     --load_weights \
#     --resume \
#     --resume_path runs/mr2ct_supervise_larger_patch/1/model_best.pt
#     # --amp

# # add ssim loss
# python3 scripts/train_sup.py \
#     --config configs/mr2ct_supervise_transformer_ssim.yml \
#     --trainer sup_mr2ct_no_patch_ssim \
#     --fold 0 \
#     --gradient_accumulation_step 1 \
#     --load_weights \
#     --resume \
#     --resume_path runs/mr2ct_supervise_transformer/no_weight_decay/model_best.pt \
#     # --amp


# # add ssim loss
# python3 scripts/train_sup.py \
#     --config configs/mr2ct_supervise_transformer_ssim.yml \
#     --trainer sup_mr2ct_no_patch_ssim \
#     --fold 0 \
#     --gradient_accumulation_step 1 \
#     --load_weights \
#     --resume \
#     --resume_path runs/mr2ct_supervise_transformer/no_weight_decay/model_best.pt \
#     # --amp


# add windowed ssim loss
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_transformer_windowed_ssim.yml \
    --trainer sup_mr2ct_no_patch_windowed \
    --fold 0 \
    --gradient_accumulation_step 1 \
    --load_weights \
    --resume \
    --resume_path runs/mr2ct_supervise_transformer/no_weight_decay/model_best.pt \
    # --amp