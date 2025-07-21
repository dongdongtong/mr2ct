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

# patch_size: [192, 192, 32]
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct.yml \
    --trainer sup_mr2ct \
    --gradient_accumulation_step 1 \

# patch_size: [256, 256, 64], first training
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct_larger_patch.yml \
    --trainer sup_mr2ct \
    --gradient_accumulation_step 1 \
    --load_weights \
    --resume \
    --resume_path runs/mr2ct_supervise_t12ct/2/model_best.pt

# patch_size: [256, 256, 64], second training
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct_larger_patch.yml \
    --trainer sup_mr2ct \
    --gradient_accumulation_step 1 \
    --load_weights \
    --resume \
    --resume_path runs/mr2ct_supervise_t12ct_larger_patch/1/model_best.pt

# patch_size: [256, 256, 64], third training
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct_larger_patch.yml \
    --trainer sup_mr2ct \
    --gradient_accumulation_step 1 \
    --load_weights \
    --resume \
    --resume_path runs/mr2ct_supervise_t12ct_larger_patch/2/model_best.pt

# patch_size: [256, 256, 64], add transformer blocks
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct_larger_patch_transformer.yml \
    --trainer sup_mr2ct \
    --gradient_accumulation_step 1 \
    --load_weights \
    --resume \
    --resume_path runs/mr2ct_supervise_t12ct_larger_patch/3/model_best.pt

# patch_size: [576, 576, 64], + transformer blocks and + windowed weighted loss
python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct_larger_patch_transformer_windowloss.yml \
    --trainer sup_mr2ct_patch_windowed_weighted_loss \
    --gradient_accumulation_step 2 \
    --load_weights \
    --resume \
    --resume_path runs/mr2ct_supervise_t12ct_larger_patch_transformer/1/model_best.pt