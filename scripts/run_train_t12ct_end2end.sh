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

python3 scripts/train_sup.py \
    --config configs/mr2ct_supervise_t12ct_larger_patch_completehw_transformer.yml \
    --trainer sup_mr2ct \
    --gradient_accumulation_step 2