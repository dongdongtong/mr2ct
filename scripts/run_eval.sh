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

python3 scripts/test.py \
    --config configs/mr2ct.yml \
    --checkpoint_path runs/mr2ct/1/model_0.pt \
    --out_dir data/generated_data