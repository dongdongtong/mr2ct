#!/bin/bash


root_dir=$1  # for example: data/test_data

python preprocess_paired_mr_ct/preprocess_and_reg_based_on_betimage_reorganize.py \
    --root_dir $root_dir