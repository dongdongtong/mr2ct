#!/bin/bash

subject_path="data/pipeline_niigz_betneck/mr/LIUCHENGYI_0000.nii.gz"
target_seg_path="data/pipeline_niigz_betneck_brainsegs/mr/LIUCHENGYI_0000_seg.nii.gz"

python /home/dingsd/workstation/SynthSeg/scripts/commands/SynthSeg_predict.py \
            --i $subject_path \
            --o $target_seg_path \
            --robust --crop 224 224 160