'''
@dsd 2022/11/01 22:19
Leverage the package of preprocess to do multi-process nnunet-like preprocess for all ct_brain data
'''


import sys
sys.path.append("/opt/dingsd/cyclegan_mr2ct")
import os
from glob import glob
from os.path import basename, dirname, join

from tqdm import tqdm
import SimpleITK as sitk
import time
import datetime
import numpy as np
import cv2

from preprocess.preprocessor import Preprocessor

# from utils.visualize_slices import visualize_img

modality = "ct"
target_spacing = [0.5, 0.5, 5.0]

def single_process_preprocess(subject_paths, target_spacing, do_separate_z):
    prev_time = time.time()
    subject_count = len(subject_paths)
    preprocessor = Preprocessor(target_spacing, do_separate_z)
    cropped_dataset_path = f"data/cropped_pipeline_niigz_betneck/{modality}"
    resampled_dataset_path = f"data/resampled_cropped_pipeline_niigz_betneck/{modality}"
    for subject_index, subject_path in enumerate(tqdm(subject_paths)):
        group = basename(dirname(dirname(subject_path)))  # HC/MSA-C/MSA-P/PD
        patient_name = basename(dirname(subject_path))
        
        brain_mask_path = join(dirname(subject_path), basename(subject_path).split(".nii")[0] + "_mask.nii.gz")
        print(subject_path, brain_mask_path)
        
        # try:
        resampled_itk_dict = preprocessor.run(subject_path, brain_mask_path=brain_mask_path)
        cropped_img_path = join(cropped_dataset_path, group, patient_name, basename(subject_path).split(".")[0]) + ".nii.gz"
        os.makedirs(dirname(cropped_img_path), exist_ok=True)
        sitk.WriteImage(resampled_itk_dict['cropped_img'], cropped_img_path)

        resampled_img_path = join(resampled_dataset_path, group, patient_name, basename(subject_path).split(".")[0]) + ".nii.gz"
        os.makedirs(dirname(resampled_img_path), exist_ok=True)
        sitk.WriteImage(resampled_itk_dict['resampled_img'], resampled_img_path)


        subjects_left = subject_count - subject_index
        time_left = datetime.timedelta(seconds=subjects_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write("\r[Processed Subjects %d/%d] [Processing: %s] ETA: %s" % (
            subject_index + 1,
            subject_count,
            basename(subject_path).split(".")[0],
            time_left,
        ))
        # break



# multi-process preprocess
from multiprocessing import Process
def multi_process_preprocess(subjects, target_spacing, do_separate_z):
    process_list = []
    cpu_count = 31
    batch = len(subjects) // cpu_count
    for i in range(cpu_count):
        if i == (cpu_count - 1):
            this_subjects = subjects[i * batch:]
        else:
            this_subjects = subjects[i*batch : (i+1)*batch]
        p = Process(target=single_process_preprocess, args=(this_subjects, target_spacing, do_separate_z))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print("preprocess end!")

import pandas as pd
if __name__ == "__main__":
    root_dir = f"/opt/dingsd/cyclegan_mr2ct/data/pipeline_niigz_betneck/{modality}"
    
    subject_paths = list(glob(join(root_dir, "*.nii.gz")))

    print(len(subject_paths))

    target_spacing, do_separate_z = (0.5, 0.5, 5.0), True
    print(target_spacing)  #

    single_process_preprocess(subject_paths, target_spacing, do_separate_z)

    # multi_process_preprocess(subject_paths, target_spacing, do_separate_z)
