import sys
sys.path.append("/data/zhaoyu/ParkinsonSWI")
import SimpleITK as sitk

from os.path import dirname, basename, join
from glob import glob
import os
import pandas as pd
import re
import json
from tqdm import tqdm


def dicom_to_nifti(dicom_path, nifti_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    os.makedirs(dirname(nifti_path), exist_ok=True)
    sitk.WriteImage(image, nifti_path)
    # print("complete")


def exist_chinese_character(file_name):
    ccs = re.findall(r'[\u4e00-\u9fff]+', file_name)
    
    if len(ccs) > 0:
        return True
    else:
        return False

# convert dicom to nii
# and save a excel to record the nii_path and label
def process_dicom2nii_folder(dicom_folder_path, output_folder_path):
    dcm_folder_paths = list(glob(join(dicom_folder_path, "*")))
    dcm_folder_paths = [dcm_folder_path for dcm_folder_path in dcm_folder_paths if os.path.isdir(dcm_folder_path)]
    
    data_df = pd.DataFrame(columns=["patient_name", "orig_dicom_series_path", "nii_path"])
    
    faliure_convert = []
    
    for dcm_folder_idx, dcm_folder_path in enumerate(dcm_folder_paths):
        patient_folder_name = basename(dcm_folder_path)
        print(f"Processing {patient_folder_name}...")
        
        if exist_chinese_character(patient_folder_name):
            nii_path = join(output_folder_path, str(dcm_folder_idx) + ".nii.gz")
        else:
            nii_path = join(output_folder_path, patient_folder_name + ".nii.gz")
        
        if os.path.exists(nii_path):
            continue
        
        try:
            dicom_to_nifti(dcm_folder_path, nii_path)
        except Exception:
            faliure_convert.append(dcm_folder_path)
            continue
        
        data_df.loc[len(data_df)] = [patient_folder_name, dcm_folder_path, nii_path]

    print("convert complemted, but found the following failure cases:")
    print(faliure_convert)

if __name__ == "__main__":
    dicom_dir_path = "data/dicom/ct"
    nii_dir_path = "data/nii/ct"
    process_dicom2nii_folder(dicom_dir_path, nii_dir_path)