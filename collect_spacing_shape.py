import SimpleITK as sitk

import os
from os.path import dirname, basename, join
from glob import glob
import json
from tqdm import tqdm
import pandas as pd

def get_data_dict():
    data_df = pd.DataFrame(columns=["nii_path", "pname", "spacing_x", "spacing_y", "spacing_z", "shape_x", "shape_y", "shape_z"])
    
    data_dir = "data/pipeline_niigz_betneck/mr"
    nii_paths = list(glob(join(data_dir, "*.nii.gz")))

    fail_niis = []
    for nii_path in nii_paths:
        pname = basename(nii_path).split(".nii")[0]
        try:
            itk_img = sitk.ReadImage(nii_path)
        except:
            fail_niis.append(nii_path)
            continue

        data_df.loc[len(data_df)] = [nii_path, pname, *itk_img.GetSpacing(), *itk_img.GetSize()]
    
    print("failure nii paths: ", fail_niis)
    
    return data_df


if __name__ == "__main__":
    # # ===============collecting info from original NIFTI files=====================
    # data_df = get_data_dict()
    # data_df.to_excel("data/mr_info.xlsx", index=False)
    # print(data_df.head())
    # print(data_df.shape)
    
    # merge nii info
    ct_data_df = pd.read_excel("data/ct_info.xlsx")
    mr_data_df = pd.read_excel("data/mr_info.xlsx")

    data_df = pd.merge(ct_data_df, mr_data_df, on="pname", how="inner")

    data_df.to_excel("data/ct_mr_info.xlsx", index=False)

    
    