import SimpleITK as sitk

from os.path import dirname, basename, join
from glob import glob
import os


def dicom_to_nifti(dicom_path, nifti_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    os.makedirs(dirname(nifti_path), exist_ok=True)
    sitk.WriteImage(image, nifti_path)
    # print("complete")


# dicom_to_nifti("/data/zhaoyu/ParkinsonSWI/data/dicom/HC/吴世雷", "./asdasd.nii.gz")