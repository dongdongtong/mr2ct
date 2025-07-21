import os
import pydicom
import SimpleITK as sitk
from collections import defaultdict
from glob import glob
import json


def read_dicom_series_from_directory(directory):
    """Read DICOM series from a directory."""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_files)
    series = reader.Execute()
    return series


def read_dicom_series_from_dcms(dicom_files):
    """Read DICOM series from a list of dcms."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    series = reader.Execute()
    return series


def convert_dicom_to_nii(dcms, save_path):
    """Convert DICOM series to NIfTI (.nii file)."""
    # Read DICOM series
    if isinstance(dcms, str):
        series = read_dicom_series_from_directory(dcms)
    elif isinstance(dcms, list):
        series = read_dicom_series_from_dcms(dcms)
    else:
        raise ValueError("Invalid input.")

    # Write NIfTI file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sitk.WriteImage(series, save_path)


def split_dicom_series(input_directory, output_directory):
    print("Processing:", input_directory, "---->", output_directory)
    # Dictionary to store DICOM files grouped by series
    series_dicoms = defaultdict(list)
    window_info = defaultdict(dict)

    # Read DICOM files and group them by series
    files = os.listdir(input_directory)
    for file in files:
        dicom_file = os.path.join(input_directory, file)
        ds = pydicom.dcmread(dicom_file)
        # print(ds)
        series_uid = ds.SeriesDescription.replace(" ", "_")
        # print(series_uid, ds.WindowWidth, ds.StudyDescription, ds.SeriesDescription, ds.AcquisitionNumber, ds.InstanceNumber)
        series_dicoms[series_uid].append(dicom_file)

        window_info[series_uid]["window_center"] = ds.WindowCenter
        window_info[series_uid]["window_width"] = ds.WindowWidth

    # Convert DICOM to NIfTI for each series
    for series_uid, dicom_files in series_dicoms.items():
        dicom_files = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

        # Convert DICOM to NIfTI
        save_path = os.path.join(output_directory, f"{os.path.basename(input_directory)}----{series_uid}.nii.gz")
        try:
            convert_dicom_to_nii(dicom_files, save_path)
            window_json = save_path.replace(".nii.gz", ".json")
            
            with open(window_json, "w") as f:
                dump_info = dict()

                def get_values(window_list):
                    try:
                        return [float(value) for value in window_list]
                    except:
                        return float(window_list)
                    
                dump_info["window_center"] = get_values(window_info[series_uid]['window_center'])
                dump_info["window_width"] = get_values(window_info[series_uid]['window_width'])
                
                json.dump(dump_info, f, sort_keys=True, indent=4)
        except Exception as e:
            print("Error:", e)
            continue


def main():
    data_dir = "data/dicom"
    centers = list(glob(os.path.join(data_dir, "*")))
    centers = [center for center in centers if os.path.isdir(center)]

    for center in centers:
        if os.path.basename(center) == "mr":
            continue
        dcm_dirs = list(glob(os.path.join(center, "*")))
        dcm_dirs = [dcm_dir for dcm_dir in dcm_dirs if os.path.isdir(dcm_dir)]

        for dcm_dir in dcm_dirs:
            split_dicom_series(dcm_dir, os.path.join("data/nii", os.path.basename(center), os.path.basename(dcm_dir)))

    

if __name__ == "__main__":
    main()
