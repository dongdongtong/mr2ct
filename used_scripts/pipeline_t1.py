import sys
sys.path.append("/data/dingsd/mr2ct")
import os
from os.path import dirname, basename, join
from glob import glob
from tqdm import tqdm
import pydicom
import numpy as np
from shutil import copyfile

import ants
import SimpleITK as sitk

from used_scripts.skull_strip import skull_stripping


def get_patient_name(dicom_dir):
    # get the first dicom file
    dicom_files = glob(join(dicom_dir, "*"))
    dicom_files = [f for f in dicom_files if os.path.isfile(f)]

    if len(dicom_files) == 0:
        raise Exception("No dicom files found in {}".format(dicom_dir))
    
    dicom_file = dicom_files[0]
    ds = pydicom.dcmread(dicom_file)
    name = str(ds.PatientName)

    processed_name = name.strip().replace(" ", "")
    return processed_name


def remove_nondirs(dirs):
    return [d for d in dirs if os.path.isdir(d)]


def dicom_to_nifti(dicom_path, nifti_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    os.makedirs(dirname(nifti_path), exist_ok=True)
    sitk.WriteImage(image, nifti_path)


def reorient_RAS(img_file, out_file):
    os.makedirs(dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        return
    
    print("Reorienting (RAS) MR image: ", img_file)

    img = ants.image_read(img_file)
    img = ants.reorient_image2(img, 'RAS')

    os.makedirs(dirname(out_file), exist_ok=True)
    ants.image_write(img, out_file)


def mr_preprocessing(img_file, out_file):
    os.makedirs(dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        return
    
    print("Processing MR image: ", img_file)

    img = ants.image_read(img_file)
    img = ants.reorient_image2(img, 'RAS')
    img_n4 = ants.n4_bias_field_correction(img)
    img_tmp = img_n4.otsu_segmentation(k=3) # otsu_segmentation
    img_tmp = ants.multi_label_morphology(img_tmp, 'MD', 2) # dilate 2
    img_tmp = ants.smooth_image(img_tmp, 3) # smooth 3
    img_tmp = ants.threshold_image(img_tmp, 0.5) # threshold 0.5
    img_mask = ants.get_mask(img_tmp)
    img_out = ants.multiply_images(img_n4, img_mask)

    os.makedirs(dirname(out_file), exist_ok=True)
    ants.image_write(img_out, out_file)
    ants.image_write(img_mask, out_file.replace('.nii.gz', '_brainmask.nii.gz'))


def mr_brainmask(img_file, out_file):
    os.makedirs(dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        return
    
    print("Processing MR image: ", img_file)

    img = ants.image_read(img_file)
    img = ants.reorient_image2(img, 'RAS')
    img_n4 = ants.n4_bias_field_correction(img)
    img_tmp = img_n4.otsu_segmentation(k=3) # otsu_segmentation
    img_tmp = ants.multi_label_morphology(img_tmp, 'MD', 2) # dilate 2
    img_tmp = ants.smooth_image(img_tmp, 3) # smooth 3
    img_tmp = ants.threshold_image(img_tmp, 0.5) # threshold 0.5
    img_mask = ants.get_mask(img_tmp)
    img_out = ants.multiply_images(img_n4, img_mask)

    os.makedirs(dirname(out_file), exist_ok=True)
    ants.image_write(img_out, out_file)
    ants.image_write(img_mask, out_file.replace('.nii.gz', '_brainmask.nii.gz'))


def extract_mr_data():
    data_root_mr_path = r"data/orig_dcm_data/MRI-CT/to Ziyang MR 20240306"

    # glob for all the subjects
    patient_dirs = list(glob(join(data_root_mr_path, "*")))
    patient_dirs = remove_nondirs(patient_dirs)
    
    patient_t1_dirs = []
    patient_t1_names = []
    patient_t1_sequence_dirs = []

    # Patient directory: a set of sequence directories
    for patient_dir in tqdm(patient_dirs):
        sequence_dirs = list(glob(join(patient_dir, "*")))
        sequence_dirs = remove_nondirs(sequence_dirs)

        # find the proper sequence directory
        # for now, we use keyword "t1_mx3d_sag_fs_0.6_a5" in directory name
        t1_sequence_dir = None
        for sequence_dir in sequence_dirs:
            sequence_name = basename(sequence_dir)

            if ("t1_mx3d_sag_fs_0.6_a5" in sequence_name or "t1_mx3d_sag_fs_0.6_a4" in sequence_name) \
                and "NIF" not in sequence_name and "MPR_TRA" not in sequence_name:
                t1_sequence_dir = sequence_dir

                print(t1_sequence_dir)
            
                patient_name = get_patient_name(t1_sequence_dir)

                if patient_name not in patient_t1_names:
                    patient_t1_names.append(patient_name)
                    patient_t1_sequence_dirs.append(t1_sequence_dir)

                break
    
    return patient_t1_names, patient_t1_sequence_dirs


def extract_ct_data():
    data_root_path = r"data/orig_dcm_data/MRI-CT/to ziyang CT 20240306"

    # glob for all the subjects
    patient_dirs = list(glob(join(data_root_path, "*")))
    patient_dirs = remove_nondirs(patient_dirs)
    patient_dirs = sorted(patient_dirs)
    # print(patient_dirs)
    
    # patient_dirs = []
    patient_names = []
    patient_sequence_dirs = []

    # Patient directory: a set of sequence directories
    for patient_dir in tqdm(patient_dirs):
        sequence_dirs = list(glob(join(patient_dir, "*")))
        sequence_dirs = remove_nondirs(sequence_dirs)
        sequence_dirs = sorted(sequence_dirs)

        # find the proper sequence directory
        # for now, we use keyword "brain" or "helical" or "head"
        for sequence_dir in sequence_dirs:
            sequence_name = basename(sequence_dir)

            sequence_name_lower = sequence_name.lower()
            # print(sequence_name_lower)

            if ("brain" in sequence_name_lower or 
                    "helical" in sequence_name_lower or 
                    "head" in sequence_name_lower) \
                and "5mm" in sequence_name_lower:

                # print(sequence_dir)
            
                patient_name = get_patient_name(sequence_dir)

                if patient_name not in patient_names:
                    patient_names.append(patient_name)
                    patient_sequence_dirs.append(sequence_dir)

                break
    
    return patient_names, patient_sequence_dirs

import subprocess
def remove_neck_slices(nii_path, betneck_nii_path):
    os.makedirs(dirname(betneck_nii_path), exist_ok=True)

    print("Processing image: ", nii_path)

    # First command
    head_top = subprocess.check_output("robustfov -i {} | grep -v Final | head -n 1 | awk '{{print $5}}'".format(nii_path), shell=True).decode().strip()

    # Second command
    subprocess.run("fslmaths {} -roi 0 -1 0 -1 {} 448 0 1 {}".format(nii_path, head_top, betneck_nii_path), shell=True)

    # remove neck slices (zero slices)
    itk_img = sitk.ReadImage(betneck_nii_path)
    itk_img_arr = sitk.GetArrayFromImage(itk_img)
    new_arr = []
    for i, img_slice in enumerate(itk_img_arr):
        if (img_slice != 0).sum() > 0:
            new_arr.append(img_slice)
        # else:
        #     if i >= itk_img_arr.shape[0] // 2:
        #         new_arr.append(img_slice)
                
    itk_new_arr = sitk.GetImageFromArray(new_arr)
    itk_new_arr.SetOrigin(itk_img.GetOrigin())
    itk_new_arr.SetDirection(itk_img.GetDirection())
    itk_new_arr.SetSpacing(itk_img.GetSpacing())
    sitk.WriteImage(itk_new_arr, betneck_nii_path)


def resample(nii, new_spacing):
    itk_img = sitk.ReadImage(nii)
    itk_img_arr = sitk.GetArrayFromImage(itk_img)
    itk_img_spacing = itk_img.GetSpacing()
    itk_img_size = itk_img.GetSize()

    new_size = [int(np.round(itk_img_size[i] * itk_img_spacing[i] / new_spacing[i])) for i in range(3)]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_img.GetDirection())
    resample.SetOutputOrigin(itk_img.GetOrigin())
    resample.SetDefaultPixelValue(itk_img.GetPixelIDValue())
    resample.SetOutputPixelType(itk_img.GetPixelID())

    resampled_img = resample.Execute(itk_img)
    return resampled_img


def registration(bet_mr_nii_path, bet_ct_nii_path, ct_nii_path, reg_out_dir):
    reged_ct_nii_path = join(reg_out_dir, "ct", basename(ct_nii_path))
    reged_brainmask_nii_path = reged_ct_nii_path.replace(".nii.gz", "_brainmask.nii.gz")
    reged_skullstrip_mask_nii_path = reged_ct_nii_path.replace(".nii.gz", "_skullstrip_mask.nii.gz")

    # registration
    head_ct_img_pre = ants.image_read(bet_ct_nii_path)
    head_mr_img_pre = ants.image_read(bet_mr_nii_path)

    print("registration starts...")
    out_dict = ants.registration(
        fixed=head_mr_img_pre,
        moving=head_ct_img_pre,
        type_of_transform="Similarity",
        aff_metric='mattes',
    )

    # apply the affine matrix to the head ct, head mask, skullstriped brain mask
    masked_ct_img = ants.image_read(ct_nii_path)
    reged_ct = ants.apply_transforms(
        fixed=head_mr_img_pre, 
        moving=masked_ct_img, 
        transformlist=out_dict['fwdtransforms'],
        interpolator='bSpline', defaultvalue=-1024)
    
    brainmask_nii_path = ct_nii_path.replace(".nii.gz", "_brainmask.nii.gz")
    brainmask_img = ants.image_read(brainmask_nii_path)
    reged_brainmask = ants.apply_transforms(
        fixed=head_mr_img_pre, 
        moving=brainmask_img, 
        transformlist=out_dict['fwdtransforms'],
        interpolator='nearestNeighbor', defaultvalue=0)
    
    skullstrip_mask_nii_path = bet_ct_nii_path.replace(".nii.gz", "_skullstrip_mask.nii.gz")
    skullstrip_mask_img = ants.image_read(skullstrip_mask_nii_path)
    reged_skullstrip_mask = ants.apply_transforms(
        fixed=head_mr_img_pre, 
        moving=skullstrip_mask_img, 
        transformlist=out_dict['fwdtransforms'],
        interpolator='nearestNeighbor', defaultvalue=0)
    
    os.makedirs(dirname(reged_ct_nii_path), exist_ok=True)
    ants.image_write(reged_ct, reged_ct_nii_path)
    ants.image_write(reged_brainmask, reged_brainmask_nii_path)
    ants.image_write(reged_skullstrip_mask, reged_skullstrip_mask_nii_path)


if __name__ == "__main__":
    # =============================
    # get dcm dirs and patient names
    # ============================
    # patient_t1_names, patient_t1_sequence_dirs = extract_mr_data()
    # print(len(patient_t1_names))
    
    # patient_ct_names, patient_ct_sequence_dirs = extract_ct_data()
    # print(len(patient_ct_names))

    # # find the corresponding CT and MR data
    # paired_patient_names = []
    # paired_patient_t1_sequence_dirs = []
    # paired_patient_ct_sequence_dirs = []
    # for i in range(len(patient_t1_names)):
    #     patient_name = patient_t1_names[i]
    #     if patient_name in patient_ct_names:
    #         paired_patient_names.append(patient_name)
    #         paired_patient_t1_sequence_dirs.append(patient_t1_sequence_dirs[i])
    #         paired_patient_ct_sequence_dirs.append(patient_ct_sequence_dirs[patient_ct_names.index(patient_name)])

    # =============================
    # dicom to nii, remove negative CTs
    # ============================
    # out_dcm2nii_dir = "data/orig_dcm_data/dcm2nii"
    # for i in tqdm(range(len(paired_patient_names))):
    #     out_t1_nii_path = join(out_dcm2nii_dir, "mr", paired_patient_names[i] + "_0000.nii.gz")
    #     out_ct_nii_path = join(out_dcm2nii_dir, "ct", paired_patient_names[i] + "_0000.nii.gz")

    #     dicom_to_nifti(paired_patient_t1_sequence_dirs[i], out_t1_nii_path)
    #     dicom_to_nifti(paired_patient_ct_sequence_dirs[i], out_ct_nii_path)

    # # remove negative ct, copy normal ct and paired mr to new dir, reorient mr
    # dcm2nii_dir = "data/orig_dcm_data/dcm2nii"
    # preprocessed_dir = "data/orig_dcm_data/preprocessed/non_negative_nii"
    # ct_nii_paths = list(glob(join(dcm2nii_dir, "ct", "*.nii.gz")))
    # for ct_nii_path in tqdm(ct_nii_paths):
    #     ct_itk = sitk.ReadImage(ct_nii_path)
    #     ct_arr = sitk.GetArrayFromImage(ct_itk)
    #     if ct_arr.max() < 0:
    #         continue

    #     ct_fn = basename(ct_nii_path)
    #     out_ct_nii_path = join(preprocessed_dir, "ct", ct_fn)
    #     out_mr_nii_path = join(preprocessed_dir, "mr", ct_fn)
    #     os.makedirs(dirname(out_ct_nii_path), exist_ok=True)
    #     os.makedirs(dirname(out_mr_nii_path), exist_ok=True)

    #     reorient_RAS(ct_nii_path, out_ct_nii_path)
    #     mr_preprocessing(join(dcm2nii_dir, "mr", ct_fn), out_mr_nii_path)

    # ==================
    # Shape and Spacing
    # ==================
    # # collect shape and spacing for mr and ct
    # normal_ct_dir = "data/orig_dcm_data/dcm2nii/normal_ct"
    # normal_ct_nii_paths = list(glob(join(normal_ct_dir, "*.nii.gz")))
    # normal_ct_pnames = [basename(nii).split(".nii")[0] for nii in normal_ct_nii_paths]

    # mr_dir_path = "data/orig_dcm_data/preprocessed/mr"
    
    # import pandas as pd
    # spacing_shape_df = pd.DataFrame(columns=[
    #     "pname", 
    #     "mr_spacing_x", "mr_spacing_y", "mr_spacing_z", 
    #     "mr_shape_x", "mr_shape_y", "mr_shape_z",
    #     "ct_spacing_x", "ct_spacing_y", "ct_spacing_z",
    #     "ct_shape_x", "ct_shape_y", "ct_shape_z"])
    # for pname in tqdm(normal_ct_pnames):
    #     ct_nii_path = join(normal_ct_dir, pname + ".nii.gz")
    #     mr_nii_path = join(mr_dir_path, pname + ".nii.gz")

    #     ct_itk = sitk.ReadImage(ct_nii_path)
    #     mr_itk = sitk.ReadImage(mr_nii_path)

    #     spacing_shape_df.loc[len(spacing_shape_df)] = [
    #         pname,
    #         mr_itk.GetSpacing()[0], mr_itk.GetSpacing()[1], mr_itk.GetSpacing()[2],
    #         mr_itk.GetSize()[0], mr_itk.GetSize()[1], mr_itk.GetSize()[2],
    #         ct_itk.GetSpacing()[0], ct_itk.GetSpacing()[1], ct_itk.GetSpacing()[2],
    #         ct_itk.GetSize()[0], ct_itk.GetSize()[1], ct_itk.GetSize()[2]
    #     ]
    # spacing_shape_df.to_csv("data/orig_dcm_data/paired_mrct_spacing_shape.csv", index=False)


    # ==================
    # MR spacing resampling
    # ==================
    # # resample ...


    # ==================
    # Bet MR neck
    # ==================
    # # bet MR neck ...
    # normal_ct_dir = "data/orig_dcm_data/dcm2nii/normal_ct"
    # normal_ct_nii_paths = list(glob(join(normal_ct_dir, "*.nii.gz")))
    # normal_ct_pnames = [basename(nii).split(".nii")[0] for nii in normal_ct_nii_paths]

    # mr_dir_path = "data/orig_dcm_data/preprocessed/mr"
    # betneck_mr_dir_path = "data/orig_dcm_data/preprocessed/mr_betneck"
    # mr_paths = [join(mr_dir_path, pname + ".nii.gz") for pname in normal_ct_pnames]
    # mr_betneck_paths = [join(betneck_mr_dir_path, pname + ".nii.gz") for pname in normal_ct_pnames]

    # import multiprocessing as mp
    # pool = mp.Pool(mp.cpu_count())
    # pool.starmap(remove_neck_slices, zip(mr_paths, mr_betneck_paths))
    # pool.close()

    # ====================================================
    # # bet get brain mask for registration (ct to mr ...)
    # ====================================================
    # normal_ct_dir = "data/orig_dcm_data/dcm2nii/masked_ct"
    # betneck_mr_dir = "/data/dingsd/mr2ct/data/orig_dcm_data/preprocessed/mr_betneck"

    # normal_ct_nii_paths = list(glob(join(normal_ct_dir, "*.nii.gz")))
    # normal_ct_nii_paths = [nii for nii in normal_ct_nii_paths if "mask" not in basename(nii)]
    # normal_ct_pnames = [basename(nii).split(".nii")[0] for nii in normal_ct_nii_paths]

    # bet_dir = "data/orig_dcm_data/preprocessed/bet_nii"
    # import multiprocessing as mp
    # pool = mp.Pool(mp.cpu_count())
    # for pname in tqdm(normal_ct_pnames):
    #     ct_nii_path = join(normal_ct_dir, pname + ".nii.gz")
    #     mr_nii_path = join(betneck_mr_dir, pname + ".nii.gz")

    #     bet_ct_nii_path = join(bet_dir, "ct", pname + ".nii.gz")
    #     bet_mr_nii_path = join(bet_dir, "mr", pname + ".nii.gz")
        
    #     pool.apply_async(skull_stripping, args=(ct_nii_path, bet_ct_nii_path))
    #     pool.apply_async(skull_stripping, args=(mr_nii_path, bet_mr_nii_path))
    # pool.close()
    # pool.join()

    # ==================
    # registration
    # ==================
    # bet_dir = "data/orig_dcm_data/preprocessed/bet_nii"
    # ct_dir = "data/orig_dcm_data/dcm2nii/masked_ct"
    # mr_dir = "/data/dingsd/mr2ct/data/orig_dcm_data/preprocessed/mr_betneck"

    # ct_nii_paths = list(glob(join(ct_dir, "*.nii.gz")))
    # file_names = [basename(nii).split(".nii")[0] for nii in ct_nii_paths if "mask" not in basename(nii)]

    # reg_out_dir = "data/orig_dcm_data/preprocessed/ct_reg2_mr_betneck"
    # import multiprocessing as mp
    # pool = mp.Pool(mp.cpu_count())
    # for fn in tqdm(file_names):
    #     ct_nii_path = join(ct_dir, fn + ".nii.gz")
    #     mr_nii_path = join(mr_dir, fn + ".nii.gz")

    #     # registration
    #     bet_ct_nii_path = join(bet_dir, "ct", fn + ".nii.gz")
    #     bet_mr_nii_path = join(bet_dir, "mr", fn + ".nii.gz")

    #     pool.apply_async(registration, args=(bet_mr_nii_path, bet_ct_nii_path, ct_nii_path, reg_out_dir))
    
    # pool.close()
    # pool.join()


    # collect brain mask
    mr_dir = "/data/dingsd/mr2ct/data/orig_dcm_data/preprocessed/mr_betneck"
    niis = list(glob(join(mr_dir, "*.nii.gz")))

    import pandas as pd
    spacing_shape_df = pd.DataFrame(columns=[
        "pname", 
        "mr_crop_x", "mr_crop_y", "mr_crop_z"])

    import nibabel as nib
    for nii in tqdm(niis):
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii))
        # img_arr = nib.load(nii).get_fdata()

        mask_voxel_coords = np.where(img_arr > 0)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

        print(bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0], bbox[2][1] - bbox[2][0],)

        spacing_shape_df.loc[len(spacing_shape_df)] = [
            basename(nii).split(".nii")[0],
            bbox[1][1] - bbox[1][0], bbox[2][1] - bbox[2][0], bbox[0][1] - bbox[0][0]
        ]
    spacing_shape_df.to_csv("data/orig_dcm_data/preprocessed/mr_foreground_shape.csv", index=False)
        