from functools import partial
import SimpleITK as sitk
import ants
import os
from os.path import join, basename, dirname
import sys
sys.path.append(dirname(dirname(os.path.abspath(__file__))))
import numpy as np
import multiprocessing
import cv2
from glob import glob

from preprocess_paired_mr_ct.skull_strip import skull_stripping

from utils.logger import setup_logger

logging_path = "./preprocess_and_reg.log"
setup_logger(logging_path)
SYNTHSTRIP_MODEL_PATH = "data/synthstrip/synthstrip.1.pt"

def do_prep_mr(img_file, out_file):
    "- N4 Bias correction + OTSU extract head mask (brain + skull) + mask out non-head values"
    "- skull stripping for more accurate registration"
    os.makedirs(dirname(out_file), exist_ok=True)
    print(f"do_prep_mr Preprocessing {img_file}")

    img = ants.image_read(img_file)
    # N4 bias correction
    img_n4 = ants.n4_bias_field_correction(img)

    # OTSU extract brain mask
    img_tmp = img_n4.otsu_segmentation(k=3) # otsu_segmentation
    img_tmp = ants.multi_label_morphology(img_tmp, 'MD', 2) # dilate 2
    img_tmp = ants.smooth_image(img_tmp, 3) # smooth 3
    img_tmp = ants.threshold_image(img_tmp, 0.5) # threshold 0.5
    img_mask = ants.get_mask(img_tmp)
    img_out = ants.multiply_images(img_n4, img_mask)   # mask out non-head values

    # save preprocessed image and head mask
    os.makedirs(dirname(out_file), exist_ok=True)
    ants.image_write(img_out, out_file)
    ants.image_write(img_mask, out_file.replace('.nii.gz', '_headmask.nii.gz'))
    print('Prepare MR head image done, output saved to: {}'.format(out_file))
    print('Prepare MR head mask done, output saved to: {}'.format(out_file.replace('.nii.gz', '_headmask.nii.gz')))


def do_prep_ct(img_file, out_file, head_mask_path):
    "- N4 Bias correction + brain mask extraction + mask out non-head values"
    "- skull stripping for more accurate registration"
    os.makedirs(dirname(out_file), exist_ok=True)
    print(f"do_prep_ct Preprocessing {img_file}")

    # n4 bias correction
    img_ants = ants.n4_bias_field_correction(ants.image_read(img_file))
    ants.image_write(img_ants, out_file)

    img_itk = sitk.ReadImage(out_file)

    # head mask has already been acquired using nnU-Net in advance
    head_mask = sitk.ReadImage(head_mask_path)

    # get masked image
    arr = sitk.GetArrayFromImage(img_itk)
    head_mask_arr = sitk.GetArrayFromImage(head_mask)

    arr[head_mask_arr == 0] = -1024
    img_out = sitk.GetImageFromArray(arr)
    img_out.CopyInformation(img_itk)

    os.makedirs(dirname(out_file), exist_ok=True)
    sitk.WriteImage(img_out, out_file)
    sitk.WriteImage(head_mask, out_file.replace('.nii.gz', '_headmask.nii.gz'))

    print('Prepare CT head image done, output saved to: {}'.format(out_file))
    print('Prepare CT head mask done, output saved to: {}'.format(out_file.replace('.nii.gz', '_headmask.nii.gz')))


def bet_nii(img_file, out_file, synthstrip_model_path=SYNTHSTRIP_MODEL_PATH):
    skull_stripping(img_file, out_file, model_path=synthstrip_model_path)


def do_registration_ct2mr(
        bet_ct_file, bet_mr_file, 
        pre_head_ct_file, pre_head_mr_file, 
        reg_ct_file):
    "1. registration from bet ct to bet mr; 2. apply the affine matrix to the head ct (maybe head_mask and brain_mask)."
    itk_head_ct = sitk.ReadImage(pre_head_ct_file)
    head_ct_arr = sitk.GetArrayFromImage(itk_head_ct)
    if head_ct_arr.max() < 0:  # for those ct images with all negative values, they are all preprocessed by machines thus not available for mr2ct translation
        return

    bet_ct_img_pre = ants.image_read(bet_ct_file)
    head_ct_img_pre = ants.image_read(pre_head_ct_file)
    # ct_brainmask = ants.image_read(bet_ct_file.replace('.nii.gz', '_brainmask.nii.gz'))
    # ct_headmask = ants.image_read(join(dirname(pre_head_ct_file), "ct_headmask", basename(pre_head_ct_file)))
    
    bet_mr_img_pre = ants.image_read(bet_mr_file)
    head_mr_img_pre = ants.image_read(pre_head_mr_file)

    print(f"Registration from {bet_ct_file} to {bet_mr_file} starts...")
    out_dict = ants.registration(
        fixed=bet_mr_img_pre,
        moving=bet_ct_img_pre,
        type_of_transform="Similarity",
        aff_metric='mattes',
    )

    # apply the affine matrix to the head ct
    reged_ct = ants.apply_transforms(
        fixed=head_mr_img_pre, 
        moving=head_ct_img_pre, 
        transformlist=out_dict['fwdtransforms'],
        interpolator='bSpline', defaultvalue=-1024)  # we use -1024 as the background value after interpolation

    os.makedirs(dirname(reg_ct_file), exist_ok=True)
    ants.image_write(reged_ct, reg_ct_file)

    # you may also want to apply the affine matrix to the head mask or brain mask
    # apply the affine matrix to the head mask
    # reged_ct_headmask = ants.apply_transforms(
    #     fixed=head_mr_img_pre, 
    #     moving=ct_headmask, 
    #     transformlist=out_dict['fwdtransforms'],
    #     interpolator='nearestNeighbor', defaultvalue=0)
    
    # reg_ct_headmask_file = reg_ct_file.replace('.nii.gz', '_headmask.nii.gz')
    # ants.image_write(reged_ct_headmask, reg_ct_headmask_file)

    # # apply the affine matrix to the brain mask
    # reged_bet_ct_brainmask = ants.apply_transforms(
    #     fixed=head_mr_img_pre, 
    #     moving=ct_brainmask, 
    #     transformlist=out_dict['fwdtransforms'],
    #     interpolator='nearestNeighbor', defaultvalue=0)
    
    # reg_ct_brainmask_file = reg_ct_file.replace('.nii.gz', '_brainmask.nii.gz')
    # ants.image_write(reged_bet_ct_brainmask, reg_ct_brainmask_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess paired MR and CT images, including skull stripping and registration.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing CT and MR images.')
    args = parser.parse_args()
    root_dir = args.root_dir
    ct_dir = join(root_dir, "ct")  # a list of files ending with .nii.gz
    mr_dir = join(root_dir, "mr")  # a list of files ending with .nii.gz

    # first step: extract head masks for ct files using nnunet (See nnunetv2 results Dataset141_CTHeadMask)
    ct_headmask_dir = join(root_dir, "ct_headmask")
    os.makedirs(ct_headmask_dir, exist_ok=True)
    print("Extracting head masks for CT files using nnUNet...")
    # os.system(f"nnUNetv2_predict -i {ct_dir} -o {ct_headmask_dir} -d 141 -c 3d_fullres -tr nnUNetTrainer_30Epoch -f 0 1 2 3 4")    # this will remove image's 0000 in basename
    print(f"Head masks for CT files extracted and saved to: {ct_headmask_dir}")
    # we need to rename the files ending with "_0000.nii.gz"
    print("Renaming head mask files to add '_0000' suffix...")
    for file in glob(join(ct_headmask_dir, "*.nii.gz")):
        if file.endswith("_0000.nii.gz"):
            continue
        new_file = join(ct_headmask_dir, basename(file).replace('.nii.gz', '_0000.nii.gz'))
        os.rename(file, new_file)

    # second step: preprocess ct/mr files (skull stripping)
    mr_files = list(glob(join(mr_dir, "*.nii.gz")))
    ct_files = list(glob(join(ct_dir, "*.nii.gz")))

    for mr_file in mr_files:
        pre_mr_file = join(root_dir, "pre_head_mr", basename(mr_file))
        do_prep_mr(mr_file, pre_mr_file)   # both pre_head_mr and headmask will be saved

        # skull stripping
        bet_mr_file = join(root_dir, "bet_mr", basename(mr_file))
        bet_nii(pre_mr_file, bet_mr_file)  # both brain and brainmask will be saved
    
    for ct_file in ct_files:
        pre_ct_file = join(root_dir, "pre_head_ct", basename(ct_file))
        ct_head_mask_file = join(ct_headmask_dir, basename(ct_file))
        do_prep_ct(ct_file, pre_ct_file, ct_head_mask_file)

        # skull stripping
        bet_ct_file = join(root_dir, "bet_ct", basename(ct_file))
        bet_nii(pre_ct_file, bet_ct_file)  # both brain and brain_mask will be saved
    
    # registration
    pre_head_ct_files = list(glob(join(root_dir, "pre_head_ct", "*.nii.gz")))

    # remove mask files in the list
    pre_head_ct_files = [file for file in pre_head_ct_files if 'headmask' not in file and 'brainmask' not in file]
    
    # Do registration for each ct file and also remove those ct files with all negative values
    print("Start registration...")
    print(f"Total ct files to register: {len(pre_head_ct_files)}")
    out_root_dir = join(root_dir, "headct_reg2_mr")
    for pre_head_ct_file in pre_head_ct_files:
        bet_ct_file = join(root_dir, "bet_ct", basename(pre_head_ct_file))
        bet_mr_file = join(root_dir, "bet_mr", basename(pre_head_ct_file))
        pre_head_mr_file = join(root_dir, "pre_head_mr", basename(pre_head_ct_file))

        reg_ct_file = join(out_root_dir, basename(pre_head_ct_file))

        do_registration_ct2mr(bet_ct_file, bet_mr_file, pre_head_ct_file, pre_head_mr_file, reg_ct_file)

    # ==============================================================================
    # the final files are ct files in "headct_reg2_mr" and mr files in "pre_head_mr"
    # ==============================================================================

    # you may want to use the following code for parallel processing
    # # Obtain nii files in the mr dir
    # mr_files = list(glob(join(mr_dir, "*.nii.gz")))

    # # Execute do_prep_ct in parallel using multi processing
    # pool = multiprocessing.Pool()
    # for mr_file in mr_files:
    #     pre_mr_file = join(root_dir, "pre_head_mr", basename(mr_file))
    #     pool.apply_async(do_prep_mr, args=(mr_file, pre_mr_file))
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    main()