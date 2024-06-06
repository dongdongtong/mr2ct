import SimpleITK as sitk
import ants
import os
from os.path import join, basename, dirname
import sys
sys.path.append(dirname(dirname(os.path.abspath(__file__))))
import numpy as np
import multiprocessing
import cv2
from preprocess.brain_extractor import BrainExtractor


# Prepare T1w MRI: bias correction and create head mask
def do_prep_mr(img_file, out_file):
    os.makedirs(dirname(out_file), exist_ok=True)
    print(img_file)

    img = ants.image_read(img_file)
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


# Prepare T1w MRI: bias correction and create head mask
def do_prep_ct(img_file, out_file):
    os.makedirs(dirname(out_file), exist_ok=True)

    # n4 bias correction
    img_ants = ants.n4_bias_field_correction(ants.image_read(img_file))
    ants.image_write(img_ants, out_file)

    img_itk = sitk.ReadImage(out_file)

    # extract head mask
    pid = basename(img_file).split(".nii")[0][:-5]
    brain_mask_path = join("data/ct_reg2_mr_betneck/original_ct_betneck_head_mask", pid + ".nii.gz")
    brain_mask = sitk.ReadImage(brain_mask_path)

    # get masked image
    arr = sitk.GetArrayFromImage(img_itk)
    brain_mask_arr = sitk.GetArrayFromImage(brain_mask)

    arr[brain_mask_arr == 0] = -1024
    img_out = sitk.GetImageFromArray(arr)
    img_out.CopyInformation(img_itk)

    os.makedirs(dirname(out_file), exist_ok=True)
    sitk.WriteImage(img_out, out_file)
    sitk.WriteImage(brain_mask, out_file.replace('.nii.gz', '_brainmask.nii.gz'))

    print('Prepare CT image done, output saved to: {}'.format(out_file))


def do_prep_aligned_ct_mr(ct_file, out_root_dir):
    itk_ct = sitk.ReadImage(ct_file)
    ct_arr = sitk.GetArrayFromImage(itk_ct)
    if ct_arr.max() < 0:  # for those ct images with all negative values, they are all preprocessed by machines thus not available for mr2ct translation
        return
    
    fn = basename(ct_file)
    mr_file = join(dirname(dirname(ct_file)), "mr", fn)

    out_ct_file = join(out_root_dir, "ct", fn)
    out_mr_file = join(out_root_dir, "mr", fn)

    if not os.path.exists(out_ct_file) or not os.path.exists(out_ct_file.replace('.nii.gz', '_brainmask.nii.gz')):
        do_prep_ct(ct_file, out_ct_file)
    else:
        print(f"CT preprocess already processed {fn}")
    ct_img_pre = ants.image_read(out_ct_file)
    ct_brainmask = ants.image_read(out_ct_file.replace('.nii.gz', '_brainmask.nii.gz'))
    
    if not os.path.exists(out_mr_file) or not os.path.exists(out_mr_file.replace('.nii.gz', '_brainmask.nii.gz')):
        do_prep_mr(mr_file, out_mr_file)
    else:
        print(f"MR preprocess already processed {fn}")
    mr_img_pre = ants.image_read(out_mr_file)

    out_dict = ants.registration(
        fixed=mr_img_pre,
        moving=ct_img_pre,
        type_of_transform="Similarity",
        aff_metric='mattes',
    )

    # # apply the affine matrix to the preprocessed ct image
    # reged_ct = ants.apply_transforms(
    #     fixed=mr_img_pre, 
    #     moving=ct_img_pre, 
    #     transformlist=out_dict['fwdtransforms'],
    #     interpolator='bSpline', defaultvalue=-1024)

    # reg_ct_dir = join(out_root_dir, "ct_reg2_mr")
    # os.makedirs(reg_ct_dir, exist_ok=True)
    # reg_ct_file = join(reg_ct_dir, fn)
    # ants.image_write(reged_ct, reg_ct_file)

    # apply the affine matrix to the brain mask
    reged_ct_brainmask = ants.apply_transforms(
        fixed=mr_img_pre, 
        moving=ct_brainmask, 
        transformlist=out_dict['fwdtransforms'],
        interpolator='nearestNeighbor', defaultvalue=0)
    
    reg_ct_brainmask_dir = join(out_root_dir, "ct_reg2_mr_brainmask")
    os.makedirs(reg_ct_brainmask_dir, exist_ok=True)
    reg_ct_brainmask_file = join(reg_ct_brainmask_dir, fn.replace('.nii.gz', '_brainmask.nii.gz'))
    ants.image_write(reged_ct_brainmask, reg_ct_brainmask_file)


def main():
    ct_data_dir = "data/pipeline_niigz_betneck/ct"
    out_root_dir = "data/ct_reg2_mr_betneck"

    # Obtain nii files in the ct_data_dir
    ct_files = [os.path.join(ct_data_dir, file) for file in os.listdir(ct_data_dir) if file.endswith('.nii.gz')]

    # Execute do_prep_ct in parallel using multi processing
    pool = multiprocessing.Pool()
    for ct_file in ct_files:
        pool.apply_async(do_prep_aligned_ct_mr, args=(ct_file, out_root_dir))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()

    # t2_path = "data/pipeline_niigz_betneck/mr/FANGLIRONG_0000.nii.gz"
    # pre_t2_path = "./FANGLIRONG_0000_otsuants.nii.gz"
    # do_prep_mr(t2_path, pre_t2_path)

    # # Example usage
    # input_image_path = "data/pipeline_niigz_betneck/ct/LUFA_0000.nii.gz"  # Replace with the path to your input CT image
    # output_image_path = "./LUFA_0000_mine.nii.gz"  # Specify the path for the output segmented image

    # do_prep_ct(input_image_path, output_image_path)


    # # do reg
    # # preprocess ct for better registration, we found that it's none of registration's bussiness
    # ct_img = sitk.ReadImage("ct_MENGKE_0000_otsu_ants_erode.nii.gz")
    # ct_arr = sitk.GetArrayFromImage(ct_img)
    # ct_arr = np.clip(ct_arr, 0, 100)
    # clipped_ct_img = sitk.GetImageFromArray(ct_arr)
    # clipped_ct_img.CopyInformation(ct_img)
    # sitk.WriteImage(clipped_ct_img, "clipped_value_ct_ants_otsu.nii.gz")
# mr_img = ants.image_read("t2_MENGKE_0000_prep.nii.gz")
# ct_img = ants.image_read("ct_MENGKE_0000_otsu_ants_erode.nii.gz")
# out_dict = ants.registration(
#     fixed=mr_img,
#     moving=ct_img,
#     type_of_transform="Similarity",
#     aff_metric='mattes',
#     # syn_metric='meansquares'
# )
# # dict containing follow key/value pairs:
# #     warpedmovout: Moving image warped to space of fixed image.
# #     warpedfixout: Fixed image warped to space of moving image.
# #     fwdtransforms: Transforms to move from moving to fixed image.
# #     invtransforms: Transforms to move from fixed to moving image.

# ants.image_write(out_dict['warpedmovout'], filename="ct2mr.nii.gz")
# ants.image_write(out_dict['warpedfixout'], filename="mr2ct.nii.gz")
# ants.image_write(out_dict['fwdtransforms'], filename="mr2ct_out.nii.gz")

# ants

# reg_aladin -ref ./BICHENQIONG_0000_otsuants.nii.gz -flo BICHENQIONG_0000_otsu_ants_erode.nii.gz –rigOnly -res BICHENQIONG_ct2mr.nii.gz

# flirt -in ct_MENGKE_0000_otsu_ants_erode.nii.gz -ref t2_MENGKE_0000_prep.nii.gz -out ct2mr_fsl.nii.gz -omat T2F_fsl.mat -bins 256 -cost mutualinfo -searchcost mutualinfo -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -interp spline -sincwidth 7 -sincwindow hanning