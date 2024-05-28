import sys
sys.path.append("/opt/dingsd/cyclegan_mr2ct")
from os.path import dirname, basename, join
import numpy as np
import SimpleITK as sitk
from glob import glob
import json
import os
from tqdm import tqdm
from shutil import copyfile
from skimage.measure import label
import nibabel as nib
import cv2
import ants
from monai.data.meta_tensor import MetaTensor

from datasets.create_dataloaders import create_loader_for_debug


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
def visualize_img_seg_overlay(image_data: np.ndarray, output_path):
    # Normalize image data to [0, 1] range
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Calculate an appropriate number of rows and columns for the grid
    num_slices = image_data.shape[0]
    num_rows = 8  # You can adjust this number to your preference
    num_cols = (num_slices + num_rows - 1) // num_rows

    # Calculate figsize to maintain aspect ratio
    aspect_ratio = image_data.shape[1] / image_data.shape[2]
    fig_width = 15
    fig_height = fig_width * aspect_ratio * num_rows / num_cols
    figsize = (fig_width, fig_height)

    # Set up the figure and subplots
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.1)

    # Iterate through axial slices and visualize
    for i, ax in enumerate(grid):
        if i < num_slices:
            ax.imshow(image_data[i, :, :], cmap='gray')
            # ax.imshow(segmentation_data[i, :, :], cmap='jet', alpha=0.5)

            ax.axis('off')

    # Save the figure as a PNG file
    os.makedirs(dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print(f"Visualization saved as {output_path}")


config = "configs/mr2ct.yml"

loader = create_loader_for_debug(config)

train_ds = loader['valid_ds']

from tqdm import tqdm
max_z = 0
monai_preprocessed_nii_out_dir = "data/monai_preprocessed_nii/pipeline_niigz_betneck"
monai_preprocessed_bm_out_dir = "data/monai_preprocessed_nii/resampled_pipeline_niigz_betneck_brainsegs/mr"
ants_reg_out_dir = "data/ants_reg/pipeline_niigz_betneck"
type_of_transform = "Affine"  # "Rigid", "Affine", "Similarity", "SyN"
for batch_idx, batch in enumerate(tqdm(train_ds)):
    # ============
    # ====for plane monai loader========
    # ============
    mr_img = batch['mr_image']
    ct_img = batch['ct_image']
    mr_bm_img = batch['mr_brainmask']

    mr_img_path = batch['mr_img_path']
    ct_img_path = batch['ct_img_path']
    print(mr_img.shape, ct_img.shape)

    pname = basename(mr_img_path).split(".")[0]

    # ==============================
    # ==== monai preprocess ========
    # ==============================
    mr_img_affine = batch['mr_image'].affine
    ct_img_affine = batch['ct_image'].affine
    mr_bm_img_affine = batch['mr_brainmask'].affine
    mr_img_nib = nib.Nifti1Image(mr_img.numpy()[0], mr_img_affine)
    ct_img_nib = nib.Nifti1Image(ct_img.numpy()[0], ct_img_affine)
    mr_bm_img_nib = nib.Nifti1Image(mr_bm_img.numpy()[0], mr_bm_img_affine)

    monai_preprocessed_mr_nii_path = join(monai_preprocessed_nii_out_dir, "mr", f"{pname}.nii.gz")
    monai_preprocessed_ct_nii_path = join(monai_preprocessed_nii_out_dir, "ct", f"{pname}.nii.gz")
    monai_preprocessed_bm_nii_path = join(monai_preprocessed_bm_out_dir, f"{pname}.nii.gz")
    os.makedirs(dirname(monai_preprocessed_mr_nii_path), exist_ok=True)
    os.makedirs(dirname(monai_preprocessed_ct_nii_path), exist_ok=True)
    os.makedirs(dirname(monai_preprocessed_bm_nii_path), exist_ok=True)
    nib.save(mr_img_nib, monai_preprocessed_mr_nii_path)
    nib.save(ct_img_nib, monai_preprocessed_ct_nii_path)
    nib.save(mr_bm_img_nib, monai_preprocessed_bm_nii_path)


    # ==============================
    # ==== ants registration ========
    # ==============================
    # type_of_transform:
    # "Rigid": rotation and translation.
    # "Affine": Affine transformation: rotation + translation + scaling.
    # "Similarity": Similarity transformation: scaling, rotation and translation.
    # "SyN": Symmetric normalization: Affine + deformable transformation, with mutual information as optimization metric.
    moving_img = ants.image_read(monai_preprocessed_mr_nii_path)
    ref_nii_path = ants.image_read(monai_preprocessed_ct_nii_path)
    mytx = ants.registration(
        fixed=moving_img, 
        moving=ref_nii_path, 
        type_of_transform=type_of_transform)

    reged_ct_nii_path = join(ants_reg_out_dir, "ct", f"{pname}.nii.gz")
    os.makedirs(dirname(reged_ct_nii_path), exist_ok=True)
    warped_moving = mytx['warpedmovout']
    warped_moving.to_file(reged_ct_nii_path)

    # output_mat_path = join(dirname(dirname(output_nii_path)), "reg_mat", basename(output_nii_path).split(".nii")[0] + ".mat")
    # os.makedirs(dirname(output_mat_path), exist_ok=True)
    # warped_mat = mytx['fwdtransforms']
    # copyfile(warped_mat[0], output_mat_path)
    

    # moved_target_seg = ants.apply_transforms(fixed=source_seg, moving=target_seg, transformlist=mytx['fwdtransforms'])
    # moved_target_seg.to_file("moved_30_to_26_seg.nii.gz")

    break

    # visualize_img_seg_overlay(
    #     mr_img.squeeze().permute(2, 1, 0).numpy()[:, ::-1, :], # (W, H, D) -> (D, H, W)
    #     f"valid_visual/{pname}/mr.png"
    # )
    # visualize_img_seg_overlay(
    #     ct_img.squeeze().permute(2, 1, 0).numpy()[:, ::-1, :], # (W, H, D) -> (D, H, W)
    #     f"valid_visual/{pname}/ct.png"
    # )