import sys
# sys.path.append("/home/dingsd/workstation/SynthSeg")
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
for batch_idx, batch in enumerate(tqdm(train_ds)):
    # ============
    # ====for plane monai loader========
    # ============
    mr_img = batch['mr_image']
    ct_img = batch['ct_image']

    mr_img_path = batch['mr_img_path']
    ct_img_path = batch['ct_img_path']
    print(mr_img.shape, ct_img.shape)

    pname = basename(mr_img_path).split(".")[0]

    visualize_img_seg_overlay(
        mr_img.squeeze().permute(2, 1, 0).numpy()[:, ::-1, :], # (W, H, D) -> (D, H, W)
        f"valid_visual/{pname}/mr.png"
    )
    visualize_img_seg_overlay(
        ct_img.squeeze().permute(2, 1, 0).numpy()[:, ::-1, :], # (W, H, D) -> (D, H, W)
        f"valid_visual/{pname}/ct.png"
    )


# from models.generators.dyunet import DynUNet
# from models.modules.upsample import UnetUpBlock
# import yaml
# import torch


# with open("configs/mr2ct.yml") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

# strides = config['strides']
# filters = config['filters'][:len(strides)]
# kernel_size = config['kernel_size']
# upsample_kernel_size = strides[1:]
# model = DynUNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     kernel_size=kernel_size,
#     upsample_kernel_size=upsample_kernel_size,
#     strides=strides,
#     filters=filters,
#     deep_supervision=False,
#     upsample_block=UnetUpBlock,
# ).cuda()

# dummy_input = torch.randn(1, 1, 160, 160, 24).cuda()

# print(model(dummy_input).shape)


# import torch
# import torch.nn as nn

# upsample_stride = [2, 2, 1]
# upsample = nn.Upsample(scale_factor=tuple(upsample_stride), mode="trilinear", align_corners=False)

# dummy_input = torch.randn(1, 1, 80, 80, 16)

# print(upsample(dummy_input).shape)