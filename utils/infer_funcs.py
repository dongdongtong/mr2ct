import sys, os, copy
import torch
from monai.config import print_config
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
    ToTensor, HistogramNormalize, ResizeWithPadOrCrop, CropForeground,
)

from utils.netdef import ShuffleUNet
import numpy as np
import ants
import matplotlib.pyplot as plt
import matplotlib.colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel as nib

import monai
from monai.data import MetaTensor


# define patch-based inference function
# patches are 256 x 256 x 32 (i.e. 32 slices in axial/transverse plane)
def do_inference_3D_sliding_window(net, x, patch_size=(384,384,48)):
    sliding_window_infer = monai.inferers.inferer.SlidingWindowInferer(
        roi_size=patch_size, sw_batch_size=1, overlap=0.5
    )

    out_tensor = sliding_window_infer(x, net)

    return out_tensor

# Prepare T1w MRI: bias correction and create head mask
def do_prep_mr(img_file):
    img = ants.image_read(img_file)
    img_n4 = ants.n4_bias_field_correction(img)
    img_tmp = img_n4.otsu_segmentation(k=3) # otsu_segmentation
    img_tmp = ants.multi_label_morphology(img_tmp, 'MD', 2) # dilate 2
    img_tmp = ants.smooth_image(img_tmp, 3) # smooth 3
    img_tmp = ants.threshold_image(img_tmp, 0.5) # threshold 0.5
    img_mask = ants.get_mask(img_tmp)
    img_out = ants.multiply_images(img_n4, img_mask)

    out_path = os.path.splitext(img_file)[0] + '_prep.nii.gz'
    ants.image_write(img_out,(os.path.splitext(img_file)[0] + '_prep.nii'))
    print('Prepare MR image done, output saved to: {}'.format((os.path.splitext(img_file)[0] + '_prep.nii')))

    return out_path


from monai.transforms.utils import allow_missing_keys_mode
def inverse_transforms(
        data: MetaTensor, 
        valid_tfs: Compose, 
        loaded_data: MetaTensor, 
        orig_img_path: str,
        save_path: str):
    # create metatensor inheriting transforms operations from loaded_data
    data = MetaTensor(data, affine=loaded_data.affine, applied_operations=loaded_data.applied_operations)

    # with allow_missing_keys_mode(valid_tfs):
    inverted_data = valid_tfs.inverse(data)
    
    inverted_data = inverted_data.squeeze(0)  # remove channel dimension, to shape (W, H, D)

    # save inverted_img as nii.gz
    img_nib = nib.load(orig_img_path)
    aff, header = img_nib.affine, img_nib.header
    volume = inverted_data.cpu().numpy().astype(dtype=img_nib.get_fdata().dtype)

    # print(volume.min(), volume.max())
    nifty = nib.Nifti1Image(volume, aff, header)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib.save(nifty, save_path)


# load MRIs
def do_mr_to_pct(input_mr_file, output_pct_file, saved_model, device, prep_t1, **kwargs):
    # set network parameters and check net shape
    transformer_layers = kwargs.get("transformer_layers", 0)
    img_size = kwargs.get("img_size", (448, 448, 56))
    sliding_window_infer = kwargs.get("sliding_window_infer", False)

    net = ShuffleUNet(dimensions=3, in_channels=1, out_channels=1,
        channels=(32, 64, 128, 256, 384), strides=(2, 2, 2, 2),
        kernel_size = 3, up_kernel_size = 3, num_res_units=0, 
        transformer_layers=transformer_layers, img_size=img_size
    )
    # n = torch.rand(1, 1, 256, 256, 32)
    # print(net(n).shape)  # should be [1, 1, 256, 256, 32]

    # specify transforms
    transform_test = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(0.5, 0.5, 3.0)),
            CropForeground(),   # crop to align the brain center into the image center
            ResizeWithPadOrCrop(spatial_size=(448, 448, 64), mode="edge"),
            HistogramNormalize(min=-1, max=1.0),
        ]
    )

    # load images
    print('Loading MR image: {}'.format(input_mr_file))
    if prep_t1:
        print('Preparing MR image: bias correction and masking...')
        pre_mr_path = do_prep_mr(input_mr_file)
    else:
        pre_mr_path = input_mr_file
    mr_tensor = transform_test(pre_mr_path).unsqueeze(0)

    net.to(device)
    net.load_state_dict(saved_model)
    net.eval()

    print('Running MR to pCT...')
    with torch.no_grad():
        if not sliding_window_infer:
            pesudo_ct_tensor = net(mr_tensor.cuda())
        else:
            pesudo_ct_tensor = do_inference_3D_sliding_window(net, mr_tensor.cuda(), patch_size=(384, 384, 48))
    
    # invert transform using monai
    print('Invert the pCT to the original MR space...')
    mr_tensor = mr_tensor.squeeze(0)
    pesudo_ct_tensor = pesudo_ct_tensor.squeeze(0)
    inverse_transforms(
        pesudo_ct_tensor, transform_test, mr_tensor, pre_mr_path, output_pct_file
    )

    print('MR to pCT done, output saved to: {}'.format(output_pct_file))
    print('')
    print('Please inspect your output.')