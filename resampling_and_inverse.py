import os

import torch
import torch.nn.functional as F

import numpy as np
import nibabel as nib

from monai.data import MetaTensor
from monai.transforms import Compose
from monai.transforms.utils import allow_missing_keys_mode

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
    ToTensor, HistogramNormalize, ResizeWithPadOrCrop, CropForeground,
)
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype
from monai.transforms.utils import generate_spatial_bounding_box, compute_divisible_spatial_size

from monai.transforms.inverse import InvertibleTransform


# allow the output of the model transformed inversely into the original space of the model input.
# i.e., model output space -> model input space
def inverse_transforms(
        data: MetaTensor, 
        valid_tfs: Compose, 
        loaded_data: MetaTensor, 
        orig_img_path: str,
        save_path: str):
    # print(loaded_data.applied_operations, len(loaded_data.applied_operations))
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


class CusCropForeground(InvertibleTransform):
    """Crop non-zero area of the input data."""
    def __init__(self,):
        pass

    def compute_bounding_box(self, img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(img) # select > 0 as bounding box
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=1))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_
    
    def crop_foreground(self, img):
        # with self.cropper.trace_transform(False):  # don't record the operation to avoid inverse call on this cropper.inverse
        #     with self.cropper.padder.trace_transform(False):
        box_start, box_end = self.compute_bounding_box(img)
        cropped_img = img[..., box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]

        coords = [[box_start[0], box_end[0]], [box_start[1], box_end[1]], [box_start[2], box_end[2]]]
        return cropped_img, coords

    def __call__(self, data):
        self.orig_shape = data.shape[1:]
        cropped_data, coords = self.crop_foreground(data)

        self.crop_coords = coords

        # print("Crop output shape:", cropped_data.shape)

        return cropped_data

    def inverse(self, cropped_data):
        orig_shape = self.orig_shape
        cur_shape = cropped_data.shape[1:]

        inverse_crop_to_pad = [[0, 0]] * 3
        for i in range(3):  # w, h, d
            inverse_crop_to_pad[i] = (self.crop_coords[i][0], orig_shape[i] - self.crop_coords[i][1])
        
        _pad = []
        for group in inverse_crop_to_pad:
            for ele in group:
                _pad.append(int(ele))
        
        inversed_data = F.pad(cropped_data.permute(0, 3, 2, 1), _pad, mode="constant", value=-1024).permute(0, 3, 2, 1)

        print("Inverse in CusCropForeground shape:", inversed_data.shape)

        return inversed_data


class CusResizeWithPadOrCrop(InvertibleTransform):
    "Resize the input data with pad or crop, no interpolation."
    def __init__(self, spatial_size, mode="constant", value=0):
        self.spatial_size = spatial_size
        self.mode = mode
        self.value = value
    
    def resize_to_size(self, data, target_size):
        # data shape: (C, W, H, D)
        _, w, h, d = data.shape
        target_w, target_h, target_d = target_size

        self.orig_size = (w, h, d)

        _pad = [0] * 3 * 2
        _crop = [0, w, 0, h, 0, d]
        if w < target_w:  # following pytorch, _pad[0] and _pad[1] for the last dimension (left and right, the width), cause pytorch seem the data as (N, C, D, H, W)
            _pad[0] = int((target_w - w) // 2)
            _pad[1] = int((target_w - w) - (target_w - w) // 2)
        else:
            _crop[0] = int((w - target_w) // 2)
            _crop[1] = int(target_w + (w - target_w) - (w - target_w) // 2)
        
        if h < target_h:
            _pad[2] = int((target_h - h) // 2)
            _pad[3] = int((target_h - h) - (target_h - h) // 2)
        else:
            _crop[2] = int((h - target_h) // 2)
            _crop[3] = int(target_h + (h - target_h) - (h - target_h) // 2)
        
        if d < target_d:
            _pad[4] = int((target_d - d) // 2)
            _pad[5] = int((target_d - d) - (target_d - d) // 2)
        else:
            _crop[4] = int((d - target_d) // 2)
            _crop[5] = int(target_d + (d - target_d) - (d - target_d) // 2)
        
        # list _crop to slicer
        self._pad = _pad
        self._crop = _crop
        data = data[:, _crop[0]:_crop[1], _crop[2]:_crop[3], _crop[4]:_crop[5]]
        data = F.pad(data.permute(0, 3, 2, 1), _pad, mode=self.mode, value=self.value).permute(0, 3, 2, 1)
        
        return data

    def __call__(self, data):

        data = self.resize_to_size(data, self.spatial_size)

        return data
    
    def inverse(self, resized_data):
        orig_w, orig_h, orig_d = self.orig_size
        cur_w, cur_h, cur_d = resized_data.shape[1:]

        orig_size = (orig_w, orig_h, orig_d)
        cur_size = (cur_w, cur_h, cur_d)

        operated_pad = self._pad
        operated_crop = self._crop
        
        inverse_pad_to_crop = [0] * 6
        inverse_crop_to_pad = [0] * 6
        
        for i in range(3):  # w, h, d
            left, right = i * 2, i * 2 + 1
            inverse_pad_to_crop[left] = int(operated_pad[left])
            inverse_pad_to_crop[right] = int(cur_size[i] - operated_pad[right])

            inverse_crop_to_pad[left] = int(operated_crop[left])
            inverse_crop_to_pad[right] = int(orig_size[i] - operated_crop[right])

        # print(operated_pad, operated_crop)
        # print(resized_data.shape, inverse_pad_to_crop, inverse_crop_to_pad)
        # list _crop to slicer
        inverse_pad_data = resized_data[:, inverse_pad_to_crop[0]:inverse_pad_to_crop[1], inverse_pad_to_crop[2]:inverse_pad_to_crop[3], inverse_pad_to_crop[4]:inverse_pad_to_crop[5]]
        inverse_crop_data = F.pad(inverse_pad_data.permute(0, 3, 2, 1), inverse_crop_to_pad, mode=self.mode, value=-1024).permute(0, 3, 2, 1)

        # print("Inverse in CusResizeWithPadOrCrop shape:", inverse_crop_data.shape)
        return inverse_crop_data


transform_test = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(0.5, 0.5, 3.0)),   # resampling
        CusCropForeground(),
        CusResizeWithPadOrCrop(spatial_size=(448, 448, 64)),
        # CropForeground(),
        # ResizeWithPadOrCrop(spatial_size=(448, 448, 64), mode="constant"),
        HistogramNormalize(min=-1, max=1.0),
    ]
)

# load the model input
mr_img = "data/test_data/mr/FANGLIRONG_0000.nii.gz"

# transform the input
model_input_meta_tensor = transform_test(mr_img)

# do model inference here
# ...

# assume the model output is model_output
model_output = torch.randn_like(model_input_meta_tensor)

# inverse transform the model output
inverse_transforms(
    data=model_output, 
    valid_tfs=transform_test, 
    loaded_data=model_input_meta_tensor, 
    orig_img_path=mr_img,
    save_path="data/test_data/mr_inverse/FANGLIRONG_0000_inverse.nii.gz"
)