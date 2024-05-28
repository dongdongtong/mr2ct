import torch
import numpy as np

import monai
import monai.transforms as transforms
from monai.transforms.inverse import InvertibleTransform


class CropMRIROId(transforms.transform.MapTransform, InvertibleTransform):
    """This transform requires the data is of channel-first shape (channels, H, W, D)"""
    def __init__(self, keys, spatial_size, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = spatial_size
    
    def get_image_slicer_to_crop(self, nonzero_mask):
        outside_value = 0
        mask_voxel_coords = np.where(nonzero_mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
        # resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        return bbox
    
    def crop_z(self, image_array, binary_brainmask):
        if monai.__version__ >= "1.0.0":
            # monai version is greater than or equal to 1.0.0
            crop_bbox = self.get_image_slicer_to_crop(binary_brainmask.numpy()[0])
        else:
            # monai version is less than 1.0.0
            crop_bbox = self.get_image_slicer_to_crop(binary_brainmask[0])
        # crop_bbox_z_end = crop_bbox[2][1] + 6
        # crop_bbox[2][1] = crop_bbox_z_end
        
        crop_center_w = (crop_bbox[0][0] + crop_bbox[0][1]) // 2
        crop_center_h = (crop_bbox[1][0] + crop_bbox[1][1]) // 2
        crop_center_d = (crop_bbox[2][0] + crop_bbox[2][1]) // 2
        crop_center_point = (crop_center_w, crop_center_h, crop_center_d)

        self.cropper = transforms.SpatialCrop(roi_center=crop_center_point, roi_size=self.spatial_size)

        cropped_image = self.cropper(image_array)
        cropped_binary_brainmask = self.cropper(binary_brainmask)
        
        return cropped_image, cropped_binary_brainmask
    
    def __call__(self, data):
        d = dict(data)
        image = d[self.keys[0]]
        binary_brainmask = d[self.keys[1]]

        cropped_image, cropped_binary_brainmask = self.crop_z(image, binary_brainmask)

        d[self.keys[0]] = cropped_image
        d[self.keys[1]] = cropped_binary_brainmask

        return d

    def inverse(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.cropper.inverse(d[key])
        return d


class AdjustCTValuesd(transforms.transform.MapTransform, InvertibleTransform):
    """This transform requires the data is of channel-first shape (channels, H, W, D)
    This transform is used to adjust intensities of all ct images to [0, 255].
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    
    def adjust_values(self, image_array):
        if monai.__version__ >= "1.0.0":
            # monai version is greater than or equal to 1.0.0
            if torch.all(image_array <= 0):
                image_array = image_array - image_array.min()   # to [0, 255]
            else:
                image_array = torch.clamp(image_array, min=0, max=100)
                image_array = image_array / 100 * 255   # to [0, 255]
        else:
            # monai version is less than 1.0.0
            if np.all(image_array <= 0):
                image_array = image_array - image_array.min()   # to [0, 255]
            else:
                image_array = np.clip(image_array, a_min=0, a_max=100)
                image_array = image_array / 100 * 255   # to [0, 255]
        
        return image_array
    
    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            d[key] = self.adjust_values(d[key])
        return d

    def inverse(self, data):
        d = dict(data)
        return d


class NormalizeMRValuesd(transforms.transform.MapTransform, InvertibleTransform):
    """This transform requires the data is of channel-first shape (channels, H, W, D)
    This transform is used to adjust intensities of all mr images to [0, 1].
    We first cutoff the outlier intensities (greater than 99%), then normalize the intensities to [0, 1].
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.clamp_min_value = 0
        self.clamp_max_value = 0
    
    def adjust_values(self, image_array):
        if monai.__version__ >= "1.0.0":
            # monai version is greater than or equal to 1.0.0
            clip_value = np.percentile(image_array.numpy(), 99.5)
            image_array = torch.clamp(image_array, min=0, max=clip_value)
        else:
            # monai version is less than 1.0.0
            clip_value = np.percentile(image_array, 99.5)
            image_array = np.clip(image_array, a_min=0, a_max=clip_value)
        self.clamp_min_value = 0
        self.clamp_max_value = clip_value
        image_array = image_array / clip_value
        
        return image_array
    
    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            # print("NormalizeMRValuesd adjust_values function: key", key)
            d[key] = self.adjust_values(d[key])
        return d

    def inverse(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # print("NormalizeMRValuesd inverse function: key", key)
            if "ct" in key:
                pass
            else:
                d[key] = d[key] * self.clamp_max_value
        return d


def get_train_transforms(cfg):
    # mr_img_size = cfg['mr_img_size']   # here use nib lib, so the layout of loaded image shape is HWD
    # ct_img_size = cfg['ct_img_size']
    mr_boder_pad = cfg['mr_boder_pad']

    mr_crop_img_size = cfg['mr_crop_img_size']
    ct_crop_img_size = cfg['ct_crop_img_size']
    mr_final_img_size = cfg['mr_final_img_size']
    ct_final_img_size = cfg['ct_final_img_size']
    
    need_keys = ["mr_image", "ct_image", "mr_brainmask"]
    
    train_transforms = [
        transforms.LoadImaged(keys=need_keys),
        transforms.EnsureChannelFirstd(keys=need_keys),
        transforms.Orientationd(keys=need_keys, axcodes="RAS"),
        # for ct image transform
        transforms.Spacingd(keys=['ct_image'], pixdim=(0.5, 0.5, 3.0), mode=['bilinear']),
        AdjustCTValuesd(keys=['ct_image']),
        transforms.CropForegroundd(keys=['ct_image'], source_key="ct_image"),   # crop to align the brain center into the image center
        transforms.ResizeWithPadOrCropd(keys=['ct_image'], spatial_size=ct_crop_img_size),
        transforms.Resized(keys=['ct_image'], spatial_size=ct_final_img_size),
        # for mr image transform, make sure the cropped image should contain the brain skull
        transforms.Spacingd(keys=['mr_image', 'mr_brainmask'], pixdim=(0.5, 0.5, 3.0), mode=['bilinear', 'nearest']),
        transforms.BorderPadd(keys=['mr_image', 'mr_brainmask'], spatial_border=mr_boder_pad),
        CropMRIROId(keys=['mr_image', 'mr_brainmask'], spatial_size=mr_crop_img_size),
        transforms.Resized(keys=['mr_image', 'mr_brainmask'], spatial_size=mr_final_img_size),
    ]

    # other spatial transforms for data augmentation
    spatial_transforms = [
        transforms.RandFlipd(keys=need_keys, prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=need_keys, prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=need_keys, prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=need_keys, prob=0.3, max_k=3),
    ]

    # intensity transforms for data augmentation
    intensity_transforms = [
        transforms.RandScaleIntensityd(keys=['mr_image',], factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys=['mr_image', ], offsets=0.1, prob=0.15),
        transforms.RandGaussianNoised(keys=['mr_image',], prob=0.15),
        transforms.RandScaleIntensityd(keys=['ct_image'], factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys=['ct_image'], offsets=0.1, prob=0.15),
        transforms.RandGaussianNoised(keys=['ct_image'], prob=0.15),
    ]

    # nomalize the intensity of the image
    normalize_transforms = [
        transforms.ScaleIntensityRanged(keys=['ct_image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        NormalizeMRValuesd(keys=['mr_image']),
    ]

    train_transforms.extend(spatial_transforms)
    train_transforms.extend(intensity_transforms)
    train_transforms.extend(normalize_transforms)

    return transforms.Compose(train_transforms)


def get_valid_transforms(cfg):
    mr_crop_img_size = cfg['mr_crop_img_size']
    ct_crop_img_size = cfg['ct_crop_img_size']
    mr_final_img_size = cfg['mr_final_img_size']
    ct_final_img_size = cfg['ct_final_img_size']
    mr_boder_pad = cfg['mr_boder_pad']
    
    need_keys = ["mr_image", "ct_image", "mr_brainmask"]
    
    train_transforms = [
        transforms.LoadImaged(keys=need_keys),
        transforms.EnsureChannelFirstd(keys=need_keys),
        transforms.Orientationd(keys=need_keys, axcodes="RAS"),
        # for ct image transform
        transforms.Spacingd(keys=['ct_image'], pixdim=(0.5, 0.5, 3.0), mode=['bilinear']),
        AdjustCTValuesd(keys=['ct_image']),
        transforms.CropForegroundd(keys=['ct_image'], source_key="ct_image"),   # crop to align the brain center into the image center
        transforms.ResizeWithPadOrCropd(keys=['ct_image'], spatial_size=ct_crop_img_size),
        transforms.Resized(keys=['ct_image'], spatial_size=ct_final_img_size),
        # for mr image transform, make sure the cropped image should contain the brain skull
        transforms.Spacingd(keys=['mr_image', 'mr_brainmask'], pixdim=(0.5, 0.5, 3.0), mode=['bilinear', 'nearest']),
        transforms.BorderPadd(keys=['mr_image', 'mr_brainmask'], spatial_border=mr_boder_pad),
        CropMRIROId(keys=['mr_image', 'mr_brainmask'], spatial_size=mr_crop_img_size),
        transforms.Resized(keys=['mr_image', 'mr_brainmask'], spatial_size=mr_final_img_size),

        # normalize the intensity of the image to [0, 1]
        transforms.ScaleIntensityRanged(keys=['ct_image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        NormalizeMRValuesd(keys=['mr_image']),
    ]

    return transforms.Compose(train_transforms)