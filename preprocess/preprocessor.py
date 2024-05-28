import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from preprocess.brain_extractor import BrainExtractor
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
import pandas as pd
import os


def get_target_spacing(spacing_statistics_npy_path="data/spacing.npy"):
    target_spacing_percentile = 10
    spacings = np.load(spacing_statistics_npy_path)
    target = np.percentile(np.vstack(spacings), target_spacing_percentile, 0)

    anisotropy_threshold = 3

    worst_spacing_axis = np.argmax(target)
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]

    has_aniso_spacing = target[worst_spacing_axis] > (anisotropy_threshold * min(other_spacings))
    # has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)
    has_aniso_voxels = True
    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < min(other_spacings):
            target_spacing_of_that_axis = max(min(other_spacings), target_spacing_of_that_axis) + 1e-5
        target[worst_spacing_axis] = target_spacing_of_that_axis

    return target  # target spacing


def get_target_spacing_from_excel(spacing_shape_statistics_excel_path="data/bet_nii_info.xlsx"):
    target_spacing_percentile = 50
    spacing_shape_info = pd.read_excel(spacing_shape_statistics_excel_path)
    spacings = np.array(spacing_shape_info[['spacing_x', 'spacing_y', 'spacing_z']])
    shapes = np.array(spacing_shape_info[['shape_x', 'shape_y', 'shape_z']])
    
    target_spacing = np.percentile(np.vstack(spacings), target_spacing_percentile, 0)
    target_size = np.percentile(np.vstack(shapes), target_spacing_percentile, 0)

    anisotropy_threshold = 3

    worst_spacing_axis = np.argmax(target_spacing)
    other_axes = [i for i in range(len(target_spacing)) if i != worst_spacing_axis]
    other_spacings = [target_spacing[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = target_spacing[worst_spacing_axis] > (anisotropy_threshold * min(other_spacings))
    has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)
    do_seperate_z = True if has_aniso_spacing and has_aniso_voxels else False
    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < min(other_spacings):
            target_spacing_of_that_axis = max(min(other_spacings), target_spacing_of_that_axis) + 1e-5
        target_spacing[worst_spacing_axis] = target_spacing_of_that_axis

    return target_spacing, do_seperate_z  # target spacing


def get_new_shape(itk_img, cropped_img, target_spacing):
    orig_spacing = itk_img.GetSpacing()
    cropped_shape = cropped_img.shape
    ratio_orig_target = (np.array(orig_spacing) / np.array(target_spacing)).astype(np.float32)
    calibrate_ratio_order = [ratio_orig_target[2], ratio_orig_target[0], ratio_orig_target[1]]
    new_shape = np.round((np.array(calibrate_ratio_order) * cropped_shape)).astype(int)
    return new_shape


def array2image(array, origin_image, new_spacing=None):
    if array is None:
        raise Exception("You want to transform a NONE array to simpleITK image!!!")
    rec_image = sitk.GetImageFromArray(array)
    rec_image.SetDirection(origin_image.GetDirection())
    if new_spacing is not None:
        rec_image.SetSpacing(new_spacing)
    else:
        rec_image.SetSpacing(origin_image.GetSpacing())
    rec_image.SetOrigin(origin_image.GetOrigin())

    return rec_image


def resample_data_or_seg(data, new_shape, axis=None, order=0, do_separate_z=False, is_seg=False):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param axis:
    :param order: order 3 for image spline interpolation, order 0 for segmentation or z-axis nearest interpolation
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    # print(shape, new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            # assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_data = []
            for slice_id in range(data.shape[0]):
                if axis == 0:
                    reshaped_data.append(
                        resize(data[slice_id], new_shape_2d, order, preserve_range=True).astype(dtype_data))
                elif axis == 1:
                    reshaped_data.append(
                        resize(data[:, slice_id], new_shape_2d, order).astype(dtype_data))
                else:
                    reshaped_data.append(
                        resize(data[:, :, slice_id], new_shape_2d, order).astype(dtype_data))
            reshaped_final_data = np.array(reshaped_data)

            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_final_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                reshaped_final_data = map_coordinates(reshaped_data, coord_map, order=0, cval=0, mode='nearest')
        else:
            # print("no separate z, order", order)
            # reshaped_final_data = resize_segmentation(data, new_shape, order, ).astype(dtype_data)
            # if is_seg:
            #     reshaped_final_data = resize_segmentation(data, new_shape, order, ).astype(dtype_data)
            reshaped_final_data = resize(data.astype(float), new_shape, order, cval=0, mode="edge").astype(dtype_data)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def get_image_slicer_to_crop(nonzero_mask):
    outside_value = 0
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return resizer


def crop_image(itk_img, itk_seg=None, brain_mask=None):
    img = sitk.GetArrayFromImage(itk_img).astype(float)
    # np.savez("figures/preprocess_example/data_source/original_ct_brain.npz", data=img)
    # threshold = np.percentile(img, 50)  # for ct brain image, we find threshold just set as 0 is just well.
    if brain_mask is not None:
        nonzero_mask = brain_mask.copy()
    else:
        brain_extractor = BrainExtractor()
        nonzero_mask = brain_extractor.get_brain_mask(itk_img)
        nonzero_mask = sitk.GetArrayFromImage(nonzero_mask)
        # nonzero_mask = img.copy()
    # np.savez("figures/preprocess_example/data_source/original_nonzero_mask.npz", data=nonzero_mask)
    resizer = get_image_slicer_to_crop(nonzero_mask)
    cropped_img = img[resizer]
    cropped_nonzero_mask = nonzero_mask[resizer]
    cropped_seg = None
    if itk_seg:
        seg = sitk.GetArrayFromImage(itk_seg)
        cropped_seg = seg[resizer]
    # print("before crop, shape: ", img.shape, "after crop shape:", cropped_img.shape)
    return cropped_img, cropped_nonzero_mask, cropped_seg


class Preprocessor(object):
    def __init__(self, target_spacing, do_separate):
        self.target_spacing = target_spacing
        self.do_separate_z = do_separate

    def resample_patient(self, itk_img, cropped_img, cropped_mask, cropped_seg=None):
        new_shape = get_new_shape(itk_img, cropped_img, self.target_spacing)
        # cropped_img = np.expand_dims(cropped_img, axis=0)
        # cropped_mask = np.expand_dims(cropped_mask, axis=0)
        resampled_img = resample_data_or_seg(cropped_img, new_shape, order=3, axis=[0],
                                             do_separate_z=self.do_separate_z, order_z=0)
        resampled_mask = resample_data_or_seg(cropped_mask, new_shape, order=0, axis=[0],
                                              do_separate_z=self.do_separate_z, order_z=0)
        resampled_seg = None
        if cropped_seg:
            resampled_seg = resample_data_or_seg(cropped_seg, new_shape, order=0, axis=[0],
                                                 do_separate_z=False, order_z=0)

        return {
            "resampled_img": array2image(resampled_img, itk_img, self.target_spacing),
            "resampled_mask": array2image(resampled_mask, itk_img, self.target_spacing),
            "resampled_seg": array2image(resampled_seg, itk_img, self.target_spacing),
        }

    def resample_data(self, cropped_array, new_shape, is_seg=False):
        if not is_seg:
            return resample_data_or_seg(cropped_array, new_shape, order=3, axis=[0], do_separate_z=self.do_separate_z, is_seg=is_seg)
        else:
            return resample_data_or_seg(cropped_array, new_shape, order=0, axis=[0], do_separate_z=self.do_separate_z, is_seg=is_seg)

    def run(self, img_path, seg_path=None, brain_mask_path=None, need_resample=True):
        resampled_dict = {}
        
        print("-> load img")
        itk_img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(itk_img)

        itk_seg = None
        if seg_path is not None and os.path.exists(seg_path):
            print("-> load seg")
            itk_seg = sitk.ReadImage(seg_path)
            seg = sitk.GetArrayFromImage(itk_seg)
        
        brain_mask = None
        if brain_mask_path is not None and os.path.exists(brain_mask_path):
            print("-> load brain mask and mask out img 0")
            itk_brain_mask = sitk.ReadImage(brain_mask_path)
            brain_mask = sitk.GetArrayFromImage(itk_brain_mask)

            img[brain_mask == 0] = 0
            itk_img = array2image(img, itk_img)
            
            if seg_path is not None and os.path.exists(seg_path):
                print("-> load brain mask and mask out seg 0")
                seg[brain_mask == 0] = 0
                itk_seg = array2image(seg, itk_seg)
        
        # crop image
        cropped_img, cropped_nonzero_mask, cropped_seg = crop_image(itk_img, itk_seg, brain_mask)
        print(f"After crop:, img shape: {img.shape} -> {cropped_img.shape}")
        resampled_dict["cropped_img"] = array2image(cropped_img, itk_img)

        if seg_path is not None and os.path.exists(seg_path):
            print(f"After crop:, seg shape: {seg.shape} -> {cropped_seg.shape}")
            resampled_dict["cropped_seg"] = array2image(cropped_seg, itk_seg)
        
        if brain_mask_path is not None and os.path.exists(brain_mask_path):
            resampled_dict["cropped_nonzero_mask"] = array2image(cropped_nonzero_mask, itk_img)

        if need_resample:
            new_shape = get_new_shape(itk_img, cropped_img, self.target_spacing)
            print("-> resampling img")
            resampled_img = self.resample_data(cropped_img, new_shape, is_seg=False)
            resampled_dict["resampled_img"] = array2image(resampled_img, itk_img, self.target_spacing)

            if seg_path is not None and os.path.exists(seg_path):
                print("-> resampling seg")
                resampled_seg = self.resample_data(cropped_seg, new_shape, is_seg=True)
                print(f"-> After resampling, seg min: {resampled_seg.min()}, max: {resampled_seg.max()}")
                # print("resampled seg sum: ", (resampled_seg > 0).sum(), resampled_seg.shape, resampled_img.shape)
                resampled_dict["resampled_seg"] = array2image(resampled_seg, itk_seg, self.target_spacing)
            else:
                resampled_dict["resampled_seg"] = None
        # cropped_img, cropped_nonzero_mask, cropped_seg = crop_image(itk_img, itk_seg)
        # np.savez("figures/preprocess_example/data_source/cropped_img.npz", data=cropped_img)
        # resampled_dict = self.resample_patient(itk_img, cropped_img, cropped_nonzero_mask, cropped_seg)
        # print("before resample, spacing: ", itk_img.GetSpacing(), "after resample, spacing: ", resampled_img.GetSpacing())
        return resampled_dict
