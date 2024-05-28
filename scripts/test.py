import os
from os.path import dirname, basename, join
import sys
sys.path.append("/opt/dingsd/cyclegan_mr2ct")
import monai.transforms
import yaml
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nibabel as nib
from monai.networks import one_hot
import shutil

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
import monai
from monai.data.meta_tensor import MetaTensor

from models.generators.unet import UNet

from datasets.aligned_dataset import AlignedDataset
from datasets.augmentations_aligned import get_valid_transforms
from datasets.create_dataloaders import load_data_dicts_from_data_root, save_json

from torch.utils.data import DataLoader


def create_test_loader(config):
    random_seed = config['random_seed']
    data_json_path = config['data_json_path']

    if os.path.exists(data_json_path):
        with open(data_json_path, encoding='utf-8') as f:
            data_json = json.load(f)   # contain training, validation, no test for cycleGAN
    else:
        data_root_dir = config['data_root_dir']
        image_subfolder_name = config['image_subfolder_name']
        mr_brainmask_subfolder_name = config['mr_brainmask_subfolder_name']
        data_dicts = load_data_dicts_from_data_root(data_root_dir, image_subfolder_name, mr_brainmask_subfolder_name)
        X_train, X_val = train_test_split(data_dicts, test_size=0.2, random_state=random_seed)

        data_json = {
            "training": X_train,
            "validation": X_val
        }

        save_json(data_json, data_json_path)
    
    val_json = data_json['validation']

    valid_tfs = get_valid_transforms(config)
    
    valid_ds = AlignedDataset(val_json, valid_tfs)
    
    return valid_ds, valid_tfs


def load_pretrained_model(args, config):
    spatial_dim = config['spatial_dim']
    in_channels = config['in_channel']

    strides = config['strides']
    filters = config['filters'][: len(strides)]
    kernel_size = config['kernel_size']
    upsample_kernel_size = strides[1:]
    G_S = UNet(in_channels, in_channels, filters, strides, kernel_size, upsample_kernel_size)
    G_T = UNet(in_channels, in_channels, filters, strides, kernel_size, upsample_kernel_size)

    state_dict = torch.load(args.checkpoint_path)
    print(f"Load checkpoint from {args.checkpoint_path}")
    G_S_state_dict = state_dict['G_S']
    G_T_state_dict = state_dict['G_T']

    G_S.load_state_dict(G_S_state_dict)
    G_T.load_state_dict(G_T_state_dict)

    G_S = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_S)
    G_T = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_T)
    
    G_S = G_S.cuda()
    G_T = G_T.cuda()
    
    return G_S, G_T


from monai.transforms.utils import allow_missing_keys_mode
def inverse_transforms(
        data: MetaTensor, 
        valid_tfs: monai.transforms.Compose, 
        loaded_data: MetaTensor, 
        loaded_data_key_in_transform: str,
        orig_img_path: str,
        save_path: str):
    # create metatensor inheriting transforms operations from loaded_data
    data = MetaTensor(data, affine=loaded_data.affine, applied_operations=loaded_data.applied_operations)

    inverse_input_data_dict = {loaded_data_key_in_transform: data}
    with allow_missing_keys_mode(valid_tfs):
        inverted_data_dict = valid_tfs.inverse(inverse_input_data_dict)
    
    inverted_data = inverted_data_dict[loaded_data_key_in_transform]
    inverted_data = inverted_data.squeeze(0)  # remove channel dimension, to shape (W, H, D)

    # save inverted_img as nii.gz
    img_nib = nib.load(orig_img_path)
    aff, header = img_nib.affine, img_nib.header
    volume = inverted_data.cpu().numpy().astype(dtype=img_nib.get_fdata().dtype)

    # print(volume.min(), volume.max())
    nifty = nib.Nifti1Image(volume, aff, header)
    os.makedirs(dirname(save_path), exist_ok=True)
    nib.save(nifty, save_path)


def test(args):
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    
    if args.data_json_path is None:
        args.data_json_path = config['data_json_path']
    else:
        config['data_json_path'] = args.data_json_path

    test_dataset, valid_tfs = create_test_loader(config)
    
    G_S, G_T = load_pretrained_model(args, config)
    G_S.eval()
    G_T.eval()
    
    out_dir = args.out_dir
    with torch.no_grad():
        skipped_patients = 0
        for idx, batch in tqdm(enumerate(test_dataset)):
            mr_img = batch['mr_image'].unsqueeze(0).cuda()
            ct_img = batch['ct_image'].unsqueeze(0).cuda()

            mr_img_path =  batch['mr_img_path']
            ct_img_path =  batch['ct_img_path']
            pid = basename(mr_img_path).split(".nii")[0]

            fake_ct = G_S(mr_img)
            fake_ct = interpolate(fake_ct, size=ct_img.shape[2:], mode='trilinear', align_corners=True)
            fake_mr = G_T(ct_img)
            fake_mr = interpolate(fake_mr, size=mr_img.shape[2:], mode='trilinear', align_corners=True)

            # we use monai inverse transfroms to convert the fake images back into the original NIFTI space.
            # inverse fake_ct and save it
            print(fake_ct.shape, fake_mr.shape)
            fake_ct = fake_ct.squeeze(0).cpu()  # remove batch dimentsion, to shape (C, H, W, D)
            save_path_fake_ct = join(out_dir, pid, "fake_ct.nii.gz")
            print(f"---> Processing fake_ct to {save_path_fake_ct}...")
            inverse_transforms(
                fake_ct, 
                valid_tfs, 
                batch['ct_image'], 
                'ct_image',
                ct_img_path,
                save_path_fake_ct)
            
            # inverse fake_mr and save it
            fake_mr = fake_mr.squeeze(0).cpu()  # remove batch dimentsion, to shape (C, H, W, D)
            save_path_fake_mr = join(out_dir, pid, "fake_mr.nii.gz")
            print(f"---> Processing fake_mr to {save_path_fake_mr}...")
            inverse_transforms(
                fake_mr, 
                valid_tfs, 
                batch['mr_image'], 
                'mr_image',
                mr_img_path,
                save_path_fake_mr)
            shutil.copyfile(mr_img_path, join(out_dir, pid, "orig_mr.nii.gz"))
            shutil.copyfile(ct_img_path, join(out_dir, pid, "orig_ct.nii.gz"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='hematoma distributed training work')
    
    parser.add_argument('--config', type=str, help='config file path')

    parser.add_argument('--checkpoint_path', default="runs/xxx/xxx/***.pt", type=str, help='Model checkpoint weight loaded')

    parser.add_argument('--data_root_dir', type=str, help='dataset root path')
    parser.add_argument('--image_subfolder_name', type=str, help='contains the mr and ct subfolders')
    parser.add_argument('--mr_brainmask_subfolder_name', type=str, help='contains the brain mask for mr')
    parser.add_argument('--data_json_path', type=str, help='if no json found in this path, it will be created using the random seed')

    parser.add_argument('--out_dir', type=str, help='directory to save the translated images')
    
    args = parser.parse_args()
    
    test(args)