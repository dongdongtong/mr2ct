"""Created by Dingsd on 11/16/2022 22:21
For pratical clinical 3D image, the axial slice is very hete but have large image size, so need to crop patch.
The patch needs contains foreground hematoma, not randomly crop.
"""

from collections.abc import Sequence
import os
from os.path import basename, dirname, join

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from glob import glob


class AlignedDataset(Dataset):
    def __init__(self, data: list, transforms):
        # each item of data is a dict, which includes the following 5 key:value
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index: int):
        
        mr_img_path = self.data[index]['mr_image']
        ct_img_path = self.data[index]['ct_image']

        data_dict = self.transforms(self.data[index])

        new_dict = {}
        if isinstance(data_dict, list):
            mr_tensor_list = [data_dict[i]['mr_image'].unsqueeze(0) for i in range(len(data_dict))]
            ct_tensor_list = [data_dict[i]['ct_image'].unsqueeze(0) for i in range(len(data_dict))]

            mr_patches = torch.concatenate(mr_tensor_list, dim=0)
            ct_patches = torch.concatenate(ct_tensor_list, dim=0)

            new_dict = {'mr_image': mr_patches, 'ct_image': ct_patches}

            new_dict['mr_img_path'] = mr_img_path
            new_dict['ct_img_path'] = ct_img_path
        else:
            mr_tensor = data_dict['mr_image']
            ct_tensor = data_dict['ct_image']
            new_dict = {'mr_image': mr_tensor, 'ct_image': ct_tensor}
            ct_brainmask = data_dict.get('ct_brainmask', None)
            if ct_brainmask is not None:
                new_dict['ct_brainmask'] = ct_brainmask.type(torch.LongTensor)

            new_dict['mr_img_path'] = mr_img_path
            new_dict['ct_img_path'] = ct_img_path

        return new_dict  # note that for monai transforms, data are all loaded in 0 index

    def __len__(self):
        return len(self.data)


from monai.data.dataset import CacheDataset
class AlignedDatasetCache(CacheDataset):
    def __init__(self, data: list, transforms, cache_num, num_workers):
        super(AlignedDatasetCache, self).__init__(data, transforms, cache_num=cache_num, num_workers=num_workers)

        self.data = data
        self.transforms = transforms
    
    def __getitem__(self, index: int | slice | Sequence[int]):
        data_dict = super().__getitem__(index)  # loaded volume intensity range [0, 255]

        mr_img_path = self.data[index]['mr_image']
        ct_img_path = self.data[index]['ct_image']

        new_dict = {}
        if isinstance(data_dict, list):
            mr_tensor_list = [data_dict[i]['mr_image'].unsqueeze(0) for i in range(len(data_dict))]
            ct_tensor_list = [data_dict[i]['ct_image'].unsqueeze(0) for i in range(len(data_dict))]

            mr_patches = torch.concatenate(mr_tensor_list, dim=0)
            ct_patches = torch.concatenate(ct_tensor_list, dim=0)

            new_dict = {'mr_image': mr_patches, 'ct_image': ct_patches}

            new_dict['mr_img_path'] = mr_img_path
            new_dict['ct_img_path'] = ct_img_path
        else:
            mr_tensor = data_dict['mr_image']
            ct_tensor = data_dict['ct_image']
            new_dict = {'mr_image': mr_tensor, 'ct_image': ct_tensor}
            ct_brainmask = data_dict.get('ct_brainmask', None)
            if ct_brainmask is not None:
                new_dict['ct_brainmask'] = ct_brainmask.type(torch.LongTensor)

            new_dict['mr_img_path'] = mr_img_path
            new_dict['ct_img_path'] = ct_img_path

        return new_dict