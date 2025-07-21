import os
from os.path import basename, dirname, join
import yaml
import json
import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from glob import glob

from datasets.sampler import Sampler
from datasets.aligned_dataset import AlignedDataset, AlignedDatasetCache
import datasets.augmentations_aligned as aug_cyclegan
import datasets.augmentations_patch as aug_sup
import datasets.augmentations_patch_t12ct as aug_sup_t12ct

from monai.data.dataset import CacheDataset

from sklearn.model_selection import train_test_split


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def worker_init_fn(worker_id):    # see: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data_dicts_from_data_root(data_root_path, image_subfolder_name, mr_brainmask_subfolder_name, file_suffix):
    image_root_path = join(data_root_path, image_subfolder_name)
    mr_brainmask_subfolder_name = join(data_root_path, mr_brainmask_subfolder_name)

    mr_image_files = list(glob(join(image_root_path, "mr", f"*{file_suffix}")))
    mr_image_files = sorted(mr_image_files, key=lambda x: basename(x).split(".")[0])  # sort by pid

    data_dicts = []
    for mr_image_file in mr_image_files:
        pid = basename(mr_image_file).split(".")[0]
        ct_image_file = join(image_root_path, "ct", pid + file_suffix)
        mr_brainmask_file = join(mr_brainmask_subfolder_name, pid + f"_brainseg{file_suffix}")

        if os.path.exists(ct_image_file) and os.path.exists(mr_brainmask_file):
            data_dicts.append({
                "mr_image": mr_image_file,
                "ct_image": ct_image_file,
                "mr_brainmask": mr_brainmask_file
            })
    
    return data_dicts


def create_loader(args, rank, tune_param=False, **kwargs):
    config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, yaml.FullLoader)
    
    random_seed = config['random_seed']
    data_json_path = config['data_json_path']
    data_file_suffix = config['file_suffix']

    if os.path.exists(data_json_path):
        with open(data_json_path, encoding='utf-8') as f:
            data_json = json.load(f)   # contain training, validation, no test for cycleGAN
    else:
        data_root_dir = config['data_root_dir']
        image_subfolder_name = config['image_subfolder_name']
        mr_brainmask_subfolder_name = config['mr_brainmask_subfolder_name']
        data_dicts = load_data_dicts_from_data_root(data_root_dir, image_subfolder_name, mr_brainmask_subfolder_name, data_file_suffix)
        X_train, X_val = train_test_split(data_dicts, test_size=0.2, random_state=random_seed)

        data_json = {
            "training": X_train,
            "validation": X_val
        }

        save_json(data_json, data_json_path)
    
    training_json = data_json['training']
    val_json = data_json['validation']

    train_tfs = aug_cyclegan.get_train_transforms(config)
    valid_tfs = aug_cyclegan.get_valid_transforms(config)
    
    train_ds = AlignedDataset(training_json, train_tfs)
    valid_ds = AlignedDataset(val_json, valid_tfs)
        
    g = torch.Generator()
    g.manual_seed(random_seed + rank)  # cause we're not using DDP, so the rank is just equals to 0.
    
    if args.ddp:
        train_sampler = Sampler(train_ds)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=config['workers'],
        sampler=train_sampler,
        # pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
        # shuffle=False,
        # persistent_workers=True,
        # collate_fn=data.utils.pad_list_data_collate,
    )
    
    val_sampler = Sampler(valid_ds, shuffle=False) if args.ddp else None
    val_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
        # pin_memory=True,
        # worker_init_fn=worker_init_fn,
        # persistent_workers=True,
    )

    
    loader = {
        "train_loader": train_loader, 
        "valid_loader": val_loader, 
        "train_ds": train_ds, 
        "valid_ds": valid_ds,
        }
    
    return loader


def create_patch_loader(args, rank, tune_param=False, **kwargs):
    config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, yaml.FullLoader)
    
    random_seed = config['random_seed']
    data_json_path = config['data_json_path']
    data_file_suffix = config['file_suffix']

    if os.path.exists(data_json_path):
        if "cross_validation_fold" in basename(data_json_path):
            data_json_path = join(dirname(data_json_path), f"cross_validation_fold_{args.fold}.json")
        
        with open(data_json_path, encoding='utf-8') as f:
            data_json = json.load(f)   # contain training, validation, no test for cycleGAN
        
    else:
        data_root_dir = config['data_root_dir']
        image_subfolder_name = config['image_subfolder_name']
        mr_brainmask_subfolder_name = config['mr_brainmask_subfolder_name']
        data_dicts = load_data_dicts_from_data_root(data_root_dir, image_subfolder_name, mr_brainmask_subfolder_name, data_file_suffix)
        X_train, X_val = train_test_split(data_dicts, test_size=0.2, random_state=random_seed)

        data_json = {
            "training": X_train,
            "validation": X_val
        }

        save_json(data_json, data_json_path)
    
    training_json = data_json['training']
    val_json = data_json['validation']

    train_tfs = aug_sup_t12ct.get_train_transforms(config)
    valid_tfs = aug_sup_t12ct.get_valid_transforms(config)
    
    # train_ds = AlignedDataset(training_json, train_tfs)
    # valid_ds = AlignedDataset(val_json, valid_tfs)

    train_ds = AlignedDatasetCache(training_json, train_tfs, cache_num=len(training_json), num_workers=17)
    valid_ds = AlignedDatasetCache(val_json, valid_tfs, cache_num=len(val_json), num_workers=5)
        
    g = torch.Generator()
    g.manual_seed(random_seed + rank)  # cause we're not using DDP, so the rank is just equals to 0.
    
    if args.ddp:
        train_sampler = Sampler(train_ds)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=config['workers'],
        sampler=train_sampler,
        # pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4,
        # shuffle=False,
        persistent_workers=True,
        # collate_fn=data.utils.pad_list_data_collate,
    )
    
    val_sampler = Sampler(valid_ds, shuffle=False) if args.ddp else None
    val_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
        # pin_memory=True,
        # worker_init_fn=worker_init_fn,
        # persistent_workers=True,
    )

    
    loader = {
        "train_loader": train_loader,
        # "train_whole_image_loader": train_whole_image_ds,
        "valid_loader": val_loader, 
        "train_ds": train_ds, 
        "valid_ds": valid_ds,
        }
    
    return loader


def create_loader_for_debug(config_path):
    with open(config_path) as f:
        config = yaml.load(f, yaml.FullLoader)
    
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
    
    training_json = data_json['training']
    val_json = data_json['validation']

    train_tfs = aug_sup.get_train_transforms(config)
    valid_tfs = aug_sup.get_valid_transforms(config)
    
    train_ds = AlignedDataset(training_json, train_tfs)
    valid_ds = AlignedDataset(val_json, valid_tfs)
        
    g = torch.Generator()
    g.manual_seed(random_seed + 0)  # cause we're not using DDP, so the rank is just equals to 0.
    
    train_sampler = None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=config['workers'],
        sampler=train_sampler,
        # pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
        # shuffle=False,
        # persistent_workers=True,
        # collate_fn=data.utils.pad_list_data_collate,
    )
    
    val_sampler = None
    val_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
        # pin_memory=True,
        # worker_init_fn=worker_init_fn,
        # persistent_workers=True,
    )

    
    loader = {
        "train_loader": train_loader, 
        "valid_loader": val_loader, 
        "train_ds": train_ds, 
        "valid_ds": valid_ds,
        }
    
    return loader