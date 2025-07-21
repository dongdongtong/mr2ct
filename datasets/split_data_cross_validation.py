import os
from os.path import join, basename, dirname
from sklearn.model_selection import KFold
import json
from glob import glob


def generate_cross_validation_json(data_root_dir, mr_subfolder_name, ct_subfolder_name, file_suffix=".nii.gz"):
    """
    Generate cross-validation JSON files for CT and MR images.
    """
    ct_dir = join(data_root_dir, ct_subfolder_name)
    mr_dir = join(data_root_dir, mr_subfolder_name)

    # Collect file names in the ct subdir
    ct_file_paths = list(glob(join(ct_dir, f"*{file_suffix}")))

    # Retrieve the same filename in the mr subdir
    pids = [basename(file_path).split(".nii")[0] for file_path in ct_file_paths if "mask" not in basename(file_path)]
    mr_file_paths = [join(mr_dir, f"{pid}{file_suffix}") for pid in pids]
    ct_file_paths = [join(ct_dir, f"{pid}{file_suffix}") for pid in pids]

    # Combine the ct and mr file paths
    aligned_data_paths = list(zip(ct_file_paths, mr_file_paths))

    # Split the aligned data paths in 5-fold cross validation
    cross_validation_json_dir = join(data_root_dir, "cross_validation_t12ct")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_i, (train_index, test_index) in enumerate(kfold.split(aligned_data_paths)):
        train_data = [aligned_data_paths[i] for i in train_index]
        test_data = [aligned_data_paths[i] for i in test_index]
        train_data_json = []
        test_data_json = []

        for ct_file, mr_file in train_data:
            train_data_json.append(
                {"ct_image": ct_file, 
                "mr_image": mr_file, 
                "ct_brainmask": ct_file.replace('.nii.gz', '_headmask.nii.gz'),
                "ct_skullstrip_mask": ct_file.replace('.nii.gz', '_brainmask.nii.gz'),
                })
        
        for ct_file, mr_file in test_data:
            test_data_json.append(
                {"ct_image": ct_file, 
                "mr_image": mr_file, 
                "ct_brainmask": ct_file.replace('.nii.gz', '_headmask.nii.gz'),
                "ct_skullstrip_mask": ct_file.replace('.nii.gz', '_brainmask.nii.gz'),
                })

        data_json = {
            "numTraining": len(train_data_json),
            "numValidation": len(test_data_json),
            "training": train_data_json,
            "validation": test_data_json,
        }

        # Save train_data_json and test_data_json to JSON files
        json_path = join(cross_validation_json_dir, f"cross_validation_fold_{fold_i}.json")
        os.makedirs(cross_validation_json_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data_json, f, indent=4, sort_keys=True)