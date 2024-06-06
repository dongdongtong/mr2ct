import os
from os.path import join, basename, dirname
from sklearn.model_selection import KFold
import json

data_root_dir = "data/ct_reg2_mr_betneck"
ct_dir = os.path.join(data_root_dir, "ct_reg2_mr")
mr_dir = os.path.join(data_root_dir, "mr")

# Collect file names in the ct subdir
ct_files = os.listdir(ct_dir)

# Retrieve the same filename in the mr subdir
mr_files = [os.path.join(mr_dir, file) for file in ct_files if "brainmask" not in basename(file)]
ct_files = [os.path.join(ct_dir, file) for file in ct_files if "brainmask" not in basename(file)]

# Combine the ct and mr file paths
aligned_data_paths = list(zip(ct_files, mr_files))

# Split the aligned data paths in 5-fold cross validation
cross_validation_json_dir = "data/cross_validation"
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
             "ct_brainmask": ct_file.replace('.nii.gz', '_brainmask.nii.gz'),})
    
    for ct_file, mr_file in test_data:
        test_data_json.append({"ct_image": ct_file, "mr_image": mr_file, "ct_brainmask": ct_file.replace('.nii.gz', '_brainmask.nii.gz')})

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