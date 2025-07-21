import os
import os.path as osp
from glob import glob



mask_path = "/data/dingsd/mr2ct/data/orig_dcm_data/dcm2nii/masked_ct"
out_dir = "data/ct_headmask"
os.makedirs(out_dir, exist_ok=True)


ct_mask_files = list(glob(osp.join(mask_path, "*_brainmask.nii.gz")))

for ct_mask_file in ct_mask_files:
    pid = osp.basename(ct_mask_file).split("_")[0]
    out_file = osp.join(out_dir, f"{pid}_0000.nii.gz")
    
    if osp.exists(out_file):
        print(f"File already exists: {out_file}")
        continue
    
    os.rename(ct_mask_file, out_file)
    print(f"Renamed {ct_mask_file} to {out_file}")