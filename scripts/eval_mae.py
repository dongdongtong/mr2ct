import nibabel as nib
import numpy as np
from glob import glob
from os.path import join
from tqdm import tqdm
from collections import defaultdict

output_data_dir = "data/orig_dcm_data/preprocessed/transformed_data"
case_dirs = list(glob(join(output_data_dir, "*")))

mae_dict = defaultdict(list)

for case_dir in tqdm(case_dirs):
    mr_path = join(case_dir, "mr.nii.gz")
    ct_path = join(case_dir, "ct.nii.gz")
    pct_path = join(case_dir, "pseudo_ct.nii.gz")
    
    mr_arr = nib.load(mr_path).get_fdata()
    ct_arr = nib.load(ct_path).get_fdata()
    pct_arr = nib.load(pct_path).get_fdata()
    
    # global MAE
    mae_global = np.abs(ct_arr - pct_arr).mean()
    
    # MAE in the window 500-1500
    window_500_1500_mask = (ct_arr > 500) & (ct_arr < 1500)
    mae_window_500_1500 = np.abs(ct_arr[window_500_1500_mask] - pct_arr[window_500_1500_mask]).mean()
    
    # MAE in the window 0-100
    window_0_100_mask = (ct_arr > 0) & (ct_arr < 100)
    mae_window_0_100 = np.abs(ct_arr[window_0_100_mask] - pct_arr[window_0_100_mask]).mean()
    
    # MAE in the window 100-1500
    window_100_1500_mask = (ct_arr > 100) & (ct_arr < 1500)
    mae_window_100_1500 = np.abs(ct_arr[window_100_1500_mask] - pct_arr[window_100_1500_mask]).mean()
    
    # MAE in the window 100-500
    window_100_500_mask = (ct_arr > 100) & (ct_arr < 500)
    mae_window_100_500 = np.abs(ct_arr[window_100_500_mask] - pct_arr[window_100_500_mask]).mean()
    
    mae_dict['global'].append(mae_global)
    mae_dict['window_500_1500'].append(mae_window_500_1500)
    mae_dict['window_0_100'].append(mae_window_0_100)
    mae_dict['window_100_1500'].append(mae_window_100_1500)
    mae_dict['window_100_500'].append(mae_window_100_500)
    

print("global MAE: ", np.mean(mae_dict['global']))
print("window 0-100 MAE: ", np.mean(mae_dict['window_0_100']))
print("window 100-500 MAE: ", np.mean(mae_dict['window_100_500']))
print("window 100-1500 MAE: ", np.mean(mae_dict['window_100_1500']))
print("window 500-1500 MAE: ", np.mean(mae_dict['window_500_1500']))