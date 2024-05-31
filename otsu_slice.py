from monai.losses import SSIMLoss
import torch
import SimpleITK as sitk
import nibabel as nib
import os
import numpy as np


# pct =  nib.load("./MAGUIXIA_0000_pseudo_ct.nii.gz").get_fdata()
# ref_ct = nib.load("./ct_MAGUIXIA_0000.nii.gz").get_fdata()

# pct = torch.tensor(pct).unsqueeze(0).unsqueeze(0).float().cuda()
# ref_ct = torch.tensor(ref_ct).unsqueeze(0).unsqueeze(0).float().cuda()
# pct.requires_grad = True
# ref_ct.requires_grad = True
# # print(pct.min(), pct.max(), ref_ct.min(), ref_ct.max())

# with torch.no_grad():
#     data_range = ref_ct.max() - ref_ct.min()
# norm_pct = (pct - pct.min()) / (pct.max() - pct.min())
# norm_ref_ct = (ref_ct - ref_ct.min()) / (ref_ct.max() - ref_ct.min())
# norm_pct.retain_grad()
# norm_ref_ct.retain_grad()

# ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)
# # loss = ssim_loss(pct - pct.min(), ref_ct - ref_ct.min())
# norm_loss = ssim_loss(norm_pct, norm_ref_ct)
# # print("before norm:", loss.item(), "after norm:", norm_loss.item())
# print("after norm:", norm_loss.item())

# norm_loss = data_range.item() * norm_loss
# norm_loss.backward(retain_graph=True)
# print(norm_pct.grad.min(), norm_pct.grad.max(), norm_ref_ct.grad.min(), norm_ref_ct.grad.max())
# print(pct.grad.min(), pct.grad.max(), ref_ct.grad.min(), ref_ct.grad.max())

# ssim_loss = SSIMLoss(spatial_dims=2,)
# # x -axis ssim loss
# a = torch.zeros(1, 1, 256, 256)
# b = torch.zeros(1, 1, 256, 256)
# a.requires_grad = True
# b.requires_grad = True
# print(ssim_loss(a, b))
# ssim_loss(a, b).backward()
# print(a.grad)

# # y -axis ssim loss
# print(ssim_loss(pct[0].permute(0, 2, 1, 3), ref_ct[0].permute(0, 2, 1, 3)))

# # z -axis ssim loss
# print(ssim_loss(pct[0].permute(0, 3, 1, 2), ref_ct[0].permute(0, 3, 1, 2)))


directory = 'data/ct_reg2_mr_betneck/ct_reg2_mr'  # Replace with the actual directory path

wmin = 999
wmax = 0
for filename in os.listdir(directory):
    if filename.endswith('.nii.gz'):
        file_path = os.path.join(directory, filename)
        data = nib.load(file_path).get_fdata()
        data_min = data.min()
        data_max = data.max()
        data_max = np.percentile(data, 99.995)
        print(f"File: {filename}, Min: {data_min}, Max: {data_max}")

        if data_min < wmin:
            wmin = data_min
        
        if data_max > wmax:
            wmax = data_max

print(wmin, wmax)