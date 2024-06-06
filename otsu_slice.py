from monai.losses import SSIMLoss
import torch
import SimpleITK as sitk
import nibabel as nib
import os
import numpy as np
import torch.nn.functional as F


pct =  nib.load("./train_LUFA_0000_pseudo_ct_transformer_1000l1loss_500_3dssim_500_2dssim_3t.nii.gz").get_fdata()
ref_ct = nib.load("./ct_LUFA_0000.nii.gz").get_fdata()

pct = torch.tensor(pct).unsqueeze(0).unsqueeze(0).float().cuda()
ref_ct = torch.tensor(ref_ct).unsqueeze(0).unsqueeze(0).float().cuda()
head_mask = ref_ct > 0
pct.requires_grad = True
ref_ct.requires_grad = True
# print(pct.min(), pct.max(), ref_ct.min(), ref_ct.max())

# with torch.no_grad():
#     pct_data_range = pct.max() - pct.min()
#     data_range = ref_ct.max() - ref_ct.min()
#     norm_pct = (pct - pct.min()) / pct_data_range
#     norm_ref_ct = (ref_ct - ref_ct.min()) / data_range
# norm_pct.retain_grad()
# norm_ref_ct.retain_grad()
clamp_pct = torch.clamp(pct, min=0, max=100) / 100 * 255
clamp_ref_ct = torch.clamp(ref_ct, min=0, max=100) / 100 * 255
clamp_pct.retain_grad()
clamp_ref_ct.retain_grad()

ssim_loss = SSIMLoss(spatial_dims=3, data_range=255)
# loss = ssim_loss(pct - pct.min(), ref_ct - ref_ct.min())
norm_loss = ssim_loss(clamp_pct, clamp_ref_ct)
# print("before norm:", loss.item(), "after norm:", norm_loss.item())
print("after norm:", norm_loss.item())

# norm_loss = data_range.item() * norm_loss
norm_loss.backward()
# print(norm_pct.grad.mean(), norm_pct.grad.std(), norm_ref_ct.grad.mean(), norm_ref_ct.grad.std())
print(pct.grad.mean(), pct.grad.std(), ref_ct.grad.mean(), ref_ct.grad.std())
print(clamp_pct.grad.mean(), clamp_pct.grad.std(), clamp_ref_ct.grad.mean(), clamp_ref_ct.grad.std())
print(torch.abs(pct - ref_ct).mean().item(), torch.abs(pct - ref_ct)[head_mask].mean().item())

from monai.metrics.regression import compute_ssim_and_cs, compute_ms_ssim
ssim_loss_2d = SSIMLoss(spatial_dims=2, data_range=255, reduction="none")
clamp_pct_2d = clamp_pct[:, :, 240:460, 176:423, 24]
clamp_ref_ct_2d = clamp_ref_ct[:, :, 240:460, 176:423, 24]
clamp_pct_2d_mask = clamp_pct[..., 24]
clamp_ref_ct_2d_mask = clamp_ref_ct[..., 24]
print(ssim_loss_2d(clamp_pct_2d_mask, clamp_ref_ct_2d_mask).item())
mask_2d = head_mask[:, :, ..., 24]
print(ssim_loss_2d(clamp_pct_2d, clamp_ref_ct_2d))
ssim, _ = compute_ssim_and_cs(
    clamp_pct_2d_mask, 
    clamp_ref_ct_2d_mask, 
    spatial_dims=2, 
    kernel_size=(11, 11),
    kernel_sigma=(1.5, 1.5),
    kernel_type='gaussian',
    data_range=255, 
    k1=0.01, 
    k2=0.03, 
)
ssim = F.pad(ssim, (5, 5, 5, 5), mode='constant', value=0)
print(ssim.shape, 1 - ssim[mask_2d].mean().item())

a = torch.randn(4, 1, 256, 256, 32)
b = torch.randn(4, 1, 256, 256, 32)


ssim, _ = compute_ssim_and_cs(
    a, 
    b, 
    spatial_dims=3, 
    kernel_size=(11, 11, 11),
    kernel_sigma=(1.5, 1.5, 1.5),
    kernel_type='gaussian',
    data_range=1, 
    k1=0.01, 
    k2=0.03, 
)
print(ssim.shape, a.min(dim=1).min(dim=1).min(dim=1), a.max(dim=0), b.min(dim=0), b.max(dim=0))
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


# directory = 'data/ct_reg2_mr_betneck/ct_reg2_mr'  # Replace with the actual directory path

# wmin = 999
# wmax = 0
# for filename in os.listdir(directory):
#     if filename.endswith('.nii.gz'):
#         file_path = os.path.join(directory, filename)
#         data = nib.load(file_path).get_fdata()
#         data_min = data.min()
#         data_max = data.max()
#         data_max = np.percentile(data, 99.995)
#         print(f"File: {filename}, Min: {data_min}, Max: {data_max}")

#         if data_min < wmin:
#             wmin = data_min
        
#         if data_max > wmax:
#             wmax = data_max

# print(wmin, wmax)