import torch.nn as nn

from models.generators.dyunet import DynUNet

from models.modules.upsample import UnetUpBlock


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters, strides, kernel_size, upsample_kernel_size, use_resblock=False):
        super().__init__()
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            strides=strides,
            kernel_size=kernel_size,
            upsample_kernel_size=upsample_kernel_size,
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            res_block=use_resblock,
            upsample_block=UnetUpBlock,
        )
    
    def forward(self, x):
        return self.model(x)