from typing import Any, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.utils import ensure_tuple_rep
from monai.networks.nets import UNet
from monai.networks.blocks import Convolution, ResidualUnit, SubpixelUpsample
from monai.networks.layers import Act, Conv, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.layers.convutils import same_padding
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding

import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.pos_emb = PositionalEmbSineCos(d_model)

    def forward(
        self,
        src,
    ):
        pos = self.pos_emb(src).permute(1, 0, 2)
        b, c, h, w, d = src.shape
        src = src.reshape(b, c, -1).permute(2, 0, 1)

        output = src

        for layer in self.layers:
            output = layer(output, pos)

        return output.permute(1, 2, 0).reshape(b, c, h, w, d)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos):
        
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(
            q, k, value=src, attn_mask=None, key_padding_mask=None
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos):

        out = self.forward_post(src, pos)

        return out


import collections.abc
from itertools import repeat
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


class PositionalEmbSineCos(nn.Module):
    def __init__(self, embed_dim: int, spatial_dims: int = 3, temperature: float = 10000.0):
        super(PositionalEmbSineCos, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dims = spatial_dims
        self.temperature = temperature
    
    def forward(self, x):
        grid_size = x.size()[2:]

        if self.spatial_dims == 2:
            to_2tuple = _ntuple(2)
            grid_size_t = to_2tuple(grid_size)
            h, w = grid_size_t
            grid_h = torch.arange(h, dtype=torch.float32)
            grid_w = torch.arange(w, dtype=torch.float32)

            grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

            if self.embed_dim % 4 != 0:
                raise AssertionError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")

            pos_dim = self.embed_dim // 4
            omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
            omega = 1.0 / (self.temperature**omega)
            out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
            out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
            pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        elif self.spatial_dims == 3:
            to_3tuple = _ntuple(3)
            grid_size_t = to_3tuple(grid_size)
            h, w, d = grid_size_t
            grid_h = torch.arange(h, dtype=torch.float32, device=x.device)
            grid_w = torch.arange(w, dtype=torch.float32, device=x.device)
            grid_d = torch.arange(d, dtype=torch.float32, device=x.device)

            grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing="ij")

            if self.embed_dim % 6 != 0:
                raise AssertionError("Embed dimension must be divisible by 6 for 3D sin-cos position embedding")

            pos_dim = self.embed_dim // 6
            omega = torch.arange(pos_dim, dtype=torch.float32, device=x.device) / pos_dim
            omega = 1.0 / (self.temperature**omega)
            out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
            out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
            out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
            pos_emb = torch.cat(
                [
                    torch.sin(out_w),
                    torch.cos(out_w),
                    torch.sin(out_h),
                    torch.cos(out_h),
                    torch.sin(out_d),
                    torch.cos(out_d),
                ],
                dim=1,
            )[None, :, :]
        else:
            raise NotImplementedError("Spatial Dimension Size {spatial_dims} Not Implemented!")

        return pos_emb


# define Shuffle UNet
class ShuffleUNet(UNet):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        img_size: Union[Sequence[int], int],
        transformer_layers: int = 3,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:

        super().__init__(
            dimensions,
            in_channels,
            out_channels,
            channels,
            strides,
        )

        self.num_res_units = num_res_units
        self.out_size = ensure_tuple_rep([s / (2 ** len(strides)) for s in img_size], self.dimensions)
        self.transformer_layers = transformer_layers
        self.bottleneck_feat_dim = self.channels[-1]
        self.attn_heads = self.bottleneck_feat_dim // 64

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self.get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

        # self.learnable_ssim_lambda_3d = nn.Parameter(torch.tensor(1.0))
        # self.learnable_ssim_lambda_2d_yz = nn.Parameter(torch.tensor(1.0))
        # self.learnable_ssim_lambda_2d_xz = nn.Parameter(torch.tensor(1.0))
        # self.learnable_ssim_lambda_2d_xy = nn.Parameter(torch.tensor(1.0))

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:

        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        decode = nn.Sequential()
        shuffle = SubpixelUpsample(self.dimensions, in_channels, out_channels, strides)

        conv = Convolution(
            self.dimensions,
            out_channels,
            out_channels,
            strides=1,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        decode.add_module('shuffle', shuffle)
        decode.add_module('conv',conv)

        return decode

    def get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        conv = self._get_down_layer(in_channels, out_channels, 1, False)

        if self.transformer_layers == 0:
            return conv
        else:
            bottleneck = nn.Sequential()

            attn_layer = TransformerEncoderLayer(self.bottleneck_feat_dim, self.attn_heads, self.bottleneck_feat_dim * 4)
            transformer_encoder = TransformerEncoder(attn_layer, self.transformer_layers, self.bottleneck_feat_dim)

            bottleneck.add_module('conv', conv)
            bottleneck.add_module('transformer_encoder', transformer_encoder)

            return bottleneck

    def load_weights(self, path):
        if isinstance(path, str):
            saved_state_dict = torch.load(path)
        else:
            saved_state_dict = path
        cur_state_dict = self.state_dict()

        new_state_dict = {}
        for key in cur_state_dict.keys():
            if key in saved_state_dict.keys():
                new_state_dict[key] = saved_state_dict[key]
            else:
                new_state_dict[key] = cur_state_dict[key]

        self.load_state_dict(new_state_dict)