# encoding=utf-8

import numpy as np
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# add for shape-preserving Loss
from collections import namedtuple
# from pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
cudnn.benchnark=True
from Generation_snow_style_S_adapt_stylegan.modules import *
from torch.nn import AvgPool2d, Conv1d, Conv2d, Embedding, LeakyReLU, Module

from models.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer
from models.skip_transformer import SkipTransformer

from Generation_snow_style_S_adapt_stylegan.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


neg = 0.01
neg_2 = 0.2


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        Conv = EqualConv1d if use_eql else nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style) #[24,128,2048]
        gamma, beta = style.chunk(2, 1) #chunk(2,1)沿1轴分两份 [24,64,2048]

        out = self.norm(input)
        out = gamma * out + beta

        return out


class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k][24,6,2048,10]
        w = self.conv_w(x[:, C:, :, :]) #[24,64,2048,10]
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk [24,64,2048,10]
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1] [24,64,2048,1]

        x = x.squeeze(3)  # BxCxN [24,64,2048]

        return x

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class EqualLinear_style(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1,knn=None, point = 8):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius

        self.mlp_ps = MLP_CONV(in_channel=dim_feat, layer_dims=[256, 128])
        self.ps = nn.ConvTranspose1d(128, dim_feat, up_factor, up_factor, bias=False)   # point-wise splitting
        # self.up_sampler = up_sample(point, scale_factor=up_factor)
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim_feat*2, hidden_dim=dim_feat*2, out_dim=dim_feat)


    def forward(self, pcd_prev, feat):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev) 点
            feat: Tensor, (B, dim_feat, N_prev) 特征
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        point_cloud = pcd_prev #[bs,3,256]
        H = feat
        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        return K_curr + H_up

class ModulatedConv3d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1


        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2


        fan_in = in_channel * kernel_size
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size)
        )

        self.modulation = EqualLinear_style(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style): #[bs,512,64] [bs,512]
        batch, in_channel, points = input.shape #[12,512,64]

        style = self.modulation(style).view(batch, 1, in_channel,  1) #[2,1,512,1]
        weight = self.scale * self.weight * style #[bs,512,512,3]

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1,  1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size
        )

        input = input.view(1, batch * in_channel, points)
        out = F.conv1d(input, weight, padding=self.padding, groups=batch) #[1,bs*512,64]
        _, _, points = out.shape
        out = out.view(batch, self.out_channel, points)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, points = image.shape
            noise = image.new_empty(batch, 1, points).normal_()

        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=64):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv3d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style) #[bs,512,64]
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out
    
class up_sample(nn.Module):
    def __init__(self, dim, scale_factor = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.up = nn.Sequential(
            nn.Conv1d(dim, int(dim*self.scale_factor), 1)
        )
    def forward(self, pts):
        pts = pts.permute(0, 2, 1).contiguous()
        pts = self.up(pts)
        return pts.permute(0, 2, 1).contiguous()


class ToPOINT(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], point = 8):
        super().__init__()

        self.conv = ModulatedConv3d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1))
        # if learnable_upsample:
        self.upsample = nn.Upsample(scale_factor=2) ###采样方法为默认
        # self.upsample = up_sample(point, scale_factor = 2)

    def forward(self, input, style, skip=None):
        out = self.conv(input, style) #【bs,3,64】
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out,skip


class Generator(nn.Module):
    def __init__(self, opts, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(Generator, self).__init__()
        self.opts = opts
        self.np = opts.np
        self.nk = opts.nk // 2
        self.nz = opts.nz
        style_dim = opts.nz
        # softmax = opts.softmax
        self.off = opts.off
        self.use_attn = opts.attn
        self.use_head = opts.use_head

        self.n_mlp = opts.n_mlp
        self.lr_mlp = opts.lr_mlp
        self.n_latent = opts.n_latent

        layers = [PixelNorm()]

        for i in range(self.n_mlp):
            layers.append(
                EqualLinear_style(
                    self.nz, self.nz, lr_mul=self.lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        # self.img_mapping = self.spatial_mapping()
        # self.pts_mapping = self.style_mapping()

        # channel_multiplier = opts.channel_multiplier

        self.channels = {
            # 2:512, ### ###const2
            # 4: 512, ###const4
            8: 512,
            16: 512,###const16
            32: 512, ###const32
            64: 512,
            128: 256,
            256: 128,
            512: 64,
            1024: 32,
            2048: 16,
            # 1024: 128 * channel_multiplier,
            # 2048: 64 * channel_multiplier,
        }
        blur_kernel = [1, 3, 3, 1]

        # self.input = ConstantInput(self.channels[2],size=2) ###const2
        # self.conv1 = StyledConv(
        #     self.channels[2], self.channels[2], 3, self.nz, blur_kernel=blur_kernel
        # )
        # self.to_point1 = ToPOINT(self.channels[2], self.nz, upsample=False)

        # self.input = ConstantInput(self.channels[4],size=4) ###const4
        # self.conv1 = StyledConv(
        #     self.channels[4], self.channels[4], 3, self.nz, blur_kernel=blur_kernel
        # )
        # self.to_point1 = ToPOINT(self.channels[4], self.nz, upsample=False)

        self.input = ConstantInput(self.channels[8],size=8) ###const8
        self.conv1 = StyledConv(
            self.channels[8], self.channels[8], 3, self.nz, blur_kernel=blur_kernel
        )
        self.to_point1 = ToPOINT(self.channels[8], self.nz, upsample=False, point=8)

        # self.input = ConstantInput(self.channels[16],size=16) ###const16
        # self.conv1 = StyledConv(
        #     self.channels[16], self.channels[16], 3, self.nz, blur_kernel=blur_kernel
        # )
        # self.to_point1 = ToPOINT(self.channels[16], self.nz, upsample=False)

        ###32
        # self.input = ConstantInput(self.channels[32],size=32) ###const8
        # self.conv1 = StyledConv(
        #     self.channels[32], self.channels[32], 3, self.nz, blur_kernel=blur_kernel
        # )
        # self.to_point1 = ToPOINT(self.channels[32], self.nz, upsample=False)

        self.convs = nn.ModuleList()
        # self.upsamples = nn.ModuleList()
        self.to_points = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channel = self.channels[16]

        # for i in range(2, 11 + 1):  ###const2
        # for i in range(3,11+1): ###const4
        for i in range(4, 11 + 1):  ###const8
        # for i in range(5, 11 + 1):  ###const16
        # for i in range(6, 11 + 1):  ###const32
            out_channel = self.channels[2 ** i]

            self.decoder.append(
                SPD(dim_feat=in_channel, up_factor=up_factors[0], i=0, radius=radius, knn=64, point=2 ** (i-1))
            )

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_points.append(ToPOINT(out_channel, style_dim, point=2 ** (i-1)))

            in_channel = out_channel

    def forward(self, z, noise = None, return_feats=False ):

        # B,N,_ = x.size() #[bs,64,3]
        arr_pcd = []
        up_pcd = []

        styles = self.style(z)
        noise = [None] * (self.n_latent-1)

        latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1)

        out = self.input(latent) #[bs,512,64]
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip,_ = self.to_point1(out, latent[:, 1])

        ###const8
        pc1 = skip.permute(0,2,1).contiguous()
        arr_pcd.append(pc1)

        i = 1
        for decoder, conv1, conv2, noise1, noise2, to_point in zip(
            self.decoder,self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_points
        ):
            out = decoder(skip, out)  # [bs,512,128]
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip, up_point = to_point(out, latent[:, i + 2], skip)  # [bs,3,128]
            pc2 = skip.permute(0, 2, 1).contiguous()
            arr_pcd.append(pc2)
            up_pcd.append(up_point)

            i += 2

        return arr_pcd, up_pcd
    
    # def style_mapping(self):
    #     self.num_style_mapping = self.num_spatial_mapping
    #     layers=[PixelNorm(self.pixel_norm_op_dim)]
    #     for i in range(self.num_style_mapping):
    #         layers.append(
    #             EqualLinear(
    #                 in_dim=self.nz, out_dim=self.nz, lr_mul=self.lr_mlp, activation='fused_lrelu'
    #             )
    #         )
    #     return nn.Sequential(*layers)
    
    # def spatial_mapping(self):
    #     self.pixel_norm_op_dim = 2
    #     num_region = 1
    #     self.num_spatial_mapping = int(16/num_region) 
    #     layers=[PixelNorm(self.pixel_norm_op_dim)]
    #     for i in range(self.num_spatial_mapping):
    #         layers.append(
    #             EqualLinear(
    #                 in_dim=self.nz, out_dim=self.nz, lr_mul=self.lr_mlp, activation='fused_lrelu'
    #             )
    #         )
    #     return nn.Sequential(*layers)
    
# class PointCloudNorm(nn.Module): 
#     def __init__(self, pointcloud_norm_op_dim):
#         super().__init__()
#         self.pointcloud_norm_op_dim = pointcloud_norm_op_dim

#     def forward(self, input):
#         # input为点云数据，假设每个点的坐标在最后一维
#         squared = torch.sum(input ** 2, dim=self.pointcloud_norm_op_dim, keepdim=True)  # 计算每个点坐标的平方和
#         norm = torch.rsqrt(squared + 1e-8)  # 求平方和的倒数的平方根
#         return input * norm  # 点云数据与归一化系数相乘



