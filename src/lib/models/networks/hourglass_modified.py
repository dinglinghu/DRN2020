# -*- coding:utf-8 -*-
# 修改版的hourglass结构

# -*- coding:utf-8 -*-
# 大型hourglass网络
# ------------------------------------------------------------------------------
# This code is base on
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .corner_pool_utils import center_pool_dcn, center_pool
from .corner_pool_utils import feature_reassemble
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M


class GrouBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(GrouBasicBlock, self).__init__()
        self.conv1 = P4MConvP4M(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 8)
        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 8)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        batch, ch, w, h = x.size()
        cn1 = self.conv1(x)
        cn1 = cn1.view(batch, ch, w, h)
        out = F.relu(self.bn1(cn1))
        cn2 = self.conv2(out)
        cn2 = cn2.view(batch, ch, w, h)
        out = self.bn2(cn2)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def make_grou_conv(in_dim, inter_dim, out_dim):
    return GrouAngleReg(in_dim, inter_dim, out_dim)


class GrouAngleReg(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim, rate=8):
        super(GrouAngleReg, self).__init__()
        self.grou = GrouBasicBlock(in_dim // rate, in_dim // rate)
        self.reg = make_kp_layer(in_dim, inter_dim, out_dim)

    def forward(self, x):
        grou = self.grou(x)
        reg = self.reg(grou)
        return reg


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


def make_merge_layer(dim):
    return MergeUp()


# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def make_inter_layer(dim):
    return residual(3, dim, dim)


def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)


def make_hm_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        center_pool_dcn(cnv_dim),
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


class kp_module(nn.Module):
    def __init__(
            self, n, dims, modules, layer=residual,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
            make_low_layer(
                3, next_dim, next_dim, next_mod,
                layer=layer, **kwargs
            )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2 = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)


class exkp(nn.Module):
    def __init__(
            self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256,
            make_tl_layer=None, make_br_layer=None,
            make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
            make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
            kp_layer=residual, center_pool_flag=True, fea_reasm_flag=True
    ):
        super(exkp, self).__init__()

        self.nstack = nstack
        self.heads = heads

        curr_dim = dims[0]

        # 茎干层
        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        # hourglass结构
        self.kps = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])

        # 构造3*3卷积
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.fea_reasm_flag = fea_reasm_flag
        if self.fea_reasm_flag:
            self.fea_reasm = nn.ModuleList([feature_reassemble(cnv_dim)
                                            for _ in range(nstack)])

        self.center_pool_flag = center_pool_flag
        if self.center_pool_flag:
            self.cps = nn.ModuleList([
                center_pool(cnv_dim, inter_dim=cnv_dim // 2)
                 for _ in range(nstack)
            ])
            # self.cpm = center_pool(cnv_dim, inter_dim=cnv_dim // 2)

        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                # 构建热力图输出头
                module = nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                # for heat in self.__getattr__(head):
                #     heat[-1].bias.data.fill_(-2.19)
            elif 'angle' in head:
                # 构建角度输出头
                module = nn.ModuleList(
                    [nn.Sequential(
                     make_grou_conv(cnv_dim, curr_dim, heads[head]))
                     # make_regr_layer(cnv_dim, curr_dim, heads[head]))
                        for _ in range(nstack)]
                )
                self.__setattr__(head, module)
            else:
                # 构建回归输出头
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs = []

        for ind in range(self.nstack):
            kp_, cnv_ = self.kps[ind], self.cnvs[ind]
            kp = kp_(inter)
            cnv = cnv_(kp)
            # print(cnv.)
            if self.fea_reasm_flag:
                fea_reasm_ = self.fea_reasm[ind]
                cnv = fea_reasm_(cnv)

            if self.center_pool_flag:

                cp = self.cps[ind]
                cp_cnv = cp(cnv)

            # 输出每个阶段的结果
            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                if self.center_pool_flag:
                    y = layer(cp_cnv)
                else:
                    y = layer(cnv)
                # if head == 'hm' and ind == self.nstack - 1:
                    # print(y)
                out[head] = y

            # 添加输出结果
            outs.append(out)

            # 添加中间跳接层
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2, center_pool_flag=True, fea_reasm=True):
        n = 5
        dims = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256,
            center_pool_flag=center_pool_flag,
            fea_reasm_flag=fea_reasm
        )


def get_large_hourglass_net_modified(num_layers, heads, head_conv, center_pool=True, fea_reasm=True):
    model = HourglassNet(heads, 2, False, fea_reasm=fea_reasm)
    print('use center pool')
    return model
