# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotation_conv_utils import  FeatureSelectionModule
from external.carafe.carafe import CARAFEPack

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
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

class make_dynamic_refine_head_cls(nn.Module):
    def __init__(self, cnv_dim, curr_dim, out_dim, ks=1, r=4, eps=0.1, init_bias=True, refine=True, dense_w=False):
        super(make_dynamic_refine_head_cls, self).__init__()
        self.out_dim = out_dim
        self.curr_dim = curr_dim
        self.r = r
        self.ks = ks
        self.refine = refine
        self.eps = eps
        self.base_conv = convolution(3, cnv_dim, curr_dim, with_bn=False)
        self.base_head = nn.Conv2d(curr_dim, out_dim, (1, 1))
        self.dense_w = dense_w
        if dense_w:
            self.refine_conv = CARAFEPack(cnv_dim, 1, up_kernel=ks, compressed_channels=curr_dim // r,
                                        normalized=False)
        else:
            self.refine_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                convolution(1, cnv_dim, cnv_dim // r, with_bn=False),
                nn.Conv2d(cnv_dim // r, int(ks*ks*curr_dim*curr_dim), 1)
            )
        # self.att = nn.Sequential(
        #     convolution(1, cnv_dim, cnv_dim // r),
        #     nn.Conv2d(cnv_dim // r, 1, 1),
        #     nn.Sigmoid()
        # )
        self._init_weight()
        if init_bias:
            self.base_head.bias.data.fill_(-2.19)

    def _init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def gene_refine_feat(self,x, base_conv):
        if self.dense_w:
            refine_conv = self.refine_conv(x, base_conv)
        else:
            ref_kernel = self.refine_conv(x)
            ref_kernel = ref_kernel.reshape(-1, self.curr_dim, self.curr_dim, self.ks, self.ks).contiguous()
            refine_convs = [F.conv2d(base_conv[i, ...].unsqueeze(0), ref_kernel[i, ...], bias=None) for i in
                         range(base_conv.size(0))]
            refine_conv = torch.cat(refine_convs, dim=0)
        return refine_conv

    def forward(self, x):
        base_conv = self.base_conv(x)
        if not self.refine:
            base_head = self.base_head(base_conv)
            return base_head
        refine_conv = self.gene_refine_feat(x, base_conv)
        # modular = self.eps*self.att(base_conv)
        modular = self.eps
        refine_norm = modular*refine_conv/(torch.norm(refine_conv, p=2, dim=1).unsqueeze(1)+1e-12)
        final_conv = base_conv + refine_norm*torch.norm(base_conv, p=2, dim=1).unsqueeze(1)
        return self.base_head(final_conv)


class make_dynamic_refine_head_reg(nn.Module):
    def __init__(self, cnv_dim, curr_dim, out_dim, ks=1, r=4, eps=0.1, refine=True, dense_v=False):
        super(make_dynamic_refine_head_reg, self).__init__()
        self.out_dim = out_dim
        self.curr_dim = curr_dim
        self.r = r
        self.ks = ks
        self.refine = refine
        self.eps = eps
        self.dense_v = dense_v
        self.base_conv = convolution(3, cnv_dim, curr_dim, with_bn=False)
        self.base_head = nn.Conv2d(curr_dim, out_dim, (1, 1))
        if self.dense_v:
            self.refine_conv = CARAFEPack(cnv_dim, 1, up_kernel=ks, compressed_channels=curr_dim // r,
                                        normalized=False)
        else:
            self.refine_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                convolution(1, cnv_dim, cnv_dim // r, with_bn=False),
                nn.Conv2d(cnv_dim // r, int(ks*ks*curr_dim*out_dim), 1)
            )
        self._init_weight()

    def _init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def gene_refine_value(self,x, base_conv):
        if self.dense_w:
            refine_conv = self.refine_conv(x,base_conv)
        else:
            ref_kernel = self.refine_conv(x)
            ref_kernel = ref_kernel.reshape(-1, self.curr_dim, self.curr_dim, self.ks, self.ks).contiguous()
            refine_convs = [F.conv2d(base_conv[i, ...].unsqueeze(0), ref_kernel[i, ...], bias=None) for i in
                         range(base_conv.size(0))]
            refine_conv = torch.cat(refine_convs, dim=0)
        return refine_conv

    def forward(self, x):
        base_conv = self.base_conv(x)
        base_head = self.base_head(base_conv)
        if not self.refine:
            return base_head
        ref_value = self.gene_refine_value(x, base_conv)
        ref_ratio = self.eps *torch.tanh(ref_value)
        # meta_att = self.att(avgpool)
        total_head = (ref_ratio+1.0)*base_head
        return total_head


def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
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
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
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
        kp_layer=residual,fsm=False, drmc=False, drmr=False, only_ls=False
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self.heads     = heads

        self.last_bra = only_ls

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
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
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## feature reassemble
        self.fea_sel = fsm
        self._fea_sel = None
        if self.fea_sel:
            self._fea_sel = nn.ModuleList([FeatureSelectionModule(cnv_dim, rot=True)
                                            for _ in range(nstack)])

        # keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                if drmc:
                    module = nn.ModuleList([
                        make_dynamic_refine_head_cls(
                            cnv_dim, curr_dim, heads[head], init_bias=True, refine=True, dense_w=False) for _ in range(nstack)
                    ])
                    self.__setattr__(head, module)
                else:
                    module =  nn.ModuleList([
                        make_heat_layer(
                            cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                    ])
                    self.__setattr__(head, module)
                    for heat in self.__getattr__(head):
                        heat[-1].bias.data.fill_(-2.19)
            elif 'wh' in head:
                if drmr:
                    module = nn.ModuleList([
                        make_dynamic_refine_head_reg(
                            cnv_dim, curr_dim, heads[head], refine=True, dense_v=False) for _ in range(nstack)
                    ])
                    self.__setattr__(head, module)
                else:
                    module = nn.ModuleList([
                        make_regr_layer(
                            cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                    ])
                    self.__setattr__(head, module)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)

    def get_angle_for_fsm(self, angle_pre, angle_gt, angle_mask=None, th=0.05):
        diff = torch.abs(angle_gt - angle_pre)
        pre_ind = diff.le(th).float()
        angle = angle_gt * (1 - pre_ind) + angle_pre * pre_ind
        if angle_mask is not None:
            angle_mask = angle_mask.gt(0).float()
            angle = angle * angle_mask + angle_pre * (1-angle_mask)
        return angle.detach()

    def forward(self, image, angle_gt=None, angle_gt_mask=None, wh_gt=None):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs  = []

        for ind in range(self.nstack):
            kp_, cnv_  = self.kps[ind], self.cnvs[ind]
            kp  = kp_(inter)
            cnv = cnv_(kp)
            out = {}
            # out['cnv'] = cnv
            if self.fea_sel:
                fea_sel_ = self._fea_sel[ind]
                if angle_gt is None:
                    raise Exception('angle gt is need for fsm module')
                else:
                    layer = self.__getattr__('angle')[ind]
                    angle_pred = layer(cnv)
                    angle_for_fsm = self.get_angle_for_fsm(angle_pred.clone(), angle_gt.clone(), angle_gt_mask)

                sel_cnv, att = fea_sel_(cnv, angle_for_fsm)
                out['att'] = att
                # out['sel_cnv'] = sel_cnv

            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                if self.fea_sel:
                    if 'angle' not in head:
                        y = layer(sel_cnv)
                    else:
                        y = layer(cnv)
                else:
                    y = layer(cnv)
                out[head] = y
            
            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        if self.last_bra:
            return outs[-1:]
        else:
            return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2,fsm=False, rot=False, drmc=False, drmr=False,only_ls=False):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256,
            fsm=fsm, drmc=drmc,drmr=drmr,only_ls=only_ls
        )

def get_large_hourglass_net(num_layers, heads, head_conv, number_stacks=2, fsm=False, drmc=False, drmr=False, only_ls=False):
  model = HourglassNet(heads, number_stacks, fsm=fsm, drmc=drmc, drmr=drmr, only_ls=only_ls)
  return model
