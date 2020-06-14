
import _init_paths
import torch
import torch.nn as nn
from models.networks.DCNv2.dcn_v2 import DCNv2, DCN, dcn_v2_conv
from models.networks.corner_pool_utils import RCN_NEW, RCN_NEW_XV
from models.networks.DCNv2_xv.modules.modulated_deform_conv import ModulatedDeformConv
import numpy as np

kH = 3
kW = 1
kernel = (kH,kW)
pH = 1
pW = 0
padding = (pH,pW)
iH = iW = 3
oH = (iH + 2 * pH - kH)//1 +1
oW = (iW + 2 * pW - kW)//1 +1

deformable_groups = 1
N, inC, inH, inW = 1, 1, 3, 3
outC = 1
def check_mdconv_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(pH, pW),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(pH, pW),
                          bias=True).cuda()

    dcn = ModulatedDeformConv(inC, outC, (kH, kW),
                   stride=1, padding=(pH, pW), dilation=1,
                   groups=1,
                   deformable_groups=deformable_groups, im2col_step=1).cuda()
    pcn = nn.Conv2d(inC, outC, (kH, kW), stride=1, padding=(pH, pW), dilation=1, groups=1).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    mask *= 2
    output_d = dcn(input, offset, mask)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('mdconv zero offset passed with {}'.format(d))
    else:
        print('mdconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())

# check_mdconv_zero_offset()
test_rcn_xv = RCN_NEW_XV(1, 1, kernel, stride=1, padding=padding, bias=False).cuda()

input = torch.arange(0,iH*iW).view(1,1,iH,iW).cuda().float()
input[0,0,2,1] = 9
input[0,0,1,2] = 10
nn.init.constant_(test_rcn_xv.weight, 1.0)
angle = torch.zeros_like(input)
# offset = [0,0,0,0,0,0,0,0,0,0]
offset = [0,0,0,0,0,0]
offset = torch.Tensor(offset).view(2*kH*kW,1)
offset = offset.expand(2*kH*kW,oH*oW).contiguous().view(-1).view(1,2*kH*kW,oH,oW).cuda()
mask = torch.ones(N,kH*kW,oH,oW).cuda()
output_xv = test_rcn_xv(input, angle, offset, mask)

# offset1 = [-2,2,-1,1,0,0,1,-1,2,-2]
# offset1 = [-1,1,0,0,1,-1]
offset1 = [1,1,0,0,-1,-1]
offset1= torch.Tensor(offset1).view(2*kH*kW,1)
offset1 = offset1.expand(2*kH*kW,oH*oW).contiguous().view(1,2*kH*kW,oH,oW).cuda()
output1_xv = test_rcn_xv(input, angle, offset1, mask)


angle1 = torch.ones_like(input)*np.pi*0.5
output1_ang_xv = test_rcn_xv(input, angle1)

angle2 = torch.ones_like(input)*np.pi*1.0
output2_ang_xv = test_rcn_xv(input, angle2)

angle3 = torch.ones_like(input)*np.pi*1.5
output3_ang_xv = test_rcn_xv(input, angle3)


# weight = torch.Tensor([1,0,0,1,0,0,1,0,0]).view(1,1,3,3,).cuda()
# test_rcn.weight.data=weight
# output3 = test_rcn(input, angle)
# output4 = test_rcn(input, angle1)
#
# angle2 = torch.ones_like(input)*np.pi*1.0
# output5 = test_rcn(input, angle2)
#
# angle3 = torch.ones_like(input)*np.pi*1.5
# output6 = test_rcn(input, angle3)

print('done.')