import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
###############################################################################
# Functions
###############################################################################

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False, opt=None):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    netG = AttenionNet()

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    return netG


def define_d(gpu_ids=[], skip=False, opt=None):
    netE = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netE = AttenionNet()

    if len(gpu_ids) > 0:
        netE.cuda(device=gpu_ids[0])
        netE = torch.nn.DataParallel(netE, gpu_ids)
    return netE


def define_H(gpu_ids=[], skip=False, opt=None):
    netE = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netE = illuminationNet(opt, skip)

    if len(gpu_ids) > 0:
        netE.cuda(device=gpu_ids[0])
        netE = torch.nn.DataParallel(netE, gpu_ids)
    return netE

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class AttenionNet(torch.nn.Module):
    def __init__(self):
        super(AttenionNet, self).__init__()

        self.fe1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.fe2 = torch.nn.Conv2d(64, 64, 3, 1, 1)

        self.sAtt_1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = torch.nn.Conv2d(64 * 2, 64, 1, 1, bias=True)
        self.sAtt_3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.sAtt_4 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_5 = torch.nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.sAtt_L1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_L2 = torch.nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.sAtt_L3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, alignedframe):

        # feature extraction
        att = self.lrelu(self.fe1(alignedframe))
        att = self.lrelu(self.fe2(att))

        # spatial attention
        att = self.lrelu(self.sAtt_1(att))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L,
                              size=[att.size(2), att.size(3)],
                              mode='bilinear', align_corners=False)


        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att,
                            size=[alignedframe.size(2), alignedframe.size(3)],
                            mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        # att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        return att


class illuminationNet(nn.Module):
    def __init__(self, opt, skip):
        super(illuminationNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)



        self.opt = opt
        self.skip = skip

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # 3
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        # 5
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=4, dilation=4)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        # 8
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=8, dilation=8)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU(inplace=True)
        # 9
        self.conv9 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv9 = nn.Conv2d(128, 128, 3, stride=1, padding=16, dilation=16)
        self.bn9 = nn.BatchNorm2d(128)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv25 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn25 = nn.BatchNorm2d(128)
        self.relu25 = nn.ReLU(inplace=True)
        # 26
        self.conv26 = nn.Conv2d(128, 3, 1, stride=1, padding=0)
        self.bn26 = nn.BatchNorm2d(3)

        if self.opt.tanh:
            self.tanh = nn.Sigmoid()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size * block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).resize(batch_size, s_height, s_width,
                                                                                     s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input):
        flag = 0
        input = torch.tensor(input)


        x = self.relu1(self.bn1(self.conv1(input)))

        x = self.relu3(self.bn3(self.conv3(x)))

        res1 = x  # c3 output

        x = self.bn4(self.conv4(x))  # r4
        x = self.relu4(x)

        x = self.bn5(self.conv5(x))  # fr5
        x = self.relu5(x + res1)  # tr5
        res3 = x

        x = self.bn8(self.conv8(x))
        x = self.relu8(x)

        x = self.bn9(self.conv9(x))
        x = self.relu9(x)
        res7 = x
        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(x)

        if self.opt.tanh:
            latent = self.tanh(latent)
        output = input / (latent + 0.00001)
        return latent, output