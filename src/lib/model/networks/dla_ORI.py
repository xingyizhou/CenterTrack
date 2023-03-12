from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .base_model import BaseModel

try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('import DCN failed')
    DCN = None


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False,
                 opt=None):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        if opt.pre_img:
            self.pre_img_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        if opt.pre_hm:
            self.pre_hm_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
                    padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')
    else:
        print('Warning: No ImageNet pretrain!!')
    return model

def dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla102', hash='d94d9790')
    return model

def dla46_c(pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla46_c', hash='2bfd52c3')
    return model


def dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla46x_c', hash='d761bae7')
    return model


def dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60', hash='24839fc4')
    return model


def dla60x(pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60x', hash='d15cacda')
    return model


def dla102x(pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla102x', hash='ad62be81')
    return model


def dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla102x2', hash='262837b6')
    return model


def dla169(pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla169', hash='0914e092')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Conv(nn.Module):
    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)


class GlobalConv(nn.Module):
    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)),
            nn.Conv2d(cho, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))))
        gcr = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))),
            nn.Conv2d(cho, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)))
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = node_type[0](c, o)
            node = node_type[1](o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, 
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


DLA_NODE = {
    'dcn': (DeformConv, DeformConv),
    'gcn': (Conv, GlobalConv),
    'conv': (Conv, Conv),
}

class DLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLASeg, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        print('Using node type:', self.node_type)
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        x = self.base(x, pre_img, pre_hm)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]
