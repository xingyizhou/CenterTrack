from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from .backbones.dla import dla34
from .backbones.resnet import Resnet
from .backbones.mobilenet import MobileNetV2
from .necks.dlaup import DLASeg
from .necks.msraup import MSRAUp

backbone_factory = {
  'dla34': dla34,
  'resnet': Resnet,
  'mobilenet': MobileNetV2
}

neck_factory = {
  'dlaup': DLASeg,
  'msraup': MSRAUp
}

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class GenericNetwork(nn.Module):
    def __init__(self, num_layers, heads, head_convs, num_stacks=1, opt=None):
        super(GenericNetwork, self).__init__()
        print('Using generic model with backbone {} and neck {}'.format(
          opt.backbone, opt.neck))
        # assert (not opt.pre_hm) and (not opt.pre_img)
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.opt = opt
        self.backbone = backbone_factory[opt.backbone](opt=opt)
        channels = self.backbone.channels
        self.neck = neck_factory[opt.neck](opt=opt, channels=channels)
        last_channel = self.neck.out_channel
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x, pre_img=None, pre_hm=None):
      y = self.backbone(x, pre_img, pre_hm)
      feats = self.neck(y)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
