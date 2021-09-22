from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('import DCN failed')
    DCN = None
BN_MOMENTUM = 0.1

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.opt = opt
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            print('Building head:', head, 'with:', head_convs[head])
            classes = self.heads[head]
            head_conv = head_convs[head] # [256]
            if len(head_conv) > 0:
              if opt.head_DCN and 'tracking' in head:
                print('using DCN.')
                conv = DCN(last_channel, head_conv[0], kernel_size=head_kernel, stride=1, padding=head_kernel // 2, dilation=1, deformable_groups=4)
              else:
                conv = nn.Conv2d(last_channel, head_conv[0],
                                kernel_size=head_kernel, 
                                padding=head_kernel // 2, bias=True)

              bn = nn.BatchNorm2d(head_conv[0], momentum=BN_MOMENTUM)
              out = nn.Conv2d(head_conv[-1], classes, 
                kernel_size=1, stride=1, padding=0, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                    convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                                kernel_size=1, bias=True))
              if len(convs) == 1:
                if opt.head_DCN:
                  fc = nn.Sequential(conv, bn, nn.ReLU(inplace=True), out)
                else:
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
        

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None, kmf_att=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None, kmf_att=None):
      # print('\nforward')
      if (pre_img is not None) and (self.opt.sch_track):
        feats = self.img2feats(x)
        pre_feats = self.img2feats(pre_img[:, 0, :])
        if torch.isnan(feats[0]).any() or torch.isnan(pre_feats[0]).any():
          print('\npre_feats', pre_feats[-1][-1][-1])
          print('feats', feats[-1][-1][-1])
          #print('pre_img', pre_img[:, 0, :])
          print('pre_img', pre_img[:, 0, :].shape, torch.max(pre_img[:, 0, :]), torch.min(pre_img[:, 0, :]))
          print('x', x.shape, torch.max(x), torch.min(x))

      elif (pre_hm is not None) or (pre_img is not None) or (kmf_att is not None):
        kmf_att_in = None if self.opt.kmf_layer_out else kmf_att
        feats = self.imgpre2feats(x, pre_img, pre_hm, kmf_att_in)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
            if 'sch' in head:
              assert pre_feats is not None
              z.append(self.__getattr__(head)(pre_feats[s]))
            else:
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              if 'sch_weight' in head:
                assert pre_feats is not None
                z[head] = self.__getattr__(head)(pre_feats[s])
              else:
                z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
