# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _nms, _topk, _sigmoid
import torch.nn.functional as F
from utils.image import draw_umich_gaussian

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _neg_loss(pred, gt):
  ''' Reimplemented focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self, opt=None):
    super(FastFocalLoss, self).__init__()
    self.only_neg_loss = _only_neg_loss

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    neg_loss = self.only_neg_loss(out, target)
    pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    ffloss =  - (pos_loss + neg_loss) / num_pos
    return ffloss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class WeightedBCELoss(nn.Module):
  def __init__(self):
    super(WeightedBCELoss, self).__init__()
    self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, output, mask, ind, target):
    # output: B x F x H x W
    # ind: B x M
    # mask: B x M x F
    # target: B x M x F
    pred = _tranpose_and_gather_feat(output, ind) # B x M x F
    loss = mask * self.bceloss(pred, target)
    loss = loss.sum() / (mask.sum() + 1e-4)
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

class SegDiceLoss(nn.Module):
    def __init__(self,feat_channel):
        super(SegDiceLoss, self).__init__()
        self.in_channels=feat_channel
        self.channels=feat_channel
        self.num_layers = 3

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                weight_nums.append((self.in_channels + 2) * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(self, mask_feats, conv_weight, ind, mask):
        """
        Arguments:
          seg_feats: B x Channel x H x W
          ind, mask: B x max_objs
        """
        n_inst = torch.sum(mask)
        N, _, H, W = mask_feat.size() # batch x channels x H x W

        conv_weights = _tranpose_and_gather_feat(conv_weight, ind) # batch x max_objs x dim

        im_inds = torch.tensor([ind.new_ones(ind.size(1)) * b for b in range(N)]).to(ind.device)
        im_inds = torch.masked_select(im_inds, mask)
        mask_head_params = torch.masked_select(conv_weights, mask) # n_inst x channels
        inst_ind = torch.masked_select(ind, mask) # n_inst 
 
        x,y = inst_ind%W,inst_ind/W
        x_range = torch.arange(W).float().to(device=mask_feat.device)
        y_range = torch.arange(H).float().to(device=mask_feat.device)
        y_grid, x_grid = torch.meshgrid([y_range, x_range])
        coords_map = torch.stack((x_grid, y_grid), dim=1)
        inst_locations = torch.stack((x, y))

        relative_coords = inst_locations.reshape(-1, 1, 2) - coords_map.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        #soi = self.sizes_of_interest.float()[instances.fpn_levels]
        relative_coords = relative_coords# / soi.reshape(-1, 1, 1)
        relative_coords = relative_coords.to(dtype=mask_feats.dtype)

        mask_head_inputs = torch.cat([
            relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
        ], dim=1)


        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        return mask_logits

    def forward(self, seg_feat, conv_weight, mask, ind, target):
        inst_target = torch.masked_select(target, mask)   
        seg_logits = self.mask_heads_forward_with_coords(
                    seg_feats, conv_weight, ind, mask)
        seg_scores = mask_logits.sigmoid()
        mask_losses = dice_coefficient(seg_scores, inst_target)     

        return hm_loss/batch_size

class MTLoss(nn.Module):
    def __init__(self, heads):
        super(MTLoss, self).__init__()
        self.heads = heads
        self.log_vars = {h: nn.Parameter(torch.zeros((1,), requires_grad=True)) for h in heads}
    def forward(self, losses):
        loss = 0
        for h in self.heads:
          if losses[h] == 0:
            continue
          log_var = self.log_vars[h]
          log_var = log_var.to(losses[h].device)
          loss += losses[h] * torch.exp(-log_var) + log_var
        return loss

class SchLoss(nn.Module):
    def __init__(self,feat_channel, opt):
        super(SchLoss, self).__init__()
        self.feat_channel=feat_channel
        self.hm_crit = FastFocalLoss(opt)

    def forward(self, sch_feat, conv_weight, mask, pre_ind, target, ind, kmf_ind=None):
        """
        Arguments:
          sch_feats, target: B x N x H x W
          pre_ind, ind, mask: B x M
        """
        hm_loss=0.
        batch_size = sch_feat.size(0)
        weight = _tranpose_and_gather_feat(conv_weight, pre_ind)
        h,w = sch_feat.size(-2), sch_feat.size(-1)
        if kmf_ind is not None:
          x,y = kmf_ind%w,kmf_ind/w
        else:
          x,y = pre_ind%w,pre_ind/w
        x_range = torch.arange(w).float().to(device=sch_feat.device)
        y_range = torch.arange(h).float().to(device=sch_feat.device)
        hm = torch.zeros_like(target)
        y_grid, x_grid = torch.meshgrid([y_range, x_range])
        for i in range(batch_size):
          num_obj = target[i].size(0)
          conv1w,conv1b,conv2w,conv2b,conv3w,conv3b= \
              torch.split(weight[i,:num_obj],[(self.feat_channel+2)*self.feat_channel,self.feat_channel,
                                        self.feat_channel**2,self.feat_channel,
                                        self.feat_channel,1],dim=-1)
          y_rel_coord = (y_grid[None,None] - y[i,:num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
          x_rel_coord = (x_grid[None,None] - x[i,:num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
          feat = sch_feat[i][None].repeat([num_obj,1,1,1])
          feat = torch.cat([feat,x_rel_coord, y_rel_coord],dim=1).view(1,-1,h,w)

          conv1w=conv1w.contiguous().view(-1,self.feat_channel+2,1,1)
          conv1b=conv1b.contiguous().flatten()
          feat = F.conv2d(feat,conv1w,conv1b,groups=num_obj).relu()

          conv2w=conv2w.contiguous().view(-1,self.feat_channel,1,1)
          conv2b=conv2b.contiguous().flatten()
          feat = F.conv2d(feat,conv2w,conv2b,groups=num_obj).relu()

          conv3w=conv3w.contiguous().view(-1,self.feat_channel,1,1)
          conv3b=conv3b.contiguous().flatten()
          hm[i] = F.conv2d(feat,conv3w,conv3b,groups=num_obj).squeeze()

        hm = _sigmoid(hm)
        cat = torch.zeros_like(ind)
        hm_loss = self.hm_crit(hm.view(-1, 1, h, w), target.view(-1, 1, h, w), ind.view(-1, 1), mask.view(-1, 1), cat.view(-1, 1))

        #hm_loss = hm_loss * 0 if torch.sum(mask) <= 0 else hm_loss / torch.sum(mask) 

        return hm_loss