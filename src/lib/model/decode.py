from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
from .utils import _nms, _topk, _topk_channel
from .losses import dice_coefficient
import torch.nn.functional as F


def _update_kps_with_hm(
  kps, output, batch, num_joints, K, bboxes=None, scores=None):
  if 'hm_hp' in output:
    hm_hp = output['hm_hp']
    hm_hp = _nms(hm_hp)
    thresh = 0.2
    kps = kps.view(batch, K, num_joints, 2).permute(
        0, 2, 1, 3).contiguous() # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
    if 'hp_offset' in output or 'reg' in output:
        hp_offset = output['hp_offset'] if 'hp_offset' in output \
                    else output['reg']
        hp_offset = _tranpose_and_gather_feat(
            hp_offset, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_joints, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]
    else:
        hm_xs = hm_xs + 0.5
        hm_ys = hm_ys + 0.5
    
    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
        2).expand(batch, num_joints, K, K, 2)
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3) # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
        batch, num_joints, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_joints, K, 2)        
    mask = (hm_score < thresh)
    
    if bboxes is not None:
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
    else:
      l = kps[:, :, :, 0:1].min(dim=1, keepdim=True)[0]
      t = kps[:, :, :, 1:2].min(dim=1, keepdim=True)[0]
      r = kps[:, :, :, 0:1].max(dim=1, keepdim=True)[0]
      b = kps[:, :, :, 1:2].max(dim=1, keepdim=True)[0]
      margin = 0.25
      l = l - (r - l) * margin
      r = r + (r - l) * margin
      t = t - (b - t) * margin
      b = b + (b - t) * margin
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
      # sc = (kps[:, :, :, :].max(dim=1, keepdim=True) - kps[:, :, :, :].min(dim=1))
    # mask = mask + (min_dist > 10)
    mask = (mask > 0).float()
    kps_score = (1 - mask) * hm_score + mask * \
      scores.unsqueeze(-1).expand(batch, num_joints, K, 1) # bJK1
    kps_score = scores * kps_score.mean(dim=1).view(batch, K)
    # kps_score[scores < 0.1] = 0
    mask = mask.expand(batch, num_joints, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3).contiguous().view(
        batch, K, num_joints * 2)
    return kps, kps_score
  else:
    return kps, kps


def seg_decode(seg_feat, conv_weight, xs, ys, inds,  K):
    ys = ys.squeeze(-1)
    xs = xs.squeeze(-1)
    batch_size = seg_feat.size(0)
    feat_channel = seg_feat.size(1)
    h, w = seg_feat.size(-2), seg_feat.size(-1)
    seg_masks = torch.zeros((batch_size, K,h,w)).to(device=seg_feat.device)
    x_range = torch.arange(w).float().to(device=seg_feat.device)
    y_range = torch.arange(h).float().to(device=seg_feat.device)
    y_grid, x_grid = torch.meshgrid([y_range, x_range])
    weight = _tranpose_and_gather_feat(conv_weight, inds)
    for i in range(batch_size):
        conv1w, conv1b, conv2w, conv2b, conv3w, conv3b = \
            torch.split(weight[i], [(feat_channel + 2) * feat_channel, feat_channel,
                                              feat_channel ** 2, feat_channel,
                                              feat_channel, 1], dim=-1)
        y_rel_coord = (y_grid[None, None] - ys[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) / 128.
        x_rel_coord = (x_grid[None, None] - xs[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) / 128.
        feat = seg_feat[i][None].repeat([K, 1, 1, 1])

        feat = torch.cat([feat, x_rel_coord, y_rel_coord], dim=1)
        feat = feat.view(1, -1, h, w)

        conv1w = conv1w.contiguous().view(-1, feat_channel + 2, 1, 1)
        conv1b = conv1b.contiguous().flatten()
        feat = F.conv2d(feat, conv1w, conv1b, groups=K).relu()

        conv2w = conv2w.contiguous().view(-1, feat_channel, 1, 1)
        conv2b = conv2b.contiguous().flatten()
        feat = F.conv2d(feat, conv2w, conv2b, groups=K).relu()

        conv3w = conv3w.contiguous().view(-1, feat_channel, 1, 1)
        conv3b = conv3b.contiguous().flatten()
        feat = F.conv2d(feat, conv3w, conv3b, groups=K).sigmoid().squeeze()
        seg_masks[i] = feat

    return seg_masks

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

class CondInst(nn.Module):
    def __init__(self,feat_channel):
        super(CondInst, self).__init__()
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

    def mask_heads_forward_with_coords(self, mask_feats, conv_weight, ind, mask=None):
        """
        Arguments:
          seg_feats: B x Channel x H x W
          ind, mask: B x max_objs
        """
        n_inst = torch.sum(mask)
        N, _, H, W = mask_feats.size() # batch x channels x H x W
        max_objs = ind.size(1)

        conv_weights = _tranpose_and_gather_feat(conv_weight, ind) # batch x max_objs x dim

        #im_inds = torch.tensor([ind.new_ones(ind.size(1)) * b for b in range(N)]).to(ind.device)
        im_inds = torch.arange(N).unsqueeze(1).expand(N, max_objs).to(ind.device)
        im_inds = im_inds[mask]
        mask_head_params = conv_weights[mask] # n_inst x channels
        inst_ind = ind[mask] # n_inst 
 
        x,y = inst_ind%W,inst_ind/W
        x_range = torch.arange(W).float().to(device=mask_feats.device)
        y_range = torch.arange(H).float().to(device=mask_feats.device)
        y_grid, x_grid = torch.meshgrid([y_range, x_range])
        coords_map = torch.stack((x_grid, y_grid), dim=1).float()
        inst_locations = torch.stack((x, y)).float()

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

    def forward(self, seg_feat, conv_weight,  ind, mask=None, target=None, is_train=True):
        if mask is None:
          mask = torch.ones_like(ind)
        if torch.sum(mask) == 0:
          return torch.sum(mask)

        batch_size, k = ind.size()
        _, _, H, W = seg_feat.size()

        mask = mask.byte() 
        seg_logits = self.mask_heads_forward_with_coords(
                    seg_feat, conv_weight, ind, mask)
        seg_scores = seg_logits.sigmoid()
        if is_train:
          inst_target = target[mask]
          seg_losses = dice_coefficient(seg_scores, inst_target)     
          loss_seg = seg_losses.mean()
          return loss_seg
        else:
          return seg_scores.reshape(batch_size, k, H, W)

def sch_decode(sch_feat, weights, pre_ind, kmf_ind=None, track_K=1, nms_kernel=5):
  """
  Arguments:
    sch_feats: B x N x H x W
    pre_ind: B x M
  """
  num_obj = pre_ind.size(1)
  batch_size = sch_feat.size(0)
  feat_channel = sch_feat.size(1)
  h, w = sch_feat.size(-2), sch_feat.size(-1)
  if kmf_ind is not None:
    x, y = kmf_ind%w,kmf_ind/w
  else:
    x, y = pre_ind%w,pre_ind/w
  x_range = torch.arange(w).float().to(device=sch_feat.device)
  y_range = torch.arange(h).float().to(device=sch_feat.device)
  hm = torch.zeros((batch_size, num_obj, h, w)).to(device=sch_feat.device)
  y_grid, x_grid = torch.meshgrid([y_range, x_range])
  for i in range(batch_size):
    conv1w,conv1b,conv2w,conv2b,conv3w,conv3b= \
        torch.split(weights[i],[(feat_channel+2)*feat_channel,feat_channel,
                                  feat_channel**2,feat_channel,
                                  feat_channel, 1],dim=-1)
    y_rel_coord = (y_grid[None,None] - y[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
    x_rel_coord = (x_grid[None,None] - x[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
    feat = sch_feat[i][None].repeat([num_obj,1,1,1])
    feat = torch.cat([feat,x_rel_coord, y_rel_coord],dim=1).view(1,-1,h,w)

    conv1w=conv1w.contiguous().view(-1, feat_channel+2,1,1)
    conv1b=conv1b.contiguous().flatten()
    feat = F.conv2d(feat,conv1w,conv1b,groups=num_obj).relu()

    conv2w=conv2w.contiguous().view(-1, feat_channel,1,1)
    conv2b=conv2b.contiguous().flatten()
    feat = F.conv2d(feat,conv2w,conv2b,groups=num_obj).relu()

    conv3w=conv3w.contiguous().view(-1, feat_channel,1,1)
    conv3b=conv3b.contiguous().flatten()
    hm[i] = F.conv2d(feat,conv3w,conv3b,groups=num_obj).sigmoid().squeeze()
    
  hm = _nms(hm, kernel=nms_kernel)
  scores, inds, ys0, xs0 = _topk_channel(hm, K=track_K)
  return scores, inds, ys0, xs0, hm
  #return scores.view(batch_size, num_obj, track_K), inds.view(batch_size, num_obj, track_K), ys0.view(batch_size, num_obj, track_K), xs0.view(batch_size, num_obj, track_K), hm

def wh_decode(wh_feat, inds, xs, ys, K):
  batch = wh_feat.size(0)
  wh = _tranpose_and_gather_feat(wh_feat, inds) # B x K x (F)
  wh = wh.view(batch, K, 2)
  wh[wh < 0] = 0
  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)
  return bboxes

def ltrb_amodal_decode(amodal_feat, inds, xs, ys, K):
    batch = amodal_feat.size(0)
    ltrb_amodal = _tranpose_and_gather_feat(amodal_feat, inds) # B x K x 4
    ltrb_amodal = ltrb_amodal.view(batch, K, 4)
    bboxes_amodal = torch.cat([xs.view(batch, K, 1) + ltrb_amodal[..., 0:1], 
                          ys.view(batch, K, 1) + ltrb_amodal[..., 1:2],
                          xs.view(batch, K, 1) + ltrb_amodal[..., 2:3], 
                          ys.view(batch, K, 1) + ltrb_amodal[..., 3:4]], dim=2)
    return bboxes_amodal

class GenericDecode():
  def __init__(self, K=100, opt=None):
    self.K = K 
    self.opt = opt
    if 'seg' in self.opt.heads:
      self.seg_decode = CondInst(self.opt.seg_feat_channel)

  def generic_decode(self, output):
    K = self.K
    opt = self.opt

    if not ('hm' in output):
      return {}

    if opt.zero_tracking:
      output['tracking'] *= 0
    
    heat = output['hm']
    batch, cat, height, width = heat.size()

    heat = _nms(heat, kernel=opt.nms_kernel)
    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

    clses  = clses.view(batch, K)
    scores = scores.view(batch, K)
    bboxes = None
    cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {'scores': scores, 'clses': clses.float(), 
          'xs': xs0, 'ys': ys0, 'cts': cts}
    if 'reg' in output:
      reg = output['reg']
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs0.view(batch, K, 1) + 0.5
      ys = ys0.view(batch, K, 1) + 0.5

    xs0 = xs0.view(batch, K, 1)
    ys0 = ys0.view(batch, K, 1)

    if 'wh' in output:
      bboxes = wh_decode(output['wh'], inds, xs, ys, K)
      ret['bboxes'] = bboxes
  
    if 'seg' in output:
      seg_feat = output['seg']
      conv_weight = output['conv_weight']
      assert not opt.flip_test,"not support flip_test"
      torch.cuda.synchronize()
      seg_masks = self.seg_decode(seg_feat, conv_weight, inds, is_train=False)
      ret['seg'] = seg_masks
    
    if 'sch' in output:
      sch_weight = output['sch_weight']
      sch_weights = _tranpose_and_gather_feat(sch_weight, inds) 
      ret['sch_weights'] = sch_weights.view(batch, K, -1)

      if 'pre_inds' in output and output['pre_inds'].size(1) > 0:
        track_K = opt.track_K 
        nms_kernel = opt.nms_kernel
        sch_feat = output['sch']
        pre_inds = output['pre_inds']
        pre_weights = output['pre_weights'] if 'pre_weights' in output else _tranpose_and_gather_feat(sch_weight, pre_inds) # for training debug
        num_pre = pre_inds.size(1)

        assert not opt.flip_test,"not support flip_test"
        torch.cuda.synchronize()
        kmf_inds = output['kmf_inds'] if ('kmf_inds' in output and output['kmf_inds'] is not None) else None
        track_score, track_inds, tys0, txs0, hms = sch_decode(sch_feat, pre_weights, pre_inds, kmf_inds, track_K=track_K, nms_kernel=nms_kernel)

        if 'reg' in output:
          reg = output['reg']
          track_reg = _tranpose_and_gather_feat(reg, track_inds.view(-1, num_pre * track_K))
          txs = txs0.view(batch, num_pre * track_K, 1) + track_reg[:, :, 0:1]
          tys = tys0.view(batch, num_pre * track_K, 1) + track_reg[:, :, 1:2]
        else:
          txs = txs0.view(batch, num_pre * track_K, 1) + 0.5
          tys = tys0.view(batch, num_pre * track_K, 1) + 0.5
        if 'wh' in output:
          track_bboxes = wh_decode(output['wh'], track_inds.view(-1, num_pre * track_K), txs, tys, num_pre*track_K)
        elif 'ltrb_amodal' in output:
          track_bboxes = ltrb_amodal_decode(output['ltrb_amodal'], track_inds.view(-1, num_pre * track_K), txs, tys, num_pre*track_K)
        ret['pre_inds'] = pre_inds # (batch, num_pre)
        ret['track_scores'] = track_score # (batch, num_pre, track_K)
        ret['track_bboxes'] = track_bboxes.view(batch, num_pre, track_K, 4)
        ret['track_hms'] = hms
    
    
    if 'ltrb' in output:
      ltrb = output['ltrb']
      ltrb = _tranpose_and_gather_feat(ltrb, inds) # B x K x 4
      ltrb = ltrb.view(batch, K, 4)
      bboxes = torch.cat([xs0.view(batch, K, 1) + ltrb[..., 0:1], 
                          ys0.view(batch, K, 1) + ltrb[..., 1:2],
                          xs0.view(batch, K, 1) + ltrb[..., 2:3], 
                          ys0.view(batch, K, 1) + ltrb[..., 3:4]], dim=2)
      ret['bboxes'] = bboxes

  
    regression_heads = ['tracking', 'dep', 'rot', 'dim', 'amodel_offset',
      'nuscenes_att', 'velocity']

    for head in regression_heads:
      if head in output:
        ret[head] = _tranpose_and_gather_feat(
          output[head], inds).view(batch, K, -1)

    if 'ltrb_amodal' in output:
      ltrb_amodal = output['ltrb_amodal']
      ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, inds) # B x K x 4
      ltrb_amodal = ltrb_amodal.view(batch, K, 4)
      bboxes_amodal = torch.cat([xs0.view(batch, K, 1) + ltrb_amodal[..., 0:1], 
                            ys0.view(batch, K, 1) + ltrb_amodal[..., 1:2],
                            xs0.view(batch, K, 1) + ltrb_amodal[..., 2:3], 
                            ys0.view(batch, K, 1) + ltrb_amodal[..., 3:4]], dim=2)
      ret['bboxes_amodal'] = bboxes_amodal
      ret['bboxes'] = bboxes_amodal

    if 'hps' in output:
      kps = output['hps']
      num_joints = kps.shape[1] // 2
      kps = _tranpose_and_gather_feat(kps, inds)
      kps = kps.view(batch, K, num_joints * 2)
      kps[..., ::2] += xs0.view(batch, K, 1).expand(batch, K, num_joints)
      kps[..., 1::2] += ys0.view(batch, K, 1).expand(batch, K, num_joints)
      kps, kps_score = _update_kps_with_hm(
        kps, output, batch, num_joints, K, bboxes, scores)
      ret['hps'] = kps
      ret['kps_score'] = kps_score

    if 'pre_inds' in output and output['pre_inds'] is not None:
      pre_inds = output['pre_inds'] # B x pre_K
      pre_K = pre_inds.shape[1]
      pre_ys = (pre_inds / width).int().float()
      pre_xs = (pre_inds % width).int().float()

      ret['pre_cts'] = torch.cat(
        [pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)


    if 'kmf_inds' in output and output['kmf_inds'] is not None:
      kmf_inds = output['kmf_inds'] # B x pre_K
      kmf_K = kmf_inds.shape[1]
      kmf_ys = (kmf_inds / width).int().float()
      kmf_xs = (kmf_inds % width).int().float()

      ret['kmf_cts'] = torch.cat(
        [kmf_xs.unsqueeze(2), kmf_ys.unsqueeze(2)], dim=2)
    return ret


  