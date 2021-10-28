from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict

import pycocotools.coco as coco
import pycocotools.mask as mask_utils
import torch
import torch.utils.data as data

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_umich_gaussian_oval, gaussian_radius_center
from utils.image import erase_seg_mask_from_image, copy_paste_with_seg_mask
from utils.utils import make_disjoint
from utils.kalman_filter import KalmanBoxTracker
import copy
import random

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class GenericDataset(data.Dataset):
  is_fusion_dataset = False
  default_resolution = None
  num_categories = None
  class_name = None
  # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
  # Not using 0 because 0 is used for don't care region and ignore loss.
  cat_ids = None
  max_objs = None
  rest_focal_length = 1200
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                      dtype=np.float32)
  _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
  ignore_val = 1
  nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 
    4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}
  def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
    super(GenericDataset, self).__init__()
    if opt is not None and split is not None:
      self.split = split
      self.opt = opt
      self._data_rng = np.random.RandomState(123)
    
    if ann_path is not None and img_dir is not None:
      print('==> initializing {} data from {}, \n images from {} ...'.format(
        split, ann_path, img_dir))
      self.coco = coco.COCO(ann_path)
      self.images = self.coco.getImgIds()

      if opt.tracking:
        if not ('videos' in self.coco.dataset):
          self.fake_video_data()
        print('Creating video index!')
        self.video_to_images = defaultdict(list)
        for image in self.coco.dataset['images']:
          self.video_to_images[image['video_id']].append(image)
      
      self.img_dir = img_dir

  def __getitem__(self, index):
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
    aug_s, rot, flipped = 1, 0, 0
    kmf_cts = None
    if self.split == 'train':
      c, aug_s, rot = self._get_aug_param(c, s, width, height)
      s = s * aug_s
      if np.random.random() < opt.flip:
        flipped = 1
        img = img[:, ::-1, :]
        anns = self._flip_anns(anns, height, width)


    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])
    inp = self._get_input(img, trans_input)
    ret = {'image': inp}
    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

    pre_cts, track_ids = None, None
    if opt.tracking:
      num_pre_data = opt.num_pre_data 
      pre_images, pre_annss, frame_dists = self._load_pre_data(
        img_info['video_id'], img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1, num_pre_data)
      if flipped:
        pre_images = [pre_image[:, ::-1, :].copy() for pre_image in pre_images]
        pre_annss = [self._flip_anns(pre_anns, height, width) for pre_anns in pre_annss]
      if opt.same_aug_pre and frame_dists[0] != 0:
        trans_input_pre = trans_input 
        trans_output_pre = trans_output
      else:
        if self.split == 'train':
          c_pre, aug_s_pre, _ = self._get_aug_param(
            c, s, width, height, disturb=True)
          s_pre = s * aug_s_pre
        else:
          c_pre = c
          s_pre = s
        trans_input_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w, opt.output_h])

      pre_imgs = [self._get_input(pre_image, trans_input_pre) for pre_image in pre_images]
      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_annss[-1], trans_input_pre, trans_output_pre, ret)
      ret['pre_img'] = np.array(pre_imgs[-1])
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm
    
        ### init samples
    self._init_ret(ret, gt_det)
    calib = self._get_calib(img_info, width, height)

    if opt.tracking and (opt.kmf_ind or opt.kmf_att) and opt.kmf_pit:
      kmf_trackers = self._gen_kmf_att_hm(ret, pre_annss, trans_output) # output format 
      kmf_cts = [kmf_trackers[tid]['ct'] for tid in track_ids]
    
    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        continue

      seg_mask = None
      if 'segmentation' in ann.keys():
        self.coco.imgs[ann['image_id']].update({'height':height, 'width':width})
        seg_mask = self._get_seg_mask_output(
           ann, trans_output, (opt.output_w, opt.output_h))
        
      if 'bbox' not in ann.keys():
        ann['bbox'] = mask_utils.toBbox(ann['segmentation'])
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      
      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        if 'segmentation' in ann.keys():
          self._mask_ignore_or_crowd_seg(ret, cls_id, seg_mask)
        else:
          self._mask_ignore_or_crowd(ret, cls_id, bbox)
        continue
      

      self._add_instance(
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
        calib, seg_mask, pre_cts, track_ids, kmf_cts)
      
      if opt.kmf_att and not opt.kmf_pit:
        _ = self._add_kmf_att(ret=ret, ann=ann, trans_input=trans_input)
    if 'kmf_att' in ret and not opt.keep_att:
      ret['kmf_att'][0] = ret['kmf_att'][0] * 0.5 + 0.5
    if self.opt.debug > 0:
      gt_det = self._format_gt_det(gt_det)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'flipped': flipped}
      ret['meta'] = meta
    return ret


  def get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _load_data(self, index):
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]
    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path

  def _rand_pick_peds_ann(self):
    ann_index = np.random.choice(self.PedsAnnIds, 1)[0]
    ann = self.coco.loadAnns(ids=int(ann_index))[0]
    img, _, _, _ = self._load_image_anns(ann['image_id'], self.coco, self.img_dir)
    return ann, img

  def _load_pre_data(self, video_id, frame_id, sensor_id=1, num_data=1):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previous" frame
    # If testing, get the exact prevous frame
    if 'train' in self.split:
      if (self.opt.kmf_att and self.opt.kmf_pit) or self.opt.one_way_pre_data: # load pre data from one-way only
        rev = random.randrange(2)
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (frame_id - img_info['frame_id']) * pow(-1, rev) <= self.opt.max_frame_dist and \
            (frame_id - img_info['frame_id']) * pow(-1, rev) > 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        img_ids.sort(key=lambda x: x[1])
        if len(img_ids) == 0:
          img_ids = [(img_info['id'], img_info['frame_id']) \
              for img_info in img_infos \
              if (img_info['frame_id'] - frame_id) == 0 and \
              (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        if len(img_ids) < num_data:
          img_ids = img_ids * num_data
        pre_ids = np.random.choice(len(img_ids), num_data, replace=False)
        pre_ids = sorted(pre_ids, reverse=(rev%2==1))
      else:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if abs(img_info['frame_id'] - frame_id) <= self.opt.max_frame_dist and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        pre_ids = np.random.choice(len(img_ids), num_data, replace=False)
    else:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (frame_id - img_info['frame_id']) <= num_data \
              and (frame_id - img_info['frame_id']) > 0 ]
      img_ids.sort(key=lambda x: x[1]) # frame: (1, 2, 3 ...)
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) < num_data:
        img_ids = img_ids * num_data
      pre_ids = np.arange(len(img_ids))
    imgs, annss, frame_dists = [], [], []
    for pre_id in pre_ids:
      img_id, pre_frame_id = img_ids[pre_id]
      frame_dist = abs(frame_id - pre_frame_id)
      img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
      imgs.append(img)
      annss.append(anns)
      frame_dists.append(frame_dist)
      
    return imgs, annss, frame_dists


  def _get_pre_dets(self, anns, trans_input, trans_output, ret):
    k = 0
    hm_h, hm_w = self.opt.input_h, self.opt.input_w
    down_ratio = self.opt.down_ratio
    trans = trans_input
    reutrn_hm = self.opt.pre_hm
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
    pre_cts, track_ids = [], []
  
    for i, ann in enumerate(anns):
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -99 or \
        ('iscrowd' in ann and ann['iscrowd'] > 0) or cls_id == 0: # cls_id add by vtsai01
        continue
      if 'bbox' not in anns[i].keys():
        ann['bbox'] = mask_utils.toBbox(ann['segmentation'])
      bbox = self._coco_box_to_bbox(ann['bbox'])
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      max_rad = 1

      track_id = ann['track_id'] if 'track_id' in ann else -1
  
      if (h > 0 and w > 0):
        if 'seg' in self.opt.task  and self.opt.seg_center:
          seg_mask = self.get_masks_as_input(ann, trans)
          if np.sum(seg_mask) <= 0:
            continue
          ct = np.array([np.mean(np.where(seg_mask>=0.5)[1]), np.mean(np.where(seg_mask>=0.5)[0])], dtype=np.float32)
        else:
          ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius)) 
        max_rad = max(max_rad, radius)
        
        ct0 = ct.copy()
        conf = 1

        ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
        ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
        conf = 1 if np.random.random() > self.opt.lost_disturb else 0
        
        ct_int = ct.astype(np.int32)
        if conf == 0:
          pre_cts.append(ct / down_ratio)
        else:
          pre_cts.append(ct0 / down_ratio)

        track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
        if reutrn_hm:
          draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

        if np.random.random() < self.opt.fp_disturb and reutrn_hm:
          ct2 = ct0.copy()
          # Hard code heatmap disturb ratio, haven't tried other numbers.
          ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
          ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
          ct2_int = ct2.astype(np.int32)
          draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

    return pre_hm, pre_cts, track_ids

  def merge_masks_as_input(self, anns, trans_input):
      rles = [ann['segmentation'] for ann in anns  if not ann['category_id'] == 10]
      mgrle = mask_utils.merge(rles)
      mask = mask_utils.decode(mgrle)
      inp = cv2.warpAffine(mask, trans_input, 
                    (self.opt.input_w, self.opt.input_h),
                    flags=cv2.INTER_LINEAR)
      return inp
  def get_masks_as_input(self, ann, trans_input):
      rle = ann['segmentation']
      mask = mask_utils.decode(rle)
      inp = cv2.warpAffine(mask, trans_input, 
                    (self.opt.input_w, self.opt.input_h),
                    flags=cv2.INTER_LINEAR)
      return inp

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


  def _get_aug_param(self, c, s, width, height, disturb=False):
    if (not self.opt.not_rand_crop) and not disturb:
      aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
      w_border = self._get_border(128, width)
      h_border = self._get_border(128, height)
      c[0] = np.random.randint(low=w_border, high=width - w_border)
      c[1] = np.random.randint(low=h_border, high=height - h_border)
    else:
      sf = self.opt.scale
      cf = self.opt.shift
      if type(s) == float:
        s = [s, s]
      c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
    if np.random.random() < self.opt.aug_rot:
      rf = self.opt.rotate
      rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    else:
      rot = 0
    
    return c, aug_s, rot

  def _copy_and_paste(self, anchor_ann, anns, image, copied_ann, copied_image, height, width):
    if len(anns) <= 0 or anns is None or copied_ann is None or len(copied_ann) <= 0 or anchor_ann['image_id'] == copied_ann['image_id']:
      return anns, image, 0
 
    copied_mask = mask_utils.decode(copied_ann['segmentation'])
    anchor_bbox = mask_utils.toBbox(anchor_ann['segmentation']) #  bbs     - [nx4] Bounding box(es) stored as [x y w h]
    copied_bbox = mask_utils.toBbox(copied_ann['segmentation']) #  bbs     - [nx4] Bounding box(es) stored as [x y w h]

    scale_ratio = min(anchor_bbox[3] / (copied_bbox[3] + 1e-8), 1)
    if copied_bbox[3] / copied_bbox[2] > 5 or copied_bbox[3] / copied_bbox[2] < 1: #dacnp v1.1
      return anns, image, 0
    dx, dy = - copied_bbox[0], - copied_bbox[1]
    jitter_x, jitter_y = np.random.random() * anchor_bbox[2] , np.random.random() * anchor_bbox[3]
    dx = dx*scale_ratio + anchor_bbox[0] + jitter_x
    dy = dy*scale_ratio + anchor_bbox[1] + jitter_y
    M = np.float32([[scale_ratio, 0, dx],[0, scale_ratio, dy]])
    _copied_image = copy.deepcopy(copied_image)
    cpimg = cv2.warpAffine(_copied_image, M, (image.shape[1], image.shape[0]))
    cpmask = cv2.warpAffine(copied_mask, M, (image.shape[1], image.shape[0]))

    result_img = copy_paste_with_seg_mask(image, cpimg, cpmask, blend=False)

    cpseg= mask_utils.encode((np.asfortranarray(cpmask > 0.5).astype(np.uint8)))
    cpseg['counts'] = cpseg['counts'].decode("utf-8")
    _copied_ann = copy.deepcopy(copied_ann)
    _copied_ann.update({'height':height, 'width':width, 'segmentation': cpseg, 'priority': 99})
    _copied_ann['bbox'] = mask_utils.toBbox(_copied_ann['segmentation'])  #dacnp v1.1

    for a in anns:
      a.update({'priority': 1})

    anns.append(_copied_ann)
    anns = make_disjoint(anns, strategy='priority')
    return anns, result_img, 1


  def _flip_anns(self, anns, height, width):
    for k in range(len(anns)):
      if 'bbox' not in anns[k].keys():
        anns[k]['bbox'] = mask_utils.toBbox(anns[k]['segmentation'])
      if 'segmentation' in anns[k].keys():
        self.coco.imgs[anns[k]['image_id']].update({'height':height, 'width':width})
        seg_mask = self.coco.annToMask(anns[k])
        seg_mask = np.asfortranarray(seg_mask[:, ::-1])
        rev_segmentation = mask_utils.encode(seg_mask)
        rev_segmentation['counts'] = rev_segmentation['counts'].decode("utf-8")
        anns[k]['segmentation'] = rev_segmentation

      bbox = anns[k]['bbox']
      anns[k]['bbox'] = [
        width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]
      
      if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
        keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
          self.num_joints, 3)
        keypoints[:, 0] = width - keypoints[:, 0] - 1
        for e in self.flip_idx:
          keypoints[e[0]], keypoints[e[1]] = \
            keypoints[e[1]].copy(), keypoints[e[0]].copy()
        anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

      if 'rot' in self.opt.heads and 'alpha' in anns[k]:
        anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                           else - np.pi - anns[k]['alpha']

      if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
        anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

      if self.opt.velocity and 'velocity' in anns[k]:
        anns[k]['velocity'] = [-10000, -10000, -10000]

    return anns


  def _get_input(self, img, trans_input):
    inp = cv2.warpAffine(img, trans_input, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    return inp


  def _init_ret(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)
    if 'seg' in self.opt.task:
      ret['seg_mask'] = np.zeros(
      (max_objs, self.opt.output_h, self.opt.output_w), np.float32)
    if self.opt.sch_track:
      ret['hm_track'] = np.zeros(
      (max_objs, self.opt.output_h, self.opt.output_w), np.float32)
      ret['pre_ind'] = np.zeros((max_objs), dtype=np.int64)
      ret['kmf_ind'] = np.zeros((max_objs), dtype=np.int64)
      ret['kmf_cts'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['pre_mask'] = np.zeros((max_objs), dtype=np.float32)
    if self.opt.kmf_att:
      ret['kmf_att'] = np.zeros(
      (1, self.opt.input_h, self.opt.input_w), 
      np.float32)
    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4, 
      'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2, 
      'dep': 1, 'dim': 3, 'amodel_offset': 2}

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []

    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)
    
    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})


  def _get_calib(self, img_info, width, height):
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _ignore_region(self, region, ignore_val=1):
    return np.maximum(region, ignore_val)


  def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
    # mask out crowd region, only rectangular mask is supported
    if cls_id == 0: # ignore all classes
      ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1] = \
        self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, \
                                        int(bbox[0]): int(bbox[2]) + 1])
    else:
      # mask out one specific class
      ret['hm'][abs(cls_id) - 1, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1] = \
                      self._ignore_region(ret['hm'][abs(cls_id) - 1, \
                                    int(bbox[1]): int(bbox[3]) + 1,  \
                                    int(bbox[0]): int(bbox[2]) + 1])
    if ('hm_hp' in ret) and cls_id <= 1:
      ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1] = \
                      self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, \
                                          int(bbox[0]): int(bbox[2]) + 1])

  def _mask_ignore_or_crowd_seg(self, ret, cls_id, seg_mask):
    # mask out crowd region, only rectangular mask is support
    if cls_id == 0: # ignore all classes
      ret['hm'][:, seg_mask.astype(np.bool)] = self._ignore_region(ret['hm'][:, seg_mask.astype(np.bool)])
    else:
      # mask out one specific class
      ret['hm'][abs(cls_id) - 1, seg_mask.astype(np.bool)] = self._ignore_region(ret['hm'][abs(cls_id) - 1, seg_mask.astype(np.bool)])

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_seg_mask_output(self, ann, trans_output, output_w_h):

    seg_mask = self.coco.annToMask(ann)
    #seg_mask = mask_utils.decode(ann['segmentation'])
    #if flipped:
    #  seg_mask = seg_mask[:, ::-1]
    seg_mask = cv2.warpAffine(seg_mask, trans_output, 
                    output_w_h, flags=cv2.INTER_NEAREST)

    return seg_mask
  def _get_bbox_output(self, bbox, trans_output, height, width):
    bbox = self._coco_box_to_bbox(bbox).copy()

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

    bbox_amodal = copy.deepcopy(bbox)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    return bbox, bbox_amodal


  def _gen_kmf_att_hm(self, ret, pre_anns, trans_input):
    trackers = {}
    trans = trans_input
    hm_h, hm_w = self.opt.input_h, self.opt.input_w
    iid = None
    for idx, anns in enumerate(pre_anns): #[..., n-2, n-1] 
      for i, ann in enumerate(anns):
        cls_id = int(self.cat_ids[ann['category_id']])
        if cls_id > self.opt.num_classes or cls_id <= -999 or cls_id == 0:
          continue
        if 'bbox' not in anns[i].keys():
          ann['bbox'] = mask_utils.toBbox(ann['segmentation'])
        bbox = self._coco_box_to_bbox(ann['bbox'])
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
          continue
        
        if ann['track_id'] not in trackers:
          trackers[ann['track_id']] = {}
          trackers[ann['track_id']]['kmf'] = KalmanBoxTracker(bbox)
          trackers[ann['track_id']]['age'] = 0
          trackers[ann['track_id']]['cts_history'] = [np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)]
        else:
          if np.random.random() > self.opt.att_track_lost_disturb or idx == len(pre_anns) - 1:
            trackers[ann['track_id']]['kmf'].predict()
            trackers[ann['track_id']]['kmf'].update(bbox)
            trackers[ann['track_id']]['age'] += 1
            trackers[ann['track_id']]['cts_history'].append(np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32))
    for k in trackers:
      bbox = trackers[k]['kmf'].predict()[0]
      pred_ct = self._add_kmf_att(ret=ret, bbox=bbox, trans_input=trans_input, init=(trackers[k]['age'] <= 0), draw=(self.opt.kmf_att))
      if pred_ct is None:
        trackers[k]['ct'] = trackers[k]['cts_history'][-1]
      else:
        trackers[k]['ct'] = pred_ct

    return trackers

  def _add_kmf_att(self, ret, trans_input, ann=None, bbox=None, init=False, conf=1, draw=True):
    trans = trans_input
    hm_h, hm_w = self.opt.input_h, self.opt.input_w
    if bbox is None and ann is not None:
      if 'bbox' not in ann.keys():
        ann['bbox'] = mask_utils.toBbox(ann['segmentation'])
      bbox = self._coco_box_to_bbox(ann['bbox'])
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

    if (h > 0 and w > 0):
      ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
      if self.opt.guss_rad:
        min_overlap = 0.2 if init else 0.6
        conf = self.opt.init_conf if init else 1
        radius = gaussian_radius_center((math.ceil(h), math.ceil(w)), min_overlap=0.2)
      else:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      radius = max(0, int(radius)) 
      
      ct0 = ct.copy()

      ct[0] = ct[0] + np.random.randn() * self.opt.att_hm_disturb * w
      ct[1] = ct[1] + np.random.randn() * self.opt.att_hm_disturb * h
      conf = conf if np.random.random() > self.opt.att_lost_disturb else 0
      ct_int = ct.astype(np.int32)
      if self.opt.guss_oval and draw:
        radius = radius if (self.opt.guss_rad and init) or (self.opt.guss_rad and self.opt.guss_rad_always) else 0
        draw_umich_gaussian_oval(ret['kmf_att'][0], ct_int, radius_h=h//2+radius, radius_w=w//2+radius, k=conf)
      elif draw:
        draw_umich_gaussian(ret['kmf_att'][0], ct_int, radius, k=conf)

      if np.random.random() < self.opt.att_fp_disturb: # generate false positive 
        ct2 = ct0.copy()
        # Hard code heatmap disturb ratio, haven't tried other numbers.
        ct2[0] = ct2[0] + np.random.randn() * self.opt.att_disturb_dist * w
        ct2[1] = ct2[1] + np.random.randn() * self.opt.att_disturb_dist * h 
        ct2_int = ct2.astype(np.int32)
        if self.opt.guss_oval and draw:
          draw_umich_gaussian_oval(ret['kmf_att'][0], ct2_int, radius_h=h//2, radius_w=w//2, k=conf)
        elif draw:
          draw_umich_gaussian(ret['kmf_att'][0], ct2_int, radius, k=conf)
    else:
      return None
    return ct_int

  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
    aug_s, calib, seg_mask=None, pre_cts=None, track_ids=None, kmf_cts=None):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0 or (seg_mask is not None and np.sum(seg_mask)<=0):
      return
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius)) 
    if 'seg' in self.opt.task and seg_mask is not None and self.opt.seg_center:
      ct = np.array([np.mean(np.where(seg_mask>=0.5)[1]), np.mean(np.where(seg_mask>=0.5)[0])], dtype=np.float32)
    else:
      ct = np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    if 'reg' in ret:
      ret['reg'][k] = ct - ct_int
      ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct)

    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        #ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct_int
        gt_det['tracking'].append(ret['tracking'][k])
      else:
        gt_det['tracking'].append(np.zeros(2, np.float32))
    
    if 'sch' in self.opt.heads:
      if ann['track_id'] in track_ids:
        draw_umich_gaussian(ret['hm_track'][k], ct_int, radius)
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        pre_ct_int = pre_ct.astype(np.int32)
        ret['pre_ind'][k] = pre_ct_int[1] * self.opt.output_w + pre_ct_int[0]
        if kmf_cts is not None:
          kmf_ct = kmf_cts[track_ids.index(ann['track_id'])]
          kmf_ct_int = kmf_ct.astype(np.int32)
          ret['kmf_ind'][k] = kmf_ct_int[1] * self.opt.output_w + kmf_ct_int[0]
          ret['kmf_cts'][k] = kmf_ct
        ret['pre_mask'][k] = 1
      
    if 'seg' in self.opt.task and seg_mask is not None:
      ret['seg_mask'][k] = seg_mask

    if 'ltrb' in self.opt.heads:
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
        bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    if 'ltrb_amodal' in self.opt.heads:
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      gt_det['ltrb_amodal'].append(bbox_amodal)

    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    if 'velocity' in self.opt.heads:
      if ('velocity' in ann) and min(ann['velocity']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, k, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1,1,1])
    
    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])
    

  def _format_gt_det(self, gt_det):
    if (len(gt_det['scores']) == 0):
      gt_det = {'bboxes': np.array([[0,0,1,1]], dtype=np.float32), 
                'scores': np.array([1], dtype=np.float32), 
                'clses': np.array([0], dtype=np.float32),
                'cts': np.array([[0, 0]], dtype=np.float32),
                'pre_cts': np.array([[0, 0]], dtype=np.float32),
                'tracking': np.array([[0, 0]], dtype=np.float32),
                'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                'ltrb_amodal': np.array([[0, 0]], dtype=np.float32),
                'hps': np.zeros((1, 17, 2), dtype=np.float32),}
    gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
    return gt_det

  def fake_video_data(self):
    self.coco.dataset['videos'] = []
    for i in range(len(self.coco.dataset['images'])):
      img_id = self.coco.dataset['images'][i]['id']
      self.coco.dataset['images'][i]['video_id'] = img_id
      self.coco.dataset['images'][i]['frame_id'] = 1
      self.coco.dataset['videos'].append({'id': img_id})
    
    if not ('annotations' in self.coco.dataset):
      return

    for i in range(len(self.coco.dataset['annotations'])):
      self.coco.dataset['annotations'][i]['track_id'] = i + 1