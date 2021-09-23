from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from .image import transform_preds_with_trans, get_affine_transform
from .ddd_utils import ddd2locrot, comput_corners_3d
from .ddd_utils import project_to_image, rot_y2alpha
import numba
from pycocotools import mask as mask_utils
import pycocotools.coco as coco
from .utils import make_disjoint



def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)

def generic_post_process(
  opt, dets, c, s, h, w, num_classes, calibs=None, height=-1, width=-1, pid2track=None):
  """
    h: out_height
    w: out_width
    height: img_height
    width: img_width
  """
  if not ('scores' in dets):
    return [{}], [{}]
  ret = []

  for i in range(len(dets['scores'])):
    preds = []
    trans = get_affine_transform(
      c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
    for j in range(len(dets['scores'][i])):
      if dets['scores'][i][j] < opt.out_thresh[int(dets['clses'][i][j])]:
        break
      item = {}
      item['score'] = dets['scores'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1
      item['ct'] = transform_preds_with_trans(
        (dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)

      if 'tracking' in dets:
        tracking = transform_preds_with_trans(
          (dets['tracking'][i][j] + dets['cts'][i][j]).reshape(1, 2), 
          trans).reshape(2)
        item['tracking'] = tracking - item['ct']

      if 'bboxes' in dets:
        bbox = transform_preds_with_trans(
          dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
        item['bbox'] = bbox

      if 'seg' in dets:
        item['seg'] = mask_utils.encode(
          (np.asfortranarray(cv2.warpAffine(dets['seg'][i][j], trans, (width, height),
				   flags=cv2.INTER_CUBIC) > 0.5).astype(np.uint8)))
        item['seg']['counts'] = item['seg']['counts'].decode("utf-8")
        if opt.wh_weight <= 0:
          item['bbox'] = _coco_box_to_bbox(mask_utils.toBbox(item['seg']))


      if 'hps' in dets:
        pts = transform_preds_with_trans(
          dets['hps'][i][j].reshape(-1, 2), trans).reshape(-1)
        item['hps'] = pts

      if 'dep' in dets and len(dets['dep'][i]) > j:
        item['dep'] = dets['dep'][i][j]
      
      if 'dim' in dets and len(dets['dim'][i]) > j:
        item['dim'] = dets['dim'][i][j]

      if 'rot' in dets and len(dets['rot'][i]) > j:
        item['alpha'] = get_alpha(dets['rot'][i][j:j+1])[0]
      
      if 'rot' in dets and 'dep' in dets and 'dim' in dets \
        and len(dets['dep'][i]) > j:
        if 'amodel_offset' in dets and len(dets['amodel_offset'][i]) > j:
          ct_output = dets['bboxes'][i][j].reshape(2, 2).mean(axis=0)
          amodel_ct_output = ct_output + dets['amodel_offset'][i][j]
          ct = transform_preds_with_trans(
            amodel_ct_output.reshape(1, 2), trans).reshape(2).tolist()
        else:
          bbox = item['bbox']
          ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        item['ct'] = ct
        item['loc'], item['rot_y'] = ddd2locrot(
          ct, item['alpha'], item['dim'], item['dep'], calibs[i])
      
      preds.append(item)

    if 'nuscenes_att' in dets:
      for j in range(len(preds)):
        preds[j]['nuscenes_att'] = dets['nuscenes_att'][i][j]

    if 'velocity' in dets:
      for j in range(len(preds)):
        preds[j]['velocity'] = dets['velocity'][i][j]

    if 'seg' in dets and not opt.not_make_mask_disjoint:
      strategy = opt.disjoint_strategy
      preds = make_disjoint(preds, strategy)

    ret.append(preds)
  pre_ret = []
  if 'track_scores' in dets:
    for i in range(len(dets['track_scores'])):
      track_preds = []
      trans = get_affine_transform(
        c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
      for j in range(len(dets['track_scores'][i])):
        item = {}
        item['track_score'] = dets['track_scores'][i][j]
        if pid2track is not None:
          item['tracking_id'] = pid2track[dets['pre_inds'][i][j]]
        if 'track_bboxes' in dets:
          bbox = transform_preds_with_trans(
            dets['track_bboxes'][i][j].reshape(2, 2), trans).reshape(4)
          item['track_bbox'] = bbox
      track_preds.append(item)
    pre_ret.append(track_preds)

  return ret, pre_ret