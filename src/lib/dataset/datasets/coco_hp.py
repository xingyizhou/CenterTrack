from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from ..generic_dataset import GenericDataset

class COCOHP(GenericDataset):
  num_categories = 1
  class_name = ['']
  num_joints = 17
  default_resolution = [512, 512]
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [4, 6], [3, 5], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [6, 12], [5, 11], [11, 12], 
                  [12, 14], [14, 16], [11, 13], [13, 15]]
  max_objs = 32
  cat_ids = {1: 1}

  def __init__(self, opt, split):
    data_dir = os.path.join(opt.data_dir, 'coco')
    img_dir = os.path.join(data_dir, '{}2017'.format(split))
    if split == 'test':
      ann_path = os.path.join(data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    else:
      ann_path = os.path.join(data_dir, 'annotations', 
        'person_keypoints_{}2017.json').format(split)
    

    self.images = None
    # load image list and coco
    super(COCOHP, self).__init__(opt, split, ann_path, img_dir)

    if split == 'train':
      image_ids = self.coco.getImgIds()
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
    
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          if item['class'] != 1:
            continue
          category_id = 1
          keypoints = np.concatenate([
            np.array(item['hps'], dtype=np.float32).reshape(-1, 2), 
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "score": float("{:.2f}".format(item['score'])),
              "keypoints": keypoints
          }
          if 'bbox' in item:
            bbox = item['bbox']
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            bbox_out  = list(map(self._to_float, bbox[0:4]))
            detection['bbox'] = bbox_out
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results_cocohp.json'.format(save_dir), 'w'))


  def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results_cocohp.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
