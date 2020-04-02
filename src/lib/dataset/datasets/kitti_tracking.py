from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math

from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image

class KITTITracking(GenericDataset):
  num_categories = 3
  default_resolution = [384, 1280]
  class_name = ['Pedestrian', 'Car', 'Cyclist']
  # negative id is for "not as negative sample for abs(id)".
  # 0 for ignore losses for all categories in the bounding box region
  # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
  #       'Tram', 'Misc', 'DontCare']
  cat_ids = {1:1, 2:2, 3:3, 4:-2, 5:-2, 6:-1, 7:-9999, 8:-9999, 9:0}
  max_objs = 50
  def __init__(self, opt, split):
    data_dir = os.path.join(opt.data_dir, 'kitti_tracking')
    split_ = 'train' if opt.dataset_version != 'test' else 'test' #'test'
    img_dir = os.path.join(
      data_dir, 'data_tracking_image_2', '{}ing'.format(split_), 'image_02')
    ann_file_ = split_ if opt.dataset_version == '' else opt.dataset_version
    print('Warning! opt.dataset_version is not set')
    ann_path = os.path.join(
      data_dir, 'annotations', 'tracking_{}.json'.format(
        ann_file_))
    self.images = None
    super(KITTITracking, self).__init__(opt, split, ann_path, img_dir)
    self.alpha_in_degree = False
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))


  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))


  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_kitti_tracking')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)

    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      
      for image_info in images:
        img_id = image_info['id']
        if not (img_id in results):
          continue
        frame_id = image_info['frame_id'] 
        for i in range(len(results[img_id])):
          item = results[img_id][i]
          category_id = item['class']
          cls_name_ind = category_id
          class_name = self.class_name[cls_name_ind - 1]
          if not ('alpha' in item):
            item['alpha'] = -1
          if not ('rot_y' in item):
            item['rot_y'] = -10
          if 'dim' in item:
            item['dim'] = [max(item['dim'][0], 0.01), 
              max(item['dim'][1], 0.01), max(item['dim'][2], 0.01)]
          if not ('dim' in item):
            item['dim'] = [-1, -1, -1]
          if not ('loc' in item):
            item['loc'] = [-1000, -1000, -1000]
          
          track_id = item['tracking_id'] if 'tracking_id' in item else -1
          f.write('{} {} {} -1 -1'.format(frame_id - 1, track_id, class_name))
          f.write(' {:d}'.format(int(item['alpha'])))
          f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
            item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]))
          
          f.write(' {:d} {:d} {:d}'.format(
            int(item['dim'][0]), int(item['dim'][1]), int(item['dim'][2])))
          f.write(' {:d} {:d} {:d}'.format(
            int(item['loc'][0]), int(item['loc'][1]), int(item['loc'][2])))
          f.write(' {:d} {:.2f}\n'.format(int(item['rot_y']), item['score']))
      f.close()

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    os.system('python tools/eval_kitti_track/evaluate_tracking.py ' + \
              '{}/results_kitti_tracking/ {}'.format(
                save_dir, self.opt.dataset_version))
    
