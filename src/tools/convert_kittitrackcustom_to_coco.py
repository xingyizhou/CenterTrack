from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import cv2
TRAIN_SEQ = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQ = [2, 6, 7, 8, 10, 13, 14, 16, 18]
DATA_PATH = '../../data/kitti_tracking_custom/'
SPLITS = ['val', 'train', 'test']
VIDEO_SETS = {'train': TRAIN_SEQ , 'test': range(29), 'val': VAL_SEQ}
CREATE_HALF_LABEL = False
DEBUG = False

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']


cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_ids['Person'] = cat_ids['Person_sitting']

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

if __name__ == '__main__':
  for split in SPLITS:
    ann_dir = DATA_PATH + '/label_02/'
    ret = {'images': [], 'annotations': [], "categories": cat_info,
           'videos': []}
    num_images = 0
    for i in VIDEO_SETS[split]:
      image_id_base = num_images
      video_name = '{:04d}'.format(i)
      ret['videos'].append({'id': i + 1, 'file_name': video_name})
      ann_dir = 'train'  if not ('test' in split) else split
      video_path = DATA_PATH + \
        '/data_tracking_image_2/{}ing/image_02/{}'.format(ann_dir, video_name)
      image_files = sorted(os.listdir(video_path))
      num_images_video = len(image_files)
      if CREATE_HALF_LABEL and 'half' in split:
        image_range = [0, num_images_video // 2 - 1] if split == 'train_half' else \
          [num_images_video // 2, num_images_video - 1]
      else:
        image_range = [0, num_images_video - 1]
      print('num_frames', video_name, image_range[1] - image_range[0] + 1)
      for j, image_name in enumerate(image_files):
        if (j < image_range[0] or j > image_range[1]):
          continue
        num_images += 1
        image_info = {'file_name': '{}/{:06d}.png'.format(video_name, j),
                      'id': num_images,
                      'video_id': i + 1,
                      'frame_id': j + 1 - image_range[0]}
        ret['images'].append(image_info)

      if split == 'test':
        continue
      # 0 -1 DontCare -1 -1 -10.000000 219.310000 188.490000 245.500000 218.560000 -1000.000000 -1000.000000 -1000.000000 -10.000000 -1.000000 -1.000000 -1.000000
      ann_path = DATA_PATH + 'label_02/{}.txt'.format(video_name)
      anns = open(ann_path, 'r')
      
      if CREATE_HALF_LABEL and 'half' in split:
        label_out_folder = DATA_PATH + 'label_02_{}/'.format(split)
        label_out_path = label_out_folder + '{}.txt'.format(video_name)
        if not os.path.exists(label_out_folder):
          os.mkdir(label_out_folder)
        label_out_file = open(label_out_path, 'w')
      
      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        frame_id = int(tmp[0])
        track_id = int(tmp[1])
        cat_id = cat_ids[tmp[2]]
        truncated = int(float(tmp[3]))
        occluded = int(tmp[4])
        alpha = float(tmp[5])
        bbox = [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])]
        dim = [float(tmp[10]), float(tmp[11]), float(tmp[12])]
        location = [float(tmp[13]), float(tmp[14]), float(tmp[15])]
        rotation_y = float(tmp[16])
        ann = {'image_id': frame_id + 1 - image_range[0] + image_id_base,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               'dim': dim,
               'bbox': _bbox_to_coco_bbox(bbox),
               'depth': location[2],
               'alpha': alpha,
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_y': rotation_y,
               'track_id': track_id + 1}
        if CREATE_HALF_LABEL and 'half' in split:
          if (frame_id < image_range[0] or frame_id > image_range[1]):
            continue
          out_frame_id = frame_id - image_range[0]
          label_out_file.write('{} {}'.format(
            out_frame_id, txt[txt.find(' ') + 1:]))
        
        ret['annotations'].append(ann)
      
    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    out_dir = '{}/annotations/'.format(DATA_PATH)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    out_path = '{}/annotations/tracking_{}.json'.format(
      DATA_PATH, split)
    json.dump(ret, open(out_path, 'w'))
