#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import copy

from typing import Any, List, Dict

DATA_PATH = '../../data/road/'
ANNOTATIONS_PATH = os.path.join(DATA_PATH, "road_trainval_v1.0.json")
VIDEOS_DIR = "videos/"
IMAGES_DIR = "rgb-images/"

'''
ROAD Dataset Annotation Structure: https://github.com/gurkirt/road-dataset#annotation-structure
'''
def load_annotations(path: str = ANNOTATIONS_PATH) -> Dict[str, Any]:
  return json.load(open(path, 'r'))

def save_splits(splits: Dict[str, Any]):
  out_dir = os.path.join(DATA_PATH, 'annotations')
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  for split in splits:
    ret = splits[split]
    print("Split:", split, "#images:", len(ret['images']), "#annotations:", len(ret['annotations']))
    out_path = os.path.join(out_dir, 'tracking_{}.json'.format(split))
    json.dump(ret, open(out_path, 'w'))

def format_classes(classes: List[str]) -> List[Dict[str, Any]]:
  return [{'name': label, 'id': i + 1} for i, label in enumerate(classes)] # TODO: Not convinced if the 1-index is necessary or not

def format_videoes(annotations: Dict[str, Any]) -> List[Dict[str, Any]]:
  videos = [os.path.join(VIDEOS_DIR, video_name) for video_name in list(annotations['db'].keys())]
  return [{'file_name': label, 'id': i + 1} for i, label in enumerate(videos)]

# format frames
# - Parameters:
#   - frame_names: List frame image files, relative to the dataset root directory
#   - video_id: the current ID of the video the frame belongs to
# - Returns: Formatted JSON of identifying the frames in a video
def format_frames(frame_names: List[str], video_id: int) -> List[Dict[str, Any]]:
  return [{'file_name': label, 
           'id': i + 1, 
           'frame_id': i + 1, 
           'video_id': video_id } for i, label in enumerate(frame_names)]

# Road BBOX: [xmin, ymin, xmax, ymax] normalized to 0, 1
# COCO BBOX: [x-top left, y-top left, width, height]
def bbox_road_to_coco(bbox: List[int], img_width: int, img_height: int) -> List[int]:
  return [bbox[0] * img_width, bbox[1] * img_height, 
          (bbox[2] - bbox[0]) * img_width, (bbox[3] - bbox[1]) * img_height]

def flatten(list: List[List[Any]]):
  return [x for sublist in list for x in sublist]

def main():
  annotations = load_annotations()
  base = {'images': [], 
         'annotations': [], 
         'categories': format_classes(annotations['agent_labels']),
         'loc_categories': format_classes(annotations['loc_labels']),
         'action_categories': format_classes(annotations['action_labels']),
         'duplex_categories': format_classes(annotations['duplex_labels']),
         'triplex_categories': format_classes(annotations['triplet_labels']),
         'videos': format_videoes(annotations)}
  num_images = 0

  category_to_id = {category['id']: category['name'] for category in base['categories']}
  db = annotations['db']

  all_splits = sorted(set(flatten([db[name]['split_ids'] for name in db])))
  print("Splits:", all_splits)
  splits = {name: copy.deepcopy(base) for name in all_splits}

  for video_info in base["videos"]:
    video_name = os.path.basename(video_info['file_name'])
    video_id = video_info['id']
    video_splits = db[video_name]['split_ids']

    frame_names = [os.path.join(IMAGES_DIR, video_name, frame) for frame in sorted(os.listdir(os.path.join(DATA_PATH, IMAGES_DIR, video_name)))]
    
    new_frames = format_frames(frame_names, video_id)
    for split in video_splits:
      splits[split]['images'].extend(new_frames)
    
    for frame_id in db[video_name]['frames']:
      frame_Dict = db[video_name]['frames'][frame_id]
      # Not annotated
      if db[video_name]['frames'][frame_id]['annotated'] == 0:
        continue

      for annotation_id in frame_Dict["annos"]:
        annotation = frame_Dict["annos"][annotation_id]
        # TODO: Tube level labels
        # tube_id = next(tube_id for tube_id in list(db[video_name]['agent_tubes'].keys()) if tube_id.startswith(annotation['tube_uid']))
        coco_annotation = {
               'image_id': frame_id,
               'id': int(annotation_id, 16), # Hex string to int
               'category_id': annotation['agent_ids'][0] + 1, # convert to 1-indexed ID 
               'loc_ids': [x + 1 for x in annotation['loc_ids']],
               'action_ids': [x + 1 for x in annotation['action_ids']],
               'action_ids': [x + 1 for x in annotation['duplex_ids']],
               'action_ids': [x + 1 for x in annotation['triplet_ids']],
               'bbox': bbox_road_to_coco(annotation['box'], frame_Dict['width'], frame_Dict['height']),
               'track_id': int(annotation['tube_uid'], 16)
        }
        for split in video_splits:
          splits[split]['annotations'].append(coco_annotation)
  
  save_splits(splits)

if __name__ == '__main__':
  main()
