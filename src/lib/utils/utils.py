from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from pycocotools import mask as mask_utils
import copy
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def make_disjoint(objects, strategy):
    def get_max_y(obj):
        _, y, _, h = mask_utils.toBbox(obj['seg'])
        return y + h
    def get_area(obj):
        return mask_utils.area(obj['seg'])

    if len(objects) == 0:
        return []
    if strategy == "y_pos":
        objects_sorted = sorted(objects, key=lambda x: get_max_y(x), reverse=True)
    elif strategy == "score":
        objects_sorted = sorted(objects, key=lambda x: x['score'], reverse=True)
    elif strategy == "area":
        objects_sorted = sorted(objects, key=lambda x: get_area(x), reverse=False)
    elif strategy == "class":
        objects_sorted = sorted(objects, key=lambda x: x['class'], reverse=True)
    elif strategy == "priority":
        objects_sorted = sorted(objects, key=lambda x: x['priority'], reverse=True) # higher is more important as score
    else:
        assert False, "Unknown mask_disjoint_strategy"
    skey = 'seg' if 'seg' in objects_sorted[0] else 'segmentation'
    objects_disjoint = copy.deepcopy(objects_sorted)
    used_pixels = objects_sorted[0][skey]
    for i, obj in enumerate(objects_sorted[1:], start=1):
        new_mask = obj[skey]
        if mask_utils.area(mask_utils.merge([used_pixels, obj[skey]], intersect=True)) > 0.0:
            obj_mask_decoded = mask_utils.decode(obj[skey])
            used_pixels_decoded = mask_utils.decode(used_pixels)
            obj_mask_decoded[np.where(used_pixels_decoded > 0)] = 0
            new_mask = mask_utils.encode(obj_mask_decoded)
            new_mask_rle = new_mask['counts'].decode("utf-8")
            objects_disjoint[i][skey]['counts'] = new_mask_rle
        used_pixels = mask_utils.merge([used_pixels, obj[skey]], intersect=False)
        


    return objects_disjoint


def np_iou(box, boxes):
    """
    Calculate IOU between a bounding box and a set of bounding boxes.
    :param box: [x1,y1,x2,y2]
    :param boxes:[[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return: corresponding IOU values
    """
    box = np.array(box)
    boxes = np.array(boxes)
    if len(boxes)  == 0:
        return np.array([])
    ww = np.maximum(np.minimum(box[2], boxes[:, 2]) 
                    - np.maximum(box[0], boxes[:, 0]), 0)
    hh = np.maximum(np.minimum(box[3], boxes[:, 3]) -
                    np.maximum(box[1], boxes[:, 1]), 0)
    uu = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]+1e-8)
    return ww * hh / (uu - ww * hh)


def np_ious(boxes1, boxes2):
   
    results = []
    for b in boxes1:
        results.append(np_iou(b, boxes2))
    return np.array(results)

def iou_score(dets, tracks):
    """
        :param tracks: [[(bbox_1, score_1)], [(bbox_2_1, score_2_1), (bbox_2_2, score_2_2)], ...]      M x K x 4
        :param dets: [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2], ....]                                 N x 4

        :return N x M

    """
    results = []
    for ts in tracks:
        ratings = []
        for prop in ts:
            bbox, score = prop
            ious = np_iou(bbox, dets)
            ious[ious < 0.01] = 0

            ratings.append(ious*score)

        results.append(np.array(ratings).max(axis=0))
    results = np.array(results) # M x N
    results = np.swapaxes(results, 0, 1) if len(results.shape) > 1 else results # N x M
    return results