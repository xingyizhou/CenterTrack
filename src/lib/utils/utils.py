from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from pycocotools import mask as cocomask
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
        _, y, _, h = cocomask.toBbox(obj['seg'])
        return y + h

    if len(objects) == 0:
        return []
    if strategy == "y_pos":
        objects_sorted = sorted(objects, key=lambda x: get_max_y(x), reverse=True)
    elif strategy == "score":
        objects_sorted = sorted(objects, key=lambda x: x['score'], reverse=True)
    else:
        assert False, "Unknown mask_disjoint_strategy"
    objects_disjoint = copy.deepcopy(objects_sorted)
    used_pixels = objects_sorted[0]['seg']
    for i, obj in enumerate(objects_sorted[1:], start=1):
        new_mask = obj['seg']
        if cocomask.area(cocomask.merge([used_pixels, obj['seg']], intersect=True)) > 0.0:
            obj_mask_decoded = cocomask.decode(obj['seg'])
            used_pixels_decoded = cocomask.decode(used_pixels)
            obj_mask_decoded[np.where(used_pixels_decoded > 0)] = 0
            new_mask = cocomask.encode(obj_mask_decoded)
            new_mask_rle = new_mask['counts'].decode("utf-8")
            objects_disjoint[i]['seg']['counts'] = new_mask_rle
        used_pixels = cocomask.merge([used_pixels, obj['seg']], intersect=False)
        


    return objects_disjoint