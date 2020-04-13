from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..generic_dataset import GenericDataset

class CustomDataset(GenericDataset):
  num_categories = 1
  default_resolution = [-1, -1]
  class_name = ['']
  max_objs = 128
  cat_ids = {1: 1}
  def __init__(self, opt, split):
    assert (opt.custom_dataset_img_path != '') and \
      (opt.custom_dataset_ann_path != '') and \
      (opt.num_classes != -1) and \
      (opt.input_h != -1) and (opt.input_w != -1), \
      'The following arguments must be specified for custom datasets: ' + \
      'custom_dataset_img_path, custom_dataset_ann_path, num_classes, ' + \
      'input_h, input_w.'
    img_dir = opt.custom_dataset_img_path
    ann_path = opt.custom_dataset_ann_path
    self.num_categories = opt.num_classes
    self.class_name = ['' for _ in range(self.num_categories)]
    self.default_resolution = [opt.input_h, opt.input_w]
    self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}

    self.images = None
    # load image list and coco
    super().__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded Custom dataset {} samples'.format(self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def run_eval(self, results, save_dir):
    pass
