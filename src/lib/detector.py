from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
import math

from model.model import create_model, load_model
from model.decode import generic_decode
from model.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius, draw_umich_gaussian_oval
from utils.post_process import generic_post_process
from utils.debugger import Debugger
from utils.tracker import Tracker
from dataset.dataset_factory import get_dataset

import pycocotools.mask as mask_utils
from utils.image import erase_seg_mask_from_image, copy_paste_with_seg_mask

class Detector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)
    self.model, _, start_epoch = load_model(self.model, opt.load_model, opt)
    self.model = self.model.to(opt.device)
    self.model.eval()
    self.start_epoch = start_epoch

    self.opt = opt
    self.trained_dataset = get_dataset(opt.dataset)
    self.mean = np.array(
      self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(
      self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
    self.pause = not opt.no_pause
    self.rest_focal_length = self.trained_dataset.rest_focal_length \
      if self.opt.test_focal_length < 0 else self.opt.test_focal_length
    self.flip_idx = self.trained_dataset.flip_idx
    self.cnt = 0
    self.pre_images = None
    self.pre_image_ori = None
    self.age_images = []
    self.tracker = Tracker(opt)
    self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)


  def run(self, image_or_path_or_tensor, meta={}):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
    self.debugger.clear()
    start_time = time.time()

    # read image
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()

      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []

    # for multi-scale testing
    for scale in self.opt.test_scales:
      scale_start_time = time.time()
      if not pre_processed:
        # not prefetch testing or demo
        images, meta = self.pre_process(image, scale, meta)
      else:
        # prefetch testing
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        if 'pre_dets' in pre_processed_images['meta']:
          meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
        if 'cur_dets' in pre_processed_images['meta']:
          meta['cur_dets'] = pre_processed_images['meta']['cur_dets']
      
      images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)

      # initializing tracker
      pre_hms, pre_inds = None, None
      if self.opt.tracking:
        # initialize the first frame
        if self.pre_images is None:
          p_images = copy.deepcopy(images)
          p_images = p_images.unsqueeze(1)
          p_images = p_images.expand(-1, self.opt.num_pre_imgs_input, -1, -1, -1)
          self.pre_images = p_images
          self.tracker.init_track(
            meta['pre_dets'] if 'pre_dets' in meta else [])
        if self.opt.pre_hm:
          # render input heatmap from tracker status
          # pre_inds is not used in the current version.
          # We used pre_inds for learning an offset from previous image to
          # the current image.
          pre_images, pre_hms, pre_inds, kmf_hms = self._get_additional_inputs(
            self.tracker.tracks, meta, self.pre_images[:, 0, :], self.age_images, 
            with_hm=not self.opt.zero_pre_hm, with_kmf=self.opt.kmf_att)
          if self.opt.num_pre_imgs_input > 1:
            #self.pre_images[:, 0, :] = pre_images # could be failed
            mask = torch.zeros_like(self.pre_images, device=self.pre_images.device, dtype=torch.bool)
            mask[:, 0, :] = True
            self.pre_images = self.pre_images.masked_scatter(mask.byte(), pre_images)
          else:
            self.pre_images = pre_images.unsqueeze(1).expand(-1, self.opt.num_pre_imgs_input, -1, -1, -1)
      
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      # run the network
      # output: the output feature maps, only used for visualizing
      # dets: output tensors after extracting peaks
      output, dets, forward_time = self.process(
        images, self.pre_images, pre_hms, pre_inds, kmf_hms,return_time=True)
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      # convert the cropped and 4x downsampled output coordinate system
      # back to the input image coordinate system
      result = self.post_process(dets, meta, scale)
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(result)
      if self.opt.debug >= 2:
        self.debug(
          self.debugger, images, result, output, scale, 
          pre_images=self.pre_images if not self.opt.no_pre_img else None, 
          pre_hms=pre_hms, kmf_hms=kmf_hms)

    # merge multi-scale testing results
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    
    if self.opt.tracking:
      # public detection mode in MOT challenge
      public_det = meta['cur_dets'] if self.opt.public_det else None
      # add tracking id to results
      results = self.tracker.step(results, public_det) 
      #self.pre_images = images
      # self.pre_images[:, :-1, :] = self.pre_images[:, 1:, :]
      # mask = torch.zeros_like(self.pre_images, device=self.pre_images.device, dtype=torch.bool)
      # mask[0, -1, :] = True
      # self.pre_images = self.pre_images.masked_scatter(mask.byte(), images.squeeze(0))
      self.age_images.append(images.squeeze(0))
      if len(self.age_images) > max(self.opt.max_age):
        self.age_images.pop(0)
      n_img = self.opt.num_pre_imgs_input
      for idx in range(n_img):
        if n_img-idx > len(self.age_images):
          continue
        mask = torch.zeros_like(self.pre_images, device=self.pre_images.device, dtype=torch.bool)
        mask[0, idx, :] = True
        self.pre_images = self.pre_images.masked_scatter(mask.byte(), self.age_images[-n_img+idx])

    tracking_time = time.time()
    track_time += tracking_time - end_time
    tot_time += tracking_time - start_time

    if self.opt.debug >= 1:
      self.show_results(self.debugger, image, results)
    self.cnt += 1

    show_results_time = time.time()
    display_time += show_results_time - end_time
    
    # return results and run time
    ret = {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time, 'track': track_time,
            'display': display_time}
    if self.opt.save_video:
      try:
        # return debug image for saving video
        ret.update({'generic': self.debugger.imgs['generic']})
      except:
        pass
    return ret


  def _transform_scale(self, image, scale=1):
    '''
      Prepare input image in different testing modes.
        Currently support: fix short size/ center crop to a fixed size/ 
        keep original resolution but pad to a multiplication of 32
    '''
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_short > 0:
      if height < width:
        inp_height = self.opt.fix_short
        inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
      else:
        inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
        inp_width = self.opt.fix_short
      c = np.array([width / 2, height / 2], dtype=np.float32)
      s = np.array([width, height], dtype=np.float32)
    elif self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
      # s = np.array([inp_width, inp_height], dtype=np.float32)
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, c, s, inp_width, inp_height, height, width


  def pre_process(self, image, scale, input_meta={}):
    '''
    Crop, resize, and normalize image. Gather meta data for post processing 
      and tracking.
    '''
    resized_image, c, s, inp_width, inp_height, height, width = \
      self._transform_scale(image)
    #print(f'input hw: {inp_height} {inp_width} ,img hw: {height} {width}')
    #print(np.shape(resized_image))
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    out_height =  inp_height // self.opt.down_ratio
    out_width =  inp_width // self.opt.down_ratio
    trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
             if 'calib' in input_meta else \
             self._get_default_calib(width, height)}
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
            'out_height': out_height, 'out_width': out_width,
            'inp_height': inp_height, 'inp_width': inp_width,
            'trans_input': trans_input, 'trans_output': trans_output})
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']
    return images, meta


  def _trans_bbox(self, bbox, trans, width, height):
    '''
    Transform bounding boxes according to image crop.
    '''
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox


  def _get_additional_inputs(self, tracks, meta, pre_images, age_images, with_hm=True, with_kmf=True):
    '''
    Render input heatmap from previous trackings.
    '''
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)
    kmf_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

    output_inds = []
    for track in tracks:
      if track['score'] < self.opt.pre_thresh[track['class']-1]: #or det['active'] == 0:
        continue
      bbox = self._trans_bbox(track['bbox'], trans_input, inp_width, inp_height)
      bbox_out = self._trans_bbox(
        track['bbox'], trans_output, out_width, out_height)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        if 'seg' in self.opt.task  and self.opt.seg_center:
          seg_mask = self.get_masks_as_input(track, trans_input)
          ct = np.array([np.mean(np.where(seg_mask>=0.5)[1]), np.mean(np.where(seg_mask>=0.5)[0])], dtype=np.float32)
        else:
          ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if with_hm:
          draw_umich_gaussian(input_hm[0], ct_int, radius)
        if with_kmf and track['age'] <= 1:
          p_bbox = track['kmf'].predict()[0]
          p_bbox = self._trans_bbox(p_bbox, trans_input, inp_width, inp_height)
          p_h, p_w = p_bbox[3] - p_bbox[1], p_bbox[2] - p_bbox[0]
          p_radius = gaussian_radius((math.ceil(p_h), math.ceil(p_w)))
          p_radius = max(0, int(p_radius))
          p_ct_int = np.array(
            [(p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2], dtype=np.float32).astype(np.int32)
          if (p_h > 0) and (p_w > 0) and (p_ct_int[0] > 0) and (p_ct_int[1]> 0) and (p_ct_int[0] < inp_width) and (p_ct_int[1] < inp_height):
            if self.opt.guss_oval:
              draw_umich_gaussian_oval(kmf_hm[0], p_ct_int, radius_h=h//2, radius_w=w//2)
            else:
              draw_umich_gaussian(kmf_hm[0], p_ct_int, p_radius)

        ct_out = np.array(
          [(bbox_out[0] + bbox_out[2]) / 2, 
           (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
        output_inds.append(ct_out[1] * out_width + ct_out[0])
      if track['age'] > 1 and self.opt.paste_up:
        track['segmentation'] = track['seg']
        masks_to_be_paste = self.merge_masks_as_input([track], trans_input)
        pre_images = copy_paste_with_seg_mask(pre_images.squeeze(0), age_images[-track['age']], masks_to_be_paste, blend=False)
        pre_images = pre_images.unsqueeze(0)

    if with_hm:
      input_hm = input_hm[np.newaxis]
      if self.opt.flip_test:
        input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
      input_hm = torch.from_numpy(input_hm).to(self.opt.device)
    if with_kmf:
      if not self.opt.keep_att:
        kmf_hm = kmf_hm * 0.5 + 0.5
      kmf_hm = kmf_hm[np.newaxis]
      if self.opt.flip_test:
        kmf_hm = np.concatenate((kmf_hm, kmf_hm[:, :, :, ::-1]), axis=0)
      kmf_hm = torch.from_numpy(kmf_hm).to(self.opt.device)
    else:
      kmf_hm = None
    
    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(self.opt.device)
    return pre_images, input_hm, output_inds, kmf_hm

  def merge_masks_as_input(self, anns, trans_input):
      rles = [ann['segmentation'] for ann in anns]
      mgrle = mask_utils.merge(rles)
      mask = mask_utils.decode(mgrle)
      inp = cv2.warpAffine(mask, trans_input, 
                    (self.opt.input_w, self.opt.input_h),
                    flags=cv2.INTER_LINEAR)
      return inp
  def get_masks_as_input(self, ann, trans_input):
      rle = ann['seg']
      mask = mask_utils.decode(rle)
      inp = cv2.warpAffine(mask, trans_input, 
                    (self.opt.input_w, self.opt.input_h),
                    flags=cv2.INTER_LINEAR)
      return inp

  def _get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = output['hm'].sigmoid_()
    if 'hm_hp' in output:
      output['hm_hp'] = output['hm_hp'].sigmoid_()
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      output['dep'] *= self.opt.depth_scale
    return output


  def _flip_output(self, output):
    average_flips = ['hm', 'wh', 'dep', 'dim']
    neg_average_flips = ['amodel_offset']
    single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
      'hp_offset', 'rot', 'tracking', 'pre_hm']
    for head in output:
      if head in average_flips:
        output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
      if head in neg_average_flips:
        flipped_tensor = flip_tensor(output[head][1:2])
        flipped_tensor[:, 0::2] *= -1
        output[head] = (output[head][0:1] + flipped_tensor) / 2
      if head in single_flips:
        output[head] = output[head][0:1]
      if head == 'hps':
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
      if head == 'hm_hp':
        output['hm_hp'] = (output['hm_hp'][0:1] + \
          flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

    return output


  def process(self, images, pre_images=None, pre_hms=None,
    pre_inds=None, kmf_hms=None, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images, pre_images, pre_hms, kmf_hms)[-1]
      output = self._sigmoid_output(output)
      output.update({'pre_inds': pre_inds})
      if self.opt.flip_test:
        output = self._flip_output(output)
      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets = generic_decode(output, K=self.opt.K, opt=self.opt)
      torch.cuda.synchronize()
      for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = generic_post_process(
      self.opt, dets, [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], self.opt.num_classes,
      [meta['calib']], meta['height'], meta['width'])
    self.this_calib = meta['calib']
    if scale != 1:
      for i in range(len(dets[0])):
        for k in ['bbox', 'hps']:
          if k in dets[0][i]:
            dets[0][i][k] = (np.array(
              dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]

  def merge_outputs(self, detections):
    assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
    results = []
    for i in range(len(detections[0])):
      if detections[0][i]['score'] > self.opt.out_thresh[detections[0][i]['class'] - 1]:
        results.append(detections[0][i])
    return results

  def debug(self, debugger, images, dets, output, scale=1, 
    pre_images=None, pre_hms=None, kmf_hms=None):
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if 'hm_hp' in output:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')

    if pre_images is not None:
      pre_img = pre_images[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
      pre_img = np.clip(((
        pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      debugger.add_img(pre_img, 'pre_img')
      if pre_images.size()[1] > 1:
        pre_img = pre_images[0, 1].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_1')
      if pre_hms is not None:
        pre_hm = debugger.gen_colormap(
          pre_hms[0].detach().cpu().numpy())
        debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')
      if kmf_hms is not None:
        kmf_hm = debugger.gen_colormap(
          kmf_hms[0].detach().cpu().numpy())
        debugger.add_blend_img(img, kmf_hm, 'kmf_hm')

    if 'tracking' in output and self.opt.show_arrowmap:
      debugger.add_img(img, img_id='tracking_arrowmap')
      debugger.add_arrows(output['tracking'], img_id='tracking_arrowmap')



  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='generic')
    if self.opt.tracking:
      debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image, 
        img_id='previous')
      self.pre_image_ori = image
    for j in range(len(results)):
      if results[j]['score'] > self.opt.vis_thresh:
        if 'active' in results[j] and results[j]['active'] == 0:
          continue
        item = results[j]

        sc = item['score'] if self.opt.demo == '' or \
          not ('tracking_id' in item) else item['tracking_id']
        sc = item['tracking_id'] if self.opt.show_track_color else sc
        if ('seg' in item):
          seg_mask = mask_utils.decode(item['seg'])
          debugger.add_coco_seg(
            seg_mask, item['class'] - 1, sc, img_id='generic')
        elif ('bbox' in item):
          debugger.add_coco_bbox(
            item['bbox'], item['class'] - 1, sc, img_id='generic')

        if 'tracking' in item:
          debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')
        
        tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
        if 'tracking_id' in item and self.opt.demo == '' and \
          not self.opt.show_track_color:
          debugger.add_tracking_id(
            item['ct'], item['tracking_id'], img_id='generic')

        if (item['class'] in [1, 2]) and 'hps' in item:
          debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
            img_id='generic')

    if len(results) > 0 and \
      'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
      debugger.add_3d_detection(
        image if not self.opt.qualitative else cv2.resize(
          debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])), 
        False, results, self.this_calib,
        vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
      debugger.add_bird_view(
        results, vis_thresh=self.opt.vis_thresh,
      img_id='bird_pred', cnt=self.cnt)
      if self.opt.show_track_color and self.opt.debug == 4:
        del debugger.imgs['generic'], debugger.imgs['bird_pred']
    if 'ddd_pred' in debugger.imgs:
      debugger.imgs['generic'] = debugger.imgs['ddd_pred']
    if self.opt.debug == 4:
      debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
    else:
      debugger.show_all_imgs(pause=self.pause)
  

  def reset_tracking(self):
    self.tracker.reset()
    self.pre_images = None
    self.pre_image_ori = None
