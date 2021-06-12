from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def load_seqmap(seqmap_filename):
  print("Loading seqmap...")
  seqmap = []
  max_frames = {}
  with open(seqmap_filename, "r") as fh:
    for i, l in enumerate(fh):
      fields = l.split(" ")
      seq = "%04d" % int(fields[0])
      seqmap.append(seq)
      max_frames[seq] = int(fields[3])
  return seqmap, max_frames

def evaluation(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  #opt.debug = max(opt.debug, 1)
  detector = Detector(opt)
  seqmaps, max_frames = load_seqmap(opt.inf_seqmap)
  for seq in seqmaps:
    inference(opt, detector, seq, max_frames[seq])
    detector.reset_tracking()
  print('inference done for:', seqmaps)



def inference(opt, detector, seqmap, max_frames):
  
  image_dir = os.path.join(opt.inf_dir, seqmap)
   
  # Demo on images sequences
  image_names = []
  ls = os.listdir(image_dir)
  for file_name in sorted(ls):
      ext = file_name[file_name.rfind('.') + 1:].lower()
      if ext in image_ext:
          image_names.append(os.path.join(image_dir, file_name))


  # Initialize output video
  out = None
  out_name = seqmap
  print('out_name', out_name)
  if opt.save_video:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('../results/{}.mp4'.format(
      opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}
  is_video = False

  while True:
      if is_video:
        _, img = cam.read()
        if img is None:
          save_and_exit(opt, out, results, out_name, detector.start_epoch)
          return 
      else:
        if cnt < len(image_names):
          img = cv2.imread(image_names[cnt])
        else:
          save_and_exit(opt, out, results, out_name, detector.start_epoch)
          return 
      cnt += 1

      # resize the original video for saving video results
      if opt.resize_video:
        img = cv2.resize(img, (opt.video_w, opt.video_h))

      # skip the first X frames of the video
      if cnt < opt.skip_first:
        continue
      
      #cv2.imshow('input', img)

      # track or detect the image.
      ret = detector.run(img)

      # log run time
      time_str = 'frame {} |'.format(cnt)
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      # results[cnt] is a list of dicts:
      #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
      results[cnt] = ret['results']

      # save debug image to video
      if opt.save_video:
        out.write(ret['generic'])
        if not is_video:
          cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])
      
      # esc to quit and finish saving video
      if cv2.waitKey(1) == 27:
        save_and_exit(opt, out, results, out_name)
        return 
  save_and_exit(opt, out, results, out_name)
  return 


def save_and_exit(opt, out=None, results=None, out_name='default', start_epoch=''):
  dirname = f"{opt.exp_id}{start_epoch}" if opt.dir_suffix is None else f"{opt.exp_id}{start_epoch}-{opt.dir_suffix}"
  save_dir = os.path.join('../results/data/', dirname)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if (results is not None):
    save_path =  os.path.join(save_dir, '{}.json'.format(out_name))
    print('saving results to', save_path)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_path, 'w'))
  if 'seg' in opt.task and 'tracking' in opt.task:
    save_path = os.path.join(save_dir, f"{out_name}.txt")
    print('saving results for mots_tools to', save_path)
    json2mots(opt, results, save_path)
  if opt.save_video and out is not None:
    out.release()
  #sys.exit(0)
coco2kitti = {1: 2, 3: 1}
def json2mots(opt, results, save_dir):
  with open(save_dir, "w") as fp:
    for time_frame in results.keys():
        for obj in results[time_frame]:
            track_id = str(obj['class']) + "{0:03}".format(obj['tracking_id'])
            if not opt.dataset == 'kitti_mots' and obj['class'] not in coco2kitti:
              continue
            class_id = obj['class'] if opt.dataset == 'kitti_mots' else coco2kitti[int(obj['class'])]
            img_height = obj['seg']['size'][0]
            img_width =  obj['seg']['size'][1]
            seg_rle = obj['seg']['counts']
            line = f"{str(int(time_frame)-1)} {track_id} {class_id} {img_height} {img_width} {seg_rle}\n"
            fp.write(line)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  evaluation(opt)
