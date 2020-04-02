import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

GT_PATH = '../../data/mot17/test/'
IMG_PATH = GT_PATH
SAVE_VIDEO = True
RESIZE = 2
IS_GT = False

def draw_bbox(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
      c, 2, lineType=cv2.LINE_AA)
    ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    txt = '{}'.format(bbox[4])
    cv2.putText(img, txt, (int(ct[0]), int(ct[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (255, 122, 255), thickness=1, lineType=cv2.LINE_AA)

if __name__ == '__main__':
  seqs = os.listdir(GT_PATH)
  if SAVE_VIDEO:
    save_path = sys.argv[1][:sys.argv[1].rfind('/res')] + '/video'
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    print('save_video_path', save_path)
  for seq in sorted(seqs):
    print('seq', seq)
    # if len(sys.argv) > 2 and not sys.argv[2] in seq:
    #   continue
    if '.DS_Store' in seq:
      continue
    # if SAVE_VIDEO:
    #   fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #   video = cv2.VideoWriter(
    #     '{}/{}.avi'.format(save_path, seq),fourcc, 10.0, (1024, 750))
    seq_path = '{}/{}/'.format(GT_PATH, seq)
    if IS_GT:
      ann_path = seq_path + 'gt/gt.txt'
    else:
      ann_path = seq_path + 'det/det.txt'
    anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
    print('anns shape', anns.shape)
    image_to_anns = defaultdict(list)
    for i in range(anns.shape[0]):
      if (not IS_GT) or (int(anns[i][6]) == 1 and float(anns[i][8]) >= 0.25):
        frame_id = int(anns[i][0])
        track_id = int(anns[i][1])
        bbox = (anns[i][2:6] / RESIZE).tolist()
        image_to_anns[frame_id].append(bbox + [track_id])
    
    image_to_preds = {}
    for K in range(1, len(sys.argv)):
      image_to_preds[K] = defaultdict(list)
      pred_path = sys.argv[K] + '/{}.txt'.format(seq)
      try:
        preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=',')
      except:
        preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=' ')
      for i in range(preds.shape[0]):
        frame_id = int(preds[i][0])
        track_id = int(preds[i][1])
        bbox = (preds[i][2:6] / RESIZE).tolist()
        image_to_preds[K][frame_id].append(bbox + [track_id])
    
    img_path = seq_path + 'img1/'
    images = os.listdir(img_path)
    num_images = len([image for image in images if 'jpg' in image])
    
    for i in range(num_images):
      frame_id = i + 1
      file_name = '{}/img1/{:06d}.jpg'.format(seq, i + 1)
      file_path = IMG_PATH + file_name
      img = cv2.imread(file_path)
      if RESIZE != 1:
        img = cv2.resize(img, (img.shape[1] // RESIZE, img.shape[0] // RESIZE))
      for K in range(1, len(sys.argv)):
        img_pred = img.copy()
        draw_bbox(img_pred, image_to_preds[K][frame_id])
        cv2.imshow('pred{}'.format(K), img_pred)
      draw_bbox(img, image_to_anns[frame_id])
      cv2.imshow('gt', img)
      cv2.waitKey()
      # if SAVE_VIDEO:
      #   video.write(img_pred)
    # if SAVE_VIDEO:
    #   video.release()
