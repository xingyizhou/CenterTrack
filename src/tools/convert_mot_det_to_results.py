import json
import numpy as np
import os
from collections import defaultdict
split = 'val_half'

DET_PATH = '../../data/mot17/'
ANN_PATH = '../../data/mot17/annotations/{}.json'.format(split)
OUT_DIR = '../../data/mot17/results/'
OUT_PATH = OUT_DIR + '{}_det.json'.format(split)

if __name__ == '__main__':
  if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
  seqs = [s for s in os.listdir(DET_PATH) if '_det' in s]
  data = json.load(open(ANN_PATH, 'r'))
  images = data['images']
  image_to_anns = defaultdict(list)
  for seq in sorted(seqs):
    print('seq', seq)
    seq_path = '{}/{}/'.format(DET_PATH, seq)
    if split == 'val_half':
      ann_path = seq_path + 'det/det_val_half.txt'
      train_ann_path = seq_path + 'det/det_train_half.txt'
      train_anns = np.loadtxt(train_ann_path, dtype=np.float32, delimiter=',')
      frame_base = int(train_anns[:, 0].max())
    else:
      ann_path = seq_path + 'det/det.txt'
      frame_base = 0
    if not IS_THIRD_PARTY:
      anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
    for i in range(len(anns)):
      frame_id = int(anns[i][0])
      file_name = '{}/img1/{:06d}.jpg'.format(seq, frame_id + frame_base)
      bbox = (anns[i][2:6]).tolist()
      score = 1 # float(anns[i][8])
      image_to_anns[file_name].append(bbox + [score])

  results = {}
  for image_info in images:
    image_id = image_info['id']
    file_name = image_info['file_name']
    dets = image_to_anns[file_name]
    results[image_id] = []
    for det in dets:
      bbox = [float(det[0]), float(det[1]), \
              float(det[0] + det[2]), float(det[1] + det[3])]
      ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
      results[image_id].append(
        {'bbox': bbox, 'score': float(det[4]), 'class': 1, 'ct': ct})
  out_path = OUT_PATH
  json.dump(results, open(out_path, 'w'))
