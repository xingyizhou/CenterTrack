import os
import numpy as np
import json
import cv2

# Use the same script for MOT16
DATA_PATH = '../../data/mots20/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['train', 'test']
HALF_VIDEO = False
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + (split if not HALF_VIDEO else 'train')
    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'pedestrain'}],
           'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      num_images = len([image for image in images if 'jpg' in image])

      image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        det_path = seq_path + 'det/det.txt'
        ## start loading anns
        anns_txts = open(ann_path, 'r')
        for ann_ind, txt in enumerate(anns_txts):
          tmp = txt[:-1].split(' ')
          frame_id = int(tmp[0])
          track_id = int(tmp[1])
          cat_id = int(tmp[2])
          img_height = int(tmp[3])
          img_width = int(tmp[4])
          seg_mask = str(tmp[5])
        
          ann = {'image_id': frame_id + image_cnt,
                'id': int(len(out['annotations']) + 1),
                'category_id': cat_id,
                'segmentation': {"counts": seg_mask, "size": (img_height, img_width)}, 
                'track_id': track_id + 1}
          
          out['annotations'].append(ann)
        
        ## end loading anns
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
        

