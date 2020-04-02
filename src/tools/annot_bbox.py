import os
import sys
import json
import cv2
import argparse
import numpy as np
image_ext = ['jpg', 'jpeg', 'png', 'webp']

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='')
parser.add_argument('--save_path', default='')
MAX_CACHE = 20
CAT_NAMES = ['cat']

def _sort_expt(pts):
  t, l, b, r = 0, 0, 0, 0
  for i in range(4):
    if pts[i][0] < pts[l][0]:
      l = i
    if pts[i][1] < pts[t][1]:
      t = i
    if pts[i][0] > pts[r][0]:
      r = i
    if pts[i][1] > pts[b][1]:
      b = i
  ret = [pts[t], pts[l], pts[b], pts[r]]
  return ret

def _expt2bbox(expt):
  expt = np.array(expt, dtype=np.int32)
  bbox = [int(expt[:, 0].min()), int(expt[:, 1].min()), 
          int(expt[:, 0].max()), int(expt[:, 1].max())]
  return bbox

def save_txt(txt_name, pts_cls):
  ret = []
  for i in range(len(pts_cls)):
    ret.append(np.array(pts_cls[i][:4], dtype=np.int32).reshape(8).tolist() \
               + [pts_cls[i][4]])
  np.savetxt(txt_name, np.array(ret, dtype=np.int32), fmt='%d')

def click(event, x, y, flags, param):
  global expt_cls, bboxes, pts
  if event == cv2.EVENT_LBUTTONDOWN:
    pts.append([x, y])
    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
    if len(pts) == 4:
      expt = _sort_expt(pts)
      bbox = _expt2bbox(expt)
      expt_cls.append(expt + [cls])
      cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                    (255, 0, 255), 2, cv2.LINE_AA)
      pts = []

if __name__ == '__main__':
  cat_info = []
  for i, cat in enumerate(CAT_NAMES):
    cat_info.append({'name': cat, 'id': i + 1})

  args = parser.parse_args()
  if args.save_path == '':
    args.save_path = os.path.join(args.image_path, '..', 'click_annotation')
  if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
  
  ann_path = os.path.join(args.save_path, 'annotations.json')
  if os.path.exists(ann_path):
    anns = json.load(open(ann_path, 'r'))
  else:
    anns = {'annotations': [], 'images': [], 'categories': cat_info}

  assert os.path.exists(args.image_path)
  ls = os.listdir(args.image_path)
  image_names = []
  for file_name in sorted(ls):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    if (ext in image_ext):
      image_names.append(file_name)
  
  i = 0
  cls = 1
  cached = 0
  while i < len(image_names):
    image_name = image_names[i]
    txt_name = os.path.join(
      args.save_path, image_name[:image_name.rfind('.')] + '.txt')
    if os.path.exists(txt_name) or image_name in anns:
      i = i + 1
      continue
    image_path = os.path.join(args.image_path, image_name)
    img = cv2.imread(image_path)
    cv2.namedWindow(image_name)
    cv2.setMouseCallback(image_name, click)
    expt_cls, pts = [], []
    while True:
      finished = False
      cv2.imshow(image_name, img)
      key = cv2.waitKey(1)
      if key == 100:
        i = i + 1
        save_txt(txt_name, expt_cls)
        image_id = len(anns['images'])
        image_info = {'file_name': image_name, 'id': image_id}
        anns['images'].append(image_info)
        for ann in expt_cls:
          ann_id = len(anns['annotations'])
          ann_dict = {'image_id': image_id, 'id': ann_id, 'categoty_id': ann[4],
                      'bbox': _expt2bbox(ann[:4]), 'extreme_points': ann[:4]}
          anns['annotations'].append(ann_dict)
          cached = cached + 1
        print('saved to ', txt_name)
        if cached > MAX_CACHE:
          print('Saving json', ann_path)
          json.dump(anns, open(ann_path, 'w'))
          cached = 0
        break
      elif key == 97:
        i = i - 1
        break
      elif key == 27:
        json.dump(anns, open(ann_path, 'w'))
        sys.exit(0)
    cv2.destroyAllWindows()
