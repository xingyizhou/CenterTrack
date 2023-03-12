from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import cv2
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d


class Debugger(object):
  def __init__(self, opt, dataset):
    self.opt = opt
    self.imgs = {}
    self.theme = opt.debugger_theme
    self.plt = plt
    self.with_3d = False
    self.names = dataset.class_name
    self.out_size = 384 if opt.dataset == 'kitti' else 512
    self.cnt = 0
    colors = [(color_list[i]).astype(np.uint8) for i in range(len(color_list))]
    while len(colors) < len(self.names):
      colors = colors + colors[:min(len(colors), len(self.names) - len(colors))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
  
    self.num_joints = 17
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [3, 5], [4, 6], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [5, 11], [6, 12], [11, 12], 
                  [11, 13], [13, 15], [12, 14], [14, 16]]
    self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
                (255, 0, 0), (0, 0, 255), (255, 0, 255),
                (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                (255, 0, 0), (0, 0, 255), (255, 0, 255),
                (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
    self.colors_hp = [(128, 0, 128), (128, 0, 0), (0, 0, 128), 
      (128, 0, 0), (0, 0, 128), (128, 0, 0), (0, 0, 128),
      (128, 0, 0), (0, 0, 128), (128, 0, 0), (0, 0, 128),
      (128, 0, 0), (0, 0, 128), (128, 0, 0), (0, 0, 128),
      (128, 0, 0), (0, 0, 128)]
    self.track_color = {}
    self.trace = {}
    # print('names', self.names)
    self.down_ratio=opt.down_ratio
    # for bird view
    self.world_size = 64


  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()
  
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    # ignore region
    img[img == 1] = 0.5
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    if self.opt.tango_color:
      colors = tango_color_dark[:c].reshape(1, 1, c, 3)
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[1], output_res[0]))
    return color_map
    
  def gen_colormap_hp(self, img, output_res=None):
    img = img.copy()
    img[img == 1] = 0.5 
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map

  def _get_rand_color(self):
    c = ((np.random.random((3)) * 0.6 + 0.2) * 255).astype(np.int32).tolist()
    return c

  def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, 
    no_bbox=False, img_id='default'): 
    if self.opt.show_track_color:
      track_id = int(conf)
      if not (track_id in self.track_color):
        self.track_color[track_id] = self._get_rand_color()
      c = self.track_color[track_id]
      # thickness = 4
      # fontsize = 0.8
    if self.opt.only_show_dots:
      ct = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
      cv2.circle(
        self.imgs[img_id], ct ,8, c, -1, lineType=cv2.LINE_AA)
      if self.opt.show_trace:
        if track_id in self.trace:
          trace = self.trace[track_id]
          cnt = 0
          t_pre = ct
          for t in trace[::-1]:
            cv2.circle(
              self.imgs[img_id], t ,6-cnt*2, c, -1, lineType=cv2.LINE_AA)
            cv2.line(self.imgs[img_id], t, t_pre, c, max(6-cnt*2, 1), lineType=cv2.LINE_AA)
            t_pre = t
            cnt = cnt + 1
            if cnt >= 3:
              break
          self.trace[track_id].append(ct)
        else:
          self.trace[track_id] = [ct]
      return
    bbox = np.array(bbox, dtype=np.int32)
    cat = int(cat)
    c = self.colors[cat][0][0].tolist()
    if self.theme == 'white':
      c = (255 - np.array(c)).tolist()
    if self.opt.tango_color:
      c = (255 - tango_color_dark[cat][0][0]).tolist()
    if conf >= 1:
      ID = int(conf) if not self.opt.not_show_number else ''
      txt = '{}{}'.format(self.names[cat], ID)
    else:
      txt = '{}{:.1f}'.format(self.names[cat], conf)
    thickness = 2
    fontsize = 0.8 if self.opt.qualitative else 0.5
    if not self.opt.not_show_bbox:
      font = cv2.FONT_HERSHEY_SIMPLEX
      cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
      if not no_bbox:
        cv2.rectangle(
          self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
          c, thickness)
        
      if show_txt:
        cv2.rectangle(self.imgs[img_id],
                      (bbox[0], bbox[1] - cat_size[1] - thickness),
                      (bbox[0] + cat_size[0], bbox[1]), c, -1)
        cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - thickness - 1), 
                    font, fontsize, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

  def add_tracking_id(self, ct, tracking_id, img_id='default'):
    txt = '{}'.format(tracking_id)
    fontsize = 0.5
    cv2.putText(self.imgs[img_id], txt, (int(ct[0]), int(ct[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, 
                (255, 0, 255), thickness=1, lineType=cv2.LINE_AA)


  def add_coco_hp(self, points, tracking_id=0, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    if not self.opt.show_track_color:
      for j in range(self.num_joints):
        cv2.circle(self.imgs[img_id],
                  (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)

    h, w = self.imgs[img_id].shape[0], self.imgs[img_id].shape[1]
    for j, e in enumerate(self.edges):
      if points[e].min() > 0 and points[e, 0].max() < w and \
        points[e, 1].max() < h:
        c = self.ec[j] if not self.opt.show_track_color else \
          self.track_color[tracking_id]
        cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                      (points[e[1], 0], points[e[1], 1]), c, 2,
                      lineType=cv2.LINE_AA)

  def clear(self):
    return

  def show_all_imgs(self, pause=False, Time=0):
    if 1:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if not self.with_3d:
        cv2.waitKey(0 if pause else 1)
      else:
        max_range = np.array([
          self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
        Xb = 0.5*max_range*np.mgrid[
          -1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
        Yb = 0.5*max_range*np.mgrid[
          -1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
        Zb = 0.5*max_range*np.mgrid[
          -1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
        for xb, yb, zb in zip(Xb, Yb, Zb):
          self.ax.plot([xb], [yb], [zb], 'w')
        if self.opt.debug == 9:
          self.plt.pause(1e-27)
        else:
          self.plt.show()
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          plt.imshow(v)
      plt.show()

  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    for i, v in self.imgs.items():
      if i in self.opt.save_imgs or self.opt.save_imgs == []:
        cv2.imwrite(
          path + '/{}{}{}.png'.format(prefix, i, self.opt.save_img_suffix), v)

  def remove_side(self, img_id, img):
    if not (img_id in self.imgs):
      return
    ws = img.sum(axis=2).sum(axis=0)
    l = 0
    while ws[l] == 0 and l < len(ws):
      l+= 1
    r = ws.shape[0] - 1
    while ws[r] == 0 and r > 0:
      r -= 1
    hs = img.sum(axis=2).sum(axis=1)
    t = 0
    while hs[t] == 0 and t < len(hs):
      t += 1
    b = hs.shape[0] - 1
    while hs[b] == 0 and b > 0:
      b -= 1
    self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

  def project_3d_to_bird(self, pt):
    pt[0] += self.world_size / 2
    pt[1] = self.world_size - pt[1]
    pt = pt * self.out_size / self.world_size
    return pt.astype(np.int32)

  def add_3d_detection(
    self, image_or_path, flipped, dets, calib, show_txt=False, 
    vis_thresh=0.3, img_id='det'):
    if self.opt.only_show_dots:
      return
    if isinstance(image_or_path, np.ndarray):
      self.imgs[img_id] = image_or_path.copy()
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    # thickness = 1
    if self.opt.show_track_color:
      # self.imgs[img_id] = (self.imgs[img_id] * 0.5 + \
      #   np.ones_like(self.imgs[img_id]) * 255 * 0.5).astype(np.uint8)
        # thickness = 3
      pass
    if flipped:
      self.imgs[img_id] = self.imgs[img_id][:, ::-1].copy()
    for item in dets:
      if item['score'] > vis_thresh \
        and 'dim' in item and 'loc' in item and 'rot_y' in item:
        cl = (self.colors[int(item['class']) - 1, 0, 0]).tolist() \
          if not self.opt.show_track_color else \
          self.track_color[int(item['tracking_id'])]
        if self.theme == 'white' and not self.opt.show_track_color:
          cl = (255 - np.array(cl)).tolist()
        if self.opt.tango_color:
          cl = (255 - tango_color_dark[int(item['class']) - 1, 0, 0]).tolist()
        dim = item['dim']
        loc = item['loc']
        rot_y = item['rot_y']
        if loc[2] > 1:
          box_3d = compute_box_3d(dim, loc, rot_y)
          box_2d = project_to_image(box_3d, calib)
          self.imgs[img_id] = draw_box_3d(
            self.imgs[img_id], box_2d.astype(np.int32), cl, 
            same_color=self.opt.show_track_color or self.opt.qualitative)
          if self.opt.show_track_color or self.opt.qualitative:
            bbox = [box_2d[:,0].min(), box_2d[:,1].min(),
                    box_2d[:,0].max(), box_2d[:,1].max()]
            sc = int(item['tracking_id']) if self.opt.show_track_color else \
              item['score']
            self.add_coco_bbox(
              bbox, item['class'] - 1, sc, show_txt=not self.opt.not_show_txt,
              no_bbox=True, img_id=img_id)
          if self.opt.show_track_color:
            self.add_arrow([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], 
              item['tracking'], img_id=img_id)

    # print('===========================')
  def compose_vis_ddd(
    self, img_path, flipped, dets, calib,
    vis_thresh, pred, bev, img_id='out'):
    self.imgs[img_id] = cv2.imread(img_path)
    if flipped:
      self.imgs[img_id] = self.imgs[img_id][:, ::-1].copy()
    h, w = pred.shape[:2]
    hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
    self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
    self.add_blend_img(self.imgs[img_id], pred, img_id)
    for item in dets:
      if item['score'] > vis_thresh:
        dim = item['dim']
        loc = item['loc']
        rot_y = item['rot_y']
        cl = (self.colors[int(item['class']) - 1, 0, 0]).tolist()
        if loc[2] > 1:
          box_3d = compute_box_3d(dim, loc, rot_y)
          box_2d = project_to_image(box_3d, calib)
          box_2d[:, 0] /= hs
          box_2d[:, 1] /= ws
          self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

    self.imgs[img_id] = np.concatenate(
      [self.imgs[img_id], self.imgs[bev]], axis=1)

  def add_bird_view(self, dets, vis_thresh=0.3, img_id='bird', cnt=0):
    if self.opt.vis_gt_bev:
      bird_view = cv2.imread(
        self.opt.vis_gt_bev + '/{}bird_pred_gt.png'.format(cnt))
    else:
      bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for item in dets:
      cl = (self.colors[int(item['class']) - 1, 0, 0]).tolist()
      lc = (250, 152, 12)
      if item['score'] > vis_thresh:
        dim = item['dim']
        loc = item['loc']
        rot_y = item['rot_y']
        rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
        for k in range(4):
          rect[k] = self.project_3d_to_bird(rect[k])
        cv2.polylines(
            bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
            True,lc,2,lineType=cv2.LINE_AA)
        for e in [[0, 1]]:
          t = 4 if e == [0, 1] else 1
          cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                  (rect[e[1]][0], rect[e[1]][1]), lc, t,
                  lineType=cv2.LINE_AA)

    self.imgs[img_id] = bird_view

  def add_bird_views(self, dets_dt, dets_gt, vis_thresh=0.3, img_id='bird'):
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for ii, (dets, lc, cc) in enumerate(
      [(dets_gt, (12, 49, 250), (0, 0, 255)), 
       (dets_dt, (250, 152, 12), (255, 0, 0))]):
      for item in dets:
        if item['score'] > vis_thresh \
          and 'dim' in item and 'loc' in item and 'rot_y' in item:
          dim = item['dim']
          loc = item['loc']
          rot_y = item['rot_y']
          rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
          for k in range(4):
            rect[k] = self.project_3d_to_bird(rect[k])
          if ii == 0:
            cv2.fillPoly(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              lc,lineType=cv2.LINE_AA)
          else:
            cv2.polylines(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              True,lc,2,lineType=cv2.LINE_AA)
          # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
          for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                    (rect[e[1]][0], rect[e[1]][1]), lc, t,
                    lineType=cv2.LINE_AA)

    self.imgs[img_id] = bird_view

  def add_arrow(self, st, ed, img_id, c=(255, 0, 255), w=2):
    if self.opt.only_show_dots:
      return
    cv2.arrowedLine(
      self.imgs[img_id], (int(st[0]), int(st[1])), 
      (int(ed[0] + st[0]), int(ed[1] + st[1])), c, 2,
      line_type=cv2.LINE_AA, tipLength=0.3)

color_list = np.array(
        [1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.333, 0.000, 0.500,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255


tango_color = [[252, 233,  79], #	Butter 1
  [237, 212,   0], #	Butter 2
  [196, 160,   0], #	Butter 3
  [138, 226,  52], #	Chameleon 1
  [115, 210,  22], #	Chameleon 2
  [ 78, 154,   6], #	Chameleon 3
  [252, 175,  62], #	Orange 1
  [245, 121,   0], #	Orange 2
  [206,  92,   0], #	Orange 3
  [114, 159, 207], #	Sky Blue 1
  [ 52, 101, 164], #	Sky Blue 2
  [ 32,  74, 135], #	Sky Blue 3
  [173, 127, 168], #	Plum 1
  [117,  80, 123], #	Plum 2
  [ 92,  53, 102], #	Plum 3
  [233, 185, 110], #	Chocolate 1
  [193, 125,  17], #	Chocolate 2
  [143,  89,   2], #	Chocolate 3
  [239,  41,  41], #	Scarlet Red 1
  [204,   0,   0], #	Scarlet Red 2
  [164,   0,   0], #	Scarlet Red 3
  [238, 238, 236], #	Aluminium 1
  [211, 215, 207], #	Aluminium 2
  [186, 189, 182], #	Aluminium 3
  [136, 138, 133], #	Aluminium 4
  [ 85,  87,  83], #	Aluminium 5
  [ 46,  52,  54], #	Aluminium 6
]
tango_color = np.array(tango_color, np.uint8).reshape((-1, 1, 1, 3))


tango_color_dark = [
  [114, 159, 207], #	Sky Blue 1
  [196, 160,   0], #	Butter 3
  [ 78, 154,   6], #	Chameleon 3
  [206,  92,   0], #	Orange 3
  [164,   0,   0], #	Scarlet Red 3
  [ 32,  74, 135], #	Sky Blue 3
  [ 92,  53, 102], #	Plum 3
  [143,  89,   2], #	Chocolate 3
  [ 85,  87,  83], #	Aluminium 5
  [186, 189, 182], #	Aluminium 3
]

tango_color_dark = np.array(tango_color_dark, np.uint8).reshape((-1, 1, 1, 3))
