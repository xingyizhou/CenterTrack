import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy
from .kalman_filter import KalmanBoxTracker

class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.reset()

  def init_track(self, results):
    for item in results:
      if item['score'] > self.opt.new_thresh[item['class']-1]:
        self.id_count += 1
        # active and age are never used in the paper
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        item['kmf'] = KalmanBoxTracker(item['bbox'])
        self.tracks.append(item)

  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step(self, results, public_det=None):
    N = len(results)
    M = len(self.tracks)

    dets = np.array(
      [det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
      (track['bbox'][3] - track['bbox'][1])) \
      for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
      (item['bbox'][3] - item['bbox'][1])) \
      for item in results], np.float32) # N
    item_cat = np.array([item['class'] for item in results], np.int32) # N
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2
    dist = (((tracks.reshape(1, -1, 2) - \
              dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M

    invalid = ((dist > track_size.reshape(1, M)) + \
      (dist > item_size.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    #invalid =  (item_cat.reshape(N, 1) != track_cat.reshape(1, M)) > 0
    dist = dist + invalid * 1e18
    
    if self.opt.hungarian:
      item_score = np.array([item['score'] for item in results], np.float32) # N
      dist[dist > 1e18] = 1e18
      matched_indices = linear_assignment(dist)
    else:
      matched_indices = greedy_assignment(copy.deepcopy(dist))
    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.opt.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
          unmatched_tracks.append(m[1])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    for m in matches:
      result = results[m[0]]
      result['tracking_id'] = self.tracks[m[1]]['tracking_id']
      result['age'] = 1
      result['active'] = self.tracks[m[1]]['active'] + 1
      result['kmf'] = self.tracks[m[1]]['kmf']
      result['kmf'].update(result['bbox'])
      ret.append(result)

    if self.opt.public_det and len(unmatched_dets) > 0:
      # Public detection: only create tracks from provided detections
      pub_dets = np.array([d['ct'] for d in public_det], np.float32)
      dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(
        axis=2)
      matched_dets = [d for d in range(dets.shape[0]) \
        if not (d in unmatched_dets)]
      dist3[matched_dets] = 1e18
      for j in range(len(pub_dets)):
        i = dist3[:, j].argmin()
        if dist3[i, j] < item_size[i]:
          dist3[i, :] = 1e18
          result = results[i]
          if result['score'] > self.opt.new_thresh[result['class']-1]:
            self.id_count += 1
            result['tracking_id'] = self.id_count
            result['age'] = 1
            result['active'] = 1
            result['kmf'] = KalmanBoxTracker(result['bbox'])
            ret.append(result)
    else:
      # Private detection: create tracks for all un-matched detections
      for i in unmatched_dets:
        result = results[i]
        if result['score'] > self.opt.new_thresh[result['class']-1]:
          self.id_count += 1
          result['tracking_id'] = self.id_count
          result['age'] = 1
          result['active'] =  1
          result['kmf'] = KalmanBoxTracker(result['bbox'])
          ret.append(result)
    
    
    tracks = copy.deepcopy(ret)

    for i in unmatched_tracks: # for "tracker" unmatched
      track = self.tracks[i]
      if track['age'] < self.opt.max_age[track['class']-1]:
        track['age'] += 1
        track['active'] = 0
        bbox = track['bbox']
        ct = track['ct']
        v = [0, 0]
        track['bbox'] = [
          bbox[0] + v[0], bbox[1] + v[1],
          bbox[2] + v[0], bbox[3] + v[1]]
        track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
        tracks.append(track)
    self.tracks = tracks
    return ret

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)
