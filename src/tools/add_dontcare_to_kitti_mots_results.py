import os
import sys

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

def add_dontcare(seq):
    gt_ann_file = os.path.join(GT_PATH, f"{seq}.txt")
    pred_ann_file = os.path.join(RESULTS_PATH, f"{seq}.txt")
   
    dcare = []
    with open(gt_ann_file, "r") as fh:
        for i, line in enumerate(fh):
            fields = line.split(" ")
            if fields[2] == "10":
                dcare.append(line)
    with open(pred_ann_file, "a") as pf:
        for line in dcare:
            pf.write(line)
RESULTS_PATH = sys.argv[1]
SEQS_MAP_PATH = sys.argv[2]
GT_PATH = '../../data/kitti_mots/instances_txt/'

if __name__ == '__main__':

    print(f'Results from {RESULTS_PATH}')

    seqmaps, max_frames = load_seqmap(SEQS_MAP_PATH)
    for seq in seqmaps:
        print('Adding segs', seq)
        add_dontcare(seq)


            