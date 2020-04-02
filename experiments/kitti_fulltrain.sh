cd src
# train
python main.py tracking --exp_id kitti_fulltrain --dataset kitti_tracking --dataset_version train --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 16 --load_model ../models/nuScenes_3Ddetection_e140.pth
# test
python test.py tracking --exp_id kitti_fulltrain --dataset kitti_tracking --dataset_version test --pre_hm --track_thresh 0.4 --resume
