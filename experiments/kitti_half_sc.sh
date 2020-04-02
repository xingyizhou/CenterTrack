cd src
# train
python main.py tracking --exp_id kitti_half_sc --dataset kitti_tracking --dataset_version train_half --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 16
# test
python test.py tracking --exp_id kitti_half_sc --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --pre_thresh 0.5 --resume