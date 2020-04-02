cd src
# train 
python main.py tracking,ddd --exp_id nuScenes_3Dtracking --dataset nuscenes --pre_hm --load_model ../models/nuScenes_3Ddetection_e140.pth --shift 0.01 --scale 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --hm_disturb 0.05 --batch_size 64 --gpus 0,1,2,3 --lr 2.5e-4 --save_point 60
# test
python test.py tracking,ddd --exp_id nuScenes_3Dtracking --dataset nuscenes --pre_hm --track_thresh 0.1 --resume
cd ..