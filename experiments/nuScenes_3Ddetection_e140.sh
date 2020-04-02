cd src
# train 
python main.py ddd --exp_id nuScenes_3Ddetection_e140 --dataset nuscenes --batch_size 128 --gpus 0,1,2,3,4,5,6,7 --lr 5e-4 --num_epochs 140 --lr_step 90,120 --save_point 90,120
# test
python test.py ddd --exp_id nuScenes_3Ddetection_e140 --dataset nuscenes --resume
cd ..