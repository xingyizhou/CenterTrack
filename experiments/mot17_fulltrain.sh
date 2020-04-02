cd src
# train
python main.py tracking --exp_id mot17_fulltrain --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1 --load_model ../models/crowdhuman.pth
# test
python test.py tracking --exp_id mot17_fulltrain --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume
cd ..