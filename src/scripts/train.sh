CUDA_VISIBLE_DEVICES=2 python main.py tracking,seg --exp_id mots_hts_mtl_ignore --dataset kitti_mots --dataset_version train_part --pre_hm --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --batch_size 10 --num_epoch 140 --lr_step 80,100 --same_aug_pre --num_workers 4  --wh_weight 0 --off_weight 0 --mtl
CUDA_VISIBLE_DEVICES=2 python main.py tracking,seg --exp_id mots_cnp_pt17 --dataset kitti_mots --dataset_version train_part --pre_hm --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --batch_size 10 --num_epoch 60 --lr_step 20,40 --same_aug_pre --num_workers 4  --wh_weight 0 --off_weight 0 --mtl --load_model ../exp/tracking,seg/mots_pt_mot17/model_best.pth --paste_up
CUDA_VISIBLE_DEVICES=7 python main.py tracking,seg --exp_id mots_cnp_dacnp_v1.1_nb_ --dataset kitti_mots --dataset_version train_part --pre_hm --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --batch_size 6 --num_epoch 60 --lr 1.25e-5 --lr_step 30,50 --same_aug_pre --num_workers 4  --wh_weight 0 --off_weight 0 --mtl --load_model ../exp/tracking,seg/mots_dacnp_pt_coco_v1.1/model_last.pth --copy_and_paste 0.4 --paste_up
CUDA_VISIBLE_DEVICES=1 python main.py tracking,seg --exp_id mots_dacnp_v2 --dataset kitti_mots --dataset_version train_part --pre_hm --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --batch_size 8 --num_epoch 140 --lr_step 80,110 --same_aug_pre --num_workers 4  --wh_weight 0 --off_weight 0 --mtl --load_model ../models/coco_tracking.pth --copy_and_paste 0.5 --pre_paste 0.8
CUDA_VISIBLE_DEVICES=0 python main.py tracking,seg --exp_id mots_kmf_att_v0.11_fs --dataset kitti_mots --dataset_version train_part --pre_hm --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0  --num_epoch 140 --lr_step 80,110 --same_aug_pre --num_workers 4  --wh_weight 0 --off_weight 0 --mtl --kmf_att --guss_oval --kmf_append --kmf_pit --kmf_layer  --att_hm_disturb 0.01 --att_lost_disturb 0.2 --att_fp_disturb 0.1 --batch_size 16 --num_pre_imgs_input 2
CUDA_VISIBLE_DEVICES=2 python main.py tracking,seg --exp_id mots_sch_v1.0 --dataset kitti_mots --dataset_version train_part --gpus 0  --num_epoch 140 --lr_step 80,110 --same_aug_pre --num_workers 4  --wh_weight 0.1  --batch_size 12 --load_model ../models/ctdet_coco_dla_2x.pth --sch_track --hm_disturb 0.05  --no_pre_img --sch_feat_channel 16
