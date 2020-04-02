# MODEL ZOO

### Common settings and notes

- The experiments are run with PyTorch 1.0, CUDA 10.0, and CUDNN 7.5.
- Training times are measured on our servers with TITAN V GPUs (12 GB Memory).
- Testing times are measured on our local machine with TITAN Xp GPU. 
- The models can be downloaded directly from [Google drive](https://drive.google.com/open?id=1u4n_WwvDOJz4ws_KKQUMCpHXyvA6tj-I).

## 2D bounding box Tracking

### MOT17

| Model                 | GPUs |Train time| Test time | Valication MOTA  | Test MOTA  | Download | 
|-----------------------|------|----------|-----------|------------------|------------|----------|
| [mot17_fulltrain](../experiments/mot17_fulltrain.sh)       |    4 |    4h    | 45ms      |          -       |67.3 (Private Detection)|  [model](https://drive.google.com/open?id=1h_8Ts11rf0GQ4_n6FgmCeBuFcWrRjJfa)     |
| [mot17_fulltrain_sc](../experiments/mot17_fulltrain_sc.sh)    |    4 |    4h    | 45ms      |          -       |61.4 (Public Detection) | [model](https://drive.google.com/open?id=1WXBlzHsxHQTELvusJSgEWw_wydC6u7XB)  |
| [mot17_half](../experiments/mot17_half.sh)            |    4 |    2h    | 45ms      |         66.1     |      -     | [model](https://drive.google.com/open?id=1sf1bWJ1LutwQ_wp176nd2Y3HII9WeFf0) |
| [mot17_half_sc](../experiments/mot17_half_sc.sh)         |    4 |    2h    | 45ms      |         60.7     |      -     | [model](https://drive.google.com/open?id=12xnXeY-kW3otNjCoQtyJAayHFiQdTTAU) |
| [crowdhuman](../experiments/crowdhuman.sh)            |    4 |    21h   | 45ms      |         52.2     |      -     |[model](https://drive.google.com/open?id=1rIVl-jSG6oiBdiJmCvIAUOeasT7YllRZ) |

#### Notes

- `*_half` corresponds to the half-half video train/ val split mentioned in the paper. 
- `*_fulltrain` corresponds to train on the full training set, and evaluate on the official test server. These models are provided for arXiv and demo purposes. It is highly NOT recommended to submit our predictions to the test server, for not abusing the test set. Usually the validation results are all you need for developing.
- `mot17_half`/ `mot17_fulltrain` are finetuned on the `crowdhuman` model, and `mot17_half_sc`/ `mot17_fulltrain_sc` are trained from ImageNet initialization.
- The validation results are both using private detection.
- All the MOT models are trained for 70 epochs, with learning rate dropped at the 60th epoch. 
- The crowdhuman model is trained on CrowdHuman dataset with the "training on static image data" technic in our paper, and evaluate directly in MOT17 validation set. The crowdhuman pretraining uses 140 epochs, with learning rate dropped at 90 and 140 epochs.
- The training schedules are not well studies.
- We observe about 1 MOTA random noise for MOT models. 
- If the resulting MOTA of your self-trained model is not desired, playing around with the `--track_thresh` and `--pre_thresh` sometimes gives a better number (See Appendix H of the paper). 
- The MOT models, even trained on the full training set, still does not look great for in-the-wild videos. The crowdhuman model is a better choice for real world application. However, be aware that both datasets are in non-commercial licenses.


### KITTI 2D Tracking

| Model                 |GPUs| Train time| Test time | Validation MOTA  | Test MOTA  | Download  |
|-----------------------|----|-----------|-----------|------------------|------------|-----------|
| [kitti_fulltrain](../experiments/kitti_fulltrain.sh) (flip)| 2  |     9h    |      66   |               -  |   89.44    | [model](https://drive.google.com/open?id=1kBX4AgQj7R7HvgMdbgBcwvIac-IFp95h)  |
| [kitti_half](../experiments/kitti_half.sh)            | 2  |     4.5h  |      40   |             88.7 |   -        | [model](https://drive.google.com/open?id=1_VtGal9UzZE3n3QcVa0brZ7nNAwqPzd-)  |
| [kitti_half_sc](../experiments/kitti_half_sc.sh)         | 2  |     4.5h  |      40   |             84.5 |   -        | [model](https://drive.google.com/open?id=1Kv8kA7VLBqVst1ZcfB9gRH8TWs5oPN_h)|

#### Notes

- We use flip-test for the model we submitted to the test server (kitti_fulltrain_flip).
- `kitti_fulltrain` are finetuned on the nuScenes_3Ddetection_e140 model (see below). 
- All the models are trained for 70 epochs.
- We observe up to 1.5 MOTA jittering due to randomness. The results are reported for the best model.

## Monocular 3D Detection/ Tracking

### nuScenes

| Model                    | GPUs |Train time| Test time | Val AMOTA@0.2 | Val AMOTA | Val mAP |  Download | 
|--------------------------|------|----------|-----------|---------------|-----------|---------|-----------|
| [nuScenes_3Ddetection_e140](../experiments/nuScenes_3Ddetection_e140.sh)| 8    |     72h  |    28ms   |        -      |   -       |  30.27  | [model](https://drive.google.com/open?id=1ZSG9swryMEfBJ104WH8CP7kcypCobFlU) |
| [nuScenes_3Dtracking](../experiments/nuScenes_3Dtracking.sh)      | 8    |     40h  |    28ms   |       28.3    |  6.8      |  -      | [model](https://drive.google.com/open?id=1e8zR1m1QMJne-Tjp-2iY_o81hn2CiQRt) |

#### Notes

- Both models are trained on our DGX servers with 8x 32G V100 GPUs.
- The 3D detection model is trained on all 6 camera images of the keyframes for 140 epochs. It does not include attributes and velocity prediction and is different from the model we used in the 3D detection leaderboard. See the CenterNet repo for details about the full 3D detection model we used for test set evaluation.
- The 3D tracking model is finetuned on the 3D detection model for 70 epochs. 
- Training on 4 GPUs or 8x 12G GPUs with smaller batchsize is OK, if the [linear learning rate rule](https://arxiv.org/abs/1706.02677) is applied.

## COCO Tracking (for demo purpose only)

| Model                 |GPUs| Train time| Test time | Download  |
|-----------------------|----|-----------|-----------|-----------|
| [coco_tracking](../experiments/coco_tracking.sh)         | 8  |     39h   |    30ms   | [model](https://drive.google.com/open?id=1tJCEJmdtYIh8VuN8CClGNws3YO7QGd40)   |
| [coco_pose_tracking](../experiments/coco_pose_tracking.sh)    | 8  |     19h   |    33ms   | [model](https://drive.google.com/open?id=1H0YvFYCOIZ06EzAkC2NxECNQGXxK27hH)|

- Both models are trained with the "training on static image data" technic in our paper.
- The models are not evaluated on any benchmarks since there are no suitable ones in this setting. We provide them for demo purpose only.