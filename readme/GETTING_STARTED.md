# Getting Started

This document provides tutorials to train and evaluate CenterTrack. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Benchmark evaluation

First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in `CenterTrack_ROOT/models/`. 

### MOT17

To test the tracking performance on MOT17 with our pretrained model, run

~~~
 python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/mot17_half.pth
~~~

This will give a MOTA of `66.1` if set up correctly. `--pre_hm` is to enable the input heatmap. `--ltrb_amodal` is to use the left, top, right, bottom bounding box representation to enable detecting out-of-image bounding box (We observed this is important for MOT datasets). And `--track_thresh` and `--pre_thresh` are the score threshold for predicting a bounding box ($\theta$ in the paper) and feeding the heatmap to the next frame ($\tau$ in the paper), respectively.

To test with public detection, run

~~~
 python test.py tracking --exp_id mot17_half_public --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/mot17_half.pth --public_det --load_results ../data/mot17/results/val_half_det.json
~~~

The expected MOTA is `63.1`.

To test on the test set, run

~~~
 python test.py tracking --exp_id mot17_fulltrain_public --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/mot17_fulltrain_sc.pth --public_det --load_results ../data/mot17/results/test_det.json
~~~

The Test set evaluation requires submitting to the official test server.
We discourage the users to submit our predictions to the test set to prevent test set abuse.
You can append `--debug 2` to above commends to visualize the predictions.

See the experiments folder for testing in other settings.


### KITTI Tracking

Run:

~~~
python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --load_model ../models/kitti_half.pth
~~~

The expected MOTA is `88.7`.

### nuScenes

Run:

~~~
python test.py tracking,ddd --exp_id nuScenes_3Dtracking --load_model ../models/nuScenes_3Dtracking.pth --dataset nuscenes --track_thresh 0.1 --pre_hm
~~~

The expected AMOTA is `6.8`.

## Training
We have packed all the training scripts in the [experiments](../experiments) folder.
The experiment names correspond to the model name in the [model zoo](MODEL_ZOO.md).
The number of GPUs for each experiment can be found in the scripts and the model zoo.
If the training is terminated before finishing, you can use the same command with `--resume` to resume training. It will found the latest model with the same `exp_id`.
Some experiments rely on pretraining on another model. In this case, download the pretrained model from our model zoo or train that model first.