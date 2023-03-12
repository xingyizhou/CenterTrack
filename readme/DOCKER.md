# Run CenterTrack using Docker

### DCNv2 Compilation Issue Fixed
```
docker pull bryanbocao/centertrack
docker run -d --ipc=host --shm-size=16384m -it -v /:/share --gpus all --network=bridge bryanbocao/centertrack /bin/bash
docker ps -a
```

Check ```<CONTAINER_ID>```:
```
CONTAINER ID   IMAGE                    COMMAND                  CREATED          STATUS                       PORTS                NAMES
89bb79551ccb   bryanbocao/centertrack   "/usr/local/bin/nvid…"   49 seconds ago   Up 38 seconds                6006/tcp, 8888/tcp   competent_northcutt
```
```
docker exec -it <CONTAINE_ID> /bin/bash
```

```
docker exec -it 89bb79551ccb /bin/bash
```

Inside the container:
```
cd /root/CenterTrack/src/lib/model/networks/DCNv2
python3 setup.py build develop
cd /root/CenterTrack/src/
python3 demo.py tracking,ddd --load_model ../models/nuScenes_3Dtracking.pth --dataset nuscenes --pre_hm --track_thresh 0.1 --demo ../videos/nuscenes_mini.mp4 --test_focal_length 633
```
The DCNv2 is built from [MatthewHowe_DCNv2](https://github.com/MatthewHowe/DCNv2). Thanks @MatthewHowe for the contribution.
Then you should be able to see the results.

Go to ```/share``` to transfer data between the container and your root system.
