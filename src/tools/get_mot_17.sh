mkdir ../../data
cd ../../data
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
mv MOT17 mot17
rm MOT17.zip
cd mot17
cd test
mkdir annotations
cd ../../../src/tools/
python convert_mot_to_coco.py
python convert_mot_det_to_results.py
