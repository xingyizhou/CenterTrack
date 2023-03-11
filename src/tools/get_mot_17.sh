mkdir ../../data
cd ../../data
mkdir ./mot17
cd ./mot17
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
rm MOT17.zip
cd ./MOT17
mv * ..
cd ..
rm -r MOT17 
mkdir annotations
cd ../../src/tools/
python convert_mot_to_coco.py
python convert_mot_det_to_results.py