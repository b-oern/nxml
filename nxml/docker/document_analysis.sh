#!/bin/sh
git clone https://github.com/b-oern/nxml.git
pip install /nxml
git clone https://github.com/THU-MIG/yolov10.git
echo "#############################################"
echo "Installing yolov10"
cd yolov10
pip install -r requirements.txt
pwd
pip install /yolov10
echo "#############################################"
wget https://huggingface.co/spaces/omoured/YOLOv10-Document-Layout-Analysis/resolve/main/models/yolov10x_best.pt


#python3 -m nwebclient.runner --rest --cfg --executor nxml.image:DocumentAnalysis