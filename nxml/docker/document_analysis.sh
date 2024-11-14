#!/bin/sh
chmod +x /document_analysis_start.sh
git clone https://github.com/b-oern/nxml.git
pip install /nxml
git clone https://github.com/THU-MIG/yolov10.git
echo "#############################################"
echo "Installing yolov10"
cd yolov10
python3 -m pip install -r requirements.txt
pwd
python3 -m pip install .
echo "#############################################"
echo "Importing ultralytics"
python3 -c "from ultralytics import YOLO"
echo $?
echo "#############################################"

#wget https://huggingface.co/spaces/omoured/YOLOv10-Document-Layout-Analysis/resolve/main/models/yolov10x_best.pt


#python3 -m nwebclient.runner --rest --cfg --executor nxml.image:DocumentAnalysis