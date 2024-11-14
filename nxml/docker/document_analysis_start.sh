#!/bin/sh
cd /yolov10
pip install .
python3 -m nwebclient.runner --rest --executor nxml.image:DocumentAnalysis