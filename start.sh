#!/bin/sh
pip install numpy nwebclient transformers requests nltk flair detoxify textblob textblob_de langchain sentence_transformers pillow

apt update
apt install -y wget git curl libjpeg-dev libpng-dev python3-dev cmake gcc libx11-dev
git clone https://github.com/b-oern/age-and-gender.git
cd age-and-gender
pip install .
cd ..

python3 -m nwebclient.runner --rest --executor nxml.analyse:AnalyseMain
