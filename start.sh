#!/bin/sh
pip install numpy nwebclient transformers requests nltk flair detoxify textblob textblob_de langchain sentence_transformers
python3 -m nwebclient.runner --rest --executor nxml.analyse:AnalyseMain
