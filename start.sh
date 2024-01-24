#!/bin/sh
pip install numpy nwebclient transformers requests nltk flair
python3 -m nwebclient.runner --rest --executor nxml.analyse:NlpPipeline
