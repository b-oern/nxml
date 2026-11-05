#!/bin/sh
apt update
apt install -y python3 python3-pip wget git curl

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# python liegt auf python3

pip install numpy nwebclient transformers requests nltk flask
