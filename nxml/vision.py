"""

https://huggingface.co/spaces/eyepallavi/FaceSimilarity/tree/main/app/Hackathon_setup

PIP: joblib

"""
import json
import time

RESSOURCES = {
    'clf_model.joblib': 'https://pi.bsnx.net/ki-models/facesimilarity/clf_model.joblib',
    'exp_recognition_net.t7': 'https://pi.bsnx.net/ki-models/facesimilarity/exp_recognition_net.t7',
    'expression_model.t7': 'https://pi.bsnx.net/ki-models/facesimilarity/expression_model.t7',
    'siamese_model.t7': 'https://pi.bsnx.net/ki-models/facesimilarity/siamese_model.t7',
    'haarcascade_frontalface_default.xml': 'https://pi.bsnx.net/ki-models/facesimilarity/haarcascade_frontalface_default.xml',
    'haarcascade_eye.xml': 'https://pi.bsnx.net/ki-models/facesimilarity/haarcascade_eye.xml'
}

from nwebclient import runner as r, base
from nwebclient.runner import TAG
from nwebclient import util as u
from nwebclient import dev as d
from nwebclient import web as w
from nwebclient import base as b

import base64
import requests

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from PIL import Image
import base64

current_path = './'

trnscm = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])


class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        # YOUR CODE HERE
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            # Pads the input tensor using the reflection of the input boundary, it similar to the padding.
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


##########################################################################################################
## Sample classification network (Specify if you are using a pytorch classifier during the training)    ##
## classifier = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear...)           ##
##########################################################################################################

# YOUR CODE HERE for pytorch classifier
num_of_classes = 6
classifier = nn.Sequential(nn.Linear(256, 64),
                           nn.BatchNorm1d(64),
                           nn.ReLU(),
                           nn.Linear(64, 32),
                           nn.BatchNorm1d(32),
                           nn.ReLU(),
                           nn.Linear(32, num_of_classes))

# Definition of classes as dictionary
classes = ['person1', 'person2', 'person3', 'person4', 'person5', 'person6']


def detected_face(image):
    eye_haar = current_path + '/haarcascade_eye.xml'
    face_haar = current_path + '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_haar)
    eye_cascade = cv2.CascadeClassifier(eye_haar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = []
    images = []
    required_image = 0
    for i, (x, y, w, h) in enumerate(faces):
        face_cropped = gray[y:y + h, x:x + w]
        face_areas.append(w * h)
        images.append(face_cropped)
        required_image = images[np.argmax(face_areas)]
        required_image = Image.fromarray(required_image)
    return required_image


# 1) Images captured from mobile is passed as parameter to the below function in the API call. It returns the similarity measure between given images.
# 2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
# 3) Define an object to your siamese network here in the function and load the weight from the trained network, set it in evaluation mode.
# 4) Get the features for both the faces from the network and return the similarity measure, Euclidean,cosine etc can be it. But choose the Relevant measure.
# 5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function
# Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_similarity(img1, img2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    det_img1 = detected_face(img1)
    det_img2 = detected_face(img2)
    if det_img1 == 0 or det_img2 == 0:
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        det_img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    face1 = trnscm(det_img1).unsqueeze(0)
    face2 = trnscm(det_img2).unsqueeze(0)

    print("Image CV")
    ##########################################################################################
    ##Example for loading a model using weight state dictionary:                            ##
    ## feature_net = light_cnn() #Example Network                                           ##
    ## model = torch.load(current_path + '/siamese_model.t7', map_location=device)          ##
    ## feature_net.load_state_dict(model['net_dict'])                                       ##
    ##                                                                                      ##
    ##current_path + '/<network_definition>' is path of the saved model if present in       ##
    ##the same path as this file, we recommend to put in the same directory                 ##
    ##########################################################################################
    ##########################################################################################

    feature_net = Siamese()
    ckpt = torch.load(current_path + '/siamese_model.t7', map_location=device)
    # model_path = current_path + "/Hackathon-setup/siamese_model.t7"
    feature_net.load_state_dict(ckpt['net_dict'])
    # model.eval()

    with torch.no_grad():
        output1, output2 = feature_net(face1.to(device), face2.to(device))
        # Calculate similarity measure - for instance, using cosine similarity
        euclidean_distance = F.pairwise_distance(output1, output2)

    return euclidean_distance.item()


def get_face_class(img1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    det_img1 = detected_face(img1)
    if det_img1 == 0:
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    face1 = trnscm(det_img1).unsqueeze(0)

    feature_net = Siamese().to(device)
    model = torch.load(current_path + '/siamese_model.t7', map_location=device)  ##
    feature_net.load_state_dict(model['net_dict'])
    feature_net.eval()

    output1, output2 = feature_net(face1.to(device), face1.to(device))
    output1 = output1.detach().numpy()

    clf_model = load(current_path + '/clf_model.joblib')
    label = clf_model.predict(output1)
    return label


class FaceSimilarity(r.ImageExecutor):

    TAGS = [TAG.IMAGE, '2-images']

    type = 'face_similarity'

    def __init__(self):
        super().__init__()
        u.download_resources('./', RESSOURCES)

    def to_numpy(self, img):
        return np.array(img).reshape(img.size[1], img.size[0], 3).astype(np.uint8)

    def execute(self, data):
        a = self.to_numpy(self.get_image('a', data))
        b = self.to_numpy(self.get_image('b', data))
        dist = get_similarity(a, b)
        return self.success('ok', distance=dist, value=dist)

    def part_index(self, p: base.Page, params={}):
        p.input('a_id', id='a_id')
        p.input('b_id', id='b_id')
        p(self.action_btn_parametric("Compare", {'type':self.type, 'a_id': '#a_id', 'b_id': '#b_id'}))
        p.pre('', id='result')



class ComfyUi(r.BaseJobExecutor):
    """
      comfyui: nxml.vision:ComfyUi
    """
    def __init__(self, server_url="http://127.0.0.1:8188", args: u.Args = {}):
        super().__init__('comfyui')
        self.cfg = args.get(self.type, {})
        self.server_url = server_url
        self.define_vars('server_url')
        self.define_sig(d.PStr('prompt', ''), d.PStr('image', ''),
                        d.PStr('workflow', ''))

    def merge(self, a: dict, b: dict, path=[]):
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    self.merge(a[key], b[key], path + [str(key)])
                elif a[key] != b[key]:
                    #raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    def send_prompt_and_image_to_comfyui(self, prompt: str, image: any, json_file='workflow.json', server_url="http://127.0.0.1:8188") -> dict:
        """
        Sendet einen Prompt und ein Bild an ComfyUI und gibt die Server-Antwort zurück.

        :param prompt: Textprompt für das Modell
        :param image: Pfad zum Eingabebild, wird vom ComfyUI-Server geladen
        :param json_file:  Prompt -> File > Export (API)
        :param server_url: URL des ComfyUI-Servers (Standard: lokal)
        :return: JSON-Antwort des Servers {prompt_id, number, node_errors}
        """
        if server_url is None:
            server_url = self.server_url
        #with open(image, "rb") as f:
        #    image_bytes = f.read()
        #image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        b = {
                "6": {
                    "inputs": {
                        "text": prompt
                    },
                    "class_type": "CLIPTextEncode"
                },
                "54": {
                    "inputs": {
                        "image": image
                    },
                    "class_type": "LoadImage"
                }
            # 28 input filename_prefix    SaveAnimateWEBP
        }
        with open(json_file, 'r') as f:
            a = json.load(f)
        workflow = self.merge(a, b)
        payload = {"prompt": workflow}  # Standard-Workflow für ComfyUI /prompt
        response = requests.post(f"{server_url}/prompt", json=payload)
        response.raise_for_status()
        return response.json()

    def inject_prompt(self, prompt, workflow):
        for k in workflow.keys():
            if (workflow[k].get('class_type', '') == 'CLIPTextEncode' and
                    "Positive Prompt" in workflow[k].get('_meta', {}).get('title', "")):
                return {
                   k: {
                       "inputs": {
                            "text": prompt
                       }
                   }
                }

    def send_prompt_to_comfyui(self, prompt: any, json_file='workflow.json',
                                         server_url="http://127.0.0.1:8188", data={}) -> dict:
        if server_url is None:
            server_url = self.server_url
        with open(json_file, 'r') as f:
            a = json.load(f)
        if isinstance(prompt, str):
            prompt = self.inject_prompt(prompt, a)
        workflow = self.merge(a, prompt)
        payload = {"prompt": workflow}  # Standard-Workflow für ComfyUI /prompt
        response = requests.post(f"{server_url}/prompt", json=payload)
        response.raise_for_status()
        if data.get('count', 1) > 1:
            for i in range(data.get('count')):
                response = requests.post(f"{server_url}/prompt", json=payload)
                time.sleep(3)
        return response.json()

    def history(self):
        response = requests.get(self.server_url + '/history')
        # object(guid: {"prompt": ..., "outputs": ..., status: { completed}})
        response.raise_for_status()
        return response.json()

    def queue(self):
        response = requests.get(self.server_url + '/queue')
        # { queue_running: [...], queue_pending: [ [nr,guid, prompt-obj, [int,int,int]] ] }
        response.raise_for_status()
        return response.json()

    def system_stats(self):
        response = requests.get(self.server_url + '/queue')
        # { system: {os:, ram_total:}, "devices": [{name:, vram_total:, vram_free},..] }
        response.raise_for_status()
        return response.json()

    def free(self):
        response = requests.post(self.server_url + '/free')
        # { system: {os:, ram_total:}, "devices": [{name:, vram_total:, vram_free},..] }
        response.raise_for_status()
        return response.json()

    def execute_rest(self, data):
        routes = ['history', 'queue', 'system_stats', 'free']
        if data['route'] in routes:
            op = getattr(self, data['route'])
            return op()
        else:
            return self.fail('route not in routes')

    def execute_queue_count(self):
        j = self.queue()
        return self.success(value=len(j.get('queue_pending', [])))

    def execute(self, data):
        if 'prompt' in data and 'image' in data:
            return self.send_prompt_and_image_to_comfyui(data['prompt'], data['image'], data['workflow'])
        elif 'prompt' in data:
            return self.send_prompt_to_comfyui(data['prompt'], data['workflow'], data=data)
        return super().execute(data)

    def part_index(self, p: base.Page, params={}):
        p.div("Start ComfyUi Jobs")
        p.div("Workflows")
        p.ul(self.cfg.get('workflows', []))
        p(self.action_btn(dict(title="Stats", type=self.type, op='rest', route='system_stats')))
        p(self.action_btn(dict(title="Queue", type=self.type, op='rest', route='queue')))
        p(self.action_btn(dict(title="Queue Count", type=self.type, op='queue_count')))
        p.ul(w.a("Prompt", self.link(self.part_prompt)))
        p.pre('', id='result')

    def part_prompt(self, p: base.Page, params={}):
        p('<textarea id="prompt"></textarea>')
        p.input('workflow', value='/mnt/d/ai/z_image.json', id='workflow')
        p(self.action_btn_parametric("Execute", dict(prompt='#prompt', type=self.type, workflow='#workflow')))



__all__ = ['FaceSimilarity', 'ComfyUi']
