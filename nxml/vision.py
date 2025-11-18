"""

https://huggingface.co/spaces/eyepallavi/FaceSimilarity/tree/main/app/Hackathon_setup

PIP: joblib

"""

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
    def __init__(self):
        super().__init__('comfyui')

    def send_prompt_and_image_to_comfyui(self, prompt: str, image: any, server_url="http://127.0.0.1:8188") -> dict:
        """
        Sendet einen Prompt und ein Bild an ComfyUI und gibt die Server-Antwort zurück.

        :param prompt: Textprompt für das Modell
        :param image_path: Pfad zum Eingabebild
        :param server_url: URL des ComfyUI-Servers (Standard: lokal)
        :return: JSON-Antwort des Servers
        """
        with open(image, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {                    # Standard-Workflow für ComfyUI /prompt
            "prompt": {
                "6": {
                    "inputs": {
                        "text": prompt
                    },
                    "class_type": "CLIPTextEncode"
                },
                "54": {
                    "inputs": {
                        "image": image_b64
                    },
                    "class_type": "LoadImage"
                },
                # hier kannst du deine weiteren Node-IDs/Workflow-Nodes hinzufügen
            }
        }
        response = requests.post(f"{server_url}/prompt", json=payload)
        response.raise_for_status()
        return response.json()



__all__ = ['FaceSimilarity', 'ComfyUi']
