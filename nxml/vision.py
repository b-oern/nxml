"""

https://huggingface.co/spaces/eyepallavi/FaceSimilarity/tree/main/app/Hackathon_setup

PIP: joblib

"""
import json
import time
import html
import re
import glob
import base64
import requests
import random
import os.path

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
        self.define_sig(d.Param('a'), d.Param('b'))\
            .set_description("Distanz zweier Gesichter (Siamese-Netz), kleiner ist ähnlicher")\
            .r_success(distance='float', value='float')

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
      comfyui:
        workflows:
            - path/w.json
        workflow_dir: /path/to/workflows/
        jobpath: /path/to/jobs/
        comfyui_path: /mnt/l/ComfyUI_windows_portable/ComfyUI

      Workflows muessen im API-Format vorliegen (ComfyUI: Workflow > Export (API)). Angeboten werden die
      workflows und alle *.json aus workflow_dir, Uploads landen in workflow_dir.

      Eingaben werden ueber die Titel der Nodes gefunden (vorbereiten mit part_prepare):
        - Prompt: Node mit Text-Eingang (text/prompt), dessen Titel "Positive Prompt" enthaelt
        - Bilder: LoadImage-Nodes, der Titel ist der Name beim Aufruf (file.<Titel> oder file_title/file_data),
                  die Bilder werden nach comfyui_path/input geschrieben
    """

    PROMPT_TITLE = 'Positive Prompt'
    PROMPT_KEYS = ('text', 'prompt')
    IMAGE_CLASSES = ('LoadImage',)
    OUTPUT_PREFIXES = ('Save', 'Preview', 'VHS_VideoCombine')
    SEED_KEYS = ('seed', 'noise_seed')

    def __init__(self, server_url="http://127.0.0.1:8188", args: u.Args = {}):
        super().__init__('comfyui')
        self.cfg = args.get(self.type, {})
        self.server_url = server_url
        if 'server_url' in self.cfg:
            self.server_url = self.cfg['server_url'].strip()
        self.wait_time = 4
        self.define_vars('server_url', 'wait_time')
        self.define_sig(d.Param('prompt'), d.Param('workflow'), d.PInt('count', 1), d.PStr('random_seed', '0'))\
            .set_description("Startet einen Workflow (API-Export) mit Prompt, Bilder als file.<Titel> (Base64)")\
            .r(prompt_id='str', number='int', node_errors='dict')
        self.define_sig(d.Param('prompt'), d.Param('image'), d.Param('workflow'))\
            .set_description("Startet einen Workflow mit Prompt (Node 6) und Bild-Dateiname (Node 54)")\
            .r(prompt_id='str', number='int', node_errors='dict')
        self.define_sig(d.PStr('op', 'queue'), d.Param('prompt'), d.Param('workflow'))\
            .set_description("Legt den Job als .job.json im jobpath ab").r_success(job_id='str')
        self.define_sig(d.PStr('op', 'queue_count')).set_description("Anzahl wartender Jobs in der ComfyUI-Queue")\
            .r_success(value='int')
        self.define_sig(d.PStr('op', 'rest'), d.PStr('route', 'system_stats'))\
            .set_description("Ruft die ComfyUI-API auf (history, queue, system_stats, free), liefert deren JSON")
        f = self.define_sig(d.PStr('op', 'workflow_info'), d.Param('workflow'))\
            .set_description("Prompt-, Bild-, Seed- und Ausgabe-Nodes eines Workflows und was fuer die Ausfuehrung fehlt")
        f.r_success(api_format='bool', ready='bool', prompt='dict', text_nodes='list', images='list', seeds='list',
                    outputs='list', warnings='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'workflow_prepare'), d.Param('workflow'), d.PStr('prompt_node', ''),
                            d.PStr('save_as', ''))\
            .set_description("Markiert den Prompt-Node, setzt Titel (title.<Node-Id>) und speichert den Workflow")
        f.r_success(workflow='str', ready='bool', warnings='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'workflow_upload'), d.Param('name'), d.Param('file_data'))\
            .set_description("Speichert einen Workflow (Base64-JSON) in workflow_dir")
        f.r_success(workflow='str')
        f.r_fail()

    def merge(self, a: dict, b: dict, path=[]):
        try:
            for key in b:
                if key in a:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        self.merge(a[key], b[key], path + [str(key)])
                    elif a[key] != b[key]:
                        #raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
                        a[key] = b[key]
                else:
                    a[key] = b[key]
        except Exception as e:
            self.info("Merge fail: " + str(e))
        return a

    def workflows(self) -> list:
        """ Workflows aus der Konfiguration und *.json aus workflow_dir """
        res = list(self.cfg.get('workflows', []))
        wf_dir = self.cfg.get('workflow_dir')
        if wf_dir:
            for file in sorted(glob.glob(os.path.join(wf_dir, '*.json'))):
                if file not in res:
                    res.append(file)
        return res

    def load_workflow(self, json_file) -> dict:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_workflow(self, json_file, workflow: dict):
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        if json_file not in self.workflows():
            self.cfg.setdefault('workflows', []).append(json_file)

    @staticmethod
    def is_api_format(workflow) -> bool:
        """ API-Export: {Node-Id: {class_type, inputs}}, der normale Export hat stattdessen nodes und links """
        return isinstance(workflow, dict) and len(workflow) > 0 and \
            all(isinstance(node, dict) and 'class_type' in node for node in workflow.values())

    @staticmethod
    def node_title(node) -> str:
        return node.get('_meta', {}).get('title', '') if isinstance(node, dict) else ''

    def prompt_key(self, node):
        """ Name des Text-Eingangs (text bzw. prompt) eines Nodes oder None """
        inputs = node.get('inputs', {}) if isinstance(node, dict) else {}
        for key in self.PROMPT_KEYS:
            if isinstance(inputs.get(key), str):
                return key
        return None

    def find_prompt_node(self, workflow):
        """ (Node-Id, Eingang) des ersten Text-Nodes mit "Positive Prompt" im Titel oder None """
        for k, node in workflow.items():
            key = self.prompt_key(node)
            if key is not None and self.PROMPT_TITLE in self.node_title(node):
                return k, key
        return None

    def seed_inputs(self, node) -> list:
        """ Seed-Eingaenge mit festem Wert (Listen sind Verbindungen zu anderen Nodes) """
        inputs = node.get('inputs', {})
        return [key for key in self.SEED_KEYS if key in inputs and not isinstance(inputs[key], list)]

    def analyze_workflow(self, workflow: dict) -> dict:
        """
        Eingaenge und Ausgaben eines Workflows, so wie dieser Runner sie findet
        :return: {api_format, ready, prompt, text_nodes, images, seeds, outputs, warnings}
        """
        res = dict(api_format=self.is_api_format(workflow), ready=False, prompt=None, text_nodes=[], images=[],
                   seeds=[], outputs=[], warnings=[])
        if not res['api_format']:
            res['warnings'].append("Kein API-Format: in ComfyUI über Workflow > Export (API) speichern")
            return res
        for k, node in workflow.items():
            cls = node.get('class_type', '')
            item = dict(node=k, title=self.node_title(node), class_type=cls)
            if self.prompt_key(node) is not None:
                res['text_nodes'].append({**item, 'key': self.prompt_key(node)})
            if cls in self.IMAGE_CLASSES:
                res['images'].append(item)
            if len(self.seed_inputs(node)) > 0:
                res['seeds'].append({**item, 'keys': self.seed_inputs(node)})
            if cls.startswith(self.OUTPUT_PREFIXES):
                res['outputs'].append(item)
        found = self.find_prompt_node(workflow)
        if found is None:
            res['warnings'].append(f'Kein Prompt-Node: einen Text-Node als "{self.PROMPT_TITLE}" markieren')
        else:
            res['prompt'] = dict(node=found[0], key=found[1], title=self.node_title(workflow[found[0]]))
            if sum(self.PROMPT_TITLE in t['title'] for t in res['text_nodes']) > 1:
                res['warnings'].append(f'Mehrere Nodes mit "{self.PROMPT_TITLE}" im Titel, verwendet wird {found[0]}')
        titles = [img['title'] for img in res['images']]
        if len(set(titles)) < len(titles):
            res['warnings'].append("Bild-Nodes brauchen eindeutige Titel, sonst erhalten alle dasselbe Bild")
        if len(titles) > 0 and not self.cfg.get('comfyui_path'):
            res['warnings'].append("comfyui_path fehlt in der Konfiguration, Bilder können nicht übergeben werden")
        if len(res['outputs']) == 0:
            res['warnings'].append("Kein Ausgabe-Node (z.B. SaveImage) gefunden")
        res['ready'] = len(res['warnings']) == 0
        return res

    def prepare_workflow(self, workflow: dict, prompt_node=None, titles: dict = None) -> dict:
        """
        Setzt Titel und markiert den Prompt-Node mit "Positive Prompt" (bei anderen Nodes wird es entfernt)
        :param titles: {Node-Id: Titel}, z.B. fuer LoadImage-Nodes
        """
        for k, title in (titles or {}).items():
            if k in workflow:
                workflow[k].setdefault('_meta', {})['title'] = title
        if prompt_node is not None and prompt_node in workflow:
            for k, node in workflow.items():
                title = self.node_title(node)
                if k != prompt_node and self.PROMPT_TITLE in title:
                    node['_meta']['title'] = title.replace(self.PROMPT_TITLE, 'Prompt')
            if self.PROMPT_TITLE not in self.node_title(workflow[prompt_node]):
                workflow[prompt_node].setdefault('_meta', {})['title'] = self.PROMPT_TITLE
        return workflow

    def send_prompt_and_image_to_comfyui(self, prompt: str, image: any, json_file='workflow.json', server_url=None) -> dict:
        """
        Sendet einen Prompt und ein Bild an ComfyUI und gibt die Server-Antwort zurück.

        :param prompt: Textprompt für das Modell
        :param image: Pfad zum Eingabebild, wird vom ComfyUI-Server geladen
        :param json_file:  Prompt -> File > Export (API)
        :param server_url: URL des ComfyUI-Servers (Standard: self.server_url)
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
        a = self.load_workflow(json_file)
        self.info(f"Loading Workflow {json_file}")
        workflow = self.merge(a, b)
        payload = {"prompt": workflow}  # Standard-Workflow für ComfyUI /prompt
        response = requests.post(f"{server_url}/prompt", json=payload)
        response.raise_for_status()
        return response.json()

    def inject_prompt(self, prompt, workflow):
        found = self.find_prompt_node(workflow)
        if found is not None:
            k, key = found
            return {
                k: {
                    "inputs": {
                        key: prompt
                    }
                }
            }

    def inject_resolution(self, width, height, workflow):
        pass # TODO

    def inject_seed(self, workflow):
        for node in workflow.values():
            for key in self.seed_inputs(node):
                node['inputs'][key] = random.randint(0, 4200000000000000)
        return workflow

    def inject_file_by_title(self, workflow, title, file_data, ext='png', input_key='image', fileid=None):
        if fileid is None:
            fileid = u.guid()
        name = fileid + '.' + ext
        self.info(f"Injecting: {title} with {name}")
        with open(os.path.join(self.cfg['comfyui_path'], 'input', name), 'wb') as f:
            f.write(base64.b64decode(file_data))
        for k in workflow.keys():
            if workflow[k].get('_meta', {}).get('title', '') == title:
                self.info(f"Injection success: {title}")
                workflow[k]['inputs'][input_key] = name
        return workflow

    def inject_value_by_title(self, workflow, title, input_key, value):
        for k in workflow.keys():
            if workflow[k].get('_meta', {}).get('title', '') == title:
                workflow[k]['inputs'][input_key] = value
        return workflow

    def collect_files(self, data: dict) -> dict:
        """
        Fasst die Bilder eines Aufrufs in data['files'] = [{title, data}] zusammen:
        files (Liste oder JSON), file_title/file_data und file.<Titel>=Base64 (leere Werte werden ignoriert)
        """
        files = data.get('files', [])
        if isinstance(files, str):
            files = json.loads(files)
        files = {file['title']: file for file in files}
        if data.get('file_title') and data.get('file_data'):
            files[data['file_title']] = dict(title=data['file_title'], data=data['file_data'])
        for key in [k for k in data.keys() if k.startswith('file.')]:
            value = data.pop(key)
            if value:
                files[key[len('file.'):]] = dict(title=key[len('file.'):], data=value)
        if len(files) > 0:
            data['files'] = list(files.values())
        return data

    def send_prompt_to_comfyui(self, prompt: any, json_file='workflow.json', server_url=None, data={}) -> dict:
        """
        :param data: files/file.<Titel> (siehe collect_files), values [{title, key, value}],
                     count (Anzahl Durchlaeufe), random_seed=1 (Seeds zufaellig setzen, bei count > 1 immer)
        """
        self.info(f"Sending prompt to comfyui, extras: " + str(data.keys()))
        if server_url is None:
            server_url = self.server_url
        a = self.load_workflow(json_file)
        if isinstance(prompt, str):
            prompt = self.inject_prompt(prompt, a)
        workflow = self.merge(a, prompt if prompt is not None else {})
        self.collect_files(data)
        for file in data.get('files', []):
            workflow = self.inject_file_by_title(workflow, file['title'], file['data'], data.get('ext', 'png'),
                                                 data.get('input_key', 'image'), data.get('fileid', None))
        values = data.get('values', [])
        if isinstance(values, str):
            values = json.loads(values)
        for elem in values:
            workflow = self.inject_value_by_title(workflow, elem['title'], elem['key'], elem['value'])
        count = max(int(data.get('count', 1)), 1)
        random_seed = count > 1 or str(data.get('random_seed', '0')) == '1'
        response = None
        for i in range(count):
            if random_seed:
                workflow = self.inject_seed(workflow)
            response = requests.post(f"{server_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            if i < count - 1:
                time.sleep(self.wait_time)
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
        response = requests.get(self.server_url + '/system_stats')
        # { system: {os:, ram_total:}, "devices": [{name:, vram_total:, vram_free},..] }
        response.raise_for_status()
        return response.json()

    def free(self):
        # gibt Modelle und Speicher frei
        response = requests.post(self.server_url + '/free', json={'unload_models': True, 'free_memory': True})
        response.raise_for_status()
        return {'success': True}

    def execute_rest(self, data):
        routes = ['history', 'queue', 'system_stats', 'free']
        if data['route'] in routes:
            op = getattr(self, data['route'])
            return op()
        else:
            return self.fail('route not in routes')

    def execute_queue_count(self, data={}):
        j = self.queue()
        return self.success(value=len(j.get('queue_pending', [])))

    def execute_queue(self, data={}):
        data.pop('op')
        path = self.cfg.get('jobpath', '.')
        guid = u.guid()
        data['guid'] = guid
        self.collect_files(data)
        with open(os.path.join(path, guid + '.job.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return self.success(job_id=guid)

    def execute_workflow_info(self, data):
        try:
            return self.success(**self.analyze_workflow(self.load_workflow(data['workflow'])))
        except (OSError, ValueError) as e:
            return self.fail(str(e))

    def execute_workflow_prepare(self, data):
        try:
            workflow = self.load_workflow(data['workflow'])
        except (OSError, ValueError) as e:
            return self.fail(str(e))
        titles = {k[len('title.'):]: v for k, v in data.items() if k.startswith('title.') and v}
        self.prepare_workflow(workflow, data.get('prompt_node') or None, titles)
        save_as = data.get('save_as') or data['workflow']
        self.save_workflow(save_as, workflow)
        return self.success(f"Gespeichert: {save_as}", workflow=save_as, **self.analyze_workflow(workflow))

    def execute_workflow_upload(self, data):
        try:
            workflow = json.loads(base64.b64decode(data['file_data']).decode('utf-8'))
        except (KeyError, ValueError) as e:
            return self.fail("Kein gültiges JSON: " + str(e))
        name = os.path.basename(data.get('name') or 'workflow.json')
        if not name.endswith('.json'):
            name += '.json'
        path = os.path.join(self.cfg.get('workflow_dir', '.'), name)
        self.save_workflow(path, workflow)
        return self.success(f"Gespeichert: {path}", workflow=path)

    def execute(self, data):
        if 'prompt' in data and 'image' in data and 'op' not in data:
            return self.send_prompt_and_image_to_comfyui(data['prompt'], data['image'], data['workflow'])
        elif 'prompt' in data and 'op' not in data:
            return self.send_prompt_to_comfyui(data['prompt'], data['workflow'], data=data)
        return super().execute(data)

    @staticmethod
    def html_id(prefix, node_id) -> str:
        """ Node-Ids koennen z.B. bei Subgraphen ':' enthalten, das ist in CSS-Selektoren nicht erlaubt """
        return prefix + '_' + re.sub(r'\W', '_', str(node_id))

    def link_workflow(self, page, wf) -> str:
        return self.link(page, w=wf)

    def js_open_workflow(self, page) -> str:
        """ on_success fuer exec_job_p: oeffnet page mit dem Workflow aus dem Ergebnis """
        return 'function(r) { if (r.success) location.href = "' + self.link(page) + 'w=" + encodeURIComponent(r.workflow); }'

    def part_index(self, p: base.Page, params={}):
        p.div("Startet ComfyUI-Workflows direkt oder als Job-Datei (jobpath). Workflows werden vorher einmal "
              "vorbereitet, damit Prompt und Bilder an der richtigen Stelle landen.")
        with p.section("Workflows"):
            self.part_workflow_table(p)
            p.ul([w.a("Workflow hochladen und vorbereiten", self.link(self.part_prepare)),
                  w.a("Workflow ausführen", self.link(self.part_execute))])
        with p.section("Server"):
            p.prop("Server", html.escape(self.server_url))
            p.prop("ComfyUI-Pfad", html.escape(self.cfg.get('comfyui_path', '- (nötig für Bilder)')))
            p.prop("Jobpfad", html.escape(self.cfg.get('jobpath', '.')))
            p.prop("Workflow-Ordner", html.escape(self.cfg.get('workflow_dir', '-')))
            p(self.action_btn(dict(title="Stats", type=self.type, op='rest', route='system_stats')))
            p(self.action_btn(dict(title="Queue", type=self.type, op='rest', route='queue')))
            p(self.action_btn(dict(title="Queue Count", type=self.type, op='queue_count')))
            p(self.action_btn(dict(title="Speicher freigeben", type=self.type, op='rest', route='free')))
        p.pre('', id='result')

    def part_workflow_table(self, p: base.Page):
        if len(self.workflows()) == 0:
            p.div("Keine Workflows konfiguriert (workflows oder workflow_dir).")
            return
        p('<table class="comfyui_workflows"><tr><th>Workflow</th><th>Prompt</th><th>Bilder</th><th>Status</th>'
          '<th></th></tr>')
        for wf in self.workflows():
            try:
                info = self.analyze_workflow(self.load_workflow(wf))
                prompt = 'ja' if info['prompt'] else '-'
                images = ', '.join(html.escape(img['title']) for img in info['images']) or '-'
                status = 'bereit' if info['ready'] else str(len(info['warnings'])) + ' Hinweis(e)'
            except (OSError, ValueError) as e:
                prompt, images, status = '', '', 'Fehler: ' + html.escape(str(e))
            actions = w.a("Ausführen", self.link_workflow(self.part_execute, wf)) + ' | ' + \
                w.a("Vorbereiten", self.link_workflow(self.part_prepare, wf))
            p(f'<tr><td title="{html.escape(wf)}">{html.escape(os.path.basename(wf))}</td><td>{prompt}</td>'
              f'<td>{images}</td><td>{status}</td><td>{actions}</td></tr>')
        p('</table>')

    def part_dropzone(self, p: base.Page, hid, title):
        """ Ablageflaeche, die die abgelegte Datei als Base64 in das Feld hid schreibt """
        p.dropzone(hid + '_drop', title)
        p.js_ready(f'nx_initFileDragArea("{hid}_drop", "{hid}");')

    def part_warnings(self, p: base.Page, info: dict):
        for msg in info['warnings']:
            p.alert_warn(html.escape(msg))
        if info['ready']:
            p.alert_success("Bereit für die Ausführung")

    def part_load(self, p: base.Page, params: dict):
        """ :return: (Pfad, Workflow, Analyse) aus params['w'] oder None, wenn er nicht geladen werden kann """
        wf = params.get('w')
        if not wf:
            return None
        try:
            workflow = self.load_workflow(wf)
        except (OSError, ValueError) as e:
            p.alert_error(f"{html.escape(wf)} kann nicht geladen werden: {html.escape(str(e))}")
            return None
        p.prop("Workflow", html.escape(wf))
        return wf, workflow, self.analyze_workflow(workflow)

    def part_prepare(self, p: base.Page, params={}):
        p.h2("Workflow vorbereiten")
        loaded = self.part_load(p, params)
        if loaded is None:
            self.part_prepare_upload(p)
            return
        wf, workflow, info = loaded
        self.part_warnings(p, info)
        if not info['api_format']:
            return
        job = dict(type=self.type, op='workflow_prepare', workflow=wf, prompt_node='#prompt_node', save_as='#save_as')
        with p.section("Prompt"):
            p.div(f'Der Prompt des Aufrufs wird in den Node geschrieben, dessen Titel "{self.PROMPT_TITLE}" enthält.')
            options = {'': '- keiner -'}
            for t in info['text_nodes']:
                text = workflow[t['node']]['inputs'][t['key']]
                options[t['node']] = f"{t['node']}: {t['title']} ({t['class_type']}) {text[:40]}"
            p.form_select('prompt_node', "Prompt-Node", options, info['prompt']['node'] if info['prompt'] else '',
                          id='prompt_node')
        with p.section("Bilder"):
            if len(info['images']) == 0:
                p.div("Keine LoadImage-Nodes, der Workflow braucht keine Bilder.")
            else:
                p.div("Der Titel ist der Name, unter dem das Bild beim Aufruf übergeben wird (file.&lt;Titel&gt;).")
            for img in info['images']:
                hid = self.html_id('title', img['node'])
                p.form_input('title.' + img['node'], f"Node {html.escape(img['node'])}", id=hid,
                             value=html.escape(img['title'], quote=True))
                job['title.' + img['node']] = '#' + hid
        with p.section("Seeds und Ausgaben"):
            p.div("Seeds (bei random_seed=1 oder count &gt; 1 zufällig): " +
                  (', '.join(html.escape(f"{s['title']} ({s['node']})") for s in info['seeds']) or '-'))
            p.div("Ausgaben: " + (', '.join(html.escape(f"{o['title']} ({o['class_type']})") for o in info['outputs'])
                                  or '-'))
        with p.expandable("Alle Nodes", expanded=False):
            self.part_nodes(p, workflow)
        with p.section("Speichern"):
            p.form_input('save_as', "Speichern als", id='save_as', value=html.escape(wf, quote=True),
                         style='width: 100%; max-width: 600px;')
            p.form_elem(self.action_btn_parametric("Speichern", job, self.js_open_workflow(self.part_prepare)) + ' ' +
                        w.a("Ausführen", self.link_workflow(self.part_execute, wf)))
        with p.section("Aufruf"):
            p.pre(json.dumps(self.example_job(wf, info), indent=2))
        p.pre('', id='result')

    def part_prepare_upload(self, p: base.Page):
        p.ul(["In ComfyUI den Workflow über Workflow &gt; Export (API) speichern",
              "Hier hochladen oder einen vorhandenen Workflow wählen",
              "Prompt-Node und Bild-Titel festlegen und speichern"])
        with p.section("Hochladen"):
            p.hidden('file_data', '', 'wf_data')
            self.part_dropzone(p, 'wf_data', "Workflow (API-JSON) hier ablegen")
            p.form_input('name', "Dateiname", id='wf_name', value='workflow.json')
            p.prop("Ziel", html.escape(self.cfg.get('workflow_dir', '.')))
            job = dict(type=self.type, op='workflow_upload', file_data='#wf_data', name='#wf_name')
            p(self.action_btn_parametric("Hochladen", job, self.js_open_workflow(self.part_prepare)))
        with p.section("Vorhandene Workflows"):
            self.part_workflow_table(p)
        p.pre('', id='result')

    def part_nodes(self, p: base.Page, workflow: dict):
        p('<table class="comfyui_nodes"><tr><th>Id</th><th>Typ</th><th>Titel</th><th>Werte</th></tr>')
        for k, node in workflow.items():
            values = ', '.join(f'{key}={str(v)[:40]}' for key, v in node.get('inputs', {}).items()
                               if not isinstance(v, list))
            p(f"<tr><td>{html.escape(k)}</td><td>{html.escape(node.get('class_type', ''))}</td>"
              f"<td>{html.escape(self.node_title(node))}</td><td>{html.escape(values)}</td></tr>")
        p('</table>')

    def example_job(self, wf, info) -> dict:
        """ Job-Daten fuer einen Aufruf des vorbereiteten Workflows """
        job = dict(type=self.type, workflow=wf, prompt='...' if info['prompt'] else '')
        for img in info['images']:
            job['file.' + img['title']] = '<Base64>'
        job['count'] = 1
        job['random_seed'] = 1
        return job

    def part_execute(self, p: base.Page, params={}):
        p.h2("Workflow ausführen")
        loaded = self.part_load(p, params)
        if loaded is None:
            self.part_workflow_table(p)
            return
        wf, workflow, info = loaded
        p.div(w.a("Vorbereiten", self.link_workflow(self.part_prepare, wf)))
        self.part_warnings(p, info)
        job = dict(type=self.type, workflow=wf, prompt='', count='#count', random_seed='#random_seed')
        if info['prompt']:
            text = workflow[info['prompt']['node']]['inputs'][info['prompt']['key']]
            p.div(f'<textarea id="prompt" style="width: 100%; max-width: 600px; height: 200px;">'
                  f'{html.escape(text)}</textarea>')
            job['prompt'] = '#prompt'
        for img in info['images']:
            hid = self.html_id('image', img['node'])
            p.hidden('file.' + img['title'], '', hid)
            p.div(html.escape(img['title']))
            self.part_dropzone(p, hid, "Bild hier ablegen")
            job['file.' + img['title']] = '#' + hid
        p.form_input('count', "Anzahl", id='count', value='1', type='number', min='1')
        p.form_select('random_seed', "Zufälliger Seed", {'1': 'ja', '0': 'nein'}, '1', id='random_seed')
        p.form_elem(self.action_btn_parametric("Ausführen", job) + ' ' +
                    self.action_btn_parametric("In Jobdatei ablegen", {**job, 'op': 'queue'}))
        p.pre('', id='result')


__all__ = ['FaceSimilarity', 'ComfyUi']
