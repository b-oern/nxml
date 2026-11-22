"""
ComfyUI-Anbindung: Workflows per REST an einen ComfyUI-Server senden und Jobs verwalten.
"""
import json
import time
import html
import re
import glob
import base64
import mimetypes
import requests
import random
import os.path
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Callable
from enum import Enum

from PIL import Image

import urllib.parse

from nwebclient import runner as r, base
from nwebclient import util as u
from nwebclient import dev as d
from nwebclient import web as w

from nxml.comfyui.ui import UiWorkflow, to_type, SEED_MAX


class Position(Enum):
    PREFIX = "prefix"
    POSTFIX = "postfix"


class ComfyUi(r.BaseJobExecutor):
    """
      comfyui: nxml.comfyui:ComfyUi
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

    MODULES = []
    PROMPT_TITLE = 'Positive Prompt'
    PROMPT_KEYS = ('text', 'prompt')
    IMAGE_CLASSES = ('LoadImage',)
    OUTPUT_PREFIXES = ('Save', 'Preview', 'VHS_VideoCombine')
    SEED_KEYS = ('seed', 'noise_seed')

    def __init__(self, server_url="http://127.0.0.1:8188", args: u.Args = {}):
        super().__init__('comfyui')
        if args is None:
            args = u.Args()
        self.cfg = args.get(self.type, {})
        self.server_url = server_url
        if 'server_url' in self.cfg:
            self.server_url = self.cfg['server_url'].strip()
        self.wait_time = 4
        self.default_count = 10
        self.define_vars('server_url', 'wait_time', 'default_count')
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
        # r_success/r_fail liefern die Ergebnis-Class, nicht die Func -> nicht verketten
        f = self.define_sig(d.PStr('op', 'workflow_info'), d.Param('workflow'))\
            .set_description("Prompt-, Bild-, Seed- und Ausgabe-Nodes eines Workflows und was fuer die Ausfuehrung fehlt")
        f.r_success(api_format='bool', ready='bool', prompt='dict', text_nodes='list', images='list', seeds='list',
                    outputs='list', warnings='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'workflow_prepare'), d.Param('workflow'), d.PStr('prompt_node', ''), d.PStr('save_as', ''))\
            .set_description("Markiert den Prompt-Node, setzt Titel (title.<Node-Id>) und speichert den Workflow")
        f.r_success(workflow='str', ready='bool', warnings='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'workflow_upload'), d.Param('name'), d.Param('file_data'))\
            .set_description("Speichert einen Workflow (Base64-JSON) in workflow_dir")
        f.r_success(workflow='str')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'instance_run'), d.Param('workflow'), d.PInt('count', 1),
                            d.PStr('wait_finish', '0'))\
            .set_description("Startet einen in der Instanz gespeicherten Workflow (UI-Format), "
                             "Parameter als p:<Node-Id>:<Name>, zufällige Seeds als r:<Node-Id>:<Name>; "
                             "wait_finish=1 wartet auf das Ende und liefert die Bilder als Base64 (images)")
        f.r_success(prompt_ids='list', warnings='list', prompts='dict', images='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'instance_list'))\
            .set_description("In der Instanz gespeicherte Workflows (user/default/workflows)")
        f.r_success(workflows='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'instance_params'), d.Param('workflow'))\
            .set_description("Wichtige Parameter und Hinweise eines in der Instanz gespeicherten Workflows")
        f.r_success(workflow='str', params='list', warnings='list')
        f.r_fail()
        f = self.define_sig(d.PStr('op', 'instance_status'), d.Param('prompt_ids'))\
            .set_description("Status von Prompts (Liste oder Komma-getrennt): done, error, running, queued, unknown")
        f.r_success(prompts='dict')
        f.r_fail()
        self.define_sig(d.PStr('op', 'workflow'), d.Param('workflow', 'dict'), d.PInt('count', 1))\
            .set_description("Sendet einen fertigen Workflow (API-Format) an ComfyUI").r_success(result='dict')
        self.define_sig(d.PStr('op', 'q_c_prompt'), d.Param('payload', 'dict'))\
            .set_description("Legt einen fertigen Workflow (payload) als Jobdatei ab").r_success()
        self.define_sig(d.PStr('op', 'prepare_workflow'), d.Param('workflow'), d.Param('title'), d.Param('name'),
                        d.Param('prompt'))\
            .set_description("Jobdatei, die prompt in den Eingang name des Nodes title schreibt")\
            .r_success(job_id='str')
        self.define_sig(d.PStr('op', 'queue_i4'), d.Param('prompt'), d.PInt('count', 10))\
            .set_description("Ideogram-4-Job als Jobdatei ablegen").r_success(job_id='str')
        self.define_sig(d.PStr('op', 'template'), d.Param('path'))\
            .set_description("Fuellt ein Workflow-Template (value_<Name>=Wert) und legt es als Jobdatei ab")\
            .r_success(job_guid='str')
        self.define_sig(d.PStr('op', 'workflows'), d.PInt('full', 0))\
            .set_description("Pfade der angebotenen Workflows, mit full=1 auch deren Inhalt (workflows, errors)")\
            .r_success(value='list', workflows='dict', errors='dict')
        f = self.define_sig(d.PStr('op', 'workflow_get'), d.Param('workflow'))\
            .set_description("Inhalt eines Workflows (JSON)")
        f.r_success(workflow='dict')
        f.r_fail()
        self.define_sig(d.PStr('op', 'server_info'))\
            .set_description("Server-URL und Pfade der Konfiguration (comfyui_path, jobpath, workflow_dir, workflows)")\
            .r_success(server_url='str', cfg='dict')

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

    def execute_q_c_prompt(self, data):
        self.info("q_c_promt("+str(data.keys())+")")
        self.create_queued_job(dict(
            type=self.type,
            op='workflow',
            workflow=data['payload']
        ))
        return self.success()

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
        return self.send_to_comfyui(workflow, server_url, data)

    def send_to_comfyui(self, workflow, server_url=None, data={}):
        """
        Sendet einen Workflow (API-Format) count-mal an ComfyUI,
        mit random_seed=1 oder count > 1 jeweils mit neuen Seeds
        """
        if server_url is None:
            server_url = self.server_url
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

    def history_job(self, jobid):
        response = requests.get(self.server_url + f'/history/{jobid}')
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

    def instance_workflow_dir(self):
        """ :return: comfyui_path/user/default/workflows oder None, wenn comfyui_path fehlt bzw. es nicht existiert """
        if not self.cfg.get('comfyui_path'):
            return None
        wf_dir = os.path.join(self.cfg['comfyui_path'], 'user', 'default', 'workflows')
        return wf_dir if os.path.isdir(wf_dir) else None

    def instance_workflows(self) -> list:
        """
         In der Instanz gespeicherte Workflows (ComfyUI: user/default/workflows).
         Die offenen Tabs speichert das ComfyUI-Frontend nur im localStorage des Browsers, sie sind per API nicht abrufbar.

         Ist comfyui_path konfiguriert, wird das Verzeichnis direkt gelesen, sonst die API der Instanz gefragt.

         :return: [{path, size, modified}], size und modified fehlen bei aelteren ComfyUI-Versionen (API)
        """
        wf_dir = self.instance_workflow_dir()
        if wf_dir is not None:
            res = []
            for file in Path(wf_dir).rglob('*.json'):
                stat = file.stat()
                res.append({'path': file.relative_to(wf_dir).as_posix(), 'file': str(file),
                            'size': stat.st_size, 'modified': stat.st_mtime})
            return sorted(res, key=lambda item: item['modified'], reverse=True)
        response = requests.get(self.server_url + '/api/userdata',
                                params={'dir': 'workflows', 'recurse': 'true', 'split': 'false', 'full_info': 'true'},
                                timeout=10)
        if response.status_code == 404:  # Verzeichnis existiert (noch) nicht
            return []
        response.raise_for_status()
        res = [item if isinstance(item, dict) else {'path': item} for item in response.json()]
        return sorted(res, key=lambda item: item.get('modified', 0), reverse=True)

    def load_instance_workflow(self, path: str) -> dict:
        """ :param path: Pfad relativ zu user/default/workflows, wie ihn instance_workflows liefert """
        wf_dir = self.instance_workflow_dir()
        if wf_dir is not None:
            base_dir = Path(wf_dir).resolve()
            file = (base_dir / path).resolve()
            if base_dir not in file.parents:
                raise ValueError(f"{path} liegt nicht im Workflow-Verzeichnis")
            return self.load_workflow(file)
        response = requests.get(self.server_url + '/api/userdata/' + urllib.parse.quote('workflows/' + path, safe=''),
                                timeout=10)
        response.raise_for_status()
        return response.json()

    def object_info(self) -> dict:
        """ Node-Definitionen der Instanz, 5 Minuten zwischengespeichert """
        cached = getattr(self, '_object_info', None)
        if cached is None or time.time() - cached[0] > 300:
            response = requests.get(self.server_url + '/object_info', timeout=30)
            response.raise_for_status()
            cached = (time.time(), response.json())
            self._object_info = cached
        return cached[1]

    def instance_converter(self, path: str, data: dict = None) -> UiWorkflow:
        """
         :param data: p:<Node-Id>:<Name> setzt einen Parameter, r:<Node-Id>:<Name> (True) einen zufaelligen Seed
        """
        workflow = self.load_instance_workflow(path)
        if not UiWorkflow.is_ui_format(workflow):
            raise ValueError("Kein Workflow im UI-Format (nodes fehlt)")
        conv = UiWorkflow(workflow, self.object_info())
        if data:
            params = {(prm['node'], prm['name']): prm for prm in conv.params()}
            for key, value in data.items():
                parts = key.split(':', 2)
                if len(parts) != 3 or parts[0] != 'p' or (parts[1], parts[2]) not in params:
                    continue
                prm = params[(parts[1], parts[2])]
                rnd = data.get(f'r:{parts[1]}:{parts[2]}')
                if prm['seed'] and rnd is not None and to_type(rnd, 'BOOLEAN'):
                    value = random.randint(0, int(min(SEED_MAX, prm['options'].get('max', SEED_MAX))))
                conv.set_param(parts[1], parts[2], to_type(value, prm['type']))
                if prm['seed'] and rnd is not None:
                    conv.set_control(parts[1], parts[2], 'randomize' if to_type(rnd, 'BOOLEAN') else 'fixed')
        return conv

    def execute_instance_run(self, data):
        path = data.get('workflow')
        if not path:
            return self.fail("workflow fehlt")
        try:
            count = max(1, int(data.get('count') or 1))
        except ValueError:
            return self.fail("count ist keine Zahl")
        prompt_ids, warnings = [], []
        try:
            for i in range(count):
                conv = self.instance_converter(path, data)  # je Durchlauf neu, damit Seeds neu gewuerfelt werden
                api = conv.to_api()
                warnings = conv.validate(api)
                response = requests.post(self.server_url + '/prompt', timeout=30, json={
                    'prompt': api, 'extra_data': {'extra_pnginfo': {'workflow': conv.workflow}}})
                if response.status_code == 400:
                    err = response.json()
                    return self.fail(err.get('error', {}).get('message', 'Fehler') + ': ' +
                                     json.dumps(err.get('node_errors', {}), ensure_ascii=False)[:2000])
                response.raise_for_status()
                prompt_ids.append(response.json().get('prompt_id'))
        except (OSError, ValueError, requests.RequestException) as e:
            return self.fail(str(e))
        if to_type(data.get('wait_finish', False), 'BOOLEAN'):
            return self.wait_for_prompts(prompt_ids, warnings)
        return self.success(f"{len(prompt_ids)} Prompt(s) gestartet", prompt_ids=prompt_ids, warnings=warnings)

    FINISH_INTERVAL = 12      # Sekunden zwischen den Abfragen bei wait_finish
    FINISH_TIMEOUT = 3600     # Sekunden, die instance_run mit wait_finish hoechstens wartet

    def wait_for_prompts(self, prompt_ids: list, warnings: list = ()) -> dict:
        """
         Wartet (Abfrage alle FINISH_INTERVAL Sekunden), bis alle Prompts fertig sind, und liefert die Bilder als Base64

         :return: success, prompt_ids, prompts (Status je Prompt), images: [{prompt_id, filename, mime, data}]
        """
        deadline = time.time() + self.FINISH_TIMEOUT
        states = {}
        try:
            while True:
                queue = self.queue()
                states = {pid: self.prompt_status(pid, queue) for pid in prompt_ids}
                if all(s['status'] in ('done', 'error', 'unknown') for s in states.values()):
                    break
                if time.time() >= deadline:
                    return self.fail(f"Nicht fertig nach {self.FINISH_TIMEOUT} s", prompt_ids=prompt_ids,
                                     prompts=states)
                time.sleep(self.FINISH_INTERVAL)
            images = []
            for pid, state in states.items():
                if state['status'] == 'done':
                    images.extend(self.prompt_images(pid))
        except (OSError, ValueError, requests.RequestException) as e:
            return self.fail(str(e), prompt_ids=prompt_ids, prompts=states)
        errors = [f"{pid}: {s['message'] or s['status']}" for pid, s in states.items() if s['status'] != 'done']
        res = dict(prompt_ids=prompt_ids, prompts=states, images=images, warnings=list(warnings))
        if errors:
            return self.fail('; '.join(errors), **res)
        return self.success(f"{len(prompt_ids)} Prompt(s) fertig, {len(images)} Bild(er)", **res)

    def prompt_images(self, prompt_id: str) -> list:
        """ Bilder (und GIFs/Videos) aus der History eines Prompts, ueber /view geladen, als Base64 """
        history = self.history_job(prompt_id).get(prompt_id) or {}
        res = []
        for out in history.get('outputs', {}).values():
            for img in out.get('images', []) + out.get('gifs', []):
                if not isinstance(img, dict) or not img.get('filename'):
                    continue
                response = requests.get(self.server_url + '/view', timeout=60, params={
                    'filename': img['filename'], 'subfolder': img.get('subfolder', ''),
                    'type': img.get('type', 'output')})
                response.raise_for_status()
                mime = response.headers.get('Content-Type') or mimetypes.guess_type(img['filename'])[0] or ''
                res.append({'prompt_id': prompt_id, 'filename': img['filename'], 'mime': mime.split(';')[0],
                            'data': base64.b64encode(response.content).decode('ascii')})
        return res

    def execute_instance_list(self, data={}):
        try:
            workflows = [{k: v for k, v in item.items() if k != 'file'} for item in self.instance_workflows()]
        except (OSError, requests.RequestException) as e:
            return self.fail(str(e))
        return self.success(f"{len(workflows)} Workflow(s)", workflows=workflows)

    def execute_instance_params(self, data):
        path = data.get('workflow')
        if not path:
            return self.fail("workflow fehlt")
        try:
            conv = self.instance_converter(path)
            warnings = conv.validate(conv.to_api(randomize=False))
        except (OSError, ValueError, requests.RequestException) as e:
            return self.fail(str(e))
        return self.success(workflow=path, params=conv.params(), warnings=warnings)

    def prompt_status(self, prompt_id: str, queue: dict) -> dict:
        """ :return: {status: done|error|running|queued|unknown, outputs: [Dateinamen], message} """
        history = self.history_job(prompt_id).get(prompt_id)
        if history:
            status = history.get('status', {})
            outputs = [img.get('filename') for out in history.get('outputs', {}).values()
                       for img in out.get('images', []) + out.get('gifs', []) if isinstance(img, dict)]
            messages = [m[1].get('exception_message') for m in status.get('messages', [])
                        if isinstance(m, list) and len(m) > 1 and m[0] == 'execution_error' and isinstance(m[1], dict)]
            ok = status.get('status_str', 'success') == 'success'
            return {'status': 'done' if ok else 'error', 'outputs': outputs, 'message': '; '.join(filter(None, messages))}
        for key, status in (('queue_running', 'running'), ('queue_pending', 'queued')):
            if any(isinstance(item, list) and len(item) > 1 and item[1] == prompt_id for item in queue.get(key, [])):
                return {'status': status, 'outputs': [], 'message': ''}
        return {'status': 'unknown', 'outputs': [], 'message': 'weder in der Queue noch in der History'}

    def execute_instance_status(self, data):
        ids = data.get('prompt_ids') or []
        if isinstance(ids, str):
            ids = [i.strip() for i in ids.split(',') if i.strip()]
        try:
            queue = self.queue()
            return self.success(prompts={pid: self.prompt_status(pid, queue) for pid in ids})
        except (ValueError, requests.RequestException) as e:
            return self.fail(str(e))

    def wait_for_job(self, jobid):
        while True:
            res = self.history_job(jobid)
            if len(list(res.keys())) > 0:
                break
            time.sleep(1)
        return res

    def run_for_str(self, workflow_file, image):
        with open(workflow_file, 'r') as f:
            workflow = json.load(f)
        filename, file_extension = os.path.splitext(image)
        with open(image, 'rb') as f:
            b64data = base64.b64encode(f.read())
            workflow = self.inject_file_by_title(workflow, 'LoadImageEx', b64data,
                                                 file_extension, 'image', None)
        payload = {"prompt": workflow}
        response = requests.post(f"{self.server_url}/prompt", json=payload)
        res = response.json()
        jobid = res['prompt_id']
        result = self.wait_for_job(jobid)
        print(str(result))
        outputs = result.get(jobid, {}).get('outputs', {})
        for output in outputs.values():
            # "text": [ "photo,brown_hair, boots" ]
            s = output.get('text', None)
            if s is not None:
                return s[0]
        return None

    def nweb_image_map(self, workflow_file, meta_ns, meta_name, instance=None, limit=10):
        from nwebclient import NWebClient, NWebDoc
        n = NWebClient(instance)
        filterArgs = 'kind=image'
        def map_fn(doc: NWebDoc, c):
            doc.downloadThumbnail('current.jpg', 'm')
            s = self.run_for_str(workflow_file, 'current.jpg')
            os.remove('current.jpg')
            return s
        n.mapDocMeta(meta_ns=meta_ns, meta_name=meta_name, filterArgs=filterArgs, limit=limit, update=True,
                     mapFunction=map_fn)

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

    def create_job_filename(self, guid):
        path = self.cfg.get('jobpath', '.')
        return os.path.join(path, guid + '.job.json')

    def create_queued_job(self, data, guid=None):
        if guid is None:
            guid = u.guid()
        with open(self.create_job_filename(guid), 'w') as f:
            json.dump(data, f)
        return self.success(job_guid=guid)

    def execute_queue(self, data={}):
        data.pop('op')
        guid = u.guid()
        data['guid'] = guid
        if 'count' not in data:
            data['count'] = self.default_count
        if 'workflow' not in data:
            data['workflow'] = self.cfg.get('workflows', [''])[0]
        self.collect_files(data)
        self.create_queued_job(data, guid)
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

    def execute_workflows(self, data={}):
        paths = self.workflows()
        res = self.success(value=paths)
        if str(data.get('full', 0)) in ('1', 'true', 'True'):
            workflows, errors = {}, {}
            for path in paths:
                try:
                    workflows[path] = self.load_workflow(path)
                except (OSError, ValueError) as e:
                    errors[path] = str(e)
            res.update(workflows=workflows, errors=errors)
        return res

    def execute_workflow_get(self, data):
        try:
            return self.success(workflow=self.load_workflow(data['workflow']))
        except (KeyError, OSError, ValueError) as e:
            return self.fail(str(e))

    SERVER_INFO_KEYS = ('comfyui_path', 'jobpath', 'workflow_dir', 'workflows')

    def execute_server_info(self, data={}):
        return self.success(server_url=self.server_url,
                            cfg={k: self.cfg[k] for k in self.SERVER_INFO_KEYS if k in self.cfg})

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
            p.ul([w.a("Workflow hochladen und vorbereiten", self.link(self.part_prepare),
                      "API-Export hochladen, Prompt-Node und Bild-Titel festlegen"),
                  w.a("Workflow ausführen", self.link(self.part_execute),
                      "Vorbereiteten Workflow mit Prompt und Bildern starten")])
        with p.section("Server"):
            self.server_section(p)
        with p.section("Weitere"):
            self.more_links(p)
        p.pre('', id='result')

    def server_section(self, p: base.Page):
        """ Abschnitt Server der Startseite """
        p.prop("Server", html.escape(self.server_url))
        p.prop("ComfyUI-Pfad", html.escape(self.cfg.get('comfyui_path', '- (nötig für Bilder)')))
        p.prop("Jobpfad", html.escape(self.cfg.get('jobpath', '.')))
        p.prop("Workflow-Ordner", html.escape(self.cfg.get('workflow_dir', '-')))
        self.server_buttons(p)
        p(self.link_btn("Instanz", self.link(self.part_instance),
                        hint="In der ComfyUI-Instanz gespeicherte Workflows (user/default/workflows)"))

    def server_buttons(self, p: base.Page):
        p(self.action_btn(dict(title="Stats", type=self.type, op='rest', route='system_stats'),
                          hint="Systeminfos des Servers: Betriebssystem, RAM, GPUs und VRAM"))
        p(self.action_btn(dict(title="Queue", type=self.type, op='rest', route='queue'),
                          hint="Laufende und wartende Prompts in der ComfyUI-Queue"))
        p(self.action_btn(dict(title="Queue Count", type=self.type, op='queue_count'),
                          hint="Anzahl der wartenden Prompts in der ComfyUI-Queue"))
        p(self.action_btn(dict(title="Speicher freigeben", type=self.type, op='rest', route='free'),
                          hint="Entlädt die Modelle und gibt RAM und VRAM des Servers frei"))

    def more_links(self, p: base.Page):
        """ Abschnitt Weitere der Startseite """
        p.ul([w.a("Prompt mit Queue-Varianten (MongoDB, Crypt, Templates)", self.link(self.part_prompt),
                  "Prompt direkt, als Jobdatei, über MongoDB oder verschlüsselt einreihen"),
              w.a("Prompt Ideogram 4", self.link(self.part_prompt_i4),
                  "Prompt mit dem festen Ideogram-4-Workflow als Jobdatei ablegen"),
              w.a("Wert per Titel setzen (Jobdatei)", self.link(self.part_prepareworkflow),
                  "Jobdatei, die einen Text in den Eingang eines Nodes mit bestimmtem Titel schreibt"),
              w.a("Image Transform", self.link(self.part_image_transform),
                  "Ein Bild in einen Workflow geben, direkt oder als Jobdatei")])

    @staticmethod
    def link_btn(title, href, hint=None) -> str:
        """ Link im Stil der action_btn-Buttons """
        attr = f' title="{html.escape(hint)}"' if hint else ''
        return f'<button type="button"{attr} onclick="location.href={html.escape(json.dumps(href))};">' \
               f'{html.escape(title)}</button>'

    ASSET_KINDS = {'output': "Ausgaben", 'input': "Eingaben"}
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
    VIDEO_EXTS = ('.mp4', '.webm', '.mov')

    def instance_assets(self, kind='output', limit=200, offset=0) -> dict:
        """
         Assets der Instanz ueber deren API, neueste zuerst: /api/assets (Tag output bzw. input), ist die Assets-API
         nicht aktiv (ComfyUI ohne --enable-assets), /internal/files/<kind> (nur Pfade, ohne Groesse und Datum).

         :return: {items: [{id, path, filename, subfolder, size, modified, mime_type}], total, has_more, api}
        """
        if kind not in self.ASSET_KINDS:
            raise ValueError(f"Unbekannte Art {kind}")
        response = requests.get(self.server_url + '/api/assets', timeout=15, params={
            'include_tags': kind, 'sort': 'created_at', 'order': 'desc', 'limit': limit, 'offset': offset})
        if response.ok:
            data = response.json()
            items = []
            for asset in data.get('assets', []):
                name = str(asset.get('name', ''))
                items.append({'id': asset.get('id'), 'path': name, 'filename': name.rsplit('/', 1)[-1],
                              'subfolder': '', 'size': asset.get('size'), 'mime_type': asset.get('mime_type'),
                              'modified': self.parse_iso(asset.get('created_at'))})
            return {'items': items, 'total': data.get('total'), 'has_more': bool(data.get('has_more')),
                    'api': '/api/assets'}
        if response.status_code not in (404, 405, 501, 503):
            response.raise_for_status()
        response = requests.get(f"{self.server_url}/internal/files/{kind}", timeout=15)
        response.raise_for_status()
        items = []
        for entry in response.json():
            path = str(entry.get('name', entry.get('path', '')) if isinstance(entry, dict) else entry).replace('\\', '/')
            subfolder, _, filename = path.rpartition('/')
            items.append({'id': None, 'path': path, 'filename': filename, 'subfolder': subfolder, 'size': None,
                          'mime_type': None, 'modified': None})
        return {'items': items[offset:offset + limit], 'total': len(items), 'has_more': len(items) > offset + limit,
                'api': f'/internal/files/{kind}'}

    @staticmethod
    def parse_iso(value):
        """ ISO-Zeitstempel der Assets-API als Unix-Zeit oder None """
        if not value:
            return None
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None

    def asset_url(self, item: dict, kind: str) -> str:
        """ URL der Datei auf der ComfyUI-Instanz (muss vom Browser erreichbar sein) """
        if item.get('id'):
            return f"{self.server_url}/api/assets/{urllib.parse.quote(str(item['id']))}/content"
        return self.server_url + '/view?' + urllib.parse.urlencode(
            {'filename': item['filename'], 'subfolder': item['subfolder'], 'type': kind})

    def part_instance_assets(self, p: base.Page, params={}):
        kind = params.get('t', 'output')
        if kind not in self.ASSET_KINDS:
            kind = 'output'
        try:
            limit = max(1, int(params.get('limit', 200)))
        except ValueError:
            limit = 200
        p.h1("ComfyUI-Assets: " + self.ASSET_KINDS[kind])
        p.div(' | '.join(w.a(title, self.link(self.part_instance_assets, t=k)) if k != kind else f'<b>{title}</b>'
                         for k, title in self.ASSET_KINDS.items()) + ' | ' +
              w.a("Zurück zur Instanz", self.link(self.part_instance)))
        try:
            result = self.instance_assets(kind, limit)
        except (ValueError, requests.RequestException) as e:
            p.alert_warn("Assets nicht verfügbar: " + html.escape(str(e)))
            return
        p.prop("Quelle", html.escape(self.server_url + result['api']))
        if result['total'] is not None:
            p.prop("Anzahl", f"{len(result['items'])} von {result['total']}")
        more = result['has_more']
        assets = result['items']
        if len(assets) == 0:
            p.div("Keine Dateien.")
            return
        p('<div class="comfyui_assets" style="display: flex; flex-wrap: wrap; gap: 8px;">')
        for item in assets:
            url = html.escape(self.asset_url(item, kind), quote=True)
            name = html.escape(item['path'])
            ext = os.path.splitext(item['filename'])[1].lower()
            mime = str(item.get('mime_type') or '')
            if ext in self.IMAGE_EXTS or mime.startswith('image/'):
                media = f'<img src="{url}" loading="lazy" alt="{name}" style="max-width: 100%; max-height: 160px;">'
            elif ext in self.VIDEO_EXTS or mime.startswith('video/'):
                media = f'<video src="{url}" preload="metadata" muted controls style="max-width: 100%; max-height: 160px;"></video>'
            else:
                media = f'<div style="padding: 60px 0; text-align: center;">{html.escape(ext or "Datei")}</div>'
            info = []
            if isinstance(item.get('modified'), (int, float)):
                info.append(time.strftime('%Y-%m-%d %H:%M', time.localtime(item['modified'])))
            if isinstance(item.get('size'), (int, float)):
                info.append(f"{item['size'] / 1024:.0f} KB")
            p(f'<div style="width: 180px; overflow: hidden;" title="{name}">'
              f'<a href="{url}" target="_blank">{media}</a>'
              f'<div style="font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</div>'
              f'<div class="muted" style="font-size: 11px;">{" · ".join(info)}</div></div>')
        p('</div>')
        if more:
            p.div(w.a("Mehr anzeigen", self.link(self.part_instance_assets, t=kind, limit=limit * 2),
                      f"Die neuesten {limit * 2} Dateien anzeigen"))

    def part_instance(self, p: base.Page, params={}):
        p.h1("ComfyUI-Instanz")
        p.prop("Server", w.a(html.escape(self.server_url), self.server_url))
        wf_dir = self.instance_workflow_dir()
        p.prop("Quelle", html.escape(wf_dir) if wf_dir else "API der Instanz (comfyui_path nicht gesetzt)")
        p(self.link_btn("Assets", self.link(self.part_instance_assets),
                        hint="Ausgaben und Eingaben der ComfyUI-Instanz (output, input)"))
        try:
            workflows = self.instance_workflows()
        except requests.RequestException as e:
            p.alert_warn("Instanz nicht erreichbar: " + html.escape(str(e)))
            workflows = None
        if workflows is not None:
            self.part_instance_table(p, workflows)
        p.div(w.a("Zurück", self.link(self.part_index)))

    def part_instance_table(self, p: base.Page, workflows: list):
        """ Tabelle der Workflows aus instance_workflows (auch fuer RemoteComfy) """
        with p.section(f"Workflows ({len(workflows)})"):
            if len(workflows) == 0:
                p.div("Keine gespeicherten Workflows.")
                return
            p('<table class="comfyui_workflows"><tr><th>Workflow</th><th>Geändert</th><th>Größe</th></tr>')
            for item in workflows:
                path = str(item.get('path', ''))
                modified = item.get('modified')
                modified = time.strftime('%Y-%m-%d %H:%M', time.localtime(modified / 1000 if modified > 1e11 else modified)) \
                    if isinstance(modified, (int, float)) else '-'
                size = item.get('size')
                size = f"{size / 1024:.1f} KB" if isinstance(size, (int, float)) else '-'
                link = w.a(html.escape(path), self.link_workflow(self.part_instance_workflow, path),
                           "Parameter einstellen und ausführen")
                p(f'<tr><td title="{html.escape(str(item.get("file", path)))}">{link}</td>'
                  f'<td>{modified}</td><td>{size}</td></tr>')
            p('</table>')

    @staticmethod
    def part_param_input(p: base.Page, prm: dict, hid: str):
        """ Eingabefeld fuer einen Parameter aus UiWorkflow.params() """
        hint = prm['options'].get('tooltip') or f"{prm['group']}: {prm['name']} ({prm['type']})"
        value = prm['value']
        if prm['type'] == 'COMBO' and prm['choices']:
            choices = prm['choices'] if value in prm['choices'] else [value, *prm['choices']]
            p.form_select(hid, prm['label'], [str(c) for c in choices], str(value), id=hid, hint=hint)
        elif prm['type'] == 'BOOLEAN':
            checked = ' checked' if value else ''
            p.form_elem(f'<input type="checkbox" id="{hid}"{checked}>', prm['label'], hint=hint)
        elif prm['type'] in ('INT', 'FLOAT'):
            attrs = ''.join(f' {k}="{html.escape(str(prm["options"][k]))}"' for k in ('min', 'max', 'step')
                            if k in prm['options'])
            if prm['type'] == 'FLOAT' and 'step' not in prm['options']:
                attrs += ' step="any"'
            field = f'<input type="number" id="{hid}" value="{html.escape(str(value))}"{attrs}>'
            if prm['seed']:
                checked = ' checked' if prm['randomize'] else ''
                field += f' <label title="Bei jedem Lauf einen zufälligen Seed verwenden">' \
                         f'<input type="checkbox" id="{hid}_rnd"{checked}> zufällig</label>'
            p.form_elem(field, prm['label'], hint=hint)
        elif prm['options'].get('multiline') or len(str(value)) > 60 or '\n' in str(value):
            p.form_elem(f'<textarea id="{hid}" rows="5" style="width: 100%; max-width: 800px;">'
                        f'{html.escape(str(value))}</textarea>', prm['label'], hint=hint)
        else:
            p.form_input(hid, prm['label'], id=hid, value=html.escape(str(value), quote=True),
                         style='width: 100%; max-width: 600px;', hint=hint)

    def part_instance_workflow(self, p: base.Page, params={}):
        path = params.get('w', '')
        p.h1("Workflow: " + html.escape(path))
        p.div(w.a("Zurück zur Instanz", self.link(self.part_instance)))
        try:
            conv = self.instance_converter(path)
            warnings = conv.validate(conv.to_api(randomize=False))
        except (OSError, ValueError, requests.RequestException) as e:
            p.alert_error(f"{html.escape(path)} kann nicht geladen werden: {html.escape(str(e))}")
            return
        for msg in warnings:
            p.alert_warn(html.escape(msg))
        self.part_params_form(p, self, conv.params(), dict(type=self.type, op='instance_run', workflow=path),
                              "Workflow mit diesen Parametern in die Queue der ComfyUI-Instanz stellen",
                              self.JS_SHOW_IMAGES, wait_finish=True)
        p.div('', id='result_images', style='display: flex; flex-wrap: wrap; gap: 8px;')
        p.pre('', id='result')

    # on_success fuer instance_run mit wait_finish: Bilder aus r.images unter dem Formular anzeigen
    JS_SHOW_IMAGES = """function(r) {
        var box = document.getElementById('result_images');
        if (!box || !r || !r.images) return;
        box.innerHTML = '';
        r.images.forEach(function(img) {
            var src = 'data:' + (img.mime || 'image/png') + ';base64,' + img.data;
            var el = document.createElement((img.mime || '').indexOf('video/') === 0 ? 'video' : 'img');
            el.src = src; el.title = img.filename; el.style.maxWidth = '100%'; el.style.maxHeight = '512px';
            if (el.tagName === 'VIDEO') { el.controls = true; }
            box.appendChild(el);
        });
    }"""

    @classmethod
    def part_params_form(cls, p: base.Page, runner: r.BaseJobExecutor, prms: list, job: dict, run_hint: str,
                         on_success='null', wait_finish=False):
        """
         Formular fuer die Parameter aus UiWorkflow.params() mit Anzahl und Run-Button (auch fuer ComfyRemote)

         :param job: Basis des Jobs, die Parameter kommen als p:<Node-Id>:<Name> bzw. r:... (zufaelliger Seed) dazu
         :param wait_finish: Haken "Wait for finish" anbieten (instance_run wartet und liefert die Bilder als Base64)
        """
        job = dict(job, count='#count')
        if wait_finish:
            job['wait_finish'] = '#wait_finish'
        groups = {}
        for prm in prms:
            groups.setdefault(prm['group'], []).append(prm)
        if len(groups) == 0:
            p.div("Keine wichtigen Parameter gefunden, der Workflow läuft mit den gespeicherten Werten.")
        for group, items in groups.items():
            with p.section(html.escape(str(group))):
                for prm in items:
                    hid = cls.html_id('p', f"{prm['node']}_{prm['name']}")
                    cls.part_param_input(p, prm, hid)
                    job[f"p:{prm['node']}:{prm['name']}"] = '#' + hid
                    if prm['seed']:
                        job[f"r:{prm['node']}:{prm['name']}"] = '#' + hid + '_rnd'
        with p.section("Ausführen"):
            p.form_input('count', "Anzahl", id='count', type='number', value='1', min='1',
                         hint="So oft wird der Workflow in die Queue der Instanz gestellt")
            if wait_finish:
                p.form_elem('<label title="Wartet, bis alle Prompts fertig sind (Abfrage alle '
                            f'{cls.FINISH_INTERVAL} s), und liefert die Bilder als Base64 in der Antwort">'
                            '<input type="checkbox" id="wait_finish"> Wait for finish</label>', "")
            p.form_elem(runner.action_btn_parametric("Run", job, on_success, hint=run_hint))

    def part_workflow_table(self, p: base.Page):
        if len(self.workflows()) == 0:
            p.div("Keine Workflows konfiguriert (workflows oder workflow_dir).")
            return
        p('<table class="comfyui_workflows"><tr>'
          f'<th title="{html.escape("Dateiname, der volle Pfad steht im Tooltip der Zeile")}">Workflow</th>'
          f'<th title="{html.escape(f"ja, wenn ein Text-Node mit {self.PROMPT_TITLE} im Titel markiert ist")}">Prompt</th>'
          f'<th title="{html.escape("Titel der LoadImage-Nodes, unter diesen Namen werden Bilder übergeben")}">Bilder</th>'
          f'<th title="{html.escape("bereit oder Anzahl der Hinweise, Details unter Vorbereiten")}">Status</th>'
          '<th></th></tr>')
        for wf in self.workflows():
            try:
                info = self.analyze_workflow(self.load_workflow(wf))
                prompt = 'ja' if info['prompt'] else '-'
                images = ', '.join(html.escape(img['title']) for img in info['images']) or '-'
                status = 'bereit' if info['ready'] else str(len(info['warnings'])) + ' Hinweis(e)'
            except (OSError, ValueError) as e:
                prompt, images, status = '', '', 'Fehler: ' + html.escape(str(e))
            actions = w.a("Ausführen", self.link_workflow(self.part_execute, wf),
                          "Workflow mit Prompt und Bildern starten") + ' | ' + \
                w.a("Vorbereiten", self.link_workflow(self.part_prepare, wf),
                    "Prompt-Node und Bild-Titel festlegen, Hinweise ansehen")
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
            p.form_select('prompt_node', "Prompt-Node", options,
                          info['prompt']['node'] if info['prompt'] else '', id='prompt_node',
                          hint=f'Dieser Node erhält den Titel "{self.PROMPT_TITLE}", '
                               f'bei allen anderen wird er entfernt')
        with p.section("Bilder"):
            if len(info['images']) == 0:
                p.div("Keine LoadImage-Nodes, der Workflow braucht keine Bilder.")
            else:
                p.div("Der Titel ist der Name, unter dem das Bild beim Aufruf übergeben wird (file.&lt;Titel&gt;).")
            for img in info['images']:
                hid = self.html_id('title', img['node'])
                p.form_input('title.' + img['node'], f"Node {html.escape(img['node'])}", id=hid,
                             value=html.escape(img['title'], quote=True),
                             hint=f"Titel von LoadImage-Node {img['node']}, eindeutig je Workflow; "
                                  f"das Bild wird als file.<Titel> übergeben")
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
                         style='width: 100%; max-width: 600px;',
                         hint="Pfad der Workflow-Datei; ein anderer Pfad legt eine Kopie an")
            p.form_elem(self.action_btn_parametric("Speichern", job, self.js_open_workflow(self.part_prepare),
                                                   hint="Titel setzen, Prompt-Node markieren und Workflow speichern") +
                        ' ' + w.a("Ausführen", self.link_workflow(self.part_execute, wf),
                                  "Workflow mit Prompt und Bildern starten"))
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
            p.form_input('name', "Dateiname", id='wf_name', value='workflow.json',
                         hint="Name im Workflow-Ordner, .json wird bei Bedarf ergänzt")
            p.prop("Ziel", html.escape(self.cfg.get('workflow_dir', '.')))
            job = dict(type=self.type, op='workflow_upload', file_data='#wf_data', name='#wf_name')
            p(self.action_btn_parametric("Hochladen", job, self.js_open_workflow(self.part_prepare),
                                         hint="Speichert den Workflow im Workflow-Ordner und öffnet ihn zum Vorbereiten"))
        with p.section("Vorhandene Workflows"):
            self.part_workflow_table(p)
        p.pre('', id='result')

    def execute_prepare_workflow(self, data={}):
        data.pop('op')
        path = self.cfg.get('jobpath', '.')
        guid = u.guid()
        data['guid'] = guid
        if 'workflow' not in data:
            data['workflow'] = self.cfg.get('workflows', [''])[0]
        data['values'] = [{'title': data['title'], 'key': data['name'], 'value': data['prompt']}]
        #if 'file_title' in data and 'file_data' in data:
        #    data['files'] = [dict(title=data['file_title'], data=data['file_data'])]
        with open(os.path.join(path, guid + '.job.json'), 'w') as f:
            json.dump(data, f)
        return self.success(job_id=guid)

    def part_prepareworkflow(self, p: base.Page, params={}):
        p.h2("Prepare Workflow")
        p.form_input('title', "Inject Title", value='Prompt', id='title', suggestions=[
            'Positive Prompt'
        ], hint="Titel des Nodes, in den der Text geschrieben wird")
        p.form_input('name', "Name", value='value', id='name',
                     hint="Name des Eingangs (inputs) im Node, z.B. text oder value")
        p.form_elem(f'<textarea id="prompt" style="width: 600px; height:200px;" '
                    f'title="{html.escape("Wert für den Eingang")}"></textarea>')
        p.form_input('workflow', "Workflow", value='/mnt/d/ai/z_image.json', id='workflow',
                     suggestions=self.cfg.get('workflows', []), hint="Pfad zum Workflow im API-Format")
        p.form_elem(
            self.action_btn_parametric("Queue", dict(
                type=self.type,
                op='prepare_workflow',
                workflow='#workflow',
                title='#title',
                name='#name',
                prompt='#prompt'
            ), hint="Legt eine Jobdatei im Jobpfad ab"))

    def part_prompt(self, p: base.Page, params={}):
        p.div(f'<textarea id="prompt" style="width: 600px; height:200px;" '
              f'title="{html.escape("Prompt für den Text-Node mit " + self.PROMPT_TITLE + " im Titel")}"></textarea>')
        p.input('workflow', value='/mnt/d/ai/z_image.json', id='workflow',
                hint="Pfad zum Workflow im API-Format")
        p.input('count', value='10', id='count', hint="Anzahl der Durchläufe, jeweils mit neuem Seed")
        base_p = dict(prompt='#prompt', type=self.type, workflow='#workflow', count='#count')
        p(self.action_btn_parametric("Execute", base_p, hint="Sendet den Prompt sofort an ComfyUI"))
        p(self.action_btn_parametric("Queue", {**base_p, 'op': 'queue'}, hint="Legt eine Jobdatei im Jobpfad ab"))
        p(self.action_btn_parametric("MQueue", dict(
            type='mongodb', op='insert_one', row_prompt='#prompt', row_type=self.type, row_workflow='#workflow',
            db='nweb', collection='jobs', row_count='#count'), hint="Job in MongoDB nweb.jobs eintragen"))
        p(self.action_btn_parametric("CMQueue", dict(
            type='crypt', op='encrypt_job',
            data_prompt='#prompt', data_type=self.type, data_workflow='#workflow', data_count='#count',
            cjob_key='row_c', cjob=dict(type='mongodb', db='nweb', collection='jobs', op='insert_one', row_type='crypt')
        ), hint="Job verschlüsselt in MongoDB nweb.jobs eintragen"))
        p(self.action_btn_parametric("T", dict(
            op='template', type=self.type, path='/mnt/d/ai/templates/zit.json', value_prompt='#prompt'
        ), hint="Template zit.json mit dem Prompt als Jobdatei ablegen"))
        p(self.action_btn_parametric("Klein9B", dict(
            op='template', type=self.type, path='/mnt/d/ai/templates/Klein9B.json', value_prompt='#prompt'
        ), hint="Template Klein9B.json mit dem Prompt als Jobdatei ablegen"))
        p(self.action_btn_parametric("Krea2", dict(
            op='template', type=self.type, path='/mnt/d/ai/templates/Krea2.json', value_prompt='#prompt'
        ), hint="Template Krea2.json mit dem Prompt als Jobdatei ablegen"))
        p.pre('', id='result')

    def part_prompt_i4(self, p: base.Page, params={}):
        p.div(f'<textarea id="prompt" style="width: 600px; height:200px;" '
              f'title="{html.escape("Prompt für Ideogram 4")}"></textarea>')
        #p.input('workflow', value='/mnt/d/ai/z_image.json', id='workflow')
        p.input('count', value='10', id='count', hint="Anzahl der Durchläufe, jeweils mit neuem Seed")
        base_p = dict(prompt='#prompt', type=self.type, count='#count')
        #p(self.action_btn_parametric("Execute", base_p))
        p(self.action_btn_parametric("Queue", {**base_p, 'op': 'queue_i4'}, hint="Legt eine Jobdatei im Jobpfad ab"))
        p.pre('', id='result')

    def execute_workflow(self, data):
        r = self.send_to_comfyui(data['workflow'], data=data)
        return self.success(result=r)

    def execute_queue_i4(self, data):
        api_workflow = '/mnt/d/ai/i4_api.json'
        cprompt = u.load_json_file(api_workflow)
        cprompt["98:24"]["inputs"]["text"] = data['prompt']
        data.pop('op')
        path = self.cfg.get('jobpath', '.')
        guid = u.guid()
        data['guid'] = guid
        data['op'] = 'workflow'
        data['workflow'] = cprompt
        if 'count' not in data:
            data['count'] = self.default_count
        with open(os.path.join(path, guid + '.job.json'), 'w') as f:
            json.dump(data, f)
        return self.success(job_id=guid)

    def part_image_transform(self, p: base.Page, params={}):
        p.js_ready('nx_initFileDragArea("dropZone", "image_data");')
        p(w.dropzone(title="Bild hier ablegen"))
        p.hidden('image_data', '', "image_data")
        p.form_input('image_title', "Image Input Title", id='image_title',
                     hint="Titel des LoadImage-Nodes, der das Bild erhält")
        p.form_input('workflow', "Workflow", value='/mnt/d/ai/z_image.json', id='workflow',
                     suggestions=self.cfg.get('workflows', []), hint="Pfad zum Workflow im API-Format")

        p.form_elem(
            self.action_btn_parametric("Queue", dict(
                type=self.type,
                op='queue',
                workflow='#workflow',
                file_data='#image_data',
                file_title='#image_title'
            ), hint="Legt eine Jobdatei im Jobpfad ab") +
            self.action_btn_parametric('Exec', {
                'type': self.type,
                'prompt': '',
                'workflow': '#workflow',
                'file_data': '#image_data',
                'file_title': '#image_title'
            }, hint="Sendet das Bild sofort an ComfyUI")
        )

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
        p.div(w.a("Vorbereiten", self.link_workflow(self.part_prepare, wf),
                  "Prompt-Node und Bild-Titel festlegen, Hinweise ansehen"))
        self.part_warnings(p, info)
        job = dict(type=self.type, workflow=wf, prompt='', count='#count', random_seed='#random_seed')
        if info['prompt']:
            text = workflow[info['prompt']['node']]['inputs'][info['prompt']['key']]
            tooltip = html.escape(f"Wird in Node {info['prompt']['node']} ({info['prompt']['title']}) geschrieben, "
                                  f"vorbelegt mit dem Text aus dem Workflow")
            p.div(f'<textarea id="prompt" style="width: 100%; max-width: 600px; height: 200px;" title="{tooltip}">'
                  f'{html.escape(text)}</textarea>')
            job['prompt'] = '#prompt'
        for img in info['images']:
            hid = self.html_id('image', img['node'])
            p.hidden('file.' + img['title'], '', hid)
            p.div(html.escape(img['title']),
                  title=html.escape(f"Bild für LoadImage-Node {img['node']}, wird nach comfyui_path/input geschrieben"))
            self.part_dropzone(p, hid, "Bild hier ablegen")
            job['file.' + img['title']] = '#' + hid
        p.form_input('count', "Anzahl", id='count', value='1', type='number', min='1',
                     hint="Anzahl der Durchläufe, ab 2 immer mit neuem Seed")
        p.form_select('random_seed', "Zufälliger Seed", {'1': 'ja', '0': 'nein'}, '1', id='random_seed',
                      hint="Seeds vor jedem Durchlauf zufällig setzen, sonst gelten die Werte im Workflow")
        p.form_elem(self.action_btn_parametric("Ausführen", job, hint="Sendet den Workflow sofort an ComfyUI") + ' ' +
                    self.action_btn_parametric("In Jobdatei ablegen", {**job, 'op': 'queue'},
                                               hint="Legt eine Jobdatei im Jobpfad ab, Bilder werden mitgespeichert"))
        p.pre('', id='result')


    def run_apiworkflow(self, workflow: Path|str|dict, values: dict = {}, create_job_only=False):
        if isinstance(workflow, Path) or isinstance(workflow, str):
            workflow = u.load_json_file(workflow)
        values[424242] = random.randint(0, sys.maxsize)
        workflow = self.replace_value(workflow, values)
        if create_job_only:
            return self.create_queued_job(dict(
                type=self.type,
                op='workflow',
                workflow=workflow
            ))
        else:
            return self.execute_workflow(dict(workflow=workflow))

    def execute_template(self, data={}):
        path = data['path']
        data.pop('type')
        data.pop('op')
        data.pop('path')
        return self.run_apiworkflow(path, data, True)

    @staticmethod
    def extract_prompt(file):
        im = Image.open(file)
        im.load()  # Needed only for .png EXIF data (see citation above)
        parameters = ''
        message = ''
        prompt = ''
        try:
            # params = json.loads(im.info)
            # parameters = params['parameters']
            if 'prompt' in im.info:
                p = im.info['prompt']
                data = json.loads(p)
                return u.extract_longest_str(data)
        except Exception as e:
            print(e)
            message = str(e)
        return ""

    @staticmethod
    def rename_files(path: str | Path, extract_feature: Callable[[Path], str], position: Position) -> None:
        """
        Benennt alle Dateien im Verzeichnis um.
        Args:
            path: Verzeichnis mit den Dateien.
            extract_feature: Callback, der für eine Datei ein Merkmal (String) liefert.
            position: Einfügeposition des Merkmals (PREFIX oder POSTFIX| "prefix" or "postfix").
        """
        path = Path(path)
        for file in path.iterdir():
            if not file.is_file():
                continue
            feature = extract_feature(file)
            if position == Position.PREFIX:
                new_name = f"{feature}_{file.name}"
            else:  # Position.POSTFIX
                new_name = f"{file.stem}_{feature}{file.suffix}"
            file.rename(file.with_name(new_name))

    @staticmethod
    def rename_with_promptid(path: str | Path):
        def mapfn(filename):
            prompt = ComfyUi.extract_prompt(filename)
            return "p_" + u.hash_text(prompt)
        ComfyUi.rename_files(path, mapfn, Position.PREFIX)

    @staticmethod
    def replace_value(data: dict, replace_vals: dict) -> dict:
        """
        Recursively traverse `data` and replace any value found in
        `replace_vals` with its corresponding replacement.
        Example:
            data = {
                "a": 1,
                "b": {"c": 2, "d": [1, 2, 3]}
            }

            replace_value(data, {1: "one", 2: "two"})
            # {'a': 'one', 'b': {'c': 'two', 'd': ['one', 'two', 3]}}
        """
        def _replace(obj):
            if isinstance(obj, Mapping):
                return {k: _replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_replace(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_replace(v) for v in obj)
            elif isinstance(obj, set):
                return {_replace(v) for v in obj}
            else:
                return replace_vals.get(obj, obj)
        return _replace(data)


__all__ = ['ComfyUi', 'Position']
