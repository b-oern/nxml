"""
ComfyUI-Anbindung: Workflows per REST an einen ComfyUI-Server senden und Jobs verwalten.
"""
import json
import time
import base64
import requests
import random
import os.path
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Callable
from enum import Enum

from PIL import Image

from nwebclient import runner as r, base
from nwebclient import util as u
from nwebclient import dev as d
from nwebclient import web as w


class Position(Enum):
    PREFIX = "prefix"
    POSTFIX = "postfix"


class ComfyUi(r.BaseJobExecutor):
    """
      comfyui: nxml.comfyui:ComfyUi
      comfyui:
        workflows:
            - path/w.json
        jobpath: /path/to/jobs/
        comfyui_path: /mnt/l/ComfyUI_windows_portable/ComfyUI

    """

    MODULES = ['opencv-python', 'torch', 'torchvision']
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
        self.define_sig(d.PStr('prompt', ''), d.PStr('image', ''), d.PStr('workflow', ''))

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

    def execute_q_c_prompt(self, data):
        self.info("q_c_promt("+str(data.keys())+")")
        self.create_queued_job(dict(
            type=self.type,
            op='workflow',
            workflow=data['payload']
        ))
        return self.success()

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

    def inject_resolution(self, width, height, workflow):
        pass # TODO

    def inject_seed(self, workflow):
        for k in workflow.keys():
            if 'seed' in workflow[k].get('inputs', {}):
                workflow[k]['inputs']['seed'] = random.randint(0, 4200000000000000)
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

    def send_prompt_to_comfyui(self, prompt: any, json_file='workflow.json', server_url=None, data={}) -> dict:
        self.info(f"Sending prompt to comfyui, extras: " + str(data.keys()))

        with open(json_file, 'r') as f:
            a = json.load(f)
        if isinstance(prompt, str):
            prompt = self.inject_prompt(prompt, a)
        workflow = self.merge(a, prompt if prompt is not None else {})
        if 'file_title' in data and 'file_data' in data:
            data['files'] = [dict(title=data['file_title'], data=data['file_data'])]
        if 'files' in data:
            for file in data['files']:
                workflow = self.inject_file_by_title(workflow, file['title'], file['data'], data.get('ext', 'png'),
                                                     data.get('input_key', 'image'), data.get('fileid', None))
        for elem in data.get('values', []):
            workflow = self.inject_value_by_title(workflow, elem['title'], elem['key'], elem['value'])
        return self.send_to_comfyui(workflow)

    def send_to_comfyui(self, workflow, server_url=None, data={}):
        if server_url is None:
            server_url = self.server_url
        payload = {"prompt": workflow}
        response = requests.post(f"{server_url}/prompt", json=payload)
        response.raise_for_status()
        if int(data.get('count', 1)) > 1:
            for i in range(int(data.get('count'))):
                payload = {"prompt": self.inject_seed(workflow)}
                response = requests.post(f"{server_url}/prompt", json=payload)
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
        response = requests.get(self.server_url + '/queue')
        # { system: {os:, ram_total:}, "devices": [{name:, vram_total:, vram_free},..] }
        response.raise_for_status()
        return response.json()

    def free(self):
        response = requests.post(self.server_url + '/free')
        # { system: {os:, ram_total:}, "devices": [{name:, vram_total:, vram_free},..] }
        response.raise_for_status()
        return response.json()

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
        if 'file_title' in data and 'file_data' in data:
            data['files'] = [dict(title=data['file_title'], data=data['file_data'])]
        self.create_queued_job(data, guid)
        return self.success(job_id=guid)

    def execute(self, data):
        if 'prompt' in data and 'image' in data and 'op' not in data:
            return self.send_prompt_and_image_to_comfyui(data['prompt'], data['image'], data['workflow'])
        elif 'prompt' in data and 'op' not in data:
            return self.send_prompt_to_comfyui(data['prompt'], data['workflow'], data=data)
        return super().execute(data)

    def part_index(self, p: base.Page, params={}):
        p.div("Start ComfyUi Jobs")
        p.div("Workflows")
        p.ul(map(lambda x: w.a(x, self.link(self.part_workflow, f'w={x}')),self.cfg.get('workflows', [])))
        p(self.action_btn(dict(title="Stats", type=self.type, op='rest', route='system_stats')))
        p(self.action_btn(dict(title="Queue", type=self.type, op='rest', route='queue')))
        p(self.action_btn(dict(title="Queue Count", type=self.type, op='queue_count')))
        p.ul([w.a("Prompt", self.link(self.part_prompt)),
              w.a("Prompt Ideogram 4", self.link(self.part_prompt_i4)),
              w.a("Prepare Workflow", self.link(self.part_prepareworkflow)),
              w.a("Image Transform", self.link(self.part_image_transform))])
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
        ])
        p.form_input('name', "Name", value='value', id='name')
        p.form_elem('<textarea id="prompt" style="width: 600px; height:200px;"></textarea>')
        p.form_input('workflow', "Workflow", value='/mnt/d/ai/z_image.json', id='workflow',
                     suggestions=self.cfg.get('workflows', []))
        p.form_elem(
            self.action_btn_parametric("Queue", dict(
                type=self.type,
                op='prepare_workflow',
                workflow='#workflow',
                title='#title',
                name='#name',
                prompt='#prompt'
            )))

    def part_workflow(self, p: base.Page, params={}):
        p.h2("Workflow")
        wf = params['w']
        wf_data = u.load_json_file(wf)
        for key, value in wf_data.items():
            p.div(f'{value.get("_meta", {}).get("title")} - {value.get("class_type")}')
            # LoadImage # TextEncodeQwenImageEditPlus # CLIPTextEncode

    def part_prompt(self, p: base.Page, params={}):
        p.div('<textarea id="prompt" style="width: 600px; height:200px;"></textarea>')
        p.input('workflow', value='/mnt/d/ai/z_image.json', id='workflow')
        p.input('count', value='10', id='count')
        base_p = dict(prompt='#prompt', type=self.type, workflow='#workflow', count='#count')
        p(self.action_btn_parametric("Execute", base_p))
        p(self.action_btn_parametric("Queue", {**base_p, 'op': 'queue'}))
        p(self.action_btn_parametric("MQueue", dict(
            type='mongodb', op='insert_one', row_prompt='#prompt', row_type=self.type, row_workflow='#workflow',
            db='nweb', collection='jobs', row_count='#count')))
        p(self.action_btn_parametric("CMQueue", dict(
            type='crypt', op='encrypt_job',
            data_prompt='#prompt', data_type=self.type, data_workflow='#workflow', data_count='#count',
            cjob_key='row_c', cjob=dict(type='mongodb', db='nweb', collection='jobs', op='insert_one', row_type='crypt')
        )))
        p(self.action_btn_parametric("T", dict(
            op='template', type=self.type, path='/mnt/d/ai/templates/zit.json', value_prompt='#prompt'
        )))
        p(self.action_btn_parametric("Klein9B", dict(
            op='template', type=self.type, path='/mnt/d/ai/templates/Klein9B.json', value_prompt='#prompt'
        )))
        p(self.action_btn_parametric("Krea2", dict(
            op='template', type=self.type, path='/mnt/d/ai/templates/Krea2.json', value_prompt='#prompt'
        )))
        p.pre('', id='result')

    def part_prompt_i4(self, p: base.Page, params={}):
        p.div('<textarea id="prompt" style="width: 600px; height:200px;"></textarea>')
        #p.input('workflow', value='/mnt/d/ai/z_image.json', id='workflow')
        p.input('count', value='10', id='count')
        base_p = dict(prompt='#prompt', type=self.type, count='#count')
        #p(self.action_btn_parametric("Execute", base_p))
        p(self.action_btn_parametric("Queue", {**base_p, 'op': 'queue_i4'}))
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
        p.form_input('image_title', "Image Input Title", id='image_title')
        p.form_input('workflow', "Workflow", value='/mnt/d/ai/z_image.json', id='workflow', suggestions=self.cfg.get('workflows', []))

        p.form_elem(
            self.action_btn_parametric("Queue", dict(
                type=self.type,
                op='queue',
                workflow='#workflow',
                file_data='#image_data',
                file_title='#image_title'
            )) +
            self.action_btn_parametric('Exec', {
                'type': self.type,
                'prompt': '',
                'workflow': '#workflow',
                'file_data': '#image_data',
                'file_title': '#image_title'
            })
        )

    def part_image_transform2(self, p: base.Page, params={}):
        p.js_ready('nx_initFileDragArea("dropZone", "image_data");')
        p('<div id="dropZone" style="width:300px;height:200px;border:2px dashed #999; display:flex;align-items:center;justify-content:center;">Bild hier ablegen</div>')
        p.hidden('image_data', '', "image_data")


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
