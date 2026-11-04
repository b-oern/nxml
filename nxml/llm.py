import os
import re

import requests
import json
import docker
import time
import base64

from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import web as w
from nwebclient import util as u
from nwebclient import crypt
from nwebclient import dev as d
from nwebclient.dev import Param, PStr

from nxml import nlp


class BaseLLM(r.BaseJobExecutor):
    """
    Basisklasse für LLM-Runner, die Prompts (auch verschlüsselt als cprompt) an ein Sprachmodell weitergeben.
    """
    TAGS = ['llm']

    def __init__(self, type='llm'):
        super().__init__()
        self.type = type
        self.count = 0
        self.define_vars('count')
        self.define_sig(Param('prompt', 'str')).set_description("Sendet einen Prompt an das LLM")\
            .r_success(response='str')
        self.define_sig(Param('cprompt', 'str')).set_description("Verschlüsselter Prompt (NPY_KEY), Antwort verschlüsselt")\
            .r_success(response='str')

    def prompt(self, prompt, data={}):
        pass

    def cprompt(self, data: dict):
        from nwebclient import crypt
        args = u.Args()
        pw = args.get('NPY_KEY', 'xxx')
        result = self.prompt(crypt.decrypt_message(data['cprompt'], pw))
        return self.success('ok', response=crypt.encrypt_message(result['response'], pw))

    def execute(self, data):
        if 'prompt' in data:
            self.count += 1
            return self.prompt(data['prompt'], data)
        if 'cprompt' in data:
            return self.cprompt(data)
        return super().execute(data)

    def part_index(self, p: b.Page, params={}):
        #p.input('prompt', id='prompt')
        p('<textarea style="width:90%; height:300px" name="prompt" id="prompt"></textarea>')
        func = """
            function x(data) {
                var resp = data["response"].replace('\\n', '<br />');
                document.getElementById("result_view").innerHTML = resp; 
            }
        """
        p(w.button_js("Prompt", 'exec_job_p({"type": "' + self.type + '", "prompt": "#prompt"}, '+func+')'))
        p.div('', id='result_view')
        p.pre('', id='result')


class OpenAiLLM(BaseLLM):
    """
    Schickt Prompts an die OpenAI-API oder einen kompatiblen Endpunkt (OPENAI_BASEURL).
    """

    MODULES = ['openai']

    MODELS = ['gpt-4o', 'gpt-4o-mini', 'gpt-4']

    def __init__(self, api_key=None, args: u.Args = None):
        super().__init__('gptllm')
        self.last_request = 0
        self.define_vars('model', 'last_request', 'base_url')
        if args is None:
            args = u.Args()
        self.model = args.get('OPENAI_MODEL', 'gpt-4o')
        self.key = args.get('OPENAI_KEY')
        self.base_url = args.get('OPENAI_BASEURL', None)

    def prompt(self, prompt, data):
        self.last_request = time.time()
        from openai import OpenAI
        if self.base_url is None:
            client = OpenAI(api_key=self.key)
        else:
            client = OpenAI(api_key=self.key, base_url=self.base_url)

        # messages = []
        # system_content = '''You are a marketing assistant called MarkBot.
        # You only respond to greetings and marketing-related questions.
        # For any other question you must answer "I'm not suitable for this type of tasks.".'''
        # messages.append({"role": "system", "content": system_content})
        # prompt_text = 'Hi, How can i improve my sellings of cakes?'
        # messages.append({"role": "user", "content": prompt_text})
        # wirft openai.error.RateLimitError: You exceeded your current quota, please check your plan and billing details.
        completion = client.chat.completions.create(
            model=self.model,  # alt gpt-4o-mini
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        resp = completion.choices[0].message.content
        #resp = str(completion)
        return self.success('ok', response=resp)


class RLLM(BaseLLM):
    """
      nxml.llm:RLLM
    """

    def __init__(self, type='rllm', args: u.Args = None):
        super().__init__(type)
        if args is None:
            args = u.Args()
        self.count = 0
        self.args = args

    def prompt(self, prompt, data):
        self.count += 1
        prompt = data['prompt']
        url = self.args.get('LLM_URL')
        pw = self.args.get('NPY_KEY', '')
        cprompt = crypt.encrypt_message(prompt, pw)
        resp = requests.post(url, {
            'cprompt': cprompt
        })
        try:
            result = json.loads(resp.text)
            text = crypt.decrypt_message(result['response'], pw)
            return self.success('ok', response=text, count=self.count)
        except Exception as e:
            return self.fail(str(e), response_text=resp.text, status_code=resp.status_code)


class OLLama(BaseLLM):
    """
    Schickt Prompts an ein lokal laufendes Ollama-Modell.
    """

    MODULES = ['ollama']

    def __init__(self, type='ollm', model=None, args: u.Args = None):
        super().__init__(type)
        import ollama as o
        if args is None:
            args = u.Args()
        if model is None:
            model = args.get('ollama_model', 'llama3')
        self.model = model
        self.ollama = o

    def prompt(self, prompt, data={}):
        response = self.ollama.generate(model=self.model, prompt=prompt)
        return self.success('ok', response=str(response.response))


class OLLamaDockerd(BaseLLM):
    """
    Startet Ollama als Docker-Container und schickt Prompts an das darin geladene Modell.
    """

    def __init__(self, type='ollm', model='llama3.2'):
        super().__init__(type)
        self.model = model
        self.define_vars('model')
        self.docker = docker.from_env()
        self.inner = OLLama()
        if self.exists():
            self.start()
        else:
            # docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
            self.container = self.docker.containers.run("ollama/ollama", detach=True, name='ollama', port=11434) # remove
            self.delayed(30, self.docker_load)

    def docker_load(self):
        self.container.exec_run('ollama run ' + self.model, detach=True)

    def start(self):
        for c in self.docker.containers.list():
            if c.name == 'ollama':
                c.start()

    def exists(self):
        for c in self.docker.containers.list():
            if c.name == 'ollama':
                return True
        return False

    def execute(self, data):
        return self.inner.execute(data)


class LMStudioError(Exception):
    pass


class LMStudioUiMixin:
    """
    Signaturen und UI (Status, Modelle, Chat, Embeddings) von LMStudio, auch für RemoteLmStudio.
    Die Seiten rufen die Ops per Job auf; lmstudio_cfg()['remote'] schaltet auf asynchrone Jobs um.
    """

    # erlaubte lms-Befehle für op=cli (nur lesend)
    CLI_COMMANDS = {
        'status': ['status'],
        'version': ['version'],
        'ps': ['ps'],
        'ls': ['ls'],
        'server_status': ['server', 'status'],
        'daemon_status': ['daemon', 'status'],
        'runtime_ls': ['runtime', 'ls'],
        'runtime_survey': ['runtime', 'survey']
    }

    def define_lmstudio_sigs(self):
        chat = [d.PStr('model', ''), d.PStr('system', ''), d.NP('temperature', 'float', 0.7),
                d.PInt('max_tokens', -1)]
        self.define_sig(d.PStr('op', 'status'))\
            .set_description("Server-URL, Erreichbarkeit, geladene Modelle und CLI")\
            .r(success='bool', url='str', online='bool', loaded='list', cli='str')
        self.define_sig(d.PStr('op', 'models'), d.PStr('kind', ''))\
            .set_description("Alle Modelle mit state (loaded/not-loaded), kind: llm, embedding oder leer")\
            .r(success='bool', models='list')
        self.define_sig(d.PStr('op', 'chat'), d.Param('messages', 'list'), *chat, d.PStr('image', ''))\
            .set_description("Chat (/v1/chat/completions), messages als Liste oder JSON, "
                             "image: Base64-Bild für Vision-Modelle zur letzten Nachricht")\
            .r(success='bool', response='str', reasoning='str', model='str', usage='dict', stats='dict')
        self.define_sig(d.PStr('op', 'chat'), d.Param('prompt'), *chat)\
            .set_description("Einzelner Prompt mit optionalem Systemprompt").r(success='bool', response='str')
        self.define_sig(d.PStr('op', 'completion'), d.Param('prompt'), d.PStr('model', ''),
                        d.NP('temperature', 'float', 0.7), d.PInt('max_tokens', 256))\
            .set_description("Textvervollständigung ohne Chat-Template (/v1/completions)")\
            .r(success='bool', response='str', usage='dict')
        self.define_sig(d.PStr('op', 'embed'), d.Param('input'), d.PStr('model', ''))\
            .set_description("Embeddings (/v1/embeddings), input: Text oder JSON-Liste von Texten")\
            .r(success='bool', embedding='list', embeddings='list', dims='int')
        self.define_sig(d.PStr('op', 'load'), d.Param('model'), d.PInt('context_length', 0),
                        d.PStr('gpu', ''), d.PInt('ttl', 0), d.PStr('identifier', ''))\
            .set_description("Modell laden (REST), gpu (off, max, 0-1), ttl (s) und identifier über das lms-CLI")\
            .r(success='bool', instance_id='str')
        self.define_sig(d.PStr('op', 'unload'), d.PStr('instance_id', ''), d.PInt('all', 0))\
            .set_description("Modell-Instanz entladen, all=1 entlädt alle").r(success='bool', unloaded='list')
        self.define_sig(d.PStr('op', 'download'), d.Param('model'), d.PStr('quantization', ''))\
            .set_description("Modell herunterladen (z.B. qwen/qwen3-4b oder Hugging-Face-URL), "
                             "Fortschritt über op=download_status").r(success='bool', job_id='str', status='str')
        self.define_sig(d.PStr('op', 'download_status'), d.Param('job_id'))\
            .set_description("Fortschritt eines Downloads")\
            .r(success='bool', status='str', downloaded_bytes='int', total_size_bytes='int')
        self.define_sig(d.PStr('op', 'server_start'), d.PInt('port', 0), d.PStr('bind', ''), d.PInt('cors', 0))\
            .set_description("LM Studio (Daemon) und Server über das lms-CLI starten, port=0: Port aus der URL")\
            .r(success='bool', output='str')
        self.define_sig(d.PStr('op', 'server_stop')).set_description("Server über das lms-CLI stoppen")\
            .r(success='bool', output='str')
        self.define_sig(d.PStr('op', 'daemon'), d.PStr('action', 'up'))\
            .set_description("LM-Studio-Daemon (llmster) starten oder beenden: action=up, down, status")\
            .r(success='bool', output='str')
        self.define_sig(d.PStr('op', 'cli'), d.PStr('cmd', 'status'))\
            .set_description("Lesender lms-Befehl: " + ', '.join(self.CLI_COMMANDS))\
            .r(success='bool', output='str')

    def lmstudio_cfg(self) -> dict:
        """ Konfiguration für das JS der Seiten """
        return {'type': self.type, 'url': self.url, 'model': self.model, 'embedding_model': self.embedding_model}

    # -- Web --

    PAGES = [('index', 'Status'), ('models', 'Modelle'), ('chat', 'Chat'), ('embeddings', 'Embeddings')]

    def part_page(self, p: b.Page, name, html, js):
        nav = ' '.join(f'<a href="{self.link(page)}" class="{"active" if page == name else ""}">{title}</a>'
                       for page, title in self.PAGES)
        cfg = self.lmstudio_cfg()
        remote = '<div class="lms-msg" id="lms-remote"></div>' if cfg.get('remote') else ''
        cfg = json.dumps(cfg).replace('</', '<\\/')
        p(f'<style>{LMSTUDIO_CSS}</style>')
        p(f'<div class="lms"><nav class="lms-nav">{nav}</nav>{remote}{html}</div>')
        # eigener Scope, damit $ nicht jQuery überdeckt
        p(f'<script>(function() {{\nconst LMS = {cfg};\n{LMSTUDIO_JS}\n{js}\n}})();</script>')

    def part_index(self, p: b.Page, params={}):
        self.part_page(p, 'index', LMSTUDIO_STATUS_HTML, LMSTUDIO_STATUS_JS)

    def part_models(self, p: b.Page, params={}):
        self.part_page(p, 'models', LMSTUDIO_MODELS_HTML, LMSTUDIO_MODELS_JS)

    def part_chat(self, p: b.Page, params={}):
        self.part_page(p, 'chat', LMSTUDIO_CHAT_HTML, LMSTUDIO_CHAT_JS)

    def part_embeddings(self, p: b.Page, params={}):
        self.part_page(p, 'embeddings', LMSTUDIO_EMBED_HTML, LMSTUDIO_EMBED_JS)


class LMStudio(LMStudioUiMixin, BaseLLM):
    """
    Nutzt einen LM-Studio-Server (Standard http://localhost:1234):
    OpenAI-kompatible API (/v1/chat/completions, /v1/completions, /v1/embeddings) für Chat, Completion und
    Embeddings, native REST-API (/api/v1, ältere Versionen /api/v0) für Modelle, Laden, Entladen und Download.
    Das lms-CLI startet Daemon und Server, falls LM Studio lokal installiert ist.

    UI: Status/Server -> Modelle -> Chat -> Embeddings

    Konfiguration: LMSTUDIO_URL, LMSTUDIO_MODEL, LMSTUDIO_EMBEDDING_MODEL, LMSTUDIO_LMS (Pfad zu lms)
    """

    ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

    def __init__(self, type='lmstudio', url=None, model=None, embedding_model=None, lms=None, timeout=600,
                 args: u.Args = None):
        """
            :param url: Server-URL, Standard LMSTUDIO_URL oder http://localhost:1234
            :param model: Standard-Modell für prompt/chat, sonst das erste geladene LLM
            :param lms: Pfad zum lms-CLI, sonst LMSTUDIO_LMS, PATH oder ~/.lmstudio/bin
        """
        super().__init__(type)
        if args is None:
            args = u.Args()
        self.url = (url or args.get('LMSTUDIO_URL', 'http://localhost:1234')).rstrip('/')
        self.model = model or args.get('LMSTUDIO_MODEL', '')
        self.embedding_model = embedding_model or args.get('LMSTUDIO_EMBEDDING_MODEL', '')
        self.lms = lms or args.get('LMSTUDIO_LMS', None) or self.find_lms()
        self.timeout = float(timeout)
        self.define_vars('url', 'model', 'embedding_model', 'lms')
        self.define_lmstudio_sigs()

    # -- HTTP --

    def api(self, path, payload=None, method=None, timeout=None):
        method = method or ('POST' if payload is not None else 'GET')
        try:
            resp = requests.request(method, self.url + path, json=payload, timeout=timeout or self.timeout)
        except requests.exceptions.ConnectionError:
            raise LMStudioError(f'LM Studio unter {self.url} nicht erreichbar (Server gestartet?)')
        try:
            data = resp.json()
        except ValueError:
            data = {'text': resp.text}
        if resp.status_code >= 400:
            error = data.get('error', data.get('text', resp.status_code)) if isinstance(data, dict) else data
            if isinstance(error, dict):
                error = error.get('message', str(error))
            raise LMStudioError(f'{path}: {error}')
        return data

    def online(self):
        try:
            self.api('/v1/models', timeout=3)
            return True
        except Exception:
            return False

    # -- Modelle --

    @staticmethod
    def kind(model_type):
        """ llm, vlm, embedding(s) -> llm oder embedding """
        return 'embedding' if str(model_type).startswith('embedding') else 'llm'

    def models(self) -> list:
        """ Modelle der nativen API, einheitlich für /api/v1 und /api/v0 """
        try:
            data = self.api('/api/v1/models', timeout=10)
            res = []
            for m in data.get('models', []):
                q = m.get('quantization') or {}
                caps = m.get('capabilities') or {}
                instances = [i.get('id') for i in m.get('loaded_instances', [])]
                res.append({
                    'id': m.get('key'), 'type': m.get('type'), 'kind': self.kind(m.get('type')),
                    'name': m.get('display_name') or m.get('key'), 'publisher': m.get('publisher'),
                    'arch': m.get('architecture'), 'quantization': q.get('name') if isinstance(q, dict) else q,
                    'size_bytes': m.get('size_bytes'), 'params': m.get('params_string'),
                    'max_context_length': m.get('max_context_length'), 'format': m.get('format'),
                    'state': 'loaded' if instances else 'not-loaded', 'instances': instances,
                    'vision': bool(caps.get('vision')), 'tool_use': bool(caps.get('trained_for_tool_use')),
                    'contexts': [(i.get('config') or {}).get('context_length') for i in m.get('loaded_instances', [])]
                })
            return res
        except LMStudioError as e:
            if 'nicht erreichbar' in str(e):
                raise
        # ältere Versionen ohne /api/v1
        data = self.api('/api/v0/models', timeout=10)
        return [{
            'id': m.get('id'), 'type': m.get('type'), 'kind': self.kind(m.get('type')), 'name': m.get('id'),
            'publisher': m.get('publisher'), 'arch': m.get('arch'), 'quantization': m.get('quantization'),
            'size_bytes': None, 'params': None, 'max_context_length': m.get('max_context_length'),
            'format': m.get('compatibility_type'), 'state': m.get('state'),
            'instances': [m.get('id')] if m.get('state') == 'loaded' else [],
            'vision': m.get('type') == 'vlm', 'tool_use': 'tool_use' in m.get('capabilities', []), 'contexts': []
        } for m in data.get('data', [])]

    def default_model(self, kind='llm', model=''):
        """ Angegebenes, konfiguriertes oder erstes geladenes Modell der Art kind """
        if model:
            return model
        configured = self.embedding_model if kind == 'embedding' else self.model
        if configured:
            return configured
        models = [m for m in self.models() if m['kind'] == kind]
        for m in models:
            if m['instances']:
                return m['instances'][0]
        if models:
            # LM Studio lädt das Modell bei Bedarf (JIT)
            return models[0]['id']
        raise LMStudioError(f'Kein Modell vom Typ {kind} vorhanden')

    # -- Inferenz --

    @staticmethod
    def as_list(value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('['):
                return json.loads(value)
            return [value]
        return list(value)

    @staticmethod
    def number(data, key, cast, default):
        value = data.get(key, '')
        return default if value in ('', None) else cast(value)

    def options(self, data, max_tokens=-1):
        res = {'temperature': self.number(data, 'temperature', float, 0.7)}
        n = self.number(data, 'max_tokens', int, max_tokens)
        if n > 0:
            res['max_tokens'] = n
        return res

    def chat(self, messages: list, data={}) -> dict:
        model = self.default_model('llm', data.get('model', ''))
        payload = {'model': model, 'messages': messages, 'stream': False, **self.options(data)}
        result = self.api('/v1/chat/completions', payload)
        message = result['choices'][0]['message']
        content = message.get('content') or ''
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if not reasoning and '</think>' in content:
            reasoning, content = content.split('</think>', 1)
            reasoning = reasoning.replace('<think>', '')
        return {'response': remove_think(content).strip(), 'reasoning': reasoning.strip(),
                'model': result.get('model', model), 'usage': result.get('usage', {}),
                'stats': result.get('stats', {}), 'finish_reason': result['choices'][0].get('finish_reason')}

    def messages(self, data) -> list:
        if 'messages' in data and data['messages'] not in ('', None):
            messages = self.as_list(data['messages'])
        else:
            messages = [{'role': 'user', 'content': data['prompt']}]
        if data.get('system') and not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {'role': 'system', 'content': data['system']})
        if data.get('image'):
            image = data['image']
            if not image.startswith('data:'):
                image = 'data:image/jpeg;base64,' + image
            last = messages[-1]
            text = last['content'] if isinstance(last['content'], str) else ''
            last['content'] = [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': image}}]
        return messages

    def prompt(self, prompt, data={}):
        try:
            return self.success('ok', **self.chat(self.messages({**data, 'prompt': prompt, 'messages': ''}), data))
        except Exception as e:
            return self.fail(str(e))

    def execute(self, data):
        if 'op' in data:
            # nicht in BaseLLM.execute, dort hat prompt Vorrang vor op
            return r.BaseJobExecutor.execute(self, data)
        return super().execute(data)

    def execute_chat(self, data):
        try:
            self.count += 1
            return self.success('ok', **self.chat(self.messages(data), data))
        except Exception as e:
            return self.fail(str(e))

    def execute_completion(self, data):
        try:
            model = self.default_model('llm', data.get('model', ''))
            result = self.api('/v1/completions', {'model': model, 'prompt': data['prompt'], 'stream': False,
                                                  **self.options(data, 256)})
            return self.success('ok', response=result['choices'][0].get('text', ''), model=result.get('model', model),
                                usage=result.get('usage', {}))
        except Exception as e:
            return self.fail(str(e))

    def embed(self, texts, model=''):
        model = self.default_model('embedding', model)
        result = self.api('/v1/embeddings', {'model': model, 'input': texts})
        return model, [e['embedding'] for e in sorted(result['data'], key=lambda e: e.get('index', 0))]

    def execute_embed(self, data):
        try:
            texts = self.as_list(data['input'])
            model, embeddings = self.embed(texts, data.get('model', ''))
            return self.success('ok', model=model, embedding=embeddings[0] if embeddings else [],
                                embeddings=embeddings, dims=len(embeddings[0]) if embeddings else 0)
        except Exception as e:
            return self.fail(str(e))

    # -- Verwaltung --

    def execute_status(self, data={}):
        res = {'url': self.url, 'online': self.online(), 'cli': self.lms or '', 'model': self.model,
               'embedding_model': self.embedding_model, 'loaded': [], 'count': 0}
        if res['online']:
            try:
                models = self.models()
                res['count'] = len(models)
                res['loaded'] = [{'id': i, 'model': m['id'], 'kind': m['kind']}
                                 for m in models for i in m['instances']]
            except Exception as e:
                res['message'] = str(e)
        return self.success(**res)

    def execute_models(self, data={}):
        try:
            models = self.models()
            if data.get('kind'):
                models = [m for m in models if m['kind'] == data['kind']]
            return self.success(models=models)
        except Exception as e:
            return self.fail(str(e))

    def execute_load(self, data):
        model = data['model']
        context_length = self.number(data, 'context_length', int, 0)
        cli_only = any(data.get(k) not in ('', None, '0', 0) for k in ['gpu', 'ttl', 'identifier'])
        if not cli_only:
            try:
                payload = {'model': model}
                if context_length > 0:
                    payload['context_length'] = context_length
                result = self.api('/api/v1/models/load', payload)
                return self.success('Geladen', **result)
            except LMStudioError as e:
                if self.lms is None or 'nicht erreichbar' in str(e):
                    return self.fail(str(e))
        args = ['load', model, '-y']
        if context_length > 0:
            args += ['-c', str(context_length)]
        for key, opt in [('gpu', '--gpu'), ('ttl', '--ttl'), ('identifier', '--identifier')]:
            if data.get(key) not in ('', None, '0', 0):
                args += [opt, str(data[key])]
        return self.cli_result(args, instance_id=data.get('identifier') or model)

    def execute_unload(self, data):
        try:
            if str(data.get('all', 0)) in ('1', 'true', 'True'):
                ids = [i for m in self.models() for i in m['instances']]
            else:
                ids = [data['instance_id']]
            for instance_id in ids:
                try:
                    self.api('/api/v1/models/unload', {'instance_id': instance_id})
                except LMStudioError as e:
                    if self.lms is None or 'nicht erreichbar' in str(e):
                        raise
                    self.cli(['unload', instance_id])
            return self.success('Entladen', unloaded=ids)
        except Exception as e:
            return self.fail(str(e))

    def execute_download(self, data):
        try:
            payload = {'model': data['model']}
            if data.get('quantization'):
                payload['quantization'] = data['quantization']
            return self.success(**self.api('/api/v1/models/download', payload))
        except Exception as e:
            return self.fail(str(e))

    def execute_download_status(self, data):
        try:
            job_id = data['job_id']
            if not re.fullmatch(r'[\w-]+', str(job_id)):
                raise ValueError(f'Invalid job_id {job_id!r}')
            return self.success(**self.api('/api/v1/models/download/status/' + job_id, timeout=10))
        except Exception as e:
            return self.fail(str(e))

    # -- lms-CLI --

    @staticmethod
    def find_lms():
        import shutil
        path = shutil.which('lms')
        if path:
            return path
        for name in ['lms.exe', 'lms']:
            path = os.path.join(os.path.expanduser('~'), '.lmstudio', 'bin', name)
            if os.path.isfile(path):
                return path
        return None

    def cli(self, args: list, timeout=300) -> str:
        if self.lms is None:
            raise LMStudioError('lms-CLI nicht gefunden (LMSTUDIO_LMS setzen oder LM Studio installieren)')
        import subprocess
        proc = subprocess.run([self.lms, *args], capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout, stdin=subprocess.DEVNULL)
        output = self.ANSI.sub('', (proc.stdout or '') + (proc.stderr or '')).strip()
        if proc.returncode != 0:
            raise LMStudioError(output or f'lms {" ".join(args)}: Exit-Code {proc.returncode}')
        return output

    def cli_result(self, args, timeout=300, **kwargs):
        try:
            return self.success(output=self.cli(args, timeout), **kwargs)
        except Exception as e:
            return self.fail(str(e))

    def port(self):
        from urllib.parse import urlparse
        return urlparse(self.url).port or 1234

    def execute_server_start(self, data={}):
        output = []
        try:
            # startet LM Studio ohne Oberfläche, falls es nicht läuft
            output.append(self.cli(['daemon', 'up'], timeout=120))
        except Exception as e:
            output.append(str(e))
        port = self.number(data, 'port', int, 0) or self.port()
        args = ['server', 'start', '--port', str(port)]
        if data.get('bind'):
            args += ['--bind', data['bind']]
        if str(data.get('cors', 0)) in ('1', 'true', 'True'):
            args.append('--cors')
        try:
            output.append(self.cli(args, timeout=120))
            return self.success('Server gestartet', output='\n'.join(output), online=self.online())
        except Exception as e:
            return self.fail(str(e), output='\n'.join(output))

    def execute_server_stop(self, data={}):
        return self.cli_result(['server', 'stop'])

    def execute_daemon(self, data={}):
        action = data.get('action', 'up')
        if action not in ('up', 'down', 'status'):
            return self.fail(f'Unknown action {action!r}')
        return self.cli_result(['daemon', action], timeout=120)

    def execute_cli(self, data={}):
        cmd = data.get('cmd', 'status')
        if cmd not in self.CLI_COMMANDS:
            return self.fail(f'Unknown cmd {cmd!r}, erlaubt: ' + ', '.join(self.CLI_COMMANDS))
        return self.cli_result(self.CLI_COMMANDS[cmd], timeout=60)


LMSTUDIO_CSS = """
.lms { max-width: 1100px; }
.lms-nav { display: flex; gap: 6px; margin: 8px 0 14px; flex-wrap: wrap; }
.lms-nav a { padding: 6px 12px; border: 1px solid #999; border-radius: 16px; text-decoration: none; color: inherit;
  opacity: .7; }
.lms-nav a.active { opacity: 1; font-weight: bold; border-width: 2px; }
.lms-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
.lms-row > label:first-child { min-width: 130px; }
.lms input[type=number] { width: 90px; }
.lms textarea { width: 100%; box-sizing: border-box; font: inherit; }
.lms-msg { margin: 6px 0; min-height: 1.2em; }
.lms-msg.error { color: #c33; }
.lms-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #c33; margin-right: 6px; }
.lms-dot.on { background: #2a7; }
.lms-table { border-collapse: collapse; margin-top: 10px; }
.lms-table td, .lms-table th { border: 1px solid #8886; padding: 3px 8px; text-align: left; vertical-align: top; }
.lms-tag { font-size: 80%; border: 1px solid #8888; border-radius: 8px; padding: 0 5px; margin-right: 3px; }
.lms-loaded { color: #2a7; font-weight: bold; }
.lms-chat { border: 1px solid #8886; border-radius: 6px; padding: 8px; min-height: 120px; max-height: 60vh;
  overflow-y: auto; margin: 8px 0; }
.lms-m { margin: 6px 0; padding: 6px 10px; border-radius: 6px; white-space: pre-wrap; }
.lms-m.user { background: #4af2; }
.lms-m.assistant { background: #8882; }
.lms-m .lms-role { font-size: 80%; opacity: .7; display: block; }
.lms-m details { font-size: 90%; opacity: .8; }
.lms-m img { max-height: 120px; display: block; }
.lms pre { white-space: pre-wrap; max-height: 400px; overflow: auto; }
.lms-progress { height: 10px; background: #8884; border-radius: 5px; overflow: hidden; max-width: 500px; }
.lms-progress div { height: 100%; width: 0; background: #2a7; transition: width .2s; }
"""

LMSTUDIO_JS = r"""
const $ = (id) => document.getElementById(id);
function post(data) {
  const body = new URLSearchParams({ type: LMS.type, ...data });
  return fetch(location.protocol + '//' + location.host + '/', { method: 'POST', body: body }).then((r) => r.json());
}
// Remote: Job wird nur eingestellt (remote_guid), das Ergebnis per op=remote_result abholen
function job(data) {
  if (!LMS.remote) return post(data);
  return post({ ...data, async: 1 }).then((r) => {
    remoteStatus();
    return r.remote_guid ? remoteWait(r.remote_guid) : r;
  });
}
function remoteWait(guid) {
  return new Promise((ok) => setTimeout(ok, LMS.poll_ms)).then(() => post({ op: 'remote_result', guid: guid }))
    .then((r) => {
      remoteStatus();
      if (r.success && r.pending) return remoteWait(guid);
      return r.success ? r.result : r;
    });
}
function remoteStatus() {
  if (!LMS.remote) return;
  post({ op: 'remote_state' }).then((r) => {
    $('lms-remote').textContent = 'Remote: ' + r.status;
    $('lms-remote').classList.toggle('error', !!r.error);
  });
}
if (LMS.remote) { remoteStatus(); setInterval(remoteStatus, 5000); }
function msg(id, text, error) {
  const el = $(id);
  el.textContent = text || '';
  el.classList.toggle('error', !!error);
}
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}
function size(bytes) {
  if (!bytes) return '';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (bytes >= 1024 && i < u.length - 1) { bytes /= 1024; i++; }
  return bytes.toFixed(i > 1 ? 1 : 0) + ' ' + u[i];
}
// Modellauswahl füllen: geladene zuerst, kind = llm oder embedding
function fillModels(select, kind, preferred) {
  return job({ op: 'models', kind: kind }).then((r) => {
    select.innerHTML = '';
    if (!r.success) { select.add(new Option('- ' + r.message + ' -', '')); return r; }
    const models = r.models.slice().sort((a, b) => (b.instances.length > 0) - (a.instances.length > 0));
    models.forEach((m) => {
      const ids = m.instances.length ? m.instances : [m.id];
      ids.forEach((id) => select.add(new Option(id + (m.instances.length ? ' (geladen)' : ''), id)));
    });
    if (!models.length) select.add(new Option('- kein Modell -', ''));
    if (preferred) select.value = preferred;
    return r;
  });
}
"""

LMSTUDIO_STATUS_HTML = """
<h2>Server</h2>
<div class="lms-row"><label>URL</label><code id="lms-url"></code></div>
<div class="lms-row"><label>Status</label><span><span class="lms-dot" id="lms-dot"></span><span id="lms-state">...</span></span></div>
<div class="lms-row"><label>Geladene Modelle</label><span id="lms-loaded"></span></div>
<div class="lms-row"><label>lms-CLI</label><code id="lms-cli"></code></div>
<div class="lms-row"><label>Server</label>
  <button id="lms-start">Starten</button><button id="lms-stop">Stoppen</button>
  <label><input type="checkbox" id="lms-cors" /> CORS</label>
  <label><input type="checkbox" id="lms-lan" /> im Netzwerk (0.0.0.0)</label>
  <button id="lms-refresh">Aktualisieren</button></div>
<div class="lms-row"><label>LM Studio (Daemon)</label>
  <button data-daemon="up">Starten</button><button data-daemon="down">Beenden</button>
  <button data-daemon="status">Status</button></div>
<div class="lms-msg" id="lms-msg"></div>
<h2>CLI</h2>
<div class="lms-row">
  <button data-cli="status">lms status</button><button data-cli="version">version</button>
  <button data-cli="ps">ps</button><button data-cli="ls">ls</button>
  <button data-cli="runtime_ls">runtime ls</button><button data-cli="runtime_survey">runtime survey</button></div>
<pre id="lms-output"></pre>
"""

LMSTUDIO_STATUS_JS = r"""
(function() {
  function refresh() {
    job({ op: 'status' }).then((r) => {
      $('lms-url').textContent = r.url;
      $('lms-dot').classList.toggle('on', r.online);
      $('lms-state').textContent = r.online ? 'online, ' + r.count + ' Modelle' : 'nicht erreichbar';
      $('lms-loaded').innerHTML = r.loaded.length
        ? r.loaded.map((m) => '<span class="lms-tag">' + esc(m.kind) + '</span>' + esc(m.id)).join('<br />') : '-';
      $('lms-cli').textContent = r.cli || 'nicht gefunden';
      document.querySelectorAll('[data-cli],[data-daemon],#lms-start,#lms-stop').forEach((b) => b.disabled = !r.cli);
    });
  }
  function output(r) {
    msg('lms-msg', r.success ? (r.message || 'ok') : r.message, !r.success);
    $('lms-output').textContent = r.output || '';
    refresh();
  }
  $('lms-start').addEventListener('click', () => {
    msg('lms-msg', 'Starte LM Studio und Server ...');
    $('lms-output').textContent = '';
    job({ op: 'server_start', cors: $('lms-cors').checked ? 1 : 0, bind: $('lms-lan').checked ? '0.0.0.0' : '' }).then(output);
  });
  $('lms-stop').addEventListener('click', () => job({ op: 'server_stop' }).then(output));
  $('lms-refresh').addEventListener('click', refresh);
  document.querySelectorAll('[data-daemon]').forEach((b) => b.addEventListener('click', () => {
    msg('lms-msg', 'lms daemon ' + b.dataset.daemon + ' ...');
    job({ op: 'daemon', action: b.dataset.daemon }).then(output);
  }));
  document.querySelectorAll('[data-cli]').forEach((b) => b.addEventListener('click', () => {
    $('lms-output').textContent = '...';
    job({ op: 'cli', cmd: b.dataset.cli }).then((r) => { $('lms-output').textContent = r.success ? r.output : r.message; });
  }));
  refresh();
})();
"""

LMSTUDIO_MODELS_HTML = """
<h2>Modelle</h2>
<div class="lms-row"><label>Filter</label>
  <select id="lms-kind"><option value="">alle</option><option value="llm">LLM</option>
    <option value="embedding">Embedding</option></select>
  <button id="lms-refresh">Aktualisieren</button><button id="lms-unload-all">Alle entladen</button></div>
<div class="lms-row"><label>Laden mit</label>
  Kontext <input type="number" id="lms-ctx" min="0" step="1024" value="0" title="0 = Standard" />
  GPU <select id="lms-gpu" title="Über das lms-CLI"><option value="">automatisch</option><option value="max">max</option>
    <option value="0.5">50 %</option><option value="off">aus</option></select>
  TTL (s) <input type="number" id="lms-ttl" min="0" value="0" title="Über das lms-CLI, entlädt nach Leerlauf" /></div>
<div class="lms-msg" id="lms-msg"></div>
<table class="lms-table" id="lms-table">
  <thead><tr><th>Modell</th><th>Typ</th><th>Parameter</th><th>Quant.</th><th>Größe</th><th>max. Kontext</th>
    <th>Status</th><th></th></tr></thead>
  <tbody></tbody>
</table>
<h2>Herunterladen</h2>
<div class="lms-row"><label for="lms-dl">Modell</label>
  <input type="text" id="lms-dl" style="width: 360px" placeholder="qwen/qwen3-4b oder Hugging-Face-URL" />
  <input type="text" id="lms-dl-q" style="width: 90px" placeholder="Quant." title="optional, z.B. q4_k_m" />
  <button id="lms-dl-btn">Herunterladen</button></div>
<div class="lms-progress"><div id="lms-dl-progress"></div></div>
<div class="lms-msg" id="lms-dl-msg"></div>
"""

LMSTUDIO_MODELS_JS = r"""
(function() {
  const tbody = document.querySelector('#lms-table tbody');
  // keep: Meldung einer Aktion stehen lassen
  function refresh(keep) {
    job({ op: 'models', kind: $('lms-kind').value }).then((r) => {
      if (!r.success) { msg('lms-msg', r.message, true); return; }
      tbody.innerHTML = '';
      r.models.forEach((m) => {
        const tr = document.createElement('tr');
        const tags = (m.vision ? '<span class="lms-tag">Vision</span>' : '') + (m.tool_use ? '<span class="lms-tag">Tools</span>' : '');
        const state = m.instances.length
          ? '<span class="lms-loaded">geladen</span><br />' + m.instances.map((id, i) =>
              esc(id) + (m.contexts[i] ? ' (' + m.contexts[i] + ')' : '') +
              ' <button data-unload="' + esc(id) + '">Entladen</button>').join('<br />')
          : 'nicht geladen';
        tr.innerHTML = '<td><b>' + esc(m.name) + '</b><br /><code>' + esc(m.id) + '</code></td>' +
          '<td>' + esc(m.type) + '<br />' + tags + '</td><td>' + esc(m.params || '') + '</td>' +
          '<td>' + esc(m.quantization || '') + '</td><td>' + size(m.size_bytes) + '</td>' +
          '<td>' + esc(m.max_context_length || '') + '</td><td>' + state + '</td>' +
          '<td><button data-load="' + esc(m.id) + '">Laden</button></td>';
        tbody.appendChild(tr);
      });
      if (keep !== true) msg('lms-msg', r.models.length + ' Modelle');
    });
  }
  tbody.addEventListener('click', (e) => {
    const b = e.target;
    if (b.dataset.load) {
      msg('lms-msg', 'Lade ' + b.dataset.load + ' ...');
      b.disabled = true;
      job({ op: 'load', model: b.dataset.load, context_length: $('lms-ctx').value, gpu: $('lms-gpu').value,
            ttl: $('lms-ttl').value }).then((r) => {
        msg('lms-msg', r.success ? 'Geladen: ' + (r.instance_id || b.dataset.load) +
          (r.load_time_seconds ? ' in ' + r.load_time_seconds + ' s' : '') : r.message, !r.success);
        refresh(true);
      });
    } else if (b.dataset.unload) {
      job({ op: 'unload', instance_id: b.dataset.unload }).then((r) => {
        msg('lms-msg', r.success ? 'Entladen: ' + r.unloaded.join(', ') : r.message, !r.success);
        refresh(true);
      });
    }
  });
  $('lms-unload-all').addEventListener('click', () => job({ op: 'unload', all: 1 }).then((r) => {
    msg('lms-msg', r.success ? 'Entladen: ' + (r.unloaded.join(', ') || '-') : r.message, !r.success);
    refresh(true);
  }));
  $('lms-kind').addEventListener('change', () => refresh());
  $('lms-refresh').addEventListener('click', () => refresh());

  function poll(jobId) {
    job({ op: 'download_status', job_id: jobId }).then((r) => {
      if (!r.success) { msg('lms-dl-msg', r.message, true); return; }
      const total = r.total_size_bytes || 0, done = r.downloaded_bytes || 0;
      $('lms-dl-progress').style.width = (total ? 100 * done / total : 0) + '%';
      msg('lms-dl-msg', r.status + ': ' + size(done) + ' / ' + size(total), r.status === 'failed');
      if (r.status === 'downloading' || r.status === 'paused') setTimeout(() => poll(jobId), 1500);
      else refresh();
    });
  }
  $('lms-dl-btn').addEventListener('click', () => {
    msg('lms-dl-msg', 'Starte Download ...');
    job({ op: 'download', model: $('lms-dl').value, quantization: $('lms-dl-q').value }).then((r) => {
      if (!r.success) { msg('lms-dl-msg', r.message, true); return; }
      if (r.job_id) poll(r.job_id); else msg('lms-dl-msg', r.status || 'ok');
    });
  });
  refresh();
})();
"""

LMSTUDIO_CHAT_HTML = """
<h2>Chat</h2>
<div class="lms-row"><label for="lms-model">Modell</label><select id="lms-model"></select>
  <label><input type="radio" name="lms-mode" value="chat" checked /> Chat</label>
  <label><input type="radio" name="lms-mode" value="completion" /> Completion (ohne Chat-Template)</label></div>
<div class="lms-row"><label for="lms-system">Systemprompt</label>
  <textarea id="lms-system" rows="2" style="flex: 1" placeholder="optional"></textarea></div>
<div class="lms-row"><label>Parameter</label>
  temperature <input type="number" id="lms-temp" min="0" max="2" step="0.1" value="0.7" />
  max_tokens <input type="number" id="lms-max" min="-1" step="1" value="-1" title="-1 = unbegrenzt" />
  <label><input type="checkbox" id="lms-history" checked /> Verlauf mitschicken</label></div>
<div class="lms-chat" id="lms-chat"></div>
<textarea id="lms-prompt" rows="4" placeholder="Nachricht (Strg+Enter sendet)"></textarea>
<div class="lms-row"><button id="lms-send">Senden</button><button id="lms-clear">Neuer Chat</button>
  <label>Bild <input type="file" id="lms-image" accept="image/*" title="Für Vision-Modelle" /></label>
  <span id="lms-stats"></span></div>
<div class="lms-msg" id="lms-msg"></div>
"""

LMSTUDIO_CHAT_JS = r"""
(function() {
  let history = [];
  fillModels($('lms-model'), 'llm', LMS.model);
  const mode = () => document.querySelector('[name=lms-mode]:checked').value;
  function add(role, text, extra) {
    const div = document.createElement('div');
    div.className = 'lms-m ' + role;
    div.innerHTML = '<span class="lms-role">' + esc(role) + '</span>' + (extra || '') + esc(text);
    $('lms-chat').appendChild(div);
    $('lms-chat').scrollTop = $('lms-chat').scrollHeight;
    return div;
  }
  function readImage() {
    const file = $('lms-image').files[0];
    if (!file) return Promise.resolve('');
    return new Promise((ok) => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.readAsDataURL(file); });
  }
  function send() {
    const text = $('lms-prompt').value;
    if (!text.trim()) return;
    $('lms-send').disabled = true;
    readImage().then((image) => {
      add('user', text, image ? '<img src="' + image + '" alt="" />' : '');
      const params = { model: $('lms-model').value, temperature: $('lms-temp').value, max_tokens: $('lms-max').value };
      let data;
      if (mode() === 'completion') {
        data = { op: 'completion', prompt: text, ...params };
      } else {
        const messages = ($('lms-history').checked ? history : []).concat([{ role: 'user', content: text }]);
        data = { op: 'chat', messages: JSON.stringify(messages), system: $('lms-system').value, image: image, ...params };
      }
      $('lms-prompt').value = '';
      $('lms-image').value = '';
      msg('lms-msg', 'Generiere ...');
      const t0 = Date.now();
      return job(data).then((r) => {
        if (!r.success) { msg('lms-msg', r.message, true); return; }
        const think = r.reasoning ? '<details><summary>Reasoning</summary>' + esc(r.reasoning) + '</details>' : '';
        add('assistant', r.response, think);
        history.push({ role: 'user', content: text }, { role: 'assistant', content: r.response });
        const u = r.usage || {}, s = r.stats || {};
        $('lms-stats').textContent = (r.model || '') + ' | ' + (u.prompt_tokens || 0) + ' + ' + (u.completion_tokens || 0) +
          ' Tokens | ' + (s.tokens_per_second ? s.tokens_per_second.toFixed(1) + ' Tokens/s | ' : '') +
          ((Date.now() - t0) / 1000).toFixed(1) + ' s';
        msg('lms-msg', r.finish_reason === 'length' ? 'Abgebrochen: max_tokens erreicht' : '');
      });
    }).finally(() => { $('lms-send').disabled = false; });
  }
  $('lms-send').addEventListener('click', send);
  $('lms-prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.ctrlKey) send(); });
  $('lms-clear').addEventListener('click', () => { history = []; $('lms-chat').innerHTML = ''; $('lms-stats').textContent = ''; });
})();
"""

LMSTUDIO_EMBED_HTML = """
<h2>Embeddings</h2>
<div class="lms-row"><label for="lms-model">Modell</label><select id="lms-model"></select></div>
<textarea id="lms-input" rows="6" placeholder="Ein Text pro Zeile"></textarea>
<div class="lms-row"><button id="lms-embed">Embeddings berechnen</button>
  <button id="lms-copy" disabled>JSON kopieren</button></div>
<div class="lms-msg" id="lms-msg"></div>
<div id="lms-sim"></div>
<pre id="lms-vectors"></pre>
"""

LMSTUDIO_EMBED_JS = r"""
(function() {
  let last = null;
  fillModels($('lms-model'), 'embedding', LMS.embedding_model);
  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb) || 1);
  }
  $('lms-embed').addEventListener('click', () => {
    const texts = $('lms-input').value.split('\n').filter((t) => t.trim());
    if (!texts.length) { msg('lms-msg', 'Bitte Text eingeben', true); return; }
    msg('lms-msg', 'Berechne ...');
    job({ op: 'embed', model: $('lms-model').value, input: JSON.stringify(texts) }).then((r) => {
      if (!r.success) { msg('lms-msg', r.message, true); return; }
      last = r.embeddings;
      $('lms-copy').disabled = false;
      msg('lms-msg', r.embeddings.length + ' Vektoren mit ' + r.dims + ' Dimensionen (' + r.model + ')');
      $('lms-vectors').textContent = r.embeddings.map((e, i) => texts[i].slice(0, 40) + ': [' +
        e.slice(0, 8).map((v) => v.toFixed(4)).join(', ') + ', ...]').join('\n');
      if (texts.length > 1) {
        let html = '<table class="lms-table"><tr><th>Kosinus-Ähnlichkeit</th>' +
          texts.map((t, i) => '<th>' + (i + 1) + '</th>').join('') + '</tr>';
        texts.forEach((t, i) => {
          html += '<tr><th>' + (i + 1) + '. ' + esc(t.slice(0, 40)) + '</th>' +
            r.embeddings.map((e) => '<td>' + cosine(r.embeddings[i], e).toFixed(3) + '</td>').join('') + '</tr>';
        });
        $('lms-sim').innerHTML = html + '</table>';
      } else {
        $('lms-sim').innerHTML = '';
      }
    });
  });
  $('lms-copy').addEventListener('click', () => navigator.clipboard.writeText(JSON.stringify(last))
    .then(() => msg('lms-msg', 'Kopiert')));
})();
"""


class CohereLlm(BaseLLM):
    """
    Schickt Prompts an die Cohere-Chat-API.
    """

    MODULES = ['cohere']

    def __init__(self, api_key=None, args:u.Args=None):
        super().__init__('cohere')
        self.model = 'command'
        self.last_request = 0
        self.define_vars('model', 'last_request')
        if args is None:
            args = u.Args()
        import cohere
        self.cohere = cohere
        self.co = cohere.Client(api_key=args.get('COHERE_API_KEY', api_key))

    def prompt(self, prompt, data={}):
        if time.time() - self.last_request < 2:
            time.sleep(2)
        self.last_request = time.time()
        resp = self.co.chat(message=prompt, model=self.model)
        return self.success('ok', response=resp.text)


class Gemini(BaseLLM):
    """
    Schickt Prompts an Google Gemini (gemini-2.5-flash).
    """

    MODULES = ['google-genai']

    def __init__(self, api_key=None, args:u.Args=None):
        super().__init__('gemini')
        from google import genai
        if api_key is None:
            if args is None:
                args = u.Args()
            api_key = args['gemini']
        os.environ["GEMINI_API_KEY"] = str(api_key)
        #genai.configure(api_key=api_key)
        self.client = genai.Client(api_key=api_key)

    def prompt(self, prompt, data={}):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return self.success('ok', response=response.text)


class TransformText(nlp.TextExecutor):
    """
      Template für einen Prompt der auf dem llm_type ausgefuehrt wird
    """

    TAGS = [r.TAG.TEXT_TRANSFORM]

    generation_strings = [
        'Ja gerne, '
    ]
    
    def __init__(self, type=None, llm_type='llm', pre='', post=''):
        super().__init__()
        if type is not None:
            self.type = type
        self.define_vars('llm_type', 'pre', 'post')
        self.llm_type = llm_type
        self.post = post
        self.pre = pre
        self.text_result("Formt den Text per LLM-Prompt (pre + text + post) um", response='str', value='str')

    def remove_generation_string(self, response: str):
        # TODO remove LLM "Hier ist der Text"
        for s in self.generation_strings:
            if response.startswith(s):
                response = response[len(s):]
                break
        return response

    def execute_text(self, text, data={}):
        d = self.getParentClass(r.LazyDispatcher)
        prompt = self.pre + text + self.post
        resp = d(type=self.llm_type, prompt=prompt)
        resp['response'] = self.remove_generation_string(resp['response'])
        resp['value'] = resp['response']
        return resp


class Tool:
    @staticmethod
    def add_two_numbers(a: int, b: int) -> int:
        """
        Add two numbers

        Args:
          a: The first integer number
          b: The second integer number

        Returns:
          int: The sum of the two numbers
        """
        print("Tool Call")
        return a + b

    def run(self):
        # via https://ollama.com/blog/functions-as-tools
        import ollama
        response = ollama.chat(
            'qwen3:14b',
            messages=[{'role': 'user', 'content': 'What is Paris?'}],
            tools=[self.add_two_numbers],  # Actual function reference
        )
        print(response)
        # response enthält tool_calls=None oder array
        # ToolCall(function=Function(name='add_two_numbers', arguments={'a': 10, 'b': 10}))]




def remove_think(text):
    """
    Removes everything between the markers ' Start' and '' in a string.
    Parameters:
    - text: The input string to process.
    Returns:
    - The processed string with the content between the markers removed.
    """
    start_marker = "[THINK]"
    end_marker = "[/THINK]"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    if start_index != -1 and end_index != -1:
        return text[:start_index] + text[end_index + len(end_marker):]
    else:
        return text


def query(image_path, prompt, model="mistralai/magistral-small-2509", api_key=None):
    """
    Query an OpenAI model with an image and a text prompt.
    Parameters:
    - api_key: Your OpenAI API key.
    - image_path: Path to the image file you want to send.
    - prompt: The text prompt to accompany the image.
    - model: The model to use (default is gpt-4-vision-preview).
    Returns:
    - The response from the OpenAI API as a JSON object.
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 900
    }
    try:
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        choice = data['choices'][0]
        message = choice['message']
        choice['message']['content'] = remove_think(message['content']).strip()
        data['choices'][0] = choice
        data['response_text'] = choice['message']['content']
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# query('/mnt/c/Users/bsalg/Desktop/QwenImageEdit/1.png', "Beschreibe das Bild")
# query('/mnt/c/Users/bsalg/Desktop/QwenImageEdit/1.png', "Beschreibe das Bild! Antworte als JSON mit {\"Beschreibung\": \"...\"} ")


class Vision(r.BaseJobExecutor):
    """
    Lässt Bilder aus nweb von einem Vision-Modell beschreiben und speichert die Beschreibung als Metadaten.
    """
    meta_ns = 'ai'
    meta_name = 'description_de'
    limit = 10
    prompt = "Beschreibe das Bild! Antworte als JSON mit {\"Beschreibung\": \"...\"} "

    def __init__(self):
        super().__init__('vision')
        self.define_vars('meta_ns', 'meta_name', 'limit', 'prompt')
        self.define_sig(PStr('op', 'map'))\
            .set_description("Beschreibt nweb-Bilder per Vision-LLM und speichert das Ergebnis als Meta-Attribut")\
            .r_success()

    def execute_map(self, data):
        self.nweb_map()
        return self.success('ok')

    def mapFn(self, doc, n):
        filename = 'current.jpg'
        doc.downloadThumbnail(file=filename, size='m')
        r = query(filename, self.prompt)
        t = r['response_text']
        response = json.loads(t.replace('```json', '').replace('```', ''))
        return response['Beschreibung']

    def nweb_map(self):
        from nwebclient import NWebClient
        nc = NWebClient(None)
        nc.mapDocMeta(self.meta_ns, self.meta_name, filterArgs='kind=image', limit=self.limit, update=True, mapFunction=self.mapFn)
