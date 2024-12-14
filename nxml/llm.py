import requests
import json
import docker
import time

from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import web as w
from nwebclient import util as u
from nwebclient import crypt
from nwebclient.nlp import ParagraphParser


class RLLM(r.BaseJobExecutor):

    def __init__(self, type='rllm', args: u.Args = None):
        super().__init__()
        if args is None:
            args = u.Args()
        self.type = type
        self.args = args

    def remote_prompt(self, data):
        prompt = data['prompt']
        url = self.args.get('LLM_URL')
        pw = self.args.get('NPY_KEY', '')
        cprompt = crypt.encrypt_message(prompt, pw)
        resp = requests.post(url, {
            'cprompt': cprompt
        })
        result = json.loads(resp.text)
        try:
            text = crypt.decrypt_message(result['response'], pw)
            return self.success('ok', response=text)
        except Exception as e:
            return self.fail(str(e), response_text=resp.text)

    def execute(self, data):
        if 'prompt' in data:
            return self.remote_prompt(data)
        return super().execute(data)

    def page_index(self, params={}):
        p = b.Page(owner=self)
        p.input('prompt', id='prompt')
        p(w.button_js("Prompt", 'exec_job_p({"type": "' + self.type + '", "prompt": "#prompt"})'))
        p.div('', id='result')
        return p.nxui()


class OLLama(r.BaseJobExecutor):

    MODULES = ['ollama']

    def __init__(self, type='ollm', args: u.Args = None):
        super().__init__()
        self.type = type
        import ollama as o
        self.ollama = o

    def remote_prompt(self, data):
        response = self.ollama.generate(model='llama3', prompt=data['prompt'])
        return response

    def cprompt(self, data:dict):
        from nwebclient import crypt
        args = u.Args()
        pw = args.get('NPY_KEY', 'xxx')
        result = self.remote_prompt(crypt.decrypt_message(data['cprompt'], pw))
        return self.success('ok', response=crypt.encrypt_message(result['response'], pw))

    def execute(self, data):
        if 'prompt' in data:
            return self.remote_prompt(data)
        if 'cprompt' in data:
            return self.cprompt(data)
        return super().execute(data)


class OLLamaDockerd(r.BaseJobExecutor):

    def __init__(self, type='ollm'):
        super().__init__()
        self.type = type
        self.docker = docker.from_env()
        self.inner = OLLama()
        if self.exists():
            self.start()
        else:
            # docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
            self.container = self.docker.run("ollama/ollama", detach=True, name='ollama', port=11434)
            self.delayed(30, self.docker_load)

    def docker_load(self):
        self.container.exec_run('ollama run llama3.2', detach=True)

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


class CohereLlm(r.BaseJobExecutor):

    MODULES = ['cohere']

    def __init__(self, api_key=None, args:u.Args={}):
        super().__init__()
        self.type = 'cohere'
        self.last_request = 0
        import cohere
        self.cohere = cohere
        self.co = cohere.Client(
            api_key=args.get('COHERE_API_KEY', api_key),
        )

    def prompt(self, prompt):
        if time.time() - self.last_request < 2:
            time.sleep(2)
        self.last_request = time.time()
        return str(self.co.chat(
            message=prompt,
            model="command"
        ))

    def execute(self, data):
        if 'prompt' in data:
            return self.prompt(data)
        #if 'cprompt' in data:
        #    return self.cprompt(data)
        return super().execute(data)
