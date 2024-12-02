import requests
import json

from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import web as w
from nwebclient import util as u
from nwebclient import crypt
from nwebclient.nlp import ParagraphParser


class LLM(r.BaseJobExecutor):

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
        if 'cpromt' in data:
            return self.cprompt(data)
        return super().execute(data)
