import os

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
from nwebclient.dev import Param

from nxml import nlp


class BaseLLM(r.BaseJobExecutor):

    def __init__(self, type='llm'):
        super().__init__()
        self.type = type
        self.count = 0
        self.define_vars('count')
        self.define_sig(Param('prompt', 'str'))
        self.define_sig(Param('cprompt', 'str'))

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

    def page_index(self, params={}):
        p = b.Page(owner=self)
        p.input('prompt', id='prompt')
        p(w.button_js("Prompt", 'exec_job_p({"type": "' + self.type + '", "prompt": "#prompt"})'))
        p.div('', id='result')
        return p.nxui()


class OpenAiLLM(BaseLLM):

    MODULES = ['openai']

    MODELS = ['gpt-4o', 'gpt-4o-mini', 'gpt-4']

    def __init__(self, api_key=None, args: u.Args = None):
        super().__init__('gptllm')
        self.model = "gpt-4o"
        self.last_request = 0
        self.define_vars('model', 'last_request')
        if args is None:
            args = u.Args()
        self.key = args.get('OPENAI_KEY')

    def prompt(self, prompt, data):
        self.last_request = time.time()
        from openai import OpenAI
        client = OpenAI(api_key=self.key)

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


class CohereLlm(BaseLLM):

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
