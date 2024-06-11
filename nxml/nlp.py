
import random

from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient.nlp import ParagraphParser


class DynamicPrompt:
    """
    {{firstname()}}   {{oneof("sagt", "schreibt")}}
    """
    def __init__(self):
        from nwebclient import llm
        self.data = llm
        from jinja2.nativetypes import NativeEnvironment
        self.env = NativeEnvironment()
        self.env.globals.update(firstname=self.firstname)
        self.env.globals.update(oneof=self.oneof)
        # fristname lastname city oneof

    def firstname(self):
        return random.choice([random.choice(self.data.de_firstname_male), random.choice(self.data.de_firstname_female)])

    def oneof(self, *args):
        return random.choice(args)

    def __call__(self, prompt, *args, **kwargs):
        t = self.env.from_string(prompt)
        return t.render()


class TextToText(r.BaseJobExecutor):
    """
    ramsrigouthamg/t5-large-paraphraser-diverse-high-quality
    """

    DE_GRAMMAR = "MRNH/mbart-german-grammar-corrector"
    DE_PARAPHRASE = "Lelon/t5-german-paraphraser-large"

    type = 'ttt'
    model_name = ''
    def __init__(self, model="Lelon/t5-german-paraphraser-large"):
        from transformers import pipeline
        self.model_name = model
        self.pipe = pipeline("text2text-generation", model="Lelon/t5-german-paraphraser-large")

    def generate(self, text):
        # pipe liefert [{'generated_text': 'Ich kann heute nicht pünktlich sein.'}]
        return self.pipe(text)[0]

    def execute(self, data):
        if 'text' in data:
            return self.generate(data['text'])
        return super().execute(data)

    def page(self, params):
        p = b.Page(owner=self)
        p.h2("Text2Text")
        p('<form>')
        p('<input type="hidden" name="type" value="' + self.type + '" />')
        p('<input type="text" name="prompt" value="" />')
        p('<input type="submit" name="submit" value="Execute" />')
        p('</form>')
        if 'prompt' in params:
            r = self.generate(params['prompt'])
            p.div(r['generated_text'])
        return p.nxui()



class QuestionAnswering(r.BaseJobExecutor):

    type = 'qa'

    pipe = None

    def __init__(self, model="deepset/gelectra-base-germanquad"):
        from transformers import pipeline
        self.pipe = pipeline("question-answering", model=model)

    def answer(self, question, context):
        return self.pipe(question=question, context=question)

    def execute(self, data):
        if 'question' in data:
            # {'score': 0.2509937584400177, 'start': 19, 'end': 63, 'answer': 'Kann ich am 13.2 an einer Führung teilnhmen?'}
            return self.answer(data['question'], data.get('context', ''))
        return super().execute(data)

    def page(self, params={}):
        p = b.Page(owner=self)
        return p.nxui()


class TextClassifier(r.BaseJobExecutor):
    """
    https://huggingface.co/facebook/bart-large-mnli - 1.7GB

    t.run_group({'group':'ds_sus', 'classes': ['Führungsanfrage', 'Mail', 'Absage', 'Presseanfrage', 'Sonstiges']})
    """
    def __init__(self):
        from transformers import pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify(self, text, classes):
        return self.classifier(text, classes, multi_label=True)

    def run_group(self, **data):
        nc = NWebClient(data.get('nweb', None))
        classes = data['classes']
        if isinstance(classes, str):
            classes = classes.split(',')
        for doc in nc.group(data['group']):
            result = self.classify(doc.content(), classes)
            print(f"{doc.title()}: {result['labels'][0]} {result['scores'][0]}")
        return {}

    def execute(self, data):
        if 'classify' in data:
            return self.classify(data['classify'], data.get('classes', []))
        if 'group' in data:
            return self.run_group(**data)
        return super().execute(data)


class TextWorker(r.BaseJobExecutor):

    sentence_runner = {}

    def __init__(self, sentence_runner={}):
        self.sentence_runner = sentence_runner
        if 'de' == sentence_runner:
            self.sentence_runner = {
                'grammar': TextToText(TextToText.DE_GRAMMAR),
                'paraphrase': TextToText(TextToText.DE_PARAPHRASE)
            }
    def execute(self, data):
        if 'text' in data:
            return self.process_text(data['text'], data)
        if 'sentence' in data:
            return self.process_sentence(data['text'], data)
        return super().execute(data)

    def process_text(self, text, data={}):
        result = []
        p = ParagraphParser(text)
        for sentence in p.paragraphs():
            result.append(self.process_sentence(sentence))
        return {'result': result}

    def process_sentence(self, text, data={}):
        result = {
            'orginal_text': text
        }
        for srk in self.sentence_runner.keys():
            result[srk] = self.sentence_runner[srk].execute({'text': text})
        return result

    def format_html(self, data):
        p = b.Page()
        if 'result' in data:
            p('<table>')
            for item in data['result']:
                p('<tr>')
                p('<td>'+item['orginal_text']+'</td>')
                p('<td>')
                if 'paraphrase' in item:
                    p(item['paraphrase'])
                p('</td>')
                p('</tr>')
            p('<table>')
        return str(p)

    def page(self, params={}):
        p = b.Page(owner=self)
        p.h1("Text Worker")
        p('<form>')
        p('<input type="hidden" name="type" value="' + self.type + '" />')
        p('<textarea name="text"></textarea>')
        p('<input type="submit" name="submit" value="Execute" />')
        p('</form>')
        if 'text' in params:
            res = self.process_text(params['prompt'])
            p.div(self.format_html(res))
        return p.nxui()

