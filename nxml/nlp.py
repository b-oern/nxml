import json
import random
import uuid

from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import web as w
from nwebclient import util as u
from nwebclient.nlp import ParagraphParser


def text_input_form(runner, p: b.Page, text='', caption="Execute",
                    t='textarea', input_name='text', html=''):
    hid = str(uuid.uuid4()).replace('-', '')
    p('<form method="POST" class="text_input_form" id="'+hid+'">')
    p('<input type="hidden" name="type" value="' + getattr(runner, 'type','') + '" />')
    if t == 'input':
        p('<input type="text" name="'+input_name+'" value="' + text + '" />')
    else:
        p('<textarea name="'+input_name+'">'+text+'</textarea>')
        p('<button>Paste</button>')
    p(html)
    p('<input type="submit" name="submit" value="'+caption+'" />')
    p('</form>')
    p.script(w.js_ready(""" 
        const hid = '""" + hid + """ ';
        const pasteButton = document.querySelector('#'+hid+' button');
        pasteButton.addEventListener('click', async () = > {
            try {
                const text = await navigator.clipboard.readText()
                document.querySelector(hid).value += text;
                console.log('Text pasted.');
            } catch (error) {
                console.log('Failed to read clipboard');
            }
        });
    """))


class DynamicPrompt:
    """
    {{firstname()}}   {{oneof("sagt", "schreibt")}}
    """
    def __init__(self):
        super().__init__()
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


class TextExecutor(r.BaseJobExecutor):
    """
        Executor zur Textarbeit
    """

    def __init__(self):
        super().__init__()

    def execute_text(self, text, data={}):
        return {'success': False, 'message': "TextExecutor::execute_text() not overwritten"}

    def remove_html(self,text):
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "lxml").text

    def execute_doc(self, document_id, data={}):
        nw = NWebClient(None)
        d = nw.doc(document_id)
        text = d.content()
        if d.is_kind('markup'):
            text = self.remove_html(text)
        return self.execute_text(text, data)


    def get_title(self):
        return getattr(self, 'title', 'Text')

    def execute(self, data):
        if 'text' in data:
            return self.execute_text(data['text'], data)
        if 'document_id' in data:
            return self.execute_doc(data['document_id'], data)
        return super().execute(data)

    def format_html(self, data):
        return '<pre>' + json.dumps(data) + '</pre>'

    def part_nweb_docs(self, p:b.Page, params={}):
        try:
            if 'document_id' in params:
                r = self.execute_doc(params['document_id'])
                p.div(self.format_html(r))
            nw = NWebClient(None)
            # TODO suchbox
            docs = nw.docs('kind=markup&limit=20&orderby=changed')
            p('<div class="nweb_docs nweb_doc_select">')
            for doc in docs:
                p.div(w.a(doc.title(), '?type' + self.type + '&document_id=' + str(doc.id())), _class='.doc')
            p('</div>')
        except Exception as e:
            p.div("Error: " + str(e))

    def page(self, params={}):
        p = b.Page(owner=self)
        if 'a' in params:
            pass
            # TODO dispatch
        p('<div class="TextExecutor runner_page">')
        p.h2(self.get_title())
        text_input_form(self, p, t='textarea', input_name='text', text=params.get('text', ''))
        if 'text' in params:
            r = self.execute_text(params['text'])
            p.div(self.format_html(r))
        self.part_nweb_docs(p, params)
        p('</div>')
        return p.nxui()


class TextToText(TextExecutor):
    """
    ramsrigouthamg/t5-large-paraphraser-diverse-high-quality
    """

    DE_GRAMMAR = "MRNH/mbart-german-grammar-corrector"
    DE_PARAPHRASE = "Lelon/t5-german-paraphraser-large"

    type = 'ttt'
    model_name = ''
    title = 'Text2Text'
    def __init__(self, model="Lelon/t5-german-paraphraser-large"):
        super().__init__()
        from transformers import pipeline
        self.param_names['text'] = "Texteingabe"
        self.model_name = model
        self.pipe = pipeline("text2text-generation", model="Lelon/t5-german-paraphraser-large")

    def generate(self, text):
        # pipe liefert [{'generated_text': 'Ich kann heute nicht pünktlich sein.'}]
        return self.pipe(text)[0]

    def execute_text(self, text, data):
        return self.generate(text)

    def page(self, params):
        p = b.Page(owner=self)
        p.h2("Text2Text")
        text_input_form(self, p, t='input', input_name='prompt')
        if 'prompt' in params:
            r = self.generate(params['prompt'])
            p.div(r['generated_text'])
        return p.nxui()



class QuestionAnswering(r.BaseJobExecutor):

    type = 'qa'

    pipe = None

    def __init__(self, model="deepset/gelectra-base-germanquad", context=''):
        super().__init__()
        from transformers import pipeline
        self.pipe = pipeline("question-answering", model=model)
        self.context = context
        self.define_vars('context')

    def answer(self, question, context):
        return self.pipe(question=question, context=question)

    def execute(self, data):
        if 'question' in data:
            # {'score': 0.2509937584400177, 'start': 19, 'end': 63, 'answer': 'Kann ich am 13.2 an einer Führung teilnhmen?'}
            return self.answer(data['question'], data.get('context', ''))
        return super().execute(data)

    def page_index(self, params={}):
        p = b.Page(owner=self)

        p.form_input('question', "Question", id='question')
        p.form_input('context', "Context", id='context')
        p(self.action_btn_parametric("Frage beantworten", type=self.type, question='#question', context='#context'))

        p.pre('', id='result')

        return p.nxui()


class TextClassifier(r.BaseJobExecutor):
    """
    https://huggingface.co/facebook/bart-large-mnli - 1.7GB

    t.run_group({'group':'ds_sus', 'classes': ['Führungsanfrage', 'Mail', 'Absage', 'Presseanfrage', 'Sonstiges']})
    """
    TAGS = [r.TAG.TEXT_EXTRACTOR]

    def __init__(self, type='classify', classes=None):
        super().__init__()
        self.type = type
        self.classes = classes
        from transformers import pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify(self, text, classes):
        """
             Return: {labels: [], scores:[]}
            :param text:
            :param classes:
            :return:
        """
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

    def read_classes(self, data, key='classes'):
        res = data.get('classes', None)
        if res is None:
            return self.classes
        if isinstance(res, str):
            res = res.split(',')
        return res

    def execute(self, data):
        if 'classify' in data:
            return self.classify(data['classify'], self.read_classes(data))
        if 'group' in data:
            return self.run_group(**data)
        return super().execute(data)

    def page_index(self, params={}):
        p = b.Page(owner=self)
        p.h1("Classify")
        p.p("Klassen mit Komman(,) trennen")
        p.form_input('classify', "Text", id='classify')
        p.form_input('classes', "Klassen", id='classes')
        p(self.action_btn_parametric("Klassifizieren", type=self.type, classify='#classify', classes='#classes'))

        p.pre('', id='result')

        return p.nxui()


class TextWorker(TextExecutor):

    sentence_runner = {}

    def __init__(self, sentence_runner={}):
        super().__init__()
        self.sentence_runner = sentence_runner
        self.param_names['text'] = "Texteingabe"
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

    def execute_text(self, text, data={}):
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


class TextSummarization(TextExecutor):
    """
    https://huggingface.co/sshleifer/distilbart-cnn-12-6
    """
    TAGS = [r.TAG.TEXT_TRANSFORM]

    type = "summarize"

    def __init__(self, model="sshleifer/distilbart-cnn-12-6"):
        super().__init__()
        from transformers import pipeline
        self.param_names['text'] = "Text der zusammengefasst werden soll"
        self.model = model
        self.pipe = pipeline("summarization", model=self.model)

    def summerize(self, text) -> dict:
        """
        Key: summary_text
        :param text:
        :return: object
        """
        res = self.pipe(text)[0]
        res['value'] = res['summary_text']
        return res

    def execute_text(self, text, data={}):
        return self.summerize(text)

    def page(self, params={}):
        p = b.Page(owner=self)
        p('<div class="TextSummarization runner_page">')
        p.h1("TextSummarization")
        p.div("Model: " + self.model)
        text_input_form(self, p, text=params.get('text', ''), caption="Summarize")
        if 'text' in params:
            res = self.summerize(params['text'])
            p.div(self.format_html(res))
        p('</div>')
        return p.nxui()

    def format_html(self, data):
        return '<div class="result">'+data['summary_text']+'</div>'


class MailRespond(r.BaseJobExecutor):

    MODULES = ['extract-msg', 'imapclient']

    type = 'mail_respond'

    llm = None
    qa: QuestionAnswering = None
    classifier: TextClassifier = None

    prompt = "Antworte auf diese Mail!"

    classes = ['anmeldung', 'information', "absage"]

    mails = []

    def __init__(self, args: u.Args = {}):
        super().__init__()
        print("Args: " + str(args))
        self.param_names['text'] = "Nachrichtentext für den eine Antwort erstellt werden soll."

    def nxitems(self):
        return [
            {'title': "Mails", 'url': '?'}
        ]

    def init_models(self):
        from nwebclient import llm
        if self.llm is None:
            self.llm = llm.LlmExecutor()
            # schauen ob als runner verfuegbar
            # self.onParentClass()
        if self.qa is None:
            self.qa = QuestionAnswering()
        if self.classifier is None:
            self.classifier = TextClassifier()

    def parse_msg(self, filedata):
        import extract_msg
        msg = extract_msg.Message(filedata)
        msg_sender = msg.sender
        msg_date = msg.date
        msg_subj = msg.subject
        msg_message = msg.body
        msg.close()
        return {
            'from': msg_sender,
            'subject': msg_subj,
            'title': msg_subj,
            'text': msg_message,
            'date': msg_date
        }

    def respond(self, text, data={}):
        self.init_models()
        sender = self.qa.answer("Von wem ist die Nachricht?", text) #{'score': 0.2509937584400177, 'start': 19, 'end': 63, 'answer': 'Kann ich am 13.2 an einer Führung teilnhmen?'}
        sender_name = ''
        if sender['score']>0.5:
            sender_name = sender['answer']
            print("Sender: " + sender['answer'])
        else:
            print("Anredename nicht mit Sicherheit bestimmbar, " + sender.get('answer', '?'))
        result = self.classifier.classify(text, self.classes)
        label = ''
        if result['scores'][0] > 0.75:
            label = result['labels'][0]
            print("Label: " + label)
        r = self.respond_with_prompt(text, self.prompt)
        return {'text': r, 'label': label, 'from': sender_name}

    def respond_with_prompt(self, text, prompt):
        self.init_models()
        result = self.llm.prompt(text + "\n" + self.prompt + " Antwort: ")
        print("Antwort:" + result['response'])
        return result['response']

    def execute(self, data):
        if 'text' in data:
            self.mails.append(data['text'])
            # TODO Verschiedene Antwortmöglichkeiten
            # self.respond_with_prompt(data['text'], "Antworte mit einer Ablehnung")
            return self.respond(data['text'], data)
        return super().execute(data)

    def a_index(self, p:b.Page, params={}):
        p.h2("Mail Respond")
        p.right("Per KI Mails schneller und besser beantworten")
        p.div("Drop here", id='upload', style="width: 300px; height:100px; background-color: #ffffcc; padding: auto;")
        upload_url = f'?type={self.type}&a=upload'
        p.script(w.js_ready('dropArea("#upload", "'+upload_url+'", function() { console.log("Upload Success"); window.location.href = "?type='+self.type+'&a=item"; })'))
        text_input_form(self, p, params.get('text', ''), "Respond")
        if 'text' in params:
            res = self.respond(params['text'])
            p.div(self.format_html(res))
        p('<hr />')
        p.a("Mails", '?type='+self.type+"&a=list")
        p(' - ')
        p.a("Load", '?type=' + self.type + "&a=load")

    def a_load(self, p:b.Page, params={}):
        p.h2("Load Data")
        p("Daten/Mails aus einem IMAP-Postfach oder einer nweb-Gruppe laden")

    def a_upload(self, p:b.Page, params={}):
        msg = self.parse_msg(params['file0'].filename)
        self.mails.append(msg['text'])
        print(params.keys())
        return {'success':True}

    def a_item(self, p:b.Page, params={}):
        # Wenn keine i da, dann letztes Element anzeigen
        i = params.get('i', -1)
        text = self.mails[int(i)]
        p.h2("Mail: " + str(i))
        p(self.format_html(self.respond(text)))
        # TODO mehr optionen für text anbieten
        p('<hr />')
        p.a("Index", '?type=' + self.type + "&a=index")
        p(' - ')
        p.a("Mails", '?type='+self.type+"&a=list")

    def a_list(self, p:b.Page, params={}):
        for i, text in enumerate(self.mails):
            p.div(text + w.a("Show", f'?type={self.type}&a=item&i={i}'))
        p('<hr />')
        p.a("Index", '?type=' + self.type + "&a=index")

    def page(self, params={}):
        p = b.Page(owner=self)
        p.script('/static/jquery.js')
        p.script('/static/js/base.js')
        p('<div class="MailRespond runner_page">')
        action = getattr(self, 'a_' + params.get('a', 'index'), None)
        res = action(p, params)
        p('</div>')
        if isinstance(res, dict):
            return json.dumps(res)
        else:
            return p.nxui()

    def format_html(self, data):
        t = data['text'].replace('\n', '<br />\n')
        return '<div class="result">'+t+'</div>'
