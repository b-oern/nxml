import html
import json
import random
import threading
import uuid

from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import web as w
from nwebclient import util as u
from nwebclient import dev as d
from nwebclient.nlp import ParagraphParser

from nxml.system import AISystemInfo, select_device
from nxml.hugging import HuggingRunner


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
        self.text_sig = self.define_sig(d.Param('text'))
        self.doc_sig = self.define_sig(d.Param('document_id', 'int'))

    def text_result(self, description, **result):
        """ Beschreibung und Ergebnistyp fuer die Signaturen text und document_id (Inhalt des nweb-Dokuments) """
        for sig in (self.text_sig, self.doc_sig):
            sig.set_description(description)
            sig.r(**result)

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


class Seq2SeqGenerator:
    """
    Ersatz für die in transformers 5 entfernten Pipelines text2text-generation und summarization.
    """

    def __init__(self, model, task=None, device_mode='auto', device=None):
        """
            :param task: z.B. "summarization", übernimmt dann task_specific_params aus der Modell-Config
            :param device: torch-Device, sonst über device_mode bestimmt
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = device or select_device(device_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device).eval()
        params = dict((getattr(self.model.config, 'task_specific_params', None) or {}).get(task, {}))
        self.prefix = params.pop('prefix', None) or getattr(self.model.config, 'prefix', None) or ''
        self.generate_kwargs = params

    def __call__(self, text) -> str:
        import torch
        enc = self.tokenizer(self.prefix + text, return_tensors='pt', truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, **self.generate_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


class TextToText(HuggingRunner, TextExecutor):
    """
    Paraphrasiert oder korrigiert deutschen Text mit einem Seq2Seq-Modell (~3 GB VRAM, T5-large fp32).

    ramsrigouthamg/t5-large-paraphraser-diverse-high-quality
    """
    TAGS = ['hugging']

    MODULES = ['beautifulsoup4', 'transformers', 'torch', 'sentencepiece']

    DE_GRAMMAR = "MRNH/mbart-german-grammar-corrector"
    DE_PARAPHRASE = "Lelon/t5-german-paraphraser-large"

    type = 'ttt'
    model_name = "Lelon/t5-german-paraphraser-large"
    title = 'Text2Text'
    HF_MODEL_ATTRS = ('pipe',)

    def __init__(self, model=None, device_mode='auto'):
        """
            :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        """
        super().__init__()
        self.hf_init(model, device_mode)
        self.param_names['text'] = "Texteingabe"
        self.text_result("Paraphrasiert bzw. korrigiert deutschen Text", generated_text='str')

    def load_model(self):
        self.pipe = Seq2SeqGenerator(self.model_name, device=self.device)

    def generate(self, text):
        # liefert {'generated_text': 'Ich kann heute nicht pünktlich sein.'} wie die frühere Pipeline
        self.load()
        return {'generated_text': self.pipe(text)}

    def execute_text(self, text, data):
        return self.generate(text)

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        p.h2("Text2Text")
        text_input_form(self, p, t='input', input_name='prompt')
        if 'prompt' in params:
            r = self.generate(params['prompt'])
            p.div(r['generated_text'])



class QuestionAnswering(HuggingRunner, r.BaseJobExecutor):
    """
    Beantwortet Fragen zu einem Kontexttext mit einem deutschen extraktiven QA-Modell (GELECTRA, ~0,5 GB VRAM fp32).
    """
    TAGS = ['hugging']

    MODULES = ['transformers', 'torch']

    type = 'qa'

    # wie die frühere question-answering-Pipeline (in transformers 5 entfernt)
    max_length = 384
    doc_stride = 128
    max_answer_len = 15

    model_name = "deepset/gelectra-base-germanquad"
    HF_MODEL_ATTRS = ('tokenizer', 'model')

    def __init__(self, model=None, context='', device_mode='auto'):
        """
            :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        """
        super().__init__()
        self.hf_init(model, device_mode)
        self.context = context
        self.define_vars('context')
        self.define_sig(d.Param('question'), d.PStr('context', ''))\
            .set_description("Beantwortet eine Frage anhand eines Kontexts")\
            .r(score='float', start='int', end='int', answer='str')

    def load_model(self):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.tokenizer = self.hf_pretrained(AutoTokenizer, to_device=False)
        self.model = self.hf_pretrained(AutoModelForQuestionAnswering)

    def answer(self, question, context):
        """
            Extraktive QA, lange Kontexte werden in überlappenden Fenstern verarbeitet.
            :return: {'score': float, 'start': int, 'end': int, 'answer': str}
        """
        import torch
        self.load()
        enc = self.tokenizer(question, context, return_tensors='pt', truncation='only_second',
                             max_length=self.max_length, stride=self.doc_stride,
                             return_overflowing_tokens=True, return_offsets_mapping=True, padding=True)
        sequence_ids = [enc.sequence_ids(i) for i in range(len(enc['input_ids']))]
        offsets = enc.pop('offset_mapping')
        enc.pop('overflow_to_sample_mapping', None)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        best = {'score': 0.0, 'start': 0, 'end': 0, 'answer': ''}
        for i in range(len(offsets)):
            # nur Tokens aus dem Kontext dürfen Teil der Antwort sein
            is_context = torch.tensor([s == 1 for s in sequence_ids[i]], device=self.device)
            start = out.start_logits[i].float().masked_fill(~is_context, -1e9).softmax(-1)
            end = out.end_logits[i].float().masked_fill(~is_context, -1e9).softmax(-1)
            scores = torch.triu(start[:, None] * end[None, :]).tril(self.max_answer_len - 1)
            idx = int(scores.argmax())
            si, ei = divmod(idx, scores.shape[1])
            score = float(scores[si, ei])
            if score > best['score']:
                s, e = int(offsets[i][si][0]), int(offsets[i][ei][1])
                best = {'score': score, 'start': s, 'end': e, 'answer': context[s:e]}
        return best

    def execute(self, data):
        if 'question' in data:
            # {'score': 0.2509937584400177, 'start': 19, 'end': 63, 'answer': 'Kann ich am 13.2 an einer Führung teilnhmen?'}
            return self.answer(data['question'], data.get('context') or self.context)
        return super().execute(data)

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        with p.section(h="Input"):
            p.form_input('question', "Question", id='question')
            p.form_input('context', "Context", id='context')
            p(self.action_btn_parametric("Frage beantworten", type=self.type, question='#question', context='#context'))
        with p.section(h="Output"):
            p.pre('', id='result')



class TextClassifier(HuggingRunner, r.BaseJobExecutor):
    """
    Zero-Shot-Klassifikation von Texten in frei wählbare Klassen (NLI-Modelle, Modell auf der Seite "Modell" wählbar).

    Standard: https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0 - multilingual (DE/EN), ~2,2 GB fp32

    multi_label=True: unabhängiger Score 0..1 pro Klasse (mehrere oder keine Klasse möglich)
    multi_label=False: Scores summieren sich zu 1 (genau eine Klasse)
    Die Scores sind nicht kalibriert, Schwellwerte an eigenen Beispielen festlegen.

    t.classify("Die Lieferung kam beschädigt an.", ['Beschwerde', 'Lob', 'Frage'], multi_label=False)
    t.execute({'op': 'configure', 'model': 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli', 'load': '1'})
    t.run_group({'group':'ds_sus', 'classes': ['Führungsanfrage', 'Mail', 'Absage', 'Presseanfrage', 'Sonstiges']})
    """

    MODULES = ['transformers']
    TAGS = [r.TAG.TEXT_EXTRACTOR, 'hugging']

    model_name = "MoritzLaurer/bge-m3-zeroshot-v2.0"
    HF_MODEL_ATTRS = ('classifier',)

    # Repo-ID -> (Sprachen, Größe, Info)
    MODELS = {
        'MoritzLaurer/bge-m3-zeroshot-v2.0': ("DE, EN, multilingual", "~2,2 GB",
                                              "Empfohlen, bis 8192 Tokens, aktuelle Trainingsdaten"),
        'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli': ("DE, EN, multilingual", "~1,1 GB",
                                                    "Kleiner und schneller, etwas ungenauer"),
        'MoritzLaurer/deberta-v3-large-zeroshot-v2.0': ("nur EN", "~1,7 GB", "Beste Wahl für rein englische Texte"),
        'MoritzLaurer/deberta-v3-base-zeroshot-v2.0': ("nur EN", "~0,7 GB", "Klein, englisch"),
        'svalabs/gbert-large-zeroshot-nli': ("nur DE", "~1,3 GB", "Deutsches BERT"),
        'Sahajtomar/German_Zeroshot': ("nur DE", "~1,3 GB", "Deutsches BERT (älter)"),
        'facebook/bart-large-mnli': ("nur EN", "~1,6 GB", "Klassiker (2019), für Deutsch ungeeignet"),
    }

    TEMPLATES = ["This example is {}.", "This text is about {}.", "Dieser Text ist {}.",
                 "Das Thema dieses Textes ist {}."]

    def __init__(self, type='classify', classes=None, model=None, device_mode='auto', multi_label=True,
                 hypothesis_template=None):
        """
            :param multi_label: Standard für Anfragen ohne multi_label
            :param hypothesis_template: z.B. "Dieser Text ist {}.", None = Standard der Pipeline ("This example is {}.")
        """
        super().__init__()
        self.type = type
        self.classes = classes
        self.multi_label = self.read_bool(multi_label)
        self.hypothesis_template = hypothesis_template or None
        self.hf_init(model, device_mode)
        self.define_vars('multi_label', 'hypothesis_template')
        self.define_sig(d.Param('classify'), d.Param('classes'))\
            .set_description("Zero-Shot-Klassifikation eines Textes (classes mit Komma getrennt, optional "
                             "multi_label und hypothesis_template)")\
            .r(sequence='str', labels='list', scores='list', probabilities='dict')
        self.define_sig(d.Param('group'), d.Param('classes'))\
            .set_description("Klassifiziert alle Dokumente einer nweb-Gruppe, Ausgabe nur im Log")

    @staticmethod
    def read_bool(value, default=True):
        if value is None or value == '':
            return default
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'ja', 'on')
        return bool(value)

    def classify(self, text, classes, multi_label=None, hypothesis_template=None):
        """
            :param multi_label: None = self.multi_label
            :param hypothesis_template: None = self.hypothesis_template
            :return: {sequence, labels: [], scores: [], probabilities: {Klasse: Score}} absteigend nach Score
        """
        self.load()
        kwargs = dict(multi_label=self.multi_label if multi_label is None else multi_label)
        template = hypothesis_template or self.hypothesis_template
        if template:
            kwargs['hypothesis_template'] = template
        res = dict(self.classifier(text, classes, **kwargs))
        res['probabilities'] = {label: float(score) for label, score in zip(res['labels'], res['scores'])}
        return res

    def load_model(self):
        self.classifier = self.hf_pipeline("zero-shot-classification")

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
        res = data.get(key, None)
        if res is None or res == '':
            return self.classes
        if isinstance(res, str):
            res = res.split(',')
        return [c.strip() for c in res if str(c).strip() != '']

    @r.op("Wählt das Modell (Repo-ID auf Hugging Face), das aktuelle wird entladen", name='configure',
          success={'loaded': 'bool', 'model': 'str'}, fail={})
    def op_configure(self, model: str = '', load: str = ''):
        model = model.strip()
        if not model:
            return self.fail("model fehlt")
        self.set_model_name(model)
        if self.read_bool(load, False):
            try:
                self.load()
            except Exception as e:
                return self.fail(f"{model} konnte nicht geladen werden: {e}")
        return self.success(f"Modell: {model}", ui_reload=True, **self.hf_status())

    def execute(self, data):
        if 'classify' in data:
            classes = self.read_classes(data)
            if not classes:
                return self.fail("Keine Klassen angegeben")
            multi_label = data.get('multi_label')
            return self.classify(data['classify'], classes,
                                 multi_label=None if multi_label in (None, '') else self.read_bool(multi_label),
                                 hypothesis_template=data.get('hypothesis_template') or None)
        if 'group' in data:
            return self.run_group(**data)
        return super().execute(data)

    RENDER_JS = """
        function textclassifier_esc(s) {
            return String(s).replace(/[&<>"']/g, function(c) {
                return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
            });
        }
        function textclassifier_render(data) {
            var view = document.getElementById('textclassifier_view');
            document.getElementById('result').innerText = JSON.stringify(data, null, 2);
            if (!data || !data.labels) {
                view.innerHTML = '<div style="color:#c0392b">' + textclassifier_esc((data && data.message) || 'Fehler') + '</div>';
                return;
            }
            var multi = document.getElementById('multi_label').checked;
            var th = parseFloat(document.getElementById('threshold').value) || 0;
            var best = data.labels[0], bestScore = data.scores[0];
            var head;
            if (multi && th > 0 && bestScore < th) {
                head = 'Keine Klasse über ' + (th * 100).toFixed(0) + '% (beste: ' + textclassifier_esc(best) + ')';
            } else {
                head = textclassifier_esc(best) + ' <span style="font-weight:normal;opacity:.7">' +
                       (bestScore * 100).toFixed(1) + '%</span>';
            }
            var h = '<div style="border:1px solid #8884;border-radius:8px;padding:12px;max-width:640px">';
            h += '<div style="font-size:1.4em;font-weight:bold;margin-bottom:10px">' + head + '</div>';
            for (var i = 0; i < data.labels.length; i++) {
                var s = data.scores[i], pct = (s * 100).toFixed(1);
                var hit = multi ? (th > 0 && s >= th) : i === 0;
                h += '<div style="display:flex;gap:8px;align-items:center;margin:4px 0">' +
                     '<span style="width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap' +
                     (hit ? ';font-weight:bold' : '') + '">' + textclassifier_esc(data.labels[i]) + '</span>' +
                     '<span style="flex:1;background:#8882;border-radius:4px;height:14px;position:relative">' +
                     '<span style="position:absolute;left:0;top:0;bottom:0;width:' + pct + '%;border-radius:4px;' +
                     'background:' + (hit ? '#2e86de' : '#9bb7d4') + '"></span>' +
                     (multi && th > 0 ? '<span style="position:absolute;top:-2px;bottom:-2px;width:2px;background:#c0392b;left:' +
                      (th * 100) + '%"></span>' : '') + '</span>' +
                     '<span style="width:60px;text-align:right;font-variant-numeric:tabular-nums">' + pct + '%</span></div>';
            }
            h += '<div style="margin-top:8px;font-size:.85em;opacity:.7">' +
                 (multi ? 'multi_label: unabhängiger Score pro Klasse' : 'single_label: Scores summieren sich zu 100%') +
                 ', nicht kalibriert</div></div>';
            view.innerHTML = h;
        }
        function textclassifier_run(type) {
            exec_job_p({'type': type, 'classify': '#classify', 'classes': '#classes',
                        'multi_label': document.getElementById('multi_label').checked ? '1' : '0',
                        'hypothesis_template': '#hypothesis_template'}, textclassifier_render);
        }
    """

    EXAMPLES = [
        dict(title="Beschwerde", text="Die Lieferung kam beschädigt an und niemand antwortet auf meine Mails.",
             classes="Beschwerde, Lob, Frage"),
        dict(title="Englisch", text="Could you tell me when the new version will be released?",
             classes="Beschwerde, Lob, Frage"),
        dict(title="Mehrere", text="Super Service, aber wann kommt eigentlich die Rechnung?",
             classes="Beschwerde, Lob, Frage, Rechnung"),
    ]

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        p.h1("Zero-Shot-Klassifikation")
        p.script(self.RENDER_JS)
        p.script('var textclassifier_examples = ' + json.dumps(self.EXAMPLES) + ';'
                 'function textclassifier_example(i) {'
                 '  document.getElementById("classify").value = textclassifier_examples[i].text;'
                 '  document.getElementById("classes").value = textclassifier_examples[i].classes;'
                 '}')
        p.div("Beispiele: " + ' '.join(w.button_js(e['title'], f'textclassifier_example({i})')
                                         for i, e in enumerate(self.EXAMPLES)))
        p.div("Text")
        p('<textarea id="classify" name="classify" style="width:90%; height:120px"></textarea>')
        p.form_input('classes', "Klassen (mit Komma getrennt)", id='classes',
                     value=', '.join(self.classes) if self.classes else '')
        p.form_input('hypothesis_template', "Hypothese ({} = Klasse, leer = Standard des Modells)",
                     id='hypothesis_template', value=self.hypothesis_template or '', suggestions=self.TEMPLATES)
        checked = ' checked' if self.multi_label else ''
        p(f'<div><label><input type="checkbox" id="multi_label"{checked} /> multi_label '
          f'(unabhängige Scores, mehrere oder keine Klasse möglich)</label></div>')
        p('<div><label>Schwellwert (nur multi_label) <input type="number" id="threshold" value="0.5" '
          'min="0" max="1" step="0.05" style="width:80px" /></label></div>')
        p(w.button_js("Klassifizieren", f'textclassifier_run({json.dumps(self.type)})'))
        p.div('', id='textclassifier_view', style='margin-top:12px')
        p('<details><summary>JSON</summary>')
        p.pre('', id='result')
        p('</details>')

    def part_hf_model_config(self, p: b.Page, params={}):
        """ Modellwahl auf der Modell-Seite (part_hf_model) """
        from nxml.hugging import hf_model_cache_dir
        import os
        with p.section(h="Modell wählen"):
            p.div("Alle Modelle sind NLI-Zero-Shot-Modelle für dieselbe Pipeline. Beim Wechsel wird das aktuelle "
                  "Modell entladen, das neue lädt beim ersten Aufruf (oder mit \"Wählen + Laden\").")
            p('<table>')
            p('<tr><th>Modell</th><th>Sprachen</th><th>Größe</th><th>Info</th><th>Im Cache</th><th></th></tr>')
            models = dict(self.MODELS)
            if self.model_name and self.model_name not in models:
                models[self.model_name] = ("?", "?", "eigenes Modell")
            for model, (langs, size, info) in models.items():
                active = model == self.model_name
                name = w.a(html.escape(model), 'https://huggingface.co/' + model)
                cached = "ja" if os.path.isdir(hf_model_cache_dir(model)) else "nein"
                btns = '<i>aktiv</i>' if active else \
                    self.action_btn(dict(title="Wählen", type=self.type, op='configure', model=model)) + \
                    self.action_btn(dict(title="Wählen + Laden", type=self.type, op='configure', model=model, load='1'))
                row_style = ' style="font-weight:bold"' if active else ''
                p(f"<tr{row_style}><td>{name}</td><td>{langs}</td>"
                  f"<td>{size}</td><td>{info}</td><td>{cached}</td><td>{btns}</td></tr>")
            p('</table>')
        with p.section(h="Eigenes Modell"):
            p.form_input('model', "Repo-ID (zero-shot-classification bzw. NLI)", value=self.model_name or '',
                         id='model', suggestions=list(self.MODELS.keys()))
            p(self.action_btn_parametric("Übernehmen", dict(type=self.type, op='configure', model='#model')))


class AiTextDetector(HuggingRunner, r.BaseJobExecutor):
    """
    Erkennt KI-generierte bzw. KI-überarbeitete deutsche wissenschaftliche Texte (ModernBERT/mmBERT-small, ~0,6 GB).

    https://huggingface.co/aliissa824/german-academic-ai-text-mmbert-small (CC BY-SA 4.0)

    Klassen: human, ai_generated, ai_assisted (menschlicher Text, von KI überarbeitet)
    Texte werden nach 512 Tokens abgeschnitten, kurze Texte (<= 70 Wörter) sind unzuverlässig.
    Das Ergebnis ist eine Wahrscheinlichkeit, kein Beweis. Kalibrierung: logits / temperature vor softmax.

    t.detect("Die vorliegende Arbeit untersucht ...")
    """
    TAGS = [r.TAG.TEXT_EXTRACTOR, 'hugging']
    MODULES = ['transformers', 'torch']

    model_name = "aliissa824/german-academic-ai-text-mmbert-small"
    HF_MODEL_ATTRS = ('tokenizer', 'model')

    max_length = 512
    # aus calibration.json des Modells
    TEMPERATURE = 3.448
    MIN_WORDS = 70

    def __init__(self, type='ai_text', model=None, device_mode='auto', temperature=None):
        """
            :param temperature: Kalibrierung, None = TEMPERATURE, 1 = unkalibriert
        """
        super().__init__()
        self.type = type
        self.temperature = float(temperature) if temperature not in (None, '') else self.TEMPERATURE
        self.hf_init(model, device_mode)
        self.define_vars('temperature')
        self.define_sig(d.Param('text'))\
            .set_description("Schätzt, ob ein deutscher wissenschaftlicher Text von Menschen, KI oder "
                             "KI-unterstützt geschrieben wurde")\
            .r(label='str', score='float', probabilities='dict', truncated='bool', words='int', reliable='bool')

    def load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = self.hf_pretrained(AutoTokenizer, to_device=False)
        self.model = self.hf_pretrained(AutoModelForSequenceClassification)

    def detect(self, text: str) -> dict:
        """
            :return: {label, score, probabilities: {human, ai_generated, ai_assisted}, truncated, words, reliable}
        """
        import torch
        self.load()
        enc = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length,
                             return_overflowing_tokens=False)
        truncated = len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_length
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            logits = self.model(**enc).logits[0].float()
        probs = (logits / (self.temperature or 1.0)).softmax(-1).tolist()
        id2label = self.model.config.id2label
        probabilities = {id2label[i]: float(p) for i, p in enumerate(probs)}
        label = max(probabilities, key=probabilities.get)
        words = len(text.split())
        return dict(label=label, score=probabilities[label], probabilities=probabilities,
                    truncated=truncated, words=words, reliable=words > self.MIN_WORDS)

    def execute(self, data):
        if 'text' in data:
            text = str(data['text']).strip()
            if not text:
                return self.fail("Kein Text angegeben")
            return self.detect(text)
        return super().execute(data)

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        p.h1("KI-Text-Erkennung (Deutsch, wissenschaftlich)")
        p.div("Klassen: human, ai_generated, ai_assisted. Nur die ersten 512 Tokens werden bewertet, "
              f"Texte bis {self.MIN_WORDS} Wörter sind unzuverlässig. Wahrscheinlichkeit, kein Beweis.")
        p('<textarea id="text" name="text" style="width:90%; height:160px"></textarea>')
        p(self.action_btn_parametric("Prüfen", type=self.type, text='#text'))
        with p.section(h="Output"):
            p.pre('', id='result')


class Jeff(HuggingRunner, r.BaseJobExecutor):
    """
    Fact-Check- und Text-Klassifikator Jeff-1 (~8 GB VRAM bf16, ~4,5 GB 8-bit, ~3 GB 4-bit).

    https://huggingface.co/GestaltLabs/Jeff-1 - LoRA auf Qwen/Qwen3-4B-Instruct-2507 (~8GB bf16)
    https://github.com/Gestalt-Lab/jeff

    Text-konditionierter Klassifikator: liefert Wahrscheinlichkeiten über Labels, keinen generierten Text.
    Fragetypen: choice (benannte Labels), noul (ja/nein), score (geordnete Stufen).

    Prompt-Format und Readout entsprechen jev_clf (scripts/jev_clf_lm_eval.py, jev_clf/readout.py).

    j.execute({'state': {'claim': '...', 'evidence': ['...']}, 'questions': {
        'verdict': {'kind': 'choice', 'instructions': '...', 'criteria': {'supported': '...', 'refuted': '...'}},
        'has_date': {'kind': 'noul', 'instructions': 'Does the evidence include a year?'},
        'strength': {'kind': 'score', 'instructions': '...', 'criteria': ['none', 'weak', 'strong']}
    }})
    j.execute({'claim': '...', 'evidence': ['...']})  # Fact-Check mit Standard-Verdict
    j.execute({'op': 'configure', 'preset': 'jeff1_4bit'})  # Modell wechseln (load/unload/status)

    Das Modell wird lazy beim ersten Aufruf geladen.
    """
    MODULES = ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'psutil']
    TAGS = [r.TAG.TEXT_EXTRACTOR, 'hugging']

    SYSTEM = (
        "You are a fact-checking classifier. You are given a claim and the evidence "
        "passages retrieved for it. You answer with exactly one verdict label."
    )

    FACTCHECK_QUESTION = {
        'kind': 'choice',
        'instructions': "Read the claim and the evidence passages in `state`. Decide whether the "
                        "evidence establishes the claim, contradicts it, or cannot decide it.",
        'criteria': {
            'supported': "The evidence passages together entail the claim: if they are true, the claim is true.",
            'refuted': "The evidence passages together contradict the claim: if they are true, the claim is false.",
            'not_enough_info': "No evidence passages are given, or they are irrelevant or too weak to "
                               "establish or contradict the claim.",
        }
    }

    # Der LoRA-Adapter passt nur auf Qwen3-4B-Instruct-2507. Kleiner wird es nur per Quantisierung
    # (bitsandbytes, nur CUDA) oder mit kleineren Qwen3-Modellen ohne Adapter (Zero-Shot, unkalibriert).
    PRESETS = {
        'jeff1': dict(title="Jeff-1 bf16", base_model="Qwen/Qwen3-4B-Instruct-2507",
                      adapter="GestaltLabs/Jeff-1", quantization=None,
                      info="~8 GB VRAM, Referenz-Qualität"),
        'jeff1_8bit': dict(title="Jeff-1 8-bit", base_model="Qwen/Qwen3-4B-Instruct-2507",
                           adapter="GestaltLabs/Jeff-1", quantization='8bit',
                           info="~4.5 GB VRAM, kaum Qualitätsverlust, nur CUDA (bitsandbytes)"),
        'jeff1_4bit': dict(title="Jeff-1 4-bit (NF4)", base_model="Qwen/Qwen3-4B-Instruct-2507",
                           adapter="GestaltLabs/Jeff-1", quantization='4bit',
                           info="~3 GB VRAM, leichter Kalibrierungsverlust, nur CUDA (bitsandbytes)"),
        'qwen3_4b': dict(title="Qwen3-4B ohne Adapter", base_model="Qwen/Qwen3-4B-Instruct-2507",
                         adapter=None, quantization=None,
                         info="~8 GB, Zero-Shot-Vergleich ohne Jeff-Training"),
        'qwen3_1.7b': dict(title="Qwen3-1.7B ohne Adapter", base_model="Qwen/Qwen3-1.7B",
                           adapter=None, quantization=None,
                           info="~3.5 GB, Zero-Shot, deutlich schwächer und unkalibriert"),
        'qwen3_0.6b': dict(title="Qwen3-0.6B ohne Adapter", base_model="Qwen/Qwen3-0.6B",
                           adapter=None, quantization=None,
                           info="~1.2 GB, Zero-Shot, läuft auch auf CPU, nur für Tests"),
    }

    def __init__(self, type='jeff', base_model="Qwen/Qwen3-4B-Instruct-2507", adapter="GestaltLabs/Jeff-1",
                 quantization=None, device=None, max_length=2048):
        super().__init__()
        self.type = type
        self.max_length = max_length
        self.base_model = base_model
        self.adapter = adapter
        self.quantization = quantization
        # festes Gerät, None = automatisch (CUDA, MPS, CPU)
        self.fixed_device = device
        self.hf_init()

    HF_MODEL_ATTRS = ('model', 'tokenizer')
    HF_PAGES = [('index', "Fact-Check"), ('questions', "Fragen (JSON)")]

    @property
    def model_name(self):
        return self.base_model

    @model_name.setter
    def model_name(self, value):
        self.base_model = value

    @property
    def loaded(self):
        return self.hf_loaded

    def hf_select_device(self):
        return self.fixed_device or select_device('auto')

    def hf_models(self):
        return [m for m in (self.base_model, self.adapter) if m]

    def hf_model_label(self):
        return self.model_id()

    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.float32 if self.device == 'cpu' else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.quantization in ('4bit', '8bit'):
            from transformers import BitsAndBytesConfig
            if self.quantization == '4bit':
                qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                          bnb_4bit_compute_dtype=torch.bfloat16)
            else:
                qcfg = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(self.base_model, quantization_config=qcfg,
                                                         dtype=torch.bfloat16, device_map='auto')
        else:
            # erst auf CPU laden, sonst ggf. "Cannot copy out of meta tensor" bei knappem Speicher
            model = AutoModelForCausalLM.from_pretrained(self.base_model, dtype=dtype, device_map='cpu')
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        if self.quantization not in ('4bit', '8bit'):
            model = model.to(self.device)
        self.model = model.eval()

    def configure(self, base_model=None, adapter=None, quantization=None, preset=None):
        """ Wechselt das Modell, ein geladenes Modell wird entladen. Neues Modell wird lazy geladen. """
        if preset is not None:
            cfg = self.PRESETS[preset]
            base_model, adapter, quantization = cfg['base_model'], cfg['adapter'], cfg['quantization']
        self.unload()
        self.base_model = base_model or self.base_model
        self.adapter = adapter or None
        self.quantization = quantization or None

    def model_id(self):
        res = self.base_model
        if self.adapter:
            res += '+' + self.adapter
        if self.quantization:
            res += ' (' + self.quantization + ')'
        return res

    def hf_status(self):
        return dict(super().hf_status(), base_model=self.base_model, adapter=self.adapter,
                    quantization=self.quantization)

    def status(self):
        return self.success(**self.hf_status())

    @staticmethod
    def label_space(q):
        kind = q.get('kind', 'choice')
        if kind == 'noul':
            return ['yes', 'no']
        if kind == 'score':
            return [str(i) for i in range(len(q['criteria']))]
        return list(q['criteria'].keys())

    def question_to_text(self, q):
        lines = [q['instructions']]
        kind = q.get('kind', 'choice')
        for label in self.label_space(q):
            if kind == 'noul':
                definition = (q.get('criteria') or {}).get(label)
            elif kind == 'score':
                definition = q['criteria'][int(label)]
            else:
                definition = q['criteria'][label]
            lines.append(f"{label}: {definition}" if definition else label)
        return "\n".join(lines)

    @staticmethod
    def state_to_text(state):
        if isinstance(state, str):
            return state
        if isinstance(state, dict) and 'claim' in state:
            lines = [f"Claim: {state['claim']}", "Evidence:"]
            evidence = state.get('evidence') or []
            if not evidence:
                lines.append("(none)")
            for i, ev in enumerate(evidence, 1):
                if isinstance(ev, dict):
                    title, text = ev.get('title'), ev.get('text', '')
                    lines.append(f"{i}. {title}: {text}" if title else f"{i}. {text}")
                else:
                    lines.append(f"{i}. {ev}")
            rest = {k: v for k, v in state.items() if k not in ('claim', 'evidence')}
            if rest:
                lines.append(json.dumps(rest, sort_keys=True))
            return "\n".join(lines)
        return json.dumps(state, sort_keys=True)

    def build_prompt(self, state, q):
        user = self.question_to_text(q) + "\n\nState:\n" + self.state_to_text(state) + "\n\nVerdict:"
        messages = [{'role': 'system', 'content': self.SYSTEM}, {'role': 'user', 'content': user}]
        # enable_thinking wirkt nur bei hybriden Qwen3-Modellen (1.7B/0.6B), das 2507-Template ignoriert es
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def label_ids(self, labels):
        out = {}
        for label in labels:
            ids = self.tokenizer(" " + label, add_special_tokens=False)['input_ids']
            if not ids:
                ids = self.tokenizer(label, add_special_tokens=False)['input_ids']
            out[label] = ids
        return out

    def distribution(self, state, q):
        """ first_token-Readout wenn alle Labels unterschiedliche erste Tokens haben, sonst sequence """
        import torch
        labels = self.label_space(q)
        variants = self.label_ids(labels)
        seqs = list(variants.values())
        first_token = all(seqs) and len({s[0] for s in seqs}) == len(seqs)
        enc = self.tokenizer(self.build_prompt(state, q), return_tensors='pt',
                             truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        if first_token:
            sub = torch.tensor([logits[variants[label][0]] for label in labels])
        else:
            prompt_ids = enc['input_ids'][0].tolist()
            totals = []
            for label in labels:
                seq = variants[label]
                with torch.no_grad():
                    out = self.model(input_ids=torch.tensor([prompt_ids + seq]).to(self.device)).logits[0]
                total = 0.0
                for k, tid in enumerate(seq):
                    total += float(torch.log_softmax(out[len(prompt_ids) - 1 + k].float(), dim=-1)[tid])
                totals.append(total)
            sub = torch.tensor(totals, dtype=torch.float32)
        probs = torch.softmax(sub, dim=-1)
        return {label: float(p) for label, p in zip(labels, probs)}, 'first_token' if first_token else 'sequence'

    def classify(self, state, questions):
        """
            :param state: str oder dict (z.B. {claim, evidence})
            :param questions: {qid: {kind: choice|noul|score, instructions, criteria}}
            :return: {choices: {}, nouls: {}, scores: {}}
        """
        self.load()
        result = {'model': self.model_id(), 'choices': {}, 'nouls': {}, 'scores': {}, 'readout_modes': {}}
        for qid, q in questions.items():
            probs, mode = self.distribution(state, q)
            result['readout_modes'][qid] = mode
            kind = q.get('kind', 'choice')
            conf = max(probs.values())
            if kind == 'noul':
                result['nouls'][qid] = {'noul': probs['yes']}
            elif kind == 'score':
                score = sum(int(label) * p for label, p in probs.items())
                result['scores'][qid] = {'score': score, 'probabilities': probs, 'confidence': conf}
            else:
                top = max(probs.items(), key=lambda kv: kv[1])[0]
                result['choices'][qid] = {'choice': top, 'probabilities': probs, 'confidence': conf}
        return result

    @staticmethod
    def read_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    def execute(self, data):
        op = data.get('op')
        if op == 'configure':
            self.configure(base_model=data.get('base_model'), adapter=data.get('adapter'),
                           quantization=data.get('quantization'), preset=data.get('preset'))
            if str(data.get('load', '')).lower() in ('1', 'true', 'yes'):
                self.load()
            return self.status()
        if 'questions' in data:
            return self.classify(self.read_json(data.get('state', '')), self.read_json(data['questions']))
        if 'claim' in data:
            evidence = self.read_json(data.get('evidence', []))
            if isinstance(evidence, str):
                evidence = [e for e in evidence.split('\n') if e.strip() != '']
            return self.classify({'claim': data['claim'], 'evidence': evidence},
                                 {'verdict': self.FACTCHECK_QUESTION})
        return super().execute(data)

    EXAMPLE_QUESTIONS = {
        'verdict': FACTCHECK_QUESTION,
        'has_date': {'kind': 'noul', 'instructions': "Does the evidence include a year?"},
        'strength': {'kind': 'score', 'instructions': "How strongly does the evidence settle the claim?",
                     'criteria': ['none', 'weak', 'moderate', 'strong']}
    }

    FACTCHECK_EXAMPLES = [
        dict(title="supported", claim="The Eiffel Tower was completed in 1889.",
             evidence=["The Eiffel Tower was built for the 1889 World's Fair in Paris.",
                       "Construction finished in March 1889."]),
        dict(title="refuted", claim="Albert Einstein was born in Vienna.",
             evidence=["Albert Einstein was born on 14 March 1879 in Ulm, in the Kingdom of Württemberg."]),
        dict(title="not_enough_info", claim="The company's CEO speaks four languages.",
             evidence=["The company was founded in 2004 and has offices in Berlin and Madrid.",
                       "The CEO studied economics in Munich."]),
        dict(title="ohne Evidence", claim="The Eiffel Tower was completed in 1889.", evidence=[]),
        dict(title="Deutsch", claim="Der Rhein mündet in die Nordsee.",
             evidence=["Der Rhein fließt durch die Niederlande und mündet bei Rotterdam in die Nordsee."]),
    ]

    RENDER_JS = """
        function jeff_render(data) {
            function bars(probs) {
                var h = '';
                for (var k in probs) {
                    var pct = (probs[k] * 100).toFixed(1);
                    h += '<div style="display:flex;gap:8px;align-items:center">' +
                         '<span style="width:140px">' + k + '</span>' +
                         '<span style="display:inline-block;height:12px;width:' + (probs[k] * 300) + 'px;background:#4a90d9"></span>' +
                         '<span>' + pct + '%</span></div>';
                }
                return h;
            }
            var h = '';
            for (var q in (data.choices || {})) {
                var c = data.choices[q];
                h += '<h3>' + q + ': ' + c.choice + '</h3>' + bars(c.probabilities);
            }
            for (var q in (data.nouls || {})) {
                var n = data.nouls[q].noul;
                h += '<h3>' + q + ': ' + (n >= 0.5 ? 'yes' : 'no') + '</h3>' + bars({'yes': n, 'no': 1 - n});
            }
            for (var q in (data.scores || {})) {
                var s = data.scores[q];
                h += '<h3>' + q + ': ' + s.score.toFixed(2) + '</h3>' + bars(s.probabilities);
            }
            document.getElementById('jeff_view').innerHTML = h;
            document.getElementById('result').innerText = JSON.stringify(data, null, 2);
        }
    """

    def part_hf_model_config(self, p: b.Page, params={}):
        """ Modellwahl auf der Modell-Seite (part_hf_model) """
        with p.section(h="Vorschläge"):
            p.div("Der Jeff-1-Adapter passt nur auf Qwen3-4B-Instruct-2507. Für weniger Speicher: 8-/4-bit Quantisierung "
                  "(benötigt CUDA und bitsandbytes). Modelle ohne Adapter sind nur Zero-Shot und nicht kalibriert.")
            p('<table>')
            p('<tr><th>Modell</th><th>Basis</th><th>Adapter</th><th>Quant.</th><th>Info</th><th></th></tr>')
            for key, cfg in self.PRESETS.items():
                active = cfg['base_model'] == self.base_model and cfg['adapter'] == self.adapter                          and cfg['quantization'] == self.quantization
                p('<tr>')
                p(f"<td>{'<b>' + cfg['title'] + '</b>' if active else cfg['title']}</td>")
                p(f"<td>{cfg['base_model']}</td><td>{cfg['adapter'] or '-'}</td>"
                  f"<td>{cfg['quantization'] or 'bf16'}</td><td>{cfg['info']}</td>")
                p('<td>' + self.action_btn(dict(title="Wählen", type=self.type, op='configure', preset=key)) +
                  self.action_btn(dict(title="Wählen + Laden", type=self.type, op='configure', preset=key, load='1')) +
                  '</td>')
                p('</tr>')
            p('</table>')
        with p.section(h="Eigenes Modell"):
            p.form_input('base_model', "Basismodell", value=self.base_model, id='base_model',
                         suggestions=sorted({c['base_model'] for c in self.PRESETS.values()}))
            p.form_input('adapter', "Adapter (leer = keiner)", value=self.adapter or '', id='adapter',
                         suggestions=['GestaltLabs/Jeff-1'])
            p.form_input('quantization', "Quantisierung (leer, 8bit, 4bit)", value=self.quantization or '',
                         id='quantization', suggestions=['8bit', '4bit'])
            p(self.action_btn_parametric("Übernehmen", dict(type=self.type, op='configure', base_model='#base_model',
                                                           adapter='#adapter', quantization='#quantization')))

    def part_result(self, p: b.Page):
        p.script(self.RENDER_JS)
        p.div('', id='jeff_view')
        p.pre('', id='result')

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        p.h1("Jeff-1 Fact-Check")
        p.div(w.a("GestaltLabs/Jeff-1 auf Hugging Face", "https://huggingface.co/GestaltLabs/Jeff-1"))
        p.script('var jeff_examples = ' + json.dumps(self.FACTCHECK_EXAMPLES) + ';'
                 'function jeff_example(i) {'
                 '  document.getElementById("claim").value = jeff_examples[i].claim;'
                 '  document.getElementById("evidence").value = jeff_examples[i].evidence.join("\\n");'
                 '}')
        p.div("Beispiele: " + ' '.join(w.button_js(e['title'], f'jeff_example({i})')
                                         for i, e in enumerate(self.FACTCHECK_EXAMPLES)))
        p.form_input('claim', "Claim", id='claim')
        p.div("Evidence (eine Passage pro Zeile)")
        p('<textarea id="evidence" name="evidence" style="width:90%; height:200px"></textarea>')
        p(w.button_js("Prüfen", 'exec_job_p({"type": "' + self.type + '", "claim": "#claim", "evidence": "#evidence"}, jeff_render)'))
        self.part_result(p)

    def part_questions(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'questions')
        p.h1("Jeff-1 Fragen")
        p.div("State (Text oder JSON)")
        p('<textarea id="state" name="state" style="width:90%; height:150px">' +
          json.dumps({'claim': "The museum opened in 1998.",
                      'evidence': ["The museum first opened to visitors in 1998."]}, indent=2) +
          '</textarea>')
        p.div("Questions (JSON, kind: choice | noul | score)")
        p('<textarea id="questions" name="questions" style="width:90%; height:300px">' +
          json.dumps(self.EXAMPLE_QUESTIONS, indent=2) + '</textarea>')
        p(w.button_js("Klassifizieren", 'exec_job_p({"type": "' + self.type + '", "state": "#state", "questions": "#questions"}, jeff_render)'))
        self.part_result(p)


class TextWorker(TextExecutor):
    """
    Zerlegt Text in Absätze und verarbeitet jeden mit konfigurierbaren Satz-Runnern (z.B. Grammatik, Paraphrase).
    """

    MODULES = ['beautifulsoup4']

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
        self.text_result("Wendet die sentence_runner auf jeden Absatz an", result='list')
    def execute(self, data):
        if 'text' in data:
            return self.execute_text(data['text'], data)
        if 'sentence' in data:
            return self.process_sentence(data['sentence'], data)
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


class TextSummarization(HuggingRunner, TextExecutor):
    """
    Fasst englische Texte zusammen (~1,2 GB VRAM, distilbart-cnn-12-6 fp32).

    https://huggingface.co/sshleifer/distilbart-cnn-12-6
    """

    MODULES = ['beautifulsoup4', 'transformers', 'torch']
    TAGS = [r.TAG.TEXT_TRANSFORM, 'hugging']

    type = "summarize"
    model_name = "sshleifer/distilbart-cnn-12-6"
    HF_MODEL_ATTRS = ('pipe',)

    def __init__(self, model=None, device_mode='auto'):
        """
            :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        """
        super().__init__()
        self.hf_init(model, device_mode)
        self.param_names['text'] = "Text der zusammengefasst werden soll"
        self.text_result("Fasst einen Text zusammen", summary_text='str', value='str')

    @property
    def model(self):
        """ früherer Name von model_name """
        return self.model_name

    def load_model(self):
        self.pipe = Seq2SeqGenerator(self.model_name, task='summarization', device=self.device)

    def summerize(self, text) -> dict:
        """
        Key: summary_text
        :param text:
        :return: object
        """
        self.load()
        summary = self.pipe(text)
        return {'summary_text': summary, 'value': summary}

    def execute_text(self, text, data={}):
        return self.summerize(text)

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        p('<div class="TextSummarization runner_page">')
        text_input_form(self, p, text=params.get('text', ''), caption="Summarize")
        if 'text' in params:
            res = self.summerize(params['text'])
            p.div(self.format_html(res))
        p('</div>')

    def format_html(self, data):
        return '<div class="result">'+data['summary_text']+'</div>'


class MailRespond(r.BaseJobExecutor):
    """
    Erstellt mit LLM, Klassifikator und Question Answering Antwortvorschläge auf E-Mails.
    """

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
        self.define_sig(d.Param('text')).set_description("Erstellt per LLM eine Antwort auf eine Mail")\
            .r(**{'text': 'str', 'label': 'str', 'from': 'str'})

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
