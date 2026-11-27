import json

from nwebclient import runner, base
from nwebclient import util
from nwebclient import dev
from nwebclient import base as b
from nwebclient import dev as d
from nwebclient import web
import os
import os.path
import base64
import io
import requests

from nxml.hugging import HuggingRunner


class Whisper(runner.BaseJobExecutor):
    """
      "whisper": "nxml.analyse:Whisper",
    """
    MODULES = ['openai-whisper']
    TAGS = ['speech-to-text', 'audio-in']

    def __init__(self):
        super().__init__('whisper')
        # TODO search for ffmpeg in path  os.environ['PATH'] os.pathsep
        import whisper
        self.model = whisper.load_model("base")
        self.define_sig(d.Param('audio')).set_description("Transkribiert eine Audiodatei (Pfad)").r(text='str')
        self.define_sig(d.Param('url')).set_description("Lädt Audio von einer URL und transkribiert es").r(text='str')

    def execute(self, data):
        if 'url' in data:
            util.download(data['url'], 'audiofile.dat', False)
            data['audio'] = 'audiofile.dat'
        if not 'audio' in data and 'file0' in data:
            util.file_put_contents('audiofile.dat', base64.b64decode(data['file0']))
            data['audio'] = 'audiofile.dat'
        result = self.model.transcribe(data['audio'])
        # print(str(result))
        data['text'] = "" + str(result['text'])
        return data

    def part_index(self, p: base.Page, params={}):
        with p.section(h="Whisper Transkription"):
            p.form_input('url', "Audio-URL", id='url')
            p(self.action_btn_parametric("Transkribieren (URL)", type=self.type, url='#url'))
            p.form_input('audio', "Audio-Pfad", id='audio')
            p(self.action_btn_parametric("Transkribieren (Pfad)", type=self.type, audio='#audio'))
            p.dropzone('audio_drop', "Audiodatei hier ablegen")
            p.hidden('audio_data', '', 'audio_data')
            p.js_ready('nx_initFileDragArea("audio_drop", "audio_data");')
            p(self.action_btn_parametric("Transkribieren (Upload)", type=self.type, file0='#audio_data'))
        p.pre('', id='result')
        p.hr()
        p.prop("Modell", "base")


class Toxity(runner.BaseJobExecutor):
    """
    Bewertet Texte mit Detoxify (multilingual) auf Toxizität, Beleidigungen, Drohungen und ähnliches.
    """
    MODULES = ['detoxify']
    TAGS = ['nlp']
    text_key = 'text'

    def __init__(self):
        super().__init__('toxity')
        from detoxify import Detoxify
        self.classifier_toxity = Detoxify('multilingual')
        self.define_sig(d.Param('text')).set_description("Bewertet die Toxizität eines Textes (mehrsprachig)")\
            .r(toxity='dict', success='bool')

    def execute(self, data):
        if isinstance(data, str):
            data = {'text': data}
        if self.text_key not in data:
            return super().execute(data)
        text = data[self.text_key]
        tv = self.classifier_toxity.predict(text)
        data['toxity'] = dict(map(lambda kv: (kv[0], float(kv[1])), tv.items()))
        # {'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
        data['success'] = True
        return data

    LABELS = {
        'toxicity': "Toxizität",
        'severe_toxicity': "Starke Toxizität",
        'obscene': "Obszön",
        'identity_attack': "Angriff auf Identität",
        'insult': "Beleidigung",
        'threat': "Drohung",
        'sexual_explicit': "Sexuell explizit",
    }

    EXAMPLES = [
        ("Freundlich", "Vielen Dank für die schnelle Hilfe, das hat mir sehr geholfen!"),
        ("Kritik", "Der Vortrag war leider ziemlich langweilig und schlecht vorbereitet."),
        ("Beleidigung", "Du bist ein kompletter Idiot und hast keine Ahnung."),
        ("English", "This is a great idea, thanks for sharing."),
    ]

    RENDER_JS = """
        function toxity_render(data) {
            var labels = %s;
            var tv = (data && data.toxity) || {};
            var h = '';
            for (var k in tv) {
                var v = tv[k];
                var color = v >= 0.5 ? '#d9534f' : (v >= 0.1 ? '#f0ad4e' : '#5cb85c');
                h += '<div style="display:flex;gap:8px;align-items:center;margin:2px 0">' +
                     '<span style="width:180px">' + (labels[k] || k) + '</span>' +
                     '<span style="display:inline-block;height:12px;width:' + Math.max(1, v * 300) + 'px;background:' + color + '"></span>' +
                     '<span>' + (v * 100).toFixed(1) + ' %%</span></div>';
            }
            if (h === '') h = '<div>Kein Ergebnis: ' + ((data && data.message) || '') + '</div>';
            document.getElementById('toxity_view').innerHTML = h;
            document.getElementById('result').innerText = JSON.stringify(data, null, 2);
        }
    """

    def part_index(self, p: base.Page, params={}):
        with p.section(h="Toxizität (Detoxify multilingual)"):
            p.div("Wahrscheinlichkeit je Kategorie, ab 50 % rot, ab 10 % gelb. "
                  "Mehrsprachig (u.a. Deutsch und Englisch).")
            p.script(self.RENDER_JS % json.dumps(self.LABELS) +
                     'var toxity_examples = ' + json.dumps([t for _, t in self.EXAMPLES]) + ';'
                     'function toxity_example(i) { document.getElementById("toxity_text").value = toxity_examples[i]; }')
            p.div("Beispiele: " + ' '.join(web.button_js(title, f'toxity_example({i})')
                                             for i, (title, _) in enumerate(self.EXAMPLES)))
            p('<textarea id="toxity_text" name="text" style="width:90%; height:120px"></textarea>')
            p.div(self.action_btn_parametric("Bewerten", dict(type=self.type, text='#toxity_text'),
                                             on_success='toxity_render', hint="Text mit Detoxify bewerten"))
            p.div('', id='toxity_view')
            p.pre('', id='result')


class Nltk(runner.BaseJobExecutor):
    """
    Erkennt die Sprache eines Textes und zerlegt ihn mit NLTK in Wörter.
    """
    MODULES = ['nltk']

    def __init__(self):
        super().__init__('nltk')
        import nltk
        try:
            nltk.find('punkt')
        except LookupError:
            nltk.download('punkt')
        self.define_sig(d.Param('text')).set_description("Erkennt die Sprache und zerlegt den Text in Wörter")\
            .r(lang='str', words='list', success='bool')

    def nltk_lang(self, lang):
        if lang == 'de':
            return 'german'
        return 'english'

    def execute(self, data):
        from nltk import word_tokenize
        from langdetect import detect
        if isinstance(data, str):
            data = {'text': data}
        text = data['text']
        data['lang'] = detect(text)
        data['words'] = word_tokenize(text, language=self.nltk_lang(data['lang']))
        data['success'] = True
        return data


class FlairRunner(runner.BaseJobExecutor):
    """
    Erkennt Named Entities (Personen, Orte, Organisationen, ...) in Texten mit einem Flair-NER-Modell.
    """

    MODULES = ['flair']

    def __init__(self, model="flair/ner-english-ontonotes-large"):
        super().__init__('flair')
        try:
            from flair.models import SequenceTagger
            self.tagger = SequenceTagger.load(model)
        except (ImportError, ModuleNotFoundError) as e:
            print("Error: Init FlairRunner, " + str(e))
        self.define_sig(d.Param('text')).set_description("Named Entity Recognition (Flair)")\
            .r(ner='list', success='bool')

    def tag_text(self, text):
        from flair.data import Sentence
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        items = []
        for entity in sentence.get_spans('ner'):
            items.append(self.span_to_dict(entity))
        return items

    def span_to_dict(self, span):
        # embedding
        return {'start_position': span.start_position,
                'end_position': span.end_position,
                'text': span.text,
                'tag': str(span.tag)}

    def execute(self, data):
        try:
            if isinstance(data, str):
                data = {'text': data}
            data['ner'] = self.tag_text(data['text'])
            data['success'] = True
        except AttributeError as e:
            data['error'] = 'Flair Error ' + str(e)
        return data

    def part_index(self, p:base.Page, params={}):
        p.h1("Flair SequenceTagger")
        p.form_input('text', "Text", id='text')
        p(self.action_btn_parametric("Tag", type=self.type, text='#text'))
        p.pre('', id='result')


class BertEmbeddings(HuggingRunner, runner.BaseJobExecutor):
    """
    Berechnet Text-Embeddings mit dem Sentence-Transformer all-MiniLM-L6-v2 (~0,1 GB VRAM, 22M Parameter fp32).
    """
    TAGS = ['hugging']
    MODULES = ['transformers', 'langchain', 'sentence_transformers', 'langchain-huggingface']
    # Kurzname wie in embedding_model gespeichert, auf dem Hub sentence-transformers/<name>
    model_name = "all-MiniLM-L6-v2"
    HF_MODEL_ATTRS = ('embeddings',)

    def __init__(self, model_name=None, device_mode='auto'):
        super().__init__()
        self.hf_init(model_name, device_mode)
        self.define_sig(d.Param('text')).set_description("Erstellt ein Text-Embedding (Sentence-Transformer)")\
            .r(embedding='list', embedding_model='str', success='bool')

    @property
    def model(self):
        """ früherer Name von model_name """
        return self.model_name

    def hf_models(self):
        return [self.model_name if '/' in self.model_name else 'sentence-transformers/' + self.model_name]

    def load_model(self):
        #from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})

    def execute(self, data):
        if 'text' not in data:
            return super().execute(data)
        try:
            self.load()
            embedding = self.embeddings.embed_documents([data['text']])
            data['embedding'] = embedding[0]
            data['embedding_model'] = self.model_name
            data['success'] = True
        except Exception as e:
            print("Error: " + str(e))
            data['success'] = False
        return data


class TextEmbeddings(HuggingRunner, runner.BaseJobExecutor):
    """
    Mehrsprachige Text-Embeddings (Deutsch + Englisch) mit sentence-transformers, z.B. für nxml.rag (embedder=text_embed).

    Standard ist BAAI/bge-m3: 100+ Sprachen, sehr gut für DE und EN, 1024 Dimensionen, bis 8192 Token je Text,
    MIT-Lizenz, ~2,3 GB (auf der GPU in fp16 ~1,2 GB VRAM). Weitere Modelle in MODELS, z.B. kleiner und schneller
    intfloat/multilingual-e5-base. Modelle wie E5 brauchen Präfixe für Anfrage und Dokument, die über
    input_type (query oder passage) automatisch gesetzt werden.

        {"type": "text_embed", "op": "embed", "input": ["Text 1", "Text 2"], "input_type": "passage"}
        {"type": "text_embed", "text": "Ein Text"}      -> embedding (wie BertEmbeddings)

    pip install sentence-transformers (dazu torch, für CUDA passend zur Grafikkarte installieren)
    """
    TAGS = ['embedding']
    MODULES = ['sentence-transformers']
    model_name = 'BAAI/bge-m3'
    HF_MODEL_ATTRS = ('st_model',)

    # Repo-ID -> dims, max. Token, Präfixe für query/passage, trust_remote_code
    MODELS = {
        'BAAI/bge-m3': dict(dims=1024, max_tokens=8192, query='', passage='',
                            info='Empfehlung: beste Qualität DE+EN, lange Texte'),
        'intfloat/multilingual-e5-large': dict(dims=1024, max_tokens=512, query='query: ', passage='passage: ',
                                               info='sehr gut, 512 Token'),
        'intfloat/multilingual-e5-base': dict(dims=768, max_tokens=512, query='query: ', passage='passage: ',
                                              info='guter Kompromiss, ~1,1 GB'),
        'intfloat/multilingual-e5-small': dict(dims=384, max_tokens=512, query='query: ', passage='passage: ',
                                               info='klein und schnell, auch auf CPU'),
        'jinaai/jina-embeddings-v2-base-de': dict(dims=768, max_tokens=8192, query='', passage='',
                                                  trust_remote_code=True,
                                                  info='zweisprachig DE/EN, lange Texte, führt Code aus dem Repo aus'),
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': dict(
            dims=384, max_tokens=128, query='', passage='', info='sehr klein, nur kurze Texte'),
    }

    def __init__(self, type='text_embed', model_name=None, device_mode='auto', batch_size=16, max_seq_length=0,
                 fp16=True):
        """
        :param model_name: Repo-ID auf dem Hugging Face Hub, Standard BAAI/bge-m3 (siehe MODELS)
        :param batch_size: Texte je Durchlauf auf GPU/CPU
        :param max_seq_length: Token je Text begrenzen (0: Modellstandard), weniger spart Speicher und Zeit
        :param fp16: auf der GPU in halber Genauigkeit rechnen
        """
        runner.BaseJobExecutor.__init__(self, type)
        self.hf_init(model_name, device_mode)
        self.batch_size = int(batch_size)
        self.max_seq_length = int(max_seq_length)
        self.fp16 = str(fp16).lower() not in ('0', 'false', 'no', '')
        self.define_vars('batch_size', 'max_seq_length', 'fp16')
        self.define_sig(d.Param('text')).set_description("Embedding eines Textes (Dokument)")\
            .r(embedding='list', embedding_model='str', success='bool')

    @property
    def model_cfg(self) -> dict:
        return self.MODELS.get(self.model_name, {})

    def load_model(self):
        from sentence_transformers import SentenceTransformer
        cfg = self.model_cfg
        model = SentenceTransformer(self.model_name, device=self.device,
                                    trust_remote_code=bool(cfg.get('trust_remote_code', False)))
        if self.fp16 and str(self.device).startswith('cuda'):
            model = model.half()
        if self.max_seq_length > 0:
            model.max_seq_length = self.max_seq_length
        self.st_model = model

    def prefix(self, input_type: str) -> str:
        return self.model_cfg.get('query' if input_type == 'query' else 'passage', '')

    def embed(self, texts: list, input_type: str = 'passage') -> list:
        """ :return: auf Länge 1 normierte Vektoren als Listen """
        if not texts:
            return []
        self.load()
        prefix = self.prefix(input_type)
        vectors = self.st_model.encode([prefix + str(t) for t in texts], batch_size=self.batch_size,
                                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(float).tolist()

    @staticmethod
    def as_list(value) -> list:
        if isinstance(value, str):
            s = value.strip()
            if s.startswith('['):
                try:
                    return list(json.loads(s))
                except ValueError:
                    pass
            return [value]
        return list(value or [])

    @runner.op("Embeddings mehrerer Texte, input_type: passage (Dokumente) oder query (Suchanfrage)",
               result={'success': 'bool', 'embeddings': 'list', 'embedding': 'list', 'dims': 'int', 'model': 'str'},
               name='embed')
    def embed_op(self, input, input_type: str = 'passage', model: str = ''):
        """ model wird ignoriert (Kompatibilität mit LMStudio), das Modell ist model_name """
        try:
            embeddings = self.embed(self.as_list(input), input_type)
            return self.success('ok', embeddings=embeddings, embedding=embeddings[0] if embeddings else [],
                                dims=len(embeddings[0]) if embeddings else 0, model=self.model_name)
        except Exception as e:
            return self.fail(str(e), embeddings=[])

    @runner.op("Unterstützte Modelle mit Dimensionen und Hinweisen", result={'success': 'bool', 'models': 'list'})
    def models(self):
        return self.success('ok', models=[dict(model=k, **{x: y for x, y in v.items() if x not in ('query', 'passage')})
                                          for k, v in self.MODELS.items()], current=self.model_name)

    def execute(self, data):
        if 'text' in data and 'op' not in data:
            try:
                embedding = self.embed([data['text']], data.get('input_type', 'passage'))[0]
                return self.success('ok', embedding=embedding, embedding_model=self.model_name)
            except Exception as e:
                return self.fail(str(e))
        return super().execute(data)

    def part_index(self, p: b.Page, params={}):
        if hasattr(self, 'part_hf_nav'):
            self.part_hf_nav(p, 'index')
        p.prop("Modell", self.model_name)
        cfg = self.model_cfg
        if cfg:
            p.prop("Dimensionen", cfg.get('dims'))
            p.prop("Max. Token", self.max_seq_length or cfg.get('max_tokens'))
            p.prop("Hinweis", cfg.get('info', ''))
        p.prop("Geladen", 'ja' if self.hf_loaded else 'nein')
        p('<textarea id="te_text" rows="3" style="width:100%" placeholder="Text"></textarea>')
        p(self.action_btn_parametric("Embedding berechnen", {'type': self.type, 'op': 'embed', 'input': '#te_text'}))
        p.pre('', id='result')


class NsfwDetector(runner.ImageExecutor):
    """
    apt update && apt install -y python-is-python3 python3-pip git
    pip install nsfw-detector nwebclient git+https://github.com/b-oern/nxml --break-system-packages

    -> https://github.com/steelcityamir/safe-content-ai

    {"is_nsfw":false,"confidence_percentage":100.0,"file_name":"current.jpg"}
    """
    MODULES = ['nsfw-detector']
    # model_url = 'https://s3.amazonaws.com/ir_public/ai/nsfw_models/nsfw.299x299.h5'
    model_url = 'https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5'
    model_filename = 'nsfw_detector.h5'

    def __init__(self):
        super().__init__()
        from nsfw_detector import predict
        if not os.path.isfile(self.model_filename):
            print("[NsfwDetector] Downloading model")
            util.download(self.model_url, self.model_filename)
        self.model = predict.load_model(self.model_filename)
        self.define_sig(d.Param('image')).set_description("Bewertet ein Bild auf NSFW-Inhalte")\
            .r(nsfw_detector='dict', neutral='float', porn='float', sexy='float')

    def executeImage(self, image, data):
        from nsfw_detector import predict
        image_preds = predict.classify(self.model, self.image_filename())
        res = list(image_preds.values())[0]
        data['nsfw_detector'] = res
        data['neutral'] = res['neutral']
        data['porn'] = res['porn']
        data['sexy'] = res['sexy']
        return data


class AgeAndGenderRunner(runner.ImageExecutor):
    """
    Schätzt Alter und Geschlecht der Gesichter in einem Bild mit age-and-gender.
    """
    MODULES = ['image', 'age-and-gender']

    MODELS_URL = 'https://raw.githubusercontent.com/b-oern/age-and-gender/master/example/models/'

    def __init__(self, args: util.Args = {}):
        super().__init__('age')
        # Bild: Base64, Pfad oder nweb-ID (image_id); prediction: je Gesicht gender, age und face-Box
        self.define_sig(dev.Param('image', is_pos=False))\
            .set_description("Schätzt Alter und Geschlecht der Gesichter im Bild")\
            .r(prediction='list', s='str', img='str')
        from age_and_gender import AgeAndGender
        # apt install -y wget git curl python3 libjpeg-dev libpng-dev python3-dev python3-pip cmake gcc libx11-dev
        self.data = AgeAndGender()
        predictor_file = args.get('age_and_gender_predictor',
                                  './age-and-gender/example/models/shape_predictor_5_face_landmarks.dat')
        classifier_file = args.get('age_and_gender_classifier',
                                   './age-and-gender/example/models/dnn_gender_classifier_v1.dat')
        dnn_file = args.get('age_and_gender_dnn', './age-and-gender/example/models/dnn_age_predictor_v1.dat')
        self.insure_files_exists({
            predictor_file: 'shape_predictor_5_face_landmarks.dat',
            classifier_file: 'dnn_gender_classifier_v1.dat',
            dnn_file: 'dnn_age_predictor_v1.dat'
        })
        self.data.load_shape_predictor(predictor_file)
        self.data.load_dnn_gender_classifier(classifier_file)
        self.data.load_dnn_age_predictor(dnn_file)

    def insure_files_exists(self, files: dict):
        """
            Lädt fehlende Modelldateien aus https://github.com/b-oern/age-and-gender (example/models) herunter.
            :param files: {lokaler Pfad: Dateiname im Repo}
        """
        for path, name in files.items():
            if os.path.isfile(path):
                continue
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.info(f"Downloading {name} -> {path}")
            util.download(self.MODELS_URL + name, path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file {path} missing, download of {self.MODELS_URL + name} failed")

    def executeImage(self, image, data):
        result = self.data.predict(image)
        # TODO min_age max_age
        return {'prediction': result, 's': str(result), 'img': str(image)}


class ClipEmbeddings(HuggingRunner, runner.BaseJobExecutor):
    """
      Erstellt CLIP-Embeddings für Texte oder Bilder (~0,9 GB VRAM: CLIPModel und CLIPTextModel, fp32).

    """
    TAGS = ['hugging']
    MODULES = ['transformers', 'pillow', 'torch']
    model_name = 'openai/clip-vit-base-patch16'
    text_key = 'text'
    image_key = 'image'
    HF_MODEL_ATTRS = ('tokenizer', 'text_encoder', 'model', 'processor')

    def __init__(self, model_name=None, device_mode='auto'):
        super().__init__()
        self.hf_init(model_name, device_mode)
        self.define_sig(d.Param(self.text_key)).set_description("CLIP-Features und Token-Embeddings eines Textes")\
            .r(features='list', embeddings='list', success='bool')
        self.define_sig(d.Param(self.image_key)).set_description("CLIP-Features eines Bildes (Base64)")\
            .r(features='list', success='bool')
        self.define_sig(d.Param('image_filename')).set_description("CLIP-Features eines Bildes (Pfad)")\
            .r(features='list', success='bool')

    def load_image(self, filename):
        with open(filename, "rb") as f:
            return base64.b64encode(f.read()).decode('ascii')

    @property
    def model_id(self):
        """ früherer Name von model_name """
        return self.model_name

    @property
    def torch_device(self):
        return self.device

    def load_model(self):
        from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel, AutoProcessor
        self.tokenizer = self.hf_pretrained(CLIPTokenizer, to_device=False)
        self.text_encoder = self.hf_pretrained(CLIPTextModel)
        self.model = self.hf_pretrained(CLIPModel)
        self.processor = self.hf_pretrained(AutoProcessor, to_device=False)

    def calculate_image_embedding(self, img):
        self.load()
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return features.cpu().detach().numpy().astype(float)

    def calculate_text_embedding(self, text):
        self.load()
        text_inputs = self.tokenizer([text], padding="max_length", return_tensors="pt").to(self.torch_device)
        text_features = self.model.get_text_features(**text_inputs)
        return text_features.cpu().detach().numpy()

    def execute(self, data):
        # https://huggingface.co/docs/transformers/model_doc/clip
        if not any(k in data for k in ('image_filename', self.text_key, self.image_key)):
            return super().execute(data)
        import torch
        self.load()
        if 'image_filename' in data:
            data[self.image_key] = self.load_image(data['image_filename'])
        if self.text_key in data:
            text_inputs = self.tokenizer([data[self.text_key]], padding="max_length", return_tensors="pt").to(
                self.torch_device)
            text_features = self.model.get_text_features(**text_inputs)
            text_embeddings = torch.flatten(
                self.text_encoder(text_inputs.input_ids.to(self.torch_device))['last_hidden_state'], 1, -1)
            data['features'] = text_features.cpu().detach().numpy().astype(float).tolist()
            data['embeddings'] = text_embeddings.cpu().detach().numpy().astype(float).tolist()
            data['success'] = True
        elif self.image_key in data:
            from PIL import Image
            image_data = base64.b64decode(data[self.image_key])
            img = Image.open(io.BytesIO(image_data))
            data['features'] = self.calculate_image_embedding(img).tolist()
            data['success'] = True
        return data


class TextBlobRunner(runner.BaseJobExecutor):
    """
    Bestimmt Sentiment (Polarität, Subjektivität) und Sätze eines deutschen oder englischen Textes mit TextBlob.
    """
    MODULES = ['textblob', 'textblob_de']
    type = 'textblob'
    text_key = 'text'

    def __init__(self):
        super().__init__()
        self.define_sig(d.Param(self.text_key), d.PStr('lang', 'en'))\
            .set_description("Sentiment-Analyse und Satzzerlegung (lang: en oder de)")\
            .r(sentimnet='dict', sentimnet_polarity='float', sentimnet_subjectivity='float', sentences='list',
               success='bool')

    def get_textblob(self, text, lang='en'):
        if lang == 'de':
            from textblob_de import TextBlobDE
            blob = TextBlobDE(text)
        else:
            from textblob import TextBlob
            blob = TextBlob(text)
        return blob

    def execute(self, data):
        # https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.BaseBlob
        text = data[self.text_key]
        blob = self.get_textblob(text, data.get('lang', 'en'))
        data['sentimnet'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        data['sentimnet_polarity'] = blob.sentiment.polarity
        data['sentimnet_subjectivity'] = blob.sentiment.subjectivity
        data['sentences'] = list(map(lambda x: str(x), blob.sentences))
        data['success'] = True
        return data

    def page(self, params):
        p = b.Page(owner=self)
        p.h2("TextBlobRunner")
        p('<form>')
        p('<input type="hidden" name="type" value="' + self.type + '" />')
        p('<input type="text" name="text" value="" />')
        p('<input type="submit" name="submit" value="Execute" />')
        p('</form>')
        if 'text' in params:
            r = self.execute(params)
            p.pre(json.dumps(r))
        return p.nxui()


# https://realpython.com/natural-language-processing-spacy-python/
# Named Entity Recognition
# nlp = spacy.load("en_core_web_sm")
# https://spacy.io/models/de


class NlpPipeline(runner.Pipeline):
    """
    Führt Flair-NER, NLTK-Tokenisierung und TextBlob-Sentiment nacheinander auf einem Text aus.
    """
    def __init__(self):
        super().__init__(FlairRunner(), Nltk(), TextBlobRunner())


class AnalyseMain(runner.AutoDispatcher):
    """
    Leitet Jobs anhand von type an NLP-Pipeline, CLIP-/BERT-Embeddings oder Alters- und Geschlechtserkennung weiter.
    """
    def __init__(self):
        super().__init__('type',
                         **self.create('nlp', NlpPipeline),
                         **self.create('clip_embeddings', ClipEmbeddings),
                         **self.create('bert_embeddings', BertEmbeddings),
                         **self.create('age_and_gender', AgeAndGenderRunner)
                         )

    def create(self, key, cls):
        res = {}
        try:
            res[key] = cls()
        except ModuleNotFoundError as e:
            print("")
            print("Error: " + str(e))
            print("Key:   " + key)
            print("")
        return res


if __name__ == '__main__':
    runner.main(NlpPipeline())
