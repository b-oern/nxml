
from nwebclient import runner
from nwebclient import util
import os
import os.path
import base64
import io
import requests

class Whisper(runner.BaseJobExecutor):
    MODULES=['openai-whisper']
    def __init__(self):
	    # TODO search for ffmpeg in path
        import whisper
        self.model = whisper.load_model("base")
    def execute(self, data):
        if not 'audio' in data and 'file0' in data:
            util.file_put_contents('audiofile.dat', base64.b64decode(data['file0']))  
            data['audio'] = 'audiofile.dat'
        result = self.model.transcribe(data['audio'])
        #print(str(result))
        data['text'] = ""+str(result['text'])
        return data

class Toxity(runner.BaseJobExecutor):
    MODULES=['detoxify']
    text_key = 'text'
    def __init__(self):
        from detoxify import Detoxify
        self.classifier_toxity = Detoxify('multilingual')
    def execute(self, data):
        if isinstance(data, str):
            data = {'text': data}
        text = data[self.text_key]
        tv =self.classifier_toxity.predict(text)
        data['toxity'] = dict(map(lambda kv: (kv[0], float(kv[1])), tv.items()))
        #{'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
        data['success'] = True
        return data
    

class Nltk(runner.BaseJobExecutor):
    MODULES=['nltk']
    def __init__(self):
        import nltk
        try:
            nltk.find('punkt')
        except LookupError:
            nltk.download('punkt')
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
    MODULES = ['flair']
    def __init__(self):
        try:
            from flair.models import SequenceTagger
            self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
        except (ImportError, ModuleNotFoundError) as e:
            print("Error: Init FlairRunner, " + str(e))
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
        return { 'start_position': span.start_position, 
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
            data['error'] = 'Flair Error '  + str(e)
        return data


class BertEmbeddings(runner.BaseJobExecutor):
    MODULES = ['transformers', 'langchain', 'sentence_transformers']
    model = "all-MiniLM-L6-v2"
    def __init__(self):
        from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model)
    def execute(self, data):
        try:
            embedding = self.embeddings.embed_documents([data['text']])
            data['embedding'] = embedding[0]
            data['embedding_model'] = self.model
            data['success'] = True
        except Exception as e:
            print("Error: " + str(e))
            data['success'] = False
        return data


class NsfwDetector(runner.ImageExecutor):
    MODULES = ['nsfw-detector']
    #model_url = 'https://s3.amazonaws.com/ir_public/ai/nsfw_models/nsfw.299x299.h5'
    model_url = 'https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5'
    model_filename = 'nsfw_detector.h5'
    def __init__(self):
        from nsfw_detector import predict
        if not os.path.isfile(self.model_filename):
            print("[NsfwDetector] Downloading model")
            util.download(self.model_url, self.model_filename)
        self.model = predict.load_model(self.model_filename)
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
    def __init__(self, args: util.Args = {}):
        from age_and_gender import AgeAndGender
        # apt install -y wget git curl python3 libjpeg-dev libpng-dev python3-dev python3-pip cmake gcc libx11-dev
        #os.system('git clone https://github.com/b-oern/age-and-gender.git')
        self.data = AgeAndGender()
        predictor_file = args.get('age_and_gender_predictor', './age-and-gender/example/models/shape_predictor_5_face_landmarks.dat')
        classifier_file = args.get('age_and_gender_classifier', './age-and-gender/example/models/dnn_gender_classifier_v1.dat')
        dnn_file = args.get('age_and_gender_dnn', './age-and-gender/example/models/dnn_age_predictor_v1.dat')
        if os.path.isfile(predictor_file):
            self.data.load_shape_predictor(predictor_file)
            self.data.load_dnn_gender_classifier(classifier_file)
            self.data.load_dnn_age_predictor(dnn_file)

    def executeImage(self, image, data):
        result = self.data.predict(image)
	print(result)
	return dict(result)
    

class ClipEmbeddings(runner.BaseJobExecutor):
    """
      Erstellt Embeddings für Texte oder Bild

    """
    MODULES = ['transformers', 'pillow']
    model_id = 'openai/clip-vit-base-patch16'
    text_key = 'text'
    image_key = 'image'
    def __init__(self):
        import torch
        from torch.nn import CosineSimilarity
        from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel, AutoProcessor
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id).to(self.torch_device)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.torch_device)
        self.img_model = CLIPModel.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    def load_image(self, filename):
        with open(filename, "rb") as f:
            return base64.b64encode(f.read()).decode('ascii')
    def execute(self, data):
        # https://huggingface.co/docs/transformers/model_doc/clip
        import torch
        if 'image_filename' in data:
            data[self.image_key] = self.load_image(data['image_filename'])
        if self.text_key in data:
            text_inputs = self.tokenizer([data[self.text_key]], padding="max_length", return_tensors="pt").to(self.torch_device)
            text_features = self.model.get_text_features(**text_inputs)
            text_embeddings = torch.flatten(self.text_encoder(text_inputs.input_ids.to(self.torch_device))['last_hidden_state'],1,-1)
            data['features'] = text_features.cpu().detach().numpy().astype(float).tolist()
            data['embeddings'] = text_embeddings.cpu().detach().numpy().astype(float).tolist()
            data['success'] = True
        elif self.image_key in data:
            from PIL import Image
            image_data = base64.b64decode(data[self.image_key])
            img = Image.open(io.BytesIO(image_data))
            inputs = self.processor(images=img, return_tensors="pt")
            features = self.img_model.get_image_features(**inputs)
            data['features'] = features.cpu().detach().numpy().astype(float).tolist()
            data['success'] = True
        return data

class TextBlobRunner(runner.BaseJobExecutor):
    MODULES = ['textblob', 'textblob_de']
    text_key = 'text'
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
        blob = self.get_textblob(text, data['lang'])
        data['sentimnet'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        data['sentimnet_polarity'] = blob.sentiment.polarity
        data['sentimnet_subjectivity'] = blob.sentiment.subjectivity
        data['sentences'] = list(map(lambda x: str(x), blob.sentences))
        data['success'] = True
        return data

# https://realpython.com/natural-language-processing-spacy-python/
# Named Entity Recognition
# nlp = spacy.load("en_core_web_sm")
# https://spacy.io/models/de

class NlpPipeline(runner.Pipeline):
    def __init__(self):
        super().__init__(FlairRunner(), Nltk(), TextBlobRunner())

class AnalyseMain(runner.AutoDispatcher):
    def __init__(self):
        super().__init__('type', **{
            'nlp': NlpPipeline(),
            'clip_embeddings': ClipEmbeddings(),
            'bert_embeddings':BertEmbeddings(),
            'age_and_gender': AgeAndGenderRunner()
        })

if __name__ == '__main__':
    runner.main(NlpPipeline())
