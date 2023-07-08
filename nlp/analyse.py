
from nwebclient import runner
import base64

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
        from flair.models import SequenceTagger
        self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
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
        if isinstance(data, str):
            data = {'text': data}
        data['ner'] = self.tag_text(data['text'])
        data['success'] = True
        return data


class BertEmbeddings():
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

class ClipEmbeddings():
    MODULES = ['transformers', 'pillow']
    model = 'openai/clip-vit-base-patch16'
    text_key = 'text'
    image_key = 'image'
    def __init__(self):
        import torch
        from torch.nn import CosineSimilarity
        from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel, CLIPProcessor
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(torch_device)
        self.model = CLIPModel.from_pretrained(model_id).to(torch_device)
        self.processor = CLIPProcessor.from_pretrained(self.model)
    def load_image(self, filename):
        with open(filename, "rb") as f:
            return base64.b64encode(f.read())
    def execute(self, data):
        # https://huggingface.co/docs/transformers/model_doc/clip
        if 'image_filename' in data:
            data[image_key] = self.load_image(data['image_filename'])
        if self.text_key in data:
            text_inputs = self.tokenizer([data[self.text_key]], padding="max_length", return_tensors="pt").to(torch_device)
            text_features = self.model.get_text_features(**text_inputs)
            text_embeddings = self.torch.flatten(self.text_encoder(text_inputs.input_ids.to(torch_device))['last_hidden_state'],1,-1)
            data['features'] = text_features
            data['embeddings'] = text_embeddings
            data['success'] = True
        elif self.image_key in data:
            from PIL import Image
            image_data = base64.b64decode(data[self.image_key])
            img = Image.open(image_data)
            inputs = self.processor(images=img, return_tensors="pt", padding=True)
            features = self.model.get_image_features(**inputs)
            data['features'] = features
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
        super().__init__(FlairRunner(), Nltk(), TextBlobRunner(), Toxity())

if __name__ == '__main__':
    runner.main(NlpPipeline())
