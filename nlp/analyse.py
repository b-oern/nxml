
from langdetect import detect
from nwebclient import runner

class Toxity(runner.BaseJobExecutor):
    MODULES=['detoxify']
    def __init__(self):
        from detoxify import Detoxify
        self.classifier_toxity = Detoxify('multilingual')
    def execute(self, data):
        if isinstance(data, str):
            data = {'text': data}
        data['toxity'] = classifier_toxity.predict(text)
        #{'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
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
        if isinstance(data, str):
            data = {'text': data}
        data['words'] = word_tokenize(data['text'], language=self.nltk_lang(data['lang']))
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
        return data


runner = runner.Pipeline(FlairRunner(), Nltk())

# https://realpython.com/natural-language-processing-spacy-python/
# Named Entity Recognition
# nlp = spacy.load("en_core_web_sm")
# https://spacy.io/models/de

def get_textblob(text, lang):
    if lang == 'de':
        from textblob_de import TextBlobDE 
        blob = TextBlobDE(text)
    else:
        from textblob import TextBlob
        blob = TextBlob(text)
    return blob

def analyse_textblob(data, text_key = 'text'):
    # https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.BaseBlob
    text = data[text_key]
    blob = get_textblob(text, data['lang'])
    data['sentimnet'] = {
        'polarity': blob.sentimnet.polarity,
        'subjectivity': blob.sentimnet.subjectivity
    }
    data['sentences'] = map(lambda x: str(x), blob.sentences)
    return data


def analyse_text(data, text_key = 'text'):
    text = data[text_key]
    data['lang'] = detect(text)
    data = analyse_textblob(data)
    return data


if __name__ == '__main__':
    from nwebclient import runner
    runner.main(analyse_text)
