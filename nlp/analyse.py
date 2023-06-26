


# pip install detoxify
from detoxify import Detoxify
classifier_toxity = Detoxify('multilingual')

from langdetect import detect

from nltk import word_tokenize


# https://realpython.com/natural-language-processing-spacy-python/
# Named Entity Recognition
# nlp = spacy.load("en_core_web_sm")
# https://spacy.io/models/de

def nltk_lang(lang):
    if lang == 'de':
        return 'german'
    return 'english'

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
    data['toxity'] = classifier_toxity.predict(text)
    #{'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
    data['lang'] = detect(text)
    data['words'] = word_tokenize(text, language=nltk_lang(data['lang']))
    data = analyse_textblob(data)
    return data


if __name__ == '__main__':
    from nwebclient import runner
    runner.main(analyse_text)
