


# pip install detoxify
from detoxify import Detoxify
classifier_toxity = Detoxify('multilingual')

from langdetect import detect

from nltk import word_tokenize


def nltk_lang(lang):
    if lang == 'de':
        return 'german'
    return 'english'


def analyse_text(data, text_key = 'text'):
    text = data[text_key]
    data['toxity'] = classifier_toxity.predict(text)
    #{'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
    data['lang'] = detect(text)
    data['words'] = word_tokenize(text, language=nltk_lang(data['lang']))
    return data

