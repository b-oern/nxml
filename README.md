# NXML

```
pip install git+github.com/b-oern/nxml
```

## NLP

Analyisert einen Text.

Folgende Features werden ermittelt

 - `lang` Sprache (`en`, `de`, ...)
 - `toxity` Wie viel Hate steckt im Text, unterteilt in {'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
 - `words` Array von Wörtern