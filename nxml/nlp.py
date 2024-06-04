


from nwebclient import NWebClient
from nwebclient import runner as r
from nwebclient import base as b


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
    https://huggingface.co/facebook/bart-large-mnli
    """
    def __init__(self):
        from transformers import pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify(self, text, classes):
        self.classifier(text, classes, multi_label=True)

    def run_group(self, **data):
        nc = NWebClient(data.get('nweb', None))
        classes = data['classes']
        if isinstance(classes, str):
            classes = classes.split(',')
        for doc in nc.group(data['group']):
            result = self.classify(doc.content(), classes)
            print(f"{doc.title()}: {result['labels'][0]} {result['scores'][0]}")

    def execute(self, data):
        if 'group' in data:
            self.run_group(**data)
        return super().execute(data)
