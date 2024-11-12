import setuptools

version = "1.0.5"

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setuptools.setup(
        name="nxml",
        version=version,
        author="Bjoern Salgert",
        author_email="bjoern.salgert@hs-duesseldorf.de",
        description="Executeable Machine Learning",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://bsnx.net/4.0/group/pynwebclient",
        packages=['nxml'],
        package_data={'nxml': ['docker/*']},
        include_package_data=True,
        entry_points={
            'console_scripts': [],
            'nweb_runner': [
                'nlp = nxml.analyse:NlpPipeline',
                'toxity = nxml.analyse:Toxity',
                'flair = nxml.analyse:FlairRunner',
                'bert = nxml.analyse:BertEmbeddings',
                'nsfw = nxml.analyse:NsfwDetector',
                'age = nxml.analyse:AgeAndGenderRunner',
                'clip = nxml.analyse:ClipEmbeddings',
                'textblob = nxml.analyse:TextBlobRunner',
                'librosa = nxml.audio:LibRosa',
                'image_similarity = nxml.image:ImageSimilarity',
                'speech_reg = nxml.audio:CommandRecognition',
                'qa = nxml.nlp:QuestionAnswering',
                'text_classifier = nxml.nlp:TextClassifier',
                'ttt = nxml.nlp:TextToText',
                'tw = nxml.nlp:TextWorker',
                'od = nxml.image:ObjectDetector',
                'ic = nxml.image:ImageClassifier',
                'summarize = nxml.nlp:TextSummarization',
                'mail_respond = nxml.nlp:MailRespond',
                'document = nxml.image:DocumentAnalysis',
                'document_dockered = nxml.image:DocumentAnalysisDockerd',
                'rllm = nxml.llm:LLM',
                'whisper = nxml.analyse:Whisper',
                'document_dockered = nxml.image:DocumentAnalysisDockerd',
                'tts = nxml.audio:ElevenLabs'
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires=["usersettings>=1.0.7"]
    )
