import setuptools

version = "1.0.10"

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
                'aesthetics = nxml.image:Aesthetics',
                'summarize = nxml.nlp:TextSummarization',
                'mail_respond = nxml.nlp:MailRespond',
                'document = nxml.image:DocumentAnalysis',
                'document_dockered = nxml.image:DocumentAnalysisDockerd',
                'rllm = nxml.llm:RLLM',
                'ollm = nxml.llm:OLLama',
                'ollmd = nxml.llm:OLLamaDockerd',
                'gptllm = nxml.llm:OpenAiLLM',
                'cohere = nxml.llm:CohereLlm',
                'gemini = nxml.llm:Gemini',
                'vision = nxml.llm:Vision',
                'tt = nxml.llm:TransformText',
                'llmproxy = nxml.proxy:LlmProxy',
                'whisper = nxml.analyse:Whisper',
                'tts = nxml.audio:ElevenLabs',
                'pipertts = nxml.audio:PiperTTS',
                'face_similarity = nxml.vision:FaceSimilarity',
                'imap = nxml.data:Imap'
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            "usersettings>=1.0.7",
            "docker",
            "ollama"
        ]
    )
