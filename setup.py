import setuptools

version = "1.0.1"

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
                'image_similarity = nxml.image:ImageSimilarity'
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires=["usersettings>=1.0.7"]
    )
