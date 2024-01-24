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
        description="NWebClient via HTTP",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://bsnx.net/4.0/group/pynwebclient",
        packages=['nxml'],
        entry_points={
            'console_scripts': []
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires=["usersettings>=1.0.7"]
    )
