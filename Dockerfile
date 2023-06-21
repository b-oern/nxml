FROM ubuntu

COPY setup_ubuntu.sh .
RUN sh setup_ubuntu.sh

COPY setup_langchain
RUN chmod +x setup_langchain

RUN pwd
