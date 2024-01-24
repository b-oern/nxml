FROM ubuntu

COPY setup_ubuntu.sh .
RUN sh setup_ubuntu.sh

COPY setup_langchain .
RUN chmod +x setup_langchain
COPY create_embeddings .
RUN chmod +x create_embeddings
COPY embeddings.py .
COPY nxml .

RUN pwd
RUN ls
RUN cd nxml && pip install .

CMD ["python", "-m", "nwebclient.runner", "--rest", "--executor"]
