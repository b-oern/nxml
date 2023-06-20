FROM ubuntu

COPY setup_ubuntu.sh .
RUN sh setup_ubuntu.sh
RUN pwd
