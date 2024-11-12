#
# ~/dev/nxml/$ docker build -t document-analysis -f docker/DocumentAnalysis.Dockerfile .
# $ docker run --rm -it -p 27201:7070 document-analysis
#
FROM nxware/nxdev
COPY document_analysis.sh /document_analysis.sh
RUN sh /document_analysis.sh

EXPOSE 7070
CMD ["python", "-m", "nwebclient.runner", "--rest", "--executor", "nxml.image:DocumentAnalysis"]