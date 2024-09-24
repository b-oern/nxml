#
# ~/dev/nxml/$ docker build -t document-analysis -f docker/DocumentAnalysis.Dockerfile .
# $ docker run --rm -it -p 7070:7070 -w /yolov10 document-analysis
#
FROM nxware/nxdev
COPY docker/document_analysis.sh /document_analysis.sh
RUN sh /document_analysis.sh

EXPOSE 7070
CMD ["python", "-m", "nwebclient.runner", "--rest", "--executor", "nxml.image:DocumentAnalysis"]