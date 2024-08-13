#
# ~/dev/nxml/docker$ docker build -t document-analysis -f DocumentAnalysis.Dockerfile ..
#
FROM nxware/nxdev
COPY docker/document_analysis.sh /document_analysis.sh
RUN sh /document_analysis.sh

EXPOSE 7070
CMD ["python", "-m", "nwebclient.runner", "--rest", "--executor", "nxml.image:DocumentAnalysis"]