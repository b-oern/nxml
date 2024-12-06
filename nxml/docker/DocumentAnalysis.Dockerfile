#
# ~/dev/nxml/$ docker build -t document-analysis -f docker/DocumentAnalysis.Dockerfile .
# $ docker run --rm -it -p 27201:7070 document-analysis
#
FROM nxware/nxdev
COPY document_analysis.sh /document_analysis.sh
COPY document_analysis_start.sh /document_analysis_start.sh
RUN sh /document_analysis.sh

EXPOSE 7070
CMD ["sh", "/document_analysis_start.sh"]