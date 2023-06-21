
import sys
import json
import os.path

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

model = "all-MiniLM-L6-v2"

def create_embeddings(infile, outfile = None):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model)
        result = {'jobs': []}
        data = json.load(open(infile))
        for job in data['jobs']:
            embedding = embeddings.embed_documents([job['text']])
            job['embedding'] = embedding[0]
            job['embedding_model'] = model
            result['jobs'].append(job)
        outcontent = json.dumps(result)
        print(outcontent)
        if not outfile is None:
            with open(outfile, 'w') as f:
                f.write(outcontent)
    except Exception as e:
        print("Error: " + str(e))
        print("Faild to execute JSON-File "+str(infile))

if __name__ == '__main__':
    if len(sys.argv)>2:
        infile = sys.argv[1]
        outfile = sys.argv[2]
        create_embeddings(infile)
    else:
        print("Usage: infile outfile")
