
import sys
import json
import os.path

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

model = "all-MiniLM-L6-v2"

def create_embeddings(infile):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model)
        result = {'jobs': []}
        data = json.load(open(infile))
        for job in data['jobs']:
            embedding = embeddings.embed_document(job['text'])
            job['embedding'] = embedding
            job['embedding_model'] = model
            result['jobs'].append(job)
        print(json.dumps(result))
    except Exception as e:
        print("Error: " + str(e))
        print("Faild to execute JSON-File "+str(infile))

if __name__ == '__main__':
    if len(sys.argv)>1:
        infile = sys.argv[1]
        create_embeddings(infile)
        
