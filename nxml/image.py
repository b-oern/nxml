#
# Project: nxml
#

from threading import Thread
import json
import os.path
import time
import glob
from scipy.spatial.distance import cosine

from nwebclient import runner as r
from nwebclient import NWebClient
from nwebclient import NWebDoc
from nwebclient import util as u

from nxml import analyse


class DatasetWriter:
    """
     für clip retrival
     Erstellt dataset nach https://github.com/rom1504/img2dataset
    """
    index = 0
    output_folder = './'

    def __init__(self, output_folder = './'):
        self.output_folder = output_folder

    def create_metadata(self, img:NWebDoc, fname:str):
        width, height = img.as_image().size
        return {
            'url': img.url(),
            'caption': img.title(),
            'key': fname, # key of the form 000010005 : the first 5 digits are the shard id, the last 4 are the index in the shard
            'status': 'success',
            'error_message': '',
            'width': width,
            'height': height,
            'original_width': width,
            'original_height': height,
            'exif': {}
            #"sha256"
        }

    def write_image(self, img:NWebDoc):
        fname = "{:9d}".format(self.index)
        img.save(self.output_folder + fname+'.jpg')
        with open(self.output_folder + fname+'.json', "w") as f:
            json.dump(self.create_metadata(img, fname), f)
        with open(self.output_folder + fname + '.txt', "w") as f:
            f.write(img.title())
        self.index += 1

class ImageEmbeddingCreator(r.BaseJobExecutor):
    """
        Erstellt Embeddings
    """
    MODULES = ['numpy', 'autofaiss']

    type = 'img_embedding'

    embedding_folder = './'
    embedder = None

    embeddings = {}

    pairs = []

    threshold = 0.95

    knn = None
    knn_ids = {}

    def __init__(self, embedding_folder='./', args:u.Args={}):
        self.var_names.append("embedding_folder")
        self.var_names.append("threshold")
        if self.embedding_folder == embedding_folder and args.get('embedding_folder', None) is not None:
            self.embedding_folder = args.get('embedding_folder', None)
        else:
            self.embedding_folder = embedding_folder
        self.embedder = analyse.ClipEmbeddings()
        self.embeddings = {}
        self.nc = NWebClient(None)
        self.knn = None
        self.knn_ids = None

    def extract_guid(self, file):
        a = file.split('/')
        return a[-1].replace('.npy', '')

    def load_embeddings(self):
        import numpy as np
        for file in glob.glob(self.embedding_folder+"*.npy"):
            if 'embeddings' not in file:
                g = self.extract_guid(file)
                self.embeddings[g] = np.load(file)

    def numpy_all(self):
        import numpy as np
        array = []
        index = {}
        i = 0
        for key in self.embeddings:
            a = self.embeddings[key].tolist()[0]
            array.append(a)
            index[i] = key
            i += 1
        a = np.array(array)
        np.save(self.embedding_folder + 'embeddings', a)
        with open("index.json", 'w') as f:
            json.dump(index, f)
        return a

    def build_knn_index(self):
        """
        https://github.com/criteo/autofaiss
        :return:
        """
        from autofaiss import build_index
        import numpy as np
        embeddings = np.load(self.embedding_folder + 'embeddings.npy')
        build_index(embeddings=embeddings, index_path="knn.index", index_infos_path="index_infos.json", max_index_memory_usage="4G", current_memory_available="4G")

    def load_knn(self):
        import faiss
        self.knn = faiss.read_index("knn.index")
        with open("index.json", "r") as f:
            self.knn_ids = json.load(f)

    def knn_q_text(self, q, k=5):
        embedding = self.embedder.calculate_text_embedding(q)
        return self.knn_q(embedding, k)

    def knn_q(self, embedding, k=5):
        if self.knn is None:
            self.load_knn()
        res = []
        distances, indices = self.knn.search(embedding, k)
        for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
            self.info(f"{i + 1}: Vector number {indice:4} with distance {dist}")
            guid = self.knn_ids[str(indice)]
            res.append({'guid': guid, 'distance': dist, 'i': i+1,    'id': guid, 'score': dist})
        return res

    def similarity(self, embedding_a, embedding_b):
        return 1 - cosine(embedding_a, embedding_b)

    def similarity_matrix(self):
        """
           ~40/Minute
        :return:
        """
        i = 0
        count = len(self.embeddings.keys())
        processed = [] # indexed list verwenden
        for a in self.embeddings.keys():
            processed.append(a)
            for b in self.embeddings.keys():
                if a != b and b not in processed:
                    score = self.similarity(self.embeddings[a][0], self.embeddings[b][0])
                    if self.threshold < score:
                        self.pairs.append({'a': a, 'b': b, 'score': score})
            i += 1
            if i % 20 == 0:
                self.info(f"{i}/{count} processed. Found: {len(self.pairs)}")
                time.sleep(0.1)

    def create_embedding(self, img: NWebDoc):
        import numpy as np
        try:
            guid = img.guid()
            self.info("Create Embedding: " + guid)
            numpy_array = self.embedder.calculate_image_embedding(img.as_image())
            np.save(self.embedding_folder + guid + '.npy', numpy_array)
            self.embeddings[guid] = numpy_array.tolist()
        except Exception as e:
            self.error("Failed to create embedding: " + str(e))

    def inference(self, q=''):
        """
        Auf dem Rpi4 ~1500 Bilder/Stunde
        Etwa 100MB pro 10.000 Embeddings
        :param q:string e.g. 'limit=10000'
        :return:
        """
        i = 0
        for img in self.nc.images(q):
            if not os.path.isfile(self.embedding_folder + img.guid() + '.npy'):
                self.create_embedding(img)
                time.sleep(0.2)
        return {'inference': 'done', 'success': True}

    def load_pairs(self):
        with open("pairs.json", 'r') as f:
            self.pairs = json.load(f)

    def save_pairs(self):
        with open("pairs.json", 'w') as f:
            json.dump(self.pairs, f)

    def publish_pairs(self):
        i = 0
        for pair in self.pairs:
            self.publish_recomendations(pair['a'], pair['b'], pair['score'])
            self.publish_recomendations(pair['b'], pair['a'], pair['score'])
            i += 1
            if i % 100 == 0:
                self.info(f"{i}/{len(self.pairs)} processed.")

    def publish_recomendations(self, guid_from, guid_to, score):
        f = self.nc.doc(guid_from)
        t = self.nc.doc(guid_to)
        f.setMetaValue('image_similarity', str(t.id()), str(score))

    def execute(self, data):
        if 'inference' in data:
            return self.inference()
        elif 'q' in data:
            q = data['q'] if data['q'] != '' else data['text']
            return self.knn_q_text(q, int(data.get('k', 5)))
        return super().execute(data)


class ImageSimilarity(r.ImageExecutor):
    """
      DocMap in nwebclient.nc

      via https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3
    """

    MODULES = ['opencv-python', 'scikit-image']
    NS = 'image_similarity'
    type = 'image_similarity'

    comparators = {
        'clip': 'compareImagesCLIP',
        'ssim': 'compareImagesSSIM'
    }

    threshold = 0.9

    def compareImagesSSIM(self, image_a, image_b):
        import cv2
        from skimage import metrics
        import numpy
        try:
            # Load images
            #image1 = cv2.imread(image_a)
            #image2 = cv2.imread(image_b)
            image1 = cv2.cvtColor(numpy.array(image_a), cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(numpy.array(image_b), cv2.COLOR_RGB2BGR)
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
            # print(image1.shape, image2.shape)
            # Convert images to grayscale
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            # Calculate SSIM
            ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
            #print(f"SSIM Score: ", round(ssim_score[0], 2))
            return round(ssim_score[0], 2)
        except Exception as e:
            self.error(str(e))
            return 0

    def compareImagesCLIP(self, image_a, image_b):
        # !pip install git+https://github.com/openai/CLIP.git
        # !pip install open_clip_torch
        # !pip install sentence_transformers
        import torch
        import open_clip
        import cv2
        from sentence_transformers import util
        from PIL import Image
        # image processing model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
        model.to(device)

        def imageEncoder(img):
            img1 = preprocess(img).unsqueeze(0).to(device)
            img1 = model.encode_image(img1)
            return img1

        def generateScore(image1, image2):
            img1 = imageEncoder(image1)
            img2 = imageEncoder(image2)
            cos_scores = util.pytorch_cos_sim(img1, img2)
            score = round(float(cos_scores[0][0]) * 100, 2)
            return score

        #print(f"similarity Score: ", round(generateScore(image_a, image_b), 2))
        return round(generateScore(image_a, image_b), 2)/100

    def compareImages(self, image_a, image_b, algo='clip'):
        method = getattr(self, self.comparators[algo])
        return method(image_a, image_b)

    def searchSimilar(self, image, data):
        n = NWebClient(None)
        result = {'images': [], 'nweb': n.url()}
        q = data['search']
        if 'kind=image' not in q:
            q += '&kind=image'
        docs = n.docs(q)
        self.info(f"Calculating Similarity with {len(docs)} Images")
        result['image_count'] = len(docs)
        for d in docs:
            if d.is_image():
                try:
                    img_b = d.as_image()
                    d.similarity = self.compareImagesSSIM(image, img_b)
                    self.info("Similarity: "+str(d.similarity))
                except:
                    d.similarity = 0
                    self.error("Similarity Error on Image: "+str(d.id()))
        docs.sort(key=lambda x: x.similarity, reverse=True)
        i = 1
        for d in docs:
            self.info(f"{i}: {d.name()} id={d.id()} (Similarity: {d.similarity})")
            result['images'].append({'id': d.id(), 'score': d.similarity})
            i += 1
            if i > 5:
                break
        self.info(str(result))
        return result

    def index(self, data=dict()):
        self.n = NWebClient(None)
        q = 'kind=image&no_meta='+self.NS+'_cfg'+'.max_id&limit='+str(data.get('limit', 10))
        if 'q' in data:
            q += '&' + data['q']
        docs = self.n.docs(q)
        for d in docs:
            self.do_doc(d)
        return {'from': 'index()'}

    def get_docs(self):
        docs = self.n.docs('kind=image&limit=1000000')
        return docs

    def do_doc(self, doc):
        # Ein Bild mit allen anderen vergleichen
        img_a = doc.as_image()
        max_id = 0
        i = 0
        hits = 0
        score = 0
        for b in self.get_docs():
            i += 1
            if b.id() != doc.id():
                try:
                    max_id = max(max_id, b.id())
                    score = self.compareImages(img_a, b.as_image())
                    if score > self.threshold:
                        hits += 1
                        doc.setMetaValue(self.NS, str(b.id()), score)
                except:
                    self.error("Similarity Error on Image: " + str(b.id()))
            if i % 100 == 0:
                self.info(f"At {i} Current Score: {score}  Doc: {b.id()}")
        doc.setMetaValue(self.NS+'_cfg', 'max_id', max_id)
        self.info(f"Done Doc ({doc.id()}) Hits: {hits}")
        return hits

    def execute(self, data):
        if 'index' in data:
            return self.index(data)
        elif 'index_async' in data:
            self.thread = Thread(target=lambda: self.index({}))
            self.thread.start()
            return {}
        else:
            return super().execute(data)


    def executeImage(self, image, data):
        b = self.get_image('b', data)
        if b is not None:
            score = self.compareImagesSSIM(image, b)
            return {'success': True, 'score': score}
        if 'search' in data:
            return self.searchSimilar(image, data)
        else:
            return {}

