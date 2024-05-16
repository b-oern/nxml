#
# Project: nxml
#

from threading import Thread
import json

from nwebclient import runner as r
from nwebclient import NWebClient
from nwebclient import NWebDoc


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
    """ Erstellt Embeddings """

    embedding_folder = '.'

    def __init__(self, embedding_folder='.'):
        self.embedding_folder = embedding_folder

    def create_embedding(self, img: NWebDoc):
        guid = img.guid()
    def execute(self, data):
        #nc = NWebClient(None)
        #nc.images()
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

    threshold = 0.5

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

