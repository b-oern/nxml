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
from nwebclient import base as b

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

class ImageEmbeddingCreator(r.ImageExecutor):
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
        self.knn = faiss.read_index(self.embedding_folder + "knn.index")
        with open(self.embedding_folder + "index.json", "r") as f:
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
            res.append({'guid': guid, 'distance': float(dist), 'i': i+1,    'id': guid, 'score': float(dist)})
        return {'images': res}

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

    def query_image(self, data, image_key='image'):
        img = self.get_image(image_key, data)
        embedding = self.embedder.calculate_image_embedding(img)
        return self.knn_q(embedding, int(data.get('k', 5)))

    def execute(self, data):
        if 'inference' in data:
            return self.inference()
        elif 'q' in data:
            q = data['q'] if data['q'] != '' else data['text']
            return self.knn_q_text(q, int(data.get('k', 5)))
        elif 'image' in data:
            return self.query_image(data, 'image')
        elif 'file0' in data:
            return self.query_image(data, 'file0')
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
            return {'success': False, 'message': 'ImageSimilarity.executeImage'}


class ObjectDetector(r.ImageExecutor):
    """
    https://huggingface.co/IDEA-Research/grounding-dino-tiny

    https://huggingface.co/spaces/EduardoPacheco/Grounding-Dino-Inference/blob/main/app.py
    """
    type = 'od'

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)


    def executeImage(self, image, data):
        import torch
        # Check for cats and remote controls
        # VERY important: text queries need to be lowercased + end with a dot
        text = data.get('text', "a cat. a remote control.")

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # convert tensor of [x0,y0,x1,y1] to list of [x0,y0,x1,y1] (int)
        boxes = results["boxes"].int().cpu().tolist()
        pred_labels = results["labels"]
        annot = [(tuple(box), pred_label) for box, pred_label in zip(boxes, pred_labels)]
        return {
            'boxes': boxes,
            'annotations': annot,
            'labels': pred_labels,
            'request_text': text
        }



class ImageClassifier(r.ImageExecutor):
    """
        https://github.com/pharmapsychotic/clip-interrogator
    """

    MODULES = ['clip_interrogator']

    type = 'ic'

    classes = []

    def __init__(self, classes=[]):
        from clip_interrogator import Config, Interrogator, LabelTable
        self.ci = Interrogator(Config())
        self.LT = LabelTable
        self.classes = classes

    def similarity(self, images_features, text):
        return self.ci.similarity(images_features, text)

    def get_list(self, list_data):
        if isinstance(list_data, str):
            return list_data.split(',')
        return ['city', 'portrait', 'river', 'landscape']

    def executeImage(self, image, data):
        features = self.ci.image_to_features(image)
        if 'rank' in data:
            table = self.LT(self.get_list(data['rank']), 'terms', self.ci)
            return {'result': table.rank(features, top_count=data.get('top', 2))}
        if 'rank_c' in data:
            table = self.LT(self.classes, 'terms', self.ci)
            return {'result': table.rank(features, top_count=data.get('top', 2))}
        if 'similarity' in data:
            return {'result': self.similarity(features, data['similarity'])}
        return {}


class DocumentAnalysis(r.ImageExecutor):
    MODULES = ['git+https://github.com/THU-MIG/yolov10.git', 'opencv-python']
    # pycocotools==2.0.7
    # PyYAML==6.0.1
    # scipy==1.13.0
    # gradio==4.31.5
    # opencv-python==4.9.0.80
    # psutil==5.9.8
    # py-cpuinfo==9.0.0

    ENTITIES_COLORS = {
        "Caption": (191, 100, 21),
        "Footnote": (2, 62, 115),
        "Formula": (140, 80, 58),
        "List-item": (168, 181, 69),
        "Page-footer": (2, 69, 84),
        "Page-header": (83, 115, 106),
        "Picture": (255, 72, 88),
        "Section-header": (0, 204, 192),
        "Table": (116, 127, 127),
        "Text": (0, 153, 221),
        "Title": (196, 51, 2)
    }
    BOX_PADDING = 2

    def __init__(self):
        import cv2
        from ultralytics import YOLO
        self.cv2 = cv2
        if not os.path.exists("yolov10x_best.pt"):
            url = 'https://huggingface.co/spaces/omoured/YOLOv10-Document-Layout-Analysis/resolve/main/models/yolov10x_best.pt'
            u.download(url, "yolov10x_best.pt")
        self.DETECTION_MODEL = YOLO("yolov10x_best.pt")

    def pillow_image_to_base64_string(self, img):
        import base64
        import io
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def page(self, params={}):
        p = b.Page(owner=self)
        p.h1("DocumentAnalysis")
        # TODO
        f = '/home/pi/data_buchseite.jpg'
        r = self.execute({'image': f})
        p(f'<img src="data:image/png;base64,{r["image"]}" />')
        return p.nxui()

    def executeImage(self, image, data):
        #image = cv2.imread(image_path)
        import numpy
        image = self.cv2.cvtColor(numpy.array(image), self.cv2.COLOR_RGB2BGR)
        results = self.DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)  # Predict on image
        boxes = results[0].boxes  # Get bounding boxes

        if len(boxes) == 0:
            return {'success': False, 'message': "No Content Found"}

        # Get bounding boxes
        for box in boxes:
            detection_class_conf = round(box.conf.item(), 2)
            cls = list(self.ENTITIES_COLORS)[int(box.cls)]
            # Get start and end points of the current box
            start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

            # 01. DRAW BOUNDING BOX OF OBJECT    # Adjust the scale factors for bounding box and label
            box_scale_factor = 0.001  # Reduce this value to make the bounding box thinner
            label_scale_factor = 0.5  # Reduce this value to make the label smaller

            # 01. DRAW BOUNDING BOX OF OBJECT
            line_thickness = round(box_scale_factor * (image.shape[0] + image.shape[1]) / 2) + 1
            image = self.cv2.rectangle(img=image, pt1=start_box, pt2=end_box,
                                  color=self.ENTITIES_COLORS[cls],
                                  thickness=line_thickness)  # Draw the box with predefined colors
            # 02. DRAW LABEL
            text = cls + " " + str(detection_class_conf)
            # Get text dimensions to draw wrapping box
            font_thickness = max(line_thickness - 1, 1)
            (font_scale_w, font_scale_h) = (line_thickness * label_scale_factor, line_thickness * label_scale_factor)
            (text_w, text_h), _ = self.cv2.getTextSize(text=text, fontFace=2, fontScale=font_scale_w, thickness=font_thickness)
            # Draw wrapping box for text
            image = self.cv2.rectangle(img=image,
                                  pt1=(start_box[0], start_box[1] - text_h - self.BOX_PADDING * 2),
                                  pt2=(start_box[0] + text_w + self.BOX_PADDING * 2, start_box[1]),
                                  color=self.ENTITIES_COLORS[cls],
                                  thickness=-1)
            # Put class name on image
            start_text = (start_box[0] + self.BOX_PADDING, start_box[1] - self.BOX_PADDING)
            image = self.cv2.putText(img=image, text=text, org=start_text, fontFace=0, color=(255, 255, 255),
                                fontScale=font_scale_w, thickness=font_thickness)
        color_converted = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(color_converted)

        return {'image': self.pillow_image_to_base64_string(pil_image)}

