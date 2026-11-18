#
# Project: nxml
#

from threading import Thread
import json
import re
import os.path
import time
import glob
from scipy.spatial.distance import cosine
import requests
import base64
from io import BytesIO

from nwebclient import runner as r
from nwebclient.runner import TAG
from nwebclient import NWebClient
from nwebclient import NWebDoc
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import dev as d

from nxml import analyse
from nxml.system import select_device


class DatasetWriter:
    """
     für clip retrival
     Erstellt dataset nach https://github.com/rom1504/img2dataset
    """
    index = 0
    output_folder = './'

    def __init__(self, output_folder = './'):
        super().__init__()
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
    TAGS = [TAG.IMAGE]

    type = 'img_embedding'

    embedding_folder = './'
    embedder = None

    embeddings = {}

    pairs = []

    threshold = 0.95

    knn = None
    knn_ids = {}

    def __init__(self, embedding_folder='./', args:u.Args={}):
        super().__init__()
        self.define_vars("embedding_folder", "threshold")
        if self.embedding_folder == embedding_folder and args.get('embedding_folder', None) is not None:
            self.embedding_folder = args.get('embedding_folder', None)
        else:
            self.embedding_folder = embedding_folder
        self.embedder = analyse.ClipEmbeddings()
        self.embeddings = {}
        self.nc = NWebClient(None)
        self.knn = None
        self.knn_ids = None
        self.define_sig(d.PStr('inference', '1'))\
            .set_description("Erstellt CLIP-Embeddings für alle nweb-Bilder ohne Embedding")\
            .r(inference='str', success='bool')
        self.define_sig(d.Param('q'), d.PInt('k', 5)).set_description("Sucht per Text die k ähnlichsten Bilder")\
            .r(images='list')
        self.define_sig(d.Param('image'), d.PInt('k', 5)).set_description("Sucht die k ähnlichsten Bilder zu einem Bild")\
            .r(images='list')

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

    MODULES = ['opencv-python', 'scikit-image', 'open_clip_torch', 'scipy', 'torch']
    TAGS = ['image', '2-images']
    NS = 'image_similarity'

    comparators = {
        'clip': 'compareImagesCLIP',
        'ssim': 'compareImagesSSIM'
    }

    threshold = 0.9

    clip_model = None
    clip_preprocess = None

    def __init__(self, args={}, device_mode='auto'):
        """
            :param device_mode: Gerät für CLIP: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        """
        super().__init__('image_similarity')
        self.device_mode = device_mode
        self.device = None
        # Bilder: Base64, Pfad oder nweb-ID (image_id / b_id)
        self.define_sig(d.Param('image', is_pos=False), d.Param('b', is_pos=False), d.PStr('algo', 'ssim'))\
            .set_description("Ähnlichkeit zweier Bilder (0..1), algo: ssim oder clip")\
            .r(success='bool', score='float', algo='str')
        self.define_sig(d.Param('image', is_pos=False), d.Param('search', is_pos=False))\
            .set_description("Die 5 ähnlichsten Bilder einer nweb-Suche")\
            .r(images='list', nweb='str', image_count='int')
        self.define_sig(d.NP('index', 'str', '1'), d.PInt('limit', 10), d.PStr('q', ''))\
            .set_description("Vergleicht noch nicht indizierte nweb-Bilder mit allen anderen, Treffer als Meta")\
            .r(**{'from': 'str'})
        self.define_sig(d.NP('index_async', 'str', '1')).set_description("Startet die Indizierung im Hintergrund").r()

    def compareImagesSSIM(self, image_a, image_b):
        import cv2
        from skimage import metrics
        import numpy
        try:
            # Load images
            #image1 = cv2.imread(image_a)
            #image2 = cv2.imread(image_b)
            image1 = cv2.cvtColor(numpy.array(image_a.convert('RGB')), cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(numpy.array(image_b.convert('RGB')), cv2.COLOR_RGB2BGR)
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
            # print(image1.shape, image2.shape)
            # Convert images to grayscale
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            # Calculate SSIM
            ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
            #print(f"SSIM Score: ", round(ssim_score[0], 2))
            return round(float(ssim_score[0]), 2)
        except Exception as e:
            self.error(str(e))
            return 0

    def compareImagesCLIP(self, image_a, image_b):
        # !pip install open_clip_torch
        import torch
        model, preprocess = self.load_clip()

        def imageEncoder(img):
            img1 = preprocess(img.convert('RGB')).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return model.encode_image(img1)

        cos = torch.nn.functional.cosine_similarity(imageEncoder(image_a), imageEncoder(image_b))
        return round(float(cos[0]), 4)

    def load_clip(self):
        """ CLIP-Modell einmalig laden, nicht bei jedem Vergleich """
        if self.clip_model is None:
            import open_clip
            self.device = select_device(self.device_mode)
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
            self.clip_model = model.to(self.device).eval()
            self.clip_preprocess = preprocess
        return self.clip_model, self.clip_preprocess

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

    def ui_imageform(self, p: b.Page):
        with p.section():
            p.js_ready('nx_initFileDragArea("dropZone", "image_data");')
            p.js_ready('nx_initFileDragArea("dropZoneB", "b_data");')
            p.dropzone('dropZone', "Bild A hier ablegen")
            p.hidden('image_data', '', "image_data")
            p.dropzone('dropZoneB', "Bild B hier ablegen")
            p.hidden('b_data', '', "b_data")
            p.form_select('algo', "Algorithmus", {'ssim': "SSIM (Struktur)", 'clip': "CLIP (Inhalt)"},
                          value='ssim', id='algo')
            p.form_elem(self.action_btn_parametric("Vergleichen", dict(
                type=self.type,
                image='#image_data',
                b='#b_data',
                algo='#algo'
            )))
        p.pre('', id='result')

    def executeImage(self, image, data):
        b = self.get_image('b', data)
        if b is not None:
            algo = data.get('algo') or 'ssim'
            if algo not in self.comparators:
                return self.fail(f'Unknown algo {algo!r}, use one of {list(self.comparators)}')
            score = self.compareImages(image, b, algo)
            return {'success': True, 'score': score, 'algo': algo}
        if 'search' in data:
            return self.searchSimilar(image, data)
        else:
            return self.fail('ImageSimilarity.executeImage')


class ObjectDetector(r.ImageExecutor):
    """
    https://huggingface.co/IDEA-Research/grounding-dino-tiny

    https://huggingface.co/spaces/EduardoPacheco/Grounding-Dino-Inference/blob/main/app.py
    """

    MODULES = ['scipy', 'torch', 'transformers']
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR, 'hugging']
    type = 'od'

    def __init__(self):
        super().__init__()
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.define_sig(d.Param('image'), d.PStr('text', "a cat. a remote control."))\
            .set_description("Zero-Shot-Objekterkennung, text: kleingeschriebene Objekte mit Punkt getrennt")\
            .r(boxes='list', annotations='list', labels='list', request_text='str')

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
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR]

    MODULES = ['clip_interrogator', 'scipy']

    type = 'ic'

    classes = []

    def __init__(self, classes=[]):
        super().__init__()
        from clip_interrogator import Config, Interrogator, LabelTable
        self.ci = Interrogator(Config())
        self.LT = LabelTable
        self.classes = classes
        # Bild: Base64, Pfad oder nweb-ID (image_id); rank: Begriffe mit Komma getrennt
        self.define_sig(d.Param('image', is_pos=False), d.Param('rank', is_pos=False), d.PInt('top', 2))\
            .set_description("Die top passendsten Begriffe aus rank").r(result='list')
        self.define_sig(d.Param('image', is_pos=False), d.NP('rank_c', 'str', '1'), d.PInt('top', 2))\
            .set_description("Die top passendsten der konfigurierten Klassen").r(result='list')
        self.define_sig(d.Param('image', is_pos=False), d.Param('similarity', is_pos=False))\
            .set_description("Ähnlichkeit zwischen Bild und Text").r(result='float')

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
    """
      Docker in nxml/docker/document_analysis.sh

      Läuft aktuell nur auf x86

    """
    MODULES = ['git+https://github.com/THU-MIG/yolov10.git', 'opencv-python', 'scipy', 'ultralytics']
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR]
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
        super().__init__()
        self.type = 'document'
        try:
            import cv2
            from ultralytics import YOLO
            self.cv2 = cv2
            if not os.path.exists("yolov10x_best.pt"):
                url = 'https://huggingface.co/spaces/omoured/YOLOv10-Document-Layout-Analysis/resolve/main/models/yolov10x_best.pt'
                u.download(url, "yolov10x_best.pt")
            self.DETECTION_MODEL = YOLO("yolov10x_best.pt")
        except Exception as e:
            self.error(str(e))
        f = self.define_sig(d.Param('image'))\
            .set_description("Layout-Analyse einer Dokumentseite, image ist das Bild mit Boxen (Base64-JPEG)")
        f.r(image='str', items='list')
        f.r_fail()

    def pillow_image_to_base64_string(self, img):
        import base64
        import io
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def page(self, params={}):
        p = b.Page(owner=self)
        p.h1("DocumentAnalysis")
        # TODO impl
        f = '/home/pi/data_buchseite.jpg'
        r = self.execute({'image': f})
        p(f'<img src="data:image/png;base64,{r["image"]}" />')
        p.pre(json.dumps(r))
        p.hr()
        return p.nxui()

    def write_to_meta(self, d: NWebDoc, result: dict):
        i = 0
        for item in result.get('items', []):
            d.setMetaValue('DocumentAnalysis', i, json.dumps(item))
            i += 0

    def execute(self, data):
        #if 'nweb_doc_map' in data
        return super().execute(data)

    def executeImage(self, image, data):
        #image = cv2.imread(image_path)
        import numpy
        image = self.cv2.cvtColor(numpy.array(image), self.cv2.COLOR_RGB2BGR)
        results = self.DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)  # Predict on image
        boxes = results[0].boxes  # Get bounding boxes

        if len(boxes) == 0:
            return {'success': False, 'message': "No Content Found"}

        items = []

        # Get bounding boxes
        for box in boxes:
            detection_class_conf = round(box.conf.item(), 2)
            cls = list(self.ENTITIES_COLORS)[int(box.cls)]
            # Get start and end points of the current box
            start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

            items.append({
                'cls': str(cls),
                'x1': int(box.xyxy[0][0]),
                'y1': int(box.xyxy[0][1]),
                'x2': int(box.xyxy[0][2]),
                'y2': int(box.xyxy[0][3]),
                'detection': float(detection_class_conf)
            })

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

        return {'image': self.pillow_image_to_base64_string(pil_image), 'items': items}


class DocumentAnalysisDockerd(r.ImageExecutor):
    """
    Baut und startet die Dokumentanalyse als Docker-Container und leitet Jobs per HTTP an ihn weiter.
    """
    MODULES = ['scipy', 'setuptools']

    TAGS = [TAG.IMAGE]

    def __init__(self, build=True):
        super().__init__()
        f = self.define_sig(d.Param('image'))\
            .set_description("Layout-Analyse einer Dokumentseite im Docker-Container (wie DocumentAnalysis)")
        f.r(image='str', items='list')
        f.r_fail()
        if build:
            self.docker_build()

    def docker_build(self):
        import pkg_resources
        path = pkg_resources.resource_filename('nxml', 'docker') + '/'
        # /home/pi/.local/lib/python3.10/site-packages/nxml/docker/
        os.chdir(path)
        dockerfile = 'DocumentAnalysis.Dockerfile'
        cmd = f'docker build -t document-analysis -f {dockerfile} .'
        self.info("Build Docker Container")
        r.ProcessExecutor(cmd, on_line=print, on_end=self.on_image_builded)

    def on_image_builded(self, p):
        cmd = 'docker run --rm -it -p 27201:7070  document-analysis'
        self.info("RUN: " + cmd)
        self.p = r.ProcessExecutor(cmd, on_line=print)

    def execute(self, data):
        return requests.post('http://127.0.0.1:27201/', data).json()


class Aesthetics(r.ImageExecutor):
    """
    https://github.com/shunk031/simple-aesthetics-predictor
    """
    TAGS = [TAG.IMAGE, 'hugging']

    META_NAME = 'aesthetics'

    MODULES = ['simple-aesthetics-predictor', 'scipy', 'torch', 'transformers']

    model = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
    
    def __init__(self, device_mode='auto'):
        """
            :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        """
        super().__init__('aesthetics')
        self.device = select_device(device_mode)
        # Bild: Base64, Pfad oder nweb-ID (image_id)
        self.define_sig(d.Param('image', is_pos=False)).set_description("Ästhetik-Score eines Bildes")\
            .r(success='bool', message='str', value='float')
        self.load()

    def load(self):
        from transformers import CLIPProcessor
        from aesthetics_predictor import AestheticsPredictorV1
        self.predictor = AestheticsPredictorV1.from_pretrained(self.model).to(self.device).eval()
        # Standard-CLIPProcessor, ohne trust_remote_code fragt transformers interaktiv nach (blockiert im Server)
        self.processor = CLIPProcessor.from_pretrained(self.model, trust_remote_code=False)

    def executeImage(self, image, data):
        import torch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():  # or `torch.inference_model` in torch 1.9+
            outputs = self.predictor(**inputs)
        prediction = outputs.logits
        self.info(f"Aesthetics score: {prediction}")
        return self.success(value=float(prediction[0][0]))


class ImageFingerprint(r.ImageExecutor):
    """
      https://pypi.org/project/ImageHash/

      select MIN(d.document_id), MAX(d.document_id) from document d join document__meta m on d.document_id = m.document_id where m.name = 'fingerprint' group by m.value having count(m.name) > 1
    """
    MODULES = ['ImageHash', 'scipy']

    def __init__(self):
        super().__init__('fingerprint')
        # Bild: Base64, Pfad oder nweb-ID (image_id); value: average_hash als Hex-String
        self.define_sig(d.Param('image', is_pos=False)).set_description("Wahrnehmungs-Hash (average_hash) eines Bildes")\
            .r(success='bool', message='str', value='str')

    def executeImage(self, image, data):
        try:
            import imagehash
            fingerprint = imagehash.average_hash(image)
            return self.success(value=str(fingerprint))
        except Exception as e:
            self.info(e)
            return self.fail(str(e))


CAMERA_RE = re.compile(r'^(/dev/video\d+|\d+)$')


class CameraGrabber:
    """
    Hält eine lokale Kamera (/dev/videoN oder Index) offen und liest im Hintergrund laufend Bilder, frame() liefert
    das neueste. So bleibt die Kamera bei periodischem Abruf offen (Öffnen dauert, die Belichtung muss sich
    einregeln) und es kommen keine alten Bilder aus dem Puffer. Nach idle Sekunden ohne Abruf wird sie geschlossen.
    """

    def __init__(self, source, idle=60.0):
        self.source = source
        self.idle = idle
        self.error = None
        self.last_used = time.time()
        self._frame = None
        self._seq = 0
        self._stop = False
        self._cond = __import__('threading').Condition()
        try:
            import cv2
        except ImportError:
            raise RuntimeError('Für lokale Kameras wird OpenCV benötigt: pip install opencv-python-headless')
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f'Kamera {source} kann nicht geöffnet werden')
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fails = 0
        try:
            while not self._stop and time.time() - self.last_used < self.idle:
                ok, frame = self.cap.read()
                if not ok:
                    fails += 1
                    if fails > 50:
                        self.error = 'Kamera liefert keine Bilder'
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
        finally:
            self._stop = True
            self.cap.release()
            with self._cond:
                self._cond.notify_all()

    @property
    def alive(self):
        return not self._stop

    def frame(self, timeout=5.0):
        """ neuestes Bild als PIL-Image (RGB), wartet auf ein Bild, das nach dem Aufruf aufgenommen wurde """
        from PIL import Image
        self.last_used = time.time()
        with self._cond:
            seq = self._seq
            end = time.time() + timeout
            while self._seq == seq and not self._stop and time.time() < end:
                self._cond.wait(0.2)
            if self._frame is None:
                raise RuntimeError(self.error or f'Kein Bild von {self.source}')
            frame = self._frame
        return Image.fromarray(frame[:, :, ::-1].copy())  # BGR -> RGB

    def close(self):
        self._stop = True


class DepthEstimation(r.ImageExecutor):
    """
    Tiefenschätzung mit Depth Anything V2, https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

    Seiten: Index (Bild analysieren), ImageRadar (Bild-URL abrufen, Hinderniswarnung, wenn im oberen Teil des
    Bildes etwas näher ist als der Schwellwert; der untere Teil, z.B. der Boden, wird ignoriert).

    ImageRadar-Modelle:
     - relative: das Standardmodell, Nähe 0..1 je Bild normiert (1 = nächster Punkt im Bild). Der Schwellwert ist
       damit relativ zum Bild: sind alle Objekte weit weg, gilt das nächste trotzdem als nah.
     - indoor / outdoor: metrische Modelle (Depth-Anything-V2-Metric-*-Small-hf), Entfernung in Metern, werden beim
       ersten Gebrauch geladen.
    """
    TAGS = ['hugging']

    MODULES = ['scipy', 'transformers']

    MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
    METRIC_MODELS = {
        'indoor': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
        'outdoor': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
    }

    def __init__(self, type='depth', max_length=600, camera_idle=60):
        """
            :param max_length: Bilder vor der Analyse auf diese längste Seite verkleinern (0 = Originalgröße)
            :param camera_idle: lokale Kameras (/dev/video0) nach so vielen Sekunden ohne Abruf schließen
        """
        super().__init__(type)
        from transformers import pipeline
        self.max_length = int(max_length)
        self.camera_idle = float(camera_idle)
        self.pipe = pipeline(task="depth-estimation", model=self.MODEL)
        self.pipes = {'relative': self.pipe}
        self._pipe_lock = __import__('threading').Lock()
        self._radar_state = {}  # url -> letzter Zustand, Event depth_obstacle nur bei Änderung
        self.define_sig(d.Param('image'), d.PInt('max_length', self.max_length))\
            .set_description("Schätzt eine Tiefenkarte (Base64-PNG), max_length: längste Seite vorher "
                             "verkleinern (0 = Originalgröße)")\
            .r_success(image='str', duration='float', width='int', height='int')
        self.define_sig(d.PStr('op', 'radar'), d.PStr('url', ''), d.PStr('model', 'relative'),
                        d.NP('ignore_bottom', 'float', 30), d.NP('threshold', 'float', 0.7),
                        d.NP('min_area', 'float', 0.5), d.PInt('max_length', self.max_length))\
            .set_description("ImageRadar: Bild von url laden, Tiefe schätzen, Hindernis, wenn im oberen Teil "
                             "(ignore_bottom % unten ignoriert) mindestens min_area % näher als threshold sind. "
                             "threshold: relative = Nähe 0..1, indoor/outdoor = Meter")\
            .r(success='bool', obstacle='bool', area='float', nearest='float', duration='float', image='str')
        self.define_sig(d.PStr('op', 'radar2'), d.PStr('url', ''), d.PStr('zones', '15,70,15'),
                        d.PStr('model', 'relative'), d.NP('ignore_bottom', 'float', 30), d.NP('threshold', 'float', 0.7),
                        d.NP('min_area', 'float', 0.5), d.PInt('max_length', self.max_length))\
            .set_description("ImageRadar2: wie radar, getrennt nach Bereichen der Bildbreite (zones in %, Standard "
                             "15,70,15 = Links, Mitte, Rechts), message z.B. 'Hindernis Links, Mitte'")\
            .r(success='bool', message='str', obstacle='bool', zones='list', duration='float')
        self.define_event('depth_zones', {'url': 'Bild-URL', 'message': "z.B. 'Hindernis Links' oder 'Weg frei'",
                                          'blocked': 'Namen der blockierten Bereiche', 'zones': 'je Bereich'})
        self.define_event('depth_obstacle', {'url': 'Bild-URL', 'obstacle': 'Hindernis (bool)',
                                             'area': 'Anteil näher als Schwellwert in %',
                                             'nearest': 'nächster Wert (Meter bzw. Nähe)'})

    @staticmethod
    def to_b64(image, format='PNG', **kwargs) -> str:
        buffer = BytesIO()
        image.save(buffer, format=format, **kwargs)
        encoded = base64.b64encode(buffer.getvalue())
        return encoded.decode("utf-8")

    def resize(self, image, data):
        """ verkleinert auf max_length (aus data, sonst Runner-Einstellung), 0 = unverändert """
        try:
            limit = int(float(data.get('max_length', self.max_length) or 0))
        except (TypeError, ValueError):
            limit = self.max_length
        if limit > 0 and max(image.size) > limit:
            image = image.copy()
            image.thumbnail((limit, limit))
        return image

    def executeImage(self, image, data):
        start = time.time()
        image = self.resize(image.convert('RGB'), data)
        depthmap = self.pipe(image)["depth"]
        return self.success(image=self.to_b64(depthmap), duration=round(time.time() - start, 3),
                            width=image.width, height=image.height)

    # -- ImageRadar --

    def get_pipe(self, model):
        if model not in self.pipes:
            if model not in self.METRIC_MODELS:
                raise ValueError(f'Unbekanntes Modell {model!r}, erlaubt: relative, ' + ', '.join(self.METRIC_MODELS))
            with self._pipe_lock:
                if model not in self.pipes:
                    from transformers import pipeline
                    self.info(f'Lade {self.METRIC_MODELS[model]}')
                    self.pipes[model] = pipeline(task="depth-estimation", model=self.METRIC_MODELS[model])
        return self.pipes[model]

    def fetch_image(self, url, timeout=15):
        """
        :param url: http(s)-URL eines Bildes oder lokale Kamera (/dev/video0, 0, 1, ...)
        """
        from PIL import Image
        url = (url or '').strip()
        if CAMERA_RE.match(url):
            return self.camera(url).frame()
        if not re.match(r'^https?://', url, re.I):
            raise ValueError('url muss mit http:// oder https:// beginnen oder eine Kamera sein (/dev/video0, 0)')
        resp = requests.get(url, timeout=timeout, headers={'User-Agent': 'nxml-depth'})
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    def camera(self, source) -> CameraGrabber:
        """ offene Kamera wiederverwenden, sonst (neu) öffnen """
        with self._pipe_lock:
            cams = self.__dict__.setdefault('_cameras', {})
            cam = cams.get(source)
            if cam is None or not cam.alive:
                cams[source] = cam = CameraGrabber(source, idle=self.camera_idle)
            return cam

    ZONE_NAMES = ['Links', 'Mitte', 'Rechts']

    @staticmethod
    def parse_zones(zones):
        """ "15,70,15" oder Liste -> Anteile in %, auf 100 normiert """
        if isinstance(zones, str):
            zones = [z for z in re.split(r'[\s,;/]+', zones.strip()) if z]
        values = [max(0.0, float(z)) for z in zones]
        if len(values) < 2:
            raise ValueError('zones braucht mindestens zwei Anteile, z.B. 15,70,15')
        if sum(values) <= 0:
            raise ValueError('Die Anteile in zones müssen zusammen größer als 0 sein')
        total = sum(values)
        return [v * 100.0 / total for v in values]

    def radar(self, image, model='relative', ignore_bottom=30.0, threshold=0.7, min_area=0.5, data={}, zones=None):
        """
        :param zones: Anteile der Bildbreite in % von links nach rechts (z.B. [15, 70, 15]), None = ein Bereich
        :return: dict mit obstacle, area (% der Pixel im oberen Teil näher als threshold), nearest und einem
                 Vorschaubild (JPEG-Base64): Tiefe als Helligkeit (hell = nah), ignorierter Teil abgedunkelt,
                 nahe Pixel rot; mit zones zusätzlich zones = [{name, start, end, obstacle, area, nearest}] und
                 message, z.B. "Hindernis Links, Mitte"
        """
        import numpy as np
        from PIL import Image
        pipe = self.get_pipe(model)  # Laden eines Modells zählt nicht zur Analysedauer
        start = time.time()
        image = self.resize(image.convert('RGB'), data)
        w, h = image.size
        out = pipe(image)
        if model == 'relative':
            near = np.asarray(out['depth'].convert('L').resize((w, h)), np.float32) / 255.0  # 1 = nah
            close = near >= threshold
            shade = near
        else:
            pred = out['predicted_depth']
            pred = pred.squeeze().detach().cpu().numpy() if hasattr(pred, 'detach') else np.squeeze(np.asarray(pred))
            dist = np.asarray(Image.fromarray(pred.astype(np.float32), 'F').resize((w, h), Image.BILINEAR))
            close = dist <= threshold
            far = float(np.percentile(dist, 98)) or 1.0
            shade = 1.0 - np.clip(dist / far, 0, 1)
        limit = int(round(h * (1 - min(95.0, max(0.0, float(ignore_bottom))) / 100.0)))
        limit = max(1, min(h, limit))
        top_close = close[:limit]
        area = float(top_close.mean() * 100.0)
        if model == 'relative':
            nearest = float(np.percentile(near[:limit], 99))
        else:
            nearest = float(np.percentile(dist[:limit], 1))
        obstacle = area >= float(min_area)
        zone_res = None
        bounds = []
        if zones:
            zone_res = []
            parts = self.parse_zones(zones)
            x = 0.0
            for i, part in enumerate(parts):
                x0, x1 = int(round(w * x / 100.0)), int(round(w * (x + part) / 100.0))
                x += part
                if i == len(parts) - 1:
                    x1 = w
                name = self.ZONE_NAMES[i] if len(parts) == 3 else f'Bereich {i + 1}'
                if x1 <= x0:
                    zone_res.append({'name': name, 'start': x0, 'end': x1, 'obstacle': False, 'area': 0.0,
                                     'nearest': None})
                    continue
                z_close = top_close[:, x0:x1]
                z_area = float(z_close.mean() * 100.0)
                z_near = float(np.percentile(near[:limit, x0:x1], 99)) if model == 'relative' \
                    else float(np.percentile(dist[:limit, x0:x1], 1))
                zone_res.append({'name': name, 'start': x0, 'end': x1, 'obstacle': z_area >= float(min_area),
                                 'area': round(z_area, 2), 'nearest': round(z_near, 3)})
                if x0 > 0:
                    bounds.append(x0)
            obstacle = any(z['obstacle'] for z in zone_res)
        # Vorschau
        g = (shade * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
        rgb[limit:] *= 0.3
        mask = np.zeros_like(close)
        mask[:limit] = top_close
        rgb[mask] = rgb[mask] * 0.35 + np.array([255, 40, 40]) * 0.65
        rgb[max(0, limit - 1):limit + 1] = (255, 200, 0)
        for bx in bounds:  # Bereichsgrenzen im oberen Teil
            rgb[:limit, max(0, bx - 1):bx + 1] = (80, 180, 255)
        preview = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        res = {
            'obstacle': obstacle, 'area': round(area, 2), 'nearest': round(nearest, 3), 'model': model,
            'unit': 'Nähe 0..1' if model == 'relative' else 'm', 'threshold': float(threshold),
            'ignore_bottom': float(ignore_bottom), 'min_area': float(min_area), 'width': w, 'height': h,
            'duration': round(time.time() - start, 3), 'image': self.to_b64(preview, 'JPEG', quality=80),
            'photo': self.to_b64(image, 'JPEG', quality=75)  # analysiertes Bild, auch für Kameras ohne URL
        }
        if zone_res is not None:
            blocked = [z['name'] for z in zone_res if z['obstacle']]
            res['zones'] = zone_res
            res['message'] = 'Hindernis ' + ', '.join(blocked) if blocked else 'Weg frei'
        return res

    def execute_radar(self, data):
        url = str(data.get('url', '')).strip()
        try:
            image = self.fetch_image(url)
            fetched = time.time()
            res = self.radar(image, data.get('model', 'relative') or 'relative',
                             float(data.get('ignore_bottom', 30) or 0), float(data.get('threshold', 0.7)),
                             float(data.get('min_area', 0.5) or 0), data)
        except Exception as e:
            return self.fail(str(e))
        if self._radar_state.get(url) != res['obstacle']:
            self._radar_state[url] = res['obstacle']
            self.emit_event('depth_obstacle', url=url, obstacle=res['obstacle'], area=res['area'],
                            nearest=res['nearest'], model=res['model'])
        return self.success(url=url, **res)

    def execute_radar2(self, data):
        """ wie radar, aber getrennt nach Bereichen (zones, Standard 15,70,15 = Links, Mitte, Rechts) """
        url = str(data.get('url', '')).strip()
        try:
            image = self.fetch_image(url)
            res = self.radar(image, data.get('model', 'relative') or 'relative',
                             float(data.get('ignore_bottom', 30) or 0), float(data.get('threshold', 0.7)),
                             float(data.get('min_area', 0.5) or 0), data, zones=data.get('zones') or '15,70,15')
        except Exception as e:
            return self.fail(str(e))
        key = 'zones:' + url
        if self._radar_state.get(key) != res['message']:
            self._radar_state[key] = res['message']
            self.emit_event('depth_zones', url=url, message=res['message'], obstacle=res['obstacle'],
                            blocked=[z['name'] for z in res['zones'] if z['obstacle']],
                            zones=[{k: z[k] for k in ('name', 'obstacle', 'area', 'nearest')} for z in res['zones']])
        return self.success(url=url, **res)

    # -- Seiten --

    PAGES = [('index', 'Index'), ('imageradar', 'ImageRadar'), ('imageradar2', 'ImageRadar2')]

    def part_nav(self, p: b.Page, active='index'):
        from nxml.webui import part_nav
        part_nav(self, p, self.PAGES, active)

    def part_index(self, p: b.Page, params={}):
        self.part_nav(p, 'index')
        super().part_index(p, params)

    def ui_imageform(self, p: b.Page):
        """ wie ImageExecutor, zusätzlich max_length; die Dauer steht im Ergebnis (duration) """
        with p.section():
            p.js_ready('nx_initFileDragArea("dropZone", "image_data");')
            p.dropzone('dropZone', "Bild hier ablegen")
            p.hidden('image_data', '', "image_data")
            p.form_input('max_length', "Längste Seite (px, 0 = Original)", value=str(self.max_length),
                         id='max_length')
            p.script(self.JS_SHOW_RESULT_IMAGE)
            p.form_elem(self.action_btn_parametric("Execute", dict(
                type=self.type,
                image='#image_data',
                max_length='#max_length'
            ), on_success='image_executor_on_result'))
        p.div('', id='result_image')
        p.pre('', id='result')

    def part_imageradar(self, p: b.Page, params={}, zones=False):
        self.part_nav(p, 'imageradar2' if zones else 'imageradar')
        cfg = {'type': self.type, 'max_length': self.max_length, 'models': ['relative', *self.METRIC_MODELS],
               'zones': zones}
        p('<style>' + DEPTH_RADAR_CSS + '</style>')
        p(DEPTH_RADAR_HTML.replace('<!--ZONES-->', DEPTH_ZONES_HTML if zones else ''))
        # Block statt globaler Konstante: die UI v2 führt die Skripte bei jedem Öffnen erneut aus
        p('<script>{ const DR_CFG = ' + json.dumps(cfg).replace('</', '<\\/') + ';\n' + DEPTH_RADAR_JS + '\n}</script>')

    def part_imageradar2(self, p: b.Page, params={}):
        """ ImageRadar mit Bereichen Links, Mitte, Rechts (Standard 15 %, 70 %, 15 % der Bildbreite) """
        self.part_imageradar(p, params, zones=True)


DEPTH_RADAR_CSS = """
.dr { max-width: 1300px; }
.dr-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
.dr-row label { min-width: 0; }
.dr input[type=number] { width: 80px; }
.dr-banner { font-size: 1.5em; font-weight: bold; padding: 12px 16px; border-radius: 10px; text-align: center;
  background: #8884; margin: 10px 0; }
.dr-banner.free { background: #1f8f4e; color: #fff; }
.dr-banner.blocked { background: #c62828; color: #fff; }
.dr-banner small { display: block; font-size: .5em; font-weight: normal; margin-top: 4px; }
.dr-images { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 800px) { .dr-images { grid-template-columns: 1fr; } }
.dr-images figure { margin: 0; }
.dr-images img { width: 100%; border-radius: 6px; background: #111; min-height: 80px; display: block; }
.dr-images figcaption { font-size: .85em; opacity: .7; margin-top: 4px; }
.dr-msg.error { color: #c33; }
.dr-zones { display: grid; gap: 8px; margin: 10px 0; }
.dr-zone { border-radius: 8px; padding: 10px; text-align: center; background: #8884; font-weight: bold; }
.dr-zone.free { background: #1f8f4e; color: #fff; }
.dr-zone.blocked { background: #c62828; color: #fff; }
.dr-zone small { display: block; font-weight: normal; font-size: .8em; opacity: .9; }
"""

DEPTH_ZONES_HTML = """
  <div class="dr-row">
    <label>Bereiche % Links / Mitte / Rechts</label>
    <input type="number" id="dr-z1" min="0" max="100" step="1" value="15" />
    <input type="number" id="dr-z2" min="0" max="100" step="1" value="70" />
    <input type="number" id="dr-z3" min="0" max="100" step="1" value="15" />
    <span id="dr-zsum"></span>
  </div>
"""

DEPTH_RADAR_HTML = """
<div class="dr">
  <div class="dr-row">
    <label for="dr-url">Bild-URL</label>
    <input type="text" id="dr-url" style="flex: 1; min-width: 280px" placeholder="http://192.168.0.20:8080/shot.jpg oder /dev/video0" />
    <button id="dr-fetch">Abrufen</button>
    <label><input type="checkbox" id="dr-auto" /> periodisch alle</label>
    <input type="number" id="dr-interval" min="0.2" step="0.1" value="2" /> s
  </div>
  <div class="dr-row">
    <label for="dr-model">Modell</label>
    <select id="dr-model"><option value="relative">relativ (Nähe 0..1)</option>
      <option value="indoor">metrisch innen (m)</option><option value="outdoor">metrisch außen (m)</option></select>
    <label for="dr-threshold">Schwellwert</label><input type="number" id="dr-threshold" step="any" />
    <span id="dr-unit"></span>
    <label for="dr-area">Mindestfläche %</label><input type="number" id="dr-area" min="0" step="0.1" value="0.5" />
    <label for="dr-ignore">unten ignorieren %</label>
    <input type="range" id="dr-ignore" min="0" max="90" step="1" value="30" /> <span id="dr-ignore-v"></span>
    <label for="dr-max">längste Seite px</label><input type="number" id="dr-max" min="0" step="10" />
  </div>
<!--ZONES-->
  <div class="dr-banner" id="dr-banner">-<small id="dr-sub"></small></div>
  <div class="dr-zones" id="dr-zones"></div>
  <div class="dr-msg" id="dr-msg"></div>
  <div class="dr-images">
    <figure><img id="dr-depth" alt="" /><figcaption>Tiefe (hell = nah), rot = näher als Schwellwert, unterhalb der gelben Linie ignoriert, blau = Bereichsgrenzen</figcaption></figure>
    <figure><img id="dr-photo" alt="" /><figcaption>Bild</figcaption></figure>
  </div>
</div>
"""

DEPTH_RADAR_JS = r"""
(function() {
  const cfg = DR_CFG;
  const $ = (id) => document.getElementById(id);
  const rootEl = $('dr-url').closest('.dr');
  const alive = () => rootEl.isConnected;
  const job = (data) => fetch(location.protocol + '//' + location.host + '/', { method: 'POST',
    body: new URLSearchParams({ type: cfg.type, ...data }) }).then((r) => r.json());
  const fmt = (v, n) => (v == null || isNaN(v)) ? '-' : (+v).toFixed(n == null ? 2 : n);
  const KEY = 'nxml_depth_radar' + (cfg.zones ? '2' : '');
  const IDS = ['dr-url', 'dr-model', 'dr-threshold', 'dr-area', 'dr-ignore', 'dr-max', 'dr-interval',
               ...(cfg.zones ? ['dr-z1', 'dr-z2', 'dr-z3'] : [])];
  function zoneSum() {
    if (!cfg.zones) return;
    const s = ['dr-z1', 'dr-z2', 'dr-z3'].reduce((a, id) => a + (+$(id).value || 0), 0);
    $('dr-zsum').textContent = s === 100 ? '' : 'Summe ' + s + ' %, wird auf 100 % umgerechnet';
  }
  if (cfg.zones) ['dr-z1', 'dr-z2', 'dr-z3'].forEach((id) => $(id).addEventListener('input', zoneSum));
  const DEFAULT_THRESHOLD = { relative: 0.7, indoor: 1.0, outdoor: 3.0 };
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
  IDS.forEach((id) => { if (saved[id] != null) $(id).value = saved[id]; });
  if ($('dr-max').value === '') $('dr-max').value = cfg.max_length;
  if ($('dr-threshold').value === '') $('dr-threshold').value = DEFAULT_THRESHOLD[$('dr-model').value];
  function save() { const st = {}; IDS.forEach((id) => st[id] = $(id).value); try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
  function unit() {
    $('dr-unit').textContent = $('dr-model').value === 'relative' ? 'Nähe ≥ (0..1)' : 'Meter ≤';
    $('dr-ignore-v').textContent = $('dr-ignore').value + ' %';
  }
  unit(); zoneSum();
  $('dr-model').addEventListener('change', () => { $('dr-threshold').value = DEFAULT_THRESHOLD[$('dr-model').value]; unit(); save(); });
  $('dr-ignore').addEventListener('input', () => { unit(); save(); });
  IDS.forEach((id) => $(id).addEventListener('change', save));

  let busy = false, timer = null;
  async function fetchOnce() {
    if (busy || !alive()) return;
    const url = $('dr-url').value.trim();
    if (!url) { $('dr-msg').textContent = 'Bitte eine Bild-URL eingeben'; $('dr-msg').classList.add('error'); return; }
    busy = true; $('dr-fetch').disabled = true;
    const t0 = performance.now();
    try {
      const req = { op: cfg.zones ? 'radar2' : 'radar', url, model: $('dr-model').value, threshold: $('dr-threshold').value,
        min_area: $('dr-area').value, ignore_bottom: $('dr-ignore').value, max_length: $('dr-max').value };
      if (cfg.zones) req.zones = [$('dr-z1').value, $('dr-z2').value, $('dr-z3').value].join(',');
      const r = await job(req);
      if (!alive()) return;
      if (!r.success) throw new Error(r.message || 'Fehler');
      const b = $('dr-banner');
      const nearText = (v) => r.model === 'relative' ? 'Nähe ' + fmt(v) : fmt(v) + ' m';
      b.className = 'dr-banner ' + (r.obstacle ? 'blocked' : 'free');
      b.firstChild.textContent = cfg.zones ? r.message : (r.obstacle ? 'Hindernis' : 'Weg frei');
      $('dr-sub').textContent = fmt(r.area, 1) + ' % näher als Schwellwert (ab ' + fmt(r.min_area, 1) + ' %), nächster ' + nearText(r.nearest);
      if (cfg.zones && r.zones) {
        const box = $('dr-zones');
        box.style.gridTemplateColumns = r.zones.map((z) => Math.max(8, z.end - z.start) + 'fr').join(' ');
        box.innerHTML = '';
        r.zones.forEach((z) => {
          const el = document.createElement('div');
          el.className = 'dr-zone ' + (z.obstacle ? 'blocked' : 'free');
          el.textContent = z.name + (z.obstacle ? ': Hindernis' : ': frei');
          const small = document.createElement('small');
          small.textContent = fmt(z.area, 1) + ' %, nächster ' + nearText(z.nearest);
          el.appendChild(small);
          box.appendChild(el);
        });
      }
      $('dr-depth').src = 'data:image/jpeg;base64,' + r.image;
      $('dr-photo').src = 'data:image/jpeg;base64,' + r.photo;
      $('dr-msg').classList.remove('error');
      $('dr-msg').textContent = r.width + 'x' + r.height + ' px, Analyse ' + fmt(r.duration, 2) + ' s, gesamt ' +
        fmt((performance.now() - t0) / 1000, 2) + ' s, Modell ' + r.model;
    } catch (e) {
      if (!alive()) return;
      $('dr-msg').textContent = 'Fehler: ' + e.message; $('dr-msg').classList.add('error');
      const b = $('dr-banner'); b.className = 'dr-banner'; b.firstChild.textContent = 'unbekannt'; $('dr-sub').textContent = '';
    } finally {
      busy = false;
      if (alive()) $('dr-fetch').disabled = false;
      schedule();
    }
  }
  function schedule() {
    clearTimeout(timer);
    if (!alive() || !$('dr-auto').checked) return;
    // nächster Abruf erst nach Ende des vorherigen plus Intervall, auch bei langsamer Analyse
    timer = setTimeout(fetchOnce, Math.max(200, 1000 * (+$('dr-interval').value || 2)));
  }
  $('dr-fetch').addEventListener('click', fetchOnce);
  $('dr-auto').addEventListener('change', () => { if ($('dr-auto').checked) fetchOnce(); else clearTimeout(timer); });
  $('dr-url').addEventListener('keydown', (e) => { if (e.key === 'Enter') fetchOnce(); });
})();
"""


class OCR(r.ImageExecutor):
    """
    Texterkennung (Bild -> Text), ausgelegt für schwache Rechner (Raspberry Pi, alte Laptops), kein PyTorch nötig.

    Engines:
     - tesseract: sehr sparsam, gut für Dokumente/Scans, Sprachen wählbar (apt install tesseract-ocr tesseract-ocr-deu)
       https://github.com/tesseract-ocr/tesseract
     - rapidocr: PP-OCR-Modelle über ONNX-Runtime (~15 MB), besser bei Fotos/Schildern/Screenshots
       https://github.com/RapidAI/RapidOCR
     - auto: tesseract falls installiert, sonst rapidocr

    Die Engine wird erst beim ersten Job geladen.
    """
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR]

    MODULES = ['scipy', 'pytesseract', 'rapidocr_onnxruntime']

    ENGINES = ['auto', 'tesseract', 'rapidocr']

    WINDOWS_TESSERACT = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    def __init__(self, type='ocr', engine='auto', lang='deu+eng', max_side=2000, threads=1, tesseract_cmd=None):
        """
            :param engine: "auto", "tesseract" oder "rapidocr"
            :param lang: Tesseract-Sprachen, z.B. "deu+eng" (nicht installierte werden ignoriert)
            :param max_side: größere Bilder werden verkleinert (spart RAM und CPU)
            :param threads: CPU-Threads für Tesseract, 1 ist auf schwachen Rechnern meist am schnellsten
            :param tesseract_cmd: Pfad zur tesseract-Binary, falls nicht im PATH
        """
        super().__init__(type)
        if engine not in self.ENGINES:
            raise ValueError(f'engine must be one of {self.ENGINES}, got {engine!r}')
        self.engine_mode = engine
        self.engine = None
        self.lang = lang
        self.max_side = int(max_side)
        self.threads = int(threads)
        self.tesseract_cmd = tesseract_cmd
        self.rapid = None
        # Bild: Base64, Pfad oder nweb-ID (image_id)
        f = self.define_sig(d.Param('image', is_pos=False), d.PStr('lang', lang), d.PInt('psm', 3),
                            d.PInt('min_confidence', 0), d.PInt('preprocess', 1))\
            .set_description("Erkennt Text in einem Bild (OCR). lang/psm nur für Tesseract, "
                             "min_confidence 0-100, preprocess 0/1")
        f.r(success='bool', text='str', lines='list', engine='str', duration='float')
        f.r_fail()

    def tesseract_available(self):
        try:
            import pytesseract
        except ImportError:
            return False
        if self.tesseract_cmd is None and os.name == 'nt' and os.path.exists(self.WINDOWS_TESSERACT):
            self.tesseract_cmd = self.WINDOWS_TESSERACT
        if self.tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def load(self):
        if self.engine is not None:
            return self.engine
        if self.engine_mode in ('auto', 'tesseract') and self.tesseract_available():
            # Tesseract nutzt sonst alle Kerne per OpenMP, was auf schwachen CPUs bremst
            os.environ['OMP_THREAD_LIMIT'] = str(self.threads)
            self.engine = 'tesseract'
        elif self.engine_mode in ('auto', 'rapidocr'):
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                from rapidocr import RapidOCR
            self.rapid = RapidOCR()
            self.engine = 'rapidocr'
        else:
            raise RuntimeError('tesseract not found, install it (apt install tesseract-ocr tesseract-ocr-deu) '
                               'or set tesseract_cmd')
        self.info(f"OCR engine: {self.engine}")
        return self.engine

    def preprocess(self, image, enabled=True):
        from PIL import Image, ImageOps
        image = ImageOps.exif_transpose(image)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        scale = 1.0
        side = max(image.size)
        if side > self.max_side:
            scale = self.max_side / side
        elif enabled and side < 1000:
            # kleine Bilder (Screenshots, Ausschnitte) werden mit mehr Pixeln deutlich besser erkannt
            scale = 2.0
        if scale != 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        if enabled and self.engine == 'tesseract':
            image = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)
        return image, scale

    def tesseract_langs(self, lang):
        import pytesseract
        try:
            available = set(pytesseract.get_languages(config=''))
        except Exception:
            return lang
        langs = [l for l in lang.split('+') if l in available]
        if len(langs) == 0:
            langs = ['eng'] if 'eng' in available else sorted(available)[:1]
        return '+'.join(langs)

    def ocr_tesseract(self, image, data, min_confidence):
        import pytesseract
        lang = self.tesseract_langs(data.get('lang', self.lang))
        psm = int(data.get('psm', 3))
        words = pytesseract.image_to_data(image, lang=lang, config=f'--psm {psm}',
                                          output_type=pytesseract.Output.DICT)
        lines = {}
        for i, word in enumerate(words['text']):
            conf = float(words['conf'][i])
            if conf < 0 or word.strip() == '' or conf < min_confidence:
                continue
            key = (words['block_num'][i], words['par_num'][i], words['line_num'][i])
            x, y, w, h = words['left'][i], words['top'][i], words['width'][i], words['height'][i]
            line = lines.setdefault(key, {'words': [], 'confs': [], 'box': [x, y, x + w, y + h]})
            line['words'].append(word)
            line['confs'].append(conf)
            box = line['box']
            line['box'] = [min(box[0], x), min(box[1], y), max(box[2], x + w), max(box[3], y + h)]
        result = []
        for (block, par, _), line in lines.items():
            result.append({
                'text': ' '.join(line['words']),
                'confidence': round(sum(line['confs']) / len(line['confs']), 1),
                'box': line['box'],
                'paragraph': f'{block}.{par}'
            })
        return result

    def ocr_rapidocr(self, image, min_confidence):
        import numpy
        output = self.rapid(numpy.array(image.convert('RGB')))
        if isinstance(output, tuple):
            # rapidocr_onnxruntime: ([[box, text, score], ...] | None, elapse)
            items = output[0] or []
        else:
            # rapidocr >= 2: Objekt mit boxes, txts, scores
            items = list(zip(output.boxes, output.txts, output.scores)) if output.txts else []
        result = []
        for box, text, score in items:
            conf = round(float(score) * 100, 1)
            if conf < min_confidence or text.strip() == '':
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            result.append({
                'text': text,
                'confidence': conf,
                'box': [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            })
        return self.merge_segments(result)

    @staticmethod
    def merge_segments(segments):
        """
          RapidOCR liefert oft mehrere Segmente pro Zeile, Segmente auf gleicher Höhe werden zusammengefasst
        """
        lines = []
        for seg in sorted(segments, key=lambda s: (s['box'][1] + s['box'][3]) / 2):
            box = seg['box']
            center = (box[1] + box[3]) / 2
            line = lines[-1] if len(lines) > 0 else None
            if line is not None and abs(center - line['center']) < (box[3] - box[1]) / 2:
                line['segments'].append(seg)
            else:
                lines.append({'center': center, 'segments': [seg]})
        result = []
        for line in lines:
            segs = sorted(line['segments'], key=lambda s: s['box'][0])
            result.append({
                'text': ' '.join(s['text'] for s in segs),
                'confidence': round(sum(s['confidence'] for s in segs) / len(segs), 1),
                'box': [min(s['box'][0] for s in segs), min(s['box'][1] for s in segs),
                        max(s['box'][2] for s in segs), max(s['box'][3] for s in segs)]
            })
        return result

    @staticmethod
    def lines_to_text(lines):
        text = ''
        last_par = None
        for line in lines:
            par = line.get('paragraph')
            if text != '':
                text += '\n\n' if par is not None and par != last_par else '\n'
            text += line['text']
            last_par = par
        return text

    def executeImage(self, image, data):
        try:
            start = time.time()
            self.load()
            min_confidence = float(data.get('min_confidence', 0))
            prepared, scale = self.preprocess(image, str(data.get('preprocess', 1)) not in ('0', 'false', 'False'))
            if self.engine == 'tesseract':
                lines = self.ocr_tesseract(prepared, data, min_confidence)
            else:
                lines = self.ocr_rapidocr(prepared, min_confidence)
            # Boxen in Koordinaten des Originalbildes
            for line in lines:
                line['box'] = [round(v / scale) for v in line['box']]
            return self.success(text=self.lines_to_text(lines), lines=lines, engine=self.engine,
                                duration=round(time.time() - start, 3))
        except Exception as e:
            self.error(str(e))
            return self.fail(str(e))


class OcrAi(r.ImageExecutor):
    """
    Texterkennung mit einem End-to-End-Transformer: GOT-OCR 2.0 (~580M Parameter)
    https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf

    Deutlich besser als OCR bei Fotos, Handschrift-ähnlichen Schriften, Tabellen, Formeln und Layouts;
    format=1 liefert formatierten Text (Markdown/LaTeX). Braucht PyTorch, auf CPU langsam, GPU empfohlen.
    """
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR, 'hugging']

    MODULES = ['scipy', 'torch', 'transformers>=4.49', 'accelerate']

    model_id = "stepfun-ai/GOT-OCR-2.0-hf"

    def __init__(self, type='ocrai', device_mode='auto', max_new_tokens=4096):
        """
            :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
            :param max_new_tokens: maximale Länge des erkannten Textes in Tokens
        """
        super().__init__(type)
        self.device = select_device(device_mode)
        self.max_new_tokens = int(max_new_tokens)
        # Bild: Base64, Pfad oder nweb-ID (image_id)
        f = self.define_sig(d.Param('image', is_pos=False), d.PInt('format', 0))\
            .set_description("Erkennt Text in einem Bild mit GOT-OCR 2.0, format=1 für Markdown/LaTeX-Ausgabe")
        f.r(success='bool', text='str', engine='str', duration='float')
        f.r_fail()
        self.load()

    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, torch_dtype=dtype)\
            .to(self.device).eval()

    def executeImage(self, image, data):
        import torch
        from PIL import ImageOps
        try:
            start = time.time()
            image = ImageOps.exif_transpose(image).convert('RGB')
            formatted = str(data.get('format', 0)) in ('1', 'true', 'True')
            inputs = self.processor(image, return_tensors="pt", format=formatted).to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**inputs, do_sample=False, tokenizer=self.processor.tokenizer,
                                          stop_strings="<|im_end|>", max_new_tokens=self.max_new_tokens)
            text = self.processor.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return self.success(text=text.strip(), engine='got-ocr2', duration=round(time.time() - start, 3))
        except Exception as e:
            self.error(str(e))
            return self.fail(str(e))

