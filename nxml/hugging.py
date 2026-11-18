"""
    Gemeinsame Basis für Runner, die Modelle von Hugging Face verwenden (transformers, sentence-transformers, ...)

    class MyRunner(HuggingRunner, r.BaseJobExecutor):
        model_name = "org/model"
        HF_MODEL_ATTRS = ('pipe',)

        def __init__(self, model_name=None, device_mode='auto'):
            super().__init__('my')
            self.hf_init(model_name, device_mode)

        def load_model(self):
            self.pipe = self.hf_pipeline('text-classification')

        def execute(self, data):
            if 'text' in data:
                self.load()
                return {'result': self.pipe(data['text'])}
            return super().execute(data)

    Das Modell wird lazy beim ersten Aufruf von load() geladen. Ops: load, unload, status, delete_cache.
    Seiten: Navigation (part_hf_nav), Modell (part_hf_model, inkl. Löschen aus dem HF-Cache), Systeminfo.
"""
import gc
import html
import json
import os
import shutil
import stat
import sys
import threading

from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import web as w

from nxml.system import AISystemInfo, select_device, format_bytes


def hf_hub_cache() -> str:
    """ Ordner des Hugging-Face-Hub-Caches (HF_HUB_CACHE, HF_HOME/hub oder ~/.cache/huggingface/hub) """
    return os.environ.get('HF_HUB_CACHE') or os.path.join(
        os.environ.get('HF_HOME') or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'), 'hub')


def hf_model_cache_dir(model_id: str) -> str:
    return os.path.join(hf_hub_cache(), 'models--' + model_id.replace('/', '--'))


def dir_size(path: str) -> int:
    """ Größe aller Dateien, Symlinks (snapshots/ unter Linux) werden nicht doppelt gezählt """
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
                if not stat.S_ISLNK(st.st_mode):
                    total += st.st_size
            except OSError:
                pass
    return total


def _rmtree(path):
    """ shutil.rmtree, schreibgeschützte Dateien (Windows) werden vorher beschreibbar gemacht """
    def retry(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


class HuggingRunner(AISystemInfo):
    """
    Mixin für Runner mit Hugging-Face-Modellen, vor r.BaseJobExecutor (bzw. TextExecutor, ...) erben:

        class X(HuggingRunner, r.BaseJobExecutor)

    Unterklassen setzen model_name, HF_MODEL_ATTRS und implementieren load_model(). Der Tag 'hugging' wird
    automatisch zu TAGS hinzugefügt.
    """
    TAGS = ['hugging']

    model_name = None
    # "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
    device_mode = 'auto'
    # Attribute, die load_model() setzt, unload() setzt sie wieder auf None
    HF_MODEL_ATTRS = ()
    # Seiten des Runners für die Navigation (name, Titel), Modell und System werden angehängt
    HF_PAGES = [('index', "Start")]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tags = list(getattr(cls, 'TAGS', None) or [])
        if 'hugging' not in tags:
            cls.TAGS = tags + ['hugging']

    def hf_init(self, model_name=None, device_mode=None):
        """ im Konstruktor nach super().__init__() aufrufen """
        if model_name:
            self.model_name = model_name
        if device_mode:
            self.device_mode = device_mode
        self.device = None
        self.hf_loaded = False
        self.hf_lock = threading.RLock()
        for attr in self.HF_MODEL_ATTRS:
            setattr(self, attr, None)
        self.define_vars('model_name', 'device_mode')

    # --- Laden / Entladen ---------------------------------------------------------------------------------

    def load_model(self):
        """ lädt die Modelle in die Attribute aus HF_MODEL_ATTRS, self.device ist bereits gesetzt """
        raise NotImplementedError(type(self).__name__ + '.load_model()')

    def unload_model(self):
        for attr in self.HF_MODEL_ATTRS:
            setattr(self, attr, None)

    def hf_select_device(self):
        return select_device(self.device_mode)

    def load(self):
        """ lädt das Modell, falls noch nicht geschehen (thread-sicher) """
        if self.hf_loaded:
            return
        with self.hf_lock:
            if self.hf_loaded:
                return
            self.device = self.hf_select_device()
            self.info(f"Lade {self.hf_model_label()} auf {self.device}")
            self.load_model()
            self.hf_loaded = True

    def unload(self):
        """ gibt das Modell und, soweit möglich, RAM und Grafikspeicher frei """
        with self.hf_lock:
            if not self.hf_loaded:
                return
            self.unload_model()
            self.hf_loaded = False
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def set_model_name(self, value):
        """ setvar model_name: anderes Modell, das aktuelle wird entladen """
        if value != self.model_name:
            self.unload()
            self.model_name = value

    def set_device_mode(self, value):
        if value not in ('cpu', 'gpu', 'auto'):
            raise ValueError('device_mode must be "cpu", "gpu" or "auto"')
        if value != self.device_mode:
            self.unload()
            self.device_mode = value

    # --- Hilfen zum Laden -----------------------------------------------------------------------------------

    def hf_pipeline(self, task, model=None, **kwargs):
        """ transformers.pipeline auf self.device """
        from transformers import pipeline
        return pipeline(task, model=model or self.model_name, device=self.device, **kwargs)

    def hf_pretrained(self, cls, model=None, to_device=True, **kwargs):
        """ cls.from_pretrained(model), Modelle (to_device=True) auf self.device und im eval-Modus """
        obj = cls.from_pretrained(model or self.model_name, **kwargs)
        if to_device:
            obj = obj.to(self.device).eval()
        return obj

    # --- Modelle und Cache ----------------------------------------------------------------------------------

    def hf_models(self) -> list:
        """ Repo-IDs der verwendeten Modelle auf dem Hub (für den Cache), z.B. Basismodell und Adapter """
        return [self.model_name] if self.model_name else []

    def hf_model_label(self) -> str:
        return ' + '.join(self.hf_models()) or '-'

    def hf_cache_info(self) -> list:
        res = []
        for model in self.hf_models():
            path = hf_model_cache_dir(model)
            cached = os.path.isdir(path)
            blobs = os.path.join(path, 'blobs')
            downloading = os.path.isdir(blobs) and any(f.endswith('.incomplete') for f in os.listdir(blobs))
            res.append({'model': model, 'path': path, 'cached': cached, 'downloading': downloading,
                        'size': dir_size(path) if cached else 0})
        return res

    def hf_delete_cache(self, model=None) -> list:
        """
            Löscht Modelle dieses Runners aus dem HF-Cache, ein geladenes Modell wird vorher entladen.
            :param model: Repo-ID, None = alle Modelle des Runners
            :return: [{'model', 'path', 'size'}] der gelöschten Ordner
        """
        models = self.hf_models()
        if model is not None and model not in models:
            raise ValueError(f"{model} gehört nicht zu diesem Runner ({', '.join(models)})")
        self.unload()
        deleted = []
        for info in self.hf_cache_info():
            if info['cached'] and (model is None or info['model'] == model):
                _rmtree(info['path'])
                deleted.append({'model': info['model'], 'path': info['path'], 'size': info['size']})
                self.info(f"Aus dem HF-Cache gelöscht: {info['model']} ({format_bytes(info['size'])})")
        return deleted

    def hf_status(self) -> dict:
        return dict(loaded=self.hf_loaded, model=self.hf_model_label(), models=self.hf_models(),
                    device=self.device, device_mode=self.device_mode, cache=self.hf_cache_info())

    # --- Operationen ----------------------------------------------------------------------------------------

    @r.op("Lädt das Modell (sonst beim ersten Aufruf)", name='load', success={'loaded': 'bool', 'device': 'str'})
    def op_hf_load(self):
        self.load()
        return self.success(**self.hf_status())

    @r.op("Entlädt das Modell und gibt RAM/Grafikspeicher frei", name='unload', success={'loaded': 'bool'})
    def op_hf_unload(self):
        self.unload()
        return self.success(**self.hf_status())

    @r.op("Modell, Gerät, geladen und Belegung im HF-Cache", name='status',
          success={'loaded': 'bool', 'model': 'str', 'models': 'list', 'device': 'str', 'device_mode': 'str',
                   'cache': 'list'})
    def op_hf_status(self):
        return self.success(**self.hf_status())

    @r.op("Löscht die Modelle des Runners aus dem Hugging-Face-Cache (model leer = alle)", name='delete_cache',
          success={'deleted': 'list', 'freed': 'int'}, fail={})
    def op_hf_delete_cache(self, model: str = ''):
        try:
            deleted = self.hf_delete_cache(model or None)
        except (ValueError, OSError) as e:
            return self.fail(str(e))
        freed = sum(d['size'] for d in deleted)
        msg = f"{len(deleted)} Modell(e) gelöscht, {format_bytes(freed)} frei" if deleted else "Nichts im Cache"
        return self.success(msg, deleted=deleted, freed=freed, ui_reload=True)

    # --- Seiten ---------------------------------------------------------------------------------------------

    def page(self, params={}):
        """ Seiten-Dispatch von BaseJobExecutor (part_*), auch wenn eine Basisklasse wie TextExecutor page() ersetzt """
        return r.BaseJobExecutor.page(self, params)

    def hf_nav_items(self) -> list:
        return list(self.HF_PAGES) + [('hf_model', "Modell"), ('systeminfo', "System")]

    def part_hf_nav(self, p: b.Page, active='index'):
        """ Navigation oben auf den Seiten des Runners """
        p.nav(*[(title, self.link(name)) for name, title in self.hf_nav_items()], active=self.link(active))
        state = f"geladen auf {self.device}" if self.hf_loaded else "nicht geladen (lädt beim ersten Aufruf)"
        p.div(f"Modell: {self.hf_model_label()} - {state}")

    def part_index(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'index')
        doc = (type(self).__doc__ or '').strip()
        if doc:
            p.pre(html.escape(doc))
        super().part_index(p, params)

    def part_hf_model(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'hf_model')
        p.h1("Modell")
        with p.section(h="Status"):
            p.prop("Modell", self.hf_model_label())
            p.prop("Geladen", "ja" if self.hf_loaded else "nein")
            p.prop("Gerät", self.device or '-')
            p.prop("device_mode", self.device_mode)
            p(self.action_btn(dict(title="Laden", type=self.type, op='load'),
                              hint="Modell jetzt laden (sonst beim ersten Aufruf)"))
            p(self.action_btn(dict(title="Entladen", type=self.type, op='unload'),
                              hint="Modell entladen und RAM/Grafikspeicher freigeben"))
            p(self.action_btn(dict(title="Status", type=self.type, op='status'),
                              hint="Status als JSON, inkl. Belegung im HF-Cache"))
        self.part_hf_model_config(p, params)
        self.part_hf_cache(p)
        p.pre('', id='result')

    def part_hf_model_config(self, p: b.Page, params={}):
        """ für eigene Abschnitte der Modell-Seite (z.B. Modellwahl) """
        pass

    def part_hf_cache(self, p: b.Page):
        with p.section(h="Hugging-Face-Cache"):
            p.prop("Cache-Ordner", hf_hub_cache())
            p('<table>')
            p('<tr><th>Modell</th><th>Im Cache</th><th>Größe</th><th>Ordner</th><th></th></tr>')
            for info in self.hf_cache_info():
                state = ("Download läuft" if info['downloading'] else "ja") if info['cached'] else "nein"
                btn = ''
                if info['cached']:
                    data = {'type': self.type, 'op': 'delete_cache', 'model': info['model']}
                    question = f"{info['model']} ({format_bytes(info['size'])}) aus dem Cache löschen? " \
                               f"Beim nächsten Laden wird es erneut heruntergeladen."
                    btn = w.button_js("Löschen", 'if (confirm(' + json.dumps(question) + ')) exec_job(' +
                                      json.dumps(data) + ');',
                                      hint="Entlädt das Modell und löscht den Ordner im HF-Cache")
                p(f"<tr><td>{html.escape(info['model'])}</td><td>{state}</td>"
                  f"<td>{format_bytes(info['size']) if info['cached'] else '-'}</td>"
                  f"<td>{html.escape(info['path'])}</td><td>{btn}</td></tr>")
            p('</table>')

    def part_systeminfo(self, p: b.Page, params={}):
        self.part_hf_nav(p, 'systeminfo')
        super().part_systeminfo(p, params)
