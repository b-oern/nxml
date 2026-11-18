#
# Project: nxml
#
# Kassenbon-Erkennung: Foto eines Kassenbons -> Positionen, Preise und Händler als JSON
#
"""
BonReader (type ``bon``), DeepSeekOcr (type ``deepseek_ocr``, DeepSeek-OCR über vLLM, siehe Klasse)

Liest einen Kassenbon mit einem kleinen Vision-Language-Modell (Standard: Qwen2.5-VL-3B-Instruct) aus.
Das Modell bekommt das Bild und gibt direkt strukturiertes JSON zurück, das anschließend geprüft und
normalisiert wird (Dezimalkomma, Mengen, fehlende Felder).

Ausgabe::

    {"success": true, "items": [{"title": "Brot", "amount": 1, "price": 3.5}], "retailer": "Rewe", ...}

``price`` ist der Gesamtpreis der Position (Menge x Einzelpreis), wie er rechts auf dem Bon steht.

Speicher (8 GB VRAM):
 - quantization ``auto``: 4-bit (bitsandbytes, ~3 GB VRAM) bei GPUs unter 12 GB, sonst fp16/bf16 (~8 GB)
 - ``max_pixels`` begrenzt die Bild-Tokens (Standard 1024*28*28 ≈ 0,8 MP, reicht für Bons gut aus)
 - ohne GPU läuft das Modell auf der CPU (langsam, ~1 min pro Bon)

Das Modell wird erst beim ersten Job geladen und nach ``idle_unload`` Sekunden ohne Job (Standard 600 = 10 min)
wieder entladen, damit der Grafikspeicher für andere Runner frei wird. ``idle_unload=0`` schaltet das ab.
"""
import atexit
import collections
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

from nwebclient import runner as r
from nwebclient.runner import TAG
from nwebclient import dev as d

from nxml.system import select_device


PROMPT = """Du siehst das Foto eines Kassenbons. Lies alle gekauften Artikel aus.
Antworte NUR mit JSON in genau diesem Format, ohne weiteren Text:
{"retailer": "Name des Geschäfts", "date": "YYYY-MM-DD oder null", "total": 12.34, "currency": "EUR",
 "items": [{"title": "Artikelname", "amount": 1, "price": 1.99}]}
Regeln:
- retailer: Name des Händlers/der Kette (z.B. Rewe, Aldi, Lidl, Edeka, dm), nicht die Adresse.
- title: Artikelbezeichnung wie auf dem Bon.
- amount: Menge/Stückzahl (Standard 1, bei Gewichtsartikeln das Gewicht in kg).
- price: Gesamtpreis der Zeile als Zahl mit Punkt als Dezimaltrenner (Menge x Einzelpreis).
- Pfand und Rabatte sind eigene Positionen (Rabatte mit negativem Preis).
- Keine Positionen für Summe, Zwischensumme, MwSt, Bar, Rückgeld, EC-Karte oder Zahlungsart.
- total: Gesamtsumme des Bons."""


def _to_float(value, default=None):
    """ "3,50 €" / "3.50" / 3.5 / "1.234,56" -> float """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace('€', '').replace('EUR', '').replace(' ', '')
    neg = s.endswith('-') or s.startswith('-')
    s = s.strip('-+')
    if ',' in s and '.' in s:
        # Tausendertrenner entfernen: der letzte Trenner ist das Dezimalzeichen
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(',', '.')
    m = re.search(r'\d+(?:\.\d+)?', s)
    if not m:
        return default
    v = float(m.group(0))
    return -v if neg else v


def _extract_json(text):
    """ Sucht das JSON-Objekt in der Modell-Antwort (auch in ```json ... ``` Blöcken). """
    text = text.strip()
    text = re.sub(r'^```(?:json)?', '', text).strip()
    text = re.sub(r'```$', '', text).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except ValueError:
            # häufige Fehler kleiner Modelle: Komma vor schließender Klammer
            return json.loads(re.sub(r',\s*([}\]])', r'\1', candidate))
    raise ValueError('Keine JSON-Antwort vom Modell: ' + text[:200])


def normalize_receipt(raw):
    """ Bringt die Modell-Antwort in das Ausgabeformat {items, retailer, total, date, currency}. """
    if isinstance(raw, list):
        raw = {'items': raw}
    items = []
    for it in raw.get('items') or []:
        if not isinstance(it, dict):
            continue
        title = str(it.get('title') or it.get('name') or '').strip()
        price = _to_float(it.get('price', it.get('total')))
        if not title or price is None:
            continue
        amount = _to_float(it.get('amount', it.get('quantity')), 1.0)
        if amount == 0:
            amount = 1.0
        items.append({
            'title': title,
            'amount': int(amount) if float(amount).is_integer() else round(amount, 3),
            'price': round(price, 2)
        })
    retailer = raw.get('retailer') or raw.get('store') or raw.get('shop') or ''
    total = _to_float(raw.get('total'))
    date = raw.get('date')
    return {
        'items': items,
        'retailer': str(retailer).strip(),
        'total': round(total, 2) if total is not None else None,
        'date': date if date not in ('', 'null', None) else None,
        'currency': raw.get('currency') or 'EUR'
    }


class BonReader(r.ImageExecutor):
    """
    Liest Kassenbons (Foto) mit einem Vision-Language-Modell aus, ausgelegt für GPUs mit 8 GB VRAM.

    Ausgabe: {success: true, items: [{title: "Brot", amount: 1, price: 3.5}], retailer: "Rewe"}
    https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
    """
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR]

    MODULES = ['torch', 'torchvision', 'transformers>=4.49', 'accelerate', 'bitsandbytes']

    QUANTIZATIONS = ['auto', '4bit', '8bit', 'none']

    def __init__(self, type='bon', model_id='Qwen/Qwen2.5-VL-3B-Instruct', device_mode='auto',
                 quantization='auto', max_pixels=1024 * 28 * 28, max_new_tokens=2048,
                 idle_unload=600):
        """
            :param model_id: Hugging-Face-Modell (Image-Text-to-Text mit Chat-Template, z.B. Qwen2.5-VL / Qwen3-VL)
            :param device_mode: "cpu", "gpu" oder "auto"
            :param quantization: "auto" (4bit bei < 12 GB VRAM), "4bit", "8bit" oder "none"
            :param max_pixels: maximale Bildgröße in Pixeln für das Modell (weniger = weniger VRAM)
            :param max_new_tokens: maximale Länge der Antwort (lange Bons brauchen mehr)
            :param idle_unload: Modell nach so vielen Sekunden ohne Job entladen, 0 = nie
        """
        super().__init__(type)
        if quantization not in self.QUANTIZATIONS:
            raise ValueError(f'quantization must be one of {self.QUANTIZATIONS}, got {quantization!r}')
        self.model_id = model_id
        self.device_mode = device_mode
        self.quantization = quantization
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.device = None
        self.model = None
        self.processor = None
        self.idle_unload = float(idle_unload)
        self.last_used = 0.0
        # RLock: read() hält ihn während der Inferenz, damit der Idle-Timer nicht mitten im Job entlädt
        self.lock = threading.RLock()
        self.idle_timer = None
        # Bild: Base64, Pfad oder nweb-ID (image_id)
        f = self.define_sig(d.Param('image', is_pos=False))\
            .set_description("Liest einen Kassenbon aus: Positionen (title, amount, price) und Händler (retailer)")
        f.r(success='bool', items='list', retailer='str', total='float', date='str', currency='str',
            duration='float')
        f.r_fail()

    def _use_quantization(self):
        import torch
        if self.device != 'cuda':
            return None
        if self.quantization in ('4bit', '8bit'):
            return self.quantization
        if self.quantization == 'none':
            return None
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            self.info("bitsandbytes nicht installiert, lade ohne Quantisierung")
            return None
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return '4bit' if vram < 12 else None

    def load(self):
        with self.lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.device = select_device(self.device_mode)
            if self.device == 'cpu':
                dtype = torch.float32
            elif self.device == 'cuda' and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=256 * 28 * 28,
                                                           max_pixels=self.max_pixels)
            quant = self._use_quantization()
            if quant:
                from transformers import BitsAndBytesConfig
                if quant == '4bit':
                    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
                else:
                    qcfg = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForImageTextToText.from_pretrained(self.model_id, quantization_config=qcfg,
                                                                    dtype=dtype, device_map={'': 0})
            else:
                model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype)\
                    .to(self.device)
            self.model = model.eval()
            self.last_used = time.time()
            self.info(f"Loaded {self.model_id} on {self.device} (quantization={quant or 'none'})")

    def _schedule_unload(self, delay=None):
        """ (Re-)startet den Idle-Timer, der das Modell nach idle_unload Sekunden ohne Job entlädt. """
        if self.idle_unload <= 0:
            return
        if self.idle_timer is not None:
            self.idle_timer.cancel()
        self.idle_timer = threading.Timer(self.idle_unload if delay is None else delay, self._idle_check)
        self.idle_timer.daemon = True
        self.idle_timer.start()

    def _idle_check(self):
        with self.lock:
            if self.model is None:
                return
            idle = time.time() - self.last_used
            if idle < self.idle_unload:
                # zwischenzeitlich gab es einen Job
                self._schedule_unload(self.idle_unload - idle)
                return
            self.info(f"Unload {self.model_id} nach {int(idle)}s ohne Job")
            self.unload()

    def unload(self):
        with self.lock:
            if self.idle_timer is not None:
                self.idle_timer.cancel()
                self.idle_timer = None
            if self.model is None:
                return
            self.model = None
            self.processor = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def read(self, image):
        """ :return: Roh-Antwort des Modells (Text) """
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image).convert('RGB')
        with self.lock:
            try:
                self.load()
                return self._generate(image)
            finally:
                self.last_used = time.time()
                self._schedule_unload()

    def _generate(self, image):
        import torch
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05)
        return self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

    def executeImage(self, image, data):
        try:
            start = time.time()
            text = self.read(image)
            result = normalize_receipt(_extract_json(text))
            return self.success(**result, duration=round(time.time() - start, 3))
        except Exception as e:
            self.error(str(e))
            return self.fail(str(e))


# ---------------------------------------------------------------------------------------------------------------
# DeepSeek-OCR über vLLM
# ---------------------------------------------------------------------------------------------------------------

DEEPSEEK_PROMPTS = {
    'free': 'Free OCR.',
    'markdown': '<|grounding|>Convert the document to markdown.',
    'ocr': '<|grounding|>OCR this image.',
    'figure': 'Parse the figure.',
    'describe': 'Describe this image in detail.',
}

# <|ref|>Text<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>  (Koordinaten normiert auf 0..999)
_DS_GROUNDING = re.compile(r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>', re.S)


def parse_deepseek_grounding(text, width, height):
    """
        Zerlegt die Grounding-Ausgabe von DeepSeek-OCR in Text und Boxen.

        :return: (Text ohne Grounding-Tags, [{'label': 'text', 'box': [x1, y1, x2, y2]}, ...]) in Pixeln
    """
    boxes = []
    for m in _DS_GROUNDING.finditer(text):
        label = m.group(1).strip()
        try:
            coords = json.loads(m.group(2))
        except ValueError:
            continue
        for c in coords if coords and isinstance(coords[0], list) else [coords]:
            if len(c) != 4:
                continue
            boxes.append({'label': label, 'box': [round(c[0] / 999 * width), round(c[1] / 999 * height),
                                                   round(c[2] / 999 * width), round(c[3] / 999 * height)]})
    clean = _DS_GROUNDING.sub('', text)
    clean = re.sub(r'<\|/?(ref|det|grounding)\|>', '', clean)
    clean = clean.replace('<｜end▁of▁sentence｜>', '')
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
    return clean, boxes


DEEPSEEK_UI_CSS = """
.ds { max-width: 1100px; margin-bottom: 16px; }
.ds-banner { padding: 10px 14px; border-radius: 8px; font-weight: bold; margin: 6px 0 10px; }
.ds-ok { background: #d9f2df; color: #145a26; }
.ds-run { background: #dde9fb; color: #17407a; }
.ds-warn { background: #fdf0d2; color: #6b4a00; }
.ds-err { background: #fadcdc; color: #7a1717; }
.ds-grid { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 8px 0; }
.ds-grid div:nth-child(odd) { opacity: .7; }
.ds-grid code { word-break: break-all; }
.ds-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
.ds-log { max-height: 260px; overflow: auto; font-size: 12px; white-space: pre-wrap; padding: 8px;
          background: rgba(127,127,127,.12); border-radius: 6px; }
.ds-msg { min-height: 1.2em; margin: 4px 0; }
"""

DEEPSEEK_UI_HTML = """
<section class="page_section ds">
  <h2>DeepSeek-OCR: Installation</h2>
  <div class="ds-banner" id="ds-banner"></div>
  <div class="ds-grid" id="ds-grid"></div>
  <div class="ds-actions">
    <button type="button" id="ds-setup">Setup starten</button>
    <button type="button" id="ds-refresh">Aktualisieren</button>
    <button type="button" id="ds-stop">Server stoppen</button>
    <button type="button" id="ds-clean">venv löschen</button>
  </div>
  <div class="ds-msg" id="ds-msg"></div>
  <details id="ds-logbox"><summary>Log</summary><pre class="ds-log" id="ds-log"></pre></details>
</section>
"""

DEEPSEEK_UI_JS = """
  const cfg = DS_CFG;
  const $ = (id) => document.getElementById(id);
  const rootEl = $('ds-banner').closest('.ds');
  const alive = () => rootEl.isConnected;
  const job = (data) => fetch(location.protocol + '//' + location.host + '/', { method: 'POST',
    body: new URLSearchParams({ type: cfg.type, ...data }) }).then((r) => r.json());
  let timer = null;

  function banner(s) {
    if (s.setup_running) return ['ds-run', 'Installation läuft: ' + (s.setup || '').replace(/^running:?\\s*/, '')];
    if (s.setup && s.setup.startsWith('failed')) return ['ds-err', 'Installation fehlgeschlagen: ' + s.setup.slice(8)];
    if (s.up) return ['ds-ok', 'vLLM-Server läuft' + (s.own_server ? ' (vom Runner gestartet)' : ' (extern)')];
    if (s.own_server) return ['ds-run', 'vLLM-Server startet (beim ersten Mal wird das Modell geladen) ...'];
    if (s.vllm_installed) return ['ds-ok', 'vLLM installiert, Server startet beim nächsten Job'];
    if (s.backend === 'server') return ['ds-warn', 'Kein vLLM-Server unter ' + s.url];
    return ['ds-warn', 'vLLM nicht installiert' + (s.auto_setup ? ', der erste Job startet die Installation' : '')];
  }

  function row(grid, label, value) {
    const a = document.createElement('div'); a.textContent = label;
    const b = document.createElement('div');
    if (value instanceof Node) b.appendChild(value); else b.textContent = value;
    grid.append(a, b);
  }
  const code = (t) => { const c = document.createElement('code'); c.textContent = t; return c; };
  const yes = (v, y, n) => v ? '✔ ' + (y || 'ja') : '✘ ' + (n || 'nein');

  function render(s) {
    const [cls, text] = banner(s);
    $('ds-banner').className = 'ds-banner ' + cls;
    $('ds-banner').textContent = text;
    const g = $('ds-grid'); g.innerHTML = '';
    row(g, 'Server', yes(s.up, 'erreichbar', 'nicht erreichbar') + '  ' + s.url);
    row(g, 'Backend', s.backend);
    row(g, 'Modell', s.model + '  (' + (s.model_cached ? 'im HF-Cache' : 'noch nicht geladen, ~7 GB') + ')');
    row(g, 'vLLM', s.vllm_installed ? '✔ ' + (s.vllm_version || 'installiert') : '✘ nicht installiert');
    row(g, 'venv', code(s.venv + (s.venv_python ? '  (Python ' + s.venv_python + ')' : '')));
    row(g, 'uv', s.uv ? code(s.uv) : '✘ fehlt, wird beim Setup nach ' + s.uv_dir + ' geladen');
    row(g, 'VRAM-Anteil', Math.round(s.gpu_memory_utilization * 100) + ' %');
    row(g, 'Auto-Setup', s.auto_setup ? 'an' : 'aus');
    $('ds-log').textContent = (s.log || []).join('\\n') || '(leer)';
    const log = $('ds-log'); log.scrollTop = log.scrollHeight;
    if (s.setup_running) $('ds-logbox').open = true;
    $('ds-setup').disabled = s.setup_running || s.vllm_installed;
    $('ds-stop').disabled = !s.own_server;
    $('ds-clean').disabled = s.setup_running || !s.vllm_installed && !s.venv_python;
  }

  function schedule(s) {
    clearTimeout(timer);
    if (!alive()) return;
    const busy = s.setup_running || (s.own_server && !s.up);
    timer = setTimeout(refresh, busy ? 3000 : 20000);
  }

  function refresh() {
    if (!alive()) return;
    job({ op: 'status' }).then((s) => { render(s); schedule(s); })
      .catch((e) => { $('ds-msg').textContent = 'Status nicht abrufbar: ' + e; schedule({}); });
  }

  function action(data, confirmText) {
    if (confirmText && !confirm(confirmText)) return;
    $('ds-msg').textContent = '...';
    job(data).then((r) => { $('ds-msg').textContent = r.message || ''; refresh(); })
      .catch((e) => { $('ds-msg').textContent = 'Fehler: ' + e; });
  }

  $('ds-setup').onclick = () => action({ op: 'setup' });
  $('ds-refresh').onclick = refresh;
  $('ds-stop').onclick = () => action({ op: 'stop_server' });
  $('ds-clean').onclick = () => action({ op: 'clean_files' },
    'Server beenden und ' + cfg.status.venv + ' löschen? uv und der Modell-Cache bleiben erhalten.');

  render(cfg.status);
  schedule(cfg.status);
"""


class DeepSeekOcr(r.ImageExecutor):
    """
    Texterkennung mit DeepSeek-OCR (~3B, https://huggingface.co/deepseek-ai/DeepSeek-OCR), ausgeführt mit vLLM.

    vLLM läuft als eigener Server-Prozess (OpenAI-kompatible API), dieser Runner spricht nur HTTP mit ihm.
    Dadurch landet vLLM (und das torch, das es mitbringt) NIE im Python dieses Runners:

     - backend ``venv``: startet ``vllm serve`` aus einem eigenen venv (Standard ``~/.venvs/vllm``).
       Das venv sieht ``~/.local`` nicht (PYTHONNOUSERSITE=1), torch/transformers im User-site bleiben unberührt.
       Einrichten mit dem Job ``{"op": "setup"}`` (oder automatisch beim ersten Job): installiert mit uv nur ins venv, auf einem von uv verwalteten
       Python ``python_version``. Fehlt uv, wird die Binary nach ``~/.venvs/.uv`` geladen (kein pip, kein PATH).
     - backend ``docker``: Image ``vllm/vllm-openai`` (braucht nvidia-container-toolkit für ``--gpus all``)
     - backend ``server``: nur einen bereits laufenden vLLM-Server unter ``url`` verwenden
     - backend ``auto``: läuft unter ``url`` schon ein Server -> den, sonst ``venv``, falls installiert

    Selbst gestartete Server werden nach ``idle_unload`` Sekunden ohne Job beendet (VRAM frei für andere Runner),
    fremde Server werden nie beendet.

    Operationen (@op, Aufruf ``{"type": "deepseek_ocr", "op": "<name>"}``): ``setup``, ``status``,
    ``stop_server``, ``clean_files`` (Server beenden und venv löschen).

    Modi (``mode``): free (reiner Text), markdown (Dokument -> Markdown mit Boxen), ocr (Text mit Boxen),
    figure (Diagramme/Grafiken), describe (Bildbeschreibung) oder eigener ``prompt``.

    Ausgabe::

        {"success": true, "text": "...", "boxes": [{"label": "text", "box": [x1, y1, x2, y2]}], "raw": "..."}
    """
    TAGS = [TAG.IMAGE, TAG.IMAGE_EXTRACTOR]

    # Absichtlich OHNE vllm/torch: nwebclient installiert MODULES per `sys.executable -m pip install` ins
    # Runner-Python, vllm würde dort das vorhandene torch ersetzen. vLLM kommt über setup() ins eigene venv.
    MODULES = ['requests']

    BACKENDS = ['auto', 'venv', 'docker', 'server']

    # Parameter aus dem vLLM-Rezept für DeepSeek-OCR: N-Gram-Sperre gegen Endlos-Wiederholungen,
    # <td>/</td> bleiben für Tabellen erlaubt
    LOGITS_PROCESSOR = 'vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor'
    VLLM_XARGS = {'ngram_size': 30, 'window_size': 90, 'whitelist_token_ids': [128821, 128822]}

    def __init__(self, type='deepseek_ocr', model_id='deepseek-ai/DeepSeek-OCR', backend='auto',
                 url='http://127.0.0.1:8017', venv='~/.venvs/vllm', python_version='3.12',
                 gpu_memory_utilization=0.75, max_model_len=8192, max_tokens=4096, startup_timeout=1200,
                 idle_unload=600, docker_image='vllm/vllm-openai:latest', extra_args='', auto_setup=True):
        """
            :param model_id: Hugging-Face-Modell (deepseek-ai/DeepSeek-OCR)
            :param backend: "auto", "venv", "docker" oder "server"
            :param url: Basis-URL des vLLM-Servers (der Port gilt auch für selbst gestartete Server)
            :param venv: Pfad des isolierten venv mit vLLM
            :param python_version: Python für das venv, wenn es mit uv angelegt wird
            :param gpu_memory_utilization: Anteil des VRAM, den vLLM reserviert (0.75 von 12 GB ≈ 9 GB)
            :param max_model_len: maximale Kontextlänge
            :param max_tokens: maximale Länge der Antwort
            :param startup_timeout: Sekunden Wartezeit auf den Server (der erste Start lädt ~7 GB Modell)
            :param idle_unload: selbst gestarteten Server nach so vielen Sekunden ohne Job beenden, 0 = nie
            :param docker_image: Image für backend "docker"
            :param extra_args: zusätzliche Argumente für "vllm serve"
            :param auto_setup: fehlt vLLM, startet der erste Job die Installation mit uv ins venv
        """
        super().__init__(type)
        if backend not in self.BACKENDS:
            raise ValueError(f'backend must be one of {self.BACKENDS}, got {backend!r}')
        self.model_id = model_id
        self.backend = backend
        self.url = url.rstrip('/')
        self.venv = os.path.expanduser(venv)
        self.python_version = str(python_version)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = int(max_model_len)
        self.max_tokens = int(max_tokens)
        self.startup_timeout = float(startup_timeout)
        self.idle_unload = float(idle_unload)
        self.docker_image = docker_image
        self.extra_args = extra_args
        self.auto_setup = str(auto_setup).lower() not in ('0', 'false', 'no', '')
        self.proc = None            # selbst gestarteter vLLM-Prozess (venv oder docker run)
        self.container_name = None  # bei backend docker
        self.server_log = collections.deque(maxlen=300)
        self.setup_thread = None
        self.setup_state = None
        self.last_used = 0.0
        self.lock = threading.RLock()
        self.idle_timer = None
        atexit.register(self.unload)
        f = self.define_sig(d.Param('image', is_pos=False), d.PStr('mode', 'free'), d.PStr('prompt', ''))\
            .set_description("Erkennt Text mit DeepSeek-OCR (vLLM), mode: free, markdown, ocr, figure, describe")
        f.r(success='bool', text='str', boxes='list', raw='str', mode='str', duration='float')
        f.r_fail()

    # --- Server -------------------------------------------------------------------------------------------------

    @property
    def port(self):
        from urllib.parse import urlparse
        return urlparse(self.url).port or 80

    @property
    def vllm_bin(self):
        return os.path.join(self.venv, 'bin', 'vllm')

    def is_up(self, timeout=2):
        import requests
        try:
            return requests.get(self.url + '/health', timeout=timeout).status_code == 200
        except requests.RequestException:
            return False

    def _isolated_env(self):
        """ Umgebung für Prozesse im venv: kein User-site (~/.local), kein PYTHONPATH des Runners """
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        env['PYTHONNOUSERSITE'] = '1'
        env['VIRTUAL_ENV'] = self.venv
        env['PATH'] = os.path.join(self.venv, 'bin') + os.pathsep + env.get('PATH', '')
        return env

    def _serve_args(self, host='127.0.0.1'):
        """ Argumente nach "vllm serve" """
        args = [self.model_id, '--host', host, '--port', str(self.port), '--served-model-name', self.model_id,
                '--logits_processors', self.LOGITS_PROCESSOR, '--no-enable-prefix-caching',
                '--mm-processor-cache-gb', '0', '--gpu-memory-utilization', str(self.gpu_memory_utilization),
                '--max-model-len', str(self.max_model_len)]
        if self.extra_args:
            args += shlex.split(self.extra_args)
        return args

    def _resolve_backend(self):
        if self.backend != 'auto':
            return self.backend
        if self.is_up():
            return 'server'
        if os.path.exists(self.vllm_bin):
            return 'venv'
        self._require_vllm()

    def _require_vllm(self):
        """
            Fehlt vLLM im venv, wird die Installation (uv, nur ins venv) bei auto_setup im Hintergrund gestartet.
            Der Job selbst wartet nicht (die Installation dauert Minuten), sondern meldet den Stand.
        """
        if os.path.exists(self.vllm_bin):
            return
        hint = f'Fortschritt mit dem Job {{"type": "{self.type}", "op": "status"}}, danach den Job wiederholen.'
        if self.setup_thread is not None and self.setup_thread.is_alive():
            raise RuntimeError(f'vLLM wird gerade mit uv nach {self.venv} installiert ({self.setup_state}). ' + hint)
        if not self.auto_setup:
            raise RuntimeError(f'Kein vLLM-Server unter {self.url} und kein vLLM im venv {self.venv}. '
                               f'Einrichten mit dem Job {{"type": "{self.type}", "op": "setup"}} '
                               f'oder backend="server" mit laufendem Server verwenden.')
        if self.setup_state and self.setup_state.startswith('failed'):
            raise RuntimeError(f'Installation von vLLM fehlgeschlagen: {self.setup_state[8:]}. Log über '
                               f'{{"op": "status"}}, neuer Versuch mit {{"op": "setup"}}.')
        res = self.setup()
        if not res.get('success'):
            raise RuntimeError(res.get('message'))
        raise RuntimeError(f'vLLM ist noch nicht installiert, Installation mit uv nach {self.venv} gestartet '
                           f'(einige GB, dauert einige Minuten). ' + hint)

    def _read_output(self, proc):
        for line in iter(proc.stdout.readline, ''):
            self.server_log.append(line.rstrip())
        proc.stdout.close()

    def start_server(self):
        with self.lock:
            if self.is_up():
                return
            if self.proc is not None and self.proc.poll() is None:
                self._wait_up()
                return
            backend = self._resolve_backend()
            if backend == 'server':
                raise RuntimeError(f'vLLM-Server unter {self.url} nicht erreichbar')
            if backend == 'venv':
                self._require_vllm()
                cmd = [self.vllm_bin, 'serve'] + self._serve_args()
                env = self._isolated_env()
                self.container_name = None
            else:
                self.container_name = f'nxml-{self.type}-{self.port}'
                hf_cache = os.path.expanduser('~/.cache/huggingface')
                # das Image startet "vllm serve" selbst, im Container auf allen Interfaces lauschen,
                # nach außen nur auf localhost veröffentlichen
                cmd = ['docker', 'run', '--rm', '--name', self.container_name, '--gpus', 'all', '--ipc=host',
                       '-p', f'127.0.0.1:{self.port}:{self.port}', '-v', f'{hf_cache}:/root/.cache/huggingface',
                       self.docker_image] + self._serve_args(host='0.0.0.0')
                env = None
            self.info('Starte vLLM: ' + ' '.join(shlex.quote(c) for c in cmd))
            self.server_log.clear()
            self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                         start_new_session=True)
            threading.Thread(target=self._read_output, args=(self.proc,), daemon=True).start()
            self._wait_up()

    def _wait_up(self):
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.is_up():
                self.info(f'vLLM bereit unter {self.url}')
                return
            if self.proc is not None and self.proc.poll() is not None:
                tail = '\n'.join(list(self.server_log)[-20:])
                self.proc = None
                raise RuntimeError('vLLM-Server hat sich beim Start beendet:\n' + tail)
            time.sleep(2)
        raise RuntimeError(f'vLLM-Server nach {int(self.startup_timeout)}s nicht bereit')

    def unload(self):
        """ Beendet den selbst gestarteten vLLM-Server, fremde Server bleiben laufen. """
        with self.lock:
            if self.idle_timer is not None:
                self.idle_timer.cancel()
                self.idle_timer = None
            proc, self.proc = self.proc, None
            if proc is None or proc.poll() is not None:
                return
            self.info('Beende vLLM-Server')
            if self.container_name:
                subprocess.run(['docker', 'stop', self.container_name], capture_output=True, timeout=60)
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(proc.pid, signal.SIGTERM)  # ganze Prozessgruppe inkl. vLLM-Worker
                else:
                    proc.terminate()
                proc.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                if hasattr(os, 'killpg'):
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()

    def _schedule_unload(self, delay=None):
        if self.idle_unload <= 0 or self.proc is None:
            return
        if self.idle_timer is not None:
            self.idle_timer.cancel()
        self.idle_timer = threading.Timer(self.idle_unload if delay is None else delay, self._idle_check)
        self.idle_timer.daemon = True
        self.idle_timer.start()

    def _idle_check(self):
        with self.lock:
            idle = time.time() - self.last_used
            if idle < self.idle_unload:
                self._schedule_unload(self.idle_unload - idle)
                return
            self.info(f'vLLM-Server nach {int(idle)}s ohne Job beenden')
            self.unload()

    # --- Einrichtung (nur ins venv) -----------------------------------------------------------------------------

    # uv als einzelne Binary von den GitHub-Releases (kein Installer-Skript, keine Änderung an .bashrc/.profile)
    UV_RELEASE_URL = 'https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-unknown-linux-gnu.tar.gz'

    @property
    def uv_dir(self):
        """ eigenes Verzeichnis für uv und die von uv verwaltete Python-Version, neben dem venv """
        return os.path.join(os.path.dirname(self.venv.rstrip('/')), '.uv')

    def find_uv(self):
        """ :return: Pfad zu uv (PATH, ~/.local/bin oder das eigene uv_dir), None wenn nicht vorhanden """
        for path in (shutil.which('uv'), os.path.expanduser('~/.local/bin/uv'), os.path.join(self.uv_dir, 'uv')):
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def install_uv(self):
        """ Lädt die uv-Binary nach uv_dir (nur dorthin, kein PATH-Eintrag, kein pip). :return: Pfad zu uv """
        import platform
        import tarfile
        import tempfile
        import urllib.request
        arch = {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'aarch64', 'arm64': 'aarch64'}\
            .get(platform.machine().lower())
        if arch is None or not sys.platform.startswith('linux'):
            raise RuntimeError(f'uv-Download nur für Linux x86_64/aarch64, hier {sys.platform}/{platform.machine()}')
        url = self.UV_RELEASE_URL.format(arch=arch)
        os.makedirs(self.uv_dir, exist_ok=True)
        self.server_log.append('Lade uv: ' + url)
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, 'uv.tar.gz')
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    name = os.path.basename(member.name)
                    if member.isfile() and name in ('uv', 'uvx'):
                        src = tar.extractfile(member)
                        dst = os.path.join(self.uv_dir, name)
                        with open(dst, 'wb') as out:
                            shutil.copyfileobj(src, out)
                        os.chmod(dst, 0o755)
        uv = os.path.join(self.uv_dir, 'uv')
        if not os.path.isfile(uv):
            raise RuntimeError('uv nicht im Archiv gefunden: ' + url)
        return uv

    def _uv_env(self):
        """ Umgebung für uv: Python-Downloads nur nach uv_dir/python, keine Shims in ~/.local/bin """
        env = self._isolated_env()
        env.pop('VIRTUAL_ENV', None)
        env['UV_PYTHON_INSTALL_DIR'] = os.path.join(self.uv_dir, 'python')
        env['UV_PYTHON_INSTALL_BIN'] = '0'
        env['UV_PYTHON_PREFERENCE'] = 'only-managed'  # nie das System-Python (3.14) verwenden
        env['UV_NO_MODIFY_PATH'] = '1'
        return env

    def _is_venv(self):
        return os.path.isfile(os.path.join(self.venv, 'pyvenv.cfg')) and \
            os.path.exists(os.path.join(self.venv, 'bin', 'python'))

    def setup_commands(self, uv):
        """
            Befehle, die vLLM mit uv ins isolierte venv installieren. Fasst weder System-Python noch ~/.local an.
            uv lädt ein eigenes Python (python_version), da vLLM nicht jede Python-Version unterstützt, und wählt
            mit --torch-backend=auto die zum NVIDIA-Treiber passenden torch-Wheels (nur im venv).
        """
        create = [] if self._is_venv() else [[uv, 'venv', '--seed', '--python', self.python_version, self.venv]]
        return create + [[uv, 'pip', 'install', '--python', os.path.join(self.venv, 'bin', 'python'),
                 '--torch-backend=auto', 'vllm', 'requests']]

    def _setup_worker(self):
        try:
            uv = self.find_uv()
            if uv is None:
                self.setup_state = 'running: uv herunterladen'
                uv = self.install_uv()
            self.server_log.append('uv: ' + uv)
            for cmd in self.setup_commands(uv):
                self.setup_state = 'running: ' + ' '.join(cmd[1:3])
                self.server_log.append('$ ' + ' '.join(shlex.quote(c) for c in cmd))
                proc = subprocess.Popen(cmd, env=self._uv_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace')
                self._read_output(proc)
                if proc.wait() != 0:
                    raise RuntimeError(f'{" ".join(cmd[1:3])} fehlgeschlagen (Exit {proc.returncode})')
            if not os.path.exists(self.vllm_bin):
                raise RuntimeError('Installation beendet, aber ' + self.vllm_bin + ' fehlt')
            self.setup_state = 'ok'
            self.info(f'vLLM in {self.venv} installiert')
        except Exception as e:
            self.setup_state = 'failed: ' + str(e)
            self.server_log.append('FEHLER: ' + str(e))
            self.error('Setup: ' + str(e))

    @r.op("Installiert vLLM im Hintergrund mit uv ins isolierte venv, Fortschritt mit op=status",
          success={'venv': 'str', 'uv': 'str'}, fail={})
    def setup(self):
        """ Installiert vLLM im Hintergrund mit uv ins venv, Fortschritt über den Job {"op": "status"} """
        if self.setup_thread is not None and self.setup_thread.is_alive():
            return self.success('Setup läuft bereits', setup=self.setup_state)
        if os.path.exists(self.vllm_bin):
            return self.success(f'vLLM ist bereits in {self.venv} installiert')
        if os.path.exists(self.venv) and not self._is_venv():
            # nie in fremde Verzeichnisse installieren; ein abgebrochenes venv von uns wird fortgesetzt
            return self.fail(f'{self.venv} existiert, ist aber kein venv. Bitte prüfen oder ein anderes venv wählen.')
        self.server_log.clear()
        self.setup_state = 'running'
        self.setup_thread = threading.Thread(target=self._setup_worker, daemon=True)
        self.setup_thread.start()
        return self.success('Setup gestartet', venv=self.venv, uv=self.find_uv() or 'wird nach ' + self.uv_dir
                            + ' geladen')

    @r.op("Räumt auf: beendet den eigenen vLLM-Server und löscht das venv (uv und Modell-Cache bleiben)",
          success={'removed': 'list'}, fail={})
    def clean_files(self):
        """
            Räumt auf: beendet den eigenen vLLM-Server und löscht das venv (nur wenn es wirklich ein venv ist).
            Unberührt bleiben uv und dessen Python (uv_dir), der Hugging-Face-Cache und alles außerhalb des venv.
        """
        if self.setup_thread is not None and self.setup_thread.is_alive():
            return self.fail('Setup läuft noch, erst danach aufräumen')
        self.unload()
        if not os.path.exists(self.venv):
            return self.success(f'{self.venv} existiert nicht, nichts zu löschen', removed=[])
        if not self._is_venv():
            return self.fail(f'{self.venv} ist kein venv (pyvenv.cfg fehlt), wird nicht gelöscht')
        shutil.rmtree(self.venv)
        self.setup_state = None
        self.info(f'venv {self.venv} gelöscht')
        return self.success(f'{self.venv} gelöscht', removed=[self.venv])

    def vllm_version(self):
        """ :return: vLLM-Version im venv (aus den dist-info-Ordnern, ohne vLLM zu importieren) oder None """
        import glob
        for path in glob.glob(os.path.join(self.venv, 'lib', 'python*', 'site-packages', 'vllm-*.dist-info')):
            return os.path.basename(path)[len('vllm-'):-len('.dist-info')]
        return None

    def venv_python(self):
        """ :return: Python-Version des venv laut pyvenv.cfg oder None """
        try:
            with open(os.path.join(self.venv, 'pyvenv.cfg')) as f:
                for line in f:
                    key, _, value = line.partition('=')
                    if key.strip() in ('version', 'version_info'):
                        return value.strip()
        except OSError:
            pass
        return None

    def model_cache_dir(self):
        hub = os.environ.get('HF_HUB_CACHE') or os.path.join(
            os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'), 'hub')
        return os.path.join(hub, 'models--' + self.model_id.replace('/', '--'))

    @r.op("Stand von Installation und Server: venv, uv, vLLM-Version, Modell-Cache, Setup-Fortschritt und Log",
          result={'success': 'bool', 'up': 'bool', 'url': 'str', 'vllm_installed': 'bool', 'vllm_version': 'str',
                  'venv': 'str', 'venv_python': 'str', 'uv': 'str', 'model_cached': 'bool', 'own_server': 'bool',
                  'setup': 'str', 'setup_running': 'bool', 'log': 'list'})
    def status(self):
        setup_running = self.setup_thread is not None and self.setup_thread.is_alive()
        own = self.proc is not None and self.proc.poll() is None
        return self.success(up=self.is_up(), url=self.url, backend=self.backend, model=self.model_id,
                            venv=self.venv, vllm_installed=os.path.exists(self.vllm_bin),
                            vllm_version=self.vllm_version(), venv_python=self.venv_python(),
                            uv=self.find_uv(), uv_dir=self.uv_dir, auto_setup=self.auto_setup,
                            model_cached=os.path.isdir(self.model_cache_dir()),
                            own_server=own, setup=self.setup_state, setup_running=setup_running,
                            gpu_memory_utilization=self.gpu_memory_utilization,
                            log=list(self.server_log)[-30:])

    # --- UI -----------------------------------------------------------------------------------------------------

    def part_index(self, p, params={}):
        """ Status der Installation (venv, uv, vLLM, Modell, Server) mit Log und Aktionen, darunter das Bildformular """
        cfg = {'type': self.type, 'status': self.status()}
        p('<style>' + DEEPSEEK_UI_CSS + '</style>')
        p(DEEPSEEK_UI_HTML)
        # Block statt globaler Konstante: die UI v2 führt Skripte bei jedem Öffnen erneut aus
        p('<script>{ const DS_CFG = ' + json.dumps(cfg).replace('</', '<\\/') + ';\n' + DEEPSEEK_UI_JS + '\n}</script>')
        super().part_index(p, params)

    # --- Jobs ---------------------------------------------------------------------------------------------------

    @r.op("Beendet den vom Runner gestarteten vLLM-Server (externe Server bleiben laufen)",
          result={'success': 'bool', 'message': 'str'})
    def stop_server(self):
        self.unload()
        return self.success('gestoppt')

    def _prompt(self, data):
        prompt = (data.get('prompt') or '').strip()
        if prompt:
            return prompt
        mode = data.get('mode') or 'free'
        if mode not in DEEPSEEK_PROMPTS:
            raise ValueError(f'mode must be one of {list(DEEPSEEK_PROMPTS)}, got {mode!r}')
        return DEEPSEEK_PROMPTS[mode]

    def read(self, image, prompt):
        """ :return: Roh-Antwort des Modells """
        import base64
        import io
        import requests
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        image_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
        with self.lock:
            self.start_server()
            self.last_used = time.time()
        try:
            resp = requests.post(self.url + '/v1/chat/completions', timeout=600, json={
                'model': self.model_id,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                    {'type': 'text', 'text': prompt}]}],
                'temperature': 0.0,
                'max_tokens': self.max_tokens,
                'skip_special_tokens': False,
                'vllm_xargs': self.VLLM_XARGS,
            })
            if resp.status_code != 200:
                raise RuntimeError(f'vLLM {resp.status_code}: {resp.text[:500]}')
            return resp.json()['choices'][0]['message']['content'] or ''
        finally:
            self.last_used = time.time()
            self._schedule_unload()

    def executeImage(self, image, data):
        from PIL import ImageOps
        try:
            start = time.time()
            image = ImageOps.exif_transpose(image).convert('RGB')
            raw = self.read(image, self._prompt(data))
            text, boxes = parse_deepseek_grounding(raw, image.width, image.height)
            return self.success(text=text, boxes=boxes, raw=raw, mode=data.get('mode') or 'free',
                                duration=round(time.time() - start, 3))
        except Exception as e:
            self.error(str(e))
            return self.fail(str(e))
