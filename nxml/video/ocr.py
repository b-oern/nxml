#
# Project: nxml
#
# OCR eines Bildausschnitts in einem Video, z.B. der Anzeige eines Messgeräts
#

import base64
import csv
import io
import json
import os
import re
import shutil
import time
from threading import Thread

from nwebclient import runner as r
from nwebclient.runner import TAG
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import dev as d

from nxml.video.clip import VideoFrames


class VideoOcr(r.BaseJobExecutor):
    """
    Liest in regelmäßigen Abständen einen Bildausschnitt (z.B. die Anzeige eines Messgeräts) aus einem Video
    und erkennt den Text mit einem OCR-Runner (ocr, ocrai, ...). Ergebnis: Tabelle mit Zeit und Wert.

    UI: Video hochladen -> Bereich im Bild wählen -> Abstand wählen und auswerten -> Export CSV/JSON

    Videos und Ergebnisse liegen in data_dir/<video_id>/.
    """
    TAGS = ['video', TAG.IMAGE_EXTRACTOR]

    MODULES = ['opencv-python-headless', 'pillow']

    NUMBER = re.compile(r'[-+]?\d+(?:[.,]\d+)?')

    # Zeichen, die OCR-Modelle statt eines Dezimalpunkts liefern
    DOTS = str.maketrans({'。': '.', '．': '.', '·': '.', '，': ','})

    # Optionen, die an den OCR-Runner weitergegeben werden
    OCR_PARAMS = ['lang', 'psm', 'min_confidence', 'preprocess', 'format']

    def __init__(self, type='video_ocr', data_dir='video_ocr', ocr_runners='ocr,ocrai', ffmpeg='ffmpeg'):
        """
            :param data_dir: Ordner für hochgeladene Videos und Ergebnisse
            :param ocr_runners: Runner, die zur Auswahl stehen (Komma-getrennt), zusätzlich alle geladenen
                                Runner mit dem Tag image_extract
            :param ffmpeg: ffmpeg-Binary, nur nötig wenn OpenCV nicht installiert ist
        """
        super().__init__(type)
        self.data_dir = data_dir
        self.ocr_runners = [s.strip() for s in ocr_runners.split(',') if s.strip()] \
            if isinstance(ocr_runners, str) else list(ocr_runners)
        self.ffmpeg = ffmpeg
        self.jobs = {}
        self.local_runners = {}
        self.define_sig(d.PStr('op', 'upload'), d.Param('file_data'), d.PStr('name', 'video.mp4'))\
            .set_description("Video (Base64) speichern, liefert video_id und Dauer, FPS, Größe")\
            .r(success='bool', video_id='str', duration='float', fps='float', width='int', height='int')
        self.define_sig(d.PStr('op', 'upload'), d.Param('path'))\
            .set_description("Video aus einem Pfad auf dem Server übernehmen")\
            .r(success='bool', video_id='str', duration='float', fps='float', width='int', height='int')
        self.define_sig(d.PStr('op', 'frame'), d.Param('video_id'), d.NP('t', 'float', 0))\
            .set_description("Einzelbild zum Zeitpunkt t (Sekunden) als JPEG-Base64").r(success='bool', image='str')
        self.define_sig(d.PStr('op', 'ocr_frame'), d.Param('video_id'), d.NP('t', 'float', 0),
                        d.PStr('box', '0,0,100,50'), d.PStr('ocr', 'ocr'), d.PInt('number_only', 1),
                        d.PStr('decimals', ''), d.PInt('invert', 0))\
            .set_description("OCR des Bereichs box=x,y,w,h (Pixel) eines Einzelbilds, decimals: feste Anzahl "
                             "Nachkommastellen, falls die OCR den Dezimalpunkt nicht erkennt")\
            .r(success='bool', text='str', value='str', confidence='float')
        f = self.define_sig(d.PStr('op', 'process'), d.Param('video_id'), d.PStr('box', '0,0,100,50'),
                            d.NP('interval', 'float', 30), d.NP('start', 'float', 0), d.NP('end', 'float', 0),
                            d.PStr('ocr', 'ocr'), d.PInt('number_only', 1), d.PStr('decimals', ''),
                            d.PInt('invert', 0))
        f.set_description("Startet die Auswertung im Hintergrund (end=0: bis zum Ende), Fortschritt über op=status")
        f.r(success='bool', job_id='str', count='int')
        self.define_sig(d.PStr('op', 'status'), d.Param('job_id'), d.PInt('offset', 0))\
            .set_description("Fortschritt und Ergebniszeilen ab offset")\
            .r(success='bool', status='str', done='int', count='int', rows='list')
        self.define_sig(d.PStr('op', 'cancel'), d.Param('job_id')).set_description("Auswertung abbrechen").r()
        self.define_sig(d.PStr('op', 'export'), d.Param('job_id'), d.PStr('format', 'csv'))\
            .set_description("Ergebnis als csv oder json").r(success='bool', content='str')
        self.define_sig(d.PStr('op', 'runners')).set_description("Verfügbare OCR-Runner").r(runners='list')

    # -- Videos --

    def video_dir(self, video_id):
        if not re.fullmatch(r'[\w-]+', str(video_id)):
            raise ValueError(f'Invalid video_id {video_id!r}')
        return os.path.join(self.data_dir, video_id)

    def video_path(self, video_id):
        folder = self.video_dir(video_id)
        if not os.path.isdir(folder):
            raise ValueError(f'Unknown video_id {video_id!r}')
        for name in os.listdir(folder):
            if name.startswith('video.'):
                return os.path.join(folder, name)
        raise ValueError(f'No video in {folder}')

    def create_video(self, name):
        """ :return: (video_id, Zielpfad) für ein neues Video """
        video_id = u.guid()
        folder = self.video_dir(video_id)
        os.makedirs(folder, exist_ok=True)
        ext = os.path.splitext(name or '')[1].lower()
        if not re.fullmatch(r'\.\w{1,5}', ext):
            ext = '.mp4'
        return video_id, os.path.join(folder, 'video' + ext)

    def video_info(self, video_id) -> dict:
        frames = VideoFrames(self.video_path(video_id), self.ffmpeg)
        try:
            return {'video_id': video_id, **frames.info()}
        finally:
            frames.close()

    def store_upload(self, name, write):
        """ write(path) speichert das Video, :return: Ergebnis mit video_id und Infos """
        video_id, path = self.create_video(name)
        write(path)
        try:
            info = self.video_info(video_id)
        except Exception as e:
            shutil.rmtree(self.video_dir(video_id), ignore_errors=True)
            return self.fail("Video kann nicht gelesen werden: " + str(e))
        with open(os.path.join(self.video_dir(video_id), 'info.json'), 'w') as f:
            json.dump({**info, 'name': os.path.basename(name or '')}, f)
        return self.success(**info)

    def execute_upload(self, data):
        if 'path' in data:
            if not os.path.isfile(data['path']):
                return self.fail(f"File not found: {data['path']}")
            return self.store_upload(data['path'], lambda path: shutil.copyfile(data['path'], path))

        def write(path):
            with open(path, 'wb') as f:
                f.write(base64.b64decode(data['file_data']))
        return self.store_upload(data.get('name', 'video.mp4'), write)

    def execute_info(self, data):
        try:
            return self.success(**self.video_info(data['video_id']))
        except Exception as e:
            return self.fail(str(e))

    def frame(self, video_id, t):
        frames = VideoFrames(self.video_path(video_id), self.ffmpeg)
        try:
            return frames.frame(t)
        finally:
            frames.close()

    @staticmethod
    def jpeg(image, quality=85) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=quality)
        return buf.getvalue()

    def execute_frame(self, data):
        try:
            image = self.frame(data['video_id'], data.get('t', 0))
            return self.success(image=base64.b64encode(self.jpeg(image)).decode('ascii'),
                                width=image.width, height=image.height)
        except Exception as e:
            return self.fail(str(e))

    # -- OCR --

    @staticmethod
    def parse_box(box):
        """ box: "x,y,w,h", Liste oder dict -> (x, y, w, h) als int """
        if isinstance(box, str):
            box = json.loads(box) if box.strip().startswith(('[', '{')) else box.split(',')
        if isinstance(box, dict):
            box = [box['x'], box['y'], box['w'], box['h']]
        x, y, w, h = [int(round(float(v))) for v in box]
        if w <= 0 or h <= 0:
            raise ValueError('box needs a width and height > 0')
        return x, y, w, h

    @staticmethod
    def crop(image, box, invert=False):
        from PIL import ImageOps
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        image = image.crop((x, y, min(image.width, x + w), min(image.height, y + h)))
        if invert:
            image = ImageOps.invert(image.convert('RGB'))
        return image

    def available_runners(self) -> list:
        dispatcher = self.getParentClass(r.LazyDispatcher)
        names = list(self.ocr_runners)
        if dispatcher is not None:
            names = [n for n in names if dispatcher.support_type(n)]
            for runner in dispatcher.byTag(TAG.IMAGE_EXTRACTOR):
                if runner is not self and runner.type not in names:
                    names.append(runner.type)
        elif 'ocr' not in names:
            names.append('ocr')
        return names

    def execute_runners(self, data={}):
        return self.success(runners=self.available_runners())

    def get_ocr_runner(self, name):
        dispatcher = self.getParentClass(r.LazyDispatcher)
        runner = dispatcher.get_runner(name) if dispatcher is not None else None
        if runner is None and name == 'ocr':
            # ohne Dispatcher (z.B. direkt in Python) die leichte OCR lokal laden
            if name not in self.local_runners:
                from nxml.image import OCR
                self.local_runners[name] = OCR()
            runner = self.local_runners[name]
        if runner is None or runner is self:
            raise ValueError(f'OCR runner {name!r} not available')
        return runner

    def extract_value(self, text, number_only=True, decimals=None):
        """
            :param decimals: feste Anzahl Nachkommastellen der Anzeige, der Punkt wird dann aus der Position gesetzt,
                             weil die OCR kleine Dezimalpunkte oft nicht erkennt ("12 50" -> "12.50")
        """
        text = text.translate(self.DOTS)
        if not number_only:
            return text.strip()
        # Leerzeichen innerhalb einer Zahl ("12 .5", "12 50") entfernen, die OCR trennt Ziffern oft
        m = self.NUMBER.search(re.sub(r'(?<=[\d.,])\s+(?=[\d.,])', '', text))
        if m is None:
            return ''
        value = m.group(0).replace(',', '.').lstrip('+')
        if decimals is None or str(decimals).strip() == '':
            return value
        n = int(decimals)
        sign = '-' if value.startswith('-') else ''
        digits = re.sub(r'\D', '', value).zfill(n + 1)
        return sign + str(int(digits[:len(digits) - n])) + ('.' + digits[-n:] if n > 0 else '')

    def ocr(self, image, data) -> dict:
        """ OCR eines (zugeschnittenen) Bildes mit dem gewählten Runner """
        runner = self.get_ocr_runner(data.get('ocr') or 'ocr')
        params = {k: data[k] for k in self.OCR_PARAMS if k in data and data[k] != ''}
        if isinstance(runner, r.ImageExecutor):
            result = runner.executeImage(image, params)
        else:
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            result = runner.execute({'image': base64.b64encode(buf.getvalue()).decode('ascii'), **params})
        if not result.get('success', True):
            raise RuntimeError(result.get('message', 'OCR failed'))
        # eine Zeile, damit die Tabelle und der CSV-Export lesbar bleiben
        text = ' '.join(str(result.get('text', '')).translate(self.DOTS).split())
        confs = [line.get('confidence', 0) for line in result.get('lines', []) if 'confidence' in line]
        number_only = str(data.get('number_only', 1)) not in ('0', 'false', 'False', '')
        return {
            'text': text,
            'value': self.extract_value(text, number_only, data.get('decimals')),
            'confidence': round(sum(confs) / len(confs), 1) if confs else None
        }

    def prepare(self, data):
        return self.parse_box(data['box']), str(data.get('invert', 0)) not in ('0', 'false', 'False', '')

    def execute_ocr_frame(self, data):
        try:
            box, invert = self.prepare(data)
            image = self.crop(self.frame(data['video_id'], data.get('t', 0)), box, invert)
            return self.success(**self.ocr(image, data), crop=base64.b64encode(self.jpeg(image)).decode('ascii'))
        except Exception as e:
            self.error(str(e))
            return self.fail(str(e))

    # -- Auswertung --

    @staticmethod
    def format_time(t):
        h, rest = divmod(float(t), 3600)
        m, s = divmod(rest, 60)
        return f'{int(h):02d}:{int(m):02d}:{s:06.3f}'.rstrip('0').rstrip('.')

    def times(self, duration, interval, start=0.0, end=0.0, fps=0.0):
        if interval <= 0:
            raise ValueError('interval must be > 0')
        # das letzte Bild beginnt eine Bilddauer vor dem Ende
        last = max(0.0, duration - 1 / fps) if fps > 0 else duration
        end = last if end <= 0 else min(end, last)
        res = []
        t = max(0.0, start)
        while t <= end + 1e-6:
            res.append(round(t, 3))
            t += interval
        return res

    def execute_process(self, data):
        try:
            video_id = data['video_id']
            box, invert = self.prepare(data)
            info = self.video_info(video_id)
            times = self.times(info['duration'], float(data.get('interval', 30)),
                               float(data.get('start', 0) or 0), float(data.get('end', 0) or 0), info['fps'])
            self.get_ocr_runner(data.get('ocr') or 'ocr')
        except Exception as e:
            return self.fail(str(e))
        job_id = u.guid()
        job = {
            'job_id': job_id, 'video_id': video_id, 'status': 'running', 'times': times, 'rows': [],
            'box': list(box), 'invert': invert, 'interval': float(data.get('interval', 30)),
            'ocr': data.get('ocr') or 'ocr', 'started': time.time(), 'message': '',
            'params': {k: data[k] for k in ['number_only', 'decimals', *self.OCR_PARAMS] if k in data}
        }
        self.jobs[job_id] = job
        if str(data.get('async', 1)) in ('0', 'false', 'False'):
            self.run_job(job)
            return self.success(job_id=job_id, count=len(times), rows=job['rows'], status=job['status'])
        Thread(target=self.run_job, args=(job,), daemon=True).start()
        return self.success(job_id=job_id, count=len(times))

    def job_dir(self, job):
        folder = os.path.join(self.video_dir(job['video_id']), job['job_id'])
        os.makedirs(folder, exist_ok=True)
        return folder

    def run_job(self, job):
        frames = VideoFrames(self.video_path(job['video_id']), self.ffmpeg)
        folder = self.job_dir(job)
        params = {**job['params'], 'ocr': job['ocr']}
        try:
            for i, t in enumerate(job['times']):
                if job['status'] != 'running':
                    break
                row = {'i': i, 't': t, 'time': self.format_time(t)}
                try:
                    image = self.crop(frames.frame(t), job['box'], job['invert'])
                    image.save(os.path.join(folder, f'{i}.jpg'), quality=85)
                    row.update(self.ocr(image, params))
                except Exception as e:
                    row.update({'text': '', 'value': '', 'confidence': None, 'error': str(e)})
                job['rows'].append(row)
            if job['status'] == 'running':
                job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            job['message'] = str(e)
            self.error(str(e))
        finally:
            frames.close()
            job['duration'] = round(time.time() - job['started'], 1)
            with open(os.path.join(folder, 'result.json'), 'w') as f:
                json.dump(self.job_result(job), f, indent=2)
        self.info(f"Job {job['job_id']}: {job['status']}, {len(job['rows'])} rows")

    @staticmethod
    def job_result(job):
        keys = ['job_id', 'video_id', 'status', 'box', 'interval', 'ocr', 'invert', 'params', 'rows', 'message']
        return {k: job.get(k) for k in keys}

    def get_job(self, job_id):
        if job_id in self.jobs:
            return self.jobs[job_id]
        # Ergebnis eines früheren Laufs
        if re.fullmatch(r'[\w-]+', str(job_id)) and os.path.isdir(self.data_dir):
            for video_id in os.listdir(self.data_dir):
                path = os.path.join(self.data_dir, video_id, job_id, 'result.json')
                if os.path.isfile(path):
                    with open(path) as f:
                        return json.load(f)
        raise ValueError(f'Unknown job_id {job_id!r}')

    def execute_status(self, data):
        try:
            job = self.get_job(data['job_id'])
        except ValueError as e:
            return self.fail(str(e))
        offset = int(data.get('offset', 0))
        rows = job['rows']
        return self.success(status=job['status'], done=len(rows), count=len(job.get('times', rows)),
                            rows=rows[offset:], message=job.get('message', ''))

    def execute_cancel(self, data):
        job = self.jobs.get(data['job_id'])
        if job is None or job['status'] != 'running':
            return self.fail('Job not running')
        job['status'] = 'canceled'
        return self.success('Abgebrochen')

    EXPORT_COLUMNS = ['t', 'time', 'value', 'text', 'confidence', 'error']

    def export(self, job, format='csv', sep=','):
        if format == 'json':
            return json.dumps(self.job_result(job), indent=2)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, self.EXPORT_COLUMNS, delimiter=sep, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in job['rows']:
            writer.writerow(row)
        return buf.getvalue()

    def execute_export(self, data):
        try:
            job = self.get_job(data['job_id'])
        except ValueError as e:
            return self.fail(str(e))
        return self.success(content=self.export(job, data.get('format', 'csv'), data.get('sep', ',')))

    # -- Web --

    def route(self, name):
        return f'/nxml/{self.type}/{name}'

    def setupRestApp(self, app):
        from flask import request, Response

        def upload():
            f = request.files.get('file')
            if f is None:
                return self.fail('No file')
            return self.store_upload(f.filename, f.save)

        def frame():
            try:
                image = self.frame(request.args['video_id'], request.args.get('t', 0))
                return Response(self.jpeg(image), mimetype='image/jpeg')
            except Exception as e:
                return Response(str(e), status=404)

        def crop():
            try:
                job = self.get_job(request.args['job_id'])
                path = os.path.join(self.video_dir(job['video_id']), job['job_id'], f"{int(request.args['i'])}.jpg")
                with open(path, 'rb') as f:
                    return Response(f.read(), mimetype='image/jpeg')
            except (ValueError, KeyError, OSError) as e:
                return Response(str(e), status=404)

        def export():
            try:
                job = self.get_job(request.args['job_id'])
            except ValueError as e:
                return Response(str(e), status=404)
            fmt = 'json' if request.args.get('format') == 'json' else 'csv'
            content = self.export(job, fmt, ';' if request.args.get('sep') == ';' else ',')
            name = f"video_ocr_{job['job_id']}.{fmt}"
            return Response(content, mimetype='application/json' if fmt == 'json' else 'text/csv',
                            headers={'Content-Disposition': f'attachment; filename="{name}"'})

        for name, func, methods in [('upload', upload, ['POST']), ('frame', frame, ['GET']),
                                    ('crop', crop, ['GET']), ('export', export, ['GET'])]:
            app.add_url_rule(self.route(name), f'{self.type}_{name}', view_func=func, methods=methods)

    def part_index(self, p: b.Page, params={}):
        cfg = {
            'type': self.type,
            'routes': {name: self.route(name) for name in ['upload', 'frame', 'crop', 'export']},
            'runners': self.available_runners()
        }
        p(f'<style>{VIDEO_OCR_CSS}</style>')
        p(VIDEO_OCR_HTML)
        p(f'<script>const VIDEO_OCR = {json.dumps(cfg).replace('</', '<\\/')};\n{VIDEO_OCR_JS}</script>')


VIDEO_OCR_CSS = """
.vo { max-width: 1100px; }
.vo-steps { display: flex; gap: 6px; margin: 8px 0 14px; flex-wrap: wrap; }
.vo-steps button { padding: 6px 12px; border: 1px solid #999; border-radius: 16px; background: transparent;
  color: inherit; cursor: pointer; opacity: .6; }
.vo-steps button.active { opacity: 1; font-weight: bold; border-width: 2px; }
.vo-steps button:disabled { cursor: default; opacity: .3; }
.vo-step { display: none; }
.vo-step.active { display: block; }
.vo-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
.vo-row label { min-width: 120px; }
.vo input[type=number] { width: 90px; }
.vo-progress { height: 10px; background: #8884; border-radius: 5px; overflow: hidden; max-width: 500px; }
.vo-progress div { height: 100%; width: 0; background: #2a7; transition: width .2s; }
.vo-stage { position: relative; display: inline-block; max-width: 100%; user-select: none; touch-action: none;
  line-height: 0; cursor: crosshair; }
.vo-stage img { max-width: 100%; max-height: 70vh; display: block; }
.vo-box { position: absolute; border: 2px solid #f33; background: #f332; cursor: move; box-sizing: border-box; }
.vo-box .vo-handle { position: absolute; width: 14px; height: 14px; right: -8px; bottom: -8px; background: #f33;
  border: 2px solid #fff; border-radius: 50%; cursor: nwse-resize; }
.vo-crop { max-height: 80px; border: 1px solid #999; image-rendering: pixelated; }
.vo-table { border-collapse: collapse; margin-top: 10px; }
.vo-table td, .vo-table th { border: 1px solid #8886; padding: 3px 8px; text-align: left; }
.vo-table img { max-height: 32px; display: block; }
.vo-table td.vo-err { color: #c33; }
.vo-msg { margin: 6px 0; min-height: 1.2em; }
.vo-msg.error { color: #c33; }
"""

VIDEO_OCR_HTML = """
<div class="vo">
  <div class="vo-steps">
    <button data-step="1" class="active">1. Video</button>
    <button data-step="2" disabled>2. Bereich wählen</button>
    <button data-step="3" disabled>3. Auswertung</button>
  </div>

  <section class="vo-step active" id="vo-step1">
    <h2>Video hochladen</h2>
    <div class="vo-row"><input type="file" id="vo-file" accept="video/*" />
      <button id="vo-upload">Hochladen</button></div>
    <div class="vo-row"><label for="vo-path">oder Pfad auf dem Server</label>
      <input type="text" id="vo-path" style="width: 360px" placeholder="/data/messung.mp4" />
      <button id="vo-use-path">Übernehmen</button></div>
    <div class="vo-progress"><div id="vo-upload-progress"></div></div>
    <div class="vo-msg" id="vo-msg1"></div>
  </section>

  <section class="vo-step" id="vo-step2">
    <h2>Bereich für die OCR wählen</h2>
    <div>Box verschieben, an der Ecke unten rechts in der Größe ändern oder außerhalb neu aufziehen.</div>
    <div class="vo-row"><label for="vo-t">Zeitpunkt (s)</label>
      <input type="range" id="vo-t-range" min="0" max="0" step="0.1" value="0" style="width: 360px" />
      <input type="number" id="vo-t" min="0" step="0.1" value="0" />
      <span id="vo-info"></span></div>
    <div class="vo-stage" id="vo-stage"><img id="vo-frame" alt="Frame" draggable="false" />
      <div class="vo-box" id="vo-box"><div class="vo-handle"></div></div></div>
    <div class="vo-row">
      <label>Box (Pixel)</label>
      x <input type="number" id="vo-bx" min="0" /> y <input type="number" id="vo-by" min="0" />
      w <input type="number" id="vo-bw" min="1" /> h <input type="number" id="vo-bh" min="1" />
    </div>
    <div class="vo-row">
      <button id="vo-test">OCR testen</button>
      <img class="vo-crop" id="vo-test-crop" style="display: none" alt="" />
      <span id="vo-test-result"></span>
    </div>
    <div class="vo-row"><button id="vo-back1">Zurück</button><button id="vo-next2">Weiter</button></div>
    <div class="vo-msg" id="vo-msg2"></div>
  </section>

  <section class="vo-step" id="vo-step3">
    <h2>Auswertung</h2>
    <div class="vo-row"><label for="vo-interval">Abstand (s)</label>
      <input type="number" id="vo-interval" min="0.1" step="any" value="30" />
      <button data-interval="1">1 s</button><button data-interval="5">5 s</button>
      <button data-interval="10">10 s</button><button data-interval="30">30 s</button>
      <button data-interval="60">1 min</button><button data-interval="300">5 min</button>
      <span id="vo-count"></span></div>
    <div class="vo-row"><label for="vo-start">Von / bis (s)</label>
      <input type="number" id="vo-start" min="0" step="any" value="0" />
      <input type="number" id="vo-end" min="0" step="any" value="0" /> <span>0 = bis zum Ende</span></div>
    <div class="vo-row"><label for="vo-ocr">OCR-Runner</label><select id="vo-ocr"></select></div>
    <div class="vo-row"><label>Optionen</label>
      <label><input type="checkbox" id="vo-number" checked /> nur Zahlen</label>
      <label for="vo-decimals" style="min-width: 0">Nachkommastellen</label>
      <select id="vo-decimals" title="Fest, falls die OCR den Dezimalpunkt nicht erkennt: 2 macht aus 1250 den Wert 12.50">
        <option value="">wie erkannt</option><option value="0">0</option><option value="1">1</option>
        <option value="2">2</option><option value="3">3</option></select>
      <label><input type="checkbox" id="vo-invert" /> invertieren (helle Schrift)</label>
      <label for="vo-psm" style="min-width: 0">Tesseract-Layout</label>
      <select id="vo-psm"><option value="7">eine Zeile</option><option value="8">ein Wort</option>
        <option value="6">Textblock</option><option value="3">automatisch</option></select></div>
    <div class="vo-row"><button id="vo-back2">Zurück</button><button id="vo-start-btn">Auswertung starten</button>
      <button id="vo-cancel" disabled>Abbrechen</button></div>
    <div class="vo-progress"><div id="vo-job-progress"></div></div>
    <div class="vo-msg" id="vo-msg3"></div>
    <div class="vo-row" id="vo-exports" style="display: none">Export:
      <a id="vo-csv" href="#">CSV</a> <a id="vo-csv-de" href="#">CSV (Excel, ;)</a> <a id="vo-json" href="#">JSON</a>
      <button id="vo-copy">Werte kopieren</button></div>
    <table class="vo-table" id="vo-table" style="display: none">
      <thead><tr><th>#</th><th>Zeit</th><th>Sekunden</th><th>Wert</th><th>Text</th><th>Konfidenz</th>
        <th>Ausschnitt</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>
  <pre id="result" style="display: none"></pre>
</div>
"""

VIDEO_OCR_JS = r"""
(function() {
  const cfg = VIDEO_OCR;
  const $ = (id) => document.getElementById(id);
  const state = { video: null, box: null, job: null, rows: [] };
  const url = (path, params) => {
    const base = location.protocol + '//' + location.host;
    return base + path + (params ? '?' + new URLSearchParams(params) : '');
  };

  function job(data) {
    const body = new URLSearchParams({ type: cfg.type, ...data });
    return fetch(url('/'), { method: 'POST', body: body }).then((r) => r.json());
  }
  function msg(step, text, error) {
    const el = $('vo-msg' + step);
    el.textContent = text || '';
    el.classList.toggle('error', !!error);
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  }

  // -- Schritte --
  const stepButtons = document.querySelectorAll('.vo-steps button');
  function showStep(n) {
    document.querySelectorAll('.vo-step').forEach((s) => s.classList.toggle('active', s.id === 'vo-step' + n));
    stepButtons.forEach((b) => b.classList.toggle('active', b.dataset.step == n));
    if (n === 3) updateCount();
  }
  function enableStep(n) { stepButtons.forEach((b) => { if (b.dataset.step <= n) b.disabled = false; }); }
  stepButtons.forEach((b) => b.addEventListener('click', () => showStep(+b.dataset.step)));

  // -- 1. Upload --
  function videoLoaded(r) {
    if (!r.success) { msg(1, r.message || 'Fehler', true); return; }
    state.video = r;
    msg(1, 'Video geladen: ' + r.duration + ' s, ' + r.fps + ' FPS, ' + r.width + 'x' + r.height);
    $('vo-info').textContent = 'Dauer ' + r.duration + ' s, ' + r.width + 'x' + r.height;
    $('vo-t-range').max = r.duration;
    $('vo-t').max = r.duration;
    $('vo-end').value = 0;
    state.box = null;
    enableStep(3);
    showStep(2);
    loadFrame(0);
  }
  $('vo-upload').addEventListener('click', () => {
    const file = $('vo-file').files[0];
    if (!file) { msg(1, 'Bitte ein Video wählen', true); return; }
    const form = new FormData();
    form.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url(cfg.routes.upload));
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) $('vo-upload-progress').style.width = (100 * e.loaded / e.total) + '%';
    };
    xhr.onload = () => {
      try { videoLoaded(JSON.parse(xhr.responseText)); } catch (e) { msg(1, xhr.responseText, true); }
    };
    xhr.onerror = () => msg(1, 'Upload fehlgeschlagen', true);
    msg(1, 'Lade hoch und lese Video ...');
    xhr.send(form);
  });
  $('vo-use-path').addEventListener('click', () => {
    msg(1, 'Lese Video ...');
    job({ op: 'upload', path: $('vo-path').value }).then(videoLoaded);
  });

  // -- 2. Bereich --
  const img = $('vo-frame'), stage = $('vo-stage'), boxEl = $('vo-box');
  let frameTimer = null;
  function loadFrame(t) {
    clearTimeout(frameTimer);
    frameTimer = setTimeout(() => {
      img.src = url(cfg.routes.frame, { video_id: state.video.video_id, t: t });
    }, 120);
  }
  img.addEventListener('load', () => {
    if (!state.box) {
      const w = img.naturalWidth, h = img.naturalHeight;
      state.box = { x: Math.round(w * 0.35), y: Math.round(h * 0.4), w: Math.round(w * 0.3), h: Math.round(h * 0.2) };
    }
    drawBox();
  });
  window.addEventListener('resize', drawBox);
  function scale() { return img.clientWidth / (img.naturalWidth || 1); }
  function clampBox() {
    const b = state.box, W = img.naturalWidth, H = img.naturalHeight;
    b.w = Math.max(4, Math.min(Math.round(b.w), W));
    b.h = Math.max(4, Math.min(Math.round(b.h), H));
    b.x = Math.max(0, Math.min(Math.round(b.x), W - b.w));
    b.y = Math.max(0, Math.min(Math.round(b.y), H - b.h));
  }
  function drawBox() {
    if (!state.box || !img.naturalWidth) return;
    clampBox();
    const s = scale(), b = state.box;
    Object.assign(boxEl.style, { left: b.x * s + 'px', top: b.y * s + 'px', width: b.w * s + 'px', height: b.h * s + 'px' });
    $('vo-bx').value = b.x; $('vo-by').value = b.y; $('vo-bw').value = b.w; $('vo-bh').value = b.h;
  }
  ['vo-bx', 'vo-by', 'vo-bw', 'vo-bh'].forEach((id) => $(id).addEventListener('change', () => {
    state.box = { x: +$('vo-bx').value, y: +$('vo-by').value, w: +$('vo-bw').value, h: +$('vo-bh').value };
    drawBox();
  }));
  function setTime(t) {
    $('vo-t').value = t; $('vo-t-range').value = t;
    loadFrame(t);
  }
  $('vo-t-range').addEventListener('input', (e) => setTime(e.target.value));
  $('vo-t').addEventListener('change', (e) => setTime(e.target.value));

  // Ziehen: Box verschieben, Ecke = Größe ändern, außerhalb = neue Box aufziehen
  let drag = null;
  function point(e) {
    const r = stage.getBoundingClientRect(), s = scale();
    return { x: (e.clientX - r.left) / s, y: (e.clientY - r.top) / s };
  }
  stage.addEventListener('pointerdown', (e) => {
    if (!state.box) return;
    e.preventDefault();
    const p = point(e), b = { ...state.box };
    if (e.target.classList.contains('vo-handle')) drag = { mode: 'resize', p, b };
    else if (e.target === boxEl) drag = { mode: 'move', p, b };
    else { drag = { mode: 'new', p, b }; state.box = { x: p.x, y: p.y, w: 4, h: 4 }; drawBox(); }
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const p = point(e), dx = p.x - drag.p.x, dy = p.y - drag.p.y, b = drag.b;
    if (drag.mode === 'move') state.box = { ...b, x: b.x + dx, y: b.y + dy };
    else if (drag.mode === 'resize') state.box = { ...b, w: b.w + dx, h: b.h + dy };
    else state.box = { x: Math.min(p.x, drag.p.x), y: Math.min(p.y, drag.p.y), w: Math.abs(dx), h: Math.abs(dy) };
    drawBox();
  });
  stage.addEventListener('pointerup', () => { drag = null; });

  function boxParam() { const b = state.box; return [b.x, b.y, b.w, b.h].join(','); }
  function ocrParams() {
    return {
      box: boxParam(), ocr: $('vo-ocr').value, number_only: $('vo-number').checked ? 1 : 0, decimals: $('vo-decimals').value,
      invert: $('vo-invert').checked ? 1 : 0, psm: $('vo-psm').value
    };
  }
  $('vo-test').addEventListener('click', () => {
    $('vo-test-result').textContent = 'OCR läuft ...';
    job({ op: 'ocr_frame', video_id: state.video.video_id, t: $('vo-t').value, ...ocrParams() }).then((r) => {
      if (!r.success) { $('vo-test-result').textContent = 'Fehler: ' + r.message; return; }
      $('vo-test-crop').src = 'data:image/jpeg;base64,' + r.crop;
      $('vo-test-crop').style.display = '';
      $('vo-test-result').innerHTML = 'Wert: <b>' + esc(r.value || '-') + '</b> &nbsp; Text: ' + esc(r.text || '-') +
        (r.confidence != null ? ' &nbsp; (' + r.confidence + ' %)' : '');
    });
  });
  $('vo-back1').addEventListener('click', () => showStep(1));
  $('vo-next2').addEventListener('click', () => showStep(3));

  // -- 3. Auswertung --
  const select = $('vo-ocr');
  cfg.runners.forEach((name) => select.add(new Option(name, name)));
  if (!cfg.runners.length) select.add(new Option('- kein OCR-Runner geladen -', ''));
  function updateCount() {
    if (!state.video) return;
    const iv = +$('vo-interval').value, start = +$('vo-start').value || 0;
    const v = state.video, last = v.fps > 0 ? Math.max(0, v.duration - 1 / v.fps) : v.duration;
    const end = +$('vo-end').value > 0 ? Math.min(+$('vo-end').value, last) : last;
    $('vo-count').textContent = iv > 0 ? '= ' + (Math.floor((end - start) / iv + 1e-6) + 1) + ' Bilder' : '';
  }
  ['vo-interval', 'vo-start', 'vo-end'].forEach((id) => $(id).addEventListener('input', updateCount));
  document.querySelectorAll('[data-interval]').forEach((b) => b.addEventListener('click', () => {
    $('vo-interval').value = b.dataset.interval;
    updateCount();
  }));
  $('vo-back2').addEventListener('click', () => showStep(2));

  const tbody = document.querySelector('#vo-table tbody');
  function addRows(rows) {
    rows.forEach((row) => {
      state.rows.push(row);
      const tr = document.createElement('tr');
      const crop = url(cfg.routes.crop, { job_id: state.job, i: row.i });
      tr.innerHTML = '<td>' + (row.i + 1) + '</td><td>' + esc(row.time) + '</td><td>' + row.t + '</td>' +
        '<td><b>' + esc(row.value) + '</b></td>' +
        (row.error ? '<td class="vo-err">' + esc(row.error) + '</td>' : '<td>' + esc(row.text) + '</td>') +
        '<td>' + (row.confidence != null ? row.confidence : '') + '</td>' +
        '<td><a href="' + crop + '" target="_blank"><img loading="lazy" src="' + crop + '" alt="" /></a></td>';
      tbody.appendChild(tr);
    });
  }
  function poll() {
    job({ op: 'status', job_id: state.job, offset: state.rows.length }).then((r) => {
      if (!r.success) { msg(3, r.message, true); return; }
      addRows(r.rows);
      $('vo-job-progress').style.width = (r.count ? 100 * r.done / r.count : 100) + '%';
      msg(3, r.done + ' / ' + r.count + ' ' + ({ running: 'läuft ...', done: 'fertig', canceled: 'abgebrochen',
        error: 'Fehler: ' + r.message })[r.status], r.status === 'error');
      if (r.status === 'running') setTimeout(poll, 1000);
      else { $('vo-start-btn').disabled = false; $('vo-cancel').disabled = true; }
    }).catch((e) => { msg(3, String(e), true); setTimeout(poll, 3000); });
  }
  $('vo-start-btn').addEventListener('click', () => {
    if (!state.box) { msg(3, 'Bitte zuerst einen Bereich wählen', true); return; }
    const data = { op: 'process', video_id: state.video.video_id, interval: $('vo-interval').value,
      start: $('vo-start').value, end: $('vo-end').value, ...ocrParams() };
    msg(3, 'Starte ...');
    job(data).then((r) => {
      if (!r.success) { msg(3, r.message, true); return; }
      state.job = r.job_id; state.rows = []; tbody.innerHTML = '';
      $('vo-table').style.display = ''; $('vo-exports').style.display = '';
      $('vo-csv').href = url(cfg.routes.export, { job_id: r.job_id, format: 'csv' });
      $('vo-csv-de').href = url(cfg.routes.export, { job_id: r.job_id, format: 'csv', sep: ';' });
      $('vo-json').href = url(cfg.routes.export, { job_id: r.job_id, format: 'json' });
      $('vo-start-btn').disabled = true; $('vo-cancel').disabled = false;
      poll();
    });
  });
  $('vo-cancel').addEventListener('click', () => job({ op: 'cancel', job_id: state.job }));
  $('vo-copy').addEventListener('click', () => {
    const text = state.rows.map((r) => r.time + '\t' + r.value).join('\n');
    navigator.clipboard.writeText(text).then(() => msg(3, state.rows.length + ' Zeilen kopiert'));
  });
})();
"""
