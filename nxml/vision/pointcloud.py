#
# Project: nxml
#
# 3D-Ansicht von Punktwolken (PLY, XYZ, SLAM-Karten) im Browser mit three.js
#
"""
PointCloud-Runner (type ``pointcloud``)

Zeigt Punktwolken im Browser an (three.js aus /static/js/three). Quellen:
 - PLY (ascii, binary_little_endian, binary_big_endian), XYZ/TXT/CSV (x y z [r g b]) als Upload oder Pfad
 - Karte eines SLAM-Laufs (SlamRunner, type slam), auch live während der Lauf noch läuft

UI: /pysys/dispatcher?type=pointcloud, direkt für einen SLAM-Lauf ...&slam=<session>
Arbeitsverzeichnis: work_dir, Standard <tmp>/nxml_pointcloud (Linux: /tmp/nxml_pointcloud)
"""
import base64
import json
import os
import re
import struct
import tempfile
import time

from nwebclient import runner as r
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import dev as d

try:
    import numpy as np
except ImportError:
    np = None


ID_RE = re.compile(r'[\w-]+')

PLY_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'
}

COLOR_NAMES = [('red', 'green', 'blue'), ('r', 'g', 'b'), ('diffuse_red', 'diffuse_green', 'diffuse_blue')]


def _colors(fields, get):
    """ RGB als uint8 (N,3) aus den Feldern, Gleitkommafarben 0..1 werden skaliert """
    for names in COLOR_NAMES:
        if all(n in fields for n in names):
            rgb = np.stack([np.asarray(get(n), np.float64) for n in names], axis=1)
            if rgb.size and rgb.max() <= 1.0 and np.issubdtype(np.asarray(get(names[0])).dtype, np.floating):
                rgb = rgb * 255
            return np.clip(rgb, 0, 255).astype(np.uint8)
    return None


def read_ply(path):
    """
    Liest die Vertices einer PLY-Datei
    :return: (xyz float32 (N,3), rgb uint8 (N,3) oder None)
    """
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Keine PLY-Datei')
        fmt = None
        elements = []  # [name, count, [(prop, type) | (prop, 'list', ...)]]
        while True:
            line = f.readline()
            if not line:
                raise ValueError('PLY-Header ohne end_header')
            parts = line.decode('ascii', 'replace').split()
            if not parts or parts[0] in ('comment', 'obj_info'):
                continue
            if parts[0] == 'end_header':
                break
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'element':
                elements.append([parts[1], int(parts[2]), []])
            elif parts[0] == 'property' and elements:
                if parts[1] == 'list':
                    elements[-1][2].append((parts[4], 'list'))
                else:
                    elements[-1][2].append((parts[2], parts[1]))
        if not elements or elements[0][0] != 'vertex':
            raise ValueError('PLY: vertex muss das erste Element sein')
        _, n, props = elements[0]
        if any(t == 'list' for _, t in props):
            raise ValueError('PLY: Listen-Eigenschaften bei vertex werden nicht unterstützt')
        names = [p for p, _ in props]
        if not all(k in names for k in 'xyz'):
            raise ValueError('PLY: x, y, z fehlen')
        if fmt == 'ascii':
            data = np.loadtxt(f, max_rows=n, ndmin=2) if n else np.zeros((0, len(names)))
            col = {name: data[:, i] for i, name in enumerate(names)}
            get = col.__getitem__
            for name, t in props:  # Ganzzahlfarben bleiben ganzzahlig erkennbar
                if PLY_TYPES.get(t, 'f4')[0] in 'iu':
                    col[name] = col[name].astype(np.int64)
        elif fmt in ('binary_little_endian', 'binary_big_endian'):
            end = '<' if fmt == 'binary_little_endian' else '>'
            dtype = np.dtype([(name, end + PLY_TYPES[t]) for name, t in props])
            data = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
            get = data.__getitem__
        else:
            raise ValueError(f'PLY-Format {fmt!r} wird nicht unterstützt')
        xyz = np.stack([np.asarray(get(k), np.float32) for k in 'xyz'], axis=1)
        return xyz, _colors(names, get)


def read_xyz(path):
    """ Textdatei mit x y z [r g b] je Zeile (Leerzeichen, Komma oder Semikolon) """
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            vals = re.split(r'[\s,;]+', line.strip())
            try:
                rows.append([float(v) for v in vals[:6]])
            except ValueError:
                continue  # Kopfzeile o.ä.
    rows = [row for row in rows if len(row) >= 3]
    if not rows:
        raise ValueError('Keine Punkte gefunden')
    width = min(len(row) for row in rows)
    data = np.array([row[:width] for row in rows], np.float64)
    xyz = data[:, :3].astype(np.float32)
    rgb = None
    if width >= 6:
        c = data[:, 3:6]
        rgb = np.clip(c * 255 if c.max() <= 1.0 else c, 0, 255).astype(np.uint8)
    return xyz, rgb


def read_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        return read_ply(path)
    if ext == '.npz':
        z = np.load(path)
        return z['xyz'].astype(np.float32), (z['rgb'].astype(np.uint8) if 'rgb' in z else None)
    return read_xyz(path)


def pack_cloud(xyz, rgb=None):
    """
    Binärformat für den Browser (little endian): b'NXPC', uint32 n, uint32 flags (1 = Farben),
    float32 xyz * n, danach uint8 rgb * n
    """
    xyz = np.ascontiguousarray(xyz, '<f4')
    head = b'NXPC' + struct.pack('<II', len(xyz), 1 if rgb is not None else 0)
    body = xyz.tobytes()
    if rgb is not None:
        body += np.ascontiguousarray(rgb, np.uint8).tobytes()
    return head + body


class PointCloudRunner(r.BaseJobExecutor):
    """
    Zeigt Punktwolken im Browser an (three.js): PLY, XYZ/TXT/CSV oder die Karte eines SLAM-Laufs.

    UI: Wolke hochladen, Pfad auf dem Server angeben oder SLAM-Lauf wählen; Drehen/Verschieben/Zoomen mit der
    Maus, Farben (RGB, Höhe, einfarbig), Punktgröße, Höhenfilter, Kamerabahn, OpenCV-Achsen umdrehen.
    Direkt für einen SLAM-Lauf: /pysys/dispatcher?type=pointcloud&slam=<session>

    three.js: three_url (Standard /static/js/three/three.module.min.js) und orbit_url (OrbitControls).
    """

    TAGS = ['vision', '3d']

    MODULES = ['numpy']

    type = 'pointcloud'

    def __init__(self, type='pointcloud', work_dir=None, max_points=1000000, slam_type='slam',
                 three_url='/static/js/three/three.module.min.js',
                 orbit_url='/static/js/three/controls/OrbitControls.js'):
        """
            :param work_dir: Arbeitsverzeichnis, Standard <tmp>/nxml_pointcloud, wird bei Bedarf angelegt
            :param max_points: höchstens so viele Punkte an den Browser schicken (zufällige Auswahl)
            :param slam_type: type des SlamRunners für SLAM-Karten
            :param three_url: three.js als ES-Modul
            :param orbit_url: OrbitControls als ES-Modul (importiert three relativ)
        """
        super().__init__(type)
        self.work_dir = work_dir or os.path.join(tempfile.gettempdir(), 'nxml_pointcloud')
        self.max_points = int(max_points)
        self.slam_type = slam_type
        self.three_url = three_url
        self.orbit_url = orbit_url
        self.define_sig(d.PStr('op', 'upload'), d.Param('file_data'), d.PStr('name', 'cloud.ply'))\
            .set_description("Punktwolke (Base64, PLY/XYZ/TXT/CSV) speichern").r(success='bool', cloud='str', points='int')
        self.define_sig(d.PStr('op', 'load'), d.Param('path'))\
            .set_description("Punktwolke aus einem Pfad auf dem Server übernehmen").r(success='bool', cloud='str', points='int')
        self.define_sig(d.PStr('op', 'clouds')).set_description("Gespeicherte Punktwolken").r(clouds='list')
        self.define_sig(d.PStr('op', 'info'), d.PStr('cloud', ''), d.PStr('slam', ''))\
            .set_description("Infos zu einer Wolke oder einem SLAM-Lauf (Anzahl, Grenzen, Kamerabahn)")\
            .r(success='bool', points='int', trajectory='list')
        self.define_sig(d.PStr('op', 'slam_sessions')).set_description("SLAM-Läufe").r(sessions='list')
        self.define_sig(d.PStr('op', 'delete'), d.Param('cloud')).set_description("Wolke löschen").r()

    # -- Wolken --

    def cloud_dir(self, cloud_id):
        if not ID_RE.fullmatch(str(cloud_id)):
            raise ValueError(f'Ungültige cloud {cloud_id!r}')
        return os.path.join(self.work_dir, 'clouds', cloud_id)

    def store(self, name, xyz, rgb, source=''):
        cloud_id = time.strftime('%Y%m%d-%H%M%S-') + u.guid()[:6]
        folder = self.cloud_dir(cloud_id)
        os.makedirs(folder, exist_ok=True)
        if rgb is None:
            np.savez(os.path.join(folder, 'cloud.npz'), xyz=xyz)
        else:
            np.savez(os.path.join(folder, 'cloud.npz'), xyz=xyz, rgb=rgb)
        meta = {'cloud': cloud_id, 'name': os.path.basename(name or 'cloud'), 'source': source,
                'points': int(len(xyz)), 'colors': rgb is not None, 'created': time.time()}
        with open(os.path.join(folder, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        return meta

    def load_file(self, path, name=None, source=''):
        if np is None:
            return self.fail('numpy fehlt')
        try:
            xyz, rgb = read_cloud(path)
        except Exception as e:
            return self.fail(f'Punktwolke nicht lesbar: {e}')
        if not len(xyz):
            return self.fail('Keine Punkte')
        meta = self.store(name or path, xyz, rgb, source)
        return self.success(**meta)

    def execute_upload(self, data):
        name = os.path.basename(data.get('name', 'cloud.ply'))
        ext = os.path.splitext(name)[1].lower()
        tmp = os.path.join(self.ensure_dir('uploads'), u.guid() + (ext if re.fullmatch(r'\.\w{1,5}', ext) else '.ply'))
        with open(tmp, 'wb') as f:
            f.write(base64.b64decode(data['file_data']))
        try:
            return self.load_file(tmp, name, 'upload')
        finally:
            os.remove(tmp)

    def execute_load(self, data):
        path = data.get('path', '')
        if not os.path.isfile(path):
            return self.fail(f'Datei nicht gefunden: {path}')
        return self.load_file(path, path, path)

    def execute_clouds(self, data={}):
        res = []
        folder = os.path.join(self.work_dir, 'clouds')
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                path = os.path.join(folder, name, 'meta.json')
                if os.path.isfile(path):
                    try:
                        with open(path) as f:
                            res.append(json.load(f))
                    except (OSError, ValueError):
                        pass
        return self.success(clouds=sorted(res, key=lambda m: m.get('created', 0), reverse=True))

    def execute_delete(self, data):
        import shutil
        try:
            folder = self.cloud_dir(data['cloud'])
        except ValueError as e:
            return self.fail(str(e))
        if not os.path.isdir(folder):
            return self.fail('Unbekannte Wolke')
        shutil.rmtree(folder)
        return self.success('Gelöscht')

    def ensure_dir(self, *parts):
        path = os.path.join(self.work_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    # -- SLAM --

    def slam_runner(self):
        dispatcher = self.getParentClass(r.LazyDispatcher)
        if dispatcher is not None and dispatcher.support_type(self.slam_type):
            return dispatcher.get_runner(self.slam_type)
        return None

    def slam_data(self, session_id):
        """ :return: (xyz, rgb, trajectory [[x,y,z]], info dict) eines SLAM-Laufs """
        if not ID_RE.fullmatch(str(session_id)):
            raise ValueError(f'Ungültige session {session_id!r}')
        slam = self.slam_runner()
        if slam is not None:
            s = slam.get_session(session_id)
            with s.lock:
                points = list(s.points)
                poses = list(s.poses)
            info = {'status': s.status, 'source': s.source, 'tracking': s.tracking}
            traj = [[p['x'], p['y'], p['z']] for p in poses]
            pts = np.array(points, np.float64).reshape(-1, 6)
            return pts[:, :3].astype(np.float32), pts[:, 3:6].astype(np.uint8), traj, info
        # ohne Dispatcher: Ergebnisse im Standard-Arbeitsverzeichnis des SlamRunners
        folder = os.path.join(tempfile.gettempdir(), 'nxml_slam', 'sessions', session_id)
        path = os.path.join(folder, 'points.ply')
        if not os.path.isfile(path):
            raise ValueError(f'SLAM-Lauf {session_id} nicht gefunden')
        xyz, rgb = read_ply(path)
        traj = []
        tpath = os.path.join(folder, 'trajectory.csv')
        if os.path.isfile(tpath):
            import csv
            with open(tpath, newline='') as f:
                traj = [[float(row['x']), float(row['y']), float(row['z'])] for row in csv.DictReader(f)]
        return xyz, rgb, traj, {'status': 'done', 'source': ''}

    def execute_slam_sessions(self, data={}):
        slam = self.slam_runner()
        if slam is None:
            return self.success(sessions=[], available=False)
        return self.success(sessions=slam.execute_sessions({}).get('sessions', []), available=True)

    # -- Daten --

    def get_cloud(self, cloud='', slam=''):
        """ :return: (xyz, rgb, trajectory, info) """
        if slam:
            return self.slam_data(slam)
        folder = self.cloud_dir(cloud)
        if not os.path.isdir(folder):
            raise ValueError(f'Unbekannte Wolke {cloud!r}')
        z = np.load(os.path.join(folder, 'cloud.npz'))
        with open(os.path.join(folder, 'meta.json')) as f:
            meta = json.load(f)
        return z['xyz'], (z['rgb'] if 'rgb' in z else None), [], meta

    def execute_info(self, data):
        try:
            xyz, rgb, traj, info = self.get_cloud(data.get('cloud', ''), data.get('slam', ''))
        except Exception as e:
            return self.fail(str(e))
        step = max(1, len(traj) // 5000)
        res = {**info, 'points': int(len(xyz)), 'colors': rgb is not None,
               'trajectory': [[round(float(v), 4) for v in p] for p in traj[::step]]}
        if len(xyz):
            res['min'] = [float(v) for v in xyz.min(axis=0)]
            res['max'] = [float(v) for v in xyz.max(axis=0)]
        return self.success(**res)

    def subsample(self, xyz, rgb, limit):
        if limit <= 0 or len(xyz) <= limit:
            return xyz, rgb
        idx = np.sort(np.random.default_rng(0).choice(len(xyz), limit, replace=False))
        return xyz[idx], (rgb[idx] if rgb is not None else None)

    # -- Web (über den Dispatcher: /pysys/dispatcher?type=pointcloud&pointcloud=<name>) --

    def route(self, name):
        return f'/pysys/dispatcher?type={self.type}&{self.type}={name}'

    def page_data(self, params={}):
        from flask import request, Response
        try:
            xyz, rgb, _, _ = self.get_cloud(request.args.get('cloud', ''), request.args.get('slam', ''))
        except Exception as e:
            return Response(str(e), status=404)
        limit = min(self.max_points, int(request.args.get('max', self.max_points) or self.max_points))
        xyz, rgb = self.subsample(xyz, rgb, limit)
        return Response(pack_cloud(xyz, rgb), mimetype='application/octet-stream',
                        headers={'Cache-Control': 'no-store'})

    def page_upload(self, params={}):
        from flask import request
        f = request.files.get('file')
        if f is None:
            return self.fail('Keine Datei (Feld file)')
        name = os.path.basename(f.filename or 'cloud.ply')
        ext = os.path.splitext(name)[1].lower()
        tmp = os.path.join(self.ensure_dir('uploads'), u.guid() + (ext if re.fullmatch(r'\.\w{1,5}', ext) else '.ply'))
        f.save(tmp)
        try:
            return self.load_file(tmp, name, 'upload')
        finally:
            os.remove(tmp)

    def part_index(self, p: b.Page, params={}):
        cfg = {
            'type': self.type,
            'routes': {name: self.route(name) for name in ['data', 'upload']},
            'three': self.three_url, 'orbit': self.orbit_url,
            'slam': str(params.get('slam', '')) if ID_RE.fullmatch(str(params.get('slam', ''))) else '',
            'cloud': str(params.get('cloud', '')) if ID_RE.fullmatch(str(params.get('cloud', ''))) else '',
            'max_points': self.max_points
        }
        p(f'<style>{PC_CSS}</style>')
        p(PC_HTML)
        cfg_json = json.dumps(cfg).replace('</', '<\\/')
        # Block statt globaler Konstante: die UI v2 führt die Skripte bei jedem Öffnen erneut aus
        p('<script>{ const PC_CFG = ' + cfg_json + ';\n' + PC_JS + '\n}</script>')


PC_CSS = """
.pc { max-width: 1500px; }
.pc-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
.pc-row label { min-width: 0; }
.pc-card { border: 1px solid #8885; border-radius: 8px; padding: 10px 12px; }
.pc-view { position: relative; height: 70vh; min-height: 360px; border-radius: 6px; overflow: hidden; background: #0d1117; }
.pc-view canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.pc-overlay { position: absolute; left: 8px; top: 6px; font-size: 12px; color: #cfd8dc; pointer-events: none;
  text-shadow: 0 0 3px #000; }
.pc-help { position: absolute; right: 8px; bottom: 6px; font-size: 11px; color: #90a4ae; pointer-events: none; }
.pc-msg { min-height: 1.2em; }
.pc-msg.error { color: #c33; }
.pc input[type=number] { width: 80px; }
.pc input[type=range] { width: 130px; }
"""

PC_HTML = """
<div class="pc">
  <div class="pc-card">
    <div class="pc-row">
      <label for="pc-source">Quelle</label>
      <select id="pc-source" style="min-width: 320px"></select>
      <button id="pc-reload">Neu laden</button>
      <label><input type="checkbox" id="pc-live" /> live (SLAM läuft)</label>
    </div>
    <div class="pc-row">
      <label>Datei</label><input type="file" id="pc-file" accept=".ply,.xyz,.txt,.csv,.npz" />
      <button id="pc-upload">Hochladen</button>
      <label for="pc-path">oder Pfad</label><input type="text" id="pc-path" style="width: 300px" placeholder="/data/karte.ply" />
      <button id="pc-load">Laden</button>
    </div>
    <div class="pc-msg" id="pc-msg"></div>
  </div>
  <div class="pc-card" style="margin-top: 12px">
    <div class="pc-row">
      <label for="pc-color">Farbe</label>
      <select id="pc-color"><option value="rgb">RGB</option><option value="height">Höhe</option><option value="plain">einfarbig</option></select>
      <label for="pc-size">Punktgröße</label><input type="range" id="pc-size" min="0.2" max="8" step="0.1" value="1.5" />
      <label><input type="checkbox" id="pc-flip" checked /> OpenCV-Achsen (y unten)</label>
      <label><input type="checkbox" id="pc-traj" checked /> Kamerabahn</label>
      <label><input type="checkbox" id="pc-axes" checked /> Achsen</label>
      <label for="pc-max">max. Punkte</label><input type="number" id="pc-max" min="1000" step="1000" />
    </div>
    <div class="pc-row">
      <label>Höhe</label>
      <input type="range" id="pc-hlo" min="0" max="1000" value="0" /> <span id="pc-hlo-v"></span>
      <span>bis</span>
      <input type="range" id="pc-hhi" min="0" max="1000" value="1000" /> <span id="pc-hhi-v"></span>
      <button id="pc-floor" title="unterste dichte Schicht ausblenden">Boden ausblenden</button>
      <button id="pc-hall">Alle</button>
      <span style="margin-left: 12px">Ansicht</span>
      <button data-view="top">Oben</button><button data-view="front">Vorne</button><button data-view="side">Seite</button>
      <button data-view="fit">Einpassen</button>
    </div>
    <div class="pc-view" id="pc-view">
      <div class="pc-overlay" id="pc-stats">lade three.js ...</div>
      <div class="pc-help">Linke Maustaste drehen · rechte / Shift verschieben · Mausrad zoomen · Doppelklick: Punkt als Drehzentrum</div>
    </div>
  </div>
</div>
"""

PC_JS = r"""
(async function() {
  const cfg = PC_CFG;
  const $ = (id) => document.getElementById(id);
  // UI v2 ersetzt den Inhalt beim Wechsel: dann Timer beenden und WebGL freigeben
  const rootEl = $('pc-view').closest('.pc');
  const alive = () => rootEl.isConnected;
  const cleanups = [];
  const watcher = setInterval(() => {
    if (alive()) return;
    clearInterval(watcher);
    cleanups.forEach((f) => { try { f(); } catch (e) {} });
  }, 1000);
  const url = (path, params) => location.protocol + '//' + location.host + path +
    (params ? (path.indexOf('?') >= 0 ? '&' : '?') + new URLSearchParams(params) : '');
  const job = (data) => fetch(url('/'), { method: 'POST', body: new URLSearchParams({ type: cfg.type, ...data }) }).then((r) => r.json());
  const fmt = (v, n) => (v == null || isNaN(v)) ? '-' : (+v).toFixed(n == null ? 2 : n);
  function msg(text, error) { $('pc-msg').textContent = text || ''; $('pc-msg').classList.toggle('error', !!error); }
  const KEY = 'nxml_pointcloud';
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
  ['pc-color', 'pc-size', 'pc-max'].forEach((id) => { if (saved[id] != null) $(id).value = saved[id]; });
  ['pc-flip', 'pc-traj', 'pc-axes'].forEach((id) => { if (saved[id] != null) $(id).checked = saved[id]; });
  if (!$('pc-max').value) $('pc-max').value = Math.min(cfg.max_points, 500000);
  function save() {
    const st = {};
    ['pc-color', 'pc-size', 'pc-max'].forEach((id) => st[id] = $(id).value);
    ['pc-flip', 'pc-traj', 'pc-axes'].forEach((id) => st[id] = $(id).checked);
    try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {}
  }

  // -- three.js --
  let THREE, OrbitControls;
  try {
    THREE = await import(cfg.three);
    OrbitControls = (await import(cfg.orbit)).OrbitControls;
  } catch (e) {
    $('pc-stats').textContent = 'three.js nicht ladbar (' + cfg.three + '): ' + e.message;
    return;
  }
  const view = $('pc-view');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  view.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1e6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;
  controls.addEventListener('change', render);
  const world = new THREE.Group();  // OpenCV -> three: Drehung 180° um x
  scene.add(world);
  const material = new THREE.PointsMaterial({ size: 1, vertexColors: true, sizeAttenuation: true });
  const geometry = new THREE.BufferGeometry();
  const points = new THREE.Points(geometry, material);
  world.add(points);
  const trajMat = new THREE.LineBasicMaterial({ color: 0xffb02e });
  let trajLine = null, camMarker = null, axes = null;

  const D = { xyz: null, rgb: null, n: 0, up: null, lo: 0, hi: 0, diag: 1, traj: [], center: [0, 0, 0], source: null,
              status: '', loading: false };

  function resize() {
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    render();
  }
  let pending = false;
  cleanups.push(() => {
    controls.dispose(); geometry.dispose(); material.dispose();
    renderer.dispose(); renderer.forceContextLoss();
  });
  function render() {
    if (!alive()) return;
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; if (controls.update()) render(); renderer.render(scene, camera); });
  }
  const observer = new ResizeObserver(resize);
  observer.observe(view);
  cleanups.push(() => observer.disconnect());

  // -- Daten laden --
  function parse(buf) {
    const dv = new DataView(buf);
    if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'NXPC') throw new Error('Unbekanntes Format');
    const n = dv.getUint32(4, true), flags = dv.getUint32(8, true);
    const xyz = new Float32Array(buf, 12, n * 3);
    const rgb = flags & 1 ? new Uint8Array(buf, 12 + n * 12, n * 3) : null;
    return { n, xyz, rgb };
  }
  function percentile(arr, q) {
    if (!arr.length) return 0;
    const a = Float32Array.from(arr).sort();
    return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))];
  }
  function sign() { return $('pc-flip').checked ? -1 : 1; }  // Höhe = -y bei OpenCV-Achsen

  async function load(src, keepView) {
    if (!src || D.loading) return;
    D.loading = true;
    const q = src.slam ? { slam: src.slam } : { cloud: src.cloud };
    $('pc-stats').textContent = 'lade ...';
    try {
      const [info, buf] = await Promise.all([
        job({ op: 'info', ...q }),
        fetch(url(cfg.routes.data, { ...q, max: $('pc-max').value })).then((r) => {
          if (!r.ok) return r.text().then((t) => { throw new Error(t); });
          return r.arrayBuffer();
        })
      ]);
      if (!info.success) throw new Error(info.message);
      const c = parse(buf);
      const first = !D.source || D.source.slam !== src.slam || D.source.cloud !== src.cloud;
      Object.assign(D, { xyz: c.xyz, rgb: c.rgb, n: c.n, traj: info.trajectory || [], total: info.points, source: src,
                         status: info.status || '' });
      computeStats();
      if (first) resetHeight();
      rebuild();
      if (first || !keepView) fit();
      msg('');
    } catch (e) {
      msg('Laden fehlgeschlagen: ' + e.message, true);
      $('pc-stats').textContent = '';
    } finally {
      D.loading = false;
    }
  }
  function computeStats() {
    const n = D.n, step = Math.max(1, Math.floor(n / 20000)), xs = [], ys = [], zs = [];
    for (let i = 0; i < n; i += step) { xs.push(D.xyz[3 * i]); ys.push(D.xyz[3 * i + 1]); zs.push(D.xyz[3 * i + 2]); }
    const lo = [percentile(xs, 0.02), percentile(ys, 0.02), percentile(zs, 0.02)];
    const hi = [percentile(xs, 0.98), percentile(ys, 0.98), percentile(zs, 0.98)];
    D.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    D.diag = Math.max(1e-3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
    const s = sign(), hs = ys.map((y) => s * y);
    D.hmin = percentile(hs, 0.005); D.hmax = percentile(hs, 0.995);
    D.hsample = hs;
  }

  // -- Höhenfilter (Höhe = oben positiv) --
  function hval(id) { return D.hmin + (D.hmax - D.hmin) * (+$(id).value / 1000); }
  function resetHeight() { $('pc-hlo').value = 0; $('pc-hhi').value = 1000; }
  function heightLabels() { $('pc-hlo-v').textContent = fmt(hval('pc-hlo')); $('pc-hhi-v').textContent = fmt(hval('pc-hhi')); }

  function rebuild() {
    if (!D.xyz) return;
    const n = D.n, s = sign();
    const lo = +$('pc-hlo').value <= 0 ? -Infinity : hval('pc-hlo');
    const hi = +$('pc-hhi').value >= 1000 ? Infinity : hval('pc-hhi');
    const mode = $('pc-color').value;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    let m = 0;
    for (let i = 0; i < n; i++) {
      const y = D.xyz[3 * i + 1], h = s * y;
      if (h < lo || h > hi) continue;
      pos[3 * m] = D.xyz[3 * i]; pos[3 * m + 1] = y; pos[3 * m + 2] = D.xyz[3 * i + 2];
      if (mode === 'rgb' && D.rgb) {
        col[3 * m] = D.rgb[3 * i] / 255; col[3 * m + 1] = D.rgb[3 * i + 1] / 255; col[3 * m + 2] = D.rgb[3 * i + 2] / 255;
      } else if (mode === 'height' || (mode === 'rgb' && !D.rgb)) {
        const t = Math.max(0, Math.min(1, (h - D.hmin) / ((D.hmax - D.hmin) || 1)));
        // blau -> türkis -> gelb -> rot
        col[3 * m] = Math.min(1, Math.max(0, 1.5 * t - 0.25 + (t > 0.66 ? 0.3 : 0)));
        col[3 * m + 1] = Math.max(0, 1 - Math.abs(2 * t - 1) * 1.1);
        col[3 * m + 2] = Math.max(0, 1 - 1.6 * t);
      } else { col[3 * m] = 0.8; col[3 * m + 1] = 0.86; col[3 * m + 2] = 0.9; }
      m++;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, 3 * m), 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, 3 * m), 3));
    geometry.computeBoundingSphere();
    material.size = +$('pc-size').value * D.diag / 600;
    material.needsUpdate = true;
    world.rotation.x = $('pc-flip').checked ? Math.PI : 0;
    // Kamerabahn
    if (trajLine) { world.remove(trajLine); trajLine.geometry.dispose(); trajLine = null; }
    if (camMarker) { world.remove(camMarker); camMarker = null; }
    if ($('pc-traj').checked && D.traj.length > 1) {
      const g = new THREE.BufferGeometry().setFromPoints(D.traj.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
      trajLine = new THREE.Line(g, trajMat);
      world.add(trajLine);
      const last = D.traj[D.traj.length - 1];
      camMarker = new THREE.Mesh(new THREE.SphereGeometry(D.diag / 150, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd34d }));
      camMarker.position.set(last[0], last[1], last[2]);
      world.add(camMarker);
    }
    if (axes) { world.remove(axes); axes = null; }
    if ($('pc-axes').checked) { axes = new THREE.AxesHelper(D.diag / 8); world.add(axes); }
    heightLabels();
    const name = D.source && D.source.slam ? 'SLAM ' + D.source.slam : (D.source ? D.source.label : '');
    $('pc-stats').textContent = name + ' · ' + m.toLocaleString() + ' von ' + (D.total || n).toLocaleString() + ' Punkten' +
      (D.n < (D.total || 0) ? ' (Auswahl)' : '') + (D.status ? ' · ' + D.status : '');
    render();
  }

  function worldPoint(x, y, z) { return world.localToWorld(new THREE.Vector3(x, y, z)); }
  function setView(kind) {
    if (!D.xyz) return;
    const c = worldPoint(D.center[0], D.center[1], D.center[2]), r = D.diag * 0.9;
    const dirs = { top: [0, 1, 0.0001], front: [0, 0.25, 1], side: [1, 0.25, 0], fit: [0.6, 0.55, 0.9] };
    const v = new THREE.Vector3(...(dirs[kind] || dirs.fit)).normalize().multiplyScalar(r);
    controls.target.copy(c);
    camera.position.copy(c).add(v);
    camera.near = D.diag / 1000; camera.far = D.diag * 100;
    camera.updateProjectionMatrix();
    controls.update();
    render();
  }
  function fit() { world.updateMatrixWorld(); setView('fit'); }
  document.querySelectorAll('[data-view]').forEach((btn) => btn.addEventListener('click', () => setView(btn.dataset.view)));

  // Doppelklick: Punkt unter der Maus als Drehzentrum
  const raycaster = new THREE.Raycaster();
  renderer.domElement.addEventListener('dblclick', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.params.Points.threshold = material.size * 2;
    raycaster.setFromCamera(mouse, camera);
    const hit = raycaster.intersectObject(points)[0];
    if (hit) { controls.target.copy(hit.point); controls.update(); render(); }
  });

  // -- Bedienung --
  ['pc-color', 'pc-flip', 'pc-traj', 'pc-axes'].forEach((id) => $(id).addEventListener('change', () => {
    save();
    if (id === 'pc-flip') { computeStats(); resetHeight(); rebuild(); fit(); } else rebuild();
  }));
  $('pc-size').addEventListener('input', () => { save(); material.size = +$('pc-size').value * D.diag / 600; render(); });
  $('pc-max').addEventListener('change', () => { save(); load(D.source, true); });
  let hTimer = null;
  ['pc-hlo', 'pc-hhi'].forEach((id) => $(id).addEventListener('input', () => {
    heightLabels(); clearTimeout(hTimer); hTimer = setTimeout(rebuild, 60);
  }));
  $('pc-hall').addEventListener('click', () => { resetHeight(); rebuild(); });
  $('pc-floor').addEventListener('click', () => {
    // Boden: Häufung im unteren Viertel der Höhen, ausgeblendet wird bis knapp darüber
    if (!D.hsample || !D.hsample.length) return;
    const span = (D.hmax - D.hmin) || 1, bins = new Array(50).fill(0);
    for (const h of D.hsample) { const k = Math.floor((h - D.hmin) / span * 50); if (k >= 0 && k < 50) bins[k]++; }
    let best = 0;
    for (let k = 1; k < 13; k++) if (bins[k] > bins[best]) best = k;
    const cut = D.hmin + span * (best + 2) / 50;
    $('pc-hlo').value = Math.round(1000 * (cut - D.hmin) / span);
    msg('Boden bei ca. ' + fmt(D.hmin + span * (best + 0.5) / 50) + ', ausgeblendet bis ' + fmt(cut));
    rebuild();
  });

  // -- Quellen --
  async function loadSources(select) {
    const sel = $('pc-source');
    const [clouds, slam] = await Promise.all([job({ op: 'clouds' }), job({ op: 'slam_sessions' })]);
    sel.innerHTML = '';
    const add = (group, items) => {
      if (!items.length) return;
      const g = document.createElement('optgroup'); g.label = group;
      items.forEach(([value, text]) => { const o = document.createElement('option'); o.value = value; o.textContent = text; g.appendChild(o); });
      sel.appendChild(g);
    };
    add('SLAM-Läufe', (slam.sessions || []).map((s) => ['slam:' + s.session,
      s.session + ' · ' + s.status + ' · ' + s.points + ' Punkte']));
    add('Punktwolken', (clouds.clouds || []).map((c) => ['cloud:' + c.cloud, c.name + ' · ' + c.points + ' Punkte']));
    if (select) sel.value = select;
    return sel.value;
  }
  function sourceFromValue(v) {
    if (!v) return null;
    const [kind, id] = [v.slice(0, v.indexOf(':')), v.slice(v.indexOf(':') + 1)];
    const label = ($('pc-source').selectedOptions[0] || {}).textContent || id;
    return kind === 'slam' ? { slam: id, label } : { cloud: id, label };
  }
  $('pc-source').addEventListener('change', () => load(sourceFromValue($('pc-source').value)));
  $('pc-reload').addEventListener('click', () => load(D.source, true));
  const livePoll = setInterval(() => {
    const running = D.status === 'running' || D.status === 'starting';
    if ($('pc-live').checked && running && D.source && D.source.slam && !D.loading) load(D.source, true);
  }, 3000);
  cleanups.push(() => clearInterval(livePoll));
  async function loaded(r) {
    if (!r.success) { msg(r.message || 'Fehler', true); return; }
    msg(r.name + ': ' + r.points.toLocaleString() + ' Punkte');
    await loadSources('cloud:' + r.cloud);
    load(sourceFromValue('cloud:' + r.cloud));
  }
  $('pc-upload').addEventListener('click', () => {
    const file = $('pc-file').files[0];
    if (!file) { msg('Bitte eine Datei wählen', true); return; }
    const form = new FormData();
    form.append('file', file);
    msg('Lade hoch ...');
    fetch(url(cfg.routes.upload), { method: 'POST', body: form }).then((r) => r.json()).then(loaded)
      .catch((e) => msg('Upload fehlgeschlagen: ' + e.message, true));
  });
  $('pc-load').addEventListener('click', () => job({ op: 'load', path: $('pc-path').value.trim() }).then(loaded));

  resize();
  const initial = cfg.slam ? 'slam:' + cfg.slam : (cfg.cloud ? 'cloud:' + cfg.cloud : '');
  const first = await loadSources(initial);
  if (cfg.slam) $('pc-live').checked = true;
  const src = sourceFromValue(initial || first);
  if (src) load(src); else $('pc-stats').textContent = 'Keine Punktwolke, bitte hochladen oder SLAM-Lauf starten';
})();
"""


__all__ = ['PointCloudRunner', 'read_ply', 'read_xyz', 'read_cloud']
