#
# Project: nxml
#
# 3D-Rekonstruktion mit HunyuanWorld-Mirror (https://huggingface.co/tencent/HunyuanWorld-Mirror)
#
"""
WorldMirror-Runner (type ``worldmirror``)

Feed-forward 3D-Rekonstruktion aus Bildern oder einem Video mit Tencent HunyuanWorld-Mirror (~1B Parameter):
Punktwolke, Tiefe, Normalen, Kameraposen/-intrinsik (und optional 3D-Gaussians) in einem Forward-Pass.

Das Modell braucht den Code aus https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror und torch 2.4 (cu124).
Damit das nicht mit dem torch des Runners kollidiert, läuft die Inferenz in einem eigenen venv
(Standard ``~/.venvs/worldmirror``, Repo in ``~/.venvs/worldmirror-src``) als Unterprozess:

 - Einrichten: Job ``{"type": "worldmirror", "op": "setup"}`` (oder automatisch beim ersten Job, auto_setup)
   klont das Repo und installiert mit uv nur ins venv. Fortschritt: ``{"op": "status"}``
 - Rekonstruktion: ``{"type": "worldmirror", "images": [<base64>, ...]}``, ``{"video": <base64>}``
   oder ``{"path": "<Ordner mit Bildern oder Video auf dem Server>"}``

Ausgabe (Ordner ``<work_dir>/runs/<run>``): points.ply, cameras.json, optional depth.npz/normals.npz,
gaussians.ply. Ist ein PointCloudRunner (type ``pointcloud``) im Dispatcher, wird die Wolke dort
übernommen und ``viewer`` verlinkt die 3D-Ansicht.

    {"success": true, "run": "...", "points": 123456, "views": 8, "cameras": [...], "cloud": "...", "viewer": "..."}
"""
import base64
import collections
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from nwebclient import runner as r
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import dev as d


ID_RE = re.compile(r'[\w-]+')

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic')
VIDEO_EXT = ('.mp4', '.mov', '.avi', '.mkv', '.webm')


# Läuft im venv von WorldMirror (cwd = Repo), nicht im Runner-Python
WORKER_SCRIPT = r'''
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

args = json.loads(sys.argv[1])
sys.path.insert(0, args['repo'])
from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import extract_load_and_preprocess_images

t0 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WorldMirror.from_pretrained(args['model_id']).to(device).eval()
imgs = extract_load_and_preprocess_images(Path(args['input']), fps=args['fps'], target_size=args['target_size'],
                                          mode=args['mode']).to(device)
if args['max_views'] > 0 and imgs.shape[1] > args['max_views']:
    idx = torch.linspace(0, imgs.shape[1] - 1, args['max_views']).round().long().to(device)
    imgs = imgs[:, idx]
views = {'img': imgs}
cond_flags = [0, 0, 0]
with torch.no_grad():
    if device.type == 'cuda':
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            pred = model(views=views, cond_flags=cond_flags)
    else:
        pred = model(views=views, cond_flags=cond_flags)

def npy(x):
    x = x.detach().float().cpu().numpy()
    return x[0] if x.ndim > 0 and x.shape[0] == 1 and args.get('squeeze', True) else x

out = Path(args['output'])
out.mkdir(parents=True, exist_ok=True)
pts = npy(pred['pts3d']).reshape(-1, imgs.shape[-2], imgs.shape[-1], 3)          # S,H,W,3
conf = npy(pred['pts3d_conf']).reshape(pts.shape[:3])                              # S,H,W
rgb = (imgs[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # S,H,W,3

mask = np.isfinite(pts).all(-1)
if args['conf_percentile'] > 0:
    thr = np.percentile(conf[mask], args['conf_percentile']) if mask.any() else 0
    mask &= conf >= thr
xyz, col = pts[mask].astype(np.float32), rgb[mask]
if args['max_points'] > 0 and len(xyz) > args['max_points']:
    sel = np.sort(np.random.default_rng(0).choice(len(xyz), args['max_points'], replace=False))
    xyz, col = xyz[sel], col[sel]

with open(out / 'points.ply', 'wb') as f:
    f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n'
             'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
             % len(xyz)).encode())
    rec = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['r'], rec['g'], rec['b'] = col[:, 0], col[:, 1], col[:, 2]
    f.write(rec.tobytes())

poses = npy(pred['camera_poses']).reshape(-1, 4, 4)
intrs = npy(pred['camera_intrs']).reshape(-1, 3, 3)
cams = [{'pose': p.tolist(), 'intrinsics': k.tolist()} for p, k in zip(poses, intrs)]
with open(out / 'cameras.json', 'w') as f:
    json.dump({'cameras': cams, 'size': [int(imgs.shape[-1]), int(imgs.shape[-2])], 'convention': 'opencv c2w'}, f)

if args['save_depth'] and 'depth' in pred:
    np.savez_compressed(out / 'depth.npz', depth=npy(pred['depth']).squeeze(-1),
                        conf=npy(pred['depth_conf']) if 'depth_conf' in pred else None)
if args['save_normals'] and 'normals' in pred:
    np.savez_compressed(out / 'normals.npz', normals=npy(pred['normals']))
if args['save_gs'] and 'splats' in pred:
    s = pred['splats']
    def g(k):
        v = s[k][0] if isinstance(s[k], (list, tuple)) else s[k]
        return v.detach().float().cpu().numpy()
    means, scales, quats = g('means').reshape(-1, 3), g('scales').reshape(-1, 3), g('quats').reshape(-1, 4)
    opac = g('opacities').reshape(-1)
    sh = g('sh').reshape(len(means), -1)[:, :3]
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3']
    logit = np.log(np.clip(opac, 1e-6, 1 - 1e-6) / (1 - np.clip(opac, 1e-6, 1 - 1e-6)))
    data = np.concatenate([means, sh, logit[:, None], np.log(np.clip(scales, 1e-9, None)), quats], 1).astype('<f4')
    with open(out / 'gaussians.ply', 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % len(data)).encode())
        f.write(''.join('property float %s\n' % n for n in names).encode() + b'end_header\n')
        f.write(data.tobytes())

print('WM_RESULT ' + json.dumps({'points': int(len(xyz)), 'views': int(imgs.shape[1]),
                                 'size': [int(imgs.shape[-1]), int(imgs.shape[-2])],
                                 'device': device.type, 'duration': round(time.time() - t0, 2)}), flush=True)
'''


class WorldMirror(r.BaseJobExecutor):
    """
    3D-Rekonstruktion mit HunyuanWorld-Mirror aus Bildern oder Video: Punktwolke, Kameras, Tiefe, Normalen, 3DGS.

    Läuft in einem eigenen venv (torch 2.4/cu124 + Repo-Code), siehe Moduldoku. Operationen (``op``):
    ``setup``, ``status``, ``runs``, ``delete``, ``clean_files``. Ohne ``op`` wird rekonstruiert.
    """

    TAGS = ['vision', '3d', 'hugging']

    # Absichtlich ohne torch: das Modell läuft im eigenen venv
    MODULES = ['numpy']

    type = 'worldmirror'

    REPO_URL = 'https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git'
    UV_RELEASE_URL = 'https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-unknown-linux-gnu.tar.gz'

    def __init__(self, type='worldmirror', model_id='tencent/HunyuanWorld-Mirror', venv='~/.venvs/worldmirror',
                 repo_dir=None, work_dir=None, python_version='3.10', torch_index='https://download.pytorch.org/whl/cu124',
                 target_size=518, fps=1, mode='crop', max_views=32, conf_percentile=10, max_points=2000000,
                 save_depth=False, save_normals=False, save_gs=False, timeout=1800, auto_setup=True,
                 pointcloud_type='pointcloud'):
        """
            :param model_id: Hugging-Face-Modell
            :param venv: isoliertes venv für WorldMirror (torch 2.4, Repo-Abhängigkeiten)
            :param repo_dir: Klon des GitHub-Repos, Standard <venv>-src
            :param work_dir: Ergebnisse, Standard <tmp>/nxml_worldmirror
            :param python_version: Python für das venv (uv), das Repo verlangt 3.10
            :param torch_index: Wheel-Index für torch 2.4.0 / torchvision 0.19.0
            :param target_size: Bildgröße für das Modell (518)
            :param fps: Bilder pro Sekunde aus Videos
            :param mode: Vorverarbeitung "crop" oder "pad"
            :param max_views: höchstens so viele Ansichten (gleichmäßig ausgewählt), 0 = alle
            :param conf_percentile: Punkte unter diesem Konfidenz-Perzentil verwerfen (0 = keine Filterung)
            :param max_points: höchstens so viele Punkte in points.ply, 0 = alle
            :param save_depth: depth.npz speichern
            :param save_normals: normals.npz speichern
            :param save_gs: gaussians.ply (3D Gaussian Splatting) speichern
            :param timeout: Sekunden pro Rekonstruktion (der erste Lauf lädt das Modell, ~5 GB)
            :param auto_setup: fehlt das venv, startet der erste Job die Installation
            :param pointcloud_type: type des PointCloudRunners für die 3D-Ansicht
        """
        super().__init__(type)
        self.model_id = model_id
        self.venv = os.path.expanduser(venv)
        self.repo_dir = os.path.expanduser(repo_dir) if repo_dir else self.venv.rstrip('/\\') + '-src'
        self.work_dir = work_dir or os.path.join(tempfile.gettempdir(), 'nxml_worldmirror')
        self.python_version = str(python_version)
        self.torch_index = torch_index
        self.target_size = int(target_size)
        self.fps = float(fps)
        self.mode = mode
        self.max_views = int(max_views)
        self.conf_percentile = float(conf_percentile)
        self.max_points = int(max_points)
        self.save_depth = self._bool(save_depth)
        self.save_normals = self._bool(save_normals)
        self.save_gs = self._bool(save_gs)
        self.timeout = float(timeout)
        self.auto_setup = self._bool(auto_setup)
        self.pointcloud_type = pointcloud_type
        self.log = collections.deque(maxlen=300)
        self.setup_thread = None
        self.setup_state = None
        self.lock = threading.Lock()
        self.define_sig(d.Param('images'), d.PStr('name', 'scene'))\
            .set_description("Rekonstruiert eine Szene aus Bildern (Liste Base64)")\
            .r(success='bool', run='str', points='int', views='int', cameras='list', cloud='str', viewer='str')
        self.define_sig(d.Param('video'), d.PStr('name', 'scene.mp4'))\
            .set_description("Rekonstruiert eine Szene aus einem Video (Base64)").r(success='bool', run='str')
        self.define_sig(d.Param('path')).set_description("Rekonstruiert aus einem Ordner/Video auf dem Server")\
            .r(success='bool', run='str')
        self.define_sig(d.PStr('op', 'setup')).set_description("Repo klonen und venv mit uv einrichten").r()
        self.define_sig(d.PStr('op', 'status')).set_description("Stand von venv, Repo, Setup und Log").r()
        self.define_sig(d.PStr('op', 'runs')).set_description("Bisherige Rekonstruktionen").r(runs='list')
        self.define_sig(d.PStr('op', 'delete'), d.Param('run')).set_description("Rekonstruktion löschen").r()
        self.define_sig(d.PStr('op', 'clean_files')).set_description("venv und Repo löschen").r(removed='list')

    @staticmethod
    def _bool(v):
        return str(v).lower() not in ('0', 'false', 'no', '', 'none')

    # --- venv / Setup -------------------------------------------------------------------------------------------

    @property
    def python(self):
        if os.name == 'nt':
            return os.path.join(self.venv, 'Scripts', 'python.exe')
        return os.path.join(self.venv, 'bin', 'python')

    @property
    def uv_dir(self):
        return os.path.join(os.path.dirname(self.venv.rstrip('/\\')), '.uv')

    def _is_venv(self):
        return os.path.isfile(os.path.join(self.venv, 'pyvenv.cfg')) and os.path.exists(self.python)

    def _is_repo(self):
        return os.path.isfile(os.path.join(self.repo_dir, 'src', 'models', 'models', 'worldmirror.py'))

    def installed(self):
        return self._is_venv() and self._is_repo() and os.path.isfile(os.path.join(self.venv, '.nxml_ok'))

    def _isolated_env(self):
        """ kein User-site, kein PYTHONPATH des Runners """
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        env['PYTHONNOUSERSITE'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        return env

    def _uv_env(self):
        env = self._isolated_env()
        env['UV_PYTHON_INSTALL_DIR'] = os.path.join(self.uv_dir, 'python')
        env['UV_PYTHON_INSTALL_BIN'] = '0'
        env['UV_PYTHON_PREFERENCE'] = 'only-managed'
        env['UV_NO_MODIFY_PATH'] = '1'
        return env

    def find_uv(self):
        exe = 'uv.exe' if os.name == 'nt' else 'uv'
        for path in (shutil.which('uv'), os.path.expanduser('~/.local/bin/' + exe), os.path.join(self.uv_dir, exe)):
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def install_uv(self):
        """ Lädt die uv-Binary nach uv_dir (nur Linux, kein PATH-Eintrag) """
        import platform
        import tarfile
        import urllib.request
        arch = {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'aarch64', 'arm64': 'aarch64'}\
            .get(platform.machine().lower())
        if arch is None or not sys.platform.startswith('linux'):
            raise RuntimeError(f'uv bitte selbst installieren (Download nur für Linux), hier {sys.platform}')
        url = self.UV_RELEASE_URL.format(arch=arch)
        os.makedirs(self.uv_dir, exist_ok=True)
        self.log.append('Lade uv: ' + url)
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, 'uv.tar.gz')
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    name = os.path.basename(member.name)
                    if member.isfile() and name in ('uv', 'uvx'):
                        dst = os.path.join(self.uv_dir, name)
                        with open(dst, 'wb') as out:
                            shutil.copyfileobj(tar.extractfile(member), out)
                        os.chmod(dst, 0o755)
        return os.path.join(self.uv_dir, 'uv')

    def setup_commands(self, uv):
        cmds = []
        if not self._is_repo():
            if not shutil.which('git'):
                raise RuntimeError('git fehlt')
            cmds.append((['git', 'clone', '--depth', '1', self.REPO_URL, self.repo_dir], None))
        if not self._is_venv():
            cmds.append(([uv, 'venv', '--seed', '--python', self.python_version, self.venv], None))
        pip = [uv, 'pip', 'install', '--python', self.python]
        cmds.append((pip + ['torch==2.4.0', 'torchvision==0.19.0', '--index-url', self.torch_index], None))
        cmds.append((pip + ['-r', os.path.join(self.repo_dir, 'requirements.txt')], None))
        if self.save_gs:
            cmds.append((pip + ['gsplat', '--find-links', 'https://docs.gsplat.studio/whl/pt24cu124'], 'optional'))
        return cmds

    def _run_logged(self, cmd, env, cwd=None, timeout=None):
        """ führt cmd aus, Ausgabe ins Log; :return: (exit code, Zeilen) """
        self.log.append('$ ' + ' '.join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace')
        lines = []
        timer = threading.Timer(timeout, proc.kill) if timeout else None
        if timer:
            timer.start()
        try:
            for line in proc.stdout:
                line = line.rstrip()
                lines.append(line)
                self.log.append(line)
            return proc.wait(), lines
        finally:
            if timer:
                timer.cancel()

    def _setup_worker(self):
        try:
            uv = self.find_uv()
            if uv is None:
                self.setup_state = 'running: uv herunterladen'
                uv = self.install_uv()
            for cmd, flag in self.setup_commands(uv):
                self.setup_state = 'running: ' + ' '.join(cmd[:3])
                code, _ = self._run_logged(cmd, self._uv_env())
                if code != 0:
                    if flag == 'optional':
                        self.log.append('WARNUNG: optionaler Schritt fehlgeschlagen')
                        continue
                    raise RuntimeError(f'{" ".join(cmd[:3])} fehlgeschlagen (Exit {code})')
            with open(os.path.join(self.venv, '.nxml_ok'), 'w') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
            self.setup_state = 'ok'
            self.info(f'WorldMirror in {self.venv} eingerichtet')
        except Exception as e:
            self.setup_state = 'failed: ' + str(e)
            self.log.append('FEHLER: ' + str(e))
            self.error('Setup: ' + str(e))

    def setup_running(self):
        return self.setup_thread is not None and self.setup_thread.is_alive()

    def execute_setup(self, data={}):
        if self.setup_running():
            return self.success('Setup läuft bereits', setup=self.setup_state)
        if self.installed():
            return self.success(f'WorldMirror ist bereits in {self.venv} eingerichtet')
        if os.path.exists(self.venv) and not self._is_venv():
            return self.fail(f'{self.venv} existiert, ist aber kein venv')
        if os.path.exists(self.repo_dir) and not self._is_repo():
            return self.fail(f'{self.repo_dir} existiert, ist aber kein WorldMirror-Repo')
        self.log.clear()
        self.setup_state = 'running'
        self.setup_thread = threading.Thread(target=self._setup_worker, daemon=True)
        self.setup_thread.start()
        return self.success('Setup gestartet', venv=self.venv, repo=self.repo_dir)

    def execute_status(self, data={}):
        return self.success(installed=self.installed(), venv=self.venv, venv_ok=self._is_venv(),
                            repo=self.repo_dir, repo_ok=self._is_repo(), uv=self.find_uv(), model=self.model_id,
                            setup=self.setup_state, setup_running=self.setup_running(),
                            busy=self.lock.locked(), log=list(self.log)[-40:])

    def execute_clean_files(self, data={}):
        if self.setup_running() or self.lock.locked():
            return self.fail('Setup oder Rekonstruktion läuft noch')
        removed = []
        if self._is_venv():
            shutil.rmtree(self.venv)
            removed.append(self.venv)
        if self._is_repo():
            shutil.rmtree(self.repo_dir)
            removed.append(self.repo_dir)
        self.setup_state = None
        return self.success('Gelöscht', removed=removed)

    # --- Läufe --------------------------------------------------------------------------------------------------

    def run_dir(self, run_id):
        if not ID_RE.fullmatch(str(run_id)):
            raise ValueError(f'Ungültiger run {run_id!r}')
        return os.path.join(self.work_dir, 'runs', run_id)

    def execute_runs(self, data={}):
        res = []
        folder = os.path.join(self.work_dir, 'runs')
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                path = os.path.join(folder, name, 'meta.json')
                if os.path.isfile(path):
                    try:
                        with open(path) as f:
                            res.append(json.load(f))
                    except (OSError, ValueError):
                        pass
        return self.success(runs=sorted(res, key=lambda m: m.get('created', 0), reverse=True))

    def execute_delete(self, data):
        try:
            folder = self.run_dir(data['run'])
        except (KeyError, ValueError) as e:
            return self.fail(str(e))
        if not os.path.isdir(folder):
            return self.fail('Unbekannter Lauf')
        shutil.rmtree(folder)
        return self.success('Gelöscht')

    def _write_inputs(self, data, folder):
        """ legt die Eingaben in folder/input ab. :return: Pfad für den Worker (Ordner oder Video) """
        inp = os.path.join(folder, 'input')
        os.makedirs(inp, exist_ok=True)
        if data.get('video'):
            ext = os.path.splitext(os.path.basename(data.get('name', 'scene.mp4')))[1].lower()
            path = os.path.join(inp, 'video' + (ext if ext in VIDEO_EXT else '.mp4'))
            with open(path, 'wb') as f:
                f.write(base64.b64decode(data['video']))
            return path
        images = data.get('images') or ([data['image']] if data.get('image') else [])
        if isinstance(images, str):
            images = [images]
        for i, img in enumerate(images):
            if isinstance(img, str) and ',' in img[:100] and img.startswith('data:'):
                img = img.split(',', 1)[1]
            raw = base64.b64decode(img)
            ext = '.png' if raw[:4] == b'\x89PNG' else '.jpg'
            with open(os.path.join(inp, f'{i:04d}{ext}'), 'wb') as f:
                f.write(raw)
        return inp

    def pointcloud_runner(self):
        dispatcher = self.getParentClass(r.LazyDispatcher)
        if dispatcher is not None and dispatcher.support_type(self.pointcloud_type):
            return dispatcher.get_runner(self.pointcloud_type)
        return None

    def reconstruct(self, data):
        if not self.installed():
            if self.auto_setup and not self.setup_running():
                self.execute_setup()
            return self.fail(f'WorldMirror ist noch nicht eingerichtet (Setup: {self.setup_state}), '
                             'Fortschritt mit {"op": "status"}', setup=self.setup_state)
        path = data.get('path')
        if path and not (os.path.isdir(path) or (os.path.isfile(path) and path.lower().endswith(VIDEO_EXT))):
            return self.fail(f'Ordner oder Video nicht gefunden: {path}')
        if not path and not (data.get('images') or data.get('image') or data.get('video')):
            return self.fail('images, video oder path fehlt')
        if not self.lock.acquire(blocking=False):
            return self.fail('Es läuft bereits eine Rekonstruktion')
        try:
            run_id = time.strftime('%Y%m%d-%H%M%S-') + u.guid()[:6]
            folder = self.run_dir(run_id)
            os.makedirs(folder, exist_ok=True)
            source = path or self._write_inputs(data, folder)
            out = os.path.join(folder, 'output')
            script = os.path.join(self.work_dir, 'worldmirror_worker.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write(WORKER_SCRIPT)
            num = lambda k, default, t: t(data.get(k, default))
            args = {'repo': self.repo_dir, 'model_id': self.model_id, 'input': source, 'output': out,
                    'fps': num('fps', self.fps, float), 'target_size': num('target_size', self.target_size, int),
                    'mode': data.get('mode', self.mode), 'max_views': num('max_views', self.max_views, int),
                    'conf_percentile': num('conf_percentile', self.conf_percentile, float),
                    'max_points': num('max_points', self.max_points, int),
                    'save_depth': self._bool(data.get('save_depth', self.save_depth)),
                    'save_normals': self._bool(data.get('save_normals', self.save_normals)),
                    'save_gs': self._bool(data.get('save_gs', self.save_gs))}
            code, lines = self._run_logged([self.python, script, json.dumps(args)], self._isolated_env(),
                                           cwd=self.repo_dir, timeout=self.timeout)
            result = next((json.loads(l[len('WM_RESULT '):]) for l in reversed(lines)
                           if l.startswith('WM_RESULT ')), None)
            if code != 0 or result is None:
                return self.fail(f'WorldMirror fehlgeschlagen (Exit {code})', run=run_id, log=lines[-30:])
            with open(os.path.join(out, 'cameras.json')) as f:
                cameras = json.load(f)['cameras']
            meta = {'run': run_id, 'name': data.get('name', os.path.basename(str(source))), 'created': time.time(),
                    **result, 'files': sorted(os.listdir(out)), 'output': out}
            pc = self.pointcloud_runner()
            if pc is not None:
                res = pc.load_file(os.path.join(out, 'points.ply'), f'worldmirror {run_id}', 'worldmirror:' + run_id)
                if res.get('success'):
                    meta['cloud'] = res['cloud']
                    meta['viewer'] = f'/pysys/dispatcher?type={self.pointcloud_type}&cloud={res["cloud"]}'
            with open(os.path.join(folder, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            return self.success(**meta, cameras=cameras)
        except Exception as e:
            return self.fail(str(e))
        finally:
            self.lock.release()

    def execute(self, data):
        if 'op' not in data and any(k in data for k in ('images', 'image', 'video', 'path')):
            return self.reconstruct(data)
        return super().execute(data)

    # --- Web ----------------------------------------------------------------------------------------------------

    def part_index(self, p: b.Page, params={}):
        st = self.execute_status()
        p.h2('HunyuanWorld-Mirror')
        p.div(f'Modell: {self.model_id}, venv: {self.venv} ({"ok" if st["venv_ok"] else "fehlt"}), '
              f'Repo: {self.repo_dir} ({"ok" if st["repo_ok"] else "fehlt"}), Setup: {self.setup_state or "-"}')
        if not self.installed():
            p(self.exec_btn('Setup starten', op='setup'))
        p(self.exec_btn('Status', op='status'))
        p(self.exec_btn('Läufe', op='runs'))
        if st['log']:
            from html import escape
            p('<pre style="max-height:300px;overflow:auto">' + escape('\n'.join(st['log'])) + '</pre>')
