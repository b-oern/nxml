#
# Project: nxml
#
# Monokulares Visual SLAM: Kameraposition und Weltkarte (Punktwolke) aus einem Video oder HTTP-Stream
#
"""
SLAM-Runner (type ``slam``)

Bestimmt aus einem Kamerabild (Videodatei, HTTP-/RTSP-Stream oder Webcam) die Position der Kamera und baut
eine Weltkarte als Punktwolke auf, ähnlich wie ARCore, aber nur mit dem Bild (ohne IMU).

Backends:
 - ``opencv``: eingebaute monokulare Visual Odometry (KLT-Tracking, Essential-Matrix-Initialisierung,
   PnP-Tracking, Triangulation neuer Punkte). Läuft live, liefert für jedes Bild ein Event ``slam_pose``.
   Braucht nur ``opencv-python-headless`` und ``numpy``.
 - ``stella``: stella_vslam (``run_video_slam`` aus stella_vslam_examples) als Subprozess, nur für Videodateien.
   Genauer (Bundle Adjustment, Loop Closing), Posen und Karte gibt es erst nach dem Durchlauf.
 - ``auto``: Streams mit opencv, Videodateien mit stella, falls installiert, sonst opencv.

Monokular ist der Maßstab unbekannt: Positionen und Punkte haben eine beliebige Einheit (erste Bewegung = 1).
Koordinaten wie OpenCV, Ursprung = erste Kameraposition: x rechts, y unten, z vorwärts. Die Karte zeigt x/z
von oben, yaw ist die Drehung um die Hochachse (0 = Blickrichtung beim Start, positiv nach rechts).

Arbeitsverzeichnis: work_dir (Standard <tmp>/nxml_slam, unter Linux /tmp/nxml_slam), je Lauf
sessions/<id>/ mit trajectory.csv, points.ply und session.json, Uploads unter uploads/.

UI unter /pysys/runner?type=slam: Quelle, Kamerabild, Radar und Karte mit der Bewegung der Kamera.
"""
import base64
import csv
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import deque

from nwebclient import runner as r
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import dev as d

try:
    import numpy as np
    import cv2
except ImportError:  # Runner bleibt ladbar, die UI zeigt die fehlenden Module
    np = None
    cv2 = None


STELLA_VOCAB_URL = 'https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow'

STREAM_RE = re.compile(r'^(https?|rtsp|rtsps|rtmp|udp|tcp|srt)://', re.I)

# Drive Radar: Kuchenausschnitt vor der Kamera, Einheiten wie die Karte (erste Bewegung = 1)
DRIVE_DEFAULTS = {
    'angle': 90.0,      # Öffnungswinkel des Ausschnitts in Grad
    'range': 5.0,       # Reichweite der Anzeige
    'check': 2.0,       # Prüfdistanz: Punkte bis zu dieser Entfernung sind "im Weg"
    'lower': 0.0,       # Untergrenze über der Kamerahöhe (Punkte darunter werden ignoriert)
    'up_angle': 45.0,   # nach oben nur bis zu diesem Sichtwinkel
    'threshold': 5,     # so viele Punkte dürfen im Weg sein, ohne dass der Weg blockiert ist
    'events': True,     # Event slam_path bei Änderung auslösen
}

# Einstellungen, die die Seite Einstellungen ändern kann (Typ zum Umwandeln)
GENERAL_SETTINGS = {'backend': str, 'fov': float, 'width': int, 'event_every': int, 'push_timeout': float,
                    'quality_level': float, 'min_distance': int}


# -- Mathematik --

def pose_center(T_cw):
    """ Kamerazentrum in Weltkoordinaten aus der Welt->Kamera-Transformation """
    return -T_cw[:3, :3].T @ T_cw[:3, 3]


def invert_pose(T):
    res = np.eye(4)
    res[:3, :3] = T[:3, :3].T
    res[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return res


def rotation_to_quaternion(R):
    """ :return: (qx, qy, qz, qw) """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def quaternion_to_rotation(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])


def pose_angles(R_wc):
    """
    Winkel der Kamera in Grad (Welt = erste Kamera, y nach unten):
    yaw um die Hochachse (positiv nach rechts), pitch (positiv nach oben), roll (positiv im Uhrzeigersinn)
    """
    f = R_wc[:, 2]   # Blickrichtung
    rgt = R_wc[:, 0]  # rechts
    yaw = math.degrees(math.atan2(f[0], f[2]))
    pitch = math.degrees(math.atan2(-f[1], math.hypot(f[0], f[2])))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, rgt[1]))))
    return yaw, pitch, roll


def camera_matrix(width, height, fov=70.0, fx=None, fy=None, cx=None, cy=None, scale=1.0):
    """
    Intrinsische Matrix für ein Bild width x height.
    fx/fy/cx/cy gelten für die Originalauflösung und werden mit scale umgerechnet, ohne fx wird sie aus dem
    horizontalen Öffnungswinkel fov geschätzt.
    """
    if fx:
        fx = float(fx) * scale
        fy = float(fy) * scale if fy else fx
    else:
        fx = fy = (width / 2.0) / math.tan(math.radians(float(fov) / 2.0))
    cx = float(cx) * scale if cx not in (None, '') else width / 2.0
    cy = float(cy) * scale if cy not in (None, '') else height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


# -- Bildquelle --

def decode_image(data):
    """ JPEG/PNG-Bytes -> BGR-Bild oder None """
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def iter_jpeg_frames(read, stopped=lambda: False, chunk_size=65536):
    """
    Zerlegt einen Bytestrom (MJPEG, multipart/x-mixed-replace oder aneinandergehängte JPEGs) in Bilder.
    :param read: read(n) -> bytes, b'' am Ende
    """
    buf = b''
    while not stopped():
        chunk = read(chunk_size)
        if not chunk:
            break
        buf += chunk
        while True:
            start = buf.find(b'\xff\xd8')
            if start < 0:
                buf = buf[-1:]
                break
            end = buf.find(b'\xff\xd9', start + 2)
            if end < 0:
                buf = buf[start:]
                break
            frame = decode_image(buf[start:end + 2])
            buf = buf[end + 2:]
            if frame is not None:
                yield frame
        if len(buf) > 20_000_000:
            buf = b''


class PushSource:
    """
    Bildquelle für Bilder, die ein Client schickt (z.B. die nx-App nach dem QR-Code):
    push(frame) legt das neueste Bild ab, read() liefert es wie bei einem Stream.
    Ohne neues Bild für timeout Sekunden endet die Quelle.
    """

    is_stream = True
    kind = 'push'
    fps = 0.0

    def __init__(self, timeout=60):
        self.timeout = timeout
        self.error = None
        self.frames = 0
        self._latest = None
        self._seq = 0
        self._read_seq = 0
        self._closed = False
        self._started = time.time()
        self._last_push = time.time()
        self._cond = threading.Condition()

    def open(self):
        return self

    def push(self, frame):
        if self._closed:
            raise ValueError('Lauf ist beendet')
        with self._cond:
            self._latest = frame
            self._seq += 1
            self.frames += 1
            self._last_push = time.time()
            self._cond.notify_all()

    def read(self):
        with self._cond:
            while self._seq == self._read_seq and not self._closed:
                if time.time() - self._last_push > self.timeout:
                    return None
                self._cond.wait(0.5)
            if self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._latest, time.time() - self._started

    def progress(self):
        return None

    def close(self):
        self._closed = True
        with self._cond:
            self._cond.notify_all()


class FrameSource:
    """
    Liest Bilder aus einer Videodatei, einem Stream (HTTP-MJPEG, JPEG-Snapshot, alles was FFmpeg kann)
    oder einer Webcam (Index als Zahl).

    Streams (live) werden in einem eigenen Thread gelesen, read() liefert immer das neueste Bild, ältere werden
    verworfen, damit die Position aktuell bleibt. Dateien werden Bild für Bild gelesen.
    """

    def __init__(self, source, frame_skip=1, realtime=False, timeout=15):
        self.source = str(source).strip()
        self.is_stream = bool(STREAM_RE.match(self.source)) or self.source.isdigit()
        self.frame_skip = max(1, int(frame_skip or 1))
        self.realtime = realtime
        self.timeout = timeout
        self.cap = None
        self.fps = 0.0
        self.frame_count = 0
        self.kind = ''
        self.error = None
        self._stop = False
        self._latest = None
        self._seq = 0
        self._read_seq = 0
        self._ended = False
        self._cond = threading.Condition()
        self._thread = None
        self._mjpeg = None
        self._started = time.time()
        self._file_index = 0
        self._last_emit = None

    # Öffnen

    def open(self):
        if not self.is_stream:
            if not os.path.isfile(self.source):
                raise ValueError(f'Video nicht gefunden: {self.source}')
            self._open_capture(self.source)
            self.kind = 'file'
            return self
        if self.source.isdigit():
            self._open_capture(int(self.source))
            self.kind = 'webcam'
        elif self.source.lower().startswith(('http://', 'https://')) and self._open_http():
            pass
        else:
            self._open_capture(self.source)
            self.kind = 'stream'
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _open_capture(self, src):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError(f'Quelle kann nicht geöffnet werden: {self.source}')
        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _open_http(self):
        """ MJPEG-Streams und JPEG-Snapshots direkt lesen (ohne FFmpeg), sonst False """
        req = urllib.request.Request(self.source, headers={'User-Agent': 'nxml-slam'})
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        ctype = (resp.headers.get('Content-Type') or '').lower()
        if ctype.startswith('multipart/'):
            self._mjpeg = resp
            self.kind = 'mjpeg'
            return True
        if ctype.startswith('image/'):
            resp.close()
            self.kind = 'snapshot'
            return True
        resp.close()
        return False

    # Lesen (Streams)

    def _publish(self, frame):
        with self._cond:
            self._latest = frame
            self._seq += 1
            self._cond.notify_all()

    def _reader(self):
        try:
            if self.kind == 'mjpeg':
                self._read_mjpeg()
            elif self.kind == 'snapshot':
                self._read_snapshots()
            else:
                fails = 0
                while not self._stop:
                    ok, frame = self.cap.read()
                    if not ok:
                        fails += 1
                        if fails > 50:
                            raise IOError('Stream liefert keine Bilder mehr')
                        time.sleep(0.05)
                        continue
                    fails = 0
                    self._publish(frame)
        except Exception as e:
            if not self._stop:
                self.error = str(e)
        finally:
            with self._cond:
                self._ended = True
                self._cond.notify_all()

    def _read_mjpeg(self):
        resp = self._mjpeg
        for frame in iter_jpeg_frames(getattr(resp, 'read1', resp.read), lambda: self._stop):
            self._publish(frame)

    def _read_snapshots(self):
        while not self._stop:
            req = urllib.request.Request(self.source, headers={'User-Agent': 'nxml-slam'})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                self._publish(frame)

    def read(self):
        """ :return: (frame BGR, t in Sekunden) oder None am Ende """
        if not self.is_stream:
            return self._read_file()
        with self._cond:
            while self._seq == self._read_seq and not self._ended and not self._stop:
                self._cond.wait(0.5)
            if self._seq == self._read_seq:
                if self.error:
                    raise IOError(self.error)
                return None
            self._read_seq = self._seq
            return self._latest, time.time() - self._started

    def _read_file(self):
        for _ in range(self.frame_skip - 1):
            if not self.cap.grab():
                return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        t = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0
        if t <= 0 and self.fps > 0:
            t = self._file_index * self.frame_skip / self.fps
        self._file_index += 1
        if self.realtime and self.fps > 0:
            # so schnell wie das Video abspielen, die UI sieht dann wie live aus
            now = time.time()
            if self._last_emit is not None:
                wait = self.frame_skip / self.fps - (now - self._last_emit)
                if wait > 0:
                    time.sleep(wait)
            self._last_emit = time.time()
        return frame, t

    def progress(self):
        if self.is_stream or self.frame_count <= 0:
            return None
        return min(1.0, self._file_index * self.frame_skip / self.frame_count)

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        if self._mjpeg is not None:
            try:
                self._mjpeg.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(2)
        if self.cap is not None:
            self.cap.release()


# -- Visual Odometry --

class MonoVO:
    """
    Monokulare Visual Odometry mit lokaler Karte.

     1. Initialisierung: Merkmale (Shi-Tomasi) per KLT verfolgen, bis die Bewegung groß genug ist, dann
        Essential-Matrix, Pose (recoverPose) und Triangulation der ersten Punkte.
     2. Tracking: Pose je Bild per PnP-RANSAC aus den verfolgten Kartenpunkten.
     3. Keyframes: alle kf_interval Bilder oder wenn zu wenige Punkte sichtbar sind, werden noch nicht
        kartierte Merkmale zwischen ihrem ersten Keyframe und dem aktuellen Bild trianguliert und neue
        Merkmale gesucht.
     4. Verloren: neue Initialisierung ab der letzten Pose, der Maßstab wird aus der bisherigen
        Geschwindigkeit übernommen.

    Posen: T_cw (Welt -> Kamera), Welt = erste Kamera.
    """

    def __init__(self, K, width, max_features=800, min_tracked=120, kf_interval=5, init_parallax=None,
                 min_angle=1.0, reproj_err=2.0, quality_level=0.01, min_distance=10):
        """
            :param quality_level: Schwellwert der Punkterkennung (goodFeaturesToTrack): eine Ecke zählt nur, wenn
                                  sie mindestens diesen Anteil der stärksten Ecke im Bild erreicht (0.001 - 0.5)
            :param min_distance: Mindestabstand der erkannten Punkte in Pixeln (bezogen auf die Verarbeitungsbreite)
        """
        self.K = np.asarray(K, np.float64)
        self.quality_level = min(0.99, max(1e-4, float(quality_level)))
        self.min_distance = max(1, int(min_distance))
        self.max_features = int(max_features)
        self.min_tracked = int(min_tracked)
        self.kf_interval = int(kf_interval)
        self.init_parallax = float(init_parallax or max(8.0, 0.03 * width))
        self.cos_min_angle = math.cos(math.radians(float(min_angle)))
        self.reproj_err = float(reproj_err)
        self.state = 'init'  # init | tracking | lost
        self.segment = 0
        self.prev_gray = None
        self.T_cw = np.eye(4)
        self.kf_poses = []
        self.L = np.zeros((0, 3))  # Kartenpunkte (Welt)
        self.speed = 0.0           # mittlerer Weg pro Bild, Maßstab für eine neue Initialisierung
        self.frames_since_kf = 0
        self.inliers = 0
        self._new = []
        self._clear_tracks()

    # Merkmale

    def _clear_tracks(self):
        self.pts = np.zeros((0, 2), np.float32)       # aktuelle Bildposition
        self.ids = np.zeros(0, np.int64)              # Kartenpunkt oder -1
        self.orig_pts = np.zeros((0, 2), np.float32)  # Bildposition im Keyframe der ersten Beobachtung
        self.orig_kf = np.zeros(0, np.int64)          # Index dieses Keyframes

    def _select(self, keep):
        self.pts = self.pts[keep]
        self.ids = self.ids[keep]
        self.orig_pts = self.orig_pts[keep]
        self.orig_kf = self.orig_kf[keep]

    def _detect(self, gray, n):
        if n <= 0:
            return np.zeros((0, 2), np.float32)
        mask = np.full(gray.shape[:2], 255, np.uint8)
        for x, y in self.pts:
            cv2.circle(mask, (int(x), int(y)), self.min_distance, 0, -1)
        c = cv2.goodFeaturesToTrack(gray, maxCorners=int(n), qualityLevel=self.quality_level,
                                    minDistance=self.min_distance, mask=mask,
                                    blockSize=7)
        return np.zeros((0, 2), np.float32) if c is None else c.reshape(-1, 2).astype(np.float32)

    def _add_keyframe(self, gray):
        self.kf_poses.append(self.T_cw.copy())
        self.frames_since_kf = 0
        new = self._detect(gray, self.max_features - len(self.pts))
        if len(new):
            n = len(new)
            self.pts = np.vstack([self.pts, new])
            self.ids = np.concatenate([self.ids, np.full(n, -1, np.int64)])
            self.orig_pts = np.vstack([self.orig_pts, new])
            self.orig_kf = np.concatenate([self.orig_kf, np.full(n, len(self.kf_poses) - 1, np.int64)])

    def _restart(self, gray):
        self._clear_tracks()
        self._add_keyframe(gray)

    LK = dict(winSize=(21, 21), maxLevel=3, criteria=(3, 30, 0.01))  # 3 = TERM_CRITERIA_EPS | COUNT

    def _flow(self, prev, cur, pts):
        if len(pts) == 0:
            return pts, np.zeros(0, bool)
        p0 = pts.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **self.LK)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **self.LK)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        p1 = p1.reshape(-1, 2)
        h, w = cur.shape[:2]
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0) \
            & (p1[:, 0] >= 0) & (p1[:, 1] >= 0) & (p1[:, 0] < w - 1) & (p1[:, 1] < h - 1)
        return p1.astype(np.float32), ok

    # Geometrie

    def _triangulate(self, T1, T2, p1, p2):
        """ :return: (Punkte Welt (N,3), gültig (N,)) """
        if len(p1) == 0:
            return np.zeros((0, 3)), np.zeros(0, bool)
        P1 = self.K @ T1[:3]
        P2 = self.K @ T2[:3]
        Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
        with np.errstate(divide='ignore', invalid='ignore'):
            X = (Xh[:3] / Xh[3]).T
            ok = np.abs(Xh[3]) > 1e-12
            for T, p in ((T1, p1), (T2, p2)):
                c = (T[:3, :3] @ X.T + T[:3, 3:4]).T
                ok &= c[:, 2] > 1e-6
                uv = (self.K @ c.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                ok &= np.linalg.norm(uv - p, axis=1) < self.reproj_err
            # genug Parallaxe, sonst ist die Tiefe zu ungenau
            r1 = X - pose_center(T1)
            r2 = X - pose_center(T2)
            cos = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
            ok &= cos < self.cos_min_angle
        ok &= np.all(np.isfinite(X), axis=1)
        return X, ok

    def _add_landmarks(self, X, track_idx):
        ids = np.arange(len(self.L), len(self.L) + len(X))
        self.L = np.vstack([self.L, X])
        self.ids[track_idx] = ids
        self._new.append((ids, self.pts[track_idx].copy()))
        return ids

    # Ablauf

    def process(self, gray):
        """
        :param gray: Graustufenbild
        :return: dict(keyframe, initialized, lost, new=[(landmark_ids, bildpunkte)])
        """
        self._new = []
        res = {'keyframe': False, 'initialized': False, 'lost': False, 'new': self._new}
        if self.prev_gray is None:
            self._restart(gray)
            self.prev_gray = gray
            return res
        pts, ok = self._flow(self.prev_gray, gray, self.pts)
        self.pts = pts
        self._select(ok)
        self.prev_gray = gray
        self.frames_since_kf += 1
        if self.state == 'tracking':
            self._track(gray, res)
        else:
            self._initialize(gray, res)
        return res

    def _initialize(self, gray, res):
        if len(self.pts) < 40 or self.frames_since_kf > 400:
            self._restart(gray)
            return
        parallax = float(np.median(np.linalg.norm(self.pts - self.orig_pts, axis=1)))
        if parallax < self.init_parallax:
            return
        E, mask = cv2.findEssentialMat(self.orig_pts, self.pts, self.K, method=cv2.RANSAC, prob=0.999,
                                       threshold=1.0)
        if E is None or E.shape[0] < 3:
            return
        n, R, t, mask = cv2.recoverPose(E[:3], self.orig_pts, self.pts, self.K, mask=mask)
        if n < 30:
            return
        T_kf = self.kf_poses[-1]
        scale = self.speed * self.frames_since_kf if self.speed > 0 else 1.0
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.ravel() * scale
        T_cur = T_rel @ T_kf
        inl = np.where(mask.ravel() > 0)[0]
        X, good = self._triangulate(T_kf, T_cur, self.orig_pts[inl], self.pts[inl])
        if good.sum() < 30:
            return
        if self.speed <= 0:
            self.speed = scale / max(1, self.frames_since_kf)
        self.T_cw = T_cur
        self._add_landmarks(X[good], inl[good])
        self.state = 'tracking'
        self.segment += 1
        self.inliers = int(good.sum())
        res['initialized'] = True
        self._add_keyframe(gray)
        res['keyframe'] = True

    def _lose(self, gray, res):
        res['lost'] = True
        self.state = 'lost'
        self.inliers = 0
        self._restart(gray)

    def _track(self, gray, res):
        m = np.where(self.ids >= 0)[0]
        if len(m) < 15:
            return self._lose(gray, res)
        obj = self.L[self.ids[m]].astype(np.float64)
        img = self.pts[m].astype(np.float64)
        rvec, _ = cv2.Rodrigues(self.T_cw[:3, :3])
        tvec = self.T_cw[:3, 3].reshape(3, 1).copy()
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, rvec, tvec, useExtrinsicGuess=True,
                                                     iterationsCount=100, reprojectionError=self.reproj_err * 2,
                                                     confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok, inl = False, None
        if not ok or inl is None or len(inl) < 12:
            return self._lose(gray, res)
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        outlier = np.ones(len(m), bool)
        outlier[inl] = False
        keep = np.ones(len(self.pts), bool)
        keep[m[outlier]] = False
        self._select(keep)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        step = float(np.linalg.norm(pose_center(T) - pose_center(self.T_cw)))
        self.speed = step if self.speed <= 0 else 0.9 * self.speed + 0.1 * step
        self.T_cw = T
        self.inliers = int(len(inl))
        if self.frames_since_kf >= self.kf_interval or len(inl) < self.min_tracked:
            self._map()
            self._add_keyframe(gray)
            res['keyframe'] = True

    def _map(self):
        """ Noch nicht kartierte Merkmale triangulieren """
        unmapped = np.where(self.ids < 0)[0]
        for kf in np.unique(self.orig_kf[unmapped]):
            sel = unmapped[self.orig_kf[unmapped] == kf]
            X, good = self._triangulate(self.kf_poses[kf], self.T_cw, self.orig_pts[sel], self.pts[sel])
            if good.any():
                self._add_landmarks(X[good], sel[good])


# -- Lauf --

class SlamSession:
    """ Ein SLAM-Lauf: Quelle, Thread, Trajektorie, Punktwolke und Ergebnisse im Arbeitsverzeichnis """

    def __init__(self, owner, session_id, folder, source='', backend='opencv', params=None):
        self.owner = owner
        self.id = session_id
        self.folder = folder
        self.source = source
        self.backend = backend
        self.params = dict(params or {})
        self.status = 'starting'  # starting | running | done | stopped | error
        self.message = ''
        self.tracking = 'init'
        self.started = time.time()
        self.finished = None
        self.frame = 0
        self.fps = 0.0
        self.progress = None
        self.size = None
        self.poses = []   # dicts, siehe pose_dict
        self.points = []  # [x, y, z, r, g, b]
        self.keyframes = 0
        self.log = deque(maxlen=200)
        self.jpeg = None
        self.jpeg_frame = -1
        self.lock = threading.Lock()
        self._stop = False
        self._thread = None
        self.push_source = None  # PushSource, wenn der Client die Bilder schickt

    # Ablauf

    def start(self):
        os.makedirs(self.folder, exist_ok=True)
        target = self.run_stella if self.backend == 'stella' else self.run_opencv
        self._thread = threading.Thread(target=self._run, args=(target,), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self.push_source is not None:
            self.push_source.close()

    def is_running(self):
        return self.status in ('starting', 'running')

    def _run(self, target):
        self.status = 'running'
        self.owner.emit_event('slam_state', session=self.id, status='running', tracking=self.tracking,
                              source=self.source, backend=self.backend)
        try:
            target()
            self.status = 'stopped' if self._stop else 'done'
        except Exception as e:
            self.status = 'error'
            self.message = str(e)
            self.owner.error(f'SLAM {self.id}: {e}')
        finally:
            self.finished = time.time()
            try:
                self.save()
            except Exception as e:
                self.owner.error(f'SLAM {self.id}: Speichern fehlgeschlagen: {e}')
        self.owner.emit_event('slam_done', session=self.id, status=self.status, message=self.message,
                              frames=self.frame, poses=len(self.poses), points=len(self.points), folder=self.folder)
        self.owner.info(f'SLAM {self.id}: {self.status}, {len(self.poses)} Posen, {len(self.points)} Punkte')

    def pose_dict(self, T_wc, t, segment=0):
        R = T_wc[:3, :3]
        x, y, z = (float(v) for v in T_wc[:3, 3])
        qx, qy, qz, qw = rotation_to_quaternion(R)
        yaw, pitch, roll = pose_angles(R)
        return {
            'frame': self.frame, 't': round(float(t), 3),
            'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4),
            'qx': round(qx, 5), 'qy': round(qy, 5), 'qz': round(qz, 5), 'qw': round(qw, 5),
            'yaw': round(yaw, 2), 'pitch': round(pitch, 2), 'roll': round(roll, 2),
            'segment': int(segment)
        }

    def add_pose(self, pose):
        with self.lock:
            self.poses.append(pose)
        every = max(1, int(self.owner.event_every))
        if len(self.poses) % every == 0:
            self.owner.emit_event('slam_pose', session=self.id, tracking=self.tracking, points=len(self.points),
                                  **pose)
        if self.owner.drive.get('events') and len(self.poses) % 5 == 0:
            try:
                self.owner.drive_check(self)
            except Exception as e:
                self.log.append(f'Drive-Prüfung: {e}')

    def points_array(self):
        """ Kartenpunkte als numpy (N,3), neue Punkte werden angehängt statt alles neu umzuwandeln """
        with self.lock:
            pts = self.points
            n = len(pts)
            arr = getattr(self, '_arr', None)
            if arr is None or getattr(self, '_arr_src', None) is not pts or len(arr) > n:
                arr = np.zeros((0, 3))  # Liste wurde ersetzt (z.B. stella), neu aufbauen
                self._arr_src = pts
            if len(arr) < n:
                new = np.array([p[:3] for p in pts[len(arr):n]], np.float64).reshape(-1, 3)
                arr = np.vstack([arr, new])
            self._arr = arr
            return arr

    def set_tracking(self, state):
        if state != self.tracking:
            self.tracking = state
            self.owner.emit_event('slam_state', session=self.id, status=self.status, tracking=state,
                                  frame=self.frame)

    # Backend opencv

    def run_opencv(self):
        p = self.params
        src = self.push_source or FrameSource(self.source, p.get('frame_skip', 1), p.get('realtime', False))
        src.open()
        self.log.append(f'Quelle geöffnet: {src.kind}, {src.fps:.1f} FPS')
        vo = None
        undistort = None
        last_jpeg = 0.0
        times = deque(maxlen=30)
        try:
            while not self._stop:
                item = src.read()
                if item is None:
                    break
                frame, t = item
                h0, w0 = frame.shape[:2]
                width = int(p.get('width') or 640)
                scale = min(1.0, width / float(w0)) if width > 0 else 1.0
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(round(w0 * scale)), int(round(h0 * scale))),
                                       interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
                if vo is None:
                    K = camera_matrix(w, h, p.get('fov', 70), p.get('fx'), p.get('fy'), p.get('cx'), p.get('cy'),
                                      scale)
                    dist = np.array([float(p.get(k) or 0) for k in ('k1', 'k2', 'p1', 'p2', 'k3')])
                    if np.any(dist != 0):
                        undistort = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
                    vo = MonoVO(K, w, max_features=p.get('max_features', 800), kf_interval=p.get('kf_interval', 5),
                                min_angle=p.get('min_angle', 1.5), quality_level=p.get('quality_level', 0.01),
                                min_distance=p.get('min_distance', 10))
                    self.log.append(f'Punkterkennung: qualityLevel={vo.quality_level}, minDistance={vo.min_distance}')
                    self.size = [w, h]
                    self.log.append(f'Bildgröße {w}x{h}, fx={K[0, 0]:.1f}')
                if undistort is not None:
                    frame = cv2.remap(frame, undistort[0], undistort[1], cv2.INTER_LINEAR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = vo.process(gray)
                self.frame += 1
                times.append(time.time())
                if len(times) > 1:
                    self.fps = round((len(times) - 1) / max(1e-6, times[-1] - times[0]), 1)
                self.progress = src.progress()
                self._apply(vo, res, frame, t)
                if time.time() - last_jpeg > 0.15:
                    self._preview(vo, frame)
                    last_jpeg = time.time()
            if src.error and not self._stop:
                raise IOError(src.error)
        finally:
            src.close()
            if vo is not None:
                self._preview(vo, None)

    def _apply(self, vo, res, frame, t):
        if res['new']:
            h, w = frame.shape[:2]
            new_points = []
            for ids, pts in res['new']:
                for lid, (px, py) in zip(ids, pts):
                    x, y, z = vo.L[lid]
                    bgr = frame[min(h - 1, max(0, int(py))), min(w - 1, max(0, int(px)))]
                    new_points.append([round(float(x), 4), round(float(y), 4), round(float(z), 4),
                                       int(bgr[2]), int(bgr[1]), int(bgr[0])])
            with self.lock:
                self.points.extend(new_points)
        if res['keyframe']:
            self.keyframes += 1
        if res['lost']:
            self.log.append(f'Bild {self.frame}: Tracking verloren, neue Initialisierung')
        if res['initialized']:
            self.log.append(f'Bild {self.frame}: initialisiert (Segment {vo.segment}), {len(vo.L)} Punkte')
        self.set_tracking(vo.state)
        if vo.state == 'tracking':
            self.add_pose(self.pose_dict(invert_pose(vo.T_cw), t, vo.segment))

    def _preview(self, vo, frame):
        if frame is None:
            return
        img = frame.copy()
        mapped = vo.ids >= 0
        for (x, y), is_mapped in zip(vo.pts, mapped):
            color = (60, 220, 60) if is_mapped else (0, 200, 255)
            cv2.circle(img, (int(x), int(y)), 2 if is_mapped else 1, color, -1, cv2.LINE_AA)
        label = f'{vo.state}  {vo.inliers} Punkte'
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
            self.jpeg_frame = self.frame

    # Backend stella_vslam

    def run_stella(self):
        p = self.params
        owner = self.owner
        exe = owner.stella_executable()
        if exe is None:
            raise RuntimeError(f'stella_vslam nicht gefunden ({owner.stella_bin}), backend=opencv verwenden '
                               f'oder stella_bin setzen')
        vocab = owner.stella_vocabulary(self.log)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f'Video kann nicht gelesen werden: {self.source}')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first = cap.read()
        cap.release()
        if ok:
            self._set_jpeg(first)
        self.size = [w, h]
        K = camera_matrix(w, h, p.get('fov', 70), p.get('fx'), p.get('fy'), p.get('cx'), p.get('cy'))
        config = os.path.join(self.folder, 'config.yaml')
        with open(config, 'w') as f:
            f.write(stella_config(w, h, fps, K, [float(p.get(k) or 0) for k in ('k1', 'k2', 'p1', 'p2', 'k3')]))
        map_db = os.path.join(self.folder, 'map.msg')
        cmd = [exe, '-v', vocab, '-m', os.path.abspath(self.source), '-c', config,
               '--frame-skip', str(max(1, int(p.get('frame_skip') or 1))), '--no-sleep', '--auto-term',
               '--map-db-out', map_db, '--eval-log-dir', self.folder]
        if owner.stella_viewer:
            cmd += ['--viewer', owner.stella_viewer]
        cmd += shlex.split(owner.stella_args or '')
        self.log.append(' '.join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd=self.folder)
        frame_re = re.compile(r'frame\D{0,10}(\d+)', re.I)
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.log.append(line)
                    m = frame_re.search(line)
                    if m and total:
                        self.progress = min(1.0, int(m.group(1)) / total)
                if self._stop:
                    proc.terminate()
                    break
            proc.wait(30)
        finally:
            if proc.poll() is None:
                proc.kill()
        if self._stop:
            return
        if proc.returncode != 0:
            raise RuntimeError(f'run_video_slam beendet mit Code {proc.returncode}: ' + ' | '.join(list(self.log)[-5:]))
        self.progress = 1.0
        self._load_stella_results(map_db)

    def _set_jpeg(self, frame):
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
            self.jpeg_frame = self.frame

    def _load_stella_results(self, map_db):
        traj = os.path.join(self.folder, 'frame_trajectory.txt')
        if not os.path.isfile(traj):
            traj = os.path.join(self.folder, 'keyframe_trajectory.txt')
        if os.path.isfile(traj):
            self.set_tracking('tracking')
            with open(traj) as f:
                for line in f:
                    vals = line.split()
                    if len(vals) != 8 or line.startswith('#'):
                        continue
                    t, tx, ty, tz, qx, qy, qz, qw = (float(v) for v in vals)
                    T = np.eye(4)
                    T[:3, :3] = quaternion_to_rotation(qx, qy, qz, qw)
                    T[:3, 3] = (tx, ty, tz)
                    self.frame += 1
                    self.add_pose(self.pose_dict(T, t, 1))
        else:
            self.log.append('Keine Trajektorie gefunden (frame_trajectory.txt)')
        if os.path.isfile(map_db):
            try:
                import msgpack
                with open(map_db, 'rb') as f:
                    data = msgpack.unpackb(f.read(), strict_map_key=False)
                pts = [[round(float(v), 4) for v in lm['pos_w']] + [200, 200, 200]
                       for lm in data.get('landmarks', {}).values()]
                kfs = data.get('keyfrms', {})
                with self.lock:
                    self.points = pts
                self.keyframes = len(kfs)
            except ImportError:
                self.log.append('msgpack fehlt (pip install msgpack), Punktwolke nicht geladen')
            except Exception as e:
                self.log.append(f'Karte nicht lesbar: {e}')

    # Speichern / Laden

    def save(self):
        os.makedirs(self.folder, exist_ok=True)
        with self.lock:
            poses = list(self.poses)
            points = list(self.points)
        with open(os.path.join(self.folder, 'trajectory.csv'), 'w', newline='') as f:
            f.write(trajectory_csv(poses))
        with open(os.path.join(self.folder, 'points.ply'), 'w') as f:
            f.write(points_ply(points))
        with open(os.path.join(self.folder, 'session.json'), 'w') as f:
            json.dump(self.summary(), f, indent=2)
        if self.jpeg:
            with open(os.path.join(self.folder, 'last_frame.jpg'), 'wb') as f:
                f.write(self.jpeg)

    def summary(self):
        return {
            'session': self.id, 'source': self.source, 'backend': self.backend, 'status': self.status,
            'message': self.message, 'started': self.started, 'finished': self.finished, 'frames': self.frame,
            'poses': len(self.poses), 'points': len(self.points), 'keyframes': self.keyframes,
            'folder': self.folder, 'params': self.params
        }

    @classmethod
    def load(cls, owner, folder):
        with open(os.path.join(folder, 'session.json')) as f:
            info = json.load(f)
        s = cls(owner, info['session'], folder, info.get('source', ''), info.get('backend', ''),
                info.get('params', {}))
        s.status = info.get('status', 'done')
        s.message = info.get('message', '')
        s.started = info.get('started', 0)
        s.finished = info.get('finished')
        s.frame = info.get('frames', 0)
        s.keyframes = info.get('keyframes', 0)
        s.tracking = '-'
        path = os.path.join(folder, 'trajectory.csv')
        if os.path.isfile(path):
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    s.poses.append({k: (int(float(v)) if k in ('frame', 'segment') else float(v))
                                    for k, v in row.items()})
        path = os.path.join(folder, 'points.ply')
        if os.path.isfile(path):
            s.points = read_ply(path)
        path = os.path.join(folder, 'last_frame.jpg')
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                s.jpeg = f.read()
        return s

    def state(self, traj_from=0, pts_from=0, max_points=5000):
        with self.lock:
            traj = self.poses[traj_from:]
            pts = self.points[pts_from:pts_from + max_points]
            n_points = len(self.points)
            n_poses = len(self.poses)
            pose = self.poses[-1] if self.poses else None
        return {
            'session': self.id, 'status': self.status, 'message': self.message, 'tracking': self.tracking,
            'backend': self.backend, 'source': self.source, 'frame': self.frame, 'fps': self.fps,
            'progress': self.progress, 'size': self.size, 'keyframes': self.keyframes, 'pose': pose,
            'jpeg_frame': self.jpeg_frame if self.jpeg else -1,
            'traj': [[p['frame'], p['t'], p['x'], p['y'], p['z'], p['yaw'], p['segment']] for p in traj],
            'traj_next': n_poses,
            'points': pts, 'pts_next': pts_from + len(pts), 'n_points': n_points,
            'log': list(self.log)[-12:], 'folder': self.folder
        }


# -- Dateiformate --

TRAJECTORY_COLUMNS = ['frame', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw', 'pitch', 'roll', 'segment']


def trajectory_csv(poses):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, TRAJECTORY_COLUMNS, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for pose in poses:
        writer.writerow(pose)
    return buf.getvalue()


def points_ply(points):
    lines = ['ply', 'format ascii 1.0', f'element vertex {len(points)}', 'property float x', 'property float y',
             'property float z', 'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
    lines += [f'{x} {y} {z} {r} {g} {b}' for x, y, z, r, g, b in points]
    return '\n'.join(lines) + '\n'


def read_ply(path):
    """ Liest die von points_ply geschriebenen Dateien """
    res = []
    with open(path) as f:
        for line in f:
            if line.strip() == 'end_header':
                break
        for line in f:
            vals = line.split()
            if len(vals) >= 6:
                res.append([float(vals[0]), float(vals[1]), float(vals[2]), int(vals[3]), int(vals[4]), int(vals[5])])
    return res


def stella_config(width, height, fps, K, dist):
    k1, k2, p1, p2, k3 = dist
    return f"""# erzeugt von nxml.vision.slam
Camera:
  name: "nxml"
  setup: "monocular"
  model: "perspective"
  fx: {K[0, 0]:.4f}
  fy: {K[1, 1]:.4f}
  cx: {K[0, 2]:.4f}
  cy: {K[1, 2]:.4f}
  k1: {k1}
  k2: {k2}
  p1: {p1}
  p2: {p2}
  k3: {k3}
  fps: {fps:.3f}
  cols: {int(width)}
  rows: {int(height)}
  color_order: "RGB"

Preprocessing:
  min_size: 800

Feature:
  name: "default ORB feature extraction setting"
  scale_factor: 1.2
  num_levels: 8
  ini_fast_threshold: 20
  min_fast_threshold: 7

Mapping:
  baseline_dist_thr_ratio: 0.02
  redundant_obs_ratio_thr: 0.9
"""


# -- Runner --

class SlamRunner(r.BaseJobExecutor):
    """
    Monokulares Visual SLAM: bestimmt aus einem Video, HTTP-Stream oder einer Webcam die Kameraposition
    (Event slam_pose) und baut eine Weltkarte (Punktwolke) auf.

    Quelle (source): Pfad einer Videodatei, URL eines Streams (MJPEG über HTTP, JPEG-Snapshot-URL,
    RTSP/HLS über FFmpeg) oder Webcam-Index (0).

    Backends: opencv (live, eingebaut), stella (stella_vslam run_video_slam, nur Videodateien), auto.
    Der Maßstab ist unbekannt (monokular), die erste Bewegung hat die Länge 1.

    Arbeitsverzeichnis: work_dir, Standard <tmp>/nxml_slam (Linux: /tmp/nxml_slam).
    UI: Quelle wählen, Kamerabild mit verfolgten Punkten, Radar um die Kamera, Karte mit Kamerabewegung.

    Events:
     - slam_pose: je Bild mit Tracking, session, frame, t, x, y, z, qx, qy, qz, qw, yaw, pitch, roll, tracking, points
     - slam_state: Status oder Tracking-Zustand geändert (init, tracking, lost)
     - slam_done: Lauf beendet, mit Anzahl Posen/Punkte und Ordner
    """

    TAGS = ['video', 'vision', 'slam']

    MODULES = ['opencv-python-headless', 'numpy', 'qrcode']

    type = 'slam'

    def __init__(self, type='slam', work_dir=None, backend='auto', width=640, fov=70, event_every=1,
                 stella_bin='run_video_slam', stella_vocab='', stella_viewer='none', stella_args='',
                 push_timeout=60, pointcloud_type='pointcloud', quality_level=0.01, min_distance=10):
        """
            :param work_dir: Arbeitsverzeichnis, Standard <tmp>/nxml_slam, wird bei Bedarf angelegt
            :param backend: auto, opencv oder stella
            :param width: Verarbeitungsbreite in Pixel für opencv (kleiner = schneller)
            :param fov: horizontaler Öffnungswinkel der Kamera in Grad, falls fx nicht bekannt ist
            :param event_every: slam_pose nur für jedes n-te Bild auslösen
            :param stella_bin: run_video_slam (Pfad oder im PATH)
            :param stella_vocab: orb_vocab.fbow, leer = wird bei Bedarf ins Arbeitsverzeichnis geladen
            :param stella_viewer: Wert für --viewer (none, pangolin_viewer, ...), leer = Option weglassen
            :param stella_args: weitere Argumente für run_video_slam
            :param push_timeout: Sekunden ohne gesendetes Bild, nach denen ein Push-Lauf endet
            :param quality_level: Schwellwert der Punkterkennung (opencv), höher = weniger, stärkere Punkte
            :param min_distance: Mindestabstand erkannter Punkte in Pixeln (opencv)
        """
        super().__init__(type)
        self.work_dir = work_dir or os.path.join(tempfile.gettempdir(), 'nxml_slam')
        self.backend = backend
        self.width = int(width)
        self.fov = float(fov)
        self.event_every = int(event_every)
        self.stella_bin = stella_bin
        self.stella_vocab = stella_vocab
        self.stella_viewer = stella_viewer
        self.stella_args = stella_args
        self.push_timeout = float(push_timeout)
        self.pointcloud_type = pointcloud_type
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.sessions = {}
        self._vocab_lock = threading.Lock()
        self.drive = dict(DRIVE_DEFAULTS)
        self._drive_free = {}  # session -> letzter Zustand, slam_path nur bei Änderung
        self.load_settings()
        self.define_event('slam_path', {'session': 'ID', 'free': 'Weg frei (bool)', 'count': 'Punkte im Weg',
                                        'threshold': 'erlaubte Punkte', 'check': 'Prüfdistanz'})
        self.define_event('slam_pose', {
            'session': 'ID des Laufs', 'frame': 'Bildnummer', 't': 'Zeit in s (Video) bzw. seit Start (Stream)',
            'x': 'Position rechts', 'y': 'Position unten', 'z': 'Position vorwärts',
            'qx': 'Quaternion', 'qy': 'Quaternion', 'qz': 'Quaternion', 'qw': 'Quaternion',
            'yaw': 'Grad um die Hochachse', 'pitch': 'Grad nach oben', 'roll': 'Grad im Uhrzeigersinn',
            'tracking': 'init, tracking, lost', 'points': 'Punkte in der Karte'
        })
        self.define_event('slam_state', {'session': 'ID', 'status': 'running, done, ...',
                                         'tracking': 'init, tracking, lost'})
        self.define_event('slam_done', {'session': 'ID', 'status': 'done, stopped, error', 'poses': 'Anzahl',
                                        'points': 'Anzahl', 'folder': 'Ergebnisordner'})
        self.define_sig(d.PStr('op', 'start'), d.PStr('source', ''), d.PStr('backend', 'auto'),
                        d.NP('fov', 'float', 70), d.PInt('width', 640), d.PInt('frame_skip', 1), d.PInt('realtime', 0))\
            .set_description("SLAM starten, source: Videopfad, Stream-URL (http/rtsp) oder Webcam-Index, "
                             "optional fx, fy, cx, cy, k1, k2, p1, p2, k3 (Kalibrierung, Originalauflösung)")\
            .r(success='bool', session='str', backend='str', folder='str')
        self.define_sig(d.PStr('op', 'start_from_stream'), d.PStr('source', ''), d.Param('image'),
                        d.PStr('session', ''))\
            .set_description("Ziel des QR-Codes: source = Stream-URL des Handys, oder image = Base64-JPEG "
                             "(wiederholt senden), ohne beides wird eine Session mit push_url angelegt "
                             "(POST JPEG oder MJPEG-Strom)")\
            .r(success='bool', session='str', mode='str', push_url='str')
        self.define_sig(d.PStr('op', 'stop'), d.PStr('session', ''))\
            .set_description("Lauf beenden (leer = alle laufenden)").r()
        self.define_sig(d.PStr('op', 'state'), d.PStr('session', ''), d.PInt('traj_from', 0), d.PInt('pts_from', 0))\
            .set_description("Status, aktuelle Pose, neue Trajektorienpunkte ab traj_from und Kartenpunkte ab pts_from")\
            .r(success='bool', status='str', pose='dict', traj='list', points='list')
        self.define_sig(d.PStr('op', 'pose'), d.PStr('session', ''))\
            .set_description("Letzte Kamerapose (leer = neuester Lauf)").r(success='bool', pose='dict')
        self.define_sig(d.PStr('op', 'sessions')).set_description("Läufe (aktuell und im Arbeitsverzeichnis)")\
            .r(sessions='list')
        self.define_sig(d.PStr('op', 'upload'), d.Param('file_data'), d.PStr('name', 'video.mp4'))\
            .set_description("Video (Base64) ins Arbeitsverzeichnis, liefert path für start").r(success='bool', path='str')
        self.define_sig(d.PStr('op', 'export'), d.PStr('session', ''), d.PStr('format', 'ply'))\
            .set_description("Karte als ply, Trajektorie als csv oder alles als json").r(success='bool', content='str')
        self.define_sig(d.PStr('op', 'drive'), d.PStr('session', ''), d.PInt('points', 1))\
            .set_description("Drive Radar: ist der Weg vor der Kamera frei? count = Punkte im Weg, points = "
                             "[seitlich, vorwärts, Höhe, im Weg] im Ausschnitt")\
            .r(success='bool', free='bool', count='int', visible='int', points='list')
        self.define_sig(d.PStr('op', 'settings'))\
            .set_description("Einstellungen lesen; mit fov, width, backend, event_every, push_timeout, drive_angle, "
                             "drive_range, drive_check, drive_lower, drive_up_angle, drive_threshold, drive_events "
                             "ändern und speichern").r(success='bool', settings='dict')
        self.define_sig(d.PStr('op', 'clear_tmp'))\
            .set_description("Tmp-Ordner leeren: beendete Läufe und Uploads löschen, laufende bleiben")\
            .r(success='bool', removed='int', freed='int')
        self.define_sig(d.PStr('op', 'clean_files'))\
            .set_description("Alles aufräumen: Läufe stoppen, Sessions, Uploads, ORB-Vokabular und Einstellungen "
                             "im Arbeitsverzeichnis löschen").r(success='bool', removed='list', freed='int')

    # Hilfen

    def ensure_dir(self, *parts):
        path = os.path.join(self.work_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def stella_executable(self):
        if self.stella_bin and os.path.isfile(self.stella_bin):
            return self.stella_bin
        return shutil.which(self.stella_bin or 'run_video_slam')

    def stella_vocabulary(self, log=None):
        if self.stella_vocab:
            if not os.path.isfile(self.stella_vocab):
                raise ValueError(f'Vokabular nicht gefunden: {self.stella_vocab}')
            return self.stella_vocab
        with self._vocab_lock:
            path = os.path.join(self.ensure_dir(), 'orb_vocab.fbow')
            if not os.path.isfile(path):
                if log is not None:
                    log.append(f'Lade ORB-Vokabular von {STELLA_VOCAB_URL}')
                urllib.request.urlretrieve(STELLA_VOCAB_URL, path + '.part')
                os.replace(path + '.part', path)
            return path

    def choose_backend(self, backend, source):
        backend = (backend or self.backend or 'auto').lower()
        is_stream = bool(STREAM_RE.match(source)) or source.isdigit()
        if backend == 'auto':
            return 'stella' if not is_stream and self.stella_executable() else 'opencv'
        if backend == 'stella' and is_stream:
            raise ValueError('stella_vslam verarbeitet nur Videodateien, für Streams backend=opencv verwenden')
        if backend not in ('opencv', 'stella'):
            raise ValueError(f'Unbekanntes Backend {backend!r}')
        return backend

    def get_session(self, session_id='') -> SlamSession:
        if not session_id:
            if not self.sessions:
                raise ValueError('Kein Lauf vorhanden')
            return max(self.sessions.values(), key=lambda s: s.started)
        if session_id in self.sessions:
            return self.sessions[session_id]
        if not re.fullmatch(r'[\w-]+', str(session_id)):
            raise ValueError(f'Ungültige session {session_id!r}')
        folder = os.path.join(self.work_dir, 'sessions', session_id)
        if os.path.isfile(os.path.join(folder, 'session.json')):
            s = SlamSession.load(self, folder)
            self.sessions[session_id] = s
            return s
        raise ValueError(f'Unbekannte session {session_id!r}')

    # Operationen

    PARAM_KEYS = ['fov', 'width', 'frame_skip', 'realtime', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3',
                  'max_features', 'kf_interval', 'min_angle', 'quality_level', 'min_distance']

    def execute_start(self, data):
        if cv2 is None:
            return self.fail('OpenCV fehlt: pip install opencv-python-headless numpy')
        source = str(data.get('source', '')).strip()
        if not source:
            return self.fail('source fehlt (Videopfad, Stream-URL oder Webcam-Index)')
        try:
            backend = self.choose_backend(data.get('backend'), source)
        except ValueError as e:
            return self.fail(str(e))
        s = self.create_session(source, backend, data)
        return self.success(session=s.id, backend=backend, folder=s.folder)

    def create_session(self, source, backend, data, push_source=None) -> SlamSession:
        params = {'fov': self.fov, 'width': self.width, 'frame_skip': 1, 'realtime': False,
                  'quality_level': self.quality_level, 'min_distance': self.min_distance}
        for key in self.PARAM_KEYS:
            if key in data and str(data[key]).strip() != '':
                params[key] = data[key]
        params['realtime'] = str(params.get('realtime', 0)) in ('1', 'true', 'True', 'on')
        for key in ('fov', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'min_angle', 'quality_level'):
            if key in params:
                params[key] = float(params[key])
        for key in ('width', 'frame_skip', 'max_features', 'kf_interval', 'min_distance'):
            if key in params:
                params[key] = int(float(params[key]))
        session_id = time.strftime('%Y%m%d-%H%M%S-') + u.guid()[:6]
        folder = self.ensure_dir('sessions', session_id)
        s = SlamSession(self, session_id, folder, source, backend, params)
        s.push_source = push_source
        self.sessions[session_id] = s
        s.start()
        self.info(f'SLAM {session_id}: {backend}, {source}')
        return s

    # Stream vom Handy (QR-Code: nx://stream?media=video&to=<url mit op=start_from_stream>)

    def push_session(self, session_id='', create=True) -> SlamSession:
        """ Laufende Session für gesendete Bilder, sonst eine neue """
        if session_id:
            s = self.get_session(session_id)
            if s.push_source is None or not s.is_running():
                raise ValueError(f'Session {session_id} nimmt keine Bilder an')
            return s
        running = [s for s in self.sessions.values() if s.push_source is not None and s.is_running()]
        if running:
            return max(running, key=lambda s: s.started)
        if not create:
            raise ValueError('Keine Session für gesendete Bilder')
        return self.create_session('push', 'opencv', {}, PushSource(timeout=self.push_timeout))

    def push_frames(self, frames, session_id=''):
        s = self.push_session(session_id)
        n = 0
        for frame in frames:
            if frame is not None:
                s.push_source.push(frame)
                n += 1
        return s, n

    def execute_start_from_stream(self, data):
        """
        Ziel des QR-Codes. Der Client schickt entweder
         - die URL seines Streams (source, stream oder url), dann liest der Runner den Stream, oder
         - Bilder (image, frame oder file_data als Base64-JPEG), wiederholt aufgerufen, oder
         - nichts, dann wird eine Session angelegt und push_url geliefert, an die der Client JPEGs bzw.
           einen MJPEG-Strom per POST schickt.
        """
        if cv2 is None:
            return self.fail('OpenCV fehlt: pip install opencv-python-headless numpy')
        stream = next((str(data[k]).strip() for k in ('source', 'stream', 'url') if str(data.get(k, '')).strip()), '')
        image = next((data[k] for k in ('image', 'frame', 'file_data') if data.get(k)), None)
        try:
            if stream and image is None:
                backend = self.choose_backend(data.get('backend') or 'opencv', stream)
                s = self.create_session(stream, backend, data)
                return self.success(session=s.id, backend=backend, mode='pull')
            if image is not None:
                if isinstance(image, str) and image.startswith('data:'):
                    image = image.split(',', 1)[1]
                frame = decode_image(base64.b64decode(image))
                if frame is None:
                    return self.fail('Bild kann nicht gelesen werden')
                s, _ = self.push_frames([frame], data.get('session', ''))
                return self.success(session=s.id, frames=s.push_source.frames, mode='push')
            s = self.push_session(data.get('session', ''))
            return self.success(session=s.id, mode='push', push_url=self.route('push') + '&session=' + s.id)
        except ValueError as e:
            return self.fail(str(e))

    def execute_lan_ip(self, data={}):
        """ IP im lokalen Netz, für den QR-Code, wenn die UI über localhost geöffnet ist """
        import socket
        ip = '127.0.0.1'
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(('10.255.255.255', 1))  # UDP: es wird nichts gesendet
                ip = sock.getsockname()[0]
        except OSError:
            pass
        return self.success(ip=ip)

    def execute_stop(self, data):
        session_id = data.get('session', '')
        targets = [self.get_session(session_id)] if session_id else \
            [s for s in self.sessions.values() if s.is_running()]
        for s in targets:
            s.stop()
        return self.success('Gestoppt', sessions=[s.id for s in targets])

    def execute_state(self, data):
        try:
            s = self.get_session(data.get('session', ''))
        except ValueError as e:
            return self.fail(str(e))
        return self.success(**s.state(int(data.get('traj_from', 0) or 0), int(data.get('pts_from', 0) or 0),
                                      int(data.get('max_points', 5000) or 5000)))

    def execute_pose(self, data):
        try:
            s = self.get_session(data.get('session', ''))
        except ValueError as e:
            return self.fail(str(e))
        with s.lock:
            pose = s.poses[-1] if s.poses else None
        return self.success(session=s.id, status=s.status, tracking=s.tracking, pose=pose)

    def execute_sessions(self, data={}):
        res = {s.id: s.summary() for s in self.sessions.values()}
        folder = os.path.join(self.work_dir, 'sessions')
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                path = os.path.join(folder, name, 'session.json')
                if name not in res and os.path.isfile(path):
                    try:
                        with open(path) as f:
                            res[name] = json.load(f)
                    except (OSError, ValueError):
                        pass
        items = sorted(res.values(), key=lambda x: x.get('started') or 0, reverse=True)
        return self.success(sessions=items)

    def store_upload(self, name, write):
        ext = os.path.splitext(name or '')[1].lower()
        if not re.fullmatch(r'\.\w{1,5}', ext):
            ext = '.mp4'
        path = os.path.join(self.ensure_dir('uploads'), u.guid() + ext)
        write(path)
        return self.success(path=path)

    def execute_upload(self, data):
        def write(path):
            with open(path, 'wb') as f:
                f.write(base64.b64decode(data['file_data']))
        return self.store_upload(data.get('name', 'video.mp4'), write)

    def export(self, s: SlamSession, fmt):
        with s.lock:
            poses = list(s.poses)
            points = list(s.points)
        if fmt == 'csv':
            return trajectory_csv(poses), 'text/csv'
        if fmt == 'json':
            return json.dumps({**s.summary(), 'trajectory': poses, 'points': points}), 'application/json'
        return points_ply(points), 'text/plain'

    def execute_export(self, data):
        try:
            s = self.get_session(data.get('session', ''))
        except ValueError as e:
            return self.fail(str(e))
        content, _ = self.export(s, data.get('format', 'ply'))
        return self.success(content=content)

    # Web

    # -- Einstellungen (work_dir/settings.json, überschreiben die Konstruktor-Werte) --

    def settings_path(self):
        return os.path.join(self.work_dir, 'settings.json')

    def load_settings(self):
        try:
            with open(self.settings_path()) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self.apply_settings(data)

    def apply_settings(self, data):
        """
        Übernimmt bekannte Werte aus data (allgemein und drive_<key>), unbekannte werden ignoriert.
        Erst werden alle Werte geprüft, bei einem ungültigen Wert (ValueError) bleibt alles unverändert.
        """
        general = {}
        for key, typ in GENERAL_SETTINGS.items():
            if key in data and str(data[key]).strip() != '':
                general[key] = typ(float(data[key])) if typ is int else typ(data[key])
        drive_in = data.get('drive', {}) if isinstance(data.get('drive'), dict) else {}
        drive_in = {**drive_in, **{k[6:]: v for k, v in data.items() if k.startswith('drive_')}}
        drive = {}
        for key, default in DRIVE_DEFAULTS.items():
            if key in drive_in and str(drive_in[key]).strip() != '':
                value = drive_in[key]
                if isinstance(default, bool):
                    drive[key] = str(value).lower() in ('1', 'true', 'on', 'yes')
                else:
                    drive[key] = type(default)(float(value))
        if 'quality_level' in general and not 0 < general['quality_level'] < 1:
            raise ValueError('qualityLevel muss zwischen 0 und 1 liegen')
        if 'min_distance' in general and general['min_distance'] < 1:
            raise ValueError('minDistance muss mindestens 1 sein')
        for key, value in general.items():
            setattr(self, key, value)
        self.drive.update(drive)

    def settings(self):
        return {**{k: getattr(self, k) for k in GENERAL_SETTINGS}, 'drive': dict(self.drive)}

    def save_settings(self):
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.settings_path(), 'w') as f:
            json.dump(self.settings(), f, indent=2)

    def execute_settings(self, data={}):
        """ op=settings: lesen; mit Werten (fov, width, ..., drive_check, drive_threshold, ...) ändern und speichern """
        changes = {k: v for k, v in data.items() if k in GENERAL_SETTINGS or k.startswith('drive_') or k == 'drive'}
        if changes:
            try:
                self.apply_settings(changes)
            except (TypeError, ValueError) as e:
                return self.fail(f'Ungültiger Wert: {e}')
            if self.backend not in ('auto', 'opencv', 'stella'):
                self.backend = 'auto'
            self.save_settings()
        return self.success(settings=self.settings(), work_dir=self.work_dir, usage=self.tmp_usage())

    # -- Tmp-Ordner --

    @staticmethod
    def folder_size(path):
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    def tmp_usage(self):
        sessions = os.path.join(self.work_dir, 'sessions')
        uploads = os.path.join(self.work_dir, 'uploads')
        return {
            'sessions': len(os.listdir(sessions)) if os.path.isdir(sessions) else 0,
            'uploads': len(os.listdir(uploads)) if os.path.isdir(uploads) else 0,
            'bytes': self.folder_size(self.work_dir) if os.path.isdir(self.work_dir) else 0
        }

    def execute_clear_tmp(self, data={}):
        """
        op=clear_tmp: löscht beendete Läufe und Uploads im Arbeitsverzeichnis. Laufende Sessions, ihre Quellvideos,
        die Einstellungen und das ORB-Vokabular bleiben erhalten.
        """
        running = {s.id for s in self.sessions.values() if s.is_running()}
        keep_files = {os.path.abspath(s.source) for s in self.sessions.values()
                      if s.is_running() and s.source and os.path.isfile(s.source)}
        before = self.tmp_usage()['bytes']
        removed = 0
        sessions = os.path.join(self.work_dir, 'sessions')
        if os.path.isdir(sessions):
            for name in os.listdir(sessions):
                if name in running:
                    continue
                shutil.rmtree(os.path.join(sessions, name), ignore_errors=True)
                self.sessions.pop(name, None)
                removed += 1
        uploads = os.path.join(self.work_dir, 'uploads')
        if os.path.isdir(uploads):
            for name in os.listdir(uploads):
                path = os.path.join(uploads, name)
                if os.path.abspath(path) in keep_files:
                    continue
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        freed = before - self.tmp_usage()['bytes']
        self.info(f'Tmp geleert: {removed} Einträge, {freed} Bytes')
        return self.success(f'{removed} Einträge gelöscht, {freed / 1e6:.1f} MB frei, {len(running)} laufende behalten',
                            removed=removed, freed=freed, kept=sorted(running), usage=self.tmp_usage())

    # vom Runner im work_dir angelegte Einträge; nur diese löscht clean_files
    OWN_FILES = ('sessions', 'uploads', 'orb_vocab.fbow', 'settings.json')

    def clean_files(self, wait=10):
        """
        Räumt alles auf, was der Runner angelegt hat: stoppt laufende Sessions und löscht im Arbeitsverzeichnis
        sessions/, uploads/, das ORB-Vokabular und settings.json. work_dir selbst wird nur entfernt, wenn danach
        nichts anderes mehr darin liegt (es kann ein eigenes Verzeichnis sein). Die Einstellungen im Speicher bleiben
        bis zum Neustart aktiv.

          :param wait: Sekunden, die auf das Ende laufender Sessions gewartet wird
        """
        running = [s for s in self.sessions.values() if s.is_running()]
        for s in running:
            s.stop()
        deadline = time.time() + wait
        while any(s.is_running() for s in running) and time.time() < deadline:
            time.sleep(0.2)
        still = [s.id for s in running if s.is_running()]
        if still:
            return self.fail(f'Sessions laufen noch: {", ".join(still)}, nichts gelöscht', running=still)
        before = self.folder_size(self.work_dir) if os.path.isdir(self.work_dir) else 0
        removed = []
        for name in self.OWN_FILES:
            path = os.path.join(self.work_dir, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.lexists(path):
                os.remove(path)
            else:
                continue
            removed.append(path)
        self.sessions.clear()
        self._drive_free.clear()
        if os.path.isdir(self.work_dir) and not os.listdir(self.work_dir):
            os.rmdir(self.work_dir)
            removed.append(self.work_dir)
        freed = before - (self.folder_size(self.work_dir) if os.path.isdir(self.work_dir) else 0)
        self.info(f'Aufgeräumt: {len(removed)} Einträge, {freed} Bytes')
        return self.success(f'{len(removed)} Einträge gelöscht, {freed / 1e6:.1f} MB frei', removed=removed,
                            freed=freed)

    def execute_clean_files(self, data={}):
        """ op=clean_files: siehe clean_files """
        return self.clean_files()

    # -- Drive Radar --

    def drive_status(self, s: SlamSession, with_points=False, max_points=4000):
        """
        Prüft, ob der Weg vor der Kamera frei ist.
        Berücksichtigt werden Punkte im Kuchenausschnitt (Öffnungswinkel angle, Reichweite range) oberhalb von
        Kamerahöhe + lower und bis zum Sichtwinkel up_angle nach oben. Im Weg sind davon die Punkte bis zur
        Prüfdistanz check; frei ist der Weg bei höchstens threshold Punkten.
        """
        cfg = self.drive
        with s.lock:
            pose = s.poses[-1] if s.poses else None
        res = {'session': s.id, 'status': s.status, 'tracking': s.tracking, 'pose': pose, 'drive': dict(cfg),
               'free': None, 'count': 0, 'visible': 0}
        if pose is None or s.tracking != 'tracking' and s.status in ('starting', 'running'):
            res['reason'] = 'keine Pose' if pose is None else 'Tracking ' + s.tracking
            return res
        P = s.points_array()
        yaw = math.radians(pose['yaw'])
        dx, dy, dz = P[:, 0] - pose['x'], P[:, 1] - pose['y'], P[:, 2] - pose['z']
        fwd = dx * math.sin(yaw) + dz * math.cos(yaw)
        side = dx * math.cos(yaw) - dz * math.sin(yaw)
        up = -dy  # y zeigt nach unten
        dist = np.hypot(fwd, side)
        ang = np.degrees(np.arctan2(side, fwd))
        with np.errstate(invalid='ignore', divide='ignore'):
            elev = np.degrees(np.arctan2(up, dist))
        mask = (fwd > 0) & (np.abs(ang) <= cfg['angle'] / 2) & (dist <= cfg['range']) \
            & (up >= cfg['lower']) & (elev <= cfg['up_angle'])
        zone = mask & (dist <= cfg['check'])
        count = int(zone.sum())
        res.update({'free': count <= int(cfg['threshold']), 'count': count, 'visible': int(mask.sum())})
        if zone.any():
            res['nearest'] = round(float(dist[zone].min()), 3)
        if with_points:
            idx = np.where(mask)[0]
            if len(idx) > max_points:
                idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(int)]
            res['points'] = [[round(float(side[i]), 3), round(float(fwd[i]), 3), round(float(up[i]), 3),
                              1 if zone[i] else 0] for i in idx]
        return res

    def drive_check(self, s: SlamSession):
        """ Event slam_path, wenn sich frei/blockiert ändert """
        st = self.drive_status(s)
        if st['free'] is None:
            return
        if self._drive_free.get(s.id) != st['free']:
            self._drive_free[s.id] = st['free']
            self.emit_event('slam_path', session=s.id, free=st['free'], count=st['count'],
                            threshold=int(self.drive['threshold']), check=self.drive['check'],
                            nearest=st.get('nearest'))

    def execute_drive(self, data):
        """ op=drive: Weg frei?, Punkte im Ausschnitt (relativ: seitlich, vorwärts, Höhe, im Weg) """
        try:
            s = self.get_session(data.get('session', ''))
        except ValueError as e:
            return self.fail(str(e))
        with_points = str(data.get('points', 1)) not in ('0', 'false', 'False')
        return self.success(**self.drive_status(s, with_points))

    # Die Endpunkte laufen über den Dispatcher (/pysys/dispatcher?type=slam&slam=<name> -> page_<name>), weil ein
    # nachgeladener Runner nach dem ersten Request keine eigenen Flask-Routen mehr anlegen kann. Wenn der Runner
    # beim Start geladen wird, gibt es sie zusätzlich unter /nxml/slam/<name>.

    HTTP_ENDPOINTS = {'upload': ['POST'], 'frame': ['GET'], 'export': ['GET'], 'push': ['POST', 'PUT'],
                      'qr': ['GET']}

    def route(self, name):
        return f'/pysys/dispatcher?type={self.type}&{self.type}={name}'

    def pointcloud_page(self):
        """ Seite des PointCloud-Runners (3D-Ansicht der Karte), falls im Dispatcher vorhanden, sonst '' """
        dispatcher = self.getParentClass(r.LazyDispatcher)
        if self.pointcloud_type and dispatcher is not None and dispatcher.support_type(self.pointcloud_type):
            return f'/pysys/dispatcher?type={self.pointcloud_type}'
        return ''

    def rest_route(self, name):
        return f'/nxml/{self.type}/{name}'

    def page_upload(self, params={}):
        return self.http_upload()

    def page_frame(self, params={}):
        return self.http_frame()

    def page_export(self, params={}):
        return self.http_export()

    def page_push(self, params={}):
        return self.http_push()

    def page_qr(self, params={}):
        return self.http_qr()

    def setupRestApp(self, app):
        for name, methods in self.HTTP_ENDPOINTS.items():
            app.add_url_rule(self.rest_route(name), f'{self.type}_{name}', view_func=getattr(self, 'http_' + name),
                             methods=methods)

    def http_upload(self):
        from flask import request
        f = request.files.get('file')
        if f is None:
            return self.fail('Keine Datei (Feld file)')
        return self.store_upload(f.filename, f.save)

    def http_frame(self):
        from flask import request, Response
        try:
            s = self.get_session(request.args.get('session', ''))
        except ValueError as e:
            return Response(str(e), status=404)
        if not s.jpeg:
            return Response('no frame', status=404)
        return Response(s.jpeg, mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

    def http_export(self):
        from flask import request, Response
        try:
            s = self.get_session(request.args.get('session', ''))
        except ValueError as e:
            return Response(str(e), status=404)
        fmt = request.args.get('format', 'ply')
        fmt = fmt if fmt in ('ply', 'csv', 'json') else 'ply'
        content, mime = self.export(s, fmt)
        name = f'slam_{s.id}_{"trajectory" if fmt == "csv" else "map"}.{fmt}'
        return Response(content, mimetype=mime, headers={'Content-Disposition': f'attachment; filename="{name}"'})

    def http_push(self):
        """
        Bilder vom Client: einzelnes JPEG (Formularfeld file/image/frame) oder ein fortlaufender
        MJPEG-Strom im Body (multipart/x-mixed-replace oder aneinandergehängte JPEGs, auch chunked)
        """
        from flask import request
        if cv2 is None:
            return self.fail('OpenCV fehlt')
        session_id = request.args.get('session', '')
        try:
            files = [f for f in (request.files.get(k) for k in ('file', 'image', 'frame')) if f is not None]
            if files:
                s, n = self.push_frames([decode_image(f.read()) for f in files], session_id)
            else:
                s = self.push_session(session_id)
                n = 0
                for frame in iter_jpeg_frames(request.stream.read, lambda: not s.is_running()):
                    s.push_source.push(frame)
                    n += 1
        except ValueError as e:
            return self.fail(str(e))
        return self.success(session=s.id, frames=n)

    def http_qr(self):
        from flask import request, Response
        text = request.args.get('text', '')
        try:
            import qrcode
            import qrcode.image.svg
        except ImportError:
            return Response('pip install qrcode', status=501)
        img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return Response(buf.getvalue(), mimetype='image/svg+xml', headers={'Cache-Control': 'no-store'})

    # -- Seiten: Main (index), Drive Radar (drive), Einstellungen (settings) --

    PAGES = [('index', 'Main'), ('drive', 'Drive Radar'), ('settings', 'Einstellungen')]

    def part_nav(self, p: b.Page, active='index'):
        # in der UI v2 werden die Seiten im Inhaltsbereich gewechselt statt die UI zu verlassen
        from nxml.webui import part_nav
        part_nav(self, p, self.PAGES, active, css_class='slam-nav')

    def script(self, p: b.Page, name, cfg, js):
        cfg_json = json.dumps(cfg).replace('</', '<\\/')
        # Block statt globaler Konstante: die UI v2 führt die Skripte bei jedem Öffnen erneut aus
        p('<script>{ const ' + name + ' = ' + cfg_json + ';\n' + js + '\n}</script>')

    def part_index(self, p: b.Page, params={}):
        self.part_nav(p, 'index')
        cfg = {
            'type': self.type,
            'routes': {name: self.route(name) for name in ['upload', 'frame', 'export', 'qr']},
            'backend': self.backend, 'width': self.width, 'fov': self.fov,
            'stella': self.stella_executable() is not None,
            'work_dir': self.work_dir,
            'pointcloud': self.pointcloud_page()
        }
        p(f'<style>{SLAM_CSS}</style>')
        p(SLAM_HTML)
        self.script(p, 'SLAM_CFG', cfg, SLAM_JS)

    def part_drive(self, p: b.Page, params={}):
        self.part_nav(p, 'drive')
        p(f'<style>{SLAM_CSS}{DRIVE_CSS}</style>')
        p(DRIVE_HTML)
        self.script(p, 'DRIVE_CFG', {'type': self.type, 'drive': self.drive, 'settings': self.dispatcher_link('settings'),
                                     'session': str(params.get('session', ''))}, DRIVE_JS)

    def part_settings(self, p: b.Page, params={}):
        self.part_nav(p, 'settings')
        p(f'<style>{SLAM_CSS}</style>')
        p(SETTINGS_HTML)
        self.script(p, 'SET_CFG', {'type': self.type}, SETTINGS_JS)


SLAM_CSS = """
.slam { max-width: 1400px; }
nav.slam-nav { margin-bottom: 12px; }
.slam-grid { display: grid; grid-template-columns: minmax(300px, 1fr) minmax(320px, 1fr); gap: 16px; }
@media (max-width: 900px) { .slam-grid { grid-template-columns: 1fr; } }
.slam-card { border: 1px solid #8885; border-radius: 8px; padding: 10px 12px; min-width: 0; }
.slam-card h3 { margin: 0 0 8px; font-size: 1em; }
.slam-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
.slam-row label { min-width: 90px; }
.slam input[type=number] { width: 80px; }
.slam #slam-source { flex: 1; min-width: 220px; }
.slam-frame { width: 100%; background: #111; border-radius: 6px; min-height: 120px; display: block; }
.slam-badge { display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: .85em; font-weight: bold;
  background: #8884; }
.slam-badge.tracking { background: #2a7; color: #fff; }
.slam-badge.lost { background: #c33; color: #fff; }
.slam-badge.init { background: #d90; color: #000; }
.slam-badge.running { background: #27c; color: #fff; }
.slam-badge.error { background: #c33; color: #fff; }
.slam-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 4px 12px;
  font-variant-numeric: tabular-nums; }
.slam-stats div span { opacity: .65; margin-right: 4px; }
.slam-canvas { width: 100%; display: block; border-radius: 6px; touch-action: none; }
#slam-radar { aspect-ratio: 1 / 1; max-width: 460px; margin: 0 auto; }
#slam-map { height: 460px; background: #0f1418; cursor: grab; }
.slam-log { font-size: .8em; max-height: 140px; overflow: auto; white-space: pre-wrap; opacity: .8; margin: 6px 0 0; }
.slam-msg.error { color: #c33; }
.slam-dialog { border: 1px solid #8886; border-radius: 10px; padding: 16px 18px; max-width: min(520px, 92vw);
  width: 520px; color: #e8ecef; background: #1d2329; }
.slam-dialog input, .slam-dialog select { color: #111; background: #fff; }
.slam-dialog code { color: #cfe8ff; }
.slam-dialog::backdrop { background: #0008; }
.slam-qr-img { display: flex; justify-content: center; margin: 12px 0; }
.slam-qr-img img { width: min(300px, 70vw); aspect-ratio: 1 / 1; background: #fff; border-radius: 6px; }
.slam-progress { height: 6px; background: #8884; border-radius: 3px; overflow: hidden; }
.slam-progress div { height: 100%; width: 0; background: #2a7; }
.slam-legend { font-size: .8em; opacity: .75; }
"""

SLAM_HTML = """
<div class="slam">
  <div class="slam-card">
    <div class="slam-row">
      <label for="slam-source">Quelle</label>
      <input type="text" id="slam-source" placeholder="http://192.168.0.20:8080/video, /data/video.mp4 oder 0 (Webcam)" />
      <button id="slam-start">Start</button><button id="slam-stop" disabled>Stop</button>
    </div>
    <div class="slam-row">
      <label></label><button id="slam-qr-btn" title="QR-Code für die nx-App: Handy streamt die Kamera hierher">QR</button>
    </div>
    <dialog class="slam-dialog" id="slam-qr">
      <h3 style="margin-top: 0">Kamera vom Handy streamen</h3>
      <div>Mit der nx-App scannen, das Handy schickt dann sein Kamerabild an diesen Runner.</div>
      <div class="slam-qr-img"><img id="slam-qr-img" alt="QR-Code" /></div>
      <div class="slam-row"><label for="slam-qr-to">Ziel</label><input type="text" id="slam-qr-to" style="flex: 1" /></div>
      <div class="slam-row"><label for="slam-qr-media">Medium</label>
        <select id="slam-qr-media"><option value="video">video</option><option value="image">image</option></select></div>
      <div class="slam-row"><code id="slam-qr-text" style="word-break: break-all; font-size: .85em"></code></div>
      <div class="slam-legend" id="slam-qr-hint"></div>
      <div class="slam-row" style="justify-content: flex-end">
        <button id="slam-qr-copy">Link kopieren</button><button id="slam-qr-close">Schließen</button></div>
    </dialog>
    <div class="slam-row">
      <label>Video</label><input type="file" id="slam-file" accept="video/*" /><button id="slam-upload">Hochladen</button>
      <div class="slam-progress" style="width: 160px"><div id="slam-upload-progress"></div></div>
    </div>
    <div class="slam-row">
      <label for="slam-backend">Backend</label>
      <select id="slam-backend"><option value="auto">auto</option><option value="opencv">opencv (live)</option>
        <option value="stella">stella_vslam (nur Video)</option></select>
      <label for="slam-fov" style="min-width: 0">FOV°</label><input type="number" id="slam-fov" min="20" max="170" step="1" />
      <label for="slam-width" style="min-width: 0">Breite</label><input type="number" id="slam-width" min="160" step="16" />
      <label for="slam-skip" style="min-width: 0">Frame-Skip</label><input type="number" id="slam-skip" min="1" value="1" />
      <label style="min-width: 0"><input type="checkbox" id="slam-realtime" /> Video in Echtzeit</label>
      <details><summary>Kalibrierung</summary>
        <div class="slam-row">fx <input type="number" id="slam-fx" step="any" /> fy <input type="number" id="slam-fy" step="any" />
          cx <input type="number" id="slam-cx" step="any" /> cy <input type="number" id="slam-cy" step="any" /></div>
        <div class="slam-row">k1 <input type="number" id="slam-k1" step="any" /> k2 <input type="number" id="slam-k2" step="any" />
          p1 <input type="number" id="slam-p1" step="any" /> p2 <input type="number" id="slam-p2" step="any" />
          k3 <input type="number" id="slam-k3" step="any" /></div>
        <div class="slam-legend">Werte für die Originalauflösung, ohne fx wird aus dem FOV geschätzt.</div>
      </details>
    </div>
    <div class="slam-row">
      <label for="slam-session">Lauf</label><select id="slam-session" style="min-width: 260px"></select>
      <button id="slam-refresh">Aktualisieren</button>
      <span id="slam-exports" style="display: none">Export: <a id="slam-ply" href="#">Karte (PLY)</a>
        <a id="slam-csv" href="#">Trajektorie (CSV)</a> <a id="slam-json" href="#">JSON</a>
        <a id="slam-3d" href="#" target="_blank" style="display: none; margin-left: 8px; font-weight: bold">3D-Ansicht</a></span>
    </div>
    <div class="slam-msg" id="slam-msg"></div>
  </div>

  <div class="slam-grid" style="margin-top: 16px">
    <div class="slam-card">
      <h3>Kamera <span class="slam-badge" id="slam-status">-</span> <span class="slam-badge" id="slam-tracking">-</span></h3>
      <img class="slam-frame" id="slam-frame" alt="" />
      <div class="slam-progress" style="margin-top: 6px"><div id="slam-progress"></div></div>
      <div class="slam-stats" style="margin-top: 8px">
        <div><span>Bild</span><b id="slam-v-frame">-</b></div>
        <div><span>FPS</span><b id="slam-v-fps">-</b></div>
        <div><span>Punkte</span><b id="slam-v-points">-</b></div>
        <div><span>Keyframes</span><b id="slam-v-kf">-</b></div>
        <div><span>x</span><b id="slam-v-x">-</b></div>
        <div><span>y</span><b id="slam-v-y">-</b></div>
        <div><span>z</span><b id="slam-v-z">-</b></div>
        <div><span>Weg</span><b id="slam-v-dist">-</b></div>
        <div><span>Yaw</span><b id="slam-v-yaw">-</b></div>
        <div><span>Pitch</span><b id="slam-v-pitch">-</b></div>
        <div><span>Roll</span><b id="slam-v-roll">-</b></div>
        <div><span>t</span><b id="slam-v-t">-</b></div>
      </div>
      <pre class="slam-log" id="slam-log"></pre>
    </div>
    <div class="slam-card">
      <h3>Radar</h3>
      <canvas class="slam-canvas" id="slam-radar"></canvas>
      <div class="slam-row slam-legend">
        <label for="slam-range" style="min-width: 0">Reichweite</label>
        <select id="slam-range"><option value="0">auto</option><option value="0.5">0.5</option><option value="1">1</option>
          <option value="2">2</option><option value="5">5</option><option value="10">10</option><option value="20">20</option>
          <option value="50">50</option></select>
        <label style="min-width: 0"><input type="checkbox" id="slam-anim" checked /> Animation</label>
      </div>
      <div class="slam-row slam-legend">
        <label style="min-width: 0" title="Höhe relativ zur Kamera, oben positiv, leer = keine Grenze">Höhe von</label>
        <input type="number" id="slam-hmin" step="any" placeholder="-∞" />
        <label for="slam-hmax" style="min-width: 0">bis</label>
        <input type="number" id="slam-hmax" step="any" placeholder="+∞" />
        <button id="slam-floor" title="Boden schätzen und darunter alles ausblenden">Boden ausblenden</button>
        <button id="slam-hreset">Alle</button>
        <label style="min-width: 0"><input type="checkbox" id="slam-hmap" checked /> auch Karte</label>
      </div>
      <div class="slam-legend" id="slam-hinfo"></div>
      <div class="slam-legend">Blickrichtung oben, Punkte der Karte von oben, Einheit = erste Bewegung.
        Höhe relativ zur Kamera (oben positiv), bezogen auf die Haltung der Kamera beim Start.</div>
    </div>
  </div>

  <div class="slam-card" style="margin-top: 16px">
    <h3>Karte</h3>
    <canvas class="slam-canvas" id="slam-map"></canvas>
    <div class="slam-row slam-legend">
      <label style="min-width: 0"><input type="checkbox" id="slam-follow" /> Kamera folgen</label>
      <label style="min-width: 0"><input type="checkbox" id="slam-colors" checked /> Punktfarben</label>
      <span>Draufsicht x/z, Linie = Bewegung der Kamera. Mausrad zoomt, Ziehen verschiebt, Doppelklick passt an.</span>
    </div>
  </div>
</div>
"""

SLAM_JS = r"""
(function() {
  const cfg = SLAM_CFG;
  const $ = (id) => document.getElementById(id);
  // UI v2 ersetzt den Inhalt beim Wechsel: dann Timer und Schleifen dieser Instanz beenden
  const rootEl = $('slam-source').closest('.slam');
  const alive = () => rootEl.isConnected;
  const TAU = Math.PI * 2;
  const S = { session: null, token: 0, status: '', poses: [], px: [], py: [], pz: [], pc: [], trajFrom: 0, ptsFrom: 0,
              pose: null, jpegFrame: -1, imgBusy: false, view: null, radar: null, range: 1, dist: 0 };
  const url = (path, params) => location.protocol + '//' + location.host + path +
    (params ? (path.indexOf('?') >= 0 ? '&' : '?') + new URLSearchParams(params) : '');
  const job = (data) => fetch(url('/'), { method: 'POST', body: new URLSearchParams({ type: cfg.type, ...data }) }).then((r) => r.json());
  const fmt = (v, n) => (v == null || isNaN(v)) ? '-' : (+v).toFixed(n == null ? 2 : n);
  function msg(text, error) { $('slam-msg').textContent = text || ''; $('slam-msg').classList.toggle('error', !!error); }

  $('slam-fov').value = cfg.fov;
  $('slam-width').value = cfg.width;
  $('slam-backend').value = cfg.backend;
  if (!cfg.stella) $('slam-backend').querySelector('option[value=stella]').textContent += ' - nicht installiert';
  try { $('slam-source').value = localStorage.getItem('nxml_slam_source') || ''; } catch (e) {}

  // -- QR-Code für die nx-App --
  const QR_DEFAULT_PORT = '7070';
  function defaultTarget(host) {
    const port = location.port || (location.protocol === 'https:' ? '' : QR_DEFAULT_PORT);
    return location.protocol + '//' + host + (port ? ':' + port : '') + '/?type=' + encodeURIComponent(cfg.type) + '&op=start_from_stream';
  }
  function qrText() {
    return 'nx://stream?media=' + encodeURIComponent($('slam-qr-media').value) + '&to=' + encodeURIComponent($('slam-qr-to').value.trim());
  }
  let qrTimer = null, qrServerOk = true, qrLib = null;
  function loadQrLib() {
    // Ersatz, wenn der Server keinen QR-Code liefert (z.B. ohne pip install qrcode)
    if (!qrLib) qrLib = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js';
      s.onload = () => window.QRCode ? resolve(window.QRCode) : reject(new Error('qrcodejs fehlt'));
      s.onerror = () => { qrLib = null; reject(new Error('qrcodejs nicht ladbar (kein Internet?)')); };
      document.head.appendChild(s);
    });
    return qrLib;
  }
  function renderQrLocal(text) {
    return loadQrLib().then((QR) => {
      const box = document.createElement('div');
      new QR(box, { text, width: 600, height: 600, correctLevel: QR.CorrectLevel.M });
      const c = box.querySelector('canvas');
      return new Promise((res) => setTimeout(() => res(c ? c.toDataURL('image/png') : box.querySelector('img').src), 30));
    }).then((src) => { $('slam-qr-img').src = src; });
  }
  function showQrError(reason) {
    $('slam-qr-hint').textContent = 'QR-Code vom Server nicht verfügbar (' + reason + ')' +
      ', erzeugt im Browser. Für den Server: pip install qrcode';
  }
  function updateQr() {
    clearTimeout(qrTimer);
    qrTimer = setTimeout(() => {
      const text = qrText();
      $('slam-qr-text').textContent = text;
      if (qrServerOk) $('slam-qr-img').src = url(cfg.routes.qr, { text });
      else renderQrLocal(text).catch((e) => $('slam-qr-hint').textContent = 'Kein QR-Code möglich: ' + e.message);
    }, 150);
  }
  $('slam-qr-img').addEventListener('error', () => {
    if (!qrServerOk) return;
    qrServerOk = false;
    const text = qrText();
    // Grund vom Server anzeigen, dann im Browser erzeugen
    fetch(url(cfg.routes.qr, { text })).then((r) => r.text().then((t) => r.status + ' ' +
      t.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 160)))
      .catch((e) => e.message).then((reason) => {
        showQrError(reason);
        return renderQrLocal(text);
      }).catch((e) => { $('slam-qr-hint').textContent += ' – Kein QR-Code möglich: ' + e.message; });
  });
  $('slam-qr-btn').addEventListener('click', async () => {
    let host = location.hostname;
    $('slam-qr-hint').textContent = '';
    if (/^(localhost|127\.|\[?::1\]?$)/.test(host)) {
      // über localhost geöffnet: das Handy braucht die IP im Netz
      try { const r = await job({ op: 'lan_ip' }); if (r.success) host = r.ip; } catch (e) {}
      $('slam-qr-hint').textContent = 'UI läuft über localhost, im QR-Code steht die Netz-IP ' + host + '.';
    }
    if (!$('slam-qr-to').value) $('slam-qr-to').value = defaultTarget(host);
    updateQr();
    $('slam-qr').showModal();
  });
  $('slam-qr-to').addEventListener('input', updateQr);
  $('slam-qr-media').addEventListener('change', updateQr);
  $('slam-qr-close').addEventListener('click', () => $('slam-qr').close());
  $('slam-qr').addEventListener('click', (e) => { if (e.target === $('slam-qr')) $('slam-qr').close(); });
  $('slam-qr-copy').addEventListener('click', () => {
    const text = qrText();
    (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(
      () => { $('slam-qr-hint').textContent = 'Kopiert.'; },
      () => { window.prompt('Link', text); });
  });
  // neue Session vom Handy automatisch anzeigen
  const qrPoll = setInterval(() => {
    if (!alive()) { clearInterval(qrPoll); return; }
    if (!$('slam-qr').open) return;
    job({ op: 'sessions' }).then((r) => {
      const s = (r.sessions || []).find((x) => (x.status === 'running' || x.status === 'starting') && x.session !== S.session);
      if (s && s.started * 1000 > Date.now() - 60000) {
        $('slam-qr').close();
        msg('Stream vom Handy: Lauf ' + s.session);
        loadSessions(s.session);
        select(s.session);
      }
    });
  }, 1500);

  // -- Steuerung --
  $('slam-start').addEventListener('click', () => {
    const data = { op: 'start', source: $('slam-source').value.trim(), backend: $('slam-backend').value,
      fov: $('slam-fov').value, width: $('slam-width').value, frame_skip: $('slam-skip').value,
      realtime: $('slam-realtime').checked ? 1 : 0 };
    ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'].forEach((k) => { const v = $('slam-' + k).value; if (v !== '') data[k] = v; });
    try { localStorage.setItem('nxml_slam_source', data.source); } catch (e) {}
    msg('Starte ...');
    job(data).then((r) => {
      if (!r.success) { msg(r.message || 'Fehler', true); return; }
      msg('Lauf ' + r.session + ' (' + r.backend + '), Ordner ' + r.folder);
      loadSessions(r.session);
      select(r.session);
    });
  });
  $('slam-stop').addEventListener('click', () => { if (S.session) job({ op: 'stop', session: S.session }); });
  $('slam-upload').addEventListener('click', () => {
    const file = $('slam-file').files[0];
    if (!file) { msg('Bitte ein Video wählen', true); return; }
    const form = new FormData();
    form.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url(cfg.routes.upload));
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) $('slam-upload-progress').style.width = (100 * e.loaded / e.total) + '%'; };
    xhr.onload = () => {
      try {
        const r = JSON.parse(xhr.responseText);
        if (!r.success) { msg(r.message, true); return; }
        $('slam-source').value = r.path;
        msg('Hochgeladen: ' + r.path + ', jetzt Start');
      } catch (e) { msg(xhr.responseText, true); }
    };
    xhr.onerror = () => msg('Upload fehlgeschlagen', true);
    xhr.send(form);
  });
  function loadSessions(selected) {
    return job({ op: 'sessions' }).then((r) => {
      const sel = $('slam-session');
      sel.innerHTML = '';
      (r.sessions || []).forEach((s) => {
        const o = document.createElement('option');
        o.value = s.session;
        o.textContent = s.session + ' · ' + s.status + ' · ' + s.backend + ' · ' + String(s.source).slice(-40);
        sel.appendChild(o);
      });
      if (selected) sel.value = selected;
      else if (!S.session && sel.options.length) select(sel.options[0].value);
    });
  }
  $('slam-session').addEventListener('change', (e) => select(e.target.value));
  $('slam-refresh').addEventListener('click', () => loadSessions(S.session));

  // -- Daten --
  function select(id) {
    S.session = id; S.token++;
    S.poses = []; S.px = []; S.py = []; S.pz = []; S.pc = []; S.trajFrom = 0; S.ptsFrom = 0; S.pose = null; S.jpegFrame = -1;
    S.view = null; S.dist = 0; S.radar = null;
    $('slam-exports').style.display = '';
    ['ply', 'csv', 'json'].forEach((f) => $('slam-' + f).href = url(cfg.routes.export, { session: id, format: f }));
    if (cfg.pointcloud) { $('slam-3d').href = url(cfg.pointcloud, { slam: id }); $('slam-3d').style.display = ''; }
    poll(S.token);
  }
  async function poll(token) {
    if (token !== S.token || !alive()) return;
    let more = false;
    try {
      const r = await job({ op: 'state', session: S.session, traj_from: S.trajFrom, pts_from: S.ptsFrom });
      if (token !== S.token) return;
      if (r.success) { apply(r); more = r.pts_next < r.n_points; } else msg(r.message, true);
    } catch (e) {}
    const live = S.status === 'running' || S.status === 'starting';
    setTimeout(() => poll(token), more ? 30 : (live ? 350 : 3000));
  }
  function apply(r) {
    const prevStatus = S.status;
    S.status = r.status;
    for (const t of r.traj) {
      const last = S.poses[S.poses.length - 1];
      if (last && last[6] === t[6]) S.dist += Math.hypot(t[2] - last[2], t[3] - last[3], t[4] - last[4]);
      S.poses.push(t);
    }
    S.trajFrom = r.traj_next;
    for (const p of r.points) { S.px.push(p[0]); S.py.push(p[1]); S.pz.push(p[2]); S.pc.push('rgb(' + p[3] + ',' + p[4] + ',' + p[5] + ')'); }
    S.ptsFrom = r.pts_next;
    S.pose = r.pose;
    const st = $('slam-status'); st.textContent = r.status; st.className = 'slam-badge ' + r.status;
    const tr = $('slam-tracking'); tr.textContent = r.tracking; tr.className = 'slam-badge ' + r.tracking;
    $('slam-stop').disabled = !(r.status === 'running' || r.status === 'starting');
    $('slam-v-frame').textContent = r.frame;
    $('slam-v-fps').textContent = fmt(r.fps, 1);
    $('slam-v-points').textContent = r.n_points;
    $('slam-v-kf').textContent = r.keyframes;
    const p = r.pose || {};
    $('slam-v-x').textContent = fmt(p.x); $('slam-v-y').textContent = fmt(p.y); $('slam-v-z').textContent = fmt(p.z);
    $('slam-v-yaw').textContent = fmt(p.yaw, 1) + '°'; $('slam-v-pitch').textContent = fmt(p.pitch, 1) + '°';
    $('slam-v-roll').textContent = fmt(p.roll, 1) + '°'; $('slam-v-t').textContent = fmt(p.t, 1) + ' s';
    $('slam-v-dist').textContent = fmt(S.dist);
    $('slam-progress').style.width = (r.progress == null ? (r.status === 'running' ? 100 : 0) : 100 * r.progress) + '%';
    $('slam-log').textContent = (r.message ? 'Fehler: ' + r.message + '\n' : '') + (r.log || []).join('\n');
    if (r.jpeg_frame >= 0 && r.jpeg_frame !== S.jpegFrame && !S.imgBusy) {
      S.imgBusy = true; S.jpegFrame = r.jpeg_frame;
      $('slam-frame').src = url(cfg.routes.frame, { session: S.session, f: r.jpeg_frame });
    }
    if (prevStatus && prevStatus !== r.status) loadSessions(S.session);
    rebuildRadar();
    drawMap();
  }
  $('slam-frame').addEventListener('load', () => S.imgBusy = false);
  $('slam-frame').addEventListener('error', () => S.imgBusy = false);

  // -- Canvas --
  function canvas(c, h) {
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, hh = h || c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(hh * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: hh };
  }
  function niceStep(v) {
    const e = Math.pow(10, Math.floor(Math.log10(v))), f = v / e;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * e;
  }
  function percentile(arr, q) {
    if (!arr.length) return 0;
    const a = arr.slice().sort((x, y) => x - y);
    return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))];
  }

  // -- Höhenfilter: Höhe relativ zur Kamera, oben positiv (y zeigt nach unten) --
  const RADAR_KEY = 'nxml_slam_radar';
  function heightLimits() {
    // leere oder halb getippte Werte ("-", "1e") gelten als keine Grenze
    const num = (v, def) => { const x = parseFloat(v); return isFinite(x) ? x : def; };
    return { lo: num($('slam-hmin').value, -Infinity), hi: num($('slam-hmax').value, Infinity) };
  }
  function camY() { return S.pose ? S.pose.y : 0; }
  function heightOk(i, lim, cy) { const hgt = cy - S.py[i]; return hgt >= lim.lo && hgt <= lim.hi; }
  function heightInfo() {
    if (!S.py.length) { $('slam-hinfo').textContent = ''; return; }
    const cy = camY(), step = Math.max(1, Math.floor(S.py.length / 4000)), hs = [];
    for (let i = 0; i < S.py.length; i += step) hs.push(cy - S.py[i]);
    const lim = heightLimits();
    let shown = 0;
    for (let i = 0; i < S.py.length; i++) if (heightOk(i, lim, cy)) shown++;
    $('slam-hinfo').textContent = 'Punkthöhen ' + fmt(percentile(hs, 0.02)) + ' bis ' + fmt(percentile(hs, 0.98)) +
      ' (Median ' + fmt(percentile(hs, 0.5)) + '), angezeigt ' + shown + ' von ' + S.py.length;
  }
  function saveRadarSettings() {
    try {
      localStorage.setItem(RADAR_KEY, JSON.stringify({ hmin: $('slam-hmin').value, hmax: $('slam-hmax').value,
        hmap: $('slam-hmap').checked, anim: $('slam-anim').checked, range: $('slam-range').value }));
    } catch (e) {}
  }
  try {
    const st = JSON.parse(localStorage.getItem(RADAR_KEY) || '{}');
    if (st.hmin != null) $('slam-hmin').value = st.hmin;
    if (st.hmax != null) $('slam-hmax').value = st.hmax;
    if (st.hmap != null) $('slam-hmap').checked = st.hmap;
    if (st.anim != null) $('slam-anim').checked = st.anim;
    if (st.range != null) $('slam-range').value = st.range;
  } catch (e) {}
  let filterTimer = null;
  function filterChanged() {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => { saveRadarSettings(); rebuildRadar(); drawMap(); }, 120);
  }
  ['slam-hmin', 'slam-hmax'].forEach((id) => $(id).addEventListener('input', filterChanged));
  $('slam-hmap').addEventListener('change', filterChanged);
  $('slam-hreset').addEventListener('click', () => { $('slam-hmin').value = ''; $('slam-hmax').value = ''; filterChanged(); });
  $('slam-floor').addEventListener('click', () => {
    // Boden = unterste dichte Punktschicht unter der Kamera, ausgeblendet wird alles bis knapp darüber
    const cy = camY(), below = [];
    for (let i = 0; i < S.py.length; i++) { const hgt = cy - S.py[i]; if (hgt < 0) below.push(hgt); }
    if (below.length < 20) { msg('Zu wenige Punkte unter der Kamera, um den Boden zu schätzen', true); return; }
    // dichteste Schicht im Histogramm der Höhen unter der Kamera (vereinzelte Ausreißer stören so nicht)
    const lo = percentile(below, 0.01), BINS = 60, width = (0 - lo) / BINS || 1, bins = new Array(BINS).fill(0);
    for (const hgt of below) { const k = Math.floor((hgt - lo) / width); if (k >= 0 && k < BINS) bins[k]++; }
    let best = 0;
    for (let k = 1; k < BINS; k++) if (bins[k] > bins[best]) best = k;
    const floor = lo + (best + 0.5) * width;
    // knapp über der Schicht abschneiden: halbe Lücke zur Kamera, höchstens 3 Histogrammstufen
    const cut = Math.min(floor + 3 * width, floor / 2);
    $('slam-hmin').value = +cut.toPrecision(3);
    msg('Boden bei ca. ' + fmt(floor) + ', ausgeblendet bis ' + $('slam-hmin').value);
    filterChanged();
  });

  // -- Radar --
  function rebuildRadar() {
    heightInfo();
    const p = S.pose;
    if (!p) { S.radar = null; requestRadar(); return; }
    const lim = heightLimits(), cy = p.y;
    const yaw = p.yaw * Math.PI / 180, s = Math.sin(yaw), c = Math.cos(yaw);
    let range = +$('slam-range').value;
    if (!range) {
      const step = Math.max(1, Math.floor(S.px.length / 4000)), d = [];
      for (let i = 0; i < S.px.length; i += step) if (heightOk(i, lim, cy)) d.push(Math.hypot(S.px[i] - p.x, S.pz[i] - p.z));
      const target = d.length ? Math.max(niceStep(percentile(d, 0.6) * 1.2), 0.05) : 1;
      S.range = target;
    } else S.range = range;
    const R = S.range, out = [];
    for (let i = 0; i < S.px.length; i++) {
      if (!heightOk(i, lim, cy)) continue;
      const dx = S.px[i] - p.x, dz = S.pz[i] - p.z;
      const f = dx * s + dz * c, rt = dx * c - dz * s;
      if (f * f + rt * rt <= R * R) out.push(rt / R, f / R, (Math.atan2(rt, f) + TAU) % TAU);
    }
    const trail = [];
    for (let i = Math.max(0, S.poses.length - 300); i < S.poses.length; i++) {
      const q = S.poses[i], dx = q[2] - p.x, dz = q[4] - p.z;
      trail.push((dx * c - dz * s) / R, (dx * s + dz * c) / R);
    }
    // das Radar zeichnet laufend: bei sehr vielen Punkten nur eine Auswahl
    let pts = out;
    if (out.length / 3 > MAX_RADAR_POINTS) {
      const k = Math.ceil(out.length / 3 / MAX_RADAR_POINTS);
      pts = [];
      for (let i = 0; i < out.length; i += 3 * k) pts.push(out[i], out[i + 1], out[i + 2]);
    }
    S.radar = { pts, trail, yaw };
    requestRadar();
  }
  // mit Animation zeichnet requestAnimationFrame laufend, ohne nur bei neuen Daten
  let radarLoop = false;
  let radarError = '';
  function radarFrame(ts) {
    if (!alive()) { radarLoop = false; return; }
    const anim = $('slam-anim').checked;
    // ein Fehler beim Zeichnen darf die Schleife nicht dauerhaft anhalten
    try {
      drawRadar(ts, anim);
      radarError = '';
    } catch (e) {
      if (String(e) !== radarError) { radarError = String(e); console.error('SLAM-Radar', e); }
    }
    radarLoop = anim;
    if (anim) requestAnimationFrame(radarFrame);
  }
  function requestRadar() { if (!radarLoop && alive()) { radarLoop = true; requestAnimationFrame(radarFrame); } }
  $('slam-anim').addEventListener('change', () => { saveRadarSettings(); requestRadar(); });
  const onResize = () => {
    if (!alive()) { window.removeEventListener('resize', onResize); return; }
    requestRadar(); drawMap();
  };
  window.addEventListener('resize', onResize);
  function drawRadar(ts, anim) {
    const c = $('slam-radar');
    const { ctx, w, h } = canvas(c, c.clientWidth);
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 22;
    ctx.clearRect(0, 0, w, h);
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
    g.addColorStop(0, '#0b2a17'); g.addColorStop(1, '#04140b');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
    ctx.strokeStyle = 'rgba(80,255,140,0.25)'; ctx.lineWidth = 1;
    ctx.font = '11px sans-serif'; ctx.fillStyle = 'rgba(120,255,160,0.7)';
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath(); ctx.arc(cx, cy, R * i / 4, 0, TAU); ctx.stroke();
      ctx.fillText(+(S.range * i / 4).toPrecision(3), cx + 4, cy - R * i / 4 + 12);
    }
    ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
    // Sichtfeld
    const fov = (+$('slam-fov').value || 70) * Math.PI / 360;
    ctx.fillStyle = 'rgba(80,255,140,0.07)';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, -Math.PI / 2 - fov, -Math.PI / 2 + fov); ctx.closePath(); ctx.fill();
    // Sweep, Winkel im Uhrzeigersinn ab oben
    const sweep = ((ts || 0) / 1000 * TAU / 3) % TAU;
    if (anim) {
      for (let i = 0; i < 24; i++) {
        const a0 = sweep - (i + 1) * 0.03, a1 = sweep - i * 0.03;
        ctx.fillStyle = 'rgba(80,255,140,' + (0.22 * (1 - i / 24)) + ')';
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, a0 - Math.PI / 2, a1 - Math.PI / 2); ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = 'rgba(140,255,180,0.9)';
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.sin(sweep), cy - R * Math.cos(sweep)); ctx.stroke();
    }
    const rd = S.radar;
    if (rd) {
      const pts = rd.pts;
      if (!anim) ctx.fillStyle = 'rgba(110,255,150,0.85)';
      for (let i = 0; i < pts.length; i += 3) {
        if (anim) {
          const delta = (sweep - pts[i + 2] + TAU) % TAU;
          const a = 0.18 + 0.82 * Math.max(0, 1 - delta / (TAU * 0.85));
          ctx.fillStyle = 'rgba(110,255,150,' + a.toFixed(2) + ')';
        }
        ctx.fillRect(cx + pts[i] * R - 1.2, cy - pts[i + 1] * R - 1.2, 2.4, 2.4);
      }
      ctx.strokeStyle = 'rgba(255,190,60,0.85)'; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let i = 0; i < rd.trail.length; i += 2) {
        const x = cx + rd.trail[i] * R, y = cy - rd.trail[i + 1] * R;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // Weltachsen am Rand: +z (Startblickrichtung) und +x
      const lbl = (txt, ang) => {
        ctx.fillStyle = 'rgba(255,255,255,0.75)'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(txt, cx + (R + 11) * Math.sin(ang), cy - (R + 11) * Math.cos(ang));
        ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      };
      lbl('Z', -rd.yaw); lbl('X', Math.PI / 2 - rd.yaw);
    }
    ctx.fillStyle = '#ffd34d';
    ctx.beginPath(); ctx.moveTo(cx, cy - 9); ctx.lineTo(cx - 6, cy + 6); ctx.lineTo(cx + 6, cy + 6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = 'rgba(120,255,160,0.85)';
    ctx.fillText(S.pose ? 'Yaw ' + fmt(S.pose.yaw, 0) + '°' : 'keine Pose', 6, 14);
  }
  requestRadar();
  $('slam-range').addEventListener('change', () => { saveRadarSettings(); rebuildRadar(); });

  // -- Karte --
  function autoView(w, h) {
    const xs = [], zs = [];
    const step = Math.max(1, Math.floor(S.px.length / 5000));
    for (let i = 0; i < S.px.length; i += step) { xs.push(S.px[i]); zs.push(S.pz[i]); }
    let x0 = percentile(xs, 0.03), x1 = percentile(xs, 0.97), z0 = percentile(zs, 0.03), z1 = percentile(zs, 0.97);
    if (!xs.length) { x0 = x1 = z0 = z1 = 0; }
    for (const q of S.poses) { x0 = Math.min(x0, q[2]); x1 = Math.max(x1, q[2]); z0 = Math.min(z0, q[4]); z1 = Math.max(z1, q[4]); }
    const sx = Math.max(x1 - x0, 0.5), sz = Math.max(z1 - z0, 0.5);
    return { cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, k: Math.min(w / sx, h / sz) * 0.88 };
  }
  function currentView(w, h) {
    const v = S.view || autoView(w, h);
    if ($('slam-follow').checked && S.pose) return { cx: S.pose.x, cz: S.pose.z, k: v.k };
    return v;
  }
  // höchstens ein Neuzeichnen pro Bild, egal wie oft Daten, Filter oder Maus es anstoßen
  let mapPending = false;
  function drawMap() {
    if (mapPending) return;
    mapPending = true;
    requestAnimationFrame(() => { mapPending = false; if (alive()) drawMapNow(); });
  }
  const MAX_MAP_POINTS = 200000, MAX_RADAR_POINTS = 25000;
  function drawMapNow() {
    const c = $('slam-map');
    const { ctx, w, h } = canvas(c);
    const v = currentView(w, h);
    const X = (x) => w / 2 + (x - v.cx) * v.k, Y = (z) => h / 2 - (z - v.cz) * v.k;
    ctx.fillStyle = '#0f1418'; ctx.fillRect(0, 0, w, h);
    // Raster
    const step = niceStep(70 / v.k);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1; ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '10px sans-serif';
    const gx0 = Math.floor((v.cx - w / 2 / v.k) / step) * step, gz0 = Math.floor((v.cz - h / 2 / v.k) / step) * step;
    ctx.beginPath();
    for (let x = gx0; X(x) < w; x += step) { ctx.moveTo(X(x), 0); ctx.lineTo(X(x), h); }
    for (let z = gz0; Y(z) > 0; z += step) { ctx.moveTo(0, Y(z)); ctx.lineTo(w, Y(z)); }
    ctx.stroke();
    ctx.fillText('Raster ' + +step.toPrecision(3), 6, h - 6);
    // Punkte
    const colors = $('slam-colors').checked;
    const filter = $('slam-hmap').checked, lim = heightLimits(), cy = camY();
    ctx.fillStyle = 'rgba(160,200,220,0.55)';
    const mstep = Math.max(1, Math.ceil(S.px.length / MAX_MAP_POINTS));
    for (let i = 0; i < S.px.length; i += mstep) {
      if (filter && !heightOk(i, lim, cy)) continue;
      const x = X(S.px[i]), y = Y(S.pz[i]);
      if (x < 0 || y < 0 || x > w || y > h) continue;
      if (colors) ctx.fillStyle = S.pc[i];
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }
    // Trajektorie
    ctx.strokeStyle = '#ffb02e'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath();
    let seg = null;
    for (const q of S.poses) {
      if (q[6] !== seg) { ctx.moveTo(X(q[2]), Y(q[4])); seg = q[6]; } else ctx.lineTo(X(q[2]), Y(q[4]));
    }
    ctx.stroke();
    if (S.poses.length) {
      const s0 = S.poses[0];
      ctx.fillStyle = '#4caf50'; ctx.beginPath(); ctx.arc(X(s0[2]), Y(s0[4]), 5, 0, TAU); ctx.fill();
    }
    if (S.pose) {
      const x = X(S.pose.x), y = Y(S.pose.z), a = S.pose.yaw * Math.PI / 180;
      const fov = (+$('slam-fov').value || 70) * Math.PI / 360, L = 40;
      ctx.fillStyle = 'rgba(255,211,77,0.15)';
      ctx.beginPath(); ctx.moveTo(x, y);
      ctx.lineTo(x + L * Math.sin(a - fov), y - L * Math.cos(a - fov));
      ctx.lineTo(x + L * Math.sin(a + fov), y - L * Math.cos(a + fov));
      ctx.closePath(); ctx.fill();
      ctx.save(); ctx.translate(x, y); ctx.rotate(a);
      ctx.fillStyle = '#ffd34d'; ctx.strokeStyle = '#000';
      ctx.beginPath(); ctx.moveTo(0, -10); ctx.lineTo(-7, 7); ctx.lineTo(0, 3); ctx.lineTo(7, 7); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.restore();
    }
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('x →   z ↑   ' + S.px.length + ' Punkte, ' + S.poses.length + ' Posen', 6, 14);
  }
  const map = $('slam-map');
  map.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = map.getBoundingClientRect(), w = map.clientWidth, h = map.clientHeight;
    const v = S.view || currentView(w, h);
    const mx = v.cx + (e.clientX - r.left - w / 2) / v.k, mz = v.cz - (e.clientY - r.top - h / 2) / v.k;
    const f = Math.exp(-e.deltaY * 0.0015);
    const k = v.k * f;
    S.view = { cx: mx - (mx - v.cx) / f, cz: mz - (mz - v.cz) / f, k };
    drawMap();
  }, { passive: false });
  let drag = null;
  map.addEventListener('pointerdown', (e) => {
    const v = S.view || currentView(map.clientWidth, map.clientHeight);
    drag = { x: e.clientX, y: e.clientY, v: { ...v } };
    map.setPointerCapture(e.pointerId); map.style.cursor = 'grabbing';
  });
  map.addEventListener('pointermove', (e) => {
    if (!drag) return;
    S.view = { cx: drag.v.cx - (e.clientX - drag.x) / drag.v.k, cz: drag.v.cz + (e.clientY - drag.y) / drag.v.k, k: drag.v.k };
    $('slam-follow').checked = false;
    drawMap();
  });
  map.addEventListener('pointerup', () => { drag = null; map.style.cursor = ''; });
  map.addEventListener('dblclick', () => { S.view = null; drawMap(); });
  ['slam-follow', 'slam-colors', 'slam-fov'].forEach((id) => $(id).addEventListener('change', drawMap));

  drawMap();
  loadSessions();
})();
"""


DRIVE_CSS = """
.drive-banner { font-size: 1.6em; font-weight: bold; padding: 14px 18px; border-radius: 10px; text-align: center;
  background: #8884; transition: background .2s; }
.drive-banner.free { background: #1f8f4e; color: #fff; }
.drive-banner.blocked { background: #c62828; color: #fff; }
.drive-banner small { display: block; font-size: .5em; font-weight: normal; opacity: .9; margin-top: 4px; }
#drive-canvas { width: 100%; height: 460px; background: #071109; border-radius: 8px; display: block; }
.drive-grid { display: grid; grid-template-columns: minmax(320px, 2fr) minmax(260px, 1fr); gap: 16px; margin-top: 14px; }
@media (max-width: 900px) { .drive-grid { grid-template-columns: 1fr; } }
.drive-form .slam-row label { min-width: 170px; }
"""

DRIVE_HTML = """
<div class="slam drive">
  <div class="slam-row">
    <label for="drive-session">Lauf</label><select id="drive-session" style="min-width: 300px"></select>
    <button id="drive-refresh">Aktualisieren</button>
  </div>
  <div class="drive-banner" id="drive-banner">-<small id="drive-sub"></small></div>
  <div class="drive-grid">
    <div class="slam-card">
      <canvas id="drive-canvas"></canvas>
      <div class="slam-legend">Blickrichtung oben. Rot = im Weg (bis zur Prüfdistanz), grün = sichtbar.
        Nicht angezeigt: Punkte unterhalb Kamerahöhe + Untergrenze und oberhalb des Sichtwinkels.</div>
    </div>
    <div class="slam-card drive-form">
      <h3>Weg-Prüfung</h3>
      <div class="slam-row"><label for="drive-threshold">erlaubte Punkte im Weg</label>
        <input type="number" id="drive-threshold" min="0" step="1" /></div>
      <div class="slam-row"><label for="drive-check">Prüfdistanz</label>
        <input type="number" id="drive-check" min="0" step="any" /></div>
      <div class="slam-row"><label for="drive-range">Reichweite</label>
        <input type="number" id="drive-range" min="0.01" step="any" /></div>
      <div class="slam-row"><label for="drive-angle">Öffnungswinkel °</label>
        <input type="number" id="drive-angle" min="5" max="180" step="1" /></div>
      <div class="slam-row"><label for="drive-lower">Untergrenze über Kamera</label>
        <input type="number" id="drive-lower" step="any" /></div>
      <div class="slam-row"><label for="drive-up">Sichtwinkel nach oben °</label>
        <input type="number" id="drive-up" min="0" max="90" step="1" /></div>
      <div class="slam-row"><button id="drive-save">Übernehmen</button><span class="slam-msg" id="drive-msg"></span></div>
      <div class="slam-stats" style="margin-top: 10px">
        <div><span>im Weg</span><b id="drive-v-count">-</b></div>
        <div><span>nächster</span><b id="drive-v-near">-</b></div>
        <div><span>sichtbar</span><b id="drive-v-visible">-</b></div>
        <div><span>Tracking</span><b id="drive-v-tracking">-</b></div>
        <div><span>Yaw</span><b id="drive-v-yaw">-</b></div>
        <div><span>Bild</span><b id="drive-v-frame">-</b></div>
      </div>
      <div class="slam-legend" style="margin-top: 8px">Einheiten wie die Karte (erste Bewegung = 1).
        Bei Änderung von frei/blockiert löst der Runner das Event <code>slam_path</code> aus.</div>
    </div>
  </div>
</div>
"""

DRIVE_JS = r"""
(function() {
  const cfg = DRIVE_CFG;
  const $ = (id) => document.getElementById(id);
  const rootEl = $('drive-canvas').closest('.slam');
  const alive = () => rootEl.isConnected;
  const url = (path, params) => location.protocol + '//' + location.host + path +
    (params ? (path.indexOf('?') >= 0 ? '&' : '?') + new URLSearchParams(params) : '');
  const job = (data) => fetch(url('/'), { method: 'POST', body: new URLSearchParams({ type: cfg.type, ...data }) }).then((r) => r.json());
  const fmt = (v, n) => (v == null || isNaN(v)) ? '-' : (+v).toFixed(n == null ? 2 : n);
  const FIELDS = { threshold: 'drive-threshold', check: 'drive-check', range: 'drive-range', angle: 'drive-angle',
                   lower: 'drive-lower', up_angle: 'drive-up' };
  const D = { session: cfg.session || '', drive: cfg.drive, last: null, token: 0 };
  function fillForm(drive) { Object.entries(FIELDS).forEach(([k, id]) => { if (document.activeElement !== $(id)) $(id).value = drive[k]; }); }
  fillForm(D.drive);

  $('drive-save').addEventListener('click', () => {
    const data = { op: 'settings' };
    Object.entries(FIELDS).forEach(([k, id]) => { if ($(id).value !== '') data['drive_' + k] = $(id).value; });
    job(data).then((r) => {
      $('drive-msg').textContent = r.success ? 'gespeichert' : (r.message || 'Fehler');
      if (r.success) { D.drive = r.settings.drive; fillForm(D.drive); }
      setTimeout(() => { if (alive()) $('drive-msg').textContent = ''; }, 2000);
    });
  });
  Object.values(FIELDS).forEach((id) => $(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') $('drive-save').click(); }));

  function loadSessions() {
    return job({ op: 'sessions' }).then((r) => {
      const sel = $('drive-session'), items = r.sessions || [];
      sel.innerHTML = '';
      items.forEach((s) => {
        const o = document.createElement('option');
        o.value = s.session; o.textContent = s.session + ' · ' + s.status + ' · ' + String(s.source).slice(-40);
        sel.appendChild(o);
      });
      if (!D.session) {
        const run = items.find((s) => s.status === 'running' || s.status === 'starting');
        D.session = run ? run.session : (items[0] ? items[0].session : '');
      }
      sel.value = D.session;
    });
  }
  $('drive-session').addEventListener('change', (e) => { D.session = e.target.value; D.token++; poll(D.token); });
  $('drive-refresh').addEventListener('click', loadSessions);

  async function poll(token) {
    if (!alive() || token !== D.token) return;
    let live = false;
    if (D.session) {
      try {
        const r = await job({ op: 'drive', session: D.session });
        if (!alive() || token !== D.token) return;
        if (r.success) { D.last = r; D.drive = r.drive; fillForm(r.drive); show(r); live = r.status === 'running' || r.status === 'starting'; }
      } catch (e) {}
    }
    setTimeout(() => poll(token), live ? 300 : 2000);
  }

  function show(r) {
    const b = $('drive-banner'), sub = $('drive-sub');
    b.className = 'drive-banner' + (r.free === true ? ' free' : r.free === false ? ' blocked' : '');
    b.firstChild.textContent = r.free === true ? 'Weg frei' : r.free === false ? 'Hindernis' : (r.reason || 'unbekannt');
    sub.textContent = r.free == null ? '' : r.count + ' Punkte im Weg (erlaubt ' + r.drive.threshold + ') bis ' + fmt(r.drive.check) +
      (r.nearest != null ? ', nächster bei ' + fmt(r.nearest) : '');
    $('drive-v-count').textContent = r.count;
    $('drive-v-near').textContent = fmt(r.nearest);
    $('drive-v-visible').textContent = r.visible;
    $('drive-v-tracking').textContent = r.tracking;
    $('drive-v-yaw').textContent = r.pose ? fmt(r.pose.yaw, 1) + '°' : '-';
    $('drive-v-frame').textContent = r.pose ? r.pose.frame : '-';
    draw(r);
  }

  function draw(r) {
    const c = $('drive-canvas'), dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const d = r.drive, half = Math.max(1, Math.min(180, d.angle)) / 2 * Math.PI / 180;
    const cx = w / 2, cy = h - 24;
    // Maßstab: Reichweite passt nach oben und (bei breitem Winkel) zur Seite
    const k = Math.max(1, Math.min((cy - 16) / d.range, (w / 2 - 16) / (d.range * Math.max(Math.sin(half), 0.2))));
    const R = d.range * k, a0 = -Math.PI / 2 - half, a1 = -Math.PI / 2 + half;
    // Kuchen
    ctx.fillStyle = '#0b2614';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, a0, a1); ctx.closePath(); ctx.fill();
    // Prüfbereich
    const rc = Math.min(d.check, d.range) * k;
    ctx.fillStyle = r.free === false ? 'rgba(229,57,53,0.28)' : r.free === true ? 'rgba(67,160,71,0.22)' : 'rgba(150,150,150,0.15)';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, rc, a0, a1); ctx.closePath(); ctx.fill();
    // Ringe und Beschriftung
    ctx.strokeStyle = 'rgba(120,255,160,0.25)'; ctx.lineWidth = 1; ctx.font = '11px sans-serif'; ctx.fillStyle = 'rgba(150,255,180,0.8)';
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath(); ctx.arc(cx, cy, R * i / 4, a0, a1); ctx.stroke();
      ctx.fillText(+(d.range * i / 4).toPrecision(3), cx + 4, cy - R * i / 4 + 12);
    }
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a0), cy + R * Math.sin(a0));
    ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a1), cy + R * Math.sin(a1)); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R); ctx.stroke();
    ctx.strokeStyle = r.free === false ? '#ff5252' : '#66bb6a'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, rc, a0, a1); ctx.stroke();
    // Punkte: Höhe als Helligkeit
    for (const p of (r.points || [])) {
      const x = cx + p[0] * k, y = cy - p[1] * k;
      ctx.fillStyle = p[3] ? '#ff5252' : 'rgba(110,255,150,0.85)';
      const s = p[3] ? 3.2 : 2.2;
      ctx.fillRect(x - s / 2, y - s / 2, s, s);
    }
    // Kamera
    ctx.fillStyle = '#ffd34d';
    ctx.beginPath(); ctx.moveTo(cx, cy - 10); ctx.lineTo(cx - 7, cy + 6); ctx.lineTo(cx + 7, cy + 6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('Öffnung ' + fmt(d.angle, 0) + '°, oben bis ' + fmt(d.up_angle, 0) + '°, ab ' + fmt(d.lower) + ' über Kamera', 8, 16);
  }
  const onResize = () => { if (!alive()) { window.removeEventListener('resize', onResize); return; } if (D.last) draw(D.last); };
  window.addEventListener('resize', onResize);

  loadSessions().then(() => poll(D.token));
})();
"""

SETTINGS_HTML = """
<div class="slam settings">
  <div class="slam-card">
    <h3>Allgemein</h3>
    <div class="slam-row"><label for="set-backend">Backend</label>
      <select id="set-backend"><option value="auto">auto</option><option value="opencv">opencv</option><option value="stella">stella</option></select></div>
    <div class="slam-row"><label for="set-fov">FOV °</label><input type="number" id="set-fov" min="20" max="170" step="1" /></div>
    <div class="slam-row"><label for="set-width">Breite (px)</label><input type="number" id="set-width" min="160" step="16" /></div>
    <div class="slam-row"><label for="set-event_every">slam_pose jedes n-te Bild</label><input type="number" id="set-event_every" min="1" step="1" /></div>
    <div class="slam-row"><label for="set-push_timeout">Push-Timeout (s)</label><input type="number" id="set-push_timeout" min="1" step="1" /></div>
  </div>
  <div class="slam-card" style="margin-top: 12px">
    <h3>Punkterkennung (opencv)</h3>
    <div class="slam-row"><label for="set-quality_level">Schwellwert (qualityLevel)</label>
      <input type="number" id="set-quality_level" min="0.0001" max="0.99" step="0.001" />
      <span class="slam-legend">Anteil der stärksten Ecke im Bild, ab dem ein Punkt zählt. Höher = weniger, stärkere Punkte
        (z.B. 0.05), niedriger = mehr Punkte auch in schwach strukturierten Bereichen (z.B. 0.005). Standard 0.01</span></div>
    <div class="slam-row"><label for="set-min_distance">Mindestabstand (minDistance, px)</label>
      <input type="number" id="set-min_distance" min="1" max="200" step="1" />
      <span class="slam-legend">Abstand zwischen erkannten Punkten, bezogen auf die Verarbeitungsbreite. Kleiner = dichter. Standard 10</span></div>
    <div class="slam-legend">Gilt für neu gestartete Läufe.</div>
  </div>
  <div class="slam-card" style="margin-top: 12px">
    <h3>Drive Radar</h3>
    <div class="slam-row"><label for="set-drive_angle">Öffnungswinkel °</label><input type="number" id="set-drive_angle" min="5" max="180" step="1" /></div>
    <div class="slam-row"><label for="set-drive_range">Reichweite</label><input type="number" id="set-drive_range" min="0.01" step="any" /></div>
    <div class="slam-row"><label for="set-drive_check">Prüfdistanz</label><input type="number" id="set-drive_check" min="0" step="any" /></div>
    <div class="slam-row"><label for="set-drive_lower">Untergrenze über Kamera</label><input type="number" id="set-drive_lower" step="any" />
      <span class="slam-legend">0 = alles unterhalb der Kamerahöhe ausblenden, positiv = zusätzlich n Einheiten darüber</span></div>
    <div class="slam-row"><label for="set-drive_up_angle">Sichtwinkel nach oben °</label><input type="number" id="set-drive_up_angle" min="0" max="90" step="1" /></div>
    <div class="slam-row"><label for="set-drive_threshold">erlaubte Punkte im Weg</label><input type="number" id="set-drive_threshold" min="0" step="1" /></div>
    <div class="slam-row"><label><input type="checkbox" id="set-drive_events" /> Event slam_path bei Änderung</label></div>
  </div>
  <div class="slam-row" style="margin-top: 10px"><button id="set-save">Speichern</button><span class="slam-msg" id="set-msg"></span></div>
  <div class="slam-card" style="margin-top: 12px">
    <h3>Aktionen</h3>
    <div class="slam-row"><span>Arbeitsverzeichnis</span><code id="set-workdir"></code></div>
    <div class="slam-row"><span id="set-usage"></span></div>
    <div class="slam-row"><button id="set-clear">Tmp-Ordner leeren</button>
      <span class="slam-legend">löscht beendete Läufe und hochgeladene Videos, laufende Läufe, Einstellungen und das ORB-Vokabular bleiben</span></div>
    <div class="slam-msg" id="set-clear-msg"></div>
  </div>
</div>
"""

SETTINGS_JS = r"""
(function() {
  const cfg = SET_CFG;
  const $ = (id) => document.getElementById(id);
  const job = (data) => fetch(location.protocol + '//' + location.host + '/', { method: 'POST',
    body: new URLSearchParams({ type: cfg.type, ...data }) }).then((r) => r.json());
  const GENERAL = ['backend', 'fov', 'width', 'event_every', 'push_timeout', 'quality_level', 'min_distance'];
  const DRIVE = ['angle', 'range', 'check', 'lower', 'up_angle', 'threshold'];
  function usage(u, dir) {
    $('set-workdir').textContent = dir || '';
    $('set-usage').textContent = u ? u.sessions + ' Läufe, ' + u.uploads + ' Uploads, ' + (u.bytes / 1e6).toFixed(1) + ' MB' : '';
  }
  function fill(r) {
    const s = r.settings;
    GENERAL.forEach((k) => $('set-' + k).value = s[k]);
    DRIVE.forEach((k) => $('set-drive_' + k).value = s.drive[k]);
    $('set-drive_events').checked = !!s.drive.events;
    usage(r.usage, r.work_dir);
  }
  function msg(id, text, error) { $(id).textContent = text; $(id).classList.toggle('error', !!error); }
  job({ op: 'settings' }).then((r) => { if (r.success) fill(r); else msg('set-msg', r.message, true); });
  $('set-save').addEventListener('click', () => {
    const data = { op: 'settings', drive_events: $('set-drive_events').checked ? 1 : 0 };
    GENERAL.forEach((k) => data[k] = $('set-' + k).value);
    DRIVE.forEach((k) => data['drive_' + k] = $('set-drive_' + k).value);
    job(data).then((r) => {
      if (!r.success) { msg('set-msg', r.message || 'Fehler', true); return; }
      fill(r); msg('set-msg', 'Gespeichert');
    });
  });
  $('set-clear').addEventListener('click', () => {
    if (!confirm('Beendete Läufe und hochgeladene Videos im Arbeitsverzeichnis löschen?')) return;
    msg('set-clear-msg', 'Lösche ...');
    job({ op: 'clear_tmp' }).then((r) => {
      msg('set-clear-msg', r.message || '', !r.success);
      if (r.usage) usage(r.usage, $('set-workdir').textContent);
    });
  });
})();
"""


__all__ = ['SlamRunner', 'MonoVO', 'FrameSource', 'SlamSession']
