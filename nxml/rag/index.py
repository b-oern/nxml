#
# Project: nxml
#
# Lokaler Vektor-Index für nxml.rag (backend=local_index), nur numpy + sqlite3:
#
#   vectors.<gen>.f32   Vektoren (float32, auf Länge 1 normiert), nur angehängt, per memmap gelesen
#   centroids.<cgen>.npy  IVF-Zentren (sphärisches k-Means), ab min_train Chunks
#   index.sqlite        chunks(row, id, doc_id, chunk_index, text, data, list) und kv(row, key, value) für Filter
#
# Suche: Anfrage mit allen Zentren vergleichen, die nprobe nächsten Listen durchsuchen (Inverted File) und dort exakt
# bewerten. Vor dem Training, oder wenn der Filter höchstens exact_limit Chunks übrig lässt, wird exakt gesucht.
# Filter laufen über den Index kv(key, value) statt über alle Metadaten.
#
# Schreiben: neue Vektoren werden angehängt (kein Neuschreiben), Löschen entfernt nur die Zeilen in sqlite.
# Ab compact_ratio gelöschter Zeilen wird in eine neue Vektordatei (Generation) umkopiert; ab doppelt so vielen
# Chunks wie beim letzten Training werden die Zentren neu berechnet. Generationen werden in derselben
# sqlite-Transaktion umgeschaltet, ein Abbruch hinterlässt so immer einen gültigen Stand.
#
import json
import os
import sqlite3
import threading
from typing import Optional

from nxml.rag.store import VectorStore, normalize_filter


def _encode(value) -> str:
    """ Wert für kv: JSON, ganzzahlige floats wie int (2026.0 == 2026) """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return json.dumps(value, ensure_ascii=False)


def _scalars(value) -> list:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [v for v in values if v is None or isinstance(v, (str, int, float, bool))]


class LocalIndexStore(VectorStore):
    """
      Effizienter lokaler Index (IVF) im Verzeichnis path.

      :param nprobe: Anzahl durchsuchter Listen, 0: automatisch (k/16, mindestens 8)
      :param min_train: ab so vielen Chunks werden IVF-Zentren trainiert, darunter exakte Suche
      :param exact_limit: Filter mit höchstens so vielen Treffern exakt durchsuchen
      :param compact_ratio: Anteil gelöschter Zeilen, ab dem die Vektordatei kompaktiert wird
    """

    name = 'local_index'
    BATCH_FLOATS = 1 << 24  # ~64 MB je Matrixprodukt-Block

    def __init__(self, path='rag_index', nprobe=0, min_train=4096, exact_limit=20000, compact_ratio=0.25,
                 seed=0):
        import numpy as np
        self.np = np
        self.path = str(path)
        self.nprobe = int(nprobe or 0)
        self.min_train = max(64, int(min_train))
        self.exact_limit = int(exact_limit)
        self.compact_ratio = float(compact_ratio)
        self.seed = int(seed)
        self.lock = threading.RLock()
        os.makedirs(self.path, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(self.path, 'index.sqlite'), check_same_thread=False,
                                  isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS chunks (
                row INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, doc_id TEXT NOT NULL, chunk_index INTEGER,
                text TEXT, data TEXT, list INTEGER NOT NULL DEFAULT -1);
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
            CREATE TABLE IF NOT EXISTS kv (row INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS kv_key ON kv (key, value, row);
            CREATE INDEX IF NOT EXISTS kv_row ON kv (row);
        """)
        self._load()

    # --- Zustand --------------------------------------------------------------------------------

    def _info(self, key, default=None):
        row = self.db.execute('SELECT value FROM info WHERE key = ?', [key]).fetchone()
        return row[0] if row else default

    def _set_info(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)', [key, str(value)])

    def _vfile(self, gen=None):
        return os.path.join(self.path, f'vectors.{self.gen if gen is None else gen}.f32')

    def _cfile(self, cgen=None):
        return os.path.join(self.path, f'centroids.{self.cgen if cgen is None else cgen}.npy')

    def _load(self):
        np = self.np
        with self.lock:
            self._mm = None
            self._inv = None
            self.dims = int(self._info('dims', 0))
            self.gen = int(self._info('gen', 0))
            self.cgen = int(self._info('cgen', 0))
            self.trained_n = int(self._info('trained_n', 0))
            self._cleanup_files()
            self.centroids = np.load(self._cfile()) if self.cgen and os.path.exists(self._cfile()) else None
            size = os.path.getsize(self._vfile()) if self.dims and os.path.exists(self._vfile()) else 0
            self.rows_total = size // (4 * self.dims) if self.dims else 0
            self.alive = np.zeros(self.rows_total, dtype=bool)
            self.lists = np.full(self.rows_total, -1, dtype=np.int32)
            data = np.array(self.db.execute('SELECT row, list FROM chunks').fetchall(), dtype=np.int64).reshape(-1, 2)
            if len(data):
                broken = data[data[:, 0] >= self.rows_total, 0]
                if len(broken):  # Zeilen ohne Vektor (sollte nicht vorkommen) entfernen
                    self._delete_rows([int(x) for x in broken])
                    data = data[data[:, 0] < self.rows_total]
                self.alive[data[:, 0]] = True
                self.lists[data[:, 0]] = data[:, 1]

    def _cleanup_files(self):
        """ Dateien älterer oder abgebrochener Generationen entfernen """
        keep = {os.path.basename(self._vfile()), os.path.basename(self._cfile())}
        for name in os.listdir(self.path):
            if (name.startswith('vectors.') or name.startswith('centroids.')) and name not in keep:
                try:
                    os.remove(os.path.join(self.path, name))
                except OSError:
                    pass

    @property
    def vectors(self):
        if self._mm is None and self.rows_total:
            self._mm = self.np.memmap(self._vfile(), dtype='<f4', mode='r', shape=(self.rows_total, self.dims))
        return self._mm

    def _normalize(self, vectors):
        np = self.np
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _assign(self, vectors, centroids=None):
        """ nächstes Zentrum je Vektor, blockweise """
        np = self.np
        centroids = self.centroids if centroids is None else centroids
        out = np.empty(len(vectors), dtype=np.int32)
        step = max(1, self.BATCH_FLOATS // max(1, len(centroids)))
        for s in range(0, len(vectors), step):
            out[s:s + step] = np.argmax(np.asarray(vectors[s:s + step]) @ centroids.T, axis=1)
        return out

    # --- Schreiben ------------------------------------------------------------------------------

    def _rows_where(self, sql, params) -> list:
        return [r[0] for r in self.db.execute(sql, params).fetchall()]

    def _rows_of_ids(self, ids) -> list:
        rows = []
        for s in range(0, len(ids), 500):
            part = ids[s:s + 500]
            rows += self._rows_where(f"SELECT row FROM chunks WHERE id IN ({','.join('?' * len(part))})", part)
        return rows

    def _delete_rows(self, rows):
        """ in laufender oder eigener Transaktion """
        self.db.executemany('DELETE FROM chunks WHERE row = ?', [(r,) for r in rows])
        self.db.executemany('DELETE FROM kv WHERE row = ?', [(r,) for r in rows])

    def add(self, chunks: list):
        if not chunks:
            return
        np = self.np
        vectors = self._normalize([c['embedding'] for c in chunks])
        with self.lock:
            if self.dims and vectors.shape[1] != self.dims:
                raise ValueError(f'Embedding hat {vectors.shape[1]} Dimensionen, Index {self.dims}')
            if not self.dims:
                self.dims = vectors.shape[1]
                self._set_info('dims', self.dims)
                self._set_info('gen', self.gen)
            # gleiche id mehrfach im Aufruf: letzte gewinnt
            last = {c['id']: i for i, c in enumerate(chunks)}
            order = sorted(last.values())
            chunks, vectors = [chunks[i] for i in order], vectors[order]
            old_rows = self._rows_of_ids([c['id'] for c in chunks])
            lists = self._assign(vectors) if self.centroids is not None else np.full(len(chunks), -1, np.int32)
            # 1. Vektoren anhängen (bei Abbruch danach bleiben sie unreferenziert)
            self._mm = None
            with open(self._vfile(), 'ab') as f:
                f.write(vectors.astype('<f4').tobytes())
                f.flush()
                os.fsync(f.fileno())
            start = self.rows_total
            # 2. Metadaten in einer Transaktion
            self.db.execute('BEGIN')
            try:
                self._delete_rows(old_rows)
                self.db.executemany(
                    'INSERT INTO chunks (row, id, doc_id, chunk_index, text, data, list) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [(start + i, c['id'], c['doc_id'], int(c['chunk_index']), c['text'],
                      json.dumps(c.get('data', {}), ensure_ascii=False), int(lists[i]))
                     for i, c in enumerate(chunks)])
                self.db.executemany('INSERT INTO kv (row, key, value) VALUES (?, ?, ?)',
                                    [(start + i, str(k), _encode(v))
                                     for i, c in enumerate(chunks) for k, value in (c.get('data') or {}).items()
                                     for v in _scalars(value)])
                self.db.execute('COMMIT')
            except Exception:
                self.db.execute('ROLLBACK')
                raise
            self.rows_total += len(chunks)
            self.alive = np.concatenate([self.alive, np.ones(len(chunks), bool)])
            self.lists = np.concatenate([self.lists, lists])
            if old_rows:
                self.alive[old_rows] = False
            self._inv = None
            self._maintain()

    def delete_document(self, doc_id: str) -> int:
        with self.lock:
            rows = self._rows_where('SELECT row FROM chunks WHERE doc_id = ?', [doc_id])
            if rows:
                self.db.execute('BEGIN')
                try:
                    self._delete_rows(rows)
                    self.db.execute('COMMIT')
                except Exception:
                    self.db.execute('ROLLBACK')
                    raise
                self.alive[rows] = False
                self._inv = None
                self._maintain()
            return len(rows)

    def _maintain(self):
        n = int(self.alive.sum())
        dead = self.rows_total - n
        if self.rows_total >= 1000 and dead > self.compact_ratio * self.rows_total:
            self.compact()
        if n >= self.min_train and (self.centroids is None or n >= 2 * self.trained_n):
            self.train()

    def compact(self):
        """ lebende Vektoren in eine neue Generation kopieren und die Zeilen neu nummerieren """
        np = self.np
        with self.lock:
            rows = np.nonzero(self.alive)[0]
            new_gen = self.gen + 1
            step = max(1, self.BATCH_FLOATS // max(1, self.dims))
            with open(self._vfile(new_gen), 'wb') as f:
                for s in range(0, len(rows), step):
                    f.write(np.asarray(self.vectors[rows[s:s + step]], dtype='<f4').tobytes())
                f.flush()
                os.fsync(f.fileno())
            mapping = [(int(new), int(old)) for new, old in enumerate(rows)]
            self.db.execute('BEGIN')
            try:
                # aufsteigend: neue Nummer <= alte, belegte Ziele wurden schon verschoben
                self.db.executemany('UPDATE chunks SET row = ? WHERE row = ?', mapping)
                self.db.executemany('UPDATE kv SET row = ? WHERE row = ?', mapping)
                self._set_info('gen', new_gen)
                self.db.execute('COMMIT')
            except Exception:
                self.db.execute('ROLLBACK')
                os.remove(self._vfile(new_gen))
                raise
            self._mm = None
            old_file = self._vfile()
            self.gen = new_gen
            self.lists = self.lists[rows].copy()
            self.alive = np.ones(len(rows), bool)
            self.rows_total = len(rows)
            self._inv = None
            try:
                os.remove(old_file)
            except OSError:
                pass  # unter Windows evtl. noch gemappt, wird beim nächsten Laden entfernt

    def train(self, k: int = 0, iterations: int = 12):
        """ IVF-Zentren per sphärischem k-Means auf einer Stichprobe, danach alle Zeilen zuordnen """
        np = self.np
        with self.lock:
            rows = np.nonzero(self.alive)[0]
            n = len(rows)
            if n == 0:
                return
            k = int(k) or int(np.clip(2 * np.sqrt(n), 8, 8192))
            k = max(1, min(k, n // 32 or 1))
            rng = np.random.default_rng(self.seed)
            sample = np.sort(rng.choice(rows, size=min(n, k * 64), replace=False))
            x = np.asarray(self.vectors[sample], dtype=np.float32)
            c = x[rng.choice(len(x), size=k, replace=False)].copy()
            for _ in range(int(iterations)):
                assign = self._assign(x, c)
                order = np.argsort(assign, kind='stable')
                counts = np.bincount(assign, minlength=k)
                nonempty = counts > 0
                starts = (np.cumsum(counts) - counts)[nonempty]
                c[nonempty] = np.add.reduceat(x[order], starts, axis=0)
                empty = np.nonzero(~nonempty)[0]
                if len(empty):  # leere Listen mit zufälligen Punkten neu belegen
                    c[empty] = x[rng.choice(len(x), size=len(empty), replace=False)]
                c = self._normalize(c)
            lists = self._assign(self.vectors[rows], c)
            new_cgen = self.cgen + 1
            np.save(self._cfile(new_cgen), c)
            self.db.execute('BEGIN')
            try:
                self.db.executemany('UPDATE chunks SET list = ? WHERE row = ?',
                                    zip(lists.tolist(), rows.tolist()))
                self._set_info('cgen', new_cgen)
                self._set_info('trained_n', n)
                self.db.execute('COMMIT')
            except Exception:
                self.db.execute('ROLLBACK')
                os.remove(self._cfile(new_cgen))
                raise
            old_file = self._cfile() if self.cgen else None
            self.cgen, self.centroids, self.trained_n = new_cgen, c, n
            self.lists[rows] = lists
            self._inv = None
            if old_file and os.path.exists(old_file):
                os.remove(old_file)

    # --- Suche ----------------------------------------------------------------------------------

    def _inverted(self):
        """ (sortierte Zeilen, Start je Liste, Anzahl je Liste, Zeilen ohne Liste) """
        np = self.np
        if self._inv is None:
            rows = np.nonzero(self.alive & (self.lists >= 0))[0]
            lst = self.lists[rows]
            order = np.argsort(lst, kind='stable')
            counts = np.bincount(lst, minlength=len(self.centroids))
            self._inv = (rows[order], np.cumsum(counts) - counts, counts,
                         np.nonzero(self.alive & (self.lists < 0))[0])
        return self._inv

    def _filter_rows(self, data_filter: dict):
        np = self.np
        result = None
        for key, value in data_filter.items():
            values = [_encode(v) for v in _scalars(value)]
            if not values:
                return np.zeros(0, np.int64)
            rows = np.unique(np.array(self._rows_where(
                f"SELECT row FROM kv WHERE key = ? AND value IN ({','.join('?' * len(values))})",
                [key] + values), dtype=np.int64))
            result = rows if result is None else np.intersect1d(result, rows, assume_unique=True)
            if len(result) == 0:
                break
        result = result[result < self.rows_total]
        return result[self.alive[result]]

    def _candidates(self, query, limit, filter_rows):
        np = self.np
        if self.centroids is None or (filter_rows is not None and len(filter_rows) <= self.exact_limit):
            return filter_rows if filter_rows is not None else np.nonzero(self.alive)[0]
        k = len(self.centroids)
        nprobe = min(k, self.nprobe or max(8, k // 16))
        probe = np.argpartition(-(self.centroids @ query), nprobe - 1)[:nprobe]
        rows, starts, counts, unassigned = self._inverted()
        cand = np.concatenate([rows[starts[p]:starts[p] + counts[p]] for p in probe] + [unassigned])
        cand.sort()
        if filter_rows is not None:
            cand = np.intersect1d(cand, filter_rows, assume_unique=True)
        if len(cand) < limit:  # zu wenige Treffer in den Listen: exakt
            return filter_rows if filter_rows is not None else np.nonzero(self.alive)[0]
        return cand

    def search(self, embedding: list, limit: int = 5, data_filter: Optional[dict] = None) -> list:
        np = self.np
        limit = max(1, int(limit))
        with self.lock:
            if not self.rows_total or not self.alive.any():
                return []
            query = self._normalize(embedding)[0]
            if query.shape[0] != self.dims:
                raise ValueError(f'Embedding hat {query.shape[0]} Dimensionen, Index {self.dims}')
            data_filter = normalize_filter(data_filter)
            filter_rows = self._filter_rows(data_filter) if data_filter else None
            if filter_rows is not None and len(filter_rows) == 0:
                return []
            cand = self._candidates(query, limit, filter_rows)
            if len(cand) == 0:
                return []
            vectors = self.vectors
            step = max(1, self.BATCH_FLOATS // max(1, self.dims))
            scores = np.concatenate([np.asarray(vectors[cand[s:s + step]]) @ query
                                     for s in range(0, len(cand), step)])
            limit = min(limit, len(cand))
            best = np.argpartition(-scores, limit - 1)[:limit]
            best = best[np.argsort(-scores[best])]
            rows = [int(cand[i]) for i in best]
            meta = {r[0]: r for r in self.db.execute(
                f"SELECT row, id, doc_id, chunk_index, text, data FROM chunks WHERE row IN ({','.join('?' * len(rows))})",
                rows).fetchall()}
            return [{'id': meta[r][1], 'doc_id': meta[r][2], 'chunk_index': meta[r][3], 'text': meta[r][4],
                     'data': json.loads(meta[r][5] or '{}'), 'score': float(scores[i])}
                    for r, i in zip(rows, best) if r in meta]

    # --- Verwaltung -----------------------------------------------------------------------------

    def count(self) -> int:
        with self.lock:
            return int(self.alive.sum())

    def documents(self, limit: int = 100) -> list:
        with self.lock:
            rows = self.db.execute('SELECT doc_id, count(*), min(data) FROM chunks GROUP BY doc_id ORDER BY doc_id '
                                   'LIMIT ?', [int(limit) or -1]).fetchall()
        return [{'doc_id': d, 'chunks': int(n), 'data': json.loads(data or '{}')} for d, n, data in rows]

    def stats(self) -> dict:
        with self.lock:
            return {'count': self.count(), 'rows': self.rows_total, 'dims': self.dims,
                    'lists': 0 if self.centroids is None else len(self.centroids), 'trained_n': self.trained_n,
                    'gen': self.gen}

    def clear(self):
        with self.lock:
            self.db.execute('BEGIN')
            self.db.execute('DELETE FROM chunks')
            self.db.execute('DELETE FROM kv')
            self.db.execute('DELETE FROM info')
            self.db.execute('COMMIT')
            self._mm = None
            self.centroids = None
            self.dims = self.gen = self.cgen = self.trained_n = self.rows_total = 0
            self.alive = self.np.zeros(0, bool)
            self.lists = self.np.zeros(0, self.np.int32)
            self._inv = None
            self._cleanup_files()

    def close(self):
        with self.lock:
            self._mm = None
            self.db.close()
