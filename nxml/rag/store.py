#
# Project: nxml
#
# Speicher für nxml.rag: Filter auf document_data und die Backends memory, file, postgres, qdrant
# (local_index: index.py)
#
# data_filter: einfache Gleichheit auf Werte von document_data, {"lang": "de", "year": 2026},
#              eine Liste als Wert bedeutet "einer davon", {"lang": ["de", "en"]}. Alle Bedingungen müssen gelten.
#
import copy
import json
import os
import re
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Optional


# --- Filter ---------------------------------------------------------------------------------------------------------

def normalize_filter(data_filter) -> dict:
    """ None, {} oder JSON-String -> dict; Schlüssel müssen einfache Namen sein (wegen SQL/Qdrant-Pfaden) """
    if data_filter is None or data_filter == '':
        return {}
    if isinstance(data_filter, str):
        data_filter = json.loads(data_filter)
    if not isinstance(data_filter, dict):
        raise ValueError('data_filter muss ein dict sein')
    for key, value in data_filter.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(key)):
            raise ValueError(f'ungültiger Filter-Schlüssel {key!r}')
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if any(isinstance(v, (dict, list, tuple, set)) for v in values):
            raise ValueError(f'Filter {key}: nur einfache Werte erlaubt')
    return dict(data_filter)


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    return a == b


def matches_filter(data: dict, data_filter: dict) -> bool:
    """ Gleichheit je Schlüssel, Liste im Filter: einer der Werte; Liste in data: enthält den Wert """
    for key, wanted in (data_filter or {}).items():
        if key not in data:
            return False
        value = data[key]
        wanted = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
        have = value if isinstance(value, list) else [value]
        if not any(_same(h, w) for h in have for w in wanted):
            return False
    return True


# --- Speicher -------------------------------------------------------------------------------------------------------

class VectorStore(ABC):
    """
      Chunks: dict mit id, doc_id, chunk_index, text, data (document_data) und embedding (list[float]).
      Treffer von search: dict mit id, doc_id, chunk_index, text, data, score (Kosinus-Ähnlichkeit, höher = näher).
    """

    name = ''

    @abstractmethod
    def add(self, chunks: list):
        """ Chunks einfügen bzw. gleiche id ersetzen """

    @abstractmethod
    def search(self, embedding: list, limit: int = 5, data_filter: Optional[dict] = None) -> list:
        ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """ :return: Anzahl gelöschter Chunks, -1 wenn unbekannt """

    @abstractmethod
    def count(self) -> int:
        ...

    def documents(self, limit: int = 100) -> list:
        """ [{doc_id, chunks, data}] """
        return []

    def clear(self):
        raise NotImplementedError

    def close(self):
        pass


class MemoryVectorStore(VectorStore):
    """
      Nur im Arbeitsspeicher, geht beim Beenden verloren: für Tests, temporäre Sammlungen oder Daten,
      die beim Start ohnehin neu indiziert werden.
      Metadaten als Liste, Vektoren als float32-Matrix (n x dims, Zeilen auf Länge 1 normiert),
      gesucht wird exakt per Matrixprodukt.
    """

    name = 'memory'

    def __init__(self):
        import numpy as np
        self.np = np
        self.lock = threading.RLock()
        self.meta = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.dims = 0

    def save(self):
        """ nach jeder Änderung, FileVectorStore schreibt hier die Dateien """

    def _normalize(self, vectors):
        np = self.np
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, chunks: list):
        if not chunks:
            return
        np = self.np
        with self.lock:
            vectors = self._normalize([c['embedding'] for c in chunks])
            if self.dims and vectors.shape[1] != self.dims:
                raise ValueError(f'Embedding hat {vectors.shape[1]} Dimensionen, Index {self.dims}')
            self.dims = vectors.shape[1]
            new_ids = {c['id'] for c in chunks}
            keep = [i for i, m in enumerate(self.meta) if m['id'] not in new_ids]
            old = self.vectors[keep] if len(self.meta) else np.zeros((0, self.dims), np.float32)
            self.meta = [self.meta[i] for i in keep] + [
                {**{k: v for k, v in c.items() if k != 'embedding'}, 'data': copy.deepcopy(c.get('data') or {})}
                for c in chunks]
            self.vectors = np.vstack([old.reshape(-1, self.dims), vectors])
            self.save()

    def search(self, embedding: list, limit: int = 5, data_filter: Optional[dict] = None) -> list:
        np = self.np
        with self.lock:
            if not self.meta:
                return []
            query = self._normalize(embedding)[0]
            if query.shape[0] != self.dims:
                raise ValueError(f'Embedding hat {query.shape[0]} Dimensionen, Index {self.dims}')
            if data_filter:
                idx = np.array([i for i, m in enumerate(self.meta) if matches_filter(m.get('data', {}), data_filter)],
                               dtype=np.int64)
                if len(idx) == 0:
                    return []
            else:
                idx = np.arange(len(self.meta))
            scores = self.vectors[idx] @ query
            limit = min(max(1, int(limit)), len(idx))
            best = np.argpartition(-scores, limit - 1)[:limit]
            best = best[np.argsort(-scores[best])]
            return [dict(self.meta[int(idx[i])], data=dict(self.meta[int(idx[i])].get('data') or {}),
                         score=float(scores[i])) for i in best]

    def delete_document(self, doc_id: str) -> int:
        with self.lock:
            keep = [i for i, m in enumerate(self.meta) if m['doc_id'] != doc_id]
            removed = len(self.meta) - len(keep)
            if removed:
                self.meta = [self.meta[i] for i in keep]
                self.vectors = self.vectors[keep]
                self.save()
            return removed

    def count(self) -> int:
        with self.lock:
            return len(self.meta)

    def documents(self, limit: int = 100) -> list:
        with self.lock:
            docs = {}
            for m in self.meta:
                d = docs.setdefault(m['doc_id'], {'doc_id': m['doc_id'], 'chunks': 0, 'data': m.get('data', {})})
                d['chunks'] += 1
            return list(docs.values())[:limit] if limit else list(docs.values())


    def stats(self) -> dict:
        with self.lock:
            return {'count': len(self.meta), 'dims': self.dims}

    def clear(self):
        with self.lock:
            self.meta = []
            self.dims = 0
            self.vectors = self.np.zeros((0, 0), self.np.float32)
            self.save()


class FileVectorStore(MemoryVectorStore):
    """
      Einfacher Index in einem Verzeichnis, wie memory, aber nach jeder Änderung gespeichert:
        index.json    dims, Anzahl
        chunks.jsonl  je Zeile ein Chunk ohne Embedding (gleiche Reihenfolge wie die Matrix)
        vectors.npy   float32-Matrix (n x dims), Zeilen auf Länge 1 normiert
      Alles liegt im Speicher, gesucht wird per Matrixprodukt (exakt, für bis zu einigen 10k Chunks geeignet).
      Geschrieben wird atomar (temporäre Datei + os.replace).
    """

    name = 'file'

    def __init__(self, path='rag_index'):
        super().__init__()
        self.path = str(path)
        self.load()

    def _file(self, name):
        return os.path.join(self.path, name)

    def load(self):
        np = self.np
        with self.lock:
            if not os.path.exists(self._file('index.json')):
                return
            with open(self._file('index.json'), encoding='utf-8') as f:
                info = json.load(f)
            self.dims = int(info.get('dims', 0))
            with open(self._file('chunks.jsonl'), encoding='utf-8') as f:
                self.meta = [json.loads(line) for line in f if line.strip()]
            self.vectors = np.load(self._file('vectors.npy')) if self.meta else np.zeros((0, self.dims), np.float32)
            if len(self.meta) != len(self.vectors):
                raise ValueError(f'Index {self.path} beschädigt: {len(self.meta)} Chunks, {len(self.vectors)} Vektoren')

    def _write(self, name, writer):
        tmp = self._file(name + '.tmp')
        writer(tmp)
        os.replace(tmp, self._file(name))

    def save(self):
        np = self.np
        with self.lock:
            os.makedirs(self.path, exist_ok=True)

            def write_vectors(tmp):
                with open(tmp, 'wb') as f:
                    np.save(f, self.vectors.astype(np.float32))

            def write_chunks(tmp):
                with open(tmp, 'w', encoding='utf-8') as f:
                    for m in self.meta:
                        f.write(json.dumps(m, ensure_ascii=False) + '\n')

            def write_info(tmp):
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump({'dims': self.dims, 'count': len(self.meta), 'version': 1}, f)

            # index.json zuletzt, Anzahl wird beim Laden gegen chunks/vectors geprüft
            self._write('vectors.npy', write_vectors)
            self._write('chunks.jsonl', write_chunks)
            self._write('index.json', write_info)



class PostgresVectorStore(VectorStore):
    """
      Postgres mit der Erweiterung pgvector (CREATE EXTENSION vector, wird versucht anzulegen).
      Tabelle <table>(id, doc_id, chunk_index, text, data jsonb, embedding vector(dims)) mit HNSW-Index (Kosinus).
      Die Tabelle wird beim ersten add angelegt, wenn die Dimension bekannt ist.
      Filter: data @> '{"key": wert}' bzw. data->'key' für Listen ("einer davon"), GIN-Index auf data.
    """

    name = 'postgres'

    def __init__(self, dsn='postgresql://localhost/rag', table='rag_chunks', connection=None):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table):
            raise ValueError(f'ungültiger Tabellenname {table!r}')
        self.dsn = dsn
        self.table = table
        self.lock = threading.RLock()
        self._conn = connection
        self.dims = None

    @property
    def conn(self):
        if self._conn is None or getattr(self._conn, 'closed', False):
            import psycopg
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def _exec(self, sql, params=None, fetch=False):
        with self.lock:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall() if fetch else cur.rowcount

    def _table_dims(self):
        if not self.dims:
            rows = self._exec("SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(%s) "
                              "AND attname = 'embedding'", [self.table], fetch=True)
            self.dims = int(rows[0][0]) if rows else 0
        return self.dims

    def create(self, dims: int):
        dims = int(dims)
        try:
            self._exec("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            pass  # ohne Rechte muss die Erweiterung vorher angelegt sein
        t = self.table
        self._exec(f"""CREATE TABLE IF NOT EXISTS {t} (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            data JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({dims}) NOT NULL)""")
        self._exec(f"CREATE INDEX IF NOT EXISTS {t}_doc_idx ON {t} (doc_id)")
        self._exec(f"CREATE INDEX IF NOT EXISTS {t}_data_idx ON {t} USING GIN (data jsonb_path_ops)")
        if dims <= 2000:  # HNSW in pgvector bis 2000 Dimensionen
            self._exec(f"CREATE INDEX IF NOT EXISTS {t}_emb_idx ON {t} USING hnsw (embedding vector_cosine_ops)")
        self.dims = dims

    @staticmethod
    def vector_literal(embedding) -> str:
        return '[' + ','.join(repr(float(x)) for x in embedding) + ']'

    def add(self, chunks: list):
        if not chunks:
            return
        dims = len(chunks[0]['embedding'])
        existing = self._table_dims()
        if not existing:
            self.create(dims)
        elif existing != dims:
            raise ValueError(f'Embedding hat {dims} Dimensionen, Tabelle {self.table} {existing}')
        sql = (f"INSERT INTO {self.table} (id, doc_id, chunk_index, text, data, embedding) "
               f"VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector) "
               f"ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, chunk_index = EXCLUDED.chunk_index, "
               f"text = EXCLUDED.text, data = EXCLUDED.data, embedding = EXCLUDED.embedding")
        rows = [(c['id'], c['doc_id'], int(c['chunk_index']), c['text'],
                 json.dumps(c.get('data', {}), ensure_ascii=False), self.vector_literal(c['embedding']))
                for c in chunks]
        with self.lock:
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    cur.executemany(sql, rows)

    @staticmethod
    def filter_sql(data_filter: dict):
        """ :return: (WHERE-Teil, Parameter) """
        conds, params = [], []
        for key, value in (data_filter or {}).items():
            values = list(value) if isinstance(value, (list, tuple, set)) else [value]
            # @> trifft gleiche Skalare und Listen, die den Wert enthalten (wie matches_filter)
            ors = ["data @> %s::jsonb" for _ in values]
            params += [json.dumps({key: v}, ensure_ascii=False) for v in values]
            conds.append('(' + ' OR '.join(ors) + ')' if ors else 'FALSE')
        return (' AND '.join(conds) if conds else 'TRUE'), params

    def search(self, embedding: list, limit: int = 5, data_filter: Optional[dict] = None) -> list:
        if not self._table_dims():
            return []
        where, params = self.filter_sql(data_filter)
        vec = self.vector_literal(embedding)
        rows = self._exec(f"SELECT id, doc_id, chunk_index, text, data, 1 - (embedding <=> %s::vector) AS score "
                          f"FROM {self.table} WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
                          [vec] + params + [vec, int(limit)], fetch=True)
        return [{'id': row[0], 'doc_id': row[1], 'chunk_index': row[2], 'text': row[3],
                 'data': row[4] if isinstance(row[4], dict) else json.loads(row[4] or '{}'), 'score': float(row[5])}
                for row in rows]

    def delete_document(self, doc_id: str) -> int:
        if not self._table_dims():
            return 0
        return self._exec(f"DELETE FROM {self.table} WHERE doc_id = %s", [doc_id])

    def count(self) -> int:
        if not self._table_dims():
            return 0
        return int(self._exec(f"SELECT count(*) FROM {self.table}", fetch=True)[0][0])

    def documents(self, limit: int = 100) -> list:
        if not self._table_dims():
            return []
        rows = self._exec(f"SELECT doc_id, count(*), min(data::text) FROM {self.table} GROUP BY doc_id "
                          f"ORDER BY doc_id LIMIT %s", [int(limit) or None], fetch=True)
        return [{'doc_id': d, 'chunks': int(n), 'data': json.loads(data or '{}')} for d, n, data in rows]

    def clear(self):
        if self._table_dims():
            self._exec(f"TRUNCATE {self.table}")

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class QdrantVectorStore(VectorStore):
    """
      Qdrant-Collection (Distanz Cosine), wird beim ersten add mit der Dimension angelegt.
      Punkt-IDs sind UUIDv5 aus der Chunk-id, Payload: chunk_id, doc_id, chunk_index, text, data.
      Filter auf data.<key> (MatchValue bzw. MatchAny für Listen), Listen in data treffen, wenn sie den Wert enthalten.
      Qdrant vergleicht nur str, int und bool exakt (keine float-Werte im Filter).
      Für Filter auf großen Collections einen Payload-Index anlegen: create_payload_index('data.lang', 'keyword').
    """

    name = 'qdrant'
    NAMESPACE = uuid.UUID('6f1c8a52-4d7e-4b0a-9a51-2b0e2d8f6c11')

    def __init__(self, url='http://localhost:6333', collection='rag', api_key=None, path=None, client=None):
        if client is None:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=path) if path else QdrantClient(url=url, api_key=api_key or None)
        self.client = client
        self.collection = collection
        self.lock = threading.RLock()

    @property
    def models(self):
        from qdrant_client import models
        return models

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(self.NAMESPACE, chunk_id))

    def add(self, chunks: list):
        if not chunks:
            return
        m = self.models
        with self.lock:
            if not self.exists():
                self.client.create_collection(self.collection, vectors_config=m.VectorParams(
                    size=len(chunks[0]['embedding']), distance=m.Distance.COSINE))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # lokaler Qdrant ignoriert Payload-Indizes
                    self.client.create_payload_index(self.collection, 'doc_id', m.PayloadSchemaType.KEYWORD)
            points = [m.PointStruct(id=self.point_id(c['id']), vector=[float(x) for x in c['embedding']],
                                    payload={'chunk_id': c['id'], 'doc_id': c['doc_id'],
                                             'chunk_index': int(c['chunk_index']), 'text': c['text'],
                                             'data': c.get('data', {})})
                      for c in chunks]
            self.client.upsert(self.collection, points=points, wait=True)

    def build_filter(self, data_filter: dict):
        if not data_filter:
            return None
        m = self.models
        must = []
        for key, value in data_filter.items():
            if isinstance(value, (list, tuple, set)):
                must.append(m.FieldCondition(key='data.' + key, match=m.MatchAny(any=list(value))))
            else:
                must.append(m.FieldCondition(key='data.' + key, match=m.MatchValue(value=value)))
        return m.Filter(must=must)

    def search(self, embedding: list, limit: int = 5, data_filter: Optional[dict] = None) -> list:
        if not self.exists():
            return []
        res = self.client.query_points(self.collection, query=[float(x) for x in embedding],
                                       query_filter=self.build_filter(data_filter), limit=int(limit),
                                       with_payload=True)
        return [{'id': p.payload.get('chunk_id', str(p.id)), 'doc_id': p.payload.get('doc_id'),
                 'chunk_index': p.payload.get('chunk_index'), 'text': p.payload.get('text', ''),
                 'data': p.payload.get('data', {}), 'score': float(p.score)}
                for p in res.points]

    def _doc_filter(self, doc_id):
        m = self.models
        return m.Filter(must=[m.FieldCondition(key='doc_id', match=m.MatchValue(value=doc_id))])

    def delete_document(self, doc_id: str) -> int:
        if not self.exists():
            return 0
        with self.lock:
            n = self.client.count(self.collection, count_filter=self._doc_filter(doc_id), exact=True).count
            if n:
                self.client.delete(self.collection, points_selector=self.models.FilterSelector(
                    filter=self._doc_filter(doc_id)), wait=True)
            return n

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def documents(self, limit: int = 100) -> list:
        if not self.exists():
            return []
        docs, offset = {}, None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_payload=True,
                                                with_vectors=False)
            for p in points:
                d = docs.setdefault(p.payload['doc_id'], {'doc_id': p.payload['doc_id'], 'chunks': 0,
                                                          'data': p.payload.get('data', {})})
                d['chunks'] += 1
            if offset is None or (limit and len(docs) > limit):
                break
        return list(docs.values())[:limit] if limit else list(docs.values())

    def clear(self):
        if self.exists():
            self.client.delete_collection(self.collection)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# Klasse oder 'modul:Klasse' (lazy, nxml.rag.index importiert dieses Modul)
BACKENDS = {'memory': MemoryVectorStore, 'mem': MemoryVectorStore, 'ram': MemoryVectorStore,
            'file': FileVectorStore, 'fs': FileVectorStore, 'filesystem': FileVectorStore,
            'local_index': 'nxml.rag.index:LocalIndexStore', 'local': 'nxml.rag.index:LocalIndexStore',
            'ivf': 'nxml.rag.index:LocalIndexStore',
            'postgres': PostgresVectorStore, 'pg': PostgresVectorStore, 'pgvector': PostgresVectorStore,
            'qdrant': QdrantVectorStore}


def backend_class(backend: str):
    cls = BACKENDS.get(str(backend).lower())
    if cls is None:
        raise ValueError(f'unbekanntes Backend {backend!r}, möglich: memory, file, local_index, postgres, qdrant')
    if isinstance(cls, str):
        import importlib
        module, name = cls.split(':')
        cls = getattr(importlib.import_module(module), name)
    return cls


def create_store(backend: str = 'file', **kwargs) -> VectorStore:
    """ create_store('file', path='data/rag'), create_store('local_index', path=...), create_store('postgres',
        dsn=...), create_store('qdrant', url=...) """
    return backend_class(backend)(**kwargs)
