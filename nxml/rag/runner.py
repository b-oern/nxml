#
# Project: nxml
#
# RagRunner: Chunking, Embeddings über einen Runner, Speicher aus nxml.rag.store, Web-UI aus nxml.rag.ui
#
import hashlib
import re
import threading
from typing import Callable

from nwebclient import runner as r
from nwebclient.runner import op
from nwebclient import util as u

from nxml.rag.store import VectorStore, backend_class, normalize_filter
from nxml.rag.ui import RagUiMixin


# --- Chunking -------------------------------------------------------------------------------------------------------

def _split_long(text: str, size: int) -> list:
    """ Text über size Zeichen an Satzgrenzen, notfalls an Leerzeichen bzw. hart teilen """
    parts = []
    for sentence in re.split(r'(?<=[.!?;:])\s+', text):
        while len(sentence) > size:
            cut = sentence.rfind(' ', 0, size)
            cut = cut if cut > size // 2 else size
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            parts.append(sentence)
    return parts


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list:
    """
      Zerlegt text in Chunks von höchstens size Zeichen. Getrennt wird bevorzugt an Absätzen, dann an Sätzen.
      Aufeinanderfolgende Chunks überlappen um bis zu overlap Zeichen (ganze Sätze/Absätze vom Ende des Vorgängers).
    """
    size = max(50, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    text = (text or '').replace('\r\n', '\n').strip()
    if not text:
        return []
    pieces = []
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'[ \t]+', ' ', para).strip()
        if para:
            pieces.extend([para] if len(para) <= size else _split_long(para, size))
    chunks, current = [], []

    def length(items):
        return sum(len(p) for p in items) + max(0, len(items) - 1)

    for piece in pieces:
        if current and length(current + [piece]) > size:
            chunks.append(' '.join(current))
            tail = []
            for p in reversed(current):
                if length([p] + tail) > overlap or length([p] + tail + [piece]) > size:
                    break
                tail.insert(0, p)
            current = tail
        current.append(piece)
    if current:
        chunks.append(' '.join(current))
    return chunks


# --- Runner ---------------------------------------------------------------------------------------------------------

class RagRunner(RagUiMixin, r.BaseJobExecutor):
    """
      Dokumente indizieren und semantisch durchsuchen.

        {"type": "rag", "op": "index_document", "text": "...", "document_data": {"doc_id": "a1", "lang": "de"}}
        {"type": "rag", "op": "query", "text": "Frage", "data_filter": {"lang": "de"}, "limit": 5}
        {"type": "rag", "op": "chat", "question": "Frage", "llm": "lmstudio", "data_filter": {"lang": "de"}}

      Web-Oberfläche: Chat (LLM aus den geladenen Runnern), Suche, Indizieren, Dokumente, Status (nxml.rag_ui).

      doc_id aus document_data (doc_id oder id), sonst SHA1 des Textes. Erneutes Indizieren derselben doc_id
      ersetzt die alten Chunks. Embeddings über einen Runner (embedder, Standard text_embed: op=embed, input=[...]);
      Runner ohne op=embed werden je Text mit {"text": ...} aufgerufen und liefern embedding (bert) oder
      features (clip).
      Alternativ embed_fn(texts) -> list[list[float]] übergeben.
    """

    TAGS = ['rag', 'embedding', 'search']
    MODULES = ['numpy']

    def __init__(self, type='rag', backend='memory', embedder='text_embed', embedding_model='', chunk_size=1000,
                 chunk_overlap=150, batch_size=32, embed_fn: Callable = None, store: VectorStore = None,
                 path='rag_index', dsn='postgresql://localhost/rag', table='rag_chunks',
                 url='http://localhost:6333', collection='rag', api_key=None, qdrant_path=None, nprobe=0,
                 min_train=4096, llm='', args: u.Args = {}):
        """
        :param backend: memory, file, local_index, postgres oder qdrant
        :param embedder: type des Embedding-Runners
        :param llm: Standard-LLM-Runner für chat, leer: erster geladener Runner mit TAGS llm
        :param path: Verzeichnis des Datei-Index (file, local_index)
        :param nprobe: local_index: durchsuchte IVF-Listen (0: automatisch), min_train: ab so vielen Chunks IVF
        :param dsn: Verbindung zu Postgres (postgres), table: Tabellenname
        :param url: Qdrant-Server, qdrant_path: lokale Qdrant-Datenbank statt Server, collection: Collection
        """
        super().__init__(type)
        self.args = args
        self.backend = backend
        self.embedder = embedder
        self.llm = llm
        self.embedding_model = embedding_model
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.batch_size = max(1, int(batch_size))
        self.embed_fn = embed_fn
        self.store_args = {'memory': {}, 'file': {'path': path},
                           'local_index': {'path': path, 'nprobe': nprobe, 'min_train': min_train},
                           'postgres': {'dsn': dsn, 'table': table},
                           'qdrant': {'url': url, 'collection': collection, 'api_key': api_key,
                                      'path': qdrant_path}}
        self._store = store
        self.lock = threading.RLock()
        self.define_vars('backend', 'embedder', 'embedding_model', 'chunk_size', 'chunk_overlap', 'llm')

    @property
    def store(self) -> VectorStore:
        with self.lock:
            if self._store is None:
                cls = backend_class(self.backend)
                self._store = cls(**self.store_args[cls.name])
            return self._store

    # --- Embeddings -----------------------------------------------------------------------------

    def run_job(self, job: dict) -> dict:
        root = self.getRoot()
        if root is None or root is self:
            raise Exception('kein Dispatcher für ' + str(job.get('type')))
        return root.execute(dict(job))

    def embed(self, texts: list, input_type: str = 'passage') -> list:
        """ :param input_type: passage (Dokument-Chunks) oder query (Suchanfrage), für Modelle mit Präfixen (E5) """
        if not texts:
            return []
        if self.embed_fn is not None:
            try:
                vectors = self.embed_fn(list(texts), input_type=input_type)
            except TypeError:
                vectors = self.embed_fn(list(texts))
            return [list(map(float, e)) for e in vectors]
        res = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            job = {'type': self.embedder, 'op': 'embed', 'input': batch, 'input_type': input_type}
            if self.embedding_model:
                job['model'] = self.embedding_model
            result = self.run_job(job)
            if result.get('success') and result.get('embeddings'):
                res.extend(result['embeddings'])
                continue
            # Runner ohne op=embed: BertEmbeddings {"text"} -> embedding, ClipEmbeddings {"text"} -> features
            for text in batch:
                single = self.run_job({'type': self.embedder, 'text': text, 'input_type': input_type})
                vector = single.get('embedding') or single.get('features')
                while isinstance(vector, list) and len(vector) == 1 and isinstance(vector[0], list):
                    vector = vector[0]  # [[...]] -> [...]
                if not vector:
                    raise Exception(f'Embedding von {self.embedder} fehlgeschlagen: '
                                    f'{result.get("message", "")} / {single.get("message", "")}')
                res.append(vector)
        return [list(map(float, e)) for e in res]

    # --- API ------------------------------------------------------------------------------------

    @staticmethod
    def doc_id_of(text: str, document_data: dict) -> str:
        doc_id = document_data.get('doc_id', document_data.get('id'))
        if doc_id in (None, ''):
            doc_id = hashlib.sha1(text.encode('utf-8')).hexdigest()
        return str(doc_id)

    def index_document(self, text: str, document_data: dict = None) -> dict:
        """ Chunks + Embeddings in den Speicher, ersetzt einen vorhandenen Stand derselben doc_id """
        document_data = dict(document_data or {})
        doc_id = self.doc_id_of(text, document_data)
        document_data.setdefault('doc_id', doc_id)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        embeddings = self.embed(chunks)
        if len(embeddings) != len(chunks):
            raise Exception(f'{len(embeddings)} Embeddings für {len(chunks)} Chunks')
        rows = [{'id': f'{doc_id}#{i}', 'doc_id': doc_id, 'chunk_index': i, 'text': chunk, 'data': document_data,
                 'embedding': emb} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        with self.lock:
            self.store.delete_document(doc_id)
            self.store.add(rows)
        return {'doc_id': doc_id, 'chunks': len(rows)}

    def query(self, text: str, data_filter: dict = None, limit: int = 5, min_score: float = None) -> list:
        """ :return: Treffer [{id, doc_id, chunk_index, text, data, score}], bester zuerst """
        data_filter = normalize_filter(data_filter)
        embedding = self.embed([text], 'query')[0]
        hits = self.store.search(embedding, int(limit), data_filter)
        if min_score is not None:
            hits = [h for h in hits if h['score'] >= float(min_score)]
        return hits

    # --- Operationen ----------------------------------------------------------------------------

    @op("Text in Chunks zerlegen, Embeddings berechnen und speichern. document_data: Metadaten (doc_id/id optional), "
        "nach denen query filtern kann", result={'success': 'bool', 'doc_id': 'str', 'chunks': 'int'},
        name='index_document')
    def index_document_op(self, text: str, document_data: dict = None):
        try:
            return self.success('indiziert', **self.index_document(text, document_data))
        except Exception as e:
            return self.fail(str(e))

    @op("Ähnliche Chunks suchen. data_filter: {key: wert} oder {key: [wert1, wert2]} auf document_data",
        result={'success': 'bool', 'hits': 'list', 'context': 'str'}, name='query')
    def query_op(self, text: str, data_filter: dict = None, limit: int = 5, min_score: float = None):
        try:
            hits = self.query(text, data_filter, limit, min_score)
            return self.success('ok', hits=hits, context='\n\n---\n\n'.join(h['text'] for h in hits))
        except Exception as e:
            return self.fail(str(e), hits=[])

    @op("Alle Chunks eines Dokuments löschen", result={'success': 'bool', 'deleted': 'int'})
    def delete_document(self, doc_id: str):
        try:
            return self.success('gelöscht', deleted=self.store.delete_document(str(doc_id)))
        except Exception as e:
            return self.fail(str(e))

    @op("Indizierte Dokumente", result={'success': 'bool', 'documents': 'list'})
    def list_documents(self, limit: int = 100):
        try:
            return self.success('ok', documents=self.store.documents(limit))
        except Exception as e:
            return self.fail(str(e))

    @op("Backend und Anzahl der Chunks", result={'success': 'bool', 'backend': 'str', 'count': 'int'})
    def index_info(self):
        try:
            stats = self.store.stats() if hasattr(self.store, 'stats') else {}
            return self.success('ok', backend=self.store.name, count=self.store.count(), embedder=self.embedder,
                                stats=stats)
        except Exception as e:
            return self.fail(str(e), backend=str(self.backend))

    @op("Text testweise in Chunks zerlegen (ohne Embeddings)", result={'success': 'bool', 'chunks': 'list'})
    def chunk(self, text: str, chunk_size: int = 0, chunk_overlap: int = -1):
        return self.success('ok', chunks=chunk_text(text, chunk_size or self.chunk_size,
                                                    self.chunk_overlap if chunk_overlap < 0 else chunk_overlap))

    @op("local_index: IVF-Zentren neu berechnen (sonst automatisch ab min_train bzw. doppelter Größe)",
        result={'success': 'bool'})
    def train_index(self):
        if not hasattr(self.store, 'train'):
            return self.fail(f'Backend {self.store.name} hat keinen trainierbaren Index')
        try:
            self.store.train()
            return self.success('trainiert', stats=self.store.stats())
        except Exception as e:
            return self.fail(str(e))
