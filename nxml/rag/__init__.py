#
# Project: nxml
#
# Retrieval Augmented Generation: Dokumente in Chunks zerlegen, Embeddings berechnen, ähnliche Chunks suchen.
#
#   index_document(text, document_data) -> chunk_text -> Embeddings (Runner, z.B. lmstudio op=embed) -> VectorStore
#   query(text, data_filter)            -> Embedding  -> VectorStore.search (Kosinus-Ähnlichkeit, Filter auf data)
#
# Speicher (backend):
#     memory       MemoryVectorStore    nur im Arbeitsspeicher, keine Persistenz
#     file         FileVectorStore      einfacher Index im Dateisystem (numpy-Matrix + JSONL), args: path
#     local_index  LocalIndexStore      effizienter lokaler IVF-Index (numpy + sqlite), args: path, nprobe
#     postgres     PostgresVectorStore  Postgres mit pgvector, args: dsn, table          (pip install psycopg[binary])
#     qdrant       QdrantVectorStore    Qdrant, args: url (oder path für lokal), collection (pip install qdrant-client)
#
# data_filter: einfache Gleichheit auf Werte von document_data, {"lang": "de", "year": 2026},
#              eine Liste als Wert bedeutet "einer davon", {"lang": ["de", "en"]}. Alle Bedingungen müssen gelten.
#
# Config (runners):
#     rag: nxml.rag:RagRunner     # args: backend=local_index, path=data/rag, embedder=text_embed
#
#
# Module:
#     nxml.rag.runner  RagRunner, chunk_text
#     nxml.rag.store   VectorStore, Filter, MemoryVectorStore, FileVectorStore, PostgresVectorStore, QdrantVectorStore, create_store
#     nxml.rag.index   LocalIndexStore (backend=local_index)
#     nxml.rag.ui      Web-Oberfläche mit Chat
#
from nxml.rag.store import (VectorStore, MemoryVectorStore, FileVectorStore, PostgresVectorStore,
                            QdrantVectorStore, BACKENDS, backend_class, create_store, matches_filter, normalize_filter)
from nxml.rag.runner import RagRunner, chunk_text

__all__ = ['RagRunner', 'chunk_text', 'VectorStore', 'MemoryVectorStore', 'FileVectorStore', 'PostgresVectorStore',
           'QdrantVectorStore', 'LocalIndexStore', 'BACKENDS', 'backend_class', 'create_store', 'matches_filter',
           'normalize_filter']


def __getattr__(name):
    if name == 'LocalIndexStore':
        from nxml.rag.index import LocalIndexStore
        return LocalIndexStore
    raise AttributeError(name)
