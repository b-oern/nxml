"""
Unittests fuer nxml.rag (RagRunner, Chunking, Filter, Speicher-Backends)

    python -m unittest tests.test_rag

Qdrant wird lokal im Speicher getestet (qdrant-client noetig), Postgres nur mit RAG_TEST_PG_DSN
(Datenbank mit pgvector, z.B. docker run -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg17).
"""
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unittest
import uuid

from nwebclient import runner as r

from nxml.rag import (RagRunner, FileVectorStore, PostgresVectorStore, LocalIndexStore, chunk_text, matches_filter,
                      normalize_filter, create_store)

DIMS = 64


def word_embed(texts):
    """ deterministisches Bag-of-Words-Embedding: gleiche Wörter -> ähnliche Vektoren """
    res = []
    for text in texts:
        v = [0.0] * DIMS
        for w in re.findall(r'\w+', text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIMS] += 1.0
        res.append(v)
    return res


DOCS = [
    ("Der Hund bellt laut im Garten. Hunde sind treue Haustiere.", {'doc_id': 'hund', 'lang': 'de', 'year': 2025,
                                                                   'tags': ['tier', 'haus']}),
    ("Die Katze schläft auf dem Sofa. Katzen jagen Mäuse.", {'doc_id': 'katze', 'lang': 'de', 'year': 2026,
                                                           'tags': ['tier']}),
    ("The dog barks loudly in the garden.", {'doc_id': 'dog', 'lang': 'en', 'year': 2026, 'public': True}),
]


class ChunkTest(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(chunk_text(''), [])
        self.assertEqual(chunk_text('   \n\n '), [])

    def test_short_text_one_chunk(self):
        self.assertEqual(chunk_text('Hallo Welt.'), ['Hallo Welt.'])

    def test_size_and_overlap(self):
        text = ' '.join(f'Satz Nummer {i} enthält ein paar Wörter.' for i in range(200))
        chunks = chunk_text(text, size=300, overlap=80)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        # Überlappung: der Anfang eines Chunks ist das Ende des vorherigen
        for a, b in zip(chunks, chunks[1:]):
            first_sentence = re.split(r'(?<=\.)\s+', b)[0]
            self.assertTrue(a.endswith(first_sentence) or first_sentence in a[-80:], (a[-80:], b[:80]))
            self.assertLessEqual(len(set(re.split(r'(?<=\.)\s+', a)) & set(re.split(r'(?<=\.)\s+', b))) * 39, 80)
        # nichts geht verloren
        for i in range(200):
            self.assertTrue(any(f'Nummer {i} ' in c for c in chunks), i)

    def test_paragraphs_and_long_words(self):
        text = 'Absatz eins.\n\nAbsatz zwei.\n\n' + 'x' * 500
        chunks = chunk_text(text, size=100, overlap=0)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertEqual(''.join(chunks).count('x'), 500)


class FilterTest(unittest.TestCase):

    def test_matches(self):
        data = {'lang': 'de', 'year': 2026, 'tags': ['a', 'b'], 'public': True}
        self.assertTrue(matches_filter(data, {}))
        self.assertTrue(matches_filter(data, {'lang': 'de', 'year': 2026}))
        self.assertFalse(matches_filter(data, {'lang': 'en'}))
        self.assertTrue(matches_filter(data, {'lang': ['en', 'de']}))
        self.assertTrue(matches_filter(data, {'tags': 'b'}))
        self.assertFalse(matches_filter(data, {'missing': 1}))
        self.assertTrue(matches_filter(data, {'public': True}))
        self.assertFalse(matches_filter({'n': 1}, {'n': True}))

    def test_normalize(self):
        self.assertEqual(normalize_filter(None), {})
        self.assertEqual(normalize_filter('{"lang": "de"}'), {'lang': 'de'})
        with self.assertRaises(ValueError):
            normalize_filter({'a; DROP': 1})
        with self.assertRaises(ValueError):
            normalize_filter({'a': {'$gt': 1}})

    def test_postgres_filter_sql(self):
        where, params = PostgresVectorStore.filter_sql({'lang': ['de', 'en'], 'year': 2026})
        self.assertEqual(where, '(data @> %s::jsonb OR data @> %s::jsonb) AND (data @> %s::jsonb)')
        self.assertEqual([json.loads(p) for p in params], [{'lang': 'de'}, {'lang': 'en'}, {'year': 2026}])
        self.assertEqual(PostgresVectorStore.filter_sql({}), ('TRUE', []))


class StoreContract:
    """ gemeinsame Tests für alle Backends, make_store() liefert einen leeren Speicher """

    def make_store(self):
        raise NotImplementedError

    def setUp(self):
        self.store = self.make_store()
        self.runner = RagRunner(store=self.store, embed_fn=word_embed, chunk_size=200, chunk_overlap=0)
        for text, data in DOCS:
            self.runner.index_document(text, data)

    def tearDown(self):
        try:
            self.store.clear()
        finally:
            self.store.close()

    def test_query_ranking(self):
        hits = self.runner.query('Katze Sofa', limit=3)
        self.assertEqual(hits[0]['doc_id'], 'katze')
        self.assertEqual(hits[0]['data']['lang'], 'de')
        self.assertGreaterEqual(hits[0]['score'], hits[-1]['score'])
        self.assertIn('text', hits[0])

    def test_filter(self):
        hits = self.runner.query('dog garden', {'lang': 'de'}, limit=10)
        self.assertTrue(hits)
        self.assertTrue(all(h['data']['lang'] == 'de' for h in hits))
        self.assertEqual({h['doc_id'] for h in self.runner.query('Tier', {'year': 2026, 'lang': 'de'})}, {'katze'})
        self.assertEqual({h['doc_id'] for h in self.runner.query('x', {'lang': ['en', 'fr']})}, {'dog'})
        self.assertEqual({h['doc_id'] for h in self.runner.query('x', {'tags': 'haus'})}, {'hund'})
        self.assertEqual({h['doc_id'] for h in self.runner.query('x', {'public': True})}, {'dog'})
        self.assertEqual(self.runner.query('x', {'lang': 'fr'}), [])

    def test_reindex_replaces(self):
        before = self.store.count()
        res = self.runner.index_document('Neuer Text über Vögel.', {'doc_id': 'hund', 'lang': 'de'})
        self.assertEqual(res, {'doc_id': 'hund', 'chunks': 1})
        hits = self.runner.query('Hund bellt', {'doc_id': 'hund'}, limit=10)
        self.assertEqual([h['text'] for h in hits], ['Neuer Text über Vögel.'])
        self.assertLessEqual(self.store.count(), before)

    def test_delete(self):
        self.assertEqual(self.store.delete_document('katze'), 1)
        self.assertEqual(self.store.delete_document('katze'), 0)
        self.assertNotIn('katze', {h['doc_id'] for h in self.runner.query('Katze', limit=10)})
        self.assertEqual(self.store.count(), 2)
        self.assertEqual({d['doc_id'] for d in self.store.documents()}, {'hund', 'dog'})


class MemoryStoreTest(StoreContract, unittest.TestCase):

    def make_store(self):
        return create_store('memory')

    def test_isolated_data(self):
        data = {'doc_id': 'neu', 'tags': ['x']}
        self.runner.index_document('Vogel singt.', data)
        data['tags'].append('y')                       # Aufrufer ändert sein dict
        hit = self.runner.query('Vogel', {'doc_id': 'neu'})[0]
        self.assertEqual(hit['data']['tags'], ['x'])
        hit['data']['tags'] = 'kaputt'                 # Treffer ändern
        self.assertEqual(self.runner.query('Vogel', {'doc_id': 'neu'})[0]['data']['tags'], ['x'])

    def test_runner_backend_and_clear(self):
        rag = RagRunner(backend='memory', embed_fn=word_embed)
        rag.index_document('Hallo Welt.', {'doc_id': 'a'})
        self.assertEqual(rag.execute({'op': 'index_info'})['backend'], 'memory')
        rag.store.clear()
        self.assertEqual(rag.store.count(), 0)
        rag.store.add([{'id': 'x#0', 'doc_id': 'x', 'chunk_index': 0, 'text': 'x', 'data': {},
                        'embedding': [1.0, 2.0]}])          # nach clear andere Dimension erlaubt
        self.assertEqual(rag.store.stats()['dims'], 2)


class FileStoreTest(StoreContract, unittest.TestCase):

    def make_store(self):
        self.dir = tempfile.mkdtemp(prefix='rag_test_')
        return FileVectorStore(self.dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_persistent(self):
        reopened = FileVectorStore(self.dir)
        self.assertEqual(reopened.count(), 3)
        query = word_embed(['Katze Sofa'])[0]
        self.assertEqual(reopened.search(query, 1)[0]['doc_id'], 'katze')
        self.assertEqual(reopened.search(query, 5, {'lang': 'en'})[0]['doc_id'], 'dog')

    def test_dims_mismatch(self):
        with self.assertRaises(ValueError):
            self.store.add([{'id': 'x#0', 'doc_id': 'x', 'chunk_index': 0, 'text': 'x', 'data': {},
                             'embedding': [1.0, 2.0]}])


try:
    import qdrant_client
except ImportError:
    qdrant_client = None


@unittest.skipIf(qdrant_client is None, 'qdrant-client nicht installiert')
class QdrantStoreTest(StoreContract, unittest.TestCase):

    def make_store(self):
        from qdrant_client import QdrantClient
        return create_store('qdrant', client=QdrantClient(location=':memory:'), collection='test')


PG_DSN = os.environ.get('RAG_TEST_PG_DSN')


@unittest.skipIf(not PG_DSN, 'RAG_TEST_PG_DSN nicht gesetzt')
class PostgresStoreTest(StoreContract, unittest.TestCase):

    def make_store(self):
        return create_store('postgres', dsn=PG_DSN, table='rag_test_' + uuid.uuid4().hex[:8])

    def tearDown(self):
        try:
            self.store._exec(f'DROP TABLE IF EXISTS {self.store.table}')
        finally:
            self.store.close()


class FakeEmbedder(r.BaseJobExecutor):
    """ wie LMStudio: {"op": "embed", "input": [...]} -> embeddings """

    def __init__(self, type='fake_embed'):
        super().__init__(type)
        self.calls = 0

    def execute_embed(self, data):
        self.calls += 1
        return {'success': True, 'embeddings': word_embed(data['input'])}


class FakeBert(r.BaseJobExecutor):
    """ wie BertEmbeddings: {"text": ...} -> embedding """

    def __init__(self, type='fake_bert'):
        super().__init__(type)

    def execute(self, data):
        if 'text' in data:
            return {'success': True, 'embedding': word_embed([data['text']])[0]}
        return super().execute(data)


class RunnerTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='rag_runner_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def dispatcher(self, embedder):
        rag = RagRunner(backend='file', path=self.dir, embedder=embedder.type, batch_size=1, chunk_size=60, chunk_overlap=0)
        dispatcher = r.LazyDispatcher()
        for runner in (rag, embedder):
            dispatcher.add_runner(runner)
        root = r.JobRunner(dispatcher, {})
        for runner in (rag, embedder):
            dispatcher.get_runner(runner.type)
        return root, rag

    def test_ops_via_dispatcher(self):
        embedder = FakeEmbedder()
        root, rag = self.dispatcher(embedder)
        text = 'Erster Satz über Hunde. Zweiter Satz über Katzen. Dritter Satz über Vögel. Vierter Satz über Fische.'
        res = root.execute({'type': 'rag', 'op': 'index_document', 'text': text,
                            'document_data': json.dumps({'id': 'tiere', 'lang': 'de'})})
        self.assertTrue(res['success'], res)
        self.assertEqual(res['doc_id'], 'tiere')
        self.assertGreaterEqual(res['chunks'], 2)
        self.assertEqual(embedder.calls, res['chunks'])  # batch_size=1
        res = root.execute({'type': 'rag', 'op': 'query', 'text': 'Vögel', 'data_filter': '{"lang": "de"}',
                            'limit': '1'})
        self.assertTrue(res['success'], res)
        self.assertEqual(len(res['hits']), 1)
        self.assertIn('Vögel', res['hits'][0]['text'])
        self.assertEqual(res['context'], res['hits'][0]['text'])
        res = root.execute({'type': 'rag', 'op': 'index_info'})
        self.assertEqual((res['backend'], res['count']), ('file', rag.store.count()))
        res = root.execute({'type': 'rag', 'op': 'delete_document', 'doc_id': 'tiere'})
        self.assertGreater(res['deleted'], 0)
        self.assertEqual(rag.store.count(), 0)

    def test_text_embedder_fallback(self):
        root, rag = self.dispatcher(FakeBert())
        res = rag.index_document('Hund. Katze.', {})
        self.assertEqual(res['doc_id'], hashlib.sha1('Hund. Katze.'.encode()).hexdigest())
        self.assertEqual(rag.query('Katze')[0]['doc_id'], res['doc_id'])

    def test_bad_filter(self):
        rag = RagRunner(path=self.dir, embed_fn=word_embed)
        res = rag.execute({'op': 'query', 'text': 'x', 'data_filter': {'a b': 1}})
        self.assertFalse(res['success'])


if __name__ == '__main__':
    unittest.main()
