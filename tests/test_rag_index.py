"""
Unittests fuer nxml.rag.index (LocalIndexStore, backend=local_index) sowie Chat und Web-UI des RagRunner

    python -m unittest tests.test_rag_index
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from nwebclient import runner as r
from nwebclient import base as b

from nxml.rag import RagRunner
from nxml.rag.index import LocalIndexStore
from tests.test_rag import StoreContract, FakeEmbedder, word_embed


class LocalIndexContractTest(StoreContract, unittest.TestCase):
    """ gleiche Tests wie für file, qdrant, postgres """

    def make_store(self):
        self.dir = tempfile.mkdtemp(prefix='rag_local_')
        return LocalIndexStore(self.dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.dir, ignore_errors=True)


class LocalIndexTest(unittest.TestCase):
    """ IVF mit vielen Zufallsvektoren in Clustern """

    N, D = 6000, 32

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='rag_ivf_')
        rng = np.random.default_rng(1)
        centers = rng.normal(size=(40, self.D))
        self.vectors = (centers[rng.integers(0, 40, self.N)] + 0.35 * rng.normal(size=(self.N, self.D))).astype('f4')
        self.store = LocalIndexStore(self.dir, min_train=2000, exact_limit=500)
        for s in range(0, self.N, 1000):
            self.store.add([{'id': f'd{i}#0', 'doc_id': f'd{i}', 'chunk_index': 0, 'text': f't{i}',
                             'data': {'group': i % 10, 'even': i % 2 == 0, 'tags': ['a', 'b' if i % 3 else 'c']},
                             'embedding': self.vectors[i].tolist()} for i in range(s, s + 1000)])

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def exact(self, q, k, mask=None):
        v = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        s = v @ (q / np.linalg.norm(q))
        if mask is not None:
            s[~mask] = -9
        return {f'd{i}' for i in np.argsort(-s)[:k]}

    def test_trained_and_recall(self):
        stats = self.store.stats()
        self.assertGreater(stats['lists'], 8)
        self.assertGreaterEqual(stats['trained_n'], 2000)
        self.assertEqual(self.store.count(), self.N)
        recall = []
        for i in range(0, 300, 10):
            q = self.vectors[i] + 0.1
            got = {h['doc_id'] for h in self.store.search(q.tolist(), 10)}
            recall.append(len(got & self.exact(q, 10)) / 10)
        self.assertGreater(sum(recall) / len(recall), 0.9)

    def test_filter(self):
        idx = np.arange(self.N)
        q = self.vectors[7]
        # viele Treffer (IVF) und wenige (exakt)
        hits = self.store.search(q.tolist(), 10, {'group': [3, 7]})
        self.assertTrue(hits and all(h['data']['group'] in (3, 7) for h in hits))
        self.assertEqual(hits[0]['doc_id'], 'd7')
        got = {h['doc_id'] for h in self.store.search(q.tolist(), 5, {'group': 7, 'even': False, 'tags': 'c'})}
        mask = (idx % 10 == 7) & (idx % 2 == 1) & (idx % 3 == 0)
        self.assertEqual(got, self.exact(q, 5, mask))
        self.assertEqual(self.store.search(q.tolist(), 5, {'group': 99}), [])
        self.assertEqual(len(self.store.search(q.tolist(), 5, {'group': 7.0})), 5)
        # bool und int werden nicht verwechselt
        self.assertEqual(self.store.search(q.tolist(), 5, {'even': 1}), [])

    def test_persistent_delete_compact(self):
        for i in range(0, 2500, 500):
            self.store.delete_document(f'd{i}')
        self.store.close()
        store = self.store = LocalIndexStore(self.dir, min_train=2000, exact_limit=500)
        self.assertEqual(store.count(), self.N - 5)
        self.assertEqual(store.search(self.vectors[1].tolist(), 1)[0]['doc_id'], 'd1')
        # viele löschen -> kompaktieren, Zeilen neu nummeriert
        gen = store.gen
        for i in range(3000):
            store.delete_document(f'd{i}')
        self.assertGreater(store.gen, gen)
        self.assertLessEqual(store.rows_total - store.count(), store.compact_ratio * store.rows_total)
        store.compact()
        self.assertEqual(store.rows_total, store.count())
        self.assertEqual(store.count(), self.N - 3000)
        self.assertEqual(store.search(self.vectors[4321].tolist(), 1)[0]['doc_id'], 'd4321')
        self.assertEqual(store.search(self.vectors[10].tolist(), 3, {'group': 0})[0]['data']['group'], 0)
        store.close()
        store = self.store = LocalIndexStore(self.dir)
        self.assertEqual(store.count(), self.N - 3000)
        self.assertEqual(store.search(self.vectors[5000].tolist(), 1)[0]['doc_id'], 'd5000')
        files = [f for f in os.listdir(self.dir) if f.startswith('vectors.')]
        self.assertEqual(files, [f'vectors.{store.gen}.f32'])

    def test_replace_same_id(self):
        self.store.add([{'id': 'd5#0', 'doc_id': 'd5', 'chunk_index': 0, 'text': 'neu', 'data': {'group': 42},
                         'embedding': self.vectors[9].tolist()}])
        self.assertEqual(self.store.count(), self.N)
        hits = self.store.search(self.vectors[9].tolist(), 5, {'group': 42})
        self.assertEqual([h['text'] for h in hits], ['neu'])


class FakeLlm(r.BaseJobExecutor):
    """ wie LMStudio: op=chat mit messages """

    TAGS = ['llm']

    def __init__(self, type='fake_llm'):
        super().__init__(type)
        self.messages = None

    def execute_chat(self, data):
        m = data['messages']
        self.messages = m if isinstance(m, list) else json.loads(m)
        return {'success': True, 'response': 'Antwort [1]', 'model': 'fake'}


class PromptLlm(r.BaseJobExecutor):
    """ LLM nur mit prompt """

    TAGS = ['llm']

    def __init__(self, type='prompt_llm'):
        super().__init__(type)
        self.prompts = []

    def prompt(self, prompt, data={}):
        self.prompts.append(prompt)
        return {'success': True, 'response': 'ok'}

    def execute(self, data):
        if 'prompt' in data:
            return self.prompt(data['prompt'], data)
        return super().execute(data)


class ChatUiTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='rag_chat_')
        self.llm, self.plain = FakeLlm(), PromptLlm()
        self.rag = RagRunner(path=self.dir, embedder='fake_embed', backend='local_index', chunk_size=60,
                             chunk_overlap=0)
        runners = (self.rag, FakeEmbedder(), self.llm, self.plain)
        dispatcher = r.LazyDispatcher()
        for runner in runners:
            dispatcher.add_runner(runner)
        self.root = r.JobRunner(dispatcher, {})
        for runner in runners:
            dispatcher.get_runner(runner.type)
        self.rag.index_document('Katzen schlafen viel. Hunde bellen laut.', {'doc_id': 'tiere', 'lang': 'de'})

    def tearDown(self):
        self.rag.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_llms(self):
        res = self.root.execute({'type': 'rag', 'op': 'llms'})
        self.assertEqual([x['type'] for x in res['llms']], ['fake_llm', 'prompt_llm'])
        self.assertEqual([x['chat'] for x in res['llms']], [True, False])
        self.assertEqual(res['default'], 'fake_llm')

    def test_chat_messages(self):
        res = self.root.execute({'type': 'rag', 'op': 'chat', 'question': 'Was machen Katzen?', 'llm': 'fake_llm',
                                 'data_filter': '{"lang": "de"}', 'limit': '2',
                                 'history': json.dumps([{'role': 'user', 'content': 'Hallo'},
                                                        {'role': 'assistant', 'content': 'Hi'}])})
        self.assertTrue(res['success'], res)
        self.assertEqual(res['response'], 'Antwort [1]')
        self.assertTrue(res['hits'])
        self.assertEqual([m['role'] for m in self.llm.messages], ['system', 'user', 'assistant', 'user'])
        self.assertIn('Katzen schlafen', self.llm.messages[0]['content'])

    def test_chat_prompt_only(self):
        res = self.root.execute({'type': 'rag', 'op': 'chat', 'question': 'Hunde?', 'llm': 'prompt_llm'})
        self.assertTrue(res['success'], res)
        self.assertIn('Hunde bellen', self.plain.prompts[0])

    def test_pages(self):
        for part in (self.rag.part_index, self.rag.part_search, self.rag.part_add, self.rag.part_documents,
                     self.rag.part_status):
            p = b.Page()
            part(p, {})
            self.assertIn('class="rag"', str(p) if not hasattr(p, 'nxui') else p.nxui())
        self.assertTrue(self.root.execute({'type': 'rag', 'op': 'train_index'})['success'])

    def test_clip_features_fallback(self):
        class FakeClip(r.BaseJobExecutor):
            def __init__(self):
                super().__init__('fake_clip')

            def execute(self, data):
                if 'text' in data:
                    return {'success': True, 'features': [word_embed([data['text']])[0]], 'embeddings': [[0.0]]}
                return super().execute(data)

        clip = FakeClip()
        self.root.jobexecutor.add_runner(clip) if hasattr(self.root, 'jobexecutor') else None
        self.root.jobexecutor.get_runner('fake_clip')
        rag = RagRunner(path=os.path.join(self.dir, 'clip'), embedder='fake_clip')
        rag.getRoot = lambda: self.root
        rag.index_document('Hund. Katze.', {'doc_id': 'c'})
        self.assertEqual(rag.query('Katze')[0]['doc_id'], 'c')


class FakeSentenceTransformer:
    """ statt sentence_transformers.SentenceTransformer, merkt sich die Texte """

    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        self.seen.extend(texts)
        v = np.array(word_embed([t.split(': ', 1)[-1] for t in texts]), dtype='f4')
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


class TextEmbeddingsTest(unittest.TestCase):

    def make(self, model_name):
        from nxml.analyse import TextEmbeddings
        te = TextEmbeddings(model_name=model_name)
        te.st_model, te.hf_loaded = FakeSentenceTransformer(), True
        return te

    def test_default_bge_m3(self):
        te = self.make(None)
        self.assertEqual(te.model_name, 'BAAI/bge-m3')
        res = te.execute({'op': 'embed', 'input': '["Hund", "Katze"]', 'input_type': 'query'})
        self.assertTrue(res['success'], res)
        self.assertEqual((len(res['embeddings']), res['dims']), (2, 64))
        self.assertEqual(te.st_model.seen, ['Hund', 'Katze'])  # bge-m3 ohne Präfix
        res = te.execute({'text': 'Ein Text'})
        self.assertEqual(len(res['embedding']), 64)
        self.assertEqual(res['embedding_model'], 'BAAI/bge-m3')

    def test_e5_prefixes_via_rag(self):
        te = self.make('intfloat/multilingual-e5-base')
        dir = tempfile.mkdtemp(prefix='rag_te_')
        try:
            rag = RagRunner(path=dir, embedder=te.type, backend='local_index')
            dispatcher = r.LazyDispatcher()
            for runner in (rag, te):
                dispatcher.add_runner(runner)
            r.JobRunner(dispatcher, {})
            for runner in (rag, te):
                dispatcher.get_runner(runner.type)
            rag.index_document('Der Hund bellt.', {'doc_id': 'h'})
            hits = rag.query('Hund')
            self.assertEqual(te.st_model.seen, ['passage: Der Hund bellt.', 'query: Hund'])
            self.assertEqual(hits[0]['doc_id'], 'h')
            rag.store.close()
        finally:
            shutil.rmtree(dir, ignore_errors=True)

    def test_models(self):
        res = self.make(None).execute({'op': 'models'})
        self.assertIn('intfloat/multilingual-e5-base', [m['model'] for m in res['models']])


class FakeNWeb:
    """ wie NWebClient: groups(), group(), doc(), req('api/documents/<id>'), url() """

    def __init__(self):
        from nwebclient import NWebGroup
        self.verbose = False
        rows = [dict(group_id=1, parent=0, title='Wissen', guid='g1'),
                dict(group_id=2, parent=1, title='Handbücher', guid='g2'),
                dict(group_id=3, parent=0, title='Bilder', guid='g3')]
        self._groups = [NWebGroup(self, r_) for r_ in rows]
        self.docs = {
            1: [dict(document_id=10, guid='d10', name='a', title='Katzen', kind='markup',
                     content='<p>Katzen schlafen <b>viel</b>.</p>'),
                dict(document_id=11, guid='d11', name='b', title='Foto', kind='image', content='')],
            2: [dict(document_id=20, guid='d20', name='c', title='Hund', kind='md')],   # ohne content in der Liste
            3: [],
        }
        self.contents = {20: 'Hunde bellen laut.'}

    def url(self):
        return 'https://nweb.example/'

    def groups(self, refresh=False):
        return self._groups

    def group(self, gid):
        return next(g for g in self._groups if str(g.id()) == str(gid))

    def req(self, path, params={}):
        return json.dumps({'items': self.docs[int(path.rsplit('/', 1)[1])]})

    def doc(self, id):
        from nwebclient import NWebDoc
        d = next(d for items in self.docs.values() for d in items if d['document_id'] == int(id))
        return NWebDoc(self, dict(d, content=self.contents.get(int(id), d.get('content'))))


class NWebUiTest(unittest.TestCase):

    def setUp(self):
        self.rag = RagRunner(backend='memory', embed_fn=word_embed)
        self.nweb = FakeNWeb()
        self.rag.nweb_instances = lambda: {'test': 'https://nweb.example/'}
        self.rag.nweb_client = lambda name: self.nweb

    def test_tree(self):
        roots, children = self.rag.nweb_group_tree(self.nweb.groups())
        self.assertEqual([g.title() for g in roots], ['Bilder', 'Wissen'])
        self.assertEqual([g.title() for g in children['1']], ['Handbücher'])

    def test_index_group(self):
        res = self.rag.execute({'op': 'index_nweb_group', 'instance': 'test', 'group_id': '1'})
        self.assertTrue(res['success'], res)
        self.assertEqual((res['indexed'], res['skipped']), (1, 1))
        hit = self.rag.query('Katzen schlafen', {'source': 'nweb'})[0]
        self.assertEqual(hit['doc_id'], 'nweb:test:d10')
        self.assertEqual(hit['data']['group'], 'Wissen')
        self.assertEqual(hit['data']['url'], 'https://nweb.example/d/10/')
        self.assertNotIn('<b>', hit['text'])
        res = self.rag.execute({'op': 'index_nweb_group', 'instance': 'test', 'group_id': '1', 'recursive': '1'})
        self.assertEqual((res['indexed'], res['groups']), (2, 2))
        hit = self.rag.query('Hunde bellen', {'group_id': '2'})[0]
        self.assertIn('Hunde bellen laut.', hit['text'])
        self.assertTrue(hit['text'].startswith('Hund'))

    def test_pages(self):
        for part, params in ((self.rag.part_nweb, {}), (self.rag.part_nweb_instance, {'n': 'test'}),
                             (self.rag.part_nweb__group, {'n': 'test', 'g': '2'})):
            p = b.Page()
            part(p, params)
            html = str(p) if not hasattr(p, 'nxui') else p.nxui()
            self.assertIn('class="rag"', html)
        p = b.Page()
        self.rag.part_nweb_instance(p, {'n': 'test'})
        html = str(p)
        self.assertIn('Handbücher', html)
        self.assertIn('rag-tree', html)
        p = b.Page()
        self.rag.part_nweb__group(p, {'n': 'test', 'g': '2'})
        html = str(p)
        self.assertIn('Gruppe indizieren', html)
        self.assertIn('Wissen', html)          # Elterngruppe im Pfad
        self.assertIn('"test"', html)          # Instanz im JS

    def test_instances_from_args(self):
        rag = RagRunner(backend='memory', embed_fn=word_embed)
        rag.args = {'meinnweb': {'url': 'https://m.example/4.0', 'username': 'u', 'password': 'p'}, 'DB': 'x'}
        self.assertEqual(rag.nweb_instances().get('meinnweb'), 'https://m.example/4.0')
        self.assertEqual(rag.nweb_client('meinnweb').url(), 'https://m.example/4.0/')
        with self.assertRaises(KeyError):
            rag.nweb_client('gibtsnicht')

    def test_unknown_instance(self):
        self.rag.nweb_client = RagRunner.nweb_client.__get__(self.rag)
        res = self.rag.execute({'op': 'index_nweb_group', 'instance': 'gibtsnicht', 'group_id': '1'})
        self.assertFalse(res['success'])


if __name__ == '__main__':
    unittest.main()
