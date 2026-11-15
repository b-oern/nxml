"""
Unittests fuer nxml.comfyui.remote (ComfyRemote, ComfyJobTable)

    python -m unittest tests.test_comfy_remote
"""
import sqlite3
import threading
import unittest

from nwebclient import runner as r

from nxrunner import jobstore as js
from nxml.comfyui import ComfyUi
from nxml.comfyui.remote import ComfyRemote, ComfyJobTable


class SqliteNWeb:
    """ nweb-Persistence (dml, select, insert wie nweb.DB) auf sqlite im Speicher """

    def __init__(self):
        self.connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.lock = threading.RLock()

    def dml(self, sql, vals=[]):
        with self.lock:
            self.connection.execute(sql, vals)
            self.connection.commit()

    def select(self, sql, params=[]):
        with self.lock:
            return self.connection.execute(sql, params).fetchall()

    def insert(self, table_name, row):
        cols = ','.join(row.keys())
        self.dml(f"INSERT INTO {table_name} ({cols}) VALUES ({','.join('?' for _ in row)})", list(row.values()))


class FakeComfy(r.BaseJobExecutor):
    """ ComfyUi der Ziel-Instanz mit festen Antworten """

    def __init__(self):
        super().__init__('comfyui')
        self.jobs = []
        self.prompts = {}

    def execute(self, data):
        self.jobs.append(dict(data))
        op = data.get('op')
        if op == 'instance_list':
            return self.success(workflows=[{'path': 'a.json', 'size': 10, 'modified': 1}])
        if op == 'instance_params':
            return self.success(workflow=data['workflow'], warnings=[], params=[
                {'node': '18', 'name': 'seed', 'label': 'seed', 'group': 'G', 'type': 'INT', 'value': 1,
                 'choices': [], 'options': {}, 'seed': True, 'randomize': True}])
        if op == 'instance_run':
            return self.success('1 Prompt(s) gestartet', prompt_ids=['P1', 'P2'], warnings=[])
        if op == 'instance_status':
            return self.success(prompts={pid: self.prompts.get(pid, {'status': 'queued', 'outputs': [], 'message': ''})
                                         for pid in data['prompt_ids']})
        return self.fail('unknown op')


class NoPoll:
    def start_remote_poller(self):
        pass


class TestComfyRemote(NoPoll, ComfyRemote):
    pass


def create_root(nw, *runners):
    dispatcher = r.LazyDispatcher()
    for runner in runners:
        dispatcher.add_runner(runner)
    root = r.JobRunner(dispatcher, {})
    root.nweb = nw
    for runner in runners:
        dispatcher.get_runner(runner.type)
    return root


class ComfyRemoteTestCase(unittest.TestCase):

    def setUp(self):
        self.nw = SqliteNWeb()
        self.comfy = FakeComfy()
        self.remote = TestComfyRemote(target='local', store='store')
        self.root = create_root(self.nw, js.DirectJobStore(type='store'), self.comfy, self.remote)

    def tearDown(self):
        self.nw.connection.close()

    def poll_all(self):
        for guid in list(self.remote.remote_pending.keys()):
            self.remote.remote_poll(guid)

    def test_workflows(self):
        self.assertTrue(self.remote.execute({'op': 'workflows'})['success'])
        self.poll_all()
        self.assertEqual(self.remote.workflows['workflows'][0]['path'], 'a.json')
        self.assertEqual(self.comfy.jobs[0], {'type': 'comfyui', 'op': 'instance_list'})

    def test_params(self):
        self.remote.execute({'op': 'params', 'workflow': 'a.json'})
        self.poll_all()
        self.assertEqual(self.remote.params['a.json']['params'][0]['name'], 'seed')

    def test_run_is_stored_and_tracked(self):
        res = self.remote.execute({'op': 'run', 'workflow': 'a.json', 'count': '2', 'p:18:seed': '5',
                                   'r:18:seed': False, 'other': 'x'})
        self.assertTrue(res['success'])
        guid = res['guid']
        row = self.remote.table.get(guid)
        self.assertEqual(row['status'], 'sent')
        self.assertEqual(row['params']['p:18:seed'], '5')
        self.assertNotIn('other', row['params'])
        self.poll_all()
        self.assertEqual(self.comfy.jobs[0]['op'], 'instance_run')
        self.assertEqual(self.comfy.jobs[0]['p:18:seed'], '5')
        row = self.remote.table.get(guid)
        self.assertEqual((row['status'], row['prompt_ids']), ('queued', ['P1', 'P2']))
        # Status: ein Prompt fertig -> running, beide fertig -> done
        self.comfy.prompts['P1'] = {'status': 'done', 'outputs': ['a.png'], 'message': ''}
        self.remote.execute({'op': 'check'})
        self.poll_all()
        self.assertEqual(self.remote.table.get(guid)['status'], 'running')
        self.comfy.prompts['P2'] = {'status': 'done', 'outputs': ['b.png'], 'message': ''}
        self.remote.execute({'op': 'check'})
        self.poll_all()
        row = self.remote.table.get(guid)
        self.assertEqual(row['status'], 'done')
        self.assertEqual(row['prompts']['P2']['outputs'], ['b.png'])

    def test_error_prompt(self):
        guid = self.remote.execute({'op': 'run', 'workflow': 'a.json'})['guid']
        self.poll_all()
        self.comfy.prompts['P1'] = {'status': 'error', 'outputs': [], 'message': 'OOM'}
        self.remote.execute({'op': 'check', 'guid': guid})
        self.poll_all()
        row = self.remote.table.get(guid)
        self.assertEqual((row['status'], row['message']), ('error', 'OOM'))

    def test_restart_restores_sent_runs(self):
        guid = self.remote.execute({'op': 'run', 'workflow': 'a.json'})['guid']
        restarted = TestComfyRemote(target='local', store='store')
        create_root(self.nw, js.DirectJobStore(type='store'), FakeComfy(), restarted)
        with restarted.remote_lock:
            restarted.jobs_table()
        self.assertIn(guid, restarted.remote_pending)

    def test_without_persistence(self):
        remote = TestComfyRemote(target='local', store='store')
        create_root(None, js.DirectJobStore(type='store'), FakeComfy(), remote)
        guid = remote.execute({'op': 'run', 'workflow': 'a.json'})['guid']
        self.assertFalse(remote.table.persistent)
        self.assertEqual(remote.table.get(guid)['status'], 'sent')

    def test_pages_render(self):
        from nwebclient import base
        self.remote.execute({'op': 'run', 'workflow': 'a.json'})
        self.poll_all()
        for part, params in ((self.remote.part_index, {}), (self.remote.part_workflow, {'w': 'a.json'}),
                             (self.remote.part_jobs, {}), (self.remote.part_job, {'guid': 'x'})):
            p = base.Page()
            part(p, params)
            self.assertIn('Workflows', str(p))
        self.poll_all()
        p = base.Page()
        self.remote.part_workflow(p, {'w': 'a.json'})
        self.assertIn('p:18:seed', str(p))
        self.assertIn('r:18:seed', str(p))


class ComfyUiStatusTestCase(unittest.TestCase):

    def test_prompt_status(self):
        c = ComfyUi(server_url='http://127.0.0.1:1', args={'comfyui': {}})
        history = {'A': {'status': {'status_str': 'success', 'messages': []},
                         'outputs': {'9': {'images': [{'filename': 'x.png'}]}}},
                   'B': {'status': {'status_str': 'error', 'messages': [
                       ['execution_error', {'exception_message': 'OOM'}]]}, 'outputs': {}}}
        c.history_job = lambda pid: {pid: history[pid]} if pid in history else {}
        queue = {'queue_running': [[0, 'C', {}]], 'queue_pending': [[1, 'D', {}]]}
        self.assertEqual(c.prompt_status('A', queue), {'status': 'done', 'outputs': ['x.png'], 'message': ''})
        self.assertEqual(c.prompt_status('B', queue)['message'], 'OOM')
        self.assertEqual(c.prompt_status('C', queue)['status'], 'running')
        self.assertEqual(c.prompt_status('D', queue)['status'], 'queued')
        self.assertEqual(c.prompt_status('E', queue)['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
