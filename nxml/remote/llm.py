"""
LM Studio einer anderen Instanz über einen Job-Speicher (Standard: MongoDB-Runner mongodb).

Die Ops von nxml.llm:LMStudio gehen verschlüsselt (cenqueue, CryptRunner crypt nötig) mit guid und target in den
Speicher, die Ziel-Instanz führt sie mit ihrem lmstudio-Runner aus. Ein Thread fragt die Ergebnisse per guid ab
(cresult). Senden und Abfragen übernimmt nxrunner.jobstore:RemoteJobMixin (Projekt nxrunner).

    Config (runners):
        rlms: nxml.remote.llm:RemoteLmStudio(target="gpu-pc")
        rlms: nxml.remote.llm:RemoteLmStudio(target="gpu-pc", store="mongodb", remote_type="lmstudio")

    Jobs (wie lmstudio, warten bis zur Antwort, höchstens wait Sekunden):
        {"type": "rlms", "prompt": "Hallo"}
        {"type": "rlms", "op": "chat", "messages": [...], "model": "..."}
        {"type": "rlms", "op": "models", "async": 1}         nur einstellen, liefert remote_guid
        {"type": "rlms", "op": "remote_result", "guid": "..."}   Ergebnis abholen (pending solange keins da ist)
        {"type": "rlms", "op": "remote_state"}               Statuszeile, ausstehende Jobs, Fehler

    Test ohne zweite Instanz:
        lmstudio: nxml.llm:LMStudio
        store: nxrunner.jobstore:DirectJobStore
        rlms: nxml.remote.llm:RemoteLmStudio(target="local", store="store", poll_interval=1)
"""
import threading
import time
from collections import OrderedDict

from nwebclient import runner as r
from nwebclient import util as u
from nwebclient import dev as d

from nxrunner.jobstore import RemoteJobMixin

from nxml.llm import BaseLLM, LMStudioUiMixin


class RemoteLmStudio(LMStudioUiMixin, BaseLLM, RemoteJobMixin):
    """
    LMStudio-Runner einer anderen Instanz über einen Job-Speicher, gleiche Ops und gleiche UI wie lmstudio.
    """

    TAGS = ['llm', 'remote']

    # Ops, die an den lmstudio-Runner der Ziel-Instanz gehen
    OPS = ['status', 'models', 'chat', 'completion', 'embed', 'load', 'unload', 'download', 'download_status',
           'server_start', 'server_stop', 'daemon', 'cli']

    # Felder, die nur lokal gelten und nicht mitgeschickt werden
    LOCAL_KEYS = ['type', 'job_type', 'async', 'wait']

    def __init__(self, type='remote_lmstudio', target='', store='mongodb', remote_type='lmstudio', model='',
                 embedding_model='', wait=600, poll_interval=10, timeout=6000, keep=100):
        """
        :param target: Name der Ziel-Instanz (nwebclient.nx.get_name() dort, i.d.R. der Hostname)
        :param store: type des Job-Speicher-Runners
        :param remote_type: type des LMStudio-Runners auf der Ziel-Instanz
        :param model: Standard-Modell für prompt/chat, leer: Standard der Ziel-Instanz
        :param wait: Sekunden, die ein Job ohne async=1 auf die Antwort wartet
        :param poll_interval: Sekunden zwischen den Abfragen der Ergebnisse
        :param timeout: Sekunden, nach denen eine Antwort verworfen wird
        :param keep: Anzahl abholbereiter Ergebnisse, ältere werden verworfen
        """
        super().__init__(type)
        self.init_remote(target, store, remote_type, poll_interval, timeout)
        self.model = model
        self.embedding_model = embedding_model
        self.wait = float(wait)
        self.keep = int(keep)
        self.results = OrderedDict()   # key -> Ergebnis der Ziel-Instanz
        self.events = {}               # key -> Event für wartende Jobs
        self.define_vars('target', 'store', 'remote_type', 'model', 'embedding_model', 'wait')
        self.define_lmstudio_sigs()
        self.define_sig(d.PStr('op', 'remote_result'), d.PStr('guid'))\
            .set_description("Ergebnis eines mit async=1 eingestellten Jobs, pending solange keins da ist")\
            .r(success='bool', pending='bool', result='dict')
        self.define_sig(d.PStr('op', 'remote_state'))\
            .set_description("Statuszeile (target, ausstehende Jobs, Fehler)")\
            .r(success='bool', status='str', pending='int', error='str')

    # --- RemoteJobMixin -------------------------------------------------------------------------

    def remote_job_label(self, job: dict):
        op = job.get('op', 'prompt')
        text = job.get('prompt') or job.get('model') or job.get('cmd') or ''
        text = str(text)
        return op + (' ' + (text if len(text) <= 30 else text[:27] + '...') if text else '')

    def on_remote_result(self, meta: dict, result: dict):
        key = meta['key']
        self.results[key] = result
        while len(self.results) > self.keep:
            self.results.popitem(last=False)
        event = self.events.get(key)
        if event is not None:
            event.set()

    # --- Senden und Abholen ---------------------------------------------------------------------

    def is_pending(self, key) -> bool:
        with self.remote_lock:
            return any(m.get('key') == key for m in self.remote_pending.values())

    def take_result(self, key):
        """ :return: Ergebnis der Ziel-Instanz oder None """
        with self.remote_lock:
            self.events.pop(key, None)
            return self.results.pop(key, None)

    def send(self, job: dict, wait=None, run_async=False):
        """ Job an die Ziel-Instanz, run_async: nur einstellen, sonst bis zu wait Sekunden auf die Antwort warten """
        key = u.guid()
        if not run_async:
            self.events[key] = threading.Event()
        try:
            self.remote_send(job, key=key)
        except Exception as e:
            self.events.pop(key, None)
            self.set_remote_error(str(e))
            return self.fail(str(e))
        if run_async:
            return self.success('Gesendet, wartet auf Antwort', pending=True, remote_guid=key)
        deadline = time.time() + (self.wait if wait in (None, '') else float(wait))
        event = self.events[key]
        while not event.wait(1):
            if not self.is_pending(key):
                break
            if time.time() > deadline:
                self.events.pop(key, None)
                return self.success('Keine Antwort, Ergebnis über op=remote_result', pending=True, remote_guid=key)
        result = self.take_result(key)
        if result is None:
            return self.fail(self.remote_error or 'Keine Antwort von ' + self.target)
        return result

    def job(self, data: dict) -> dict:
        job = {k: v for k, v in data.items() if k not in self.LOCAL_KEYS}
        if job.get('op', 'chat') in ('chat', 'completion') and not job.get('model') and self.model:
            job['model'] = self.model
        if job.get('op') == 'embed' and not job.get('model') and self.embedding_model:
            job['model'] = self.embedding_model
        return job

    @staticmethod
    def is_async(data) -> bool:
        return str(data.get('async', 0)) in ('1', 'true', 'True')

    def prompt(self, prompt, data={}):
        job = self.job({**data, 'op': 'chat', 'prompt': prompt})
        job.pop('cprompt', None)
        return self.send(job, data.get('wait'), self.is_async(data))

    def execute(self, data):
        op = data.get('op')
        if op in self.OPS:
            self.count += 1
            return self.send(self.job(data), data.get('wait'), self.is_async(data))
        if op is not None:
            return r.BaseJobExecutor.execute(self, data)
        return super().execute(data)

    def execute_remote_result(self, data):
        key = str(data.get('guid', ''))
        result = self.take_result(key)
        if result is not None:
            return self.success('ok', pending=False, result=result)
        if self.is_pending(key):
            return self.success('pending', pending=True)
        return self.fail(self.remote_error or f'Kein Ergebnis für {key}')

    def execute_remote_state(self, data={}):
        error = self.check_store() if self.target else 'target (Name der Ziel-Instanz) ist nicht gesetzt'
        with self.remote_lock:
            status = self.remote_status()
            error = error or self.remote_error
            pending = len(self.remote_pending)
        if error and error not in status:
            status += ' – ' + error
        return self.success(status=status, pending=pending, error=error, target=self.target, store=self.store)

    # --- Seite ----------------------------------------------------------------------------------

    def lmstudio_cfg(self) -> dict:
        return {'type': self.type, 'url': f'{self.remote_type}@{self.target}', 'model': self.model,
                'embedding_model': self.embedding_model, 'remote': True,
                'poll_ms': int(max(1.0, min(self.poll_interval, 5.0)) * 1000)}
