"""
opencode-Agent einer anderen Instanz über einen Job-Speicher (Standard: MongoDB-Runner mongodb).

Die Ops von nxml.agents:OpenCodeAgent gehen verschlüsselt (cenqueue, CryptRunner crypt nötig) mit guid und target in
den Speicher, die Ziel-Instanz führt sie mit ihrem opencode-Runner aus. Ein Thread fragt die Ergebnisse per guid ab
(cresult). Senden und Abfragen übernimmt nxrunner.jobstore:RemoteJobMixin (Projekt nxrunner).

Die Seiten sind die von OpenCodeAgent (Issues, Repos, Server), ihre Aufrufe laufen als asynchrone Jobs, die
Statuszeile zeigt ausstehende Jobs und Fehler. Die Provider- und Modellliste (list_providers) ändert sich selten und
wird zwischengespeichert, im Speicher und, falls vorhanden, in der nweb-Persistence (root.nweb, Setting
<type>.cache.<target>). set_model aktualisiert den Eintrag, op=cache_clear bzw. refresh=1 erzwingt einen neuen Abruf.

    Config (runners):
        ropencode: nxml.remote.opencode:RemoteOpenCode(target="gpu-pc")
        ropencode: nxml.remote.opencode:RemoteOpenCode(target="gpu-pc", store="mongodb", remote_type="opencode")

    Jobs (wie opencode, warten bis zur Antwort, höchstens wait Sekunden):
        {"type": "ropencode", "op": "do_issue", "name": "login", "task_text": "...", "path": "nxml"}
        {"type": "ropencode", "op": "list_issues"}
        {"type": "ropencode", "op": "list_providers", "refresh": 1}     neu abrufen statt aus dem Cache
        {"type": "ropencode", "op": "status", "async": 1}               nur einstellen, liefert remote_guid
        {"type": "ropencode", "op": "remote_result", "guid": "..."}     Ergebnis abholen (pending solange keins da ist)
        {"type": "ropencode", "op": "remote_state"}                     Statuszeile, ausstehende Jobs, Fehler, Cache
        {"type": "ropencode", "op": "cache_clear"}                      Cache (für target) leeren
        {"type": "ropencode", "op": "list_issues", "refresh": 1}        gemerkte Abfrage ignorieren

    Lesende Ops (QUERY_OPS) merkt er sich im RAM: eine gleiche, noch ausstehende Abfrage wird nicht erneut gesendet
    (gleiche remote_guid), ihr Ergebnis wird query_ttl Sekunden wiederverwendet. Schreibende Ops verwerfen das.

    Bei do_issue ist wait der Parameter von do_issue (auf das Ende des Issues warten), die Antwort der Ziel-Instanz
    wird dann bis zu timeout Sekunden abgewartet.

    Test ohne zweite Instanz:
        opencode: nxml.agents:OpenCodeAgent(project_path="...")
        store: nxrunner.jobstore:DirectJobStore
        ropencode: nxml.remote.opencode:RemoteOpenCode(target="local", store="store", poll_interval=1)
"""
import json
import threading
import time
from collections import OrderedDict

from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import util as u
from nwebclient import dev as d

from nxrunner.jobstore import RemoteJobMixin

from nxml.agents import OpenCodeAgent


def is_true(value) -> bool:
    return str(value) in ('1', 'true', 'True')


class RemoteOpenCode(OpenCodeAgent, RemoteJobMixin):
    """
    OpenCodeAgent einer anderen Instanz über einen Job-Speicher, gleiche Ops und gleiche Seiten wie opencode.
    """

    TAGS = ['agent', 'coding', 'remote']

    # Ops, die an den opencode-Runner der Ziel-Instanz gehen
    OPS = ['do_issue', 'status', 'server_start', 'server_stop', 'list_providers', 'set_model', 'list_issues',
           'session_messages', 'pairing', 'list_sessions', 'abort', 'ask', 'list_repos', 'select_repo']

    # lesende Ops: gleiche Abfragen werden im RAM gemerkt und nicht erneut gesendet (siehe query_ttl)
    QUERY_OPS = ['status', 'list_issues', 'session_messages', 'list_sessions', 'list_repos']

    # Felder, die nur lokal gelten und nicht mitgeschickt werden (wait außer bei do_issue)
    LOCAL_KEYS = ['type', 'job_type', 'async', 'wait', 'refresh']

    CACHE_PROVIDERS = 'providers'

    PAGES = OpenCodeAgent.PAGES + [('pending', 'Pending')]

    def __init__(self, type='remote_opencode', target='', store='mongodb', remote_type='opencode', wait=600,
                 cache_ttl=0, poll_interval=60, timeout=12000, keep=100, query_ttl=300):
        """
        :param target: Name der Ziel-Instanz (nwebclient.nx.get_name() dort, i.d.R. der Hostname)
        :param store: type des Job-Speicher-Runners
        :param remote_type: type des OpenCodeAgent-Runners auf der Ziel-Instanz
        :param wait: Sekunden, die ein Job ohne async=1 auf die Antwort wartet
        :param cache_ttl: Sekunden, die die Providerliste gilt, 0: bis zum Verwerfen (cache_clear, refresh=1)
        :param poll_interval: Sekunden zwischen den Abfragen der Ergebnisse
        :param timeout: Sekunden, nach denen eine Antwort verworfen wird
        :param keep: Anzahl abholbereiter Ergebnisse, ältere werden verworfen
        :param query_ttl: Sekunden, die das Ergebnis einer lesenden Op (QUERY_OPS) wiederverwendet wird, 0: unbegrenzt;
                          eine gleiche, noch ausstehende Abfrage wird nie erneut gesendet
        """
        OpenCodeAgent.__init__(self, start=False, type=type, args={})
        self.init_remote(target, store, remote_type, poll_interval, timeout)
        self.wait = float(wait)
        self.cache_ttl = float(cache_ttl)
        self.keep = int(keep)
        self.query_ttl = float(query_ttl)
        self.queries = {}              # Abfrage (JSON) -> {key, ts}, ts: Zeitpunkt der Antwort, None: ausstehend
        self.results = OrderedDict()   # key -> Ergebnis der Ziel-Instanz
        self.events = {}               # key -> Event für wartende Jobs
        self.cache = {}                # Cache-Schlüssel -> {ts, result}
        self.cache_loaded = False
        # gelten auf der Ziel-Instanz (set_model, select_repo), nicht lokal
        self.var_names = [v for v in self.var_names if v not in ('project_path', 'url', 'provider', 'model',
                                                                 'remote', 'repo')]
        self.define_vars('target', 'store', 'remote_type', 'wait', 'cache_ttl', 'query_ttl')
        self.define_sig(d.PStr('op', 'remote_result'), d.PStr('guid'))\
            .set_description("Ergebnis eines mit async=1 eingestellten Jobs, pending solange keins da ist")\
            .r(success='bool', pending='bool', result='dict')
        self.define_sig(d.PStr('op', 'remote_state'))\
            .set_description("Statuszeile (target, ausstehende Jobs, Fehler) und Cache-Einträge")\
            .r(success='bool', status='str', pending='int', error='str', cache='list')
        self.define_sig(d.PStr('op', 'remote_pending'))\
            .set_description("Jobs, die auf die Antwort der Ziel-Instanz warten")\
            .r(success='bool', jobs='list', target='str', error='str')
        self.define_sig(d.PStr('op', 'cache_clear'))\
            .set_description("Cache (Providerliste) für target leeren").r_success()

    def start_server(self, wait=30):
        raise RuntimeError('RemoteOpenCode startet keinen Server, op=server_start geht an die Ziel-Instanz')

    # --- RemoteJobMixin -------------------------------------------------------------------------

    def remote_job_label(self, job: dict):
        op = job.get('op', '')
        text = str(job.get('name') or job.get('text') or job.get('session_id') or job.get('model') or '')
        return op + (' ' + (text if len(text) <= 30 else text[:27] + '...') if text else '')

    def on_remote_result(self, meta: dict, result: dict):
        """ unter remote_lock """
        key = meta['key']
        if result.get('success', False):
            if meta.get('cache_key'):
                self.cache_put(meta['cache_key'], result)
            if meta.get('set_model'):
                self.cache_set_model(**meta['set_model'])
        query = self.queries.get(meta.get('query'))
        if query is not None and query['key'] == key:
            if result.get('success', False):
                query['ts'] = time.time()
            else:
                self.queries.pop(meta['query'], None)  # Fehler nicht merken, der nächste Aufruf fragt neu
        self.results[key] = result
        while len(self.results) > self.keep:
            self.results.popitem(last=False)
        event = self.events.pop(key, None)
        if event is not None:
            event.set()

    # --- Cache (nweb-Persistence, system__setting) --------------------------------------------

    def cache_store(self):
        """ lokale nweb-Persistence mit setting/set_setting oder None """
        root = self.getRoot()
        nw = getattr(root, 'nweb', None) if root is not None else None
        return nw if nw is not None and hasattr(nw, 'set_setting') and hasattr(nw, 'setting') else None

    def cache_setting_key(self) -> str:
        return f'{self.type}.cache.{self.target}'

    def ensure_cache_loaded(self):
        """ unter remote_lock: Cache einmalig laden, neuere Einträge haben Vorrang """
        if self.cache_loaded:
            return
        nw = self.cache_store()
        if nw is None:
            return
        self.cache_loaded = True
        try:
            value = json.loads(nw.setting(self.cache_setting_key(), '') or '{}')
            for k, entry in (value if isinstance(value, dict) else {}).items():
                if isinstance(entry, dict) and isinstance(entry.get('result'), dict):
                    self.cache.setdefault(str(k), dict(ts=entry.get('ts') or 0, result=entry['result']))
        except Exception as e:
            self.error("Cache nicht geladen: " + str(e))

    def save_cache(self):
        """ unter remote_lock """
        nw = self.cache_store()
        if nw is None:
            return
        self.ensure_cache_loaded()
        try:
            nw.set_setting(self.cache_setting_key(), json.dumps(self.cache, default=str))
        except Exception as e:
            self.error("Cache nicht gespeichert: " + str(e))

    def cache_put(self, cache_key, result: dict):
        """ unter remote_lock """
        self.cache[cache_key] = dict(ts=time.time(), result=result)
        self.save_cache()

    def cache_set_model(self, provider='', model=''):
        """ unter remote_lock: Standard-Modell in der gespeicherten Providerliste nachführen """
        entry = self.cache.get(self.CACHE_PROVIDERS)
        if entry is not None:
            entry['result'] = {**entry['result'], 'provider': provider, 'model': model}
            self.save_cache()

    def cache_lookup(self, cache_key):
        """ unter remote_lock: gültiger Eintrag oder None """
        self.ensure_cache_loaded()
        entry = self.cache.get(cache_key)
        if entry is None:
            return None
        if self.cache_ttl > 0 and time.time() - entry['ts'] > self.cache_ttl:
            return None
        return entry

    def cache_invalidate(self):
        with self.remote_lock:
            self.ensure_cache_loaded()
            self.cache = {}
            self.save_cache()

    # --- Senden und Abholen ---------------------------------------------------------------------

    def is_pending(self, key) -> bool:
        with self.remote_lock:
            return any(m.get('key') == key for m in self.remote_pending.values())

    def wait_for(self, key, wait: float) -> bool:
        """ :return: True, wenn die Antwort da bzw. nichts mehr aussteht """
        event = self.events.get(key)
        if event is None:
            return not self.is_pending(key)
        deadline = time.time() + wait
        while not event.wait(min(1.0, max(0.0, deadline - time.time()))):
            if not self.is_pending(key):
                return True
            if time.time() >= deadline:
                return False
        return True

    def take_result(self, key):
        """ :return: Ergebnis der Ziel-Instanz oder None (bleibt abholbar, mehrere Seiten können dieselbe guid haben) """
        with self.remote_lock:
            return self.results.get(key)

    def query_key(self, sig):
        """ unter remote_lock: guid einer gleichen ausstehenden bzw. noch gültigen beantworteten Abfrage oder None """
        query = self.queries.get(sig)
        if query is None:
            return None
        key = query['key']
        if query['ts'] is None:
            if key in self.events or any(m.get('key') == key for m in self.remote_pending.values()):
                return key
        elif key in self.results and (self.query_ttl <= 0 or time.time() - query['ts'] <= self.query_ttl):
            return key
        self.queries.pop(sig, None)
        return None

    def send_query(self, job: dict, wait=None, run_async=False, refresh=False):
        """ lesende Op: eine gleiche Abfrage nicht erneut senden, sondern deren guid bzw. Ergebnis verwenden """
        sig = json.dumps(job, sort_keys=True, default=str)
        with self.remote_lock:
            key = None if refresh else self.query_key(sig)
            if key is None:
                key = u.guid()
                self.queries[sig] = dict(key=key, ts=None)
                new = True
            else:
                new = False
        if new:
            return self.send(job, wait, run_async, key=key, query=sig)
        return self.answer(key, wait, run_async)

    def send(self, job: dict, wait=None, run_async=False, key=None, **meta):
        """ Job an die Ziel-Instanz, run_async: nur einstellen, sonst bis zu wait Sekunden auf die Antwort warten """
        key = key or u.guid()
        with self.remote_lock:
            # Events von Jobs, die nicht mehr ausstehen (verworfen), aufräumen
            pending = {m.get('key') for m in self.remote_pending.values()}
            for k in [k for k in self.events if k not in pending]:
                self.events.pop(k, None)
            self.events[key] = threading.Event()
        try:
            self.remote_send(job, key=key, **meta)
        except Exception as e:
            with self.remote_lock:
                self.events.pop(key, None)
                self.queries.pop(meta.get('query'), None)
            self.set_remote_error(str(e))
            return self.fail(str(e))
        return self.answer(key, wait, run_async)

    def answer(self, key, wait=None, run_async=False):
        """ Ergebnis zu key, run_async: nicht warten (pending mit remote_guid, solange keins da ist) """
        result = self.take_result(key)
        if result is not None:
            return result
        if run_async:
            return self.success('Gesendet, wartet auf Antwort', pending=True, remote_guid=key)
        if not self.wait_for(key, self.wait if wait in (None, '') else float(wait)):
            return self.success('Keine Antwort, Ergebnis über op=remote_result', pending=True, remote_guid=key)
        result = self.take_result(key)
        if result is None:
            return self.fail(self.remote_error or 'Keine Antwort von ' + self.target)
        return result

    def job(self, data: dict) -> dict:
        job = {k: v for k, v in data.items() if k not in self.LOCAL_KEYS}
        if data.get('op') == 'do_issue' and 'wait' in data:
            job['wait'] = data['wait']  # Parameter von do_issue, nicht die lokale Wartezeit
        return job

    def execute(self, data):
        op = data.get('op')
        if op == 'list_providers':
            return self.execute_list_providers(data)
        if op in self.OPS:
            run_async = is_true(data.get('async', 0))
            wait = self.timeout if op == 'do_issue' and is_true(data.get('wait', 0)) else data.get('wait')
            if op in self.QUERY_OPS:
                return self.send_query(self.job(data), wait, run_async, is_true(data.get('refresh', 0)))
            with self.remote_lock:
                self.queries = {}  # schreibende Op: gemerkte Abfragen sind danach veraltet
            meta = {}
            if op == 'set_model':
                meta['set_model'] = {'provider': str(data.get('provider', '')), 'model': str(data.get('model', ''))}
            return self.send(self.job(data), wait, run_async, **meta)
        return r.BaseJobExecutor.execute(self, data)

    # --- gecachte Ops ---------------------------------------------------------------------------

    def execute_list_providers(self, data={}):
        """ Providerliste aus dem Cache, sonst von der Ziel-Instanz (refresh=1 erzwingt einen Abruf) """
        if not is_true(data.get('refresh', 0)):
            with self.remote_lock:
                entry = self.cache_lookup(self.CACHE_PROVIDERS)
            if entry is not None:
                return {**entry['result'], 'cached_at': entry['ts']}
        return self.send(dict(op='list_providers'), data.get('wait'), is_true(data.get('async', 0)),
                         cache_key=self.CACHE_PROVIDERS)

    def execute_cache_clear(self, data={}):
        self.cache_invalidate()
        return self.success('Cache geleert')

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
            self.ensure_cache_loaded()
            status = self.remote_status()
            error = error or self.remote_error
            pending = len(self.remote_pending)
            queries = len(self.queries)
            cache = [dict(key=k, ts=e['ts'], success=bool(e['result'].get('success')))
                     for k, e in sorted(self.cache.items())]
        if error and error not in status:
            status += ' – ' + error
        return self.success(status=status, pending=pending, error=error, target=self.target, store=self.store,
                            cache=cache, queries=queries)

    def execute_remote_pending(self, data={}):
        """ wartende Jobs, älteste zuerst """
        now = time.time()
        with self.remote_lock:
            pending = sorted(self.remote_pending.values(), key=lambda m: m.get('seq', 0))
            jobs = [dict(guid=m.get('key') or m.get('guid'), label=m.get('label', ''), ts=m.get('ts'),
                         age=int(now - (m.get('ts') or now)), query=bool(m.get('query')),
                         cached=bool(m.get('cache_key')))
                    for m in pending]
            error = self.remote_error
        return self.success(jobs=jobs, target=self.target, error=error, timeout=self.timeout)

    def set_target(self, value):
        """ setvar target: Cache und Ergebnisse gelten je Ziel """
        value = str(value or '')
        with self.remote_lock:
            if value == self.target:
                return
            self.target = value
            self.cache, self.remote_pending, self.results, self.queries = {}, {}, OrderedDict(), {}
            self.remote_error = ''
            self.cache_loaded = False

    # --- Seiten ---------------------------------------------------------------------------------

    def opencode_cfg(self) -> dict:
        return {**super().opencode_cfg(), 'remote': True, 'target': self.target,
                'poll_ms': int(max(1.0, min(self.poll_interval, 5.0)) * 1000)}

    def part_pending(self, p: b.Page, params={}):
        self.part_page(p, 'pending', PENDING_HTML, PENDING_JS)


PENDING_HTML = """
<h3>Wartende Jobs</h3>
<div class="oc-row"><button id="oc-refresh">Aktualisieren</button> <span id="oc-count"></span></div>
<div class="oc-msg" id="oc-msg"></div>
<table class="oc-table" id="oc-pending"></table>
"""

PENDING_JS = r"""
function age(s) {
  return s < 60 ? s + ' s' : s < 3600 ? Math.floor(s / 60) + ' min ' + (s % 60) + ' s' : Math.floor(s / 3600) + ' h ' + Math.floor(s % 3600 / 60) + ' min';
}
function load() {
  send({ op: 'remote_pending' }).then((r) => {
    if (!r.success) { msg('oc-msg', r.message, true); return; }
    msg('oc-msg', r.error, !!r.error);
    $('oc-count').textContent = r.jobs.length + ' Job(s) warten auf ' + (r.target || '(kein target)');
    $('oc-pending').innerHTML = '<tr><th>Job</th><th>Gesendet</th><th>Wartet seit</th><th>Art</th><th>guid</th></tr>'
      + r.jobs.map((x) => '<tr><td><b>' + esc(x.label) + '</b></td>'
        + '<td>' + (x.ts ? new Date(x.ts * 1000).toLocaleTimeString() : '') + '</td>'
        + '<td>' + age(x.age) + (x.age > r.timeout * 0.8 ? ' <span class="oc-st err">läuft bald ab</span>' : '') + '</td>'
        + '<td>' + (x.query ? 'Abfrage (gemerkt)' : x.cached ? 'Abfrage (Cache)' : 'Aktion') + '</td>'
        + '<td><small><code>' + esc(x.guid) + '</code></small></td></tr>').join('')
      + (r.jobs.length ? '' : '<tr><td colspan="5"><i>keine wartenden Jobs</i></td></tr>');
  });
}
$('oc-refresh').onclick = load;
load();
setInterval(load, 5000);
"""
