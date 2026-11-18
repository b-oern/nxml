"""
ComfyUI einer anderen Instanz über einen Job-Speicher (Standard: MongoDB-Runner mongodb).

Die Ops von nxml.comfyui:ComfyUi gehen verschlüsselt (cenqueue, CryptRunner crypt nötig) mit guid und target in den
Speicher, die Ziel-Instanz führt sie mit ihrem comfyui-Runner aus. Ein Thread fragt die Ergebnisse per guid ab
(cresult). Senden und Abfragen übernimmt nxrunner.jobstore:RemoteJobMixin (Projekt nxrunner).

Die Seiten sind die von ComfyUi (Workflow-Tabelle, Vorbereiten, Ausführen). Was sich selten ändert – Konfiguration
der Ziel-Instanz (server_info), Liste und Inhalt der Workflows – wird zwischengespeichert, im Speicher und, falls
vorhanden, in der nweb-Persistence (root.nweb, Setting <type>.cache.<target>). So kosten die Seiten nach dem ersten
Abruf keine Remote-Aufrufe mehr. workflow_info wird lokal aus dem gespeicherten Workflow berechnet.
workflow_prepare und workflow_upload verwerfen die betroffenen Einträge, op=cache_clear bzw. refresh=1 erzwingt
einen neuen Abruf. Offene Abrufe für den Cache überstehen einen Neustart (Setting <type>.pending.<target>).

    Config (runners):
        rcomfy: nxml.remote.comfy:RemoteComfy(target="gpu-pc")
        rcomfy: nxml.remote.comfy:RemoteComfy(target="gpu-pc", store="mongodb", remote_type="comfyui", cache_ttl=86400)

    Jobs (wie comfyui, warten bis zur Antwort, höchstens wait Sekunden):
        {"type": "rcomfy", "workflow": "/pfad/w.json", "prompt": "Katze", "count": 2}
        {"type": "rcomfy", "op": "rest", "route": "system_stats"}
        {"type": "rcomfy", "op": "queue", "prompt": "...", "async": 1}    nur einstellen, liefert remote_guid
        {"type": "rcomfy", "op": "workflows", "refresh": 1}               Liste neu abrufen statt aus dem Cache
        {"type": "rcomfy", "op": "workflow_info", "workflow": "..."}      lokal aus dem gespeicherten Workflow
        {"type": "rcomfy", "op": "remote_result", "guid": "..."}          Ergebnis abholen (pending solange keins da ist)
        {"type": "rcomfy", "op": "remote_state"}                          Statuszeile, ausstehende Jobs, Fehler, Cache
        {"type": "rcomfy", "op": "cache_clear"}                           Cache (für target) leeren

    Test ohne zweite Instanz:
        comfyui: nxml.comfyui:ComfyUi
        store: nxrunner.jobstore:DirectJobStore
        rcomfy: nxml.remote.comfy:RemoteComfy(target="local", store="store", poll_interval=1)
"""
import copy
import html
import json
import threading
import time
from collections import OrderedDict

from nwebclient import runner as r
from nwebclient import base
from nwebclient import util as u
from nwebclient import dev as d

from nxrunner.jobstore import RemoteJobMixin

from nxml.comfyui import ComfyUi


def is_true(value) -> bool:
    return str(value) in ('1', 'true', 'True')


class RemoteComfy(ComfyUi, RemoteJobMixin):
    """
    ComfyUi-Runner einer anderen Instanz über einen Job-Speicher, gleiche Ops und gleiche Seiten wie comfyui.
    """

    TAGS = ['image', 'remote']

    # Ops, die unverändert an den comfyui-Runner der Ziel-Instanz gehen (Jobs ohne op ebenso)
    OPS = ['rest', 'queue', 'queue_count', 'workflow', 'workflow_prepare', 'workflow_upload', 'q_c_prompt',
           'prepare_workflow', 'queue_i4', 'template']

    # Felder, die nur lokal gelten und nicht mitgeschickt werden
    LOCAL_KEYS = ['type', 'job_type', 'async', 'wait', 'refresh']

    CACHE_WORKFLOWS = 'workflows'
    CACHE_SERVER = 'server_info'
    CACHE_WORKFLOW = 'workflow:'

    def __init__(self, type='remote_comfy', target='', store='mongodb', remote_type='comfyui', wait=300,
                 page_wait=15, cache_ttl=0, poll_interval=10, timeout=6000, keep=100):
        """
        :param target: Name der Ziel-Instanz (nwebclient.nx.get_name() dort, i.d.R. der Hostname)
        :param store: type des Job-Speicher-Runners
        :param remote_type: type des ComfyUi-Runners auf der Ziel-Instanz
        :param wait: Sekunden, die ein Job ohne async=1 auf die Antwort wartet
        :param page_wait: Sekunden, die eine Seite auf fehlende Cache-Einträge wartet, danach Hinweis "wird geladen"
        :param cache_ttl: Sekunden, die ein Cache-Eintrag gilt, 0: bis zum Verwerfen (cache_clear, refresh=1)
        :param poll_interval: Sekunden zwischen den Abfragen der Ergebnisse
        :param timeout: Sekunden, nach denen eine Antwort verworfen wird
        :param keep: Anzahl abholbereiter Ergebnisse, ältere werden verworfen
        """
        self.remote_cfg = {}
        ComfyUi.__init__(self, server_url='', args={})
        self.type = type
        self.init_remote(target, store, remote_type, poll_interval, timeout)
        self.wait = float(wait)
        self.page_wait = float(page_wait)
        self.cache_ttl = float(cache_ttl)
        self.keep = int(keep)
        self.results = OrderedDict()   # key -> Ergebnis der Ziel-Instanz
        self.events = {}               # key -> Event für wartende Jobs
        self.cache = {}                # Cache-Schlüssel -> {ts, result}
        self.cache_loaded = False      # cache und offene Abrufe aus der nweb-Persistence geladen
        self.pending_saved = None      # zuletzt gespeicherte offene Abrufe (JSON)
        self.define_vars('target', 'store', 'remote_type', 'wait', 'page_wait', 'cache_ttl')
        self.define_sig(d.PStr('op', 'remote_result'), d.PStr('guid'))\
            .set_description("Ergebnis eines mit async=1 eingestellten Jobs, pending solange keins da ist")\
            .r(success='bool', pending='bool', result='dict')
        self.define_sig(d.PStr('op', 'remote_state'))\
            .set_description("Statuszeile (target, ausstehende Jobs, Fehler) und Cache-Einträge")\
            .r(success='bool', status='str', pending='int', error='str', cache='list')
        self.define_sig(d.PStr('op', 'cache_clear'), d.PStr('key', ''))\
            .set_description("Cache für target leeren, mit key nur diesen Eintrag (z.B. workflows)")\
            .r_success()

    # --- Konfiguration der Ziel-Instanz statt der lokalen -------------------------------------

    @property
    def cfg(self) -> dict:
        """ Konfiguration der Ziel-Instanz (server_info), fehlt sie, wird sie im Hintergrund geholt """
        if not hasattr(self, 'cache'):
            return self.remote_cfg  # während ComfyUi.__init__
        res = self.cached_get(self.CACHE_SERVER, dict(op='server_info'), wait=0)
        return (res.get('cfg') or {}) if res.get('success') else {}

    @cfg.setter
    def cfg(self, value):
        self.remote_cfg = value  # ComfyUi.__init__ setzt die lokale Konfiguration, sie wird nicht verwendet

    @property
    def remote_server_url(self) -> str:
        if not hasattr(self, 'cache'):
            return ''
        res =self.cached_get(self.CACHE_SERVER, dict(op='server_info'), wait=0)
        return str(res.get('server_url', '')) if res.get('success') else ''

    # --- RemoteJobMixin -------------------------------------------------------------------------

    def remote_job_label(self, job: dict):
        op = job.get('op', 'prompt')
        text = job.get('workflow') if isinstance(job.get('workflow'), str) else job.get('prompt') or job.get('route')
        text = str(text or '')
        return op + (' ' + (text if len(text) <= 30 else '...' + text[-27:]) if text else '')

    def on_remote_result(self, meta: dict, result: dict):
        """ unter remote_lock """
        key = meta['key']
        if result.get('success', False):
            for cache_key in meta.get('invalidate') or []:
                self.cache.pop(cache_key, None)
            if meta.get('invalidate') and isinstance(result.get('workflow'), str):
                self.cache.pop(self.CACHE_WORKFLOW + result['workflow'], None)
            if meta.get('cache_key'):
                self.cache_put(meta['cache_key'], result)
            if meta.get('invalidate') or meta.get('cache_key'):
                self.save_cache()
        self.results[key] = result
        while len(self.results) > self.keep:
            self.results.popitem(last=False)
        event = self.events.get(key)
        if event is not None:
            event.set()

    def remote_changed(self):
        """ unter remote_lock: offene Cache-Abrufe speichern """
        self.save_pending()

    # --- Cache (nweb-Persistence, system__setting) --------------------------------------------

    def cache_store(self):
        """ lokale nweb-Persistence mit setting/set_setting oder None """
        root = self.getRoot()
        nw = getattr(root, 'nweb', None) if root is not None else None
        return nw if nw is not None and hasattr(nw, 'set_setting') and hasattr(nw, 'setting') else None

    def cache_setting_key(self) -> str:
        return f'{self.type}.cache.{self.target}'

    def pending_setting_key(self) -> str:
        return f'{self.type}.pending.{self.target}'

    def ensure_cache_loaded(self):
        """ unter remote_lock: Cache und offene Abrufe einmalig laden, neuere Einträge haben Vorrang """
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
        try:
            value = json.loads(nw.setting(self.pending_setting_key(), '') or '[]')
            for meta in value if isinstance(value, list) else []:
                if not isinstance(meta, dict) or not meta.get('guid') or meta['guid'] in self.remote_pending:
                    continue
                self.remote_seq += 1
                self.remote_pending[meta['guid']] = dict(meta, seq=self.remote_seq, ts=meta.get('ts') or time.time())
            if self.remote_pending:
                self.start_remote_poller()
        except Exception as e:
            self.error("Offene Abrufe nicht geladen: " + str(e))

    def save_cache(self):
        """ unter remote_lock """
        nw = self.cache_store()
        if nw is None:
            return
        self.ensure_cache_loaded()  # erst laden, sonst überschreibt ein frischer Start den gespeicherten Cache
        try:
            nw.set_setting(self.cache_setting_key(), json.dumps(self.cache, default=str))
        except Exception as e:
            self.error("Cache nicht gespeichert: " + str(e))

    def save_pending(self):
        """ unter remote_lock: offene Cache-Abrufe (nicht die Jobs) speichern """
        nw = self.cache_store()
        if nw is None:
            return
        self.ensure_cache_loaded()
        pending = json.dumps([m for m in self.remote_pending.values() if m.get('cache_key')], default=str)
        if pending == self.pending_saved:
            return
        try:
            nw.set_setting(self.pending_setting_key(), pending)
            self.pending_saved = pending
        except Exception as e:
            self.error("Offene Abrufe nicht gespeichert: " + str(e))

    def cache_put(self, cache_key, result: dict):
        """ unter remote_lock, workflows mit full=1 füllt auch die Einträge der einzelnen Workflows """
        now = time.time()
        if cache_key == self.CACHE_WORKFLOWS and isinstance(result.get('workflows'), dict):
            for path, workflow in result['workflows'].items():
                self.cache[self.CACHE_WORKFLOW + path] = dict(ts=now, result=dict(success=True, workflow=workflow))
            for path, message in (result.get('errors') or {}).items():
                self.cache[self.CACHE_WORKFLOW + path] = dict(ts=now, result=dict(success=False, message=message))
            result = {k: v for k, v in result.items() if k not in ('workflows', 'errors')}
        self.cache[cache_key] = dict(ts=now, result=result)

    def cache_lookup(self, cache_key):
        """ unter remote_lock: gültiges Ergebnis oder None """
        self.ensure_cache_loaded()
        entry = self.cache.get(cache_key)
        if entry is None:
            return None
        if self.cache_ttl > 0 and time.time() - entry['ts'] > self.cache_ttl:
            return None
        return entry

    def pending_key(self, cache_key):
        """ unter remote_lock: key eines laufenden Abrufs für cache_key oder None """
        for meta in self.remote_pending.values():
            if meta.get('cache_key') == cache_key:
                return meta.get('key')
        return None

    def cached_get(self, cache_key, job: dict, wait=None, refresh=False) -> dict:
        """
          Ergebnis aus dem Cache, sonst job an die Ziel-Instanz (ein laufender Abruf wird nicht wiederholt).
          :param wait: Sekunden auf die Antwort warten, 0: nur im Hintergrund abrufen
          :return: Ergebnis mit cached_at oder success=True, pending=True, solange keine Antwort da ist
        """
        wait = self.wait if wait in (None, '') else float(wait)
        with self.remote_lock:
            entry = None if refresh else self.cache_lookup(cache_key)
            if entry is not None:
                return {**entry['result'], 'cached_at': entry['ts']}
            key = self.pending_key(cache_key)
        if not self.target:
            return self.fail('target (Name der Ziel-Instanz) ist nicht gesetzt')
        if key is None:
            res = self.send(job, run_async=True, cache_key=cache_key)
            key = res.get('remote_guid')
            if key is None:
                return res
        if wait > 0:
            with self.remote_lock:
                if self.is_pending_locked(key):
                    self.events.setdefault(key, threading.Event())
            self.wait_for(key, wait)
        with self.remote_lock:
            result = self.results.get(key)
            entry = self.cache.get(cache_key)
            if result is not None:
                return {**entry['result'], 'cached_at': entry['ts']} if result.get('success') and entry else result
            if self.is_pending_locked(key):
                return self.success(f'Wird von {self.target} abgerufen', pending=True, remote_guid=key)
            return self.fail(self.remote_error or 'Keine Antwort von ' + self.target)

    def cache_invalidate(self, *keys):
        with self.remote_lock:
            self.ensure_cache_loaded()
            for k in keys or list(self.cache.keys()):
                self.cache.pop(k, None)
            self.save_cache()

    # --- Senden und Abholen ---------------------------------------------------------------------

    def is_pending_locked(self, key) -> bool:
        return any(m.get('key') == key for m in self.remote_pending.values())

    def is_pending(self, key) -> bool:
        with self.remote_lock:
            return self.is_pending_locked(key)

    def wait_for(self, key, wait: float) -> bool:
        """ :return: True, wenn die Antwort da bzw. nichts mehr aussteht """
        event = self.events.get(key)
        if event is None:
            return not self.is_pending(key)
        deadline = time.time() + wait
        try:
            while not event.wait(min(1.0, max(0.0, deadline - time.time()))):
                if not self.is_pending(key):
                    return True
                if time.time() >= deadline:
                    return False
            return True
        finally:
            self.events.pop(key, None)

    def send(self, job: dict, wait=None, run_async=False, **meta):
        """ Job an die Ziel-Instanz, run_async: nur einstellen, sonst bis zu wait Sekunden auf die Antwort warten """
        key = u.guid()
        if not run_async:
            self.events[key] = threading.Event()
        try:
            with self.remote_lock:
                self.ensure_cache_loaded()
            self.remote_send(job, key=key, **meta)
        except Exception as e:
            self.events.pop(key, None)
            self.set_remote_error(str(e))
            return self.fail(str(e))
        if run_async:
            return self.success('Gesendet, wartet auf Antwort', pending=True, remote_guid=key)
        if not self.wait_for(key, self.wait if wait in (None, '') else float(wait)):
            return self.success('Keine Antwort, Ergebnis über op=remote_result', pending=True, remote_guid=key)
        with self.remote_lock:
            result = self.results.get(key)
        if result is None:
            return self.fail(self.remote_error or 'Keine Antwort von ' + self.target)
        return result

    def job(self, data: dict) -> dict:
        return {k: v for k, v in data.items() if k not in self.LOCAL_KEYS}

    def invalidates(self, data: dict) -> list:
        """ Cache-Einträge, die ein Job ändert """
        op = data.get('op')
        if op == 'workflow_prepare':
            return [self.CACHE_WORKFLOWS, self.CACHE_WORKFLOW + str(data.get('workflow', '')),
                    self.CACHE_WORKFLOW + str(data.get('save_as') or data.get('workflow', ''))]
        if op == 'workflow_upload':
            return [self.CACHE_WORKFLOWS]  # der Pfad steht im Ergebnis (workflow), siehe on_remote_result
        return []

    def execute(self, data):
        op = data.get('op')
        if op in self.OPS or (op is None and 'prompt' in data):
            return self.send(self.job(data), data.get('wait'), is_true(data.get('async', 0)),
                             invalidate=self.invalidates(data))
        return r.BaseJobExecutor.execute(self, data)

    # --- gecachte Ops ---------------------------------------------------------------------------

    def cached_op(self, cache_key, job: dict, data: dict) -> dict:
        wait = 0 if is_true(data.get('async', 0)) else data.get('wait')
        return self.cached_get(cache_key, job, wait, refresh=is_true(data.get('refresh', 0)))

    def execute_workflows(self, data={}):
        return self.cached_op(self.CACHE_WORKFLOWS, dict(op='workflows', full=1), data)

    def execute_workflow_get(self, data):
        wf = str(data.get('workflow', ''))
        return self.cached_op(self.CACHE_WORKFLOW + wf, dict(op='workflow_get', workflow=wf), data)

    def execute_server_info(self, data={}):
        return self.cached_op(self.CACHE_SERVER, dict(op='server_info'), data)

    def execute_cache_clear(self, data={}):
        key = str(data.get('key', '') or '')
        self.cache_invalidate(*([key] if key else []))
        return self.success('Cache geleert' + (f': {key}' if key else ''), ui_reload=True)

    def execute_remote_result(self, data):
        key = str(data.get('guid', ''))
        with self.remote_lock:
            result = self.results.get(key)
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
            cache = [dict(key=k, ts=e['ts'], success=bool(e['result'].get('success')))
                     for k, e in sorted(self.cache.items())]
        if error and error not in status:
            status += ' – ' + error
        return self.success(status=status, pending=pending, error=error, target=self.target, store=self.store,
                            cache=cache)

    def set_target(self, value):
        """ setvar target: Cache und offene Abrufe sind je Ziel gespeichert """
        value = str(value or '')
        with self.remote_lock:
            if value == self.target:
                return
            self.target = value
            self.cache, self.remote_pending, self.results = {}, {}, OrderedDict()
            self.remote_error = ''
            self.cache_loaded = False
            self.ensure_cache_loaded()

    # --- Workflows der Ziel-Instanz für die Seiten von ComfyUi ----------------------------------

    def page_wait_for(self, cache_key) -> float:
        """ nur der erste Zugriff wartet, läuft der Abruf schon, liefert die Seite den Hinweis "wird geladen" """
        with self.remote_lock:
            return 0 if self.pending_key(cache_key) is not None else self.page_wait

    def workflows(self) -> list:
        res = self.cached_get(self.CACHE_WORKFLOWS, dict(op='workflows', full=1),
                              wait=self.page_wait_for(self.CACHE_WORKFLOWS))
        return list(res.get('value') or []) if res.get('success') else []

    def load_workflow(self, json_file) -> dict:
        cache_key = self.CACHE_WORKFLOW + str(json_file)
        res = self.cached_get(cache_key, dict(op='workflow_get', workflow=json_file),
                              wait=self.page_wait_for(cache_key))
        if res.get('pending'):
            raise OSError(f'wird von {self.target} abgerufen, Seite gleich neu laden')
        if not res.get('success'):
            raise OSError(str(res.get('message', 'Fehler auf ' + self.target)))
        return copy.deepcopy(res['workflow'])  # die Seiten dürfen den Workflow ändern, der Cache bleibt

    def save_workflow(self, json_file, workflow: dict):
        raise NotImplementedError('RemoteComfy speichert Workflows über op=workflow_upload auf der Ziel-Instanz')

    # --- Seiten ---------------------------------------------------------------------------------

    def cache_age(self, cache_key) -> str:
        with self.remote_lock:
            entry = self.cache.get(cache_key)
        if entry is None:
            return '-'
        return time.strftime('%d.%m. %H:%M:%S', time.localtime(entry['ts']))

    def part_workflow_table(self, p: base.Page):
        super().part_workflow_table(p)
        with self.remote_lock:
            loading = self.pending_key(self.CACHE_WORKFLOWS) is not None
        if loading:
            p.alert_warn(f"Workflows werden von {html.escape(self.target)} abgerufen, Seite gleich neu laden.")
        p.div(f"Stand: {self.cache_age(self.CACHE_WORKFLOWS)} " +
              self.action_btn(dict(title="Neu abrufen", type=self.type, op='cache_clear'),
                              hint="Cache leeren; Workflows und Konfiguration werden beim nächsten Laden neu "
                                   "von der Ziel-Instanz geholt"))

    def server_section(self, p: base.Page):
        error = self.check_store() if self.target else 'target (Name der Ziel-Instanz) ist nicht gesetzt'
        if error:
            p.alert_error(html.escape(error))
        with self.remote_lock:
            status = self.remote_status()
        p.prop("Ziel-Instanz", html.escape(f'{self.remote_type}@{self.target}'))
        p.prop("Job-Speicher", html.escape(self.store))
        p.prop("Status", html.escape(status))
        cfg = self.cfg
        p.prop("Server", html.escape(self.remote_server_url or '-'))
        p.prop("ComfyUI-Pfad", html.escape(cfg.get('comfyui_path', '- (nötig für Bilder)')))
        p.prop("Jobpfad", html.escape(cfg.get('jobpath', '.')))
        p.prop("Workflow-Ordner", html.escape(cfg.get('workflow_dir', '-')))
        p.prop("Konfiguration", "Stand " + self.cache_age(self.CACHE_SERVER))
        self.server_buttons(p)
        p(self.action_btn(dict(title="Remote-Status", type=self.type, op='remote_state'),
                          hint="Ausstehende Jobs, letzter Fehler und Cache-Einträge"))
        p.div("Die Ziel-Instanz arbeitet die Jobs im Takt ihres Tickers ab; Server-Abfragen warten auf die Antwort.")
