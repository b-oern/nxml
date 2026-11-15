"""
ComfyRemote: die Instanz-Seiten des ComfyUi-Runners einer anderen nx-Instanz ueber einen Job-Speicher bedienen.

Die Anfragen gehen wie beim RemoteRunner (nxrunner.jobstore) verschluesselt ueber den Job-Speicher (cenqueue) an
den ComfyUi-Runner der Ziel-Instanz, die Antworten werden per guid abgeholt (cresult). Jeder Lauf (op=run) wird mit
guid, Parametern, Prompt-IDs und Ergebnis in der nweb-Persistence (root.nweb) in npy_comfy_jobs gespeichert; offene
Laeufe werden nach einem Neustart weiter abgefragt, der Status der Prompts in ComfyUI mit op=check nachgefuehrt.
"""
import json
import threading
import time
from html import escape as esc
from typing import Optional

from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import dev as d
from nwebclient import web as w

from nxrunner.jobstore import RemoteJobMixin, fmt_ts
from nxml.comfyui import ComfyUi


class ComfyJobTable:
    """
      Laeufe in npy_comfy_jobs der nweb-Persistence (sqlite oder MySQL), ohne Persistence nur im Speicher.

      status: sent (wartet auf die Ziel-Instanz), queued/running (in ComfyUI), done, error, timeout
    """

    TABLE = 'npy_comfy_jobs'
    COLS = ('guid', 'target', 'workflow', 'params', 'job_count', 'status', 'sent', 'done', 'prompt_ids',
            'prompts', 'message', 'result')
    JSON_COLS = ('params', 'prompt_ids', 'prompts', 'result')
    OPEN = ('sent', 'queued', 'running')

    def __init__(self, nw=None):
        self.nw = nw
        self.memory = {}
        if nw is not None:
            nw.dml(f"""CREATE TABLE IF NOT EXISTS {self.TABLE} (
                guid varchar(64) NOT NULL PRIMARY KEY,
                target varchar(255) DEFAULT NULL,
                workflow varchar(512) DEFAULT NULL,
                params text,
                job_count INTEGER DEFAULT 1,
                status varchar(32) DEFAULT NULL,
                sent REAL DEFAULT NULL,
                done REAL DEFAULT NULL,
                prompt_ids text,
                prompts text,
                message text,
                result text
            )""")

    @property
    def persistent(self) -> bool:
        return self.nw is not None

    def _encode(self, row: dict) -> dict:
        return {k: (json.dumps(v, default=str) if k in self.JSON_COLS and v is not None else v)
                for k, v in row.items() if k in self.COLS}

    def _decode(self, values) -> dict:
        row = dict(zip(self.COLS, values))
        for k in self.JSON_COLS:
            try:
                row[k] = json.loads(row[k]) if row[k] else None
            except ValueError:
                pass
        return row

    def insert(self, row: dict):
        if self.nw is None:
            self.memory[row['guid']] = dict(row)
        else:
            self.nw.insert(self.TABLE, self._encode(row))

    def update(self, guid: str, **values):
        if self.nw is None:
            if guid in self.memory:
                self.memory[guid].update(values)
            return
        values = self._encode(values)
        if values:
            sets = ', '.join(f'{k} = ?' for k in values)
            self.nw.dml(f"UPDATE {self.TABLE} SET {sets} WHERE guid = ?", [*values.values(), guid])

    def get(self, guid: str) -> Optional[dict]:
        if self.nw is None:
            row = self.memory.get(guid)
            return dict(row) if row is not None else None
        rows = self.nw.select(f"SELECT {', '.join(self.COLS)} FROM {self.TABLE} WHERE guid = ?", [guid])
        return self._decode(rows[0]) if rows else None

    def rows(self, target: str = None, status=None, limit=500) -> list:
        """ :return: neueste zuerst """
        if self.nw is None:
            res = [dict(row) for row in self.memory.values()
                   if (target is None or row['target'] == target) and (status is None or row['status'] in status)]
            return sorted(res, key=lambda row: row['sent'] or 0, reverse=True)[:limit]
        where, params = [], []
        if target is not None:
            where.append('target = ?')
            params.append(target)
        if status is not None:
            where.append(f"status IN ({', '.join('?' for _ in status)})")
            params.extend(status)
        sql = f"SELECT {', '.join(self.COLS)} FROM {self.TABLE}"
        sql += (' WHERE ' + ' AND '.join(where)) if where else ''
        sql += f" ORDER BY sent DESC LIMIT {int(limit)}"
        return [self._decode(values) for values in self.nw.select(sql, params)]

    def delete(self, guid: str):
        if self.nw is None:
            self.memory.pop(guid, None)
        else:
            self.nw.dml(f"DELETE FROM {self.TABLE} WHERE guid = ?", [guid])


class ComfyRemote(r.BaseJobExecutor, RemoteJobMixin):
    """
      Instanz-Seiten des ComfyUi-Runners (gespeicherte Workflows, Parameter, Run) einer anderen nx-Instanz.

        Config (runners):
            comfy_remote: nxml.comfyui.remote:ComfyRemote(target="gpu-pc")
            comfy_remote: nxml.comfyui.remote:ComfyRemote(target="gpu-pc", store="nweb_jobs", remote_type="comfyui")
            comfy_remote: nxml.comfyui.remote:ComfyRemote(target="local", store="store", poll_interval=1)  # DirectJobStore

        Jobs:
            op=workflows                         gespeicherte Workflows der Ziel-Instanz abrufen
            op=params workflow                   Parameter eines Workflows abrufen
            op=run workflow [count] p:... r:...  Workflow auf der Ziel-Instanz starten, liefert guid
            op=check [guid]                      Status der Prompts offener (bzw. eines) Laufs in ComfyUI abfragen
            op=jobs                              Laeufe aus npy_comfy_jobs
            op=delete guid                       Lauf aus npy_comfy_jobs entfernen
    """

    TAGS = ['remote', 'ml']

    def __init__(self, type='comfy_remote', target='', store='mongodb', remote_type='comfyui', poll_interval=10,
                 timeout=6000, check_interval=60):
        """
        :param target: Name der Ziel-Instanz (nwebclient.nx.get_name() dort)
        :param store: type des Job-Speicher-Runners (JobStoreMixin)
        :param remote_type: type des ComfyUi-Runners auf der Ziel-Instanz
        :param check_interval: Sekunden, nach denen die Jobs-Seite den Status offener Laeufe erneut abfragt
        """
        super().__init__(type)
        self.init_remote(target, store, remote_type, poll_interval, timeout)
        self.check_interval = float(check_interval)
        self.table = None              # ComfyJobTable, beim ersten Zugriff (root.nweb ist erst dann gesetzt)
        self.workflows = None          # Antwort von instance_list: {workflows, ts, error}
        self.params = {}               # Workflow -> Antwort von instance_params mit ts
        self.last_check = 0
        self.define_vars('target', 'store', 'remote_type', 'check_interval')
        self.define_sig(d.PStr('op', 'workflows')).r(success='bool', message='str')
        self.define_sig(d.PStr('op', 'params'), d.PStr('workflow')).r(success='bool', message='str')
        self.define_sig(d.PStr('op', 'run'), d.PStr('workflow'), d.PInt('count', 1)).r(success='bool', guid='str')
        self.define_sig(d.PStr('op', 'check'), d.PStr('guid', '')).r(success='bool', message='str')
        self.define_sig(d.PStr('op', 'jobs')).r(success='bool', jobs='list')
        self.define_sig(d.PStr('op', 'delete'), d.PStr('guid')).r(success='bool', message='str')

    # --- Persistenz -----------------------------------------------------------------------------

    def nweb_store(self):
        root = self.getRoot()
        nw = getattr(root, 'nweb', None) if root is not None else None
        return nw if nw is not None and all(hasattr(nw, m) for m in ('dml', 'select', 'insert')) else None

    def jobs_table(self) -> ComfyJobTable:
        """ unter remote_lock: Tabelle anlegen und offene Laeufe wieder abfragen (einmalig) """
        if self.table is None:
            try:
                self.table = ComfyJobTable(self.nweb_store())
            except Exception as e:
                self.error("npy_comfy_jobs nicht verfuegbar: " + str(e))
                self.table = ComfyJobTable(None)
            self.restore_pending()
        return self.table

    def restore_pending(self):
        """ unter remote_lock: Laeufe mit status=sent wieder abfragen, z.B. nach einem Neustart """
        for row in self.table.rows(self.target, ('sent',)):
            if row['guid'] not in self.remote_pending:
                self.remote_seq += 1
                self.remote_pending[row['guid']] = dict(kind='run', guid=row['guid'], seq=self.remote_seq,
                                                        ts=row['sent'] or time.time(), label='Run ' + str(row['workflow']))
        if self.remote_pending:
            self.start_remote_poller()

    # --- RemoteJobMixin -------------------------------------------------------------------------

    def remote_job_label(self, job: dict):
        op = job.get('op', '')
        return {'instance_list': 'Workflows', 'instance_status': 'Status'}.get(op) or \
            f"{op.replace('instance_', '')} {job.get('workflow', '')}".strip()

    def on_remote_result(self, meta: dict, result: dict):
        """ unter remote_lock """
        kind = meta.get('kind')
        ok = result.get('success', False)
        if kind == 'workflows':
            self.workflows = dict(workflows=result.get('workflows') if ok else None, ts=time.time(),
                                  error='' if ok else str(result.get('message', 'Fehler')))
        elif kind == 'params':
            self.params[meta['workflow']] = dict(result, ts=time.time())
        elif kind == 'run':
            ids = result.get('prompt_ids') or []
            self.jobs_table().update(meta['guid'], status='queued' if ok and ids else 'error', done=time.time(),
                                     prompt_ids=ids, message=str(result.get('message', '')), result=result)
        elif kind == 'status':
            self.on_status(result)

    def on_status(self, result: dict):
        """ unter remote_lock: Status der Prompts auf die offenen Laeufe uebertragen """
        prompts = result.get('prompts') if result.get('success', False) else None
        if not isinstance(prompts, dict):
            self.remote_error = 'Status: ' + str(result.get('message', 'keine Antwort'))
            return
        table = self.jobs_table()
        for row in table.rows(self.target, ('queued', 'running')):
            states = {pid: prompts[pid] for pid in row['prompt_ids'] or [] if pid in prompts}
            if not states:
                continue
            merged = dict(row['prompts'] or {}, **states)
            values = [s.get('status') for s in merged.values()]
            if 'error' in values:
                status = 'error'
            elif values and all(v == 'done' for v in values) and len(merged) >= len(row['prompt_ids']):
                status = 'done'
            elif 'running' in values or 'done' in values:
                status = 'running'
            elif values and all(v == 'unknown' for v in values):
                status = 'error'
            else:
                status = 'queued'
            message = '; '.join(filter(None, (s.get('message') for s in merged.values()))) or row['message']
            table.update(row['guid'], prompts=merged, status=status, message=message,
                         done=time.time() if status in ('done', 'error') else row['done'])

    def remote_changed(self):
        """ unter remote_lock: Laeufe, deren Antwort verworfen wurde (Timeout), markieren """
        if self.table is None:
            return
        for row in self.table.rows(self.target, ('sent',)):
            if row['guid'] not in self.remote_pending:
                self.table.update(row['guid'], status='timeout', done=time.time(),
                                  message=f'Keine Antwort von {self.target}')

    def pending_kind(self, kind: str, **match) -> bool:
        """ unter remote_lock: Anfrage dieser Art steht noch aus """
        return any(m.get('kind') == kind and all(m.get(k) == v for k, v in match.items())
                   for m in self.remote_pending.values())

    def send(self, job: dict, **meta) -> str:
        return self.remote_send(job, **meta)

    # --- Ops ------------------------------------------------------------------------------------

    def execute_workflows(self, data={}):
        try:
            self.send({'op': 'instance_list'}, kind='workflows')
        except Exception as e:
            self.set_remote_error(str(e))
            return self.fail(str(e))
        return self.success('Abfrage gesendet, wartet auf Antwort')

    def execute_params(self, data):
        path = str(data.get('workflow') or '')
        if not path:
            return self.fail('workflow fehlt')
        try:
            self.send({'op': 'instance_params', 'workflow': path}, kind='params', workflow=path)
        except Exception as e:
            self.set_remote_error(str(e))
            return self.fail(str(e))
        return self.success('Abfrage gesendet, wartet auf Antwort')

    def execute_run(self, data):
        path = str(data.get('workflow') or '')
        if not path:
            return self.fail('workflow fehlt')
        job = {k: v for k, v in data.items() if k.startswith(('p:', 'r:'))}
        job.update(op='instance_run', workflow=path, count=data.get('count') or 1)
        with self.remote_lock:
            table = self.jobs_table()
        try:
            guid = self.send(job, kind='run', workflow=path)
        except Exception as e:
            self.set_remote_error(str(e))
            return self.fail(str(e))
        with self.remote_lock:
            table.insert(dict(guid=guid, target=self.target, workflow=path, params=job, job_count=int(job['count']),
                              status='sent', sent=time.time(), done=None, prompt_ids=None, prompts=None, message='',
                              result=None))
        return self.success(f'Run {path} gesendet', guid=guid, link=self.link(self.part_job, guid=guid))

    def execute_check(self, data={}):
        """ Status der Prompts offener Laeufe (bzw. des Laufs guid) in ComfyUI abfragen """
        guid = str(data.get('guid') or '')
        with self.remote_lock:
            table = self.jobs_table()
            rows = [table.get(guid)] if guid else table.rows(self.target, ('queued', 'running'))
            ids = [pid for row in rows if row for pid in (row['prompt_ids'] or [])]
            if self.pending_kind('status'):
                return self.success('Statusabfrage laeuft bereits')
        if not ids:
            return self.success('Keine offenen Prompts')
        self.last_check = time.time()
        try:
            self.send({'op': 'instance_status', 'prompt_ids': ids}, kind='status')
        except Exception as e:
            self.set_remote_error(str(e))
            return self.fail(str(e))
        return self.success(f'Statusabfrage fuer {len(ids)} Prompt(s) gesendet')

    def execute_jobs(self, data={}):
        with self.remote_lock:
            return self.success(jobs=self.jobs_table().rows(self.target))

    def execute_delete(self, data):
        guid = str(data.get('guid') or '')
        with self.remote_lock:
            self.jobs_table().delete(guid)
        return self.success('Entfernt', ui_reload=True)

    # --- Seiten ---------------------------------------------------------------------------------

    def nav(self, p: b.Page, active=None):
        with self.remote_lock:
            rows = self.jobs_table().rows(self.target)
        open_count = sum(1 for row in rows if row['status'] in ComfyJobTable.OPEN)
        p.nav(("Workflows", self.link(self.part_index), "Gespeicherte Workflows der Ziel-Instanz"),
              (f"Jobs ({len(rows)}, {open_count} offen)", self.link(self.part_jobs),
               "Gestartete Laeufe aus npy_comfy_jobs"),
              active=self.link(active) if active is not None else None)

    def part_header(self, p: b.Page):
        error = self.check_store() if self.target else 'target (Name der Ziel-Instanz) ist nicht gesetzt'
        if error:
            p(w.alert_error(esc(error)))
        with self.remote_lock:
            p.prop("Status", esc(self.remote_status()))
            persistent = self.jobs_table().persistent
        p.prop("Job-Speicher", esc(self.store) + ' → ' + esc(self.remote_type))
        p.prop("Jobs gespeichert", f'ja ({ComfyJobTable.TABLE})' if persistent else
               'nein, keine nweb-Persistence (NPY_DB), nur bis zum Neustart')

    def reload_later(self, p: b.Page):
        p.js_ready(f'setTimeout(function() {{ window.location.reload(); }}, {int(max(self.poll_interval, 2) * 1000)});')

    def part_index(self, p: b.Page, params={}):
        self.nav(p, self.part_index)
        self.part_header(p)
        with self.remote_lock:
            pending = self.pending_kind('workflows')
            workflows = dict(self.workflows) if self.workflows else None
        if workflows is None and not pending and self.target:
            self.execute_workflows()
            pending = True
        p.div(self.action_btn(dict(type=self.type, op='workflows', title="Workflows abrufen"),
                              hint="Gespeicherte Workflows der Ziel-Instanz ueber den Job-Speicher abfragen"))
        if pending:
            p(w.alert_info(f"Abfrage an {esc(self.target)} laeuft, die Seite laedt neu."))
            self.reload_later(p)
        if workflows is None:
            return
        if workflows['error']:
            p(w.alert_error(esc(workflows['error'])))
            return
        p.prop("Stand", fmt_ts(workflows['ts']))
        p('<table class="comfyui_workflows"><tr><th>Workflow</th><th>Geändert</th><th>Größe</th></tr>')
        for item in workflows['workflows'] or []:
            path = str(item.get('path', ''))
            modified = item.get('modified')
            if isinstance(modified, (int, float)):
                modified = fmt_ts(modified / 1000 if modified > 1e11 else modified)
            size = item.get('size')
            size = f"{size / 1024:.1f} KB" if isinstance(size, (int, float)) else '-'
            link = w.a(esc(path), self.link(self.part_workflow, w=path), "Parameter einstellen und ausführen")
            p(f'<tr><td>{link}</td><td>{modified or "-"}</td><td>{size}</td></tr>')
        p('</table>')

    def part_workflow(self, p: b.Page, params={}):
        path = str(params.get('w', ''))
        self.nav(p, self.part_index)
        p.h1("Workflow: " + esc(path))
        with self.remote_lock:
            pending = self.pending_kind('params', workflow=path)
            info = self.params.get(path)
        if (info is None or params.get('reload')) and not pending and self.target:
            self.execute_params({'workflow': path})
            pending, info = True, None
        p.div(self.action_btn(dict(type=self.type, op='params', workflow=path, title="Parameter neu abrufen"),
                              hint="Parameter erneut von der Ziel-Instanz abfragen"))
        if pending:
            p(w.alert_info(f"Parameter werden von {esc(self.target)} abgefragt, die Seite laedt neu."))
            self.reload_later(p)
            return
        if not info.get('success', False):
            p(w.alert_error(esc(str(info.get('message', 'Fehler')))))
            return
        p.prop("Stand", fmt_ts(info['ts']))
        for msg in info.get('warnings') or []:
            p.alert_warn(esc(msg))
        js_done = 'function(r) { if (r.success && r.link) location.href = r.link; }'
        ComfyUi.part_params_form(p, self, info.get('params') or [], dict(type=self.type, op='run', workflow=path),
                                 f"Workflow auf {self.target} starten, der Lauf wird in npy_comfy_jobs gespeichert",
                                 js_done)
        p.pre('', id='result')

    STATUS_BADGES = {'sent': ('gesendet', 'tag'), 'queued': ('in Queue', 'tag'), 'running': ('läuft', 'tag'),
                     'done': ('fertig', 'ok'), 'error': ('Fehler', 'warn'), 'timeout': ('Timeout', 'warn')}

    def auto_check(self):
        """ offene Laeufe in ComfyUI hoechstens alle check_interval Sekunden abfragen """
        if time.time() - self.last_check >= self.check_interval:
            self.execute_check({})

    def part_jobs(self, p: b.Page, params={}):
        self.nav(p, self.part_jobs)
        self.part_header(p)
        self.auto_check()
        with self.remote_lock:
            rows = self.jobs_table().rows(self.target)
        p.div(self.action_btn(dict(type=self.type, op='check', title="Status prüfen"),
                              hint="Status der Prompts offener Laeufe in ComfyUI abfragen"))
        with p.section(h=f"Jobs ({ComfyJobTable.TABLE})"):
            p.grid(rows, self.job_cols(), params, search=True, empty="Noch keine Laeufe gestartet.")
        if any(row['status'] in ComfyJobTable.OPEN for row in rows):
            self.reload_later(p)

    def job_cols(self):
        col = w.Grid.col
        return [
            col('sent', "Gesendet", value=lambda e: fmt_ts(e['sent']), sort=lambda e: e['sent'] or 0),
            col('workflow', "Workflow", html=True, sort=lambda e: e['workflow'] or '',
                value=lambda e: w.a(esc(str(e['workflow'])), self.link(self.part_job, guid=e['guid']))),
            col('job_count', "Anzahl"),
            col('status', "Status", html=True, value=self.col_status),
            col('outputs', "Ausgaben", value=lambda e: ', '.join(o for s in (e['prompts'] or {}).values()
                                                                for o in s.get('outputs', []))[:120]),
            col('message', "Meldung", value=lambda e: str(e['message'] or '')[:120]),
            col('actions', '', html=True, sortable=False, value=self.col_actions),
        ]

    def col_status(self, e):
        text, css = self.STATUS_BADGES.get(e['status'], (e['status'], ''))
        return f'<span class="badge {css}">{esc(str(text))}</span>'

    def col_actions(self, e):
        params = {k: v for k, v in (e['params'] or {}).items() if k.startswith(('p:', 'r:'))}
        rerun = dict(type=self.type, op='run', workflow=e['workflow'], count=e['job_count'], title="Erneut", **params)
        delete = dict(type=self.type, op='delete', guid=e['guid'], title="Entfernen")
        return (self.action_btn(rerun, hint="Mit denselben Parametern noch einmal starten") +
                self.action_btn(delete, hint="Lauf aus npy_comfy_jobs entfernen"))

    def part_job(self, p: b.Page, params={}):
        self.nav(p, self.part_jobs)
        guid = str(params.get('guid', ''))
        with self.remote_lock:
            row = self.jobs_table().get(guid)
        if row is None:
            p(w.alert_error("Unbekannte guid"))
            return
        if row['status'] in ('queued', 'running'):
            self.auto_check()
        with p.section(h=f"Lauf {esc(str(row['workflow']))}"):
            p.prop("guid", esc(guid))
            p.prop("Ziel-Instanz", esc(str(row['target'])))
            p.prop("Status", self.col_status(row))
            p.prop("Gesendet", fmt_ts(row['sent']))
            p.prop("Aktualisiert", fmt_ts(row['done']))
            p.prop("Meldung", esc(str(row['message'] or '')))
            p.div(self.action_btn(dict(type=self.type, op='check', guid=guid, title="Status prüfen"),
                                  hint="Status der Prompts dieses Laufs in ComfyUI abfragen") + self.col_actions(row))
            p.h3("Prompts")
            p.pre(esc(json.dumps(row['prompts'] or {pid: 'noch kein Status' for pid in row['prompt_ids'] or []},
                                 indent=2, default=str)))
            p.h3("Parameter")
            p.pre(esc(json.dumps(row['params'], indent=2, default=str)))
            p.h3("Ergebnis")
            p.pre(esc(json.dumps(row['result'], indent=2, default=str)) if row['result'] is not None else 'wartet...')
        if row['status'] in ComfyJobTable.OPEN:
            self.reload_later(p)
