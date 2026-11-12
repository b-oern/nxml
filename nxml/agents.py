"""
    Coding-Agenten

    opencode: nxml.agents:OpenCodeAgent(project_path="C:/dev/projekt")
"""
import os
import re
import json
import time
import shutil
import socket
import atexit
import threading
import subprocess
import datetime
import secrets

import requests

from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import util as u
from nwebclient.runner import op


class OpenCodeError(Exception):
    pass


class OpenCodeAgent(r.BaseJobExecutor):
    """
    Startet einen opencode-Server (opencode serve, https://opencode.ai/docs/server/) und nutzt dessen HTTP-API.

    do_issue(name, task_text) legt den Branch feat_<name> an, lässt opencode die Aufgabe bearbeiten,
    committet und pusht den Branch und wechselt zurück auf den Ausgangs-Branch.

    UI: Issues (Aufgabe, Provider/Modell, Verlauf) -> Server (Status, Start/Stopp, Sessions)

    Konfiguration: OPENCODE_BIN (Pfad zu opencode), OPENCODE_URL (vorhandener Server bei start=False),
    OPENCODE_PROVIDER, OPENCODE_MODEL, OPENCODE_SERVER_PASSWORD / OPENCODE_SERVER_USERNAME (Basic Auth)
    """

    TAGS = ['agent', 'coding']

    def __init__(self, start=True, port=None, project_path=None, type='opencode', url=None, provider=None,
                 model=None, opencode_bin=None, hostname='127.0.0.1', remote='origin', timeout=3600,
                 password=None, args: u.Args = None):
        """
            :param start: opencode-Server beim Erzeugen starten (sonst start_server() oder op=server_start)
            :param port: Port des Servers, None: freier Port (bzw. Port aus url)
            :param project_path: Git-Repository, in dem opencode arbeitet, Standard: aktuelles Verzeichnis
            :param url: vorhandener Server (statt selbst zu starten)
            :param provider: Standard-Provider (z.B. anthropic), sonst Standard des Servers
            :param model: Standard-Modell (z.B. claude-sonnet-4-5 oder anthropic/claude-sonnet-4-5)
            :param remote: Git-Remote für den Push
            :param timeout: max. Sekunden für eine Antwort von opencode
            :param password: Passwort bzw. Pairing-Token des Servers (Basic Auth), sonst OPENCODE_SERVER_PASSWORD;
                             beim Selbststarten ohne Passwort wird eines erzeugt
        """
        super().__init__(type)
        if args is None:
            args = u.Args()
        self.project_path = os.path.abspath(project_path or os.getcwd())
        self.hostname = hostname
        url = url or args.get('OPENCODE_URL', None)
        if url and not port:
            m = re.search(r':(\d+)', url.split('//')[-1])
            port = int(m.group(1)) if m else None
        self.port = int(port) if port else None
        self.url = url.rstrip('/') if url else None
        self.provider = provider or args.get('OPENCODE_PROVIDER', '')
        self.model = model or args.get('OPENCODE_MODEL', '')
        if '/' in self.model and not self.provider:
            self.provider, self.model = self.model.split('/', 1)
        self.opencode_bin = opencode_bin or args.get('OPENCODE_BIN', None) or shutil.which('opencode')
        self.remote = remote
        self.timeout = float(timeout)
        self.password = password or os.environ.get('OPENCODE_SERVER_PASSWORD', '')
        self.username = os.environ.get('OPENCODE_SERVER_USERNAME', 'opencode')
        self.process = None
        self.server_log = []
        self.issues = []
        self.lock = threading.Lock()  # nur ein Issue gleichzeitig (gleiches Arbeitsverzeichnis)
        self.define_vars('project_path', 'url', 'provider', 'model', 'remote')
        self.define_event('issue_done', {'name': 'Name', 'branch': 'Branch', 'success': 'bool', 'message': 'str'})
        atexit.register(self.stop_server)
        if start:
            try:
                self.start_server()
            except Exception as e:
                self.error('opencode-Server: ' + str(e))

    # -- Server --

    @staticmethod
    def free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    def start_server(self, wait=30):
        """ Startet opencode serve im project_path und wartet, bis /global/health antwortet """
        if self.process is not None and self.process.poll() is None:
            return self.url
        if self.opencode_bin is None:
            raise OpenCodeError('opencode nicht gefunden (npm i -g opencode-ai oder OPENCODE_BIN setzen)')
        if not self.port:
            self.port = self.free_port()
        self.url = f'http://{self.hostname}:{self.port}'
        env = dict(os.environ)
        if not self.password:
            self.password = secrets.token_urlsafe(24)
        env['OPENCODE_SERVER_PASSWORD'] = self.password
        env['OPENCODE_SERVER_USERNAME'] = self.username
        cmd = [self.opencode_bin, 'serve', '--port', str(self.port), '--hostname', self.hostname]
        self.info('Starte ' + ' '.join(cmd))
        self.process = subprocess.Popen(cmd, cwd=self.project_path, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                        encoding='utf-8', errors='replace')
        threading.Thread(target=self._read_server_output, daemon=True).start()
        end = time.time() + wait
        while time.time() < end:
            if self.process.poll() is not None:
                raise OpenCodeError('opencode beendet: ' + '\n'.join(self.server_log[-10:]))
            if self.online():
                return self.url
            time.sleep(0.5)
        raise OpenCodeError(f'opencode-Server unter {self.url} antwortet nicht')

    def _read_server_output(self):
        proc = self.process
        for line in proc.stdout:
            self.server_log.append(line.rstrip())
            del self.server_log[:-500]

    def stop_server(self):
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # -- HTTP-API (/api, https://opencode.ai/docs/api) --

    # alles erlauben: der Server läuft ohne TUI, in der niemand Berechtigungen bestätigt
    PERMISSIONS = [{'action': '*', 'resource': '*', 'effect': 'allow'}]

    def api(self, path, payload=None, method=None, params=None, timeout=60, location=False):
        """
            Aufruf der opencode-API, liefert das JSON (bei {data: ...} nur data)
            :param location: location[directory]=project_path mitsenden (Endpunkte mit location-Parameter)
        """
        if not self.url:
            raise OpenCodeError('opencode-Server nicht gestartet')
        method = method or ('POST' if payload is not None else 'GET')
        auth = (self.username, self.password) if self.password else None
        params = {k: v for k, v in (params or {}).items() if v not in (None, '')}
        if location:
            params['location[directory]'] = self.project_path
        try:
            resp = requests.request(method, self.url + path, json=payload, auth=auth, timeout=timeout,
                                    params=params, headers={'Accept': 'application/json'})
        except requests.exceptions.ConnectionError:
            raise OpenCodeError(f'opencode unter {self.url} nicht erreichbar')
        if resp.status_code == 401:
            raise OpenCodeError('opencode: nicht angemeldet (401), password bzw. OPENCODE_SERVER_PASSWORD setzen')
        if resp.status_code >= 400:
            raise OpenCodeError(f'{method} {path}: {resp.status_code} {resp.text[:500]}')
        if resp.status_code == 204 or not resp.content:
            return True
        if 'json' not in resp.headers.get('Content-Type', ''):
            raise OpenCodeError(f'{path}: keine API-Antwort ({resp.headers.get("Content-Type")}), '
                                f'opencode zu alt oder falsche URL?')
        data = resp.json()
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        return data

    def server_info(self, timeout=5):
        """ {version, pid, hostname, urls, paths} """
        return self.api('/api/info', timeout=timeout)

    def online(self):
        try:
            return 'version' in self.server_info(timeout=3)
        except Exception:
            return False

    def pair(self):
        """ Pairing-Link für Browser/Apps (einmalig, kurzlebig) """
        code = self.api('/api/pair', {})
        return {'url': f"{self.url}/auth/connect/{code['code']}", **code}

    def providers(self):
        """ {providers: [{id, name, models: [{id, name}]}], default: {providerID: modelID}} """
        names = {}
        try:
            names = {p['id']: p.get('name') or p['id'] for p in self.api('/api/provider', location=True)}
        except Exception:
            pass
        grouped = {}
        for m in self.api('/api/model', location=True):
            grouped.setdefault(m['providerID'], []).append({'id': m['id'], 'name': m.get('name') or m['id']})
        res = [{'id': pid, 'name': names.get(pid, pid), 'models': sorted(ms, key=lambda m: m['name'].lower())}
               for pid, ms in grouped.items()]
        res.sort(key=lambda p: p['name'].lower())
        default = {}
        try:
            d = self.api('/api/model/default', location=True)
            if isinstance(d, dict) and d.get('providerID'):
                default[d['providerID']] = d['id']
        except Exception:
            pass
        return {'providers': res, 'default': default}

    def sessions(self, limit=50):
        return self.api('/api/session', params={'directory': self.project_path, 'limit': limit, 'order': 'desc'})

    def active_sessions(self):
        return self.api('/api/session/active') or {}

    def create_session(self, title=None, provider='', model=''):
        body = {'location': {'directory': self.project_path}, 'permissions': self.PERMISSIONS}
        if title:
            body['title'] = title
        ref = self.model_ref(provider, model)
        if ref:
            body['model'] = ref
        return self.api('/api/session', body)

    def delete_session(self, session_id):
        return self.api(f'/api/session/{session_id}', method='DELETE')

    def abort_session(self, session_id):
        return self.api(f'/api/session/{session_id}/interrupt', {}, params={'resume': 'false'})

    def messages(self, session_id, limit=200, order='asc'):
        return self.api(f'/api/session/{session_id}/message', params={'limit': limit, 'order': order})

    def diff(self, session_id):
        return self.api(f'/api/session/{session_id}/diff')

    def model_ref(self, provider='', model=''):
        provider = provider or self.provider
        model = model or self.model
        if '/' in model and not provider:
            provider, model = model.split('/', 1)
        if provider and model:
            return {'providerID': provider, 'id': model}
        return None

    def prompt(self, session_id, text):
        """ Stellt text in die Session ein (asynchron), liefert die ID der User-Nachricht """
        return self.api(f'/api/session/{session_id}/prompt', {'text': text}).get('id')

    def approve_permissions(self, session_id):
        """ Offene Berechtigungsanfragen der Session erlauben """
        for req in self.api(f'/api/session/{session_id}/permission') or []:
            self.api(f"/api/session/{session_id}/permission/{req['id']}/reply", {'decision': 'once'})

    def wait_idle(self, session_id, message_id, timeout=None, poll=2.0):
        """
            Wartet, bis die Session nach der Nachricht message_id fertig ist (idle-Nachricht)
            :return: (outcome, error, text) mit den Antworttexten nach message_id
        """
        timeout = timeout or self.timeout
        end = time.time() + timeout
        while time.time() < end:
            try:
                self.approve_permissions(session_id)
            except Exception as e:
                self.debug('Berechtigungen: ' + str(e))
            msgs = self.messages(session_id, order='asc')
            ids = [m.get('id') for m in msgs]
            after = msgs[ids.index(message_id) + 1:] if message_id in ids else []
            idle = [m for m in after if m.get('type') == 'idle']
            if idle:
                texts = [c.get('text', '') for m in after if m.get('type') == 'assistant'
                         for c in m.get('content', []) if c.get('type') == 'text']
                err = idle[-1].get('error') or {}
                return idle[-1].get('outcome', 'succeeded'), err.get('message', ''), '\n'.join(texts).strip()
            time.sleep(poll)
        self.abort_session(session_id)
        raise OpenCodeError(f'Zeitüberschreitung nach {timeout} s')

    def run(self, session_id, text, timeout=None):
        """ Prompt senden und auf das Ergebnis warten, liefert den Antworttext """
        outcome, error, response = self.wait_idle(session_id, self.prompt(session_id, text), timeout)
        if outcome != 'succeeded':
            raise OpenCodeError(f'opencode {outcome}: {error}'.rstrip(': '))
        return response

    @staticmethod
    def summarize_messages(msgs):
        """ Kurzfassung für die UI: [{role, parts: [{type: text|reasoning|tool|info, ...}]}] """
        res = []
        for m in msgs or []:
            t = m.get('type')
            if t == 'assistant':
                parts = []
                for c in m.get('content', []):
                    if c.get('type') in ('text', 'reasoning') and c.get('text'):
                        parts.append({'type': c['type'], 'text': c['text']})
                    elif c.get('type') == 'tool':
                        state = c.get('state') or {}
                        err = state.get('error') or {}
                        parts.append({'type': 'tool', 'tool': c.get('name'), 'status': state.get('status'),
                                      'title': json.dumps(state.get('input', {}), ensure_ascii=False)[:200],
                                      'error': err.get('message') if isinstance(err, dict) else str(err)})
                if m.get('error'):
                    parts.append({'type': 'info', 'text': 'Fehler: ' + str(m['error'].get('message'))})
                res.append({'role': 'assistant', 'parts': parts})
            elif t in ('user', 'synthetic', 'system'):
                res.append({'role': t, 'parts': [{'type': 'text', 'text': m.get('text', '')}]})
            elif t == 'shell':
                res.append({'role': 'shell', 'parts': [{'type': 'tool', 'tool': 'shell', 'status': m.get('status'),
                                                        'title': m.get('command', ''), 'error': None}]})
            elif t == 'idle':
                err = m.get('error') or {}
                res.append({'role': 'idle', 'parts': [{'type': 'info', 'text': 'fertig: ' + str(m.get('outcome'))
                                                       + (' - ' + err.get('message', '') if err else '')}]})
        return res

    # -- Git --

    def git(self, *args, check=True):
        res = subprocess.run(['git', *args], cwd=self.project_path, capture_output=True, text=True,
                             encoding='utf-8', errors='replace')
        if check and res.returncode != 0:
            raise OpenCodeError(f'git {" ".join(args)}: {(res.stderr or res.stdout).strip()}')
        return res.stdout.strip()

    @staticmethod
    def branch_name(name):
        slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name).strip()).strip('_.-')
        if not slug:
            raise OpenCodeError('Name für den Branch fehlt')
        return 'feat_' + slug

    # -- Issues --

    def _issue(self, name, task_text, provider, model):
        issue = {'id': len(self.issues) + 1, 'name': name, 'branch': self.branch_name(name), 'task': task_text,
                 'provider': provider, 'model': model, 'status': 'wartet',
                 'session_id': None, 'response': '', 'commit': '', 'log': [], 'success': None,
                 'started': datetime.datetime.now().isoformat(timespec='seconds'), 'finished': None}
        self.issues.append(issue)
        return issue

    def _run_issue(self, issue):
        def log(msg):
            issue['log'].append(datetime.datetime.now().strftime('%H:%M:%S ') + msg)
            self.info(f"Issue {issue['name']}: {msg}")
        with self.lock:
            base_branch = None
            try:
                if not self.online():
                    issue['status'] = 'starte Server'
                    self.start_server()
                if self.git('status', '--porcelain'):
                    raise OpenCodeError('Arbeitsverzeichnis hat nicht committete Änderungen')
                base_branch = self.git('rev-parse', '--abbrev-ref', 'HEAD')
                branch = issue['branch']
                exists = self.git('rev-parse', '--verify', '--quiet', 'refs/heads/' + branch, check=False)
                if exists:
                    self.git('checkout', branch)
                    log(f'Branch {branch} ausgecheckt (existierte schon)')
                else:
                    self.git('checkout', '-b', branch)
                    log(f'Branch {branch} von {base_branch} angelegt')
                issue['status'] = 'opencode arbeitet'
                session = self.create_session(f"{branch}: {issue['task'][:60]}", issue['provider'], issue['model'])
                issue['session_id'] = session['id']
                log(f"Session {session['id']}, Modell {issue['provider'] or '-'}/{issue['model'] or 'Standard'}")
                issue['response'] = self.run(session['id'], issue['task'])
                log('opencode fertig')
                issue['status'] = 'commit'
                self.git('add', '-A')
                if not self.git('status', '--porcelain'):
                    raise OpenCodeError('opencode hat nichts geändert, kein Commit')
                message = f"feat: {issue['name']}\n\n{issue['task']}\n\nopencode session {session['id']}"
                self.git('commit', '-m', message)
                issue['commit'] = self.git('rev-parse', '--short', 'HEAD')
                log('Commit ' + issue['commit'])
                issue['status'] = 'push'
                self.git('push', '-u', self.remote, branch)
                log(f'Push nach {self.remote}/{branch}')
                issue['status'] = 'fertig'
                issue['success'] = True
            except Exception as e:
                issue['status'] = 'Fehler'
                issue['success'] = False
                log('Fehler: ' + str(e))
            finally:
                if base_branch and base_branch != 'HEAD':
                    try:
                        if self.git('status', '--porcelain'):
                            # Reste eines Fehlers bleiben im Branch, nicht in den Ausgangs-Branch mitnehmen
                            self.git('stash', 'push', '-u', '-m', f"opencode {issue['branch']}")
                            log('Nicht committete Änderungen in git stash gesichert')
                        self.git('checkout', base_branch)
                        log(f'Zurück auf {base_branch}')
                    except Exception as e:
                        log('Fehler beim Zurückwechseln: ' + str(e))
                issue['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
                self.emit_event('issue_done', name=issue['name'], branch=issue['branch'],
                                success=bool(issue['success']), message=issue['log'][-1] if issue['log'] else '')
        return issue

    @op("Erstellt den Branch feat_<name>, lässt opencode task_text bearbeiten, committet und pusht den Branch. "
        "wait=False: läuft im Hintergrund, Stand über op=issues",
        result={'success': 'bool', 'message': 'str', 'issue': 'dict'})
    def do_issue(self, name: str, task_text: str, wait: bool = False):
        """ erstellt einen Branch feat_$name und lässt opencode den task_text bearbeiten,
            danach wird der Branch committet und gepusht. Provider/Modell: self.provider, self.model (set_model) """
        try:
            issue = self._issue(name, task_text, self.provider, self.model)
        except OpenCodeError as e:
            return self.fail(str(e))
        if wait:
            self._run_issue(issue)
            return {'success': bool(issue['success']), 'message': issue['log'][-1] if issue['log'] else '',
                    'issue': issue}
        threading.Thread(target=self._run_issue, args=(issue,), daemon=True).start()
        return self.success(f"Issue {issue['branch']} gestartet", issue=issue)

    # -- Ops --

    @op("Server-URL, Erreichbarkeit, Version, Projekt und Branch",
        result={'success': 'bool', 'url': 'str', 'online': 'bool', 'version': 'str', 'running': 'bool'})
    def status(self):
        res = {'success': True, 'url': self.url or '', 'online': False, 'version': '',
               'running': self.process is not None and self.process.poll() is None,
               'project_path': self.project_path, 'bin': self.opencode_bin or '', 'provider': self.provider,
               'model': self.model, 'log': self.server_log[-30:]}
        try:
            res['version'] = self.server_info().get('version', '')
            res['online'] = True
        except Exception as e:
            res['error'] = str(e)
        try:
            res['branch'] = self.git('rev-parse', '--abbrev-ref', 'HEAD')
            res['dirty'] = bool(self.git('status', '--porcelain'))
        except Exception as e:
            res['branch'] = ''
            res['git_error'] = str(e)
        return res

    @op("opencode-Server starten", result={'success': 'bool', 'url': 'str'})
    def server_start(self):
        try:
            return self.success('gestartet', url=self.start_server())
        except Exception as e:
            return self.fail(str(e))

    @op("opencode-Server stoppen", result={'success': 'bool'})
    def server_stop(self):
        self.stop_server()
        return self.success('gestoppt')

    @op("Provider mit Modellen und Standard-Modellen", result={'success': 'bool', 'providers': 'list', 'default': 'dict'})
    def list_providers(self):
        try:
            return self.success('ok', **self.providers(), provider=self.provider, model=self.model)
        except Exception as e:
            return self.fail(str(e))

    @op("Standard-Provider und -Modell setzen", result={'success': 'bool'})
    def set_model(self, provider: str = '', model: str = ''):
        self.provider, self.model = provider, model
        return self.success('ok')

    @op("Alle Issues mit Status und Log", result={'success': 'bool', 'issues': 'list'})
    def list_issues(self):
        return self.success('ok', issues=list(reversed(self.issues)))

    @op("Nachrichten einer Session (Kurzfassung)", result={'success': 'bool', 'messages': 'list'})
    def session_messages(self, session_id: str, limit: int = 200):
        try:
            msgs = list(reversed(self.messages(session_id, limit, order='desc')))
            return self.success('ok', messages=self.summarize_messages(msgs))
        except Exception as e:
            return self.fail(str(e))

    @op("Pairing-Link für die opencode-Web-UI (einmalig, kurzlebig)", result={'success': 'bool', 'url': 'str'})
    def pairing(self):
        try:
            return self.success('ok', **self.pair())
        except Exception as e:
            return self.fail(str(e))

    @op("Sessions des Servers", result={'success': 'bool', 'sessions': 'list'})
    def list_sessions(self):
        try:
            return self.success('ok', sessions=self.sessions())
        except Exception as e:
            return self.fail(str(e))

    @op("Laufende Session abbrechen", result={'success': 'bool'})
    def abort(self, session_id: str):
        try:
            self.abort_session(session_id)
            return self.success('abgebrochen')
        except Exception as e:
            return self.fail(str(e))

    @op("Einzelne Nachricht an eine (neue) Session, ohne Git", result={'success': 'bool', 'response': 'str'})
    def ask(self, text: str, session_id: str = ''):
        try:
            if not session_id:
                session_id = self.create_session()['id']
            return self.success('ok', response=self.run(session_id, text), session_id=session_id)
        except Exception as e:
            return self.fail(str(e))

    # -- Web --

    PAGES = [('index', 'Issues'), ('server', 'Server')]

    def part_page(self, p: b.Page, name, html, js):
        p.nav(*[(title, self.link(page)) for page, title in self.PAGES], active=self.link(name))
        cfg = json.dumps({'type': self.type, 'provider': self.provider, 'model': self.model}).replace('</', '<\\/')
        p(f'<style>{OPENCODE_CSS}</style>')
        p(f'<div class="oc">{html}</div>')
        p(f'<script>(function() {{\nconst OC = {cfg};\n{OPENCODE_JS}\n{js}\n}})();</script>')

    def part_index(self, p: b.Page, params={}):
        self.part_page(p, 'index', OPENCODE_ISSUES_HTML, OPENCODE_ISSUES_JS)

    def part_server(self, p: b.Page, params={}):
        self.part_page(p, 'server', OPENCODE_SERVER_HTML, OPENCODE_SERVER_JS)


OPENCODE_CSS = """
.oc { max-width: 1100px; }
.oc-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
.oc-row > label:first-child { min-width: 110px; }
.oc textarea { width: 100%; box-sizing: border-box; font: inherit; }
.oc select { max-width: 360px; }
.oc-msg { margin: 6px 0; min-height: 1.2em; }
.oc-msg.error { color: #c33; }
.oc-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #c33; margin-right: 6px; }
.oc-dot.on { background: #2a7; }
.oc-issue { border: 1px solid #8886; border-radius: 6px; padding: 8px 10px; margin: 10px 0; }
.oc-issue h4 { margin: 0 0 4px; }
.oc-st { font-size: 85%; border: 1px solid #8888; border-radius: 8px; padding: 0 6px; margin-left: 6px; }
.oc-st.ok { color: #2a7; border-color: #2a7; }
.oc-st.err { color: #c33; border-color: #c33; }
.oc pre { white-space: pre-wrap; max-height: 360px; overflow: auto; margin: 4px 0; }
.oc-part { margin: 3px 0; font-size: 92%; }
.oc-part.tool { opacity: .8; font-family: monospace; }
.oc-part.reasoning { opacity: .6; font-style: italic; }
.oc-part.info { font-size: 85%; opacity: .7; }
.oc-table { border-collapse: collapse; }
.oc-table td, .oc-table th { border: 1px solid #8886; padding: 3px 8px; text-align: left; }
"""

OPENCODE_JS = r"""
const $ = (id) => document.getElementById(id);
function post(data) {
  const body = new URLSearchParams({ type: OC.type, ...data });
  return fetch(location.protocol + '//' + location.host + '/', { method: 'POST', body: body }).then((r) => r.json());
}
function msg(id, text, error) {
  const el = $(id);
  el.textContent = text || '';
  el.classList.toggle('error', !!error);
}
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}
"""

OPENCODE_ISSUES_HTML = """
<h3>Neues Issue</h3>
<div class="oc-row"><label for="oc-name">Name</label>
  <input id="oc-name" placeholder="z.B. login_form" size="30"> <span>Branch: <code id="oc-branch">feat_</code></span></div>
<div class="oc-row"><label for="oc-provider">Provider</label><select id="oc-provider"></select>
  <label for="oc-model">Modell</label><select id="oc-model"></select>
  <button id="oc-default" title="Als Standard speichern">als Standard</button></div>
<textarea id="oc-task" rows="8" placeholder="Aufgabe für opencode"></textarea>
<div class="oc-row"><button id="oc-run">Branch anlegen, bearbeiten, committen und pushen</button></div>
<div class="oc-msg" id="oc-msg"></div>
<h3>Issues</h3>
<div id="oc-issues"></div>
"""

OPENCODE_ISSUES_JS = r"""
let providers = [];
const open = {};
function fillModels() {
  const p = providers.find((x) => x.id === $('oc-provider').value);
  const sel = $('oc-model');
  const keep = sel.value || OC.model;
  sel.innerHTML = '';
  sel.add(new Option('- Standard -', ''));
  (p ? p.models : []).forEach((m) => sel.add(new Option(m.name + (m.name !== m.id ? ' (' + m.id + ')' : ''), m.id)));
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
  else if (p && OC.defaults[p.id]) sel.value = OC.defaults[p.id];
}
function loadProviders() {
  return post({ op: 'list_providers' }).then((r) => {
    const sel = $('oc-provider');
    sel.innerHTML = '';
    sel.add(new Option('- Standard -', ''));
    if (!r.success) { msg('oc-msg', r.message, true); return; }
    providers = r.providers;
    OC.defaults = r.default || {};
    providers.forEach((p) => sel.add(new Option(p.name, p.id)));
    sel.value = r.provider || '';
    OC.model = r.model;
    fillModels();
  });
}
$('oc-provider').onchange = fillModels;
$('oc-name').oninput = () => {
  $('oc-branch').textContent = 'feat_' + $('oc-name').value.trim().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[_.-]+|[_.-]+$/g, '');
};
$('oc-default').onclick = () => post({ op: 'set_model', provider: $('oc-provider').value, model: $('oc-model').value })
  .then((r) => msg('oc-msg', r.success ? 'Standard gespeichert' : r.message, !r.success));
$('oc-run').onclick = () => {
  const name = $('oc-name').value.trim(), task = $('oc-task').value.trim();
  if (!name || !task) { msg('oc-msg', 'Name und Aufgabe angeben', true); return; }
  $('oc-run').disabled = true;
  post({ op: 'set_model', provider: $('oc-provider').value, model: $('oc-model').value })
    .then(() => post({ op: 'do_issue', name: name, task_text: task }))
    .then((r) => {
      msg('oc-msg', r.message, !r.success);
      if (r.success) { $('oc-task').value = ''; open[r.issue.id] = true; }
      loadIssues();
    }).finally(() => { $('oc-run').disabled = false; });
};
function statusClass(i) { return i.success === true ? 'ok' : (i.success === false ? 'err' : ''); }
function renderParts(el, sid) {
  post({ op: 'session_messages', session_id: sid }).then((r) => {
    if (!r.success) { el.textContent = r.message; return; }
    el.innerHTML = r.messages.filter((m) => m.role !== 'user').map((m) => m.parts.map((p) => p.type === 'tool'
      ? '<div class="oc-part tool">' + esc(p.tool) + ' [' + esc(p.status) + '] ' + esc(p.title) + (p.error ? ' - ' + esc(p.error) : '') + '</div>'
      : '<div class="oc-part ' + p.type + '">' + esc(p.text) + '</div>').join('')).join('<hr>');
  });
}
function loadIssues() {
  return post({ op: 'list_issues' }).then((r) => {
    if (!r.success) return;
    const box = $('oc-issues');
    if (!r.issues.length) { box.innerHTML = '<i>noch keine</i>'; return; }
    box.innerHTML = r.issues.map((i) => '<div class="oc-issue"><h4>' + esc(i.branch)
      + '<span class="oc-st ' + statusClass(i) + '">' + esc(i.status) + '</span>'
      + (i.commit ? ' <code>' + esc(i.commit) + '</code>' : '') + '</h4>'
      + '<div><small>' + esc(i.started) + ' · ' + esc(i.provider || 'Standard') + '/' + esc(i.model || 'Standard') + '</small></div>'
      + '<details><summary>Aufgabe</summary><pre>' + esc(i.task) + '</pre></details>'
      + '<pre>' + esc(i.log.join('\n')) + '</pre>'
      + (i.response ? '<details><summary>Antwort</summary><pre>' + esc(i.response) + '</pre></details>' : '')
      + (i.session_id ? '<details data-id="' + i.id + '" data-sid="' + esc(i.session_id) + '"' + (open[i.id] ? ' open' : '')
         + '><summary>Verlauf (Session ' + esc(i.session_id) + ')</summary><div class="oc-parts">…</div>'
         + (i.success == null ? '<button data-abort="' + esc(i.session_id) + '">abbrechen</button>' : '') + '</details>' : '')
      + '</div>').join('');
    box.querySelectorAll('details[data-sid]').forEach((d) => {
      d.ontoggle = () => { open[d.dataset.id] = d.open; if (d.open) renderParts(d.querySelector('.oc-parts'), d.dataset.sid); };
      if (d.open) renderParts(d.querySelector('.oc-parts'), d.dataset.sid);
    });
    box.querySelectorAll('button[data-abort]').forEach((b) => {
      b.onclick = () => post({ op: 'abort', session_id: b.dataset.abort }).then((x) => msg('oc-msg', x.message, !x.success));
    });
    if (r.issues.some((i) => i.success == null)) setTimeout(loadIssues, 3000);
  });
}
loadProviders();
loadIssues();
"""

OPENCODE_SERVER_HTML = """
<h3>Server</h3>
<div class="oc-row"><span class="oc-dot" id="oc-dot"></span><span id="oc-url"></span> <span id="oc-version"></span></div>
<table class="oc-table">
  <tr><th>Projekt</th><td id="oc-project"></td></tr>
  <tr><th>Branch</th><td id="oc-git"></td></tr>
  <tr><th>opencode</th><td id="oc-bin"></td></tr>
</table>
<div class="oc-row"><button id="oc-start">Server starten</button><button id="oc-stop">Server stoppen</button>
  <button id="oc-refresh">Aktualisieren</button><button id="oc-pair">Web-UI öffnen (Pairing)</button></div>
<div class="oc-msg" id="oc-msg"></div>
<details><summary>Server-Ausgabe</summary><pre id="oc-log"></pre></details>
<h3>Sessions</h3>
<div id="oc-sessions"></div>
"""

OPENCODE_SERVER_JS = r"""
function load() {
  post({ op: 'status' }).then((r) => {
    $('oc-dot').classList.toggle('on', r.online);
    $('oc-url').textContent = (r.url || '-') + (r.online ? ' erreichbar' : ' nicht erreichbar') + (r.error && !r.online ? ' (' + r.error + ')' : '');
    $('oc-version').textContent = r.version ? 'v' + r.version : '';
    $('oc-project').textContent = r.project_path;
    $('oc-git').textContent = r.branch ? r.branch + (r.dirty ? ' (nicht committete Änderungen)' : '') : (r.git_error || '');
    $('oc-bin').textContent = (r.bin || 'nicht gefunden') + (r.running ? ' (läuft)' : '');
    $('oc-log').textContent = (r.log || []).join('\n');
  });
  post({ op: 'list_sessions' }).then((r) => {
    if (!r.success) { $('oc-sessions').textContent = r.message; return; }
    $('oc-sessions').innerHTML = '<table class="oc-table"><tr><th>ID</th><th>Titel</th><th>Geändert</th></tr>'
      + r.sessions.map((s) => '<tr><td><code>' + esc(s.id) + '</code></td><td>' + esc(s.title) + '</td><td>'
        + esc(s.time && s.time.updated ? new Date(s.time.updated).toLocaleString() : '') + '</td></tr>').join('') + '</table>';
  });
}
function action(op) {
  msg('oc-msg', op + ' …');
  post({ op: op }).then((r) => { msg('oc-msg', r.message, !r.success); load(); });
}
$('oc-start').onclick = () => action('server_start');
$('oc-stop').onclick = () => action('server_stop');
$('oc-refresh').onclick = load;
$('oc-pair').onclick = () => post({ op: 'pairing' }).then((r) => {
  if (r.success) window.open(r.url, '_blank'); else msg('oc-msg', r.message, true);
});
load();
"""
