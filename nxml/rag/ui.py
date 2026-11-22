#
# Project: nxml
#
# Web-Oberfläche und Chat für nxml.rag.RagRunner (Mixin, wird von RagRunner eingebunden):
#
#   Chat        Frage -> query (Kontext aus dem Index) -> LLM aus den geladenen Runnern -> Antwort mit Quellen
#   Suche       query mit data_filter, Treffer mit Score
#   Indizieren  Text + document_data (JSON) oder Text-Datei
#   Dokumente   indizierte Dokumente, löschen
#   Status      Backend, Anzahl, Index-Statistik, Embedder
#   nweb        nweb-Instanzen aus der Konfiguration, Gruppenbaum, Gruppe mit Dokumenten indizieren
#
import json
from html import escape as esc

from nwebclient import NWebClient
from nwebclient import base as b
from nwebclient import web as w
from nwebclient.runner import op


DEFAULT_SYSTEM = ("Beantworte die Frage ausschließlich anhand des folgenden Kontexts. Wenn der Kontext die Antwort "
                  "nicht enthält, sage das. Nenne die verwendeten Quellen als [1], [2], ...")


def build_context(hits: list, max_chars: int = 12000) -> str:
    parts, total = [], 0
    for i, h in enumerate(hits, 1):
        part = f"[{i}] (Dokument {h.get('doc_id')}, Abschnitt {h.get('chunk_index')})\n{h.get('text', '')}"
        if parts and total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return '\n\n'.join(parts)


class RagUiMixin:
    """ erwartet von RagRunner: type, store, query(), index_document(), run_job(), link(), success(), fail() """

    # --- LLM-Auswahl ----------------------------------------------------------------------------

    def _dispatcher(self):
        parents = self.getParents() if hasattr(self, 'getParents') else []
        for x in list(parents or []) + [self.getRoot()]:
            if x is not None and hasattr(x, 'loaded'):
                return x
            inner = getattr(x, 'jobexecutor', None)
            if inner is not None and hasattr(inner, 'loaded'):
                return inner
        return None

    @staticmethod
    def is_llm(runner) -> bool:
        tags = getattr(runner, 'TAGS', []) or []
        if 'llm' not in tags or type(runner).__name__ in ('LlmQueue', 'LlmProxy'):
            return False
        return callable(getattr(runner, 'execute_chat', None)) or callable(getattr(runner, 'prompt', None))

    def llm_runners(self) -> list:
        dispatcher = self._dispatcher()
        if dispatcher is None:
            return []
        res = []
        for runner in dispatcher.loaded():
            try:
                if runner is not self and self.is_llm(runner):
                    res.append({'type': runner.type, 'class': type(runner).__name__,
                                'chat': callable(getattr(runner, 'execute_chat', None))})
            except Exception:
                pass
        return sorted(res, key=lambda x: x['type'])

    @op("Geladene LLM-Runner (TAGS llm), chat: unterstützt op=chat mit messages",
        result={'success': 'bool', 'llms': 'list', 'default': 'str'})
    def llms(self):
        llms = self.llm_runners()
        default = self.llm if any(x['type'] == self.llm for x in llms) else (llms[0]['type'] if llms else '')
        return self.success('ok', llms=llms, default=default)

    # --- Chat -----------------------------------------------------------------------------------

    def ask_llm(self, llm: str, messages: list, data: dict) -> dict:
        info = next((x for x in self.llm_runners() if x['type'] == llm), None)
        if info is not None and info['chat']:
            job = {'type': llm, 'op': 'chat', 'messages': messages}
            for key in ('model', 'temperature', 'max_tokens'):
                if data.get(key) not in (None, ''):
                    job[key] = data[key]
        else:
            # nur prompt: Verlauf als Text
            text = '\n\n'.join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
            job = {'type': llm, 'prompt': text + '\n\nASSISTANT:\n'}
        result = self.run_job(job)
        if not result.get('success', True) or 'response' not in result:
            raise Exception(str(result.get('message') or result.get('error_message') or 'keine Antwort vom LLM'))
        return result

    @op("Frage an ein LLM mit Kontext aus dem Index. history: [{role, content}], data_filter wie bei query",
        result={'success': 'bool', 'response': 'str', 'hits': 'list', 'llm': 'str'})
    def chat(self, question: str, llm: str = '', data_filter: dict = None, limit: int = 5, history: list = None,
             system: str = '', min_score: float = None, model: str = '', temperature: float = None,
             max_tokens: int = None):
        try:
            llm = llm or self.llm or (self.llm_runners() or [{}])[0].get('type', '')
            if not llm:
                return self.fail('kein LLM-Runner geladen', response='', hits=[])
            hits = self.query(question, data_filter, limit, min_score) if limit > 0 else []
            context = build_context(hits)
            messages = [{'role': 'system', 'content': (system or DEFAULT_SYSTEM) + '\n\nKontext:\n'
                         + (context or '(keine passenden Abschnitte gefunden)')}]
            for m in history or []:
                if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and m.get('content'):
                    messages.append({'role': m['role'], 'content': str(m['content'])})
            messages.append({'role': 'user', 'content': question})
            result = self.ask_llm(llm, messages, dict(model=model, temperature=temperature, max_tokens=max_tokens))
            return self.success('ok', response=str(result.get('response', '')),
                                reasoning=result.get('reasoning', ''), hits=hits, llm=llm,
                                model=result.get('model', ''))
        except Exception as e:
            return self.fail(str(e), response='', hits=[])

    # --- Web ------------------------------------------------------------------------------------

    RAG_PAGES = [('index', 'Chat'), ('search', 'Suche'), ('add', 'Indizieren'), ('documents', 'Dokumente'),
                 ('status', 'Status'), ('nweb', 'nweb')]

    def rag_page(self, p: b.Page, name, html, js=''):
        p.nav(*[(title, self.link(page)) for page, title in self.RAG_PAGES], active=self.link(name))
        cfg = json.dumps({'type': self.type, 'llm': self.llm}).replace('</', '<\\/')
        p(f'<style>{RAG_CSS}</style>')
        p(f'<div class="rag">{html}</div>')
        p(f'<script>(function() {{\nconst RAG = {cfg};\n{RAG_JS}\n{js}\n}})();</script>')

    def part_index(self, p: b.Page, params={}):
        self.rag_page(p, 'index', CHAT_HTML, CHAT_JS)

    def part_search(self, p: b.Page, params={}):
        self.rag_page(p, 'search', SEARCH_HTML, SEARCH_JS)

    def part_add(self, p: b.Page, params={}):
        self.rag_page(p, 'add', ADD_HTML, ADD_JS)

    def part_documents(self, p: b.Page, params={}):
        self.rag_page(p, 'documents', DOCS_HTML, DOCS_JS)

    def part_status(self, p: b.Page, params={}):
        rows = [('Backend', self.backend), ('Embedder', self.embedder + (f' ({self.embedding_model})'
                                                                       if self.embedding_model else '')),
                ('Chunk-Größe', f'{self.chunk_size} Zeichen, Überlappung {self.chunk_overlap}'),
                ('Standard-LLM', self.llm or '(erstes geladenes)')]
        try:
            store = self.store
            rows.append(('Chunks', store.count()))
            for key, value in (store.stats() if hasattr(store, 'stats') else {}).items():
                rows.append((key, value))
        except Exception as e:
            rows.append(('Fehler', str(e)))
        llms = ', '.join(x['type'] for x in self.llm_runners()) or '(keine geladen)'
        rows.append(('Geladene LLMs', llms))
        html = '<table class="rag-table">' + ''.join(f'<tr><th>{esc(str(k))}</th><td>{esc(str(v))}</td></tr>'
                                                     for k, v in rows) + '</table>'
        if hasattr(getattr(self, '_store', None), 'train'):
            html += '<p><button id="rag-train">IVF-Index neu trainieren</button> <span id="rag-msg" ' \
                    'class="rag-msg"></span></p>'
        self.rag_page(p, 'status', html, STATUS_JS)

    # --- nweb -----------------------------------------------------------------------------------

    NWEB_SKIP_KINDS = ('image', 'file', 'binary', 'audio', 'video', 'pdf')
    NWEB_PREVIEW_DOCS = 25

    def nweb_instances(self) -> dict:
        """ nweb-Instanzen aus Runner-Args, nweb.json und /etc/nweb.json: Name -> URL """
        args = getattr(self, 'args', None)
        res = {}
        for name in NWebClient.list(args):
            try:
                res[name] = NWebClient.config_of(name, args).get('url', '')
            except KeyError:
                pass
        return res

    def nweb_client(self, name: str) -> NWebClient:
        """ :raises KeyError: unbekannte Instanz """
        return NWebClient(name, args=getattr(self, 'args', None))

    @staticmethod
    def nweb_group_tree(groups) -> tuple:
        """ :return: (Wurzeln, {group_id: [Untergruppen]}), jeweils nach Titel sortiert """
        ids = {str(g.id()) for g in groups}
        children, roots = {}, []
        for g in groups:
            parent = str(g.get('parent', '0') or '0')
            if parent in ids and parent != str(g.id()):
                children.setdefault(parent, []).append(g)
            else:
                roots.append(g)
        key = lambda g: str(g.get('title', '') or '').lower()
        for items in children.values():
            items.sort(key=key)
        return sorted(roots, key=key), children

    def nweb_doc_text(self, nc, doc) -> str:
        """ Text eines nweb-Dokuments (Titel + Inhalt), leer bei Bildern/Dateien """
        if doc.get('kind', '') in self.NWEB_SKIP_KINDS:  # TODO bei image steht der text in meta, pdf oder description
            return ''
        if doc.get('content') is None:
            doc = nc.doc(doc.id())  # Listen liefern den Inhalt nicht immer mit
        try:
            content = doc.content_text()
        except Exception:
            # content_text braucht bs4 für markup, sonst HTML ohne Zusatzpakete umwandeln
            content = str(doc.get('content', '') or '')
            if doc.get('kind', '') == 'markup' or '<' in content:
                from nxml.order_extract import html_to_text, looks_like_html
                if looks_like_html(content):
                    content = html_to_text(content)
        content = str(content or '').strip()
        if not content:
            return ''
        title = str(doc.get('title', '') or '').strip()
        return (title + '\n\n' + content) if title and not content.startswith(title) else content

    @op("Dokumente einer nweb-Gruppe indizieren (Bilder/Dateien werden übersprungen), recursive: mit Untergruppen. "
        "doc_id: nweb:<instance>:<guid>, document_data: source, instance, group_id, group, nweb_id, title, kind, url",
        result={'success': 'bool', 'indexed': 'int', 'chunks': 'int', 'skipped': 'int', 'errors': 'list'})
    def index_nweb_group(self, instance: str, group_id: str, recursive: bool = False):
        try:
            nc = self.nweb_client(instance)
            group = nc.group(group_id)
            groups, todo = [], [group]
            while todo:
                g = todo.pop(0)
                groups.append(g)
                if recursive:
                    todo.extend(g.groups())
        except Exception as e:
            return self.fail(str(e), indexed=0, chunks=0, skipped=0, errors=[str(e)])
        indexed = chunks = skipped = 0
        errors = []
        for g in groups:
            for doc in g.docs():
                try:
                    text = self.nweb_doc_text(nc, doc)
                    if not text:
                        skipped += 1
                        continue
                    data = {'source': 'nweb', 'instance': instance, 'group_id': str(g.id()),
                            'group': str(g.get('title', '') or ''), 'nweb_id': str(doc.id()),
                            'title': str(doc.get('title', '') or ''), 'kind': str(doc.get('kind', '') or ''),
                            'url': doc.url(), 'doc_id': f"nweb:{instance}:{doc.get('guid') or doc.id()}"}
                    res = self.index_document(text, data)
                    indexed += 1
                    chunks += res['chunks']
                except Exception as e:
                    errors.append(f"{doc.get('title', doc.get('document_id', '?'))}: {e}")
        msg = f'{indexed} Dokumente indiziert ({chunks} Chunks), {skipped} übersprungen'
        return self.success(msg, indexed=indexed, chunks=chunks, skipped=skipped, errors=errors[:50],
                            groups=len(groups))

    def part_nweb(self, p: b.Page, params={}):
        instances = self.nweb_instances()
        if not instances:
            html = '<p>Keine nweb-Instanz konfiguriert. Eine Instanz ist ein Eintrag mit <code>url</code>, ' \
                   '<code>username</code> und <code>password</code> in der Konfiguration (z.B. nweb.json).</p>'
        else:
            html = '<table class="rag-table"><tr><th>Instanz</th><th>URL</th></tr>' + ''.join(
                f'<tr><td>{w.a(esc(key), self.link(self.part_nweb_instance, n=key))}</td>'
                f'<td>{esc(str(url))}</td></tr>' for key, url in sorted(instances.items())) + '</table>'
        self.rag_page(p, 'nweb', '<h3>nweb-Instanzen</h3>' + html)

    def part_nweb_instance(self, p: b.Page, params={}):
        name = params.get('n', '')
        html = f'<p>{w.a("nweb-Instanzen", self.link(self.part_nweb))} › <b>{esc(name)}</b></p>'
        try:
            nc = self.nweb_client(name)
            groups = list(nc.groups())
        except Exception as e:
            self.rag_page(p, 'nweb', html + f'<p class="rag-msg error">{esc(str(e))}</p>')
            return
        roots, children = self.nweb_group_tree(groups)

        def tree(items, depth=0):
            if not items or depth > 30:
                return ''
            res = '<ul class="rag-tree">'
            for g in items:
                gid = str(g.id())
                sub = children.get(gid, [])
                link = w.a(esc(str(g.get('title', '') or gid)), self.link(self.part_nweb__group, n=name, g=gid))
                count = f' <span class="rag-meta">({len(sub)})</span>' if sub else ''
                res += f'<li>{link}{count}{tree(sub, depth + 1)}</li>'
            return res + '</ul>'

        html += f'<h3>Gruppen ({len(groups)})</h3>' + (tree(roots) or '<p>Keine Gruppen.</p>')
        self.rag_page(p, 'nweb', html)

    def part_nweb__group(self, p: b.Page, params={}):
        name, gid = params.get('n', ''), str(params.get('g', ''))
        crumbs = f'{w.a("nweb-Instanzen", self.link(self.part_nweb))} › ' \
                 f'{w.a(esc(name), self.link(self.part_nweb_instance, n=name))}'
        try:
            nc = self.nweb_client(name)
            group = nc.group(gid)
            subgroups = sorted(group.groups(), key=lambda g: str(g.get('title', '') or '').lower())
            docs = list(group.docs())
        except Exception as e:
            self.rag_page(p, 'nweb', f'<p>{crumbs}</p><p class="rag-msg error">{esc(str(e))}</p>')
            return
        parent = str(group.get('parent', '0') or '0')
        if parent != '0' and any(str(g.id()) == parent for g in nc.groups()):
            parent_group = nc.group(parent)
            crumbs += ' › ' + w.a(esc(str(parent_group.get('title', '') or parent)),
                                  self.link(self.part_nweb__group, n=name, g=parent))
        title = str(group.get('title', '') or gid)
        html = f'<p>{crumbs} › <b>{esc(title)}</b></p>'
        textual = [d for d in docs if d.get('kind', '') not in self.NWEB_SKIP_KINDS]
        html += f'<p>{len(docs)} Dokumente, davon {len(textual)} mit Text, {len(subgroups)} Untergruppen</p>'
        html += '<div class="rag-row"><button id="rag-nweb-index">Gruppe indizieren</button>' \
                '<label><input type="checkbox" id="rag-nweb-rec" /> mit Untergruppen</label>' \
                '<span id="rag-msg" class="rag-msg"></span></div><div id="rag-out"></div>'
        if subgroups:
            html += '<h4>Untergruppen</h4><ul class="rag-tree">' + ''.join(
                f'<li>{w.a(esc(str(g.get("title", "") or g.id())), self.link(self.part_nweb__group, n=name, g=g.id()))}'
                f'</li>' for g in subgroups) + '</ul>'
        if docs:
            html += '<h4>Dokumente</h4><table class="rag-table"><tr><th>Titel</th><th>Art</th><th>ID</th></tr>'
            for d in docs[:self.NWEB_PREVIEW_DOCS]:
                try:
                    url = d.url()
                except Exception:
                    url = ''
                t = esc(str(d.get('title', '') or d.get('name', '') or ''))
                html += f'<tr><td>{w.a(t, url) if url else t}</td><td>{esc(str(d.get("kind", "") or ""))}</td>' \
                        f'<td>{esc(str(d.get("document_id", "")))}</td></tr>'
            html += '</table>'
            if len(docs) > self.NWEB_PREVIEW_DOCS:
                html += f'<p class="rag-meta">… und {len(docs) - self.NWEB_PREVIEW_DOCS} weitere</p>'
        js_str = lambda v: json.dumps(v).replace('</', '<\\/')
        js = NWEB_GROUP_JS.replace('__INSTANCE__', js_str(name)).replace('__GROUP__', js_str(gid))
        self.rag_page(p, 'nweb', html, js)


RAG_CSS = """
.rag { max-width: 1100px; }
.rag-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
.rag-row > label:first-child { min-width: 110px; }
.rag textarea, .rag input[type=text] { width: 100%; box-sizing: border-box; font: inherit; }
.rag input[type=number] { width: 80px; }
.rag-msg { margin: 6px 0; min-height: 1.2em; }
.rag-msg.error { color: #c33; }
.rag-chat { border: 1px solid #8886; border-radius: 6px; padding: 8px; min-height: 160px; max-height: 60vh;
  overflow-y: auto; margin: 8px 0; }
.rag-m { margin: 6px 0; padding: 6px 10px; border-radius: 6px; white-space: pre-wrap; }
.rag-m.user { background: #4af2; }
.rag-m.assistant { background: #8882; }
.rag-m .rag-role { font-size: 80%; opacity: .7; display: block; }
.rag-m details { font-size: 90%; margin-top: 4px; white-space: normal; }
.rag-hit { border-left: 3px solid #4af8; padding: 4px 8px; margin: 6px 0; white-space: pre-wrap; }
.rag-hit .rag-meta { font-size: 80%; opacity: .75; white-space: normal; }
.rag-table { border-collapse: collapse; margin-top: 10px; }
.rag-table td, .rag-table th { border: 1px solid #8886; padding: 3px 8px; text-align: left; vertical-align: top; }
.rag-flex { display: flex; gap: 10px; }
.rag-tree { list-style: none; padding-left: 18px; margin: 2px 0; }
.rag-tree li { margin: 2px 0; }
.rag-tree li::before { content: '▸ '; opacity: .5; }
.rag-meta { font-size: 80%; opacity: .75; }
.rag-flex > * { flex: 1; }
"""

RAG_JS = r"""
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
function post(data) {
  const body = new URLSearchParams({ type: RAG.type });
  for (const [k, v] of Object.entries(data)) {
    if (v === null || v === undefined || v === '') continue;
    body.append(k, typeof v === 'object' ? JSON.stringify(v) : v);
  }
  return fetch(location.protocol + '//' + location.host + '/', { method: 'POST', body: body }).then((r) => r.json());
}
function msg(id, text, error) { const e = $(id); if (e) { e.textContent = text || ''; e.className = 'rag-msg' + (error ? ' error' : ''); } }
function parseJson(id) {
  const v = ($(id).value || '').trim();
  if (!v) return null;
  try { return JSON.parse(v); } catch (e) { throw new Error(id + ': kein gültiges JSON'); }
}
function hitHtml(h, i) {
  return '<div class="rag-hit"><div class="rag-meta">[' + (i + 1) + '] ' + esc(h.doc_id) + ' #' + esc(h.chunk_index)
    + ' · Score ' + Number(h.score).toFixed(3) + ' · ' + esc(JSON.stringify(h.data)) + '</div>' + esc(h.text) + '</div>';
}
"""

FILTER_ROW = """
<div class="rag-row"><label for="rag-filter">Filter (JSON)</label>
  <input type="text" id="rag-filter" placeholder='{"lang": "de", "year": [2025, 2026]}' style="flex:1" />
  <label>Treffer <input type="number" id="rag-limit" value="5" min="0" max="50" /></label></div>
"""

CHAT_HTML = """
<div class="rag-row"><label for="rag-llm">LLM</label><select id="rag-llm"></select>
  <label><input type="checkbox" id="rag-history" checked /> Verlauf mitsenden</label>
  <button id="rag-clear">Neuer Chat</button></div>
""" + FILTER_ROW + """
<details><summary>System-Prompt</summary><textarea id="rag-system" rows="3"></textarea></details>
<div class="rag-chat" id="rag-chat"></div>
<textarea id="rag-question" rows="3" placeholder="Frage (Strg+Enter sendet)"></textarea>
<div class="rag-row"><button id="rag-send">Senden</button><span id="rag-msg" class="rag-msg"></span></div>
"""

CHAT_JS = """
let history = [];
$('rag-system').value = """ + json.dumps(DEFAULT_SYSTEM) + """;
post({ op: 'llms' }).then((r) => {
  const sel = $('rag-llm');
  (r.llms || []).forEach((l) => {
    const o = document.createElement('option');
    o.value = l.type; o.textContent = l.type + ' (' + l['class'] + (l.chat ? '' : ', nur prompt') + ')';
    sel.appendChild(o);
  });
  if (r.default) sel.value = r.default;
  if (!(r.llms || []).length) msg('rag-msg', 'Kein LLM-Runner geladen (Runner mit TAGS llm).', true);
});
function add(role, html) {
  const div = document.createElement('div');
  div.className = 'rag-m ' + role;
  div.innerHTML = '<span class="rag-role">' + esc(role) + '</span>' + html;
  $('rag-chat').appendChild(div);
  $('rag-chat').scrollTop = $('rag-chat').scrollHeight;
}
function send() {
  const q = $('rag-question').value;
  if (!q.trim()) return;
  let filter;
  try { filter = parseJson('rag-filter'); } catch (e) { msg('rag-msg', e.message, true); return; }
  add('user', esc(q));
  $('rag-question').value = '';
  $('rag-send').disabled = true;
  msg('rag-msg', 'Suche Kontext und frage ' + $('rag-llm').value + ' ...');
  const t0 = Date.now();
  post({ op: 'chat', question: q, llm: $('rag-llm').value, data_filter: filter, limit: $('rag-limit').value,
         system: $('rag-system').value, history: $('rag-history').checked ? history : [] }).then((r) => {
    if (!r.success) { msg('rag-msg', r.message, true); return; }
    const think = r.reasoning ? '<details><summary>Reasoning</summary>' + esc(r.reasoning) + '</details>' : '';
    const src = (r.hits || []).length ? '<details><summary>Quellen (' + r.hits.length + ')</summary>'
      + r.hits.map(hitHtml).join('') + '</details>' : '';
    add('assistant', esc(r.response) + think + src);
    history.push({ role: 'user', content: q }, { role: 'assistant', content: r.response });
    msg('rag-msg', r.llm + (r.model ? ' / ' + r.model : '') + ' · ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s');
  }).catch((e) => msg('rag-msg', String(e), true)).finally(() => { $('rag-send').disabled = false; });
}
$('rag-send').onclick = send;
$('rag-question').addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.ctrlKey) send(); });
$('rag-clear').onclick = () => { history = []; $('rag-chat').innerHTML = ''; msg('rag-msg', ''); };
"""

SEARCH_HTML = """
<textarea id="rag-q" rows="2" placeholder="Suchtext"></textarea>
""" + FILTER_ROW + """
<div class="rag-row"><button id="rag-go">Suchen</button><span id="rag-msg" class="rag-msg"></span></div>
<div id="rag-hits"></div>
"""

SEARCH_JS = """
function search() {
  let filter;
  try { filter = parseJson('rag-filter'); } catch (e) { msg('rag-msg', e.message, true); return; }
  msg('rag-msg', 'Suche ...');
  const t0 = Date.now();
  post({ op: 'query', text: $('rag-q').value, data_filter: filter, limit: $('rag-limit').value || 5 }).then((r) => {
    if (!r.success) { msg('rag-msg', r.message, true); return; }
    $('rag-hits').innerHTML = r.hits.map(hitHtml).join('') || '<p>Keine Treffer.</p>';
    msg('rag-msg', r.hits.length + ' Treffer · ' + (Date.now() - t0) + ' ms');
  });
}
$('rag-go').onclick = search;
$('rag-q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); search(); } });
"""

ADD_HTML = """
<div class="rag-row"><label for="rag-file">Datei</label><input type="file" id="rag-file" accept=".txt,.md,.csv,.json,.html,.xml" multiple /></div>
<textarea id="rag-text" rows="12" placeholder="Text"></textarea>
<div class="rag-row"><label for="rag-data">document_data</label>
  <input type="text" id="rag-data" placeholder='{"doc_id": "handbuch-1", "lang": "de"}' style="flex:1" /></div>
<div class="rag-row"><button id="rag-index">Indizieren</button><button id="rag-chunk">Nur Chunks anzeigen</button>
  <span id="rag-msg" class="rag-msg"></span></div>
<div id="rag-out"></div>
"""

ADD_JS = """
function readFile(f) { return new Promise((ok) => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.readAsText(f); }); }
async function indexText(text, data) {
  const r = await post({ op: 'index_document', text: text, document_data: data || {} });
  if (!r.success) throw new Error(r.message);
  return r;
}
$('rag-index').onclick = async () => {
  let data;
  try { data = parseJson('rag-data') || {}; } catch (e) { msg('rag-msg', e.message, true); return; }
  $('rag-index').disabled = true;
  try {
    const files = Array.from($('rag-file').files);
    const out = [];
    if (files.length) {
      for (const f of files) {
        msg('rag-msg', 'Indiziere ' + f.name + ' ...');
        const d = Object.assign({ doc_id: f.name, filename: f.name }, data);
        if (files.length > 1) d.doc_id = f.name;
        const r = await indexText(await readFile(f), d);
        out.push(esc(r.doc_id) + ': ' + r.chunks + ' Chunks');
      }
    } else {
      if (!$('rag-text').value.trim()) { msg('rag-msg', 'Kein Text', true); return; }
      msg('rag-msg', 'Indiziere ...');
      const r = await indexText($('rag-text').value, data);
      out.push(esc(r.doc_id) + ': ' + r.chunks + ' Chunks');
    }
    $('rag-out').innerHTML = '<p>' + out.join('<br>') + '</p>';
    msg('rag-msg', 'Fertig');
  } catch (e) { msg('rag-msg', e.message, true); }
  finally { $('rag-index').disabled = false; }
};
$('rag-chunk').onclick = () => {
  post({ op: 'chunk', text: $('rag-text').value }).then((r) => {
    $('rag-out').innerHTML = '<p>' + r.chunks.length + ' Chunks</p>' + r.chunks.map((c, i) =>
      '<div class="rag-hit"><div class="rag-meta">#' + i + ' · ' + c.length + ' Zeichen</div>' + esc(c) + '</div>').join('');
  });
};
"""

DOCS_HTML = """
<div class="rag-row"><span id="rag-msg" class="rag-msg"></span></div>
<table class="rag-table" id="rag-docs"></table>
"""

DOCS_JS = """
function load() {
  post({ op: 'list_documents', limit: 1000 }).then((r) => {
    if (!r.success) { msg('rag-msg', r.message, true); return; }
    msg('rag-msg', r.documents.length + ' Dokumente');
    $('rag-docs').innerHTML = '<tr><th>doc_id</th><th>Chunks</th><th>document_data</th><th></th></tr>'
      + r.documents.map((d) => '<tr><td>' + esc(d.doc_id) + '</td><td>' + d.chunks + '</td><td><code>'
        + esc(JSON.stringify(d.data)) + '</code></td><td><button data-doc="' + esc(d.doc_id) + '">Löschen</button></td></tr>').join('');
    $('rag-docs').querySelectorAll('button[data-doc]').forEach((b) => b.onclick = () => {
      if (!confirm('Dokument ' + b.dataset.doc + ' löschen?')) return;
      post({ op: 'delete_document', doc_id: b.dataset.doc }).then(load);
    });
  });
}
load();
"""

STATUS_JS = """
if ($('rag-train')) $('rag-train').onclick = () => {
  msg('rag-msg', 'Trainiere ...');
  post({ op: 'train_index' }).then((r) => { msg('rag-msg', r.message, !r.success); if (r.success) location.reload(); });
};
"""

NWEB_GROUP_JS = """
$('rag-nweb-index').onclick = () => {
  $('rag-nweb-index').disabled = true;
  msg('rag-msg', 'Indiziere ... (kann dauern)');
  const t0 = Date.now();
  post({ op: 'index_nweb_group', instance: __INSTANCE__, group_id: __GROUP__, recursive: $('rag-nweb-rec').checked ? 1 : 0 })
    .then((r) => {
      msg('rag-msg', (r.message || '') + ' · ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s', !r.success);
      $('rag-out').innerHTML = (r.errors || []).length ? '<details><summary>Fehler (' + r.errors.length + ')</summary><pre>'
        + r.errors.map(esc).join('\\n') + '</pre></details>' : '';
    }).catch((e) => msg('rag-msg', String(e), true)).finally(() => { $('rag-nweb-index').disabled = false; });
};
"""
