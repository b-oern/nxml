#
# Project: nxml
#
# Bestellungen (Shop, Bestellnummer, Positionen, Summen) aus Bestellbestätigungs-Mails per LLM extrahieren.
#
#   Mail (text/html/eml/msg) -> Vorfilter -> LlmQueue (tags mailextract, order) -> on_done: store
#       Bestellung  -> mail_orders + order_posten (nweb-Persistence, sonst im Speicher)
#       keine       -> Liste "Übersprungen" (nur im RAM)
#
# Config (runners):
#     llm_queue: nxml.llm:LlmQueue
#     order_extract: nxml.order_extract:OrderExtractor     # args: llm=lmstudio, batch=1
#
import collections
import email
import email.policy
import email.utils
import html as html_lib
import json
import re
import threading
import time
from html import escape as esc
from html.parser import HTMLParser

from nwebclient import runner as r
from nwebclient.runner import op
from nwebclient import base as b
from nwebclient import util as u

from nxml.store import NwebTable, nweb_store_of, parse_tags


# --- Mail laden -----------------------------------------------------------------------------------------------------

class _HtmlText(HTMLParser):
    """ HTML -> Text, Tabellenzellen werden mit ' | ' getrennt, damit Spalten (Menge, Preis) erhalten bleiben """

    BLOCK = {'p', 'div', 'br', 'li', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'section', 'hr'}
    SKIP = {'script', 'style', 'head', 'title'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0
        self.cell = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == 'tr':
            self.parts.append('\n')
            self.cell = False
        elif tag in ('td', 'th'):
            if self.cell:
                self.parts.append(' | ')
            self.cell = True
        elif tag in self.BLOCK:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in ('tr', 'table') or tag in self.BLOCK:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(re.sub(r'\s+', ' ', data))


def html_to_text(html: str) -> str:
    parser = _HtmlText()
    try:
        parser.feed(html)
        parser.close()
        text = ''.join(parser.parts)
    except Exception:
        text = html_lib.unescape(re.sub(r'<[^>]+>', ' ', html))
    lines = []
    for line in text.split('\n'):
        line = re.sub(r'[ \t ​‌͏]+', ' ', line).strip()
        line = re.sub(r'^(\|\s*)+|(\s*\|)+$', '', line).strip()
        line = re.sub(r'(\s*\|\s*){2,}', ' | ', line)
        if line:
            lines.append(line)
    return '\n'.join(lines).strip()


def looks_like_html(text: str) -> bool:
    return bool(re.search(r'<(html|body|table|div|p|br|td|span)\b', text or '', re.I))


AMOUNT_RE = re.compile(r'(?:€|EUR|USD|\$|CHF|£)\s*\d|\d[\d.,]*\s*(?:€|EUR|USD|\$|CHF|£)', re.I)


def _from_email_message(msg) -> dict:
    plain, html = '', ''
    try:
        part = msg.get_body(preferencelist=('plain',))
        plain = part.get_content() if part is not None else ''
        part = msg.get_body(preferencelist=('html',))
        html = part.get_content() if part is not None else ''
    except Exception:
        payload = msg.get_payload(decode=True)
        plain = payload.decode(msg.get_content_charset() or 'utf-8', 'replace') if payload else ''
    # Text-Teil nur, wenn er Beträge enthält, sonst ist die HTML-Tabelle meist vollständiger
    text = plain if plain and (AMOUNT_RE.search(plain) or not html) else html_to_text(html)
    date = msg.get('Date', '')
    try:
        date = email.utils.parsedate_to_datetime(date).isoformat()
    except Exception:
        pass
    return {'message_id': str(msg.get('Message-ID', '') or '').strip(), 'sender': str(msg.get('From', '') or ''),
            'subject': str(msg.get('Subject', '') or ''), 'date': str(date or ''), 'text': text}


def parse_eml(eml) -> dict:
    """ RFC822-Mail (bytes oder str) """
    if isinstance(eml, str):
        eml = eml.encode('utf-8', 'surrogateescape')
    return _from_email_message(email.message_from_bytes(eml, policy=email.policy.default))


def parse_msg(filedata) -> dict:
    """ Outlook-.msg (Pfad oder bytes), braucht extract-msg """
    import extract_msg
    msg = extract_msg.Message(filedata)
    try:
        text = msg.body or ''
        if not text.strip() and getattr(msg, 'htmlBody', None):
            text = html_to_text(msg.htmlBody.decode('utf-8', 'replace'))
        return {'message_id': str(getattr(msg, 'messageId', '') or ''), 'sender': str(msg.sender or ''),
                'subject': str(msg.subject or ''), 'date': str(msg.date or ''), 'text': text}
    finally:
        msg.close()


def load_mail(data: dict) -> dict:
    """
      Mail aus einem Job: eml (RFC822), msg (Outlook) oder text/html (+ optional subject, sender/from, date, message_id)
      :return: {message_id, sender, subject, date, text, folder}
    """
    if data.get('eml'):
        mail = parse_eml(data['eml'])
    elif data.get('msg'):
        mail = parse_msg(data['msg'])
    else:
        text = str(data.get('html') or data.get('text') or '')
        if data.get('html') or looks_like_html(text):
            text = html_to_text(text)
        mail = {'message_id': '', 'sender': '', 'subject': '', 'date': '', 'text': text}
    for key, alias in (('message_id', None), ('sender', 'from'), ('subject', None), ('date', None)):
        value = data.get(key) or (data.get(alias) if alias else None)
        if value and not mail.get(key):
            mail[key] = str(value)
    mail['folder'] = str(data.get('folder') or '')
    mail['text'] = cut_text(mail['text'])
    return mail


def cut_text(text: str) -> str:
    """ zitierte Antworten und lange Fußzeilen (AGB, Widerruf) abschneiden """
    m = re.search(r'\n(?:-{2,}\s*Original|Am .{5,80} schrieb|On .{5,80} wrote|>{1,2} )', text)
    if m and m.start() > 200:
        text = text[:m.start()]
    m = re.search(r'\n[^\n]{0,20}(?:Widerrufsbelehrung|Widerrufsrecht|Allgemeine Geschäftsbedingungen)', text)
    if m and m.start() > 500:
        text = text[:m.start()]
    return text.strip()


# --- Vorfilter ------------------------------------------------------------------------------------------------------

ORDER_WORDS = re.compile(
    r'bestell|auftrag|order|kauf|rechnung|invoice|warenkorb|artikel|menge|stückpreis|zwischensumme|gesamtbetrag|'
    r'gesamtsumme|subtotal|receipt|quittung|buchung', re.I)


def prefilter(mail: dict):
    """ ohne LLM: nur Mails mit Betrag und Bestell-Stichwort sind Kandidaten. :return: (bool, Grund) """
    text = (mail.get('subject') or '') + '\n' + (mail.get('text') or '')
    if len((mail.get('text') or '').strip()) < 20:
        return False, 'kein Text'
    if not AMOUNT_RE.search(text):
        return False, 'kein Betrag mit Währung'
    if not ORDER_WORDS.search(text):
        return False, 'kein Bestell-Stichwort'
    return True, ''


# --- Prompt und Auswertung ------------------------------------------------------------------------------------------

SCHEMA = {
    "is_order": True,
    "kind": "order",
    "shop": "Name des Shops/Händlers",
    "order_id": "Bestellnummer",
    "order_date": "YYYY-MM-DD",
    "currency": "EUR",
    "items": [{"title": "Produktname", "sku": "Artikelnummer oder null", "quantity": 1,
               "unit_price": 9.99, "total": 9.99}],
    "shipping": 4.90,
    "discount": 0,
    "total": 14.89,
}

KINDS = ('order', 'invoice', 'shipping', 'cancellation', 'return', 'newsletter', 'other')
ORDER_KINDS = ('order', 'invoice')

PROMPT = """Prüfe die folgende E-Mail. Ist es eine Bestellbestätigung (oder Rechnung) mit bestellten Produkten?
Extrahiere dann die Bestellung. Antworte nur mit JSON in genau diesem Format, ohne weiteren Text:
{schema}

Regeln:
- kind: "order" für Bestellbestätigungen, "invoice" für Rechnungen, "shipping" für Versand-/Lieferbenachrichtigungen,
  "cancellation" für Stornierungen, "return" für Rücksendungen, "newsletter" für Werbung, sonst "other".
- is_order: true nur bei kind "order" oder "invoice" und wenn bestellte Produkte aufgeführt sind, sonst false und items leer.
- items: jede bestellte Position einzeln, keine Versandkosten, Gutscheine oder Summen als Position.
- Zahlen als JSON-Zahl mit Punkt (12.90), ohne Währungszeichen. quantity ist die Stückzahl.
- Unbekannte Werte: null. Nichts erfinden.

Betreff: {subject}
Von: {sender}

{text}
"""


def build_prompt(mail: dict, max_chars: int = 12000) -> str:
    text = mail.get('text') or ''
    if len(text) > max_chars:
        text = text[:max_chars] + '\n[...]'
    return PROMPT.replace('{schema}', json.dumps(SCHEMA, ensure_ascii=False, indent=1)) \
        .replace('{subject}', mail.get('subject') or '').replace('{sender}', mail.get('sender') or '') \
        .replace('{text}', text)


def extract_json(text: str):
    """ JSON-Objekt aus der Modell-Antwort (auch in ```json ... ``` Blöcken, mit <think>-Teil) """
    text = re.sub(r'<think>.*?</think>', '', str(text or ''), flags=re.S).strip()
    text = re.sub(r'^```(?:json)?', '', text).strip()
    text = re.sub(r'```$', '', text).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except ValueError:
            return json.loads(re.sub(r',\s*([}\]])', r'\1', candidate))
    raise ValueError('keine JSON-Antwort: ' + text[:200])


def parse_number(value):
    """ 12.9, '12,90 €', '1.234,56', '1,234.56', 'EUR 3' -> float, sonst None """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r'[^\d,.\-]', '', str(value))
    if not re.search(r'\d', s):
        return None
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        s = s.replace('.', '').replace(',', '.') if s.count(',') == 1 else s.replace(',', '')
    elif s.count('.') > 1 or re.fullmatch(r'-?\d{1,3}\.\d{3}', s):
        s = s.replace('.', '')  # Tausenderpunkte
    try:
        return float(s)
    except ValueError:
        return None


def _str(value):
    if value is None:
        return ''
    s = str(value).strip()
    return '' if s.lower() in ('null', 'none', 'n/a', '-') else s


def normalize_order(data: dict) -> dict:
    items = []
    for item in data.get('items') or []:
        if not isinstance(item, dict):
            continue
        title = _str(item.get('title') or item.get('name') or item.get('product'))
        if not title:
            continue
        qty = parse_number(item.get('quantity', item.get('amount')))
        qty = qty if qty and qty > 0 else 1.0
        unit = parse_number(item.get('unit_price', item.get('price')))
        total = parse_number(item.get('total'))
        if total is None and unit is not None:
            total = round(unit * qty, 2)
        if unit is None and total is not None:
            unit = round(total / qty, 2)
        items.append({'title': title, 'sku': _str(item.get('sku')), 'quantity': int(qty) if qty == int(qty) else qty,
                      'unit_price': unit, 'total': total})
    kind = _str(data.get('kind')).lower() or 'other'
    is_order = data.get('is_order')
    if isinstance(is_order, str):
        is_order = is_order.strip().lower() in ('true', '1', 'yes', 'ja')
    return {
        'is_order': bool(is_order) if is_order is not None else kind in ORDER_KINDS,
        'kind': kind if kind in KINDS else 'other',
        'shop': _str(data.get('shop') or data.get('retailer')),
        'order_id': _str(data.get('order_id')),
        'order_date': _str(data.get('order_date')),
        'currency': _str(data.get('currency')).upper() or 'EUR',
        'items': items,
        'shipping': parse_number(data.get('shipping')),
        'discount': abs(parse_number(data.get('discount')) or 0.0),
        'total': parse_number(data.get('total')),
    }


def check_order(order: dict) -> list:
    """ Plausibilität: Summe der Positionen + Versand - Rabatt == Gesamtbetrag. :return: Warnungen """
    warnings = []
    if any(i['total'] is None for i in order['items']):
        warnings.append('Positionen ohne Preis')
    if order['total'] is None:
        warnings.append('kein Gesamtbetrag')
    elif order['items'] and not any(i['total'] is None for i in order['items']):
        expected = sum(i['total'] for i in order['items']) + (order['shipping'] or 0) - (order['discount'] or 0)
        if abs(expected - order['total']) > max(0.05, 0.01 * abs(order['total'])):
            warnings.append(f"Summe der Positionen {expected:.2f} passt nicht zum Gesamtbetrag {order['total']:.2f}")
    if not order['order_id']:
        warnings.append('keine Bestellnummer')
    return warnings


def evaluate(response: str) -> dict:
    """
      LLM-Antwort auswerten.
      :return: {decision: order|uncertain|skip, reason, kind, order, warnings}
    """
    try:
        data = extract_json(response)
        if not isinstance(data, dict):
            raise ValueError('JSON ist kein Objekt')
    except ValueError as e:
        return {'decision': 'skip', 'reason': 'ungültige Antwort: ' + str(e)[:150], 'kind': 'error', 'order': None,
                'warnings': []}
    order = normalize_order(data)
    if not order['is_order'] or order['kind'] not in ORDER_KINDS:
        return {'decision': 'skip', 'reason': f"keine Bestellung ({order['kind']})", 'kind': order['kind'],
                'order': order, 'warnings': []}
    if not order['items']:
        return {'decision': 'skip', 'reason': 'Bestellung ohne Positionen', 'kind': order['kind'], 'order': order,
                'warnings': []}
    warnings = check_order(order)
    sum_wrong = any(w.startswith('Summe') for w in warnings)
    return {'decision': 'uncertain' if sum_wrong else 'order', 'reason': '', 'kind': order['kind'], 'order': order,
            'warnings': warnings}


# --- Persistenz -----------------------------------------------------------------------------------------------------

class MailOrderTable(NwebTable):
    """ Bestellungen aus Mails, status: order (geprüft) oder uncertain (Summen passen nicht) """
    TABLE = 'mail_orders'
    SCHEMA = {
        'guid': 'varchar(64)',
        'message_id': 'varchar(512) DEFAULT NULL',
        'folder': 'varchar(255) DEFAULT NULL',
        'sender': 'varchar(512) DEFAULT NULL',
        'subject': 'text',
        'mail_date': 'varchar(64) DEFAULT NULL',
        'status': 'varchar(32) DEFAULT NULL',
        'kind': 'varchar(32) DEFAULT NULL',
        'shop': 'varchar(255) DEFAULT NULL',
        'order_id': 'varchar(255) DEFAULT NULL',
        'order_date': 'varchar(32) DEFAULT NULL',
        'currency': 'varchar(8) DEFAULT NULL',
        'shipping': 'REAL DEFAULT NULL',
        'discount': 'REAL DEFAULT NULL',
        'total': 'REAL DEFAULT NULL',
        'item_count': 'INTEGER DEFAULT NULL',
        'warnings': 'text',
        'prompt_guid': 'varchar(64) DEFAULT NULL',
        'created': 'REAL DEFAULT NULL',
    }
    JSON_COLS = ('warnings',)


class OrderPostenTable(NwebTable):
    """ Positionen einer Bestellung (order_guid -> mail_orders.guid) """
    TABLE = 'order_posten'
    SCHEMA = {
        'guid': 'varchar(64)',
        'order_guid': 'varchar(64) DEFAULT NULL',
        'pos': 'INTEGER DEFAULT NULL',
        'title': 'text',
        'sku': 'varchar(255) DEFAULT NULL',
        'quantity': 'REAL DEFAULT NULL',
        'unit_price': 'REAL DEFAULT NULL',
        'total': 'REAL DEFAULT NULL',
        'created': 'REAL DEFAULT NULL',
    }
    ORDER = 'pos'


# --- Runner ---------------------------------------------------------------------------------------------------------

class OrderExtractor(r.BaseJobExecutor):
    """
      Extrahiert Bestellungen mit Positionen aus Bestellbestätigungs-Mails.

        {"type": "order_extract", "op": "submit", "text": "<Mailtext oder HTML>", "subject": "..."}
        {"type": "order_extract", "op": "submit", "eml": "<RFC822>"}

      Ist der LlmQueue-Runner (queue) vorhanden, laufen die LLM-Aufrufe darüber (tags mailextract, order),
      mit batch=1 werden sie gesammelt und erst mit run_batch verarbeitet. Sonst direkt über den LLM-Runner.
      Mails ohne Bestellung werden übersprungen und nur im RAM gelistet.
    """

    TAGS = [r.TAG.TEXT_EXTRACTOR, 'mail', 'llm']

    MODULES = ['extract-msg']

    def __init__(self, type='order_extract', llm='lmstudio', queue='llm_queue', batch=False,
                 tags='mailextract,order', prefilter=True, max_chars=12000, max_skipped=500):
        """
        :param llm: type des LLM-Runners
        :param queue: type des LlmQueue-Runners, leer: LLM direkt aufrufen
        :param batch: Prompts in der Queue liegen lassen (hold), bis run_batch aufgerufen wird
        :param tags: Tags der LLM-Aufrufe in der Queue, der letzte ist der Stapel-Tag
        :param prefilter: Mails ohne Betrag oder Bestell-Stichwort ohne LLM überspringen
        :param max_skipped: so viele übersprungene Mails im RAM behalten
        """
        super().__init__(type)
        self.llm = llm
        self.queue = queue
        self.batch = to_bool(batch)
        self.tags = parse_tags(tags) or ['mailextract', 'order']
        self.prefilter = to_bool(prefilter)
        self.max_chars = int(max_chars)
        self.skipped = collections.deque(maxlen=int(max_skipped))
        self.pending = {}
        self.lock = threading.RLock()
        self.tables = None
        self.define_vars('llm', 'queue', 'batch', 'prefilter')

    @property
    def batch_tag(self) -> str:
        return self.tags[-1]

    # --- Persistenz -----------------------------------------------------------------------------

    def orders(self) -> MailOrderTable:
        return self._tables()[0]

    def posten(self) -> OrderPostenTable:
        return self._tables()[1]

    def _tables(self):
        with self.lock:
            if self.tables is None:
                nw = nweb_store_of(self)
                try:
                    self.tables = (MailOrderTable(nw), OrderPostenTable(nw))
                except Exception as e:
                    self.error("mail_orders nicht verfügbar: " + str(e))
                    self.tables = (MailOrderTable(None), OrderPostenTable(None))
            return self.tables

    # --- Ablauf ---------------------------------------------------------------------------------

    def run_job(self, job: dict) -> dict:
        root = self.getRoot()
        if root is None or root is self:
            raise Exception('kein Dispatcher für ' + str(job.get('type')))
        return root.execute(dict(job))

    def skip(self, mail: dict, reason: str, kind: str = '') -> dict:
        entry = {'time': time.time(), 'message_id': mail.get('message_id', ''), 'sender': mail.get('sender', ''),
                 'subject': mail.get('subject', ''), 'date': mail.get('date', ''), 'folder': mail.get('folder', ''),
                 'reason': reason, 'kind': kind}
        with self.lock:
            self.skipped.appendleft(entry)
        return entry

    def is_known(self, message_id: str) -> bool:
        if not message_id:
            return False
        with self.lock:
            if any(m.get('message_id') == message_id for m in self.pending.values()):
                return True
            if any(m.get('message_id') == message_id for m in self.skipped):
                return True
        return bool(self.orders().find({'message_id': message_id}, limit=1))

    @staticmethod
    def mail_meta(mail: dict) -> dict:
        """ Kopfdaten für on_done (ohne Text) """
        return {k: mail.get(k, '') for k in ('mail_guid', 'message_id', 'sender', 'subject', 'date', 'folder')}

    @op("Mail einreihen: text/html, eml (RFC822) oder msg (Outlook), optional subject, sender, date, message_id. "
        "Ohne Bestellung wird sie übersprungen",
        result={'success': 'bool', 'status': 'str', 'guid': 'str', 'reason': 'str'})
    def submit(self, text: str = '', html: str = '', eml: str = '', msg: str = '', subject: str = '',
               sender: str = '', date: str = '', message_id: str = '', folder: str = '', llm: str = '',
               batch: str = ''):
        mail = load_mail(dict(text=text, html=html, eml=eml, msg=msg, subject=subject, sender=sender, date=date,
                              message_id=message_id, folder=folder))
        return self.submit_mail(mail, llm=llm or None, batch=to_bool(batch) if batch != '' else None)

    def submit_mail(self, mail: dict, llm: str = None, batch: bool = None) -> dict:
        if self.is_known(mail.get('message_id')):
            return self.success('schon bekannt', status='known', guid='', reason='Message-ID schon verarbeitet')
        if self.prefilter:
            ok, reason = prefilter(mail)
            if not ok:
                self.skip(mail, reason, 'prefilter')
                return self.success('übersprungen', status='skipped', guid='', reason=reason)
        llm = llm or self.llm
        batch = self.batch if batch is None else batch
        mail['mail_guid'] = u.guid()
        prompt = build_prompt(mail, self.max_chars)
        meta = self.mail_meta(mail)
        if self.queue:
            try:
                res = self.run_job({'type': self.queue, 'op': 'prompt', 'llm': llm, 'prompt': prompt,
                                    'tags': self.tags, 'hold': batch,
                                    'on_done': {'type': self.type, 'op': 'store', 'mail': meta,
                                                'response': '#response'}})
            except Exception as e:
                res = {'success': False, 'message': str(e)}
            if isinstance(res, dict) and res.get('success'):
                meta['prompt_guid'] = res.get('guid', '')
                meta['queued'] = time.time()
                with self.lock:
                    self.pending[mail['mail_guid']] = meta
                return self.success('eingereiht', status=res.get('status', 'queued'), guid=mail['mail_guid'],
                                    reason='')
            self.info("LlmQueue nicht nutzbar, rufe LLM direkt auf: " + str((res or {}).get('message', '')))
        try:
            response = self.ask_llm(llm, prompt)
        except Exception as e:
            return self.fail('LLM: ' + str(e), status='error', guid='', reason=str(e))
        return self.store(meta, response)

    def ask_llm(self, llm: str, prompt: str) -> str:
        res = self.run_job({'type': llm, 'prompt': prompt})
        if not isinstance(res, dict) or 'response' not in res:
            msg = res.get('message', '') if isinstance(res, dict) else str(res)
            raise Exception('keine Antwort vom LLM ' + str(llm) + (': ' + str(msg) if msg else ''))
        return str(res['response'])

    @op("Ziel von on_done: LLM-Antwort auswerten und als Bestellung speichern oder überspringen",
        result={'success': 'bool', 'status': 'str', 'guid': 'str', 'reason': 'str'})
    def store(self, mail: dict, response: str):
        mail = mail if isinstance(mail, dict) else {}
        with self.lock:
            pending = self.pending.pop(mail.get('mail_guid', ''), None)
        if pending and not mail.get('prompt_guid'):
            mail['prompt_guid'] = pending.get('prompt_guid', '')
        res = evaluate(response)
        if res['decision'] == 'skip':
            self.skip(mail, res['reason'], res['kind'])
            return self.success('übersprungen', status='skipped', guid='', reason=res['reason'])
        order = res['order']
        if order['order_id']:
            same = self.orders().find({'shop': order['shop'], 'order_id': order['order_id']}, limit=1)
            if same:
                reason = 'Bestellung schon vorhanden: ' + (order['shop'] + ' ' + order['order_id']).strip()
                self.skip(mail, reason, 'duplicate')
                return self.success('übersprungen', status='skipped', guid=same[0]['guid'], reason=reason)
        guid = mail.get('mail_guid') or u.guid()
        now = time.time()
        self.orders().insert({
            'guid': guid, 'message_id': mail.get('message_id', ''), 'folder': mail.get('folder', ''),
            'sender': mail.get('sender', ''), 'subject': mail.get('subject', ''), 'mail_date': mail.get('date', ''),
            'status': res['decision'], 'kind': order['kind'], 'shop': order['shop'], 'order_id': order['order_id'],
            'order_date': order['order_date'], 'currency': order['currency'], 'shipping': order['shipping'],
            'discount': order['discount'], 'total': order['total'], 'item_count': len(order['items']),
            'warnings': res['warnings'], 'prompt_guid': mail.get('prompt_guid', ''), 'created': now})
        for i, item in enumerate(order['items'], 1):
            self.posten().insert({'guid': u.guid(), 'order_guid': guid, 'pos': i, 'title': item['title'],
                                  'sku': item['sku'], 'quantity': item['quantity'], 'unit_price': item['unit_price'],
                                  'total': item['total'], 'created': now})
        self.emit_event('order', guid=guid, status=res['decision'], shop=order['shop'], order_id=order['order_id'],
                        total=order['total'])
        return self.success('gespeichert', status=res['decision'], guid=guid, reason='')

    @op("Direkt (ohne Queue) auswerten, ohne zu speichern: Ergebnis des LLM für eine Mail",
        result={'success': 'bool', 'decision': 'str', 'reason': 'str', 'order': 'dict', 'warnings': 'list'})
    def extract(self, text: str = '', html: str = '', eml: str = '', subject: str = '', sender: str = '',
                llm: str = ''):
        mail = load_mail(dict(text=text, html=html, eml=eml, subject=subject, sender=sender))
        if self.prefilter:
            ok, reason = prefilter(mail)
            if not ok:
                return self.success('übersprungen', decision='skip', reason=reason, order={}, warnings=[])
        try:
            response = self.ask_llm(llm or self.llm, build_prompt(mail, self.max_chars))
        except Exception as e:
            return self.fail(str(e), decision='error', reason=str(e), order={}, warnings=[])
        res = evaluate(response)
        return self.success('ok', decision=res['decision'], reason=res['reason'], order=res['order'] or {},
                            warnings=res['warnings'], response=response)

    @op("Wartende LLM-Aufrufe (Stapel-Tag) jetzt verarbeiten", result={'success': 'bool', 'count': 'int'})
    def run_batch(self, background: bool = True):
        if not self.queue:
            return self.fail('keine Queue konfiguriert', count=0)
        return self.run_job({'type': self.queue, 'op': 'run_batch', 'tag': self.batch_tag, 'background': background})

    @op("Gespeicherte Bestellungen, status: order oder uncertain, leer: alle",
        result={'success': 'bool', 'orders': 'list'})
    def list_orders(self, status: str = '', limit: int = 100):
        return self.success('ok', orders=self.orders().find({'status': status or None}, limit=limit))

    @op("Bestellung mit Positionen", result={'success': 'bool', 'order': 'dict', 'items': 'list'})
    def get(self, guid: str):
        order = self.orders().get(guid)
        if order is None:
            return self.fail('unbekannte guid')
        return self.success('ok', order=order, items=self.posten().find({'order_guid': guid}, limit=0,
                                                                        oldest_first=True))

    @op("Bestellung mit Positionen löschen", result={'success': 'bool'})
    def delete(self, guid: str):
        self.posten().delete_where('order_guid', guid)
        self.orders().delete(guid)
        return self.success('gelöscht')

    @op("Übersprungene Mails (nur im RAM)", result={'success': 'bool', 'skipped': 'list'})
    def list_skipped(self):
        with self.lock:
            return self.success('ok', skipped=list(self.skipped))

    @op("Liste der übersprungenen Mails leeren", result={'success': 'bool'})
    def clear_skipped(self):
        with self.lock:
            self.skipped.clear()
        return self.success('geleert')

    # --- Web ------------------------------------------------------------------------------------

    def nav(self, p: b.Page, active=None):
        with self.lock:
            pending, skipped = len(self.pending), len(self.skipped)
        p.nav((f"Bestellungen ({self.orders().count()})", self.link(self.part_index), "Gespeicherte Bestellungen"),
              (f"Ausstehend ({pending})", self.link(self.part_pending), "Mails, deren LLM-Aufruf noch aussteht"),
              (f"Übersprungen ({skipped})", self.link(self.part_skipped),
               "Mails ohne Bestellung, nur im RAM bis zum Neustart"),
              ("Testen", self.link(self.part_test), "Mail einfügen und auswerten"),
              active=self.link(active) if active is not None else None)

    def part_index(self, p: b.Page, params={}):
        self.nav(p, self.part_index)
        orders = self.orders()
        p.prop("Gespeichert", f'ja ({MailOrderTable.TABLE}, {OrderPostenTable.TABLE})' if orders.persistent else
               'nein, keine nweb-Persistence, nur bis zum Neustart')
        p.prop("LLM", f"{self.llm} über {self.queue}" if self.queue else str(self.llm))
        rows = orders.find(limit=200)
        if not rows:
            p.div('Keine Bestellungen.')
            return
        html = '<table class="table"><tr><th>Datum</th><th>Shop</th><th>Bestellnummer</th><th>Positionen</th>' \
               '<th>Summe</th><th>Status</th><th>Betreff</th></tr>'
        for row in rows:
            total = f"{row['total']:.2f} {row.get('currency') or ''}" if row.get('total') is not None else ''
            warn = '; '.join(row.get('warnings') or [])
            html += f'<tr title="{esc(warn)}"><td>{esc(str(row.get("order_date") or row.get("mail_date") or ""))}</td>' \
                    f'<td>{esc(str(row.get("shop") or ""))}</td>' \
                    f'<td><a href="{esc(self.link(self.part_order, guid=row["guid"]))}">' \
                    f'{esc(str(row.get("order_id") or "?"))}</a></td><td>{row.get("item_count") or 0}</td>' \
                    f'<td>{esc(total)}</td><td>{esc(str(row.get("status") or ""))}</td>' \
                    f'<td>{esc(str(row.get("subject") or "")[:120])}</td></tr>'
        p(html + '</table>')

    def part_order(self, p: b.Page, params={}):
        self.nav(p)
        guid = params.get('guid', '')
        order = self.orders().get(guid)
        if order is None:
            p.div('Unbekannte Bestellung.')
            return
        p.h3(f"{order.get('shop') or ''} {order.get('order_id') or ''}".strip() or 'Bestellung')
        for key, title in (('order_date', 'Bestelldatum'), ('status', 'Status'), ('sender', 'Von'),
                           ('subject', 'Betreff'), ('mail_date', 'Mail vom'), ('folder', 'Ordner')):
            if order.get(key):
                p.prop(title, esc(str(order[key])))
        for w in order.get('warnings') or []:
            p.div('⚠ ' + esc(w))
        cur = order.get('currency') or ''
        money = lambda v: f'{v:.2f} {cur}' if v is not None else ''
        html = '<table class="table"><tr><th>Pos</th><th>Artikel</th><th>Art.-Nr.</th><th>Menge</th>' \
               '<th>Einzelpreis</th><th>Summe</th></tr>'
        for item in self.posten().find({'order_guid': guid}, limit=0, oldest_first=True):
            html += f'<tr><td>{item.get("pos")}</td><td>{esc(str(item.get("title") or ""))}</td>' \
                    f'<td>{esc(str(item.get("sku") or ""))}</td><td>{item.get("quantity") or ""}</td>' \
                    f'<td>{esc(money(item.get("unit_price")))}</td><td>{esc(money(item.get("total")))}</td></tr>'
        for key, title in (('shipping', 'Versand'), ('discount', 'Rabatt'), ('total', 'Gesamt')):
            if order.get(key):
                html += f'<tr><td></td><td colspan="4"><b>{title}</b></td><td>{esc(money(order[key]))}</td></tr>'
        p(html + '</table>')
        p(self.action_btn_parametric("Löschen", {'type': self.type, 'op': 'delete', 'guid': guid}))

    def part_pending(self, p: b.Page, params={}):
        self.nav(p, self.part_pending)
        if self.queue:
            p(self.action_btn_parametric(f"Stapel »{self.batch_tag}« jetzt verarbeiten",
                                         {'type': self.type, 'op': 'run_batch'}))
        with self.lock:
            rows = sorted(self.pending.values(), key=lambda m: m.get('queued') or 0)
        if not rows:
            p.div('Keine ausstehenden Mails.')
            return
        html = '<table class="table"><tr><th>Eingereiht</th><th>Von</th><th>Betreff</th><th>Prompt</th></tr>'
        for m in rows:
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m.get('queued') or 0))
            html += f'<tr><td>{t}</td><td>{esc(str(m.get("sender") or ""))}</td>' \
                    f'<td>{esc(str(m.get("subject") or ""))}</td><td>{esc(str(m.get("prompt_guid") or ""))}</td></tr>'
        p(html + '</table>')

    def part_skipped(self, p: b.Page, params={}):
        self.nav(p, self.part_skipped)
        p.div('Mails ohne Bestellung, nur im RAM (bis zum Neustart).')
        with self.lock:
            rows = list(self.skipped)
        if not rows:
            p.div('Keine übersprungenen Mails.')
            return
        p(self.action_btn_parametric("Liste leeren", {'type': self.type, 'op': 'clear_skipped'}))
        html = '<table class="table"><tr><th>Zeit</th><th>Von</th><th>Betreff</th><th>Art</th><th>Grund</th></tr>'
        for m in rows:
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m['time']))
            html += f'<tr><td>{t}</td><td>{esc(str(m.get("sender") or ""))}</td>' \
                    f'<td>{esc(str(m.get("subject") or ""))}</td><td>{esc(str(m.get("kind") or ""))}</td>' \
                    f'<td>{esc(str(m.get("reason") or ""))}</td></tr>'
        p(html + '</table>')

    def part_test(self, p: b.Page, params={}):
        self.nav(p, self.part_test)
        p('<input id="subject" placeholder="Betreff" style="width:100%" />')
        p('<textarea id="text" rows="14" style="width:100%" placeholder="Mailtext oder HTML"></textarea>')
        p(self.action_btn_parametric("Auswerten (ohne Speichern)",
                                     {'type': self.type, 'op': 'extract', 'text': '#text', 'subject': '#subject'}))
        p(self.action_btn_parametric("Einreihen",
                                     {'type': self.type, 'op': 'submit', 'text': '#text', 'subject': '#subject'}))
        p.pre('', id='result')


def to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on', 'ja')
    return bool(value)
