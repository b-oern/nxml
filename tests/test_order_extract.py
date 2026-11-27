"""
Unittests fuer nxml.order_extract (OrderExtractor) und die Stapelverarbeitung von nxml.llm.LlmQueue

    python -m unittest tests.test_order_extract
"""
import json
import sqlite3
import threading
import unittest

from nwebclient import runner as r

from nxml.llm import LlmQueue, QueuePromptTable
from nxml.order_extract import (OrderExtractor, load_mail, prefilter, parse_number, normalize_order, evaluate,
                                html_to_text, parse_eml)
from nxml.store import parse_tags


class SqliteNweb:
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


ORDER_JSON = {"is_order": True, "kind": "order", "shop": "Beispielshop", "order_id": "A-1001",
              "order_date": "2026-11-20", "currency": "EUR",
              "items": [{"title": "Kabel USB-C", "sku": "K1", "quantity": 2, "unit_price": "4,95", "total": None},
                        {"title": "Netzteil", "sku": None, "quantity": 1, "unit_price": 19.9, "total": 19.9}],
              "shipping": 3.9, "discount": 0, "total": "33,70"}

NO_ORDER_JSON = {"is_order": False, "kind": "shipping", "shop": "Beispielshop", "order_id": "A-1001", "items": []}


class FakeLlm(r.BaseJobExecutor):
    """ LLM mit fester Antwort, abhängig vom Prompt """

    def __init__(self, type='fake_llm'):
        super().__init__(type)
        self.calls = []

    def execute(self, data):
        if 'prompt' not in data:
            return super().execute(data)
        self.calls.append(data['prompt'])
        if 'Betreff: Versand' in data['prompt']:
            return {'success': True, 'response': json.dumps(NO_ORDER_JSON)}
        return {'success': True, 'response': '```json\n' + json.dumps(ORDER_JSON) + '\n```'}


def create_root(nw, *runners):
    dispatcher = r.LazyDispatcher()
    for runner in runners:
        dispatcher.add_runner(runner)
    root = r.JobRunner(dispatcher, {})
    root.nweb = nw
    for runner in runners:
        dispatcher.get_runner(runner.type)
    return root


ORDER_MAIL = """Vielen Dank für Ihre Bestellung A-1001!
Artikel | Menge | Preis
Kabel USB-C | 2 | 9,90 €
Netzteil | 1 | 19,90 €
Versand 3,90 €
Gesamtbetrag 33,70 €"""

SHIPPING_MAIL = "Ihre Bestellung A-1001 ist unterwegs. Versand 3,90 € wurde berechnet."


class HelperTest(unittest.TestCase):

    def test_parse_number(self):
        self.assertEqual(parse_number('12,90 €'), 12.9)
        self.assertEqual(parse_number('1.234,56'), 1234.56)
        self.assertEqual(parse_number('1,234.56'), 1234.56)
        self.assertEqual(parse_number('EUR 3'), 3.0)
        self.assertEqual(parse_number('1.234'), 1234.0)
        self.assertEqual(parse_number(4.5), 4.5)
        self.assertIsNone(parse_number('null'))

    def test_parse_tags(self):
        self.assertEqual(parse_tags('mailextract, order'), ['mailextract', 'order'])
        self.assertEqual(parse_tags('["a", "b"]'), ['a', 'b'])
        self.assertEqual(parse_tags(None), [])

    def test_html_table(self):
        text = html_to_text('<html><style>x{}</style><table><tr><td>Kabel</td><td>2</td><td>9,90 €</td></tr>'
                            '<tr><td>Summe</td><td></td><td>9,90 €</td></tr></table></html>')
        self.assertEqual(text, 'Kabel | 2 | 9,90 €\nSumme | 9,90 €')

    def test_load_mail_eml(self):
        eml = ("From: Shop <shop@example.org>\r\nSubject: Ihre Bestellung\r\nMessage-ID: <m1@example.org>\r\n"
               "Date: Fri, 20 Nov 2026 10:00:00 +0100\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
               "<p>Bestellung</p><table><tr><td>Kabel</td><td>9,90 &euro;</td></tr></table>")
        mail = load_mail({'eml': eml, 'folder': 'INBOX'})
        self.assertEqual(mail['message_id'], '<m1@example.org>')
        self.assertEqual(mail['subject'], 'Ihre Bestellung')
        self.assertIn('Kabel | 9,90 €', mail['text'])
        self.assertEqual(mail['folder'], 'INBOX')
        self.assertTrue(mail['date'].startswith('2026-11-20'))

    def test_prefilter(self):
        self.assertTrue(prefilter({'subject': '', 'text': ORDER_MAIL})[0])
        ok, reason = prefilter({'subject': 'Newsletter', 'text': 'Unsere neuen Angebote im Herbst, jetzt ansehen!'})
        self.assertFalse(ok)
        self.assertEqual(reason, 'kein Betrag mit Währung')

    def test_normalize_and_evaluate(self):
        order = normalize_order(ORDER_JSON)
        self.assertEqual(order['items'][0]['total'], 9.9)
        self.assertEqual(order['total'], 33.7)
        self.assertEqual(evaluate(json.dumps(ORDER_JSON))['decision'], 'order')
        self.assertEqual(evaluate(json.dumps(NO_ORDER_JSON))['decision'], 'skip')
        self.assertEqual(evaluate('kein json')['decision'], 'skip')
        wrong = dict(ORDER_JSON, total=50)
        res = evaluate(json.dumps(wrong))
        self.assertEqual(res['decision'], 'uncertain')
        self.assertTrue(res['warnings'])
        empty = dict(ORDER_JSON, items=[])
        self.assertEqual(evaluate(json.dumps(empty))['reason'], 'Bestellung ohne Positionen')


class OrderExtractorTest(unittest.TestCase):

    def setUp(self):
        self.nw = SqliteNweb()
        self.llm = FakeLlm()
        self.queue = LlmQueue(interval=0.05)
        self.ox = OrderExtractor(llm='fake_llm', batch=True)
        self.root = create_root(self.nw, self.llm, self.queue, self.ox)

    def test_batch_flow(self):
        res = self.ox.submit(text=ORDER_MAIL, subject='Bestellbestätigung', message_id='<m1>')
        self.assertTrue(res['success'], res)
        self.assertEqual(res['status'], 'held')
        self.ox.submit(text=SHIPPING_MAIL, subject='Versand', message_id='<m2>')
        self.ox.submit(text='Hallo, wie geht es dir? Liebe Grüße', subject='Privat', message_id='<m3>')
        # Vorfilter: ohne LLM übersprungen
        self.assertEqual(len(self.ox.skipped), 1)
        self.assertEqual(len(self.ox.pending), 2)
        self.assertEqual(self.llm.calls, [])
        # schon bekannt
        self.assertEqual(self.ox.submit(text=ORDER_MAIL, message_id='<m1>')['status'], 'known')

        rows = self.queue.prompts_table().rows(status=('held',), tag='order')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['tags'], ['mailextract', 'order'])
        self.assertEqual(self.queue.prompts_table().tag_counts(), {'mailextract': 2, 'order': 2})

        res = self.ox.run_batch(background=False)
        self.assertEqual((res['count'], res['done'], res['errors']), (2, 2, 0), res)
        self.assertEqual(len(self.llm.calls), 2)
        self.assertEqual(self.ox.pending, {})

        orders = self.ox.list_orders()['orders']
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual((order['shop'], order['order_id'], order['status'], order['total']),
                         ('Beispielshop', 'A-1001', 'order', 33.7))
        self.assertEqual(order['message_id'], '<m1>')
        items = self.ox.get(order['guid'])['items']
        self.assertEqual([(i['pos'], i['title'], i['quantity'], i['total']) for i in items],
                         [(1, 'Kabel USB-C', 2, 9.9), (2, 'Netzteil', 1, 19.9)])
        # Versandmail: übersprungen, nur im RAM
        reasons = [s['reason'] for s in self.ox.list_skipped()['skipped']]
        self.assertIn('keine Bestellung (shipping)', reasons)
        # in der nweb-Persistence gespeichert
        self.assertEqual(self.nw.select('SELECT COUNT(*) FROM order_posten')[0][0], 2)

    def test_duplicate_order(self):
        self.ox.batch = False
        self.ox.queue = ''
        self.assertEqual(self.ox.submit(text=ORDER_MAIL, message_id='<a>')['status'], 'order')
        res = self.ox.submit(text=ORDER_MAIL, message_id='<b>')
        self.assertEqual(res['status'], 'skipped')
        self.assertIn('schon vorhanden', res['reason'])
        self.assertEqual(len(self.ox.list_orders()['orders']), 1)

    def test_delete(self):
        self.ox.queue = ''
        guid = self.ox.submit(text=ORDER_MAIL)['guid']
        self.ox.delete(guid)
        self.assertEqual(self.ox.list_orders()['orders'], [])
        self.assertEqual(self.nw.select('SELECT COUNT(*) FROM order_posten')[0][0], 0)

    def test_pages(self):
        self.ox.queue = ''
        guid = self.ox.submit(text=ORDER_MAIL)['guid']
        self.ox.submit(text='kein Inhalt hier', subject='x')
        for page, params in (('index', {}), ('order', {'guid': guid}), ('pending', {}), ('skipped', {}),
                             ('test', {})):
            html = str(self.ox.page(dict(params, **{self.ox.type: page})))
            self.assertIn('Übersprungen (1)', html, page)
            self.assertIn('Bestellungen (1)', html, page)


class QueueMigrationTest(unittest.TestCase):

    def test_adds_tags_column(self):
        nw = SqliteNweb()
        nw.dml("CREATE TABLE queue_prompts (guid varchar(64) NOT NULL PRIMARY KEY, llm varchar(255), prompt text, "
               "on_done text, status varchar(32), created REAL, started REAL, done REAL, response text, message text)")
        nw.insert('queue_prompts', {'guid': 'old', 'llm': 'x', 'prompt': 'p', 'status': 'done', 'created': 1})
        table = QueuePromptTable(nw)
        self.assertIsNone(table.get('old')['tags'])
        table.insert({'guid': 'new', 'llm': 'x', 'prompt': 'p', 'status': 'held', 'created': 2, 'tags': ['order']})
        self.assertEqual([row['guid'] for row in table.rows(tag='order')], ['new'])

    def test_hold_needs_tags(self):
        q = LlmQueue()
        create_root(None, q)
        self.assertFalse(q.prompt('p', 'x', hold=True)['success'])


if __name__ == '__main__':
    unittest.main()
