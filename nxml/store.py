#
# Project: nxml
#
# Kleine Tabellen in der nweb-Persistence (root.nweb mit dml/select/insert, sqlite oder MySQL),
# ohne Persistence nur im Speicher.
#
import json
import threading
from typing import Optional


def nweb_store_of(runner):
    """ nweb-Persistence (root.nweb) mit dml/select/insert oder None """
    root = runner.getRoot()
    nw = getattr(root, 'nweb', None) if root is not None else None
    return nw if nw is not None and all(hasattr(nw, m) for m in ('dml', 'select', 'insert')) else None


def parse_tags(tags) -> list:
    """ Tags als Liste: None, ['a', 'b'], '["a", "b"]' oder 'a, b' """
    if tags is None or tags == '':
        return []
    if isinstance(tags, str):
        s = tags.strip()
        if s.startswith('['):
            try:
                tags = json.loads(s)
            except ValueError:
                tags = s.strip('[]').split(',')
        else:
            tags = s.split(',')
    if not isinstance(tags, (list, tuple, set)):
        tags = [tags]
    res = []
    for t in tags:
        t = str(t).strip().strip('"\'').strip()
        if t and t not in res:
            res.append(t)
    return res


class NwebTable:
    """
      Tabelle mit Primärschlüssel KEY. SCHEMA: Spalte -> SQL-Typ, die Reihenfolge ergibt COLS.
      JSON_COLS werden als JSON gespeichert. Fehlende Spalten werden bei bestehenden Tabellen ergänzt.
    """

    TABLE = ''
    KEY = 'guid'
    SCHEMA = {}
    JSON_COLS = ()
    ORDER = 'created'

    def __init__(self, nw=None):
        self.nw = nw
        self.memory = {}
        self.lock = threading.RLock()
        if nw is not None:
            self.create()

    @property
    def COLS(self) -> tuple:
        return tuple(self.SCHEMA.keys())

    @property
    def persistent(self) -> bool:
        return self.nw is not None

    def create(self):
        cols = ',\n'.join(f'{name} {typ}' + (' NOT NULL PRIMARY KEY' if name == self.KEY else '')
                          for name, typ in self.SCHEMA.items())
        self.nw.dml(f"CREATE TABLE IF NOT EXISTS {self.TABLE} (\n{cols}\n)")
        for name, typ in self.SCHEMA.items():
            if name == self.KEY:
                continue
            try:
                self.nw.select(f"SELECT {name} FROM {self.TABLE} LIMIT 1", [])
            except Exception:
                # Spalte fehlt in einer älteren Tabelle
                self.nw.dml(f"ALTER TABLE {self.TABLE} ADD COLUMN {name} {typ}")

    def _encode(self, row: dict) -> dict:
        return {k: (json.dumps(v, default=str) if k in self.JSON_COLS and v is not None else v)
                for k, v in row.items() if k in self.SCHEMA}

    def _decode(self, values) -> dict:
        row = dict(zip(self.COLS, values))
        for k in self.JSON_COLS:
            try:
                row[k] = json.loads(row[k]) if row[k] else None
            except (ValueError, TypeError):
                pass
        return row

    def insert(self, row: dict):
        with self.lock:
            if self.nw is None:
                self.memory[row[self.KEY]] = {c: row.get(c) for c in self.COLS}
            else:
                self.nw.insert(self.TABLE, self._encode(row))

    def update(self, key, **values):
        with self.lock:
            if self.nw is None:
                if key in self.memory:
                    self.memory[key].update({k: v for k, v in values.items() if k in self.SCHEMA})
                return
            values = self._encode(values)
            if values:
                sets = ', '.join(f'{k} = ?' for k in values)
                self.nw.dml(f"UPDATE {self.TABLE} SET {sets} WHERE {self.KEY} = ?", [*values.values(), key])

    def get(self, key) -> Optional[dict]:
        with self.lock:
            if self.nw is None:
                row = self.memory.get(key)
                return dict(row) if row is not None else None
            rows = self.nw.select(f"SELECT {', '.join(self.COLS)} FROM {self.TABLE} WHERE {self.KEY} = ?", [key])
            return self._decode(rows[0]) if rows else None

    def find(self, where: dict = None, limit=500, oldest_first=False, tag: str = None, tag_col='tags') -> list:
        """
          where: Spalte -> Wert oder Tupel von Werten (IN), None-Werte werden ignoriert
          tag: nur Zeilen, deren JSON-Liste tag_col den Tag enthält
          :return: neueste zuerst (oldest_first: älteste zuerst)
        """
        where = {k: v for k, v in (where or {}).items() if v is not None}
        with self.lock:
            if self.nw is None:
                res = []
                for row in self.memory.values():
                    if all(row.get(k) in v if isinstance(v, (tuple, list, set)) else row.get(k) == v
                           for k, v in where.items()):
                        if tag is None or tag in (row.get(tag_col) or []):
                            res.append(dict(row))
                res.sort(key=lambda row: row.get(self.ORDER) or 0, reverse=not oldest_first)
                return res[:limit] if limit else res
            sql_where, params = [], []
            for k, v in where.items():
                if isinstance(v, (tuple, list, set)):
                    v = list(v)
                    sql_where.append(f"{k} IN ({', '.join('?' for _ in v)})")
                    params.extend(v)
                else:
                    sql_where.append(f'{k} = ?')
                    params.append(v)
            if tag is not None:
                sql_where.append(f'{tag_col} LIKE ?')
                params.append('%' + json.dumps(tag) + '%')
            sql = f"SELECT {', '.join(self.COLS)} FROM {self.TABLE}"
            sql += (' WHERE ' + ' AND '.join(sql_where)) if sql_where else ''
            sql += f" ORDER BY {self.ORDER} {'ASC' if oldest_first else 'DESC'}"
            if limit and tag is None:
                sql += f' LIMIT {int(limit)}'
            res = [self._decode(values) for values in self.nw.select(sql, params)]
            if tag is not None:
                # LIKE ist nur eine Vorauswahl
                res = [row for row in res if tag in (row.get(tag_col) or [])]
                res = res[:limit] if limit else res
            return res

    def count(self, where: dict = None) -> int:
        where = {k: v for k, v in (where or {}).items() if v is not None}
        with self.lock:
            if self.nw is None:
                return len(self.find(where, limit=0))
            sql_where, params = [], []
            for k, v in where.items():
                sql_where.append(f'{k} = ?')
                params.append(v)
            sql = f"SELECT COUNT(*) FROM {self.TABLE}" + ((' WHERE ' + ' AND '.join(sql_where)) if sql_where else '')
            rows = self.nw.select(sql, params)
            return int(rows[0][0]) if rows else 0

    def delete(self, key):
        with self.lock:
            if self.nw is None:
                self.memory.pop(key, None)
            else:
                self.nw.dml(f"DELETE FROM {self.TABLE} WHERE {self.KEY} = ?", [key])

    def delete_where(self, col, value):
        with self.lock:
            if self.nw is None:
                for k in [k for k, row in self.memory.items() if row.get(col) == value]:
                    self.memory.pop(k)
            else:
                self.nw.dml(f"DELETE FROM {self.TABLE} WHERE {col} = ?", [value])
