# RagRunner

`nxml.rag:RagRunner` (Runner-Typ `rag`) ist Retrieval Augmented Generation für nxml:
Texte werden in Abschnitte (Chunks) zerlegt, als Embeddings gespeichert und semantisch durchsucht.
Ein Chat beantwortet Fragen mit einem LLM anhand der gefundenen Abschnitte.

```
index_document(text, document_data)
    -> chunk_text               Absätze, Sätze, Überlappung
    -> Embedding-Runner         z.B. text_embed, lmstudio (input_type=passage)
    -> Speicher                 memory | file | local_index | postgres | qdrant

query(text, data_filter)
    -> Embedding-Runner         (input_type=query)
    -> Speicher.search          Kosinus-Ähnlichkeit, Filter auf document_data
    -> Treffer                  bester zuerst

chat(question, llm)
    -> query -> Kontext [1], [2], ... -> LLM-Runner -> Antwort mit Quellen
```

## Schnellstart

```
pip install "nxml[rag-hugging] @ git+https://github.com/b-oern/nxml"
```

Konfiguration der Runner:

```
text_embed: nxml.analyse:TextEmbeddings     # args: model_name=BAAI/bge-m3
lmstudio:   nxml.llm:LMStudio                # LLM für den Chat
rag:        nxml.rag:RagRunner               # args: backend=local_index, path=data/rag, embedder=text_embed, llm=lmstudio
```

Dokument indizieren und suchen:

```json
{"type": "rag", "op": "index_document", "text": "Langer Text ...", "document_data": {"doc_id": "handbuch", "lang": "de"}}
{"type": "rag", "op": "query", "text": "Wie setze ich das Gerät zurück?", "data_filter": {"lang": "de"}, "limit": 5}
{"type": "rag", "op": "chat", "question": "Wie setze ich das Gerät zurück?", "llm": "lmstudio"}
```

Weboberfläche: `/pysys/runner?type=rag&rag=index`

Direkt in Python, ohne Dispatcher, mit eigener Embedding-Funktion:

```python
from nxml.rag import RagRunner

rag = RagRunner(backend='local_index', path='data/rag', embed_fn=my_embed)   # my_embed(texts) -> list[list[float]]
rag.index_document('Der Hund bellt. Die Katze schläft.', {'doc_id': 'tiere', 'lang': 'de'})
for hit in rag.query('Was macht die Katze?', {'lang': 'de'}, limit=3):
    print(round(hit['score'], 3), hit['doc_id'], hit['text'])
```

## Parameter

| Parameter | Standard | Bedeutung |
|---|---|---|
| `backend` | `memory` | Speicher: `memory`, `file`, `local_index`, `postgres`, `qdrant` (siehe [Speicher](#speicher)) |
| `embedder` | `text_embed` | `type` des Embedding-Runners (siehe [Embedder](#embedder)) |
| `embedding_model` | leer | wird als `model` an den Embedder gegeben (LM Studio: welches Embedding-Modell) |
| `batch_size` | `32` | Texte je Embedding-Aufruf |
| `chunk_size` | `1000` | maximale Zeichen je Chunk (mindestens 50) |
| `chunk_overlap` | `150` | Überlappung benachbarter Chunks in Zeichen (höchstens `chunk_size / 2`) |
| `llm` | leer | vorausgewähltes LLM für den Chat, leer: erster geladener LLM-Runner |
| `path` | `rag_index` | Verzeichnis für `file` und `local_index` |
| `nprobe` | `0` | `local_index`: durchsuchte Listen, 0 = automatisch |
| `min_train` | `4096` | `local_index`: ab so vielen Chunks wird der IVF-Index trainiert |
| `dsn` | `postgresql://localhost/rag` | `postgres`: Verbindung |
| `table` | `rag_chunks` | `postgres`: Tabelle |
| `url` | `http://localhost:6333` | `qdrant`: Server |
| `qdrant_path` | leer | `qdrant`: lokale Datenbank statt Server |
| `collection` | `rag` | `qdrant`: Collection |
| `api_key` | leer | `qdrant`: API-Key |
| `embed_fn` | – | nur Python: `embed_fn(texts[, input_type])` statt Embedding-Runner |
| `store` | – | nur Python: fertiger `VectorStore` statt `backend` |

## Operationen

Aufruf per Job `{"type": "rag", "op": "<name>", ...}`. Listen und dicts dürfen als JSON-String kommen
(z.B. aus Formularen). Jedes Ergebnis hat `success` und `message`.

### index_document

| Parameter | |
|---|---|
| `text` | der Text |
| `document_data` | Metadaten (dict), nach denen später gefiltert werden kann |

Ergebnis: `doc_id`, `chunks` (Anzahl).

- Die `doc_id` kommt aus `document_data["doc_id"]` oder `document_data["id"]`, sonst ist es der SHA1 des Textes.
- Wird dieselbe `doc_id` erneut indiziert, werden ihre alten Chunks ersetzt.
- `document_data` wird an jedem Chunk gespeichert (inklusive `doc_id`) und bei Treffern mitgeliefert.

### query

| Parameter | Standard | |
|---|---|---|
| `text` | | Suchtext |
| `data_filter` | leer | siehe [Filter](#filter) |
| `limit` | `5` | Anzahl Treffer |
| `min_score` | leer | nur Treffer mit mindestens diesem Score |

Ergebnis: `hits` und `context` (alle Treffertexte, getrennt durch `---`). Ein Treffer:

```json
{"id": "handbuch#3", "doc_id": "handbuch", "chunk_index": 3, "text": "...",
 "data": {"doc_id": "handbuch", "lang": "de"}, "score": 0.83}
```

`score` ist die Kosinus-Ähnlichkeit (1 = gleich). Wie hoch ein guter Wert ist, hängt vom Embedding-Modell ab,
`min_score` deshalb am eigenen Material ausprobieren.

### chat

| Parameter | Standard | |
|---|---|---|
| `question` | | Frage |
| `llm` | Parameter `llm` | `type` des LLM-Runners |
| `data_filter` | leer | Filter für die Kontextsuche |
| `limit` | `5` | Abschnitte als Kontext, 0 = ohne Kontext |
| `history` | leer | bisheriger Verlauf `[{"role": "user"/"assistant", "content": "..."}]` |
| `system` | siehe unten | System-Prompt, der Kontext wird angehängt |
| `min_score`, `model`, `temperature`, `max_tokens` | | werden durchgereicht |

Ergebnis: `response`, `reasoning` (falls das Modell eines liefert), `hits` (die Quellen), `llm`, `model`.

Standard-System-Prompt: *Beantworte die Frage ausschließlich anhand des folgenden Kontexts. Wenn der Kontext die
Antwort nicht enthält, sage das. Nenne die verwendeten Quellen als [1], [2], ...*

LLM-Runner mit `op=chat` (z.B. `lmstudio`) bekommen System-Prompt, Verlauf und Frage als Nachrichten.
Andere LLM-Runner (nur `prompt`) bekommen alles als einen Text.

### Weitere

| op | Parameter | Ergebnis |
|---|---|---|
| `llms` | – | geladene LLM-Runner: `llms` (`type`, `class`, `chat`), `default` |
| `delete_document` | `doc_id` | `deleted` (Anzahl Chunks) |
| `list_documents` | `limit` (100, 0 = alle) | `documents`: `doc_id`, `chunks`, `data` |
| `index_info` | – | `backend`, `count`, `embedder`, `stats` (bei `local_index`) |
| `chunk` | `text`, `chunk_size`, `chunk_overlap` | `chunks`: Zerlegung ohne Embeddings, zum Ausprobieren |
| `train_index` | – | `local_index`: IVF-Index sofort neu trainieren |
| `index_nweb_group` | `instance`, `group_id`, `recursive` | Dokumente einer nweb-Gruppe indizieren, siehe [nweb](#nweb): `indexed`, `chunks`, `skipped`, `errors` |

Als LLM gilt ein geladener Runner mit `llm` in `TAGS`, der `op=chat` oder `prompt` kann
(`LlmQueue` und `LlmProxy` ausgenommen).

## Filter

`data_filter` vergleicht einfache Werte aus `document_data` auf Gleichheit. Alle Bedingungen müssen gelten.

```json
{"lang": "de"}                       // lang ist "de"
{"lang": "de", "year": 2026}         // beides
{"lang": ["de", "en"]}               // eine der Möglichkeiten
{"tags": "handbuch"}                 // bei document_data {"tags": ["handbuch", "intern"]}: Liste enthält den Wert
{"public": true}                     // true ist nicht dasselbe wie 1
```

- Erlaubt sind Zeichenketten, Zahlen, `true`/`false` und `null`, keine verschachtelten Objekte.
- Schlüssel müssen einfache Namen sein (`[A-Za-z_][A-Za-z0-9_]*`).
- Fehlt der Schlüssel in `document_data`, passt der Chunk nicht.
- Es gibt keine Bereiche (`year > 2020`). Dafür z.B. `year: [2021, 2022, 2023]` oder ein eigenes Feld.
- Qdrant vergleicht keine Kommazahlen. Bei `local_index` gilt `2026.0` gleich `2026`.

## Speicher

| `backend` | Klasse | Installation | geeignet für                                        |
|---|---|---|-----------------------------------------------------|
| `memory` | `nxml.rag.store:MemoryVectorStore` | nur numpy | Tests, temporäre Daten, **nichts wird gespeichert** |
| `local_index` | `nxml.rag.index:LocalIndexStore` | nur numpy | lokal, auch große Datenmengen (empfohlen) IVF  |
| `file` | `nxml.rag.store:FileVectorStore` | nur numpy | kleine Indizes, einfach zu lesendes Format          |
| `postgres` | `nxml.rag.store:PostgresVectorStore` | `psycopg[binary]`, Postgres mit pgvector | vorhandene Postgres-Datenbank                       |
| `qdrant` | `nxml.rag.store:QdrantVectorStore` | `qdrant-client`, Qdrant-Server | mehrere Clients, Server-Betrieb                     |

Alle Backends speichern dieselben Daten und liefern dieselben Treffer. Ein Wechsel heißt aber neu indizieren.

### memory

```
rag: nxml.rag:RagRunner   # args: backend=memory
```

Alles liegt nur im Arbeitsspeicher und ist nach einem Neustart weg. Gedacht für Tests, kurzlebige Sammlungen
(z.B. ein Chat über gerade hochgeladene Dateien) oder Daten, die beim Start ohnehin neu indiziert werden.
Gesucht wird exakt wie bei `file`, also gut bis einige zehntausend Chunks. Nach `clear()` darf ein anderes
Embedding-Modell mit anderer Dimension verwendet werden. Aliase: `mem`, `ram`.

### local_index

Effizienter lokaler Index im Verzeichnis `path`:

```
index.sqlite           Chunks (Text, document_data) und eine Tabelle für die Filter
vectors.<gen>.f32      Vektoren, float32, auf Länge 1 normiert, wird nur angehängt
centroids.<gen>.npy    Zentren der IVF-Listen (ab min_train Chunks)
```

- **Suche:** Die Vektoren werden per k-Means in Listen eingeteilt (Inverted File, IVF). Eine Anfrage wird mit
  allen Zentren verglichen, nur die `nprobe` nächsten Listen werden exakt durchsucht. Unter `min_train` Chunks
  wird alles exakt durchsucht.
- **Filter** laufen über einen Index in sqlite. Bleiben höchstens 20.000 Chunks übrig, wird unter diesen exakt
  gesucht, sonst in den IVF-Listen.
- **Schreiben:** Neue Vektoren werden angehängt, Löschen entfernt nur die Einträge in sqlite. Ab 25 % gelöschten
  Zeilen wird die Vektordatei kompaktiert. Hat sich die Anzahl seit dem letzten Training verdoppelt, wird neu
  trainiert.
- **Absturzsicher:** Neue Dateien bekommen eine neue Nummer (`<gen>`) und werden erst in derselben
  sqlite-Transaktion gültig. Reste abgebrochener Vorgänge werden beim nächsten Start gelöscht.

Messung (100.000 Chunks, 384 Dimensionen, CPU):

| | `local_index` | `file` |
|---|---|---|
| Suche | 6,7 ms | 73 ms |
| Suche mit Filter | 22 ms | 180 ms |
| Recall@10 gegen exakte Suche | 1,0 | 1,0 |

`nprobe` größer: genauer, aber langsamer. Automatisch ist es 1/16 der Listen, mindestens 8.
`index_info` zeigt in `stats` Anzahl, Dimensionen, Listen und die Größe beim letzten Training.

### file

Wie `memory`, aber nach jeder Änderung im Verzeichnis `path` gespeichert: `index.json`, `chunks.jsonl`
(ein Chunk je Zeile) und `vectors.npy`. Alles liegt im Speicher, jede Suche vergleicht mit allen Vektoren,
jede Änderung schreibt die Dateien neu. Gut bis einige zehntausend Chunks.

### postgres

```
rag: nxml.rag:RagRunner   # args: backend=postgres, dsn=postgresql://user:pw@host/db, table=rag_chunks
```

- Braucht die Erweiterung [pgvector](https://github.com/pgvector/pgvector). Der Runner versucht
  `CREATE EXTENSION vector`, ohne Rechte muss sie vorher angelegt sein.
- Die Tabelle wird beim ersten Indizieren mit der Dimension des Embedding-Modells angelegt, mit HNSW-Index
  (bis 2000 Dimensionen) und GIN-Index auf `data` (jsonb) für die Filter.
- Zum Ausprobieren: `docker run -d -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg17`

### qdrant

```
rag: nxml.rag:RagRunner   # args: backend=qdrant, url=http://localhost:6333, collection=rag
```

- Die Collection wird beim ersten Indizieren angelegt (Distanz Cosine). `document_data` liegt im Payload unter
  `data`.
- Für schnelle Filter auf großen Collections einen Payload-Index anlegen, z.B. für `data.lang` (Typ keyword).
- Mit `qdrant_path` läuft Qdrant lokal ohne Server (nur zum Testen gedacht).

## Embedder

Der RagRunner ruft den Runner `embedder` auf:

1. `{"op": "embed", "input": [texte], "input_type": "passage" | "query", "model": embedding_model}`,
   erwartet `embeddings`.
2. Klappt das nicht, einzeln: `{"text": text, "input_type": ...}`, erwartet `embedding` oder `features`.

Beim Indizieren ist `input_type` `passage`, bei Suche und Chat `query`. Modelle wie E5 brauchen das
(Präfixe `passage: ` und `query: `).

| `embedder` | Runner | Hinweis |
|---|---|---|
| `text_embed` | `nxml.analyse:TextEmbeddings` | **empfohlen für Deutsch und Englisch**, siehe unten |
| `lmstudio` | `nxml.llm:LMStudio` | Embedding-Modell in LM Studio laden, z.B. bge-m3 oder multilingual-e5 |
| `bert` | `nxml.analyse:BertEmbeddings` | all-MiniLM-L6-v2, 384 Dimensionen, fürs Englische trainiert |
| `clip` | `nxml.analyse:ClipEmbeddings` | Text- und Bildvektoren im selben Raum, schneidet Texte nach 77 Token ab |

Jeder andere Runner mit einer der beiden Schnittstellen funktioniert ebenso.

**Alle Chunks eines Index müssen vom selben Modell stammen.** Eine andere Dimension melden die Backends als
Fehler. Ein anderes Modell mit gleicher Dimension fällt nicht auf, die Treffer werden aber unbrauchbar.
Nach einem Modellwechsel neu indizieren, mit neuem `path`, neuer `collection` bzw. `table` oder geleertem Index.

### text_embed

`nxml.analyse:TextEmbeddings`, Hugging Face über sentence-transformers:

```
pip install sentence-transformers        # dazu torch, für CUDA passend zur Grafikkarte
```

| Parameter | Standard | |
|---|---|---|
| `model_name` | `BAAI/bge-m3` | Repo-ID auf dem Hub |
| `device_mode` | `auto` | `auto`, `gpu` oder `cpu` |
| `batch_size` | `16` | Texte je Durchlauf |
| `max_seq_length` | `0` | Token je Text begrenzen, 0 = Modellstandard |
| `fp16` | `true` | auf der GPU in halber Genauigkeit |

| Modell | Dim. | max. Token | |
|---|---|---|---|
| `BAAI/bge-m3` | 1024 | 8192 | Standard: beste Qualität DE+EN, lange Texte, ~2,3 GB (GPU fp16 ~1,2 GB VRAM) |
| `intfloat/multilingual-e5-large` | 1024 | 512 | sehr gut |
| `intfloat/multilingual-e5-base` | 768 | 512 | guter Kompromiss |
| `intfloat/multilingual-e5-small` | 384 | 512 | klein und schnell, auch auf CPU |
| `jinaai/jina-embeddings-v2-base-de` | 768 | 8192 | zweisprachig DE/EN, führt Code aus dem Modell-Repo aus |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | sehr klein, nur kurze Texte |

Das Modell wird beim ersten Aufruf geladen. Die Ops `load`, `unload`, `status` und `delete_cache` kommen vom
`HuggingRunner` (z.B. `{"type": "text_embed", "op": "unload"}` gibt den Grafikspeicher frei), `models` listet die
Modelle oben.

**Chunk-Größe passend zum Modell:** `chunk_size` zählt Zeichen, die Modelle zählen Token
(Deutsch etwa 3–4 Zeichen je Token). Bei 512 Token also höchstens etwa 1500 Zeichen, sonst wird der Rest
abgeschnitten.

## Weboberfläche

`/pysys/runner?type=rag&rag=index`

| Seite | |
|---|---|
| **Chat** | LLM aus den geladenen Runnern wählen, Filter, Anzahl Abschnitte, System-Prompt, Verlauf an/aus. Antworten zeigen Quellen und ggf. Reasoning. Strg+Enter sendet. |
| **Suche** | Treffer mit Score und `document_data` |
| **Indizieren** | Text mit `document_data` (JSON) oder Textdateien (`.txt`, `.md`, `.csv`, `.json`, `.html`, `.xml`). Bei Dateien ist `doc_id` der Dateiname. „Nur Chunks anzeigen“ zeigt die Zerlegung ohne Embeddings. |
| **Dokumente** | indizierte Dokumente mit Anzahl Chunks, löschen |
| **Status** | Backend, Embedder, Chunk-Größe, Anzahl, Index-Statistik, geladene LLMs. Bei `local_index` Knopf zum Neutrainieren. |
| **nweb** | nweb-Instanzen aus der Konfiguration, Gruppenbaum je Instanz, Gruppe mit Untergruppen und den ersten 25 Dokumenten, Knopf „Gruppe indizieren“ (optional mit Untergruppen) |

### nweb

Dokumente aus einer nweb-Instanz übernehmen. Instanzen sind Einträge mit `url`, `username` und `password`;
gesucht wird wie bei `NWebClient(name)` in den Runner-Args, dann in `nweb.json`, dann in `/etc/nweb.json`:

```json
{"meinnweb": {"url": "https://example.org/4.0/", "username": "...", "password": "..."}}
```

Per Job oder über die Seite **nweb**:

```json
{"type": "rag", "op": "index_nweb_group", "instance": "meinnweb", "group_id": "754", "recursive": 1}
```

- Bilder, Dateien, Audio, Video und PDF werden übersprungen. HTML-Inhalte (`markup`) werden in Text umgewandelt,
  der Titel steht vor dem Inhalt.
- `doc_id` ist `nweb:<instance>:<guid>`, erneutes Indizieren ersetzt also den alten Stand.
- `document_data`: `source` (`nweb`), `instance`, `group_id`, `group` (Titel), `nweb_id`, `title`, `kind`, `url`.
  Damit lässt sich z.B. mit `{"source": "nweb", "group_id": "754"}` filtern.
- Das Indizieren läuft im Request, bei großen Gruppen kann das dauern.

## Chunking

`chunk_text(text, size, overlap)` aus `nxml.rag`:

1. trennt an Leerzeilen (Absätzen),
2. zu lange Absätze an Satzenden (`.!?;:`), zu lange Sätze an Leerzeichen, notfalls hart,
3. fügt die Teile zu Chunks bis `size` Zeichen zusammen,
4. setzt an den Anfang jedes Chunks ganze Sätze bzw. Absätze vom Ende des vorigen, zusammen höchstens `overlap`
   Zeichen.

Innerhalb eines Chunks werden die Teile mit Leerzeichen verbunden, Zeilenumbrüche gehen dabei verloren.

## Eigenes Backend

Ein Backend erbt von `nxml.rag.store.VectorStore` und implementiert `add(chunks)`,
`search(embedding, limit, data_filter)`, `delete_document(doc_id)` und `count()`, optional `documents()`,
`clear()`, `close()` und `stats()`. Eintragen in `nxml.rag.store.BACKENDS` (Klasse oder `'modul:Klasse'`)
oder als fertiges Objekt mit `RagRunner(store=...)` übergeben.

Ein Chunk für `add`:

```python
{'id': 'doc#0', 'doc_id': 'doc', 'chunk_index': 0, 'text': '...', 'data': {...}, 'embedding': [0.1, ...]}
```

`search` liefert dieselben Felder ohne `embedding`, dafür mit `score`, besten Treffer zuerst.
Für `data_filter` gibt es `nxml.rag.store.matches_filter(data, data_filter)` als Referenz.

## Module

| Modul | Inhalt |
|---|---|
| `nxml.rag` | importiert alles Wichtige: `RagRunner`, `chunk_text`, `create_store`, Backends |
| `nxml.rag.runner` | `RagRunner`, `chunk_text` |
| `nxml.rag.store` | `VectorStore`, Filter, `MemoryVectorStore`, `FileVectorStore`, `PostgresVectorStore`, `QdrantVectorStore`, `create_store` |
| `nxml.rag.index` | `LocalIndexStore` |
| `nxml.rag.ui` | Weboberfläche und Chat (Mixin) |

## Tests

```
python -m unittest tests.test_rag tests.test_rag_index
```

Qdrant wird lokal im Speicher getestet, wenn `qdrant-client` installiert ist. Postgres nur mit einer Datenbank:
`RAG_TEST_PG_DSN=postgresql://postgres:pw@localhost/postgres`.
