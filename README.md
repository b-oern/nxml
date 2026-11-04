# NXML

```
pip install git+github.com/b-oern/nxml
```

## NLP

Analyisert einen Text.

Folgende Features werden ermittelt

 - `lang` Sprache (`en`, `de`, ...)
 - `toxity` Wie viel Hate steckt im Text, unterteilt in {'toxicity': 0.00019621708, 'severe_toxicity': 0.00019254998, 'obscene': 0.0012626372, 'identity_attack': 0.0003226225, 'insult': 0.0008828422, 'threat': 0.00013756882, 'sexual_explicit': 9.029167e-05}
 - `words` Array von Wörtern
## OCR

Texterkennung (Bild -> Text).

 - `ocr` (`nxml.image:OCR`) läuft auch auf schwachen Rechnern (Raspberry Pi), ohne PyTorch.
   Nutzt Tesseract (`apt install tesseract-ocr tesseract-ocr-deu`), falls installiert, sonst RapidOCR (ONNX).
   Liefert `text` und `lines` (Text, Konfidenz 0-100, Box `[x1, y1, x2, y2]`).
 - `ocrai` (`nxml.image:OcrAi`) nutzt GOT-OCR 2.0 (PyTorch), genauer bei Fotos, Tabellen und Formeln; `format=1` gibt Markdown/LaTeX aus. GPU empfohlen.

## Video-OCR

`video_ocr` (`nxml.video.ocr:VideoOcr`) liest einen Bildausschnitt eines Videos in festen Abständen aus,
z.B. die Anzeige eines Messgeräts. UI unter `/pysys/runner?type=video_ocr`:

 1. Video hochladen (oder Pfad auf dem Server angeben)
 2. Bereich im Bild mit einer verschiebbaren Box wählen, mit "OCR testen" prüfen
 3. Abstand (z.B. alle 30 s), Zeitraum und OCR-Runner (`ocr`, `ocrai`, ...) wählen, Auswertung starten

Ergebnis ist eine Tabelle mit Zeit, Wert und erkanntem Text, Export als CSV oder JSON.
Optionen: `number_only` (nur die erste Zahl als Wert), `decimals` (feste Nachkommastellen, falls die OCR den Dezimalpunkt nicht erkennt: 2 macht aus `12 50` den Wert `12.50`), `invert` (helle Schrift auf dunkler Anzeige), `psm` (Tesseract, 7 = eine Zeile).
Braucht OpenCV (`pip install opencv-python-headless`) oder ffmpeg.

## LM Studio

`lmstudio` (`nxml.llm:LMStudio`) nutzt einen lokalen [LM-Studio](https://lmstudio.ai)-Server (Standard `http://localhost:1234`).
UI unter `/pysys/runner?type=lmstudio`:

 - Status: Server-URL, erreichbar/nicht erreichbar, geladene Modelle; LM Studio (Daemon) und Server über das `lms`-CLI starten und stoppen, lesende CLI-Befehle (`lms status`, `ps`, `ls`, `runtime ls`, ...)
 - Modelle: geladene und verfügbare Modelle mit Typ, Quantisierung, Größe und max. Kontext; laden (Kontextlänge, GPU-Anteil, TTL), entladen, herunterladen
 - Chat: Modellauswahl, Systemprompt, `temperature`, `max_tokens`, Verlauf, Bild für Vision-Modelle, alternativ Completion ohne Chat-Template
 - Embeddings: ein Text pro Zeile, Vektoren und Kosinus-Ähnlichkeit

Jobs (`op=...`): `status`, `models`, `chat`, `completion`, `embed`, `load`, `unload`, `download`, `download_status`, `server_start`, `server_stop`, `daemon`, `cli`.
Wie die anderen LLM-Runner versteht er auch `prompt` (und `cprompt`).

Konfiguration: `LMSTUDIO_URL`, `LMSTUDIO_MODEL` (sonst das erste geladene LLM), `LMSTUDIO_EMBEDDING_MODEL`, `LMSTUDIO_LMS` (Pfad zu `lms`, sonst PATH oder `~/.lmstudio/bin`).
Chat und Embeddings laufen über die OpenAI-kompatible API, Modelle, Laden und Download über die native REST-API `/api/v1` (Modellliste auch `/api/v0`).

### Remote LM Studio

`remote_lmstudio` (`nxml.remote.llm:RemoteLmStudio`) steuert den `lmstudio`-Runner einer anderen Instanz über einen Job-Speicher (Standard: MongoDB-Runner `mongodb`, verschlüsselt über den CryptRunner `crypt`). Senden und Abfragen übernimmt `RemoteJobMixin` aus dem Projekt nxrunner (`nxrunner.jobstore`), nxrunner muss installiert sein.

```
rlms: nxml.remote.llm:RemoteLmStudio(target="gpu-pc", store="mongodb", remote_type="lmstudio")
```

Gleiche Ops und gleiche UI wie `lmstudio`, die Statuszeile zeigt ausstehende Jobs und Fehler.
Jobs warten bis zur Antwort (höchstens `wait` Sekunden, Standard 600); mit `async=1` wird der Job nur eingestellt und liefert `remote_guid`, das Ergebnis holt `op=remote_result guid=...` ab. `op=remote_state` liefert die Statuszeile.
Die Antwortzeit hängt vom Ticker der Ziel-Instanz ab (z.B. `mongo_job_cron`) und von `poll_interval` (Standard 10 s).
Test ohne zweite Instanz: `store: nxrunner.jobstore:DirectJobStore` und `RemoteLmStudio(target="local", store="store", poll_interval=1)`.
