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

## SLAM

`slam` (`nxml.vision.slam:SlamRunner`) bestimmt aus einem Kamerabild die Position der Kamera und baut eine Weltkarte (Punktwolke) auf, ähnlich wie ARCore, aber nur mit dem Bild.
Quelle: Videodatei, Stream-URL (MJPEG über HTTP, JPEG-Snapshot-URL, RTSP/HLS über FFmpeg) oder Webcam-Index (`0`).
UI unter `/pysys/runner?type=slam`: Quelle wählen oder Video hochladen, Kamerabild mit verfolgten Punkten, Radar um die Kamera (Blickrichtung oben) und Karte von oben mit der Bewegung der Kamera als Linie.

 - Backend `opencv` (Standard für Streams): eingebaute monokulare Visual Odometry, live, braucht nur `opencv-python-headless` und `numpy`
 - Backend `stella`: [stella_vslam](https://github.com/stella-cv/stella_vslam) `run_video_slam` als Subprozess, nur Videodateien, Ergebnis nach dem Durchlauf (`stella_bin`, `stella_vocab`, `stella_viewer`, `stella_args`; das ORB-Vokabular wird bei Bedarf geladen, für die Karte `pip install msgpack`)

Der Button `QR` unter der Quelle zeigt einen QR-Code `nx://stream?media=video&to=<url>` für die nx-App (Ziel editierbar, bei Aufruf über localhost wird die Netz-IP eingesetzt, braucht `pip install qrcode`).
Das Ziel `op=start_from_stream` nimmt entweder die Stream-URL des Handys (`source`/`stream`/`url`), einzelne Bilder (`image` als Base64-JPEG) oder legt eine Session an und liefert `push_url` (`/pysys/dispatcher?type=slam&slam=push&session=...`), an die JPEGs oder ein fortlaufender MJPEG-Strom per POST gehen.

Events: `slam_pose` (je Bild: `x`, `y`, `z`, Quaternion, `yaw`, `pitch`, `roll`, `frame`, `t`), `slam_state` (`init`, `tracking`, `lost`), `slam_done`.
Jobs (`op=...`): `start` (`source`, `backend`, `fov`, `width`, `frame_skip`, `realtime`, Kalibrierung `fx fy cx cy k1 k2 p1 p2 k3`), `stop`, `state`, `pose`, `sessions`, `upload`, `export` (`ply`, `csv`, `json`).
Arbeitsverzeichnis `work_dir`, Standard `/tmp/nxml_slam` (je Lauf `sessions/<id>/` mit `trajectory.csv`, `points.ply`, `session.json`).
Monokular ist der Maßstab unbekannt (erste Bewegung = 1), Koordinaten: Ursprung = erste Kameraposition, x rechts, y unten, z vorwärts.

```
slam: nxml.vision.slam:SlamRunner(fov=70, width=640)
```

## Punktwolken

`pointcloud` (`nxml.vision.pointcloud:PointCloudRunner`) zeigt Punktwolken im Browser an (three.js aus `/static/js/three/three.module.min.js` und `controls/OrbitControls.js`, änderbar mit `three_url`/`orbit_url`).
Quellen: PLY (ascii und binär), XYZ/TXT/CSV (`x y z [r g b]`) als Upload oder Pfad auf dem Server, sowie die Karte eines SLAM-Laufs, auch live. Im SLAM-Runner öffnet der Link `3D-Ansicht` den gewählten Lauf (`/pysys/dispatcher?type=pointcloud&slam=<session>`).
Ansicht: drehen, verschieben, zoomen, Doppelklick setzt das Drehzentrum, Farben nach RGB/Höhe/einfarbig, Punktgröße, Höhenfilter mit `Boden ausblenden`, Kamerabahn, OpenCV-Achsen (y nach unten) umdrehen. Große Wolken werden auf `max_points` Punkte reduziert (Standard 1 Mio.).
Arbeitsverzeichnis `work_dir`, Standard `/tmp/nxml_pointcloud`.

```
pointcloud: nxml.vision.pointcloud:PointCloudRunner
```

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

## opencode-Agent

`opencode` (`nxml.agents:OpenCodeAgent`) startet einen [opencode-Server](https://opencode.ai/docs/server/) (`opencode serve`) im Projektverzeichnis und nutzt dessen HTTP-API.

```
opencode: nxml.agents:OpenCodeAgent(project_path="C:/dev/projekt", model="anthropic/claude-sonnet-4-5")
```

Parameter: `start=True` (Server sofort starten), `port=None` (freier Port), `project_path=None` (aktuelles Verzeichnis), `url` (vorhandenen Server nutzen), `password` (Basic Auth bzw. Pairing-Token, beim Selbststarten wird eines erzeugt), `provider`, `model`, `remote="origin"`.
Genutzt wird die neue API unter `/api` (https://opencode.ai/docs/api), Prompts laufen asynchron, gewartet wird auf die `idle`-Nachricht der Session.

`do_issue(name, task_text, wait=False)` legt den Branch `feat_<name>` an, lässt opencode `task_text` bearbeiten, committet (`feat: <name>`) und pusht den Branch und wechselt zurück auf den Ausgangs-Branch. Läuft ohne `wait` im Hintergrund, Stand über `op=list_issues`.
Provider und Modell kommen aus `provider`/`model` des Agents (`set_model`, UI). Das Arbeitsverzeichnis muss sauber sein; Reste eines fehlgeschlagenen Laufs landen in `git stash`.

UI unter `/pysys/runner?type=opencode`: Issues (Name, Provider/Modell-Auswahl, Aufgabe, Verlauf mit Tool-Aufrufen, Abbrechen) und Server (Status, Start/Stopp, Pairing-Link für die opencode-Web-UI, Sessions).
Weitere Ops: `status`, `server_start`, `server_stop`, `pairing`, `list_providers`, `set_model`, `list_sessions`, `session_messages`, `abort`, `ask`.
Konfiguration: `OPENCODE_BIN`, `OPENCODE_URL`, `OPENCODE_PROVIDER`, `OPENCODE_MODEL`, `OPENCODE_SERVER_PASSWORD`. Sessions werden mit allen Berechtigungen angelegt, offene Berechtigungsanfragen werden automatisch erlaubt.
