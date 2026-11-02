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
