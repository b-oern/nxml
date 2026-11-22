#
# Project: nxml
#
# Prompt-Editor: Teile eines Bild-Prompts regelbasiert entfernen oder ersetzen (ohne LLM)
#
#   pe = PromptEditor()
#   pe.edit("a knight, comic style, halftone dots, red hair", "comic style entfernen\nrot haar -> blonde haare")
#
# Änderungen (eine pro Zeile), Deutsch oder Englisch:
#   comic style entfernen | entferne comic style | - comic style | remove comic style
#   rot haar -> blonde haare | ersetze rote haare durch blonde haare | red hair => blonde hair
#   + film grain | hinzufügen film grain
#   ? comic | markiere comic | comic hervorheben   (ändert nichts, markiert Fundstellen mit Aliasen/Beschreibung)
#
# Option strip_angle (Standard True): alles in <...> (LoRA, Hypernetwork, Embedding) wird automatisch entfernt.
#
# Was entfernt/ersetzt wird, gibt der Nutzer frei ein. Die Wortlisten (EXPANSIONS) erweitern die Eingabe nur
# um verwandte Begriffe (z.B. "comic style" -> halftone, cel shading, ink outlines ...).
#
import json
import re
import difflib

from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import dev as d


# Erweiterungen: Eingabe (Alias) -> alle Begriffe, die dazu gehören (Regex, englisch + deutsch)
EXPANSIONS = {
    'comic': dict(
        label="Comic-Stil",
        description="Gibt dem Bild einen Comic-Look: Comic-Hefte, Rasterpunkte (halftone), Cel-Shading, dicke Konturen, Sprechblasen",
        aliases=['comic', 'comics', 'comic book', 'comicbook', 'cartoon', 'zeichentrick', 'graphic novel', 'toon'],
        terms=[r"comic[\s-]?books?", r"comics?", r"cartoon(?:ish|y|s)?", r"toon[\s-]?shad(?:ed|ing|er)",
               r"graphic[\s-]novels?", r"manga", r"halftone(?:[\s-](?:dots?|pattern|shading|texture))?",
               r"ben[\s-]?day[\s-]dots?", r"cel[\s-]?shad(?:ed|ing)",
               r"(?:bold|thick|heavy|clean|strong)\s+(?:black\s+)?(?:ink\s+)?(?:out)?lines?",
               r"ink(?:ed)?\s+(?:out)?lines?", r"inking", r"line[\s-]?art", r"pop[\s-]art", r"speech[\s-]bubbles?",
               r"comic[\s-]panels?", r"flat\s+colou?rs?", r"ligne\s+claire", r"onomatopoeia", r"zeichentrick",
               r"comic[\s-]?stil"]),
    'anime': dict(
        label="Anime-Stil",
        description="Japanischer Zeichentrick- und Manga-Look",
        aliases=['anime', 'manga', 'chibi'],
        terms=[r"anime", r"manga", r"chibi", r"cel[\s-]?shad(?:ed|ing)", r"studio\s+ghibli", r"kawaii",
               r"visual\s+novel", r"waifu"]),
    'photo': dict(
        label="Foto-Stil",
        description="Fotografischer Look: Kamera, Objektiv, Tiefenschärfe, Filmkorn",
        aliases=['photo', 'foto', 'photorealistic', 'fotorealistisch', 'realistic', 'realistisch'],
        terms=[r"photo[\s-]?realistic", r"photo[\s-]?realism", r"hyper[\s-]?realistic", r"raw\s+photo",
               r"photograph(?:y|ic)?", r"photo", r"dslr", r"\d{2,3}\s?mm(?:\s+lens)?", r"bokeh", r"f/\d(?:\.\d)?",
               r"depth\s+of\s+field", r"film\s+grain", r"kodak\s+\w+", r"canon\s+eos\s*\w*"]),
    'painting': dict(
        label="Malerei",
        description="Malerische Techniken: Öl, Aquarell, Pinselstriche, Leinwand",
        aliases=['painting', 'malerei', 'gemälde', 'oil painting', 'ölgemälde', 'watercolor', 'aquarell'],
        terms=[r"oil\s+painting", r"painting", r"painted", r"water[\s-]?colou?r", r"acrylic", r"gouache",
               r"impasto", r"brush\s?strokes?", r"canvas(?:\s+texture)?", r"aquarell", r"gemälde"]),
    'sketch': dict(
        label="Skizze",
        description="Skizzen- und Zeichentechniken: Bleistift, Kohle, Schraffur",
        aliases=['sketch', 'skizze', 'drawing', 'zeichnung', 'pencil'],
        terms=[r"(?:pencil|charcoal|graphite|ink)\s+(?:sketch|drawing)", r"sketch(?:y|ed)?", r"charcoal",
               r"graphite", r"hatching", r"cross[\s-]?hatching", r"skizze"]),
    '3d': dict(
        label="3D-Render",
        description="3D/CGI-Look: Render-Engines und Raytracing",
        aliases=['3d', '3d render', 'render', 'cgi'],
        terms=[r"3d\s+render(?:ing|ed)?", r"3d", r"octane\s+render", r"unreal\s+engine\s*\d*", r"blender",
               r"cgi", r"ray[\s-]?tracing", r"v-?ray", r"render(?:ed|ing)?"]),
    'pixel': dict(
        label="Pixel-Art",
        description="Retro-Pixelgrafik: 8/16-Bit, Sprites",
        aliases=['pixel', 'pixel art', 'pixelart', '8bit', '8-bit', '16-bit'],
        terms=[r"pixel[\s-]?art", r"pixelated", r"\d{1,2}[\s-]?bit", r"sprite(?:\s+sheet)?"]),
    'background_people': dict(
        label="Leute im Hintergrund",
        description="Personen und Menschenmengen im Hintergrund",
        aliases=['people in background', 'people in the background', 'leute im hintergrund',
                 'menschen im hintergrund', 'personen im hintergrund', 'crowd', 'menschenmenge',
                 'background people', 'background characters', 'bystanders', 'passanten'],
        terms=[r"(?:many\s+|other\s+|some\s+)?(?:people|persons|figures|crowds?|characters|bystanders|pedestrians|"
               r"passers[\s-]?by|onlookers|spectators|tourists)\s+in\s+(?:the\s+)?background",
               r"background\s+(?:people|crowd|characters|figures|extras)", r"crowds?(?:\s+of\s+people)?",
               r"bystanders", r"pedestrians", r"passers[\s-]?by", r"onlookers", r"spectators",
               r"busy\s+street", r"menschenmenge", r"passanten"]),
}

# Stil-Füllwörter: werden beim Erkennen einer Erweiterung ignoriert ("comic style" == "comic")
STYLE_FILLER = {'style', 'styled', 'stil', 'look', 'aussehen', 'optik', 'aesthetic', 'ästhetik', 'art', 'artwork',
                'alles', 'all', 'everything', 'like', 'artig', 'effekt', 'effect', 'vibe', 'feeling'}

# Wörter, die in einem Prompt-Teil übrig bleiben dürfen, damit der ganze Teil als "nur Stil" gilt
CHUNK_FILLER = STYLE_FILLER | {'in', 'a', 'an', 'the', 'of', 'with', 'by', 'and', 'drawn', 'drawing', 'rendered',
                               'render', 'illustration', 'illustrated', 'shading', 'shaded', 'texture', 'dots',
                               'lines', 'outlines', 'cover', 'page', 'strip', 'panel', 'colors', 'colours', 'im',
                               'mit', 'und', 'ein', 'eine', 'der', 'die', 'das', 'book', 'background', 'hintergrund'}

# Deutsch -> Englisch (je Wort Alternativen), für Eingaben auf Deutsch bei englischen Prompts
DE_EN = {
    'haar': ['hair'], 'haare': ['hair'], 'haaren': ['hair'], 'frisur': ['hairstyle', 'hair'],
    'augen': ['eyes'], 'auge': ['eye'], 'kleid': ['dress'], 'hemd': ['shirt'], 'jacke': ['jacket'],
    'mantel': ['coat'], 'hut': ['hat'], 'mütze': ['beanie', 'cap'], 'bart': ['beard'], 'haut': ['skin'],
    'lippen': ['lips'], 'schuhe': ['shoes'], 'stiefel': ['boots'], 'hose': ['pants', 'trousers'],
    'rock': ['skirt'], 'brille': ['glasses'], 'katze': ['cat'], 'hund': ['dog'], 'auto': ['car'],
    'himmel': ['sky'], 'hintergrund': ['background'], 'vordergrund': ['foreground'],
    'leute': ['people', 'persons', 'crowd'], 'menschen': ['people', 'persons', 'crowd'],
    'personen': ['people', 'persons'], 'person': ['person'], 'frau': ['woman', 'girl'], 'mann': ['man', 'guy'],
    'kind': ['child', 'kid'], 'mädchen': ['girl'], 'junge': ['boy'], 'baum': ['tree'], 'bäume': ['trees'],
    'haus': ['house'], 'stadt': ['city'], 'wald': ['forest'], 'strand': ['beach'], 'meer': ['sea', 'ocean'],
    'sonnenuntergang': ['sunset'], 'nacht': ['night'], 'regen': ['rain'], 'schnee': ['snow'],
    'text': ['text'], 'schrift': ['text', 'lettering'], 'wasserzeichen': ['watermark'], 'logo': ['logo'],
    'lang': ['long'], 'lange': ['long'], 'langen': ['long'], 'kurz': ['short'], 'kurze': ['short'],
    'lockig': ['curly'], 'lockige': ['curly'], 'glatt': ['straight'], 'glatte': ['straight'],
    'tag': ['day'], 'morgen': ['morning'], 'abend': ['evening'], 'sonne': ['sun'], 'mond': ['moon'],
    'wolken': ['clouds'], 'berg': ['mountain'], 'berge': ['mountains'], 'fluss': ['river'], 'see': ['lake'],
    'blume': ['flower'], 'blumen': ['flowers'], 'burg': ['castle'], 'schloss': ['castle', 'palace'],
    'ritter': ['knight'], 'pferd': ['horse'], 'drache': ['dragon'], 'krieger': ['warrior'], 'nebel': ['fog', 'mist'],
    'feuer': ['fire'], 'wasser': ['water'], 'gras': ['grass'], 'wiese': ['meadow'], 'straße': ['street'],
    'markt': ['market'], 'licht': ['light'], 'schatten': ['shadow', 'shadows'],
    'stil': ['style'], 'im': ['in'], 'mit': ['with'], 'und': ['and'],
}

# Farben: kanonisch (EN) -> englische Synonyme / deutsche Stämme
COLORS = {
    'red': dict(en=['red', 'ginger', 'auburn', 'copper', 'crimson', 'scarlet', 'cherry red'], de=['rot'],
                holder='redhead'),
    'blonde': dict(en=['blonde', 'blond', 'golden', 'platinum blonde', 'platinum', 'strawberry blonde',
                       'honey blonde', 'flaxen'], de=['blond', 'platinblond', 'goldblond'], holder='blonde'),
    'black': dict(en=['black', 'jet black', 'raven', 'ebony'], de=['schwarz'], holder='black-haired'),
    'brown': dict(en=['brown', 'brunette', 'chestnut', 'chocolate', 'dark brown', 'light brown'], de=['braun'],
                  holder='brunette'),
    'white': dict(en=['white', 'snow white'], de=['weiß', 'weiss'], holder='white-haired'),
    'silver': dict(en=['silver', 'gray', 'grey', 'ash', 'ash gray'], de=['silber', 'grau'], holder='silver-haired'),
    'green': dict(en=['green', 'emerald'], de=['grün', 'gruen'], holder='green-haired'),
    'blue': dict(en=['blue', 'azure', 'navy'], de=['blau'], holder='blue-haired'),
    'pink': dict(en=['pink', 'rose'], de=['rosa', 'pink'], holder='pink-haired'),
    'purple': dict(en=['purple', 'violet', 'lavender', 'lilac'], de=['lila', 'violett'], holder='purple-haired'),
    'orange': dict(en=['orange'], de=['orange'], holder='orange-haired'),
    'yellow': dict(en=['yellow'], de=['gelb'], holder='yellow-haired'),
}

# Substantive: kanonisch (EN) -> Formen im Prompt (EN + DE)
NOUNS = {
    'hair': ['hair', 'hairs', 'hairstyle', 'locks', 'curls', 'braids', 'ponytail', 'mane', 'haar', 'haare', 'haaren'],
    'eyes': ['eyes', 'eye', 'augen', 'auge'],
    'dress': ['dress', 'gown', 'kleid'], 'shirt': ['shirt', 't-shirt', 'blouse', 'hemd'],
    'jacket': ['jacket', 'jacke'], 'coat': ['coat', 'mantel'], 'hat': ['hat', 'hut'], 'beard': ['beard', 'bart'],
    'skin': ['skin', 'haut'], 'lips': ['lips', 'lippen'], 'shoes': ['shoes', 'sneakers', 'schuhe'],
    'boots': ['boots', 'stiefel'], 'pants': ['pants', 'trousers', 'jeans', 'hose'], 'skirt': ['skirt', 'rock'],
    'car': ['car', 'auto'], 'sky': ['sky', 'himmel'], 'background': ['background', 'hintergrund'],
}

DE_ENDINGS = ('en', 'em', 'er', 'es', 'e')
EN_STOP = {'a', 'an', 'the', 'with', 'and', 'of', 'in', 'on', 'at', 'is', 'wearing', 'standing', 'sitting', 'her',
           'his', 'by', 'from', 'detailed', 'highly', 'masterpiece', 'best', 'quality'}
DE_STOP = {'der', 'die', 'das', 'den', 'dem', 'ein', 'eine', 'einen', 'mit', 'und', 'im', 'auf', 'trägt', 'steht',
           'sitzt', 'ihr', 'sein', 'von', 'vor', 'hat', 'ist'}
ARTICLES = r"(?:the|a|an|der|die|das|den|dem|des|ein|eine|einem|einer)"
# Anhängsel eines Stilbegriffs: "comic" + " book style"
STYLE_SUFFIX = r"(?:[\s-]+(?:book|art|style|styled|look|aesthetic|illustration|drawing|effect|stil|optik)(?![\w]))*"


def to_bool(value, default=True):
    """ '1', 'true', 'ja', 'on' -> True; None/'' -> default """
    if value is None or value == '':
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'ja', 'on')
    return bool(value)


def _norm(s):
    return re.sub(r"\s+", ' ', s.strip().lower())


def _words(s):
    return re.findall(r"[\wäöüß]+(?:-[\wäöüß]+)*", s.lower())


def _color_of(word):
    """ :return: kanonische Farbe für ein Wort (EN oder dekliniertes DE) oder None """
    w = word.lower()
    for canon, c in COLORS.items():
        if w in c['en'] or w in c['de']:
            return canon
        for stem in c['de']:
            if any(w == stem + e for e in DE_ENDINGS):
                return canon
    return None


def _noun_of(word):
    w = word.lower()
    for canon, forms in NOUNS.items():
        if w in forms:
            return canon
    return None


def _de_color_regex(canon):
    stems = COLORS[canon]['de']
    return [re.escape(s) + r"(?:en|em|er|es|e)?" for s in stems]


def _alt(patterns):
    """ Regex-Alternation, längste Muster zuerst """
    return '(?:' + '|'.join(sorted(set(patterns), key=len, reverse=True)) + ')'


def _pretty(term):
    """ Regex-Begriff lesbar machen: cel[\\s-]?shad(?:ed|ing) -> cel shad(ed|ing) """
    t = re.sub(r"\[\\s-?\]\??|\[\\s\\-\]\+|\\s[+?]?", ' ', term)
    t = t.replace('(?:', '(').replace(')?', ')').replace(r'\d', '#').replace(r'\w', '…')
    t = re.sub(r"\{[\d,]+\}|\?", '', t)
    return re.sub(r"\s+", ' ', t).strip()


def prompt_language(text):
    ws = _words(text)
    en = sum(1 for w in ws if w in EN_STOP)
    de = sum(1 for w in ws if w in DE_STOP or any(c in w for c in 'äöüß'))
    return 'de' if de > en else 'en'


class Chunk:
    """ Ein durch Komma (oberste Ebene) getrennter Teil des Prompts mit Position """

    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text

    @property
    def inner(self):
        """ Text ohne Klammern und Gewichtung: "((comic style:1.2))" -> "comic style" """
        t = self.text.strip()
        t = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", '', t)
        t = re.sub(r":\s*-?\d+(?:\.\d+)?$", '', t)
        return t.strip()


def split_chunks(prompt):
    """ Teilt am Komma/Zeilenumbruch (nicht in Klammern oder <lora:...>), Leerraum außen abgeschnitten """
    chunks, depth, start = [], 0, 0
    for i, ch in enumerate(prompt + ','):
        if ch in '([{<':
            depth += 1
        elif ch in ')]}>':
            depth = max(0, depth - 1)
        elif (ch in ',\n' and depth == 0) or i == len(prompt):
            seg = prompt[start:i]
            lead = len(seg) - len(seg.lstrip())
            s, e = start + lead, start + len(seg.rstrip())
            if e > s:
                chunks.append(Chunk(s, e, prompt[s:e]))
            start = i + 1
    return chunks


class Instruction:
    def __init__(self, line_no, raw, kind, source='', target=''):
        self.line_no, self.raw, self.kind, self.source, self.target = line_no, raw, kind, source, target
        self.matches = 0
        self.info = ''
        self.expansion = None
        self.suggestion = None
        self.description = None

    def to_dict(self):
        return dict(line=self.line_no, raw=self.raw, kind=self.kind, source=self.source, target=self.target,
                    matches=self.matches, info=self.info, expansion=self.expansion, suggestion=self.suggestion,
                    description=self.description)


REMOVE_PATTERNS = [
    r"^(?:entferne|entfernen|lösche|löschen|ohne|kein|keine|weg mit|remove|delete|drop|no|without|-)\s+(?P<x>.+)$",
    r"^(?P<x>.+?)\s+(?:entfernen|löschen|weg|raus|rausnehmen|weglassen|remove)$",
]
REPLACE_PATTERNS = [
    r"^(?:ersetze|ersetzen|ändere|tausche|mach|mache|replace|change)\s+(?P<a>.+?)\s+(?:durch|zu|mit|in|gegen|with|to|into)\s+(?P<b>.+)$",
    r"^(?P<a>.+?)\s*(?:->|=>|→|>>)\s*(?P<b>.+)$",
    r"^(?P<a>.+?)\s+(?:statt|anstatt|instead of)\s+(?P<b>.+)$",  # umgekehrt: "blond statt rot"
]
HIGHLIGHT_PATTERNS = [
    r"^(?:markiere|markieren|hervorheben|hebe hervor|zeige|zeig|finde|suche|highlight|mark|find|show|\?)\s*:?\s*(?P<x>.+)$",
    r"^(?P<x>.+?)\s+(?:markieren|hervorheben|anzeigen|finden|zeigen|highlight)$",
]
ADD_PATTERNS = [
    r"^(?:\+|füge hinzu|hinzufügen|ergänze|add)\s*:?\s*(?P<x>.+)$",
    r"^(?P<x>.+?)\s+(?:hinzufügen|ergänzen|dazu)$",
]


def instruction_lines(changes):
    """ Anweisungen als String (eine pro Zeile), Liste/Tupel oder JSON-Liste als String -> Zeilen """
    if changes is None:
        return []
    if isinstance(changes, str):
        s = changes.strip()
        if s.startswith('['):
            try:
                changes = json.loads(s)
            except ValueError:
                return changes.splitlines()
        else:
            return changes.splitlines()
    if isinstance(changes, (list, tuple)):
        return [line for item in changes for line in instruction_lines(str(item) if item is not None else None)]
    return str(changes).splitlines()


def parse_instructions(text):
    """ :param text: str (eine Anweisung pro Zeile), list/tuple oder JSON-Liste """
    res = []
    for no, line in enumerate(instruction_lines(text), start=1):
        raw = line.strip()
        if not raw or raw.startswith('#'):
            continue
        low = raw.rstrip('.!')
        inst = None
        for i, pat in enumerate(REPLACE_PATTERNS):
            m = re.match(pat, low, re.I)
            if m:
                a, bb = m.group('a').strip(), m.group('b').strip()
                if i == 2:
                    a, bb = bb, a
                inst = Instruction(no, raw, 'replace', a, bb)
                break
        if inst is None:
            for pat in HIGHLIGHT_PATTERNS:
                m = re.match(pat, low, re.I)
                if m:
                    inst = Instruction(no, raw, 'highlight', source=m.group('x').strip())
                    break
        if inst is None:
            for pat in ADD_PATTERNS:
                m = re.match(pat, low, re.I)
                if m:
                    inst = Instruction(no, raw, 'add', target=m.group('x').strip())
                    break
        if inst is None:
            for pat in REMOVE_PATTERNS:
                m = re.match(pat, low, re.I)
                if m:
                    inst = Instruction(no, raw, 'remove', source=m.group('x').strip())
                    break
        if inst is None:
            inst = Instruction(no, raw, 'unknown')
            inst.info = "Nicht verstanden. Beispiele: »comic style entfernen«, »rot haar -> blonde haare«, »+ film grain«, »? comic«"
        res.append(inst)
    return res


class Edit:
    def __init__(self, start, end, new, reason, kind, inst):
        self.start, self.end, self.new, self.reason, self.kind, self.inst = start, end, new, reason, kind, inst


class PromptEdit:
    """ Wendet Anweisungen auf einen Prompt an. Alle Fundstellen beziehen sich auf den Original-Prompt. """

    ANGLE_RX = re.compile(r"<[^<>]*>")

    def __init__(self, prompt, instructions, strip_angle=True):
        """ :param strip_angle: alles in <...> (z.B. <lora:name:0.8>) automatisch entfernen """
        self.prompt = prompt
        self.instructions = instructions
        self.strip_angle = strip_angle
        self.option_inst = Instruction(0, 'Option: <…> entfernen', 'option')
        self.angle_spans = [(m.start(), m.end()) for m in self.ANGLE_RX.finditer(prompt)]
        self.lang = prompt_language(prompt)
        self.chunks = split_chunks(prompt)
        self.edits = []
        self.highlights = []
        self.warnings = []

    # ---------- Hilfen ----------

    def find_expansion(self, phrase):
        """ :return: (key, group) wenn die Eingabe zu einer Erweiterung passt """
        words = [w for w in _words(phrase) if w not in STYLE_FILLER]
        core = ' '.join(words)
        full = _norm(phrase)
        for key, g in EXPANSIONS.items():
            if core in g['aliases'] or full in g['aliases']:
                return key, g
        for key, g in EXPANSIONS.items():
            if core and any(re.fullmatch(t, core) for t in g['terms']):
                return key, g
        aliases = {a: k for k, g in EXPANSIONS.items() for a in g['aliases']}
        close = difflib.get_close_matches(core, list(aliases), n=1, cutoff=0.78) if core else []
        if close:
            return aliases[close[0]], EXPANSIONS[aliases[close[0]]]
        return None, None

    def word_regex(self, word):
        """ Regex für ein Wort der Eingabe inkl. Übersetzung DE->EN, Farb- und Substantiv-Synonymen """
        alts = [re.escape(word)]
        color = _color_of(word)
        noun = _noun_of(word)
        if color:
            alts += [re.escape(x).replace(r'\ ', r'[\s-]+') for x in COLORS[color]['en']] + _de_color_regex(color)
        elif noun:
            alts += [re.escape(x) for x in NOUNS[noun]]
        else:
            stem = word
            for e in DE_ENDINGS:
                if word.endswith(e) and word[:-len(e)] in DE_EN:
                    stem = word[:-len(e)]
            alts += [re.escape(x) for x in DE_EN.get(word, DE_EN.get(stem, []))]
            if word.endswith('s') and len(word) > 3:
                alts.append(re.escape(word[:-1]))
            else:
                alts.append(re.escape(word) + 's')
        return _alt(alts)

    def phrase_core(self, phrase):
        """ Wörter in Reihenfolge, dazwischen Leerraum/Bindestrich und optional Artikel/Füllwörter """
        words = _words(phrase)
        if not words:
            return None
        sep = r"(?:[\s\-]+(?:" + ARTICLES + r"|in|im|of|with|mit)\b)*[\s\-]+"
        return sep.join(self.word_regex(w) for w in words)

    def phrase_regex(self, phrase):
        core = self.phrase_core(phrase)
        return None if core is None else r"(?<![\w])" + core + r"(?![\w])"

    def suggest(self, phrase):
        cands = [c.inner for c in self.chunks]
        close = difflib.get_close_matches(phrase.lower(), [c.lower() for c in cands], n=1, cutoff=0.5)
        return close[0] if close else None

    def in_angle(self, start, end):
        """ True, wenn start..end einen <...>-Ausdruck schneidet, ohne ihn ganz zu umfassen (LoRA nicht beschädigen) """
        return any(start < e and s < end and not (start <= s and e <= end) for s, e in self.angle_spans)

    def add_edit(self, edit):
        if edit.inst.kind != 'option' and self.in_angle(edit.start, edit.end):
            return False
        for e in self.edits:
            if edit.start < e.end and e.start < edit.end or (edit.start == edit.end == e.start == e.end):
                if edit.start != edit.end or e.start != e.end:
                    self.warnings.append(f"Zeile {edit.inst.line_no}: überschneidet sich mit Zeile "
                                         f"{e.inst.line_no}, »{self.prompt[edit.start:edit.end]}« übersprungen")
                    return False
        self.edits.append(edit)
        edit.inst.matches += 1
        return True

    # ---------- Entfernen ----------

    def removal_spans(self, regex):
        """
        Sucht regex in allen Teilen. Besteht ein Teil nur aus Treffern + Füllwörtern, wird der ganze Teil
        (mit Komma) entfernt, sonst nur die Treffer (inkl. "in a ... style", "with ...").
        :return: (ganze Teile [(chunk, [treffer])], inline [(start, end, treffer)])
        """
        whole, inline = [], []
        prefix = r"(?:(?:(?:drawn|rendered|illustrated|painted|done|made|gezeichnet)\s+)?(?:in|im|as)\s+(?:" + \
                 ARTICLES + r"\s+)?|with\s+(?:" + ARTICLES + r"\s+)?|mit\s+)?"
        rx = re.compile(prefix + r"(?P<hit>" + regex + r")" + STYLE_SUFFIX, re.I)
        for c in self.chunks:
            inner = c.inner
            hits = list(rx.finditer(c.text))
            if not hits:
                continue
            rest = rx.sub(' ', inner)
            if all(w in CHUNK_FILLER for w in _words(rest)):
                whole.append((c, [h.group('hit') for h in hits]))
                continue
            spans = []
            for h in hits:
                s, e = c.start + h.start(), c.start + h.end()
                if spans and re.fullmatch(r"[\s,]*(?:and|und|&|with|mit)?[\s,]*", self.prompt[spans[-1][1]:s], re.I):
                    spans[-1] = (spans[-1][0], e, spans[-1][2] + [h.group('hit')])
                else:
                    spans.append((s, e, [h.group('hit')]))
            for s, e, hs in spans:
                if s > 0 and self.prompt[s - 1] == ' ':
                    s -= 1
                elif e < len(self.prompt) and self.prompt[e] == ' ':
                    e += 1
                inline.append((s, e, hs))
        return whole, inline

    def apply_remove(self, inst: Instruction):
        key, group = self.find_expansion(inst.source)
        if group:
            regex = r"(?<![\w])" + _alt(group['terms'] + [self.phrase_core(inst.source)]) + r"(?![\w])"
            inst.expansion = dict(key=key, label=group['label'], terms=group['terms'])
            inst.info = f"Erweitert um Gruppe »{group['label']}« ({len(group['terms'])} Begriffe)"
            what = group['label']
        else:
            regex = self.phrase_regex(inst.source)
            if regex is None:
                inst.info = "Leere Eingabe"
                return
            inst.info = "Wörtlich gesucht (mit Übersetzung DE→EN und Synonymen)"
            what = inst.source
        whole, inline = self.removal_spans(regex)
        self.pending_whole.extend((c, hits, inst, what) for c, hits in whole)
        for s, e, hits in inline:
            self.add_edit(Edit(s, e, '', f"Entfernt: »{', '.join(hits)}« gehört zu »{what}« (Zeile {inst.line_no})",
                               'del', inst))
        if not whole and not inline:
            inst.suggestion = self.suggest(inst.source)

    def apply_strip_angle(self):
        """ <lora:...>, <hypernet:...>, <embedding> usw. entfernen: eigener Teil mit Komma, sonst nur der Ausdruck """
        inst = self.option_inst
        for c in self.chunks:
            hits = list(self.ANGLE_RX.finditer(c.text))
            if not hits:
                continue
            if not self.ANGLE_RX.sub('', c.text).strip(' ()[]{}:.0123456789'):
                self.pending_whole.append((c, [h.group(0) for h in hits], inst, '<…>'))
                continue
            for h in hits:
                s, e = c.start + h.start(), c.start + h.end()
                if s > 0 and self.prompt[s - 1] == ' ':
                    s -= 1
                elif e < len(self.prompt) and self.prompt[e] == ' ':
                    e += 1
                self.add_edit(Edit(s, e, '', f"Automatisch entfernt: »{h.group(0)}« (Option »<…> entfernen«)",
                                   'del', inst))

    def flush_whole_chunks(self):
        """ Ganze Teile inkl. Komma entfernen, ohne doppelte/fehlende Kommas zu erzeugen """
        removed = {id(c) for c, _, _, _ in self.pending_whole}
        by_chunk = {id(c): (c, hits, inst, what) for c, hits, inst, what in self.pending_whole}
        idx_kept = [i for i, c in enumerate(self.chunks) if id(c) not in removed]
        for i, c in enumerate(self.chunks):
            if id(c) not in removed:
                continue
            _, hits, inst, what = by_chunk[id(c)]
            prev_kept = any(j < i for j in idx_kept)
            if prev_kept:
                # Komma davor mitnehmen: vom Ende des vorherigen Teils bis zum Ende dieses Teils
                s, e = self.chunks[i - 1].end, c.end
            else:
                # erster (verbleibender) Teil: bis zum Anfang des nächsten Teils
                # davor liegende Teile wurden bereits samt Komma bis c.start entfernt
                s = 0 if i == 0 else c.start
                e = self.chunks[i + 1].start if i + 1 < len(self.chunks) else len(self.prompt)
            if inst.kind == 'option':
                reason = f"Automatisch entfernt: »{c.text}« (Option »<…> entfernen«)"
            else:
                reason = f"Ganzer Teil entfernt: »{c.inner}« — {', '.join('»' + h + '«' for h in hits)} gehört " \
                         f"zu »{what}« (Zeile {inst.line_no})"
            self.add_edit(Edit(s, e, '', reason, 'del', inst))
        self.pending_whole = []

    # ---------- Ersetzen ----------

    def parse_attr(self, phrase):
        ws = _words(phrase)
        color = next((_color_of(w) for w in ws if _color_of(w)), None)
        color_word = next((w for w in ws if _color_of(w)), None)
        noun = next((_noun_of(w) for w in ws if _noun_of(w)), None)
        if noun is None:
            for w in ws:
                for en in DE_EN.get(w, []):
                    if _noun_of(en):
                        noun = _noun_of(en)
        return color, color_word, noun

    def target_text(self, phrase):
        """ Zieltext in der Sprache des Prompts (DE-Wörter bei englischem Prompt übersetzen) """
        if self.lang == 'de':
            return phrase
        out = []
        for w in re.split(r"(\s+)", phrase):
            lw = w.lower()
            color = _color_of(lw)
            if color and lw not in COLORS[color]['en']:
                out.append(color)
            elif _noun_of(lw) and lw not in NOUNS[_noun_of(lw)][:3]:
                out.append(_noun_of(lw))
            elif lw in DE_EN:
                out.append(DE_EN[lw][0])
            else:
                out.append(w)
        return ''.join(out)

    @staticmethod
    def inflect(old, sc, new, tc):
        """ Deutsche Endung übernehmen: roten -> blonden (statt "blonde Haaren") """
        low = old.lower()
        stem = next((s for s in sorted(COLORS[sc]['de'], key=len, reverse=True) if low.startswith(s)), None)
        if stem is None or low not in [stem + e for e in ('',) + DE_ENDINGS]:
            return new
        ending = low[len(stem):]
        nl = new.lower()
        new_stem = next((s for s in sorted(COLORS[tc]['de'], key=len, reverse=True) if nl.startswith(s)), None)
        if new_stem is None:
            return new
        return new[:len(new_stem)] + ending

    @staticmethod
    def keep_case(old, new):
        if old[:1].isupper():
            return new[:1].upper() + new[1:]
        return new

    def apply_replace(self, inst: Instruction):
        sc, scw, sn = self.parse_attr(inst.source)
        tc, tcw, tn = self.parse_attr(inst.target)
        if sc and tc and sn and (tn is None or tn == sn):
            self.replace_color(inst, sc, tc, tcw, sn)
            if inst.matches:
                return
        self.replace_phrase(inst)

    def replace_color(self, inst, sc, tc, tcw, noun):
        """ Ändert nur das Farbwort vor dem Substantiv: "long red hair" -> "long blonde hair" """
        new_color = tcw if self.lang == 'de' else (tcw if tcw in COLORS[tc]['en'] else tc)
        colors = _alt([re.escape(x).replace(r'\ ', r'[\s-]+') for x in COLORS[sc]['en']] + _de_color_regex(sc))
        nouns = _alt([re.escape(x) for x in NOUNS[noun]])
        pats = [
            # Farbe [0-2 Adjektive] Substantiv
            re.compile(r"(?<![\w-])(?P<c>" + colors + r")(?=(?:[\s-]+[\wäöüß]+){0,2}?[\s-]+" + nouns + r"(?![\w]))",
                       re.I),
            # Substantiv [ist/:] Farbe  ("hair: red", "Haare rot")
            re.compile(r"(?<![\w])" + nouns + r"\s*(?::|is|are|ist|sind)?\s+(?P<c>" + colors + r")(?![\w])", re.I),
        ]
        if noun == 'hair':
            pats.append(re.compile(r"(?<![\w-])(?P<c>" + colors + r")(?=-haired)", re.I))
            holder = COLORS[sc].get('holder')
            if holder and ' ' not in holder and '-' not in holder:
                for m in re.finditer(r"(?<![\w])" + holder + r"s?(?![\w])", self.prompt, re.I):
                    new = self.keep_case(m.group(0), COLORS[tc].get('holder') or f"{new_color}-haired")
                    self.add_edit(Edit(m.start(), m.end(), new,
                                       f"»{m.group(0)}« = {sc} hair → {tc} (Zeile {inst.line_no})", 'rep', inst))
        for rx in pats:
            for m in rx.finditer(self.prompt):
                s, e = m.start('c'), m.end('c')
                old = m.group('c')
                syn = '' if old.lower() == sc else f" (»{old}« gilt als {sc})"
                self.add_edit(Edit(s, e, self.keep_case(old, self.inflect(old, sc, new_color, tc)),
                                   f"Farbe von {noun}: {sc} → {tc}{syn} (Zeile {inst.line_no})", 'rep', inst))
        if inst.matches:
            inst.info = f"Farbe {sc} → {tc} bei »{noun}« (Synonyme wie ginger, auburn werden erkannt)"

    def replace_phrase(self, inst):
        regex = self.phrase_regex(inst.source)
        if regex is None:
            return
        new = self.target_text(inst.target)
        for m in re.finditer(regex, self.prompt, re.I):
            self.add_edit(Edit(m.start(), m.end(), self.keep_case(m.group(0), new),
                               f"»{m.group(0)}« → »{new}« (Zeile {inst.line_no})", 'rep', inst))
        if inst.matches:
            inst.info = f"Wörtlich ersetzt durch »{new}«"
        else:
            inst.info = "Nicht im Prompt gefunden"
            inst.suggestion = self.suggest(inst.source)

    # ---------- Hinzufügen ----------

    def apply_add(self, inst):
        text = self.target_text(inst.target)
        if re.search(self.phrase_regex(text) or r"$^", self.prompt, re.I):
            inst.info = "Bereits im Prompt enthalten"
            return
        end = len(self.prompt.rstrip())
        sep = ', ' if self.prompt.strip() and not self.prompt.rstrip().endswith(',') else ''
        self.add_edit(Edit(end, end, sep + text, f"Hinzugefügt: »{text}« (Zeile {inst.line_no})", 'ins', inst))
        inst.info = "Am Ende angehängt"

    # ---------- Markieren ----------

    def describe(self, phrase):
        """ Beschreibung einer Eingabe: Wortliste (Beschreibung, Aliase, Begriffe) oder Übersetzung/Synonyme je Wort """
        key, group = self.find_expansion(phrase)
        if group:
            return dict(key=key, label=group['label'], text=group.get('description', ''), aliases=group['aliases'],
                        terms=list(dict.fromkeys(_pretty(t) for t in group['terms'])))
        words = []
        for w in _words(phrase):
            color, noun = _color_of(w), _noun_of(w)
            if color:
                alts, kind = COLORS[color]['en'] + COLORS[color]['de'], f"Farbe {color}"
            elif noun:
                alts, kind = NOUNS[noun], f"Substantiv {noun}"
            else:
                alts = DE_EN.get(w, [])
                kind = "Übersetzung" if alts else "wörtlich"
            words.append(dict(word=w, kind=kind, alternatives=[a for a in alts if a != w]))
        return dict(key=None, label=phrase, text="Keine Wortliste, Suche wörtlich mit Übersetzung und Synonymen",
                    words=words)

    @staticmethod
    def describe_text(desc):
        if desc.get('key'):
            return f"{desc['label']}: {desc['text']}. Aliase: {', '.join(desc['aliases'])}. " \
                   f"Begriffe: {', '.join(desc['terms'])}"
        parts = [f"{w['word']} ({w['kind']}" + (': ' + ', '.join(w['alternatives']) if w['alternatives'] else '') + ')'
                 for w in desc.get('words', [])]
        return f"{desc['text']}. " + '; '.join(parts)

    def apply_highlight(self, inst):
        """ Markiert Fundstellen ohne den Prompt zu ändern, mit Beschreibung der Wortliste im Tooltip """
        desc = self.describe(inst.source)
        inst.description = desc
        core = self.phrase_core(inst.source)
        if core is None:
            inst.info = "Leere Eingabe"
            return
        if desc.get('key'):
            group = EXPANSIONS[desc['key']]
            inst.expansion = dict(key=desc['key'], label=group['label'], terms=group['terms'])
            core = _alt(group['terms'] + [core])
            core = r"(?<![\w])" + core + r"(?![\w])" + STYLE_SUFFIX
        else:
            core = r"(?<![\w])" + core + r"(?![\w])"
        for m in re.finditer(core, self.prompt, re.I):
            if self.in_angle(m.start(), m.end()):
                continue
            self.highlights.append(dict(start=m.start(), end=m.end(), line=inst.line_no,
                                        tip=f"»{m.group(0)}« gehört zu »{desc['label']}« (Zeile {inst.line_no}). "
                                            + self.describe_text(desc)))
            inst.matches += 1
        inst.info = self.describe_text(desc)
        if not inst.matches:
            inst.suggestion = self.suggest(inst.source)

    # ---------- Ausführen ----------

    def run(self):
        self.pending_whole = []
        if self.strip_angle:
            self.apply_strip_angle()
        for inst in self.instructions:
            if inst.kind == 'remove':
                self.apply_remove(inst)
        self.flush_whole_chunks()
        for inst in self.instructions:
            if inst.kind == 'replace':
                self.apply_replace(inst)
            elif inst.kind == 'add':
                self.apply_add(inst)
            elif inst.kind == 'highlight':
                self.apply_highlight(inst)
        for inst in self.instructions:
            if inst.kind in ('remove', 'replace', 'add') and inst.matches == 0 and not inst.info.startswith('Bereits'):
                inst.info = (inst.info + ' — ' if inst.info else '') + 'keine Fundstelle'
        return self

    def eq_segments(self, a, b):
        """ Unveränderter Text a..b, aufgeteilt in Markierungen (t='hl') und normalen Text (t='eq') """
        segs, pos = [], a
        for h in sorted(self.highlights, key=lambda x: (x['start'], -x['end'])):
            s, e = max(h['start'], pos), min(h['end'], b)
            if e <= s:
                continue
            if s > pos:
                segs.append(dict(t='eq', text=self.prompt[pos:s]))
            segs.append(dict(t='hl', text=self.prompt[s:e], tip=h['tip'], line=h['line']))
            pos = e
        if pos < b:
            segs.append(dict(t='eq', text=self.prompt[pos:b]))
        return segs

    def segments(self):
        """ t: eq (gleich), hl (markiert, unverändert), del, ins, rep """
        segs, pos = [], 0
        for e in sorted(self.edits, key=lambda x: (x.start, x.end)):
            if e.start > pos:
                segs.extend(self.eq_segments(pos, e.start))
            old = self.prompt[e.start:e.end]
            if e.kind == 'del':
                segs.append(dict(t='del', old=old, new='', tip=e.reason, line=e.inst.line_no))
            elif e.kind == 'ins':
                segs.append(dict(t='ins', old='', new=e.new, tip=e.reason, line=e.inst.line_no))
            else:
                segs.append(dict(t='rep', old=old, new=e.new, tip=e.reason, line=e.inst.line_no))
            pos = max(pos, e.end)
        if pos < len(self.prompt):
            segs.extend(self.eq_segments(pos, len(self.prompt)))
        return segs

    def result(self):
        out = ''.join(s['text'] if s['t'] in ('eq', 'hl') else s['new'] for s in self.segments())
        out = re.sub(r"[ \t]{2,}", ' ', out)
        out = re.sub(r"\s+,", ',', out)
        out = re.sub(r",\s*,+", ',', out)
        return out.strip().strip(',').strip()

    def to_dict(self):
        return dict(success=True, result=self.result(), segments=self.segments(), language=self.lang,
                    changes=len(self.edits), highlights=len(self.highlights), instructions=[i.to_dict() for i in self.instructions],
                    warnings=self.warnings, strip_angle=self.strip_angle,
                    auto_removed=self.option_inst.matches)


def edit_prompt(prompt, changes, strip_angle=True):
    """
    :param prompt: Bild-Prompt
    :param changes: Anweisungen, eine pro Zeile (str, list oder JSON-Liste)
    :param strip_angle: alles in <...> (LoRA-Aufrufe) automatisch entfernen
    :return: dict(result, segments, instructions, warnings, changes, highlights, auto_removed)
    """
    return PromptEdit(prompt or '', parse_instructions(changes), strip_angle=strip_angle).run().to_dict()


class PromptEditor(r.BaseJobExecutor):
    """
    Entfernt/ersetzt Teile eines Bild-Prompts regelbasiert, ohne LLM.

    pe.execute({'prompt': 'a knight, comic style, red hair', 'changes': 'comic style entfernen\\nrot haar -> blonde haare'})

    Seite: Live-Vorschau mit markierten Änderungen, Tooltips mit Begründung, Kopieren in die Zwischenablage.
    """

    TAGS = [r.TAG.TEXT_EXTRACTOR, 'prompt']

    EXAMPLES = [
        dict(title="Comic → neutral",
             prompt="portrait of a young woman with long red hair, comic book style, (halftone dots:1.2), "
                    "bold black outlines, cel shading, city street at night, people in the background, vibrant colors, "
                    "<lora:comicStyle_v2:0.8>, <lora:detailTweaker:0.5>",
             changes="comic style entfernen\nrot haar -> blonde haare\nleute im hintergrund entfernen"),
        dict(title="Fließtext",
             prompt="A knight with ginger hair drawn in a comic book style with bold outlines and halftone dots, riding a "
                    "horse through a crowded market, crowd of people in the background, sunset",
             changes="- comic\nrote haare -> blonde haare\nentferne leute im hintergrund\n+ photorealistic, 35mm"),
        dict(title="Deutsch",
             prompt="Eine Frau mit roten Haaren im Comic-Stil, Halftone, Stadt bei Nacht, Passanten im Hintergrund",
             changes="comic style entfernen\nrot haar -> blonde haare\npassanten entfernen"),
    ]

    def __init__(self, type='prompt_edit', strip_angle=True):
        """ :param strip_angle: Standard für Anfragen ohne strip_angle: <...> (LoRA) automatisch entfernen """
        super().__init__()
        self.type = type
        self.strip_angle = to_bool(strip_angle)
        self.define_vars('strip_angle')
        self.define_sig(d.Param('prompt'), d.Param('changes')) \
            .set_description("Entfernt/ersetzt Teile eines Prompts, Anweisungen zeilenweise "
                             "(»comic style entfernen«, »rot haar -> blonde haare«, »+ film grain«, »? comic«), "
                             "optional strip_angle=0, um <lora:...> nicht automatisch zu entfernen") \
            .r(result='str', segments='list', instructions='list', changes='int')

    def edit(self, prompt, changes, strip_angle=None):
        return edit_prompt(prompt, changes, strip_angle=self.strip_angle if strip_angle is None else strip_angle)

    def execute(self, data):
        if 'prompt' in data:
            return self.edit(data.get('prompt', ''), data.get('changes', ''),
                             strip_angle=to_bool(data.get('strip_angle'), self.strip_angle))
        if data.get('op') == 'groups':
            return self.success(groups={k: dict(label=g['label'], aliases=g['aliases']) for k, g in EXPANSIONS.items()})
        return super().execute(data)

    def part_index(self, p: b.Page, params={}):
        p('<style>' + PAGE_CSS + '</style>')
        groups = [dict(key=k, label=g['label'], alias=g['aliases'][0],
                       terms=', '.join(re.sub(r"\(\?:|\)|\[\\s-\]\??|\\s\+|\?|\\d|\{[\d,]+\}", lambda m: ' ' if 's' in m.group(0) else '', t)
                                       .replace('|', '/') for t in g['terms'][:10]))
                  for k, g in EXPANSIONS.items()]
        cfg = dict(type=self.type, examples=self.EXAMPLES, groups=groups, strip_angle=self.strip_angle)
        p(PAGE_HTML)
        p('<script>{ const PE_CFG = ' + json.dumps(cfg) + ';\n' + PAGE_JS + '\n}</script>')


PAGE_CSS = r"""
.pe { --pe-del:#c0392b; --pe-del-bg:#c0392b1f; --pe-ins:#1e8449; --pe-ins-bg:#27ae6026; --pe-acc:#2e86de;
      --pe-line:#8884; --pe-soft:#8881; max-width:1200px; font-size:15px; }
.pe * { box-sizing:border-box; }
.pe h1 { margin:0 0 4px; font-size:1.5em; }
.pe .pe-sub { opacity:.7; margin:0 0 14px; }
.pe-grid { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:16px; }
@media (max-width: 860px) { .pe-grid { grid-template-columns: 1fr; } }
.pe-card { border:1px solid var(--pe-line); border-radius:10px; padding:12px; background:var(--pe-soft); }
.pe-label { display:flex; align-items:center; gap:6px; font-weight:600; margin:0 0 6px; }
.pe textarea { width:100%; font:13px/1.5 ui-monospace, Consolas, monospace; padding:8px; border-radius:6px;
               border:1px solid var(--pe-line); background:transparent; color:inherit; resize:vertical; }
.pe #pe-prompt { min-height:140px; } .pe #pe-changes { min-height:110px; }
.pe-row { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0; }
.pe-btn { cursor:pointer; border:1px solid var(--pe-line); background:transparent; color:inherit; border-radius:6px;
          padding:4px 10px; font:inherit; font-size:13px; }
.pe-btn:hover { border-color:var(--pe-acc); }
.pe-btn.pe-primary { background:var(--pe-acc); border-color:var(--pe-acc); color:#fff; font-weight:600; padding:6px 14px; }
.pe-btn.pe-ok { background:var(--pe-ins); border-color:var(--pe-ins); }
.pe-chip { cursor:pointer; border:1px dashed var(--pe-line); border-radius:999px; padding:2px 10px; font-size:12px;
           background:transparent; color:inherit; }
.pe-chip:hover { border-style:solid; border-color:var(--pe-acc); }
.pe-help { display:inline-flex; width:18px; height:18px; border-radius:50%; border:1px solid var(--pe-line);
           align-items:center; justify-content:center; font-size:11px; font-weight:700; cursor:help; opacity:.75; }
.pe-out { font:14px/1.7 ui-monospace, Consolas, monospace; white-space:pre-wrap; word-break:break-word; min-height:90px;
          padding:10px; border-radius:6px; border:1px solid var(--pe-line); }
.pe-out del { color:var(--pe-del); background:var(--pe-del-bg); text-decoration:line-through; border-radius:3px; padding:0 2px; }
.pe-out ins { color:var(--pe-ins); background:var(--pe-ins-bg); text-decoration:none; border-radius:3px; padding:0 2px; font-weight:600; }
.pe-out .pe-mark { cursor:help; }
.pe-out .pe-mark.pe-hl { outline:2px solid var(--pe-acc); }
.pe-out.pe-clean del { display:none; }
.pe-out mark { background:#f1c40f55; color:inherit; border-radius:3px; padding:0 2px; box-shadow:inset 0 -2px 0 #f1c40f; }
.pe-insts li.pe-hl-inst { border-left-color:#f1c40f; }
.pe-desc { font-size:12px; opacity:.75; margin-top:2px; }
.pe-out.pe-clean ins { color:inherit; background:transparent; font-weight:normal; }
.pe-stats { font-size:13px; opacity:.8; }
.pe-insts { list-style:none; margin:8px 0 0; padding:0; display:flex; flex-direction:column; gap:4px; }
.pe-insts li { display:flex; gap:8px; align-items:baseline; font-size:13px; padding:4px 8px; border-radius:6px;
               border-left:3px solid var(--pe-line); background:var(--pe-soft); cursor:default; }
.pe-insts li.pe-hit { border-left-color:var(--pe-ins); }
.pe-insts li.pe-miss { border-left-color:#e67e22; }
.pe-insts li.pe-bad { border-left-color:var(--pe-del); }
.pe-insts .pe-kind { font-weight:700; min-width:76px; }
.pe-insts .pe-n { margin-left:auto; font-variant-numeric:tabular-nums; opacity:.8; white-space:nowrap; }
.pe-sugg { cursor:pointer; text-decoration:underline dotted; }
.pe-warn { color:#e67e22; font-size:13px; }
.pe-legend { display:flex; gap:12px; font-size:12px; opacity:.85; flex-wrap:wrap; }
.pe-tip { position:fixed; z-index:200; max-width:340px; background:#222; color:#fff; font:12px/1.45 system-ui, sans-serif;
          padding:6px 9px; border-radius:6px; box-shadow:0 4px 14px #0006; pointer-events:none; opacity:0; transition:opacity .12s; }
.pe-tip.pe-show { opacity:1; }
.pe-toast { position:fixed; right:20px; bottom:20px; background:var(--pe-ins); color:#fff; padding:8px 14px; border-radius:8px;
            opacity:0; transform:translateY(8px); transition:.2s; pointer-events:none; z-index:100; }
.pe-toast.pe-show { opacity:1; transform:none; }
"""

PAGE_HTML = r"""
<div class="pe">
  <p class="pe-sub">Teile eines Prompts entfernen oder ersetzen, regelbasiert und ohne LLM. Die Vorschau aktualisiert sich beim Tippen.</p>
  <div class="pe-row"><span>Beispiele:</span><span id="pe-examples" class="pe-row" style="margin:0"></span></div>
  <div class="pe-grid">
    <div class="pe-card">
      <div class="pe-label">Prompt
        <span class="pe-help" tabindex="0" data-tip="Der Original-Prompt (Englisch oder Deutsch). Teile werden an Kommas getrennt; Gewichtungen wie (comic:1.2) bleiben erhalten, <lora:...> wird je nach Option entfernt.">?</span>
        <button class="pe-btn" id="pe-paste" style="margin-left:auto;font-weight:normal" data-tip="Ersetzt den Prompt durch den Inhalt der Zwischenablage. Die Änderungen bleiben stehen und werden sofort angewendet.">Einfügen</button>
      </div>
      <textarea id="pe-prompt" spellcheck="false" placeholder="z.B. portrait of a woman, red hair, comic style, halftone dots"></textarea>
      <div class="pe-label" style="margin-top:10px">Änderungen <span style="font-weight:normal;opacity:.7">(eine pro Zeile)</span>
        <span class="pe-help" tabindex="0" data-tip="Entfernen: »comic style entfernen«, »- leute im hintergrund«, »ohne text«. Ersetzen: »rot haar -> blonde haare«, »ersetze sunset durch night«. Hinzufügen: »+ film grain«. Markieren (ändert nichts, zeigt Aliase und Beschreibung): »? comic«, »markiere leute im hintergrund«. Zeilen mit # werden ignoriert.">?</span>
      </div>
      <textarea id="pe-changes" spellcheck="false" placeholder="comic style entfernen&#10;rot haar -> blonde haare"></textarea>
      <div class="pe-row"><span style="font-size:13px;opacity:.8"
           data-tip="Diese Eingaben werden automatisch um verwandte Begriffe erweitert. Klick fügt eine Entfernen-Zeile hinzu, Umschalt+Klick eine Markieren-Zeile.">Schnell entfernen:</span>
        <span id="pe-groups" class="pe-row" style="margin:0"></span></div>
      <ul id="pe-insts" class="pe-insts"></ul>
    </div>
    <div class="pe-card">
      <div class="pe-label">Ergebnis
        <span class="pe-help" tabindex="0" data-tip="Rot durchgestrichen = entfernt, grün = neu, gelb = markiert (unverändert). Über eine Markierung fahren zeigt die Begründung; ein Klick hebt die zugehörige Anweisung hervor.">?</span>
        <span id="pe-stats" class="pe-stats" style="margin-left:auto"></span>
      </div>
      <div id="pe-out" class="pe-out" aria-live="polite"></div>
      <div class="pe-row">
        <button class="pe-btn pe-primary" id="pe-copy" data-tip="Kopiert den bereinigten Prompt (ohne Markierungen) in die Zwischenablage — Strg+Enter">Kopieren</button>
        <button class="pe-btn" id="pe-copy-orig" data-tip="Kopiert den unveränderten Original-Prompt">Original kopieren</button>
        <button class="pe-btn" id="pe-apply" data-tip="Übernimmt das Ergebnis als neuen Prompt und leert die Änderungen — für weitere Runden">Als Prompt übernehmen</button>
        <label style="font-size:13px" data-tip="Entfernt alles in spitzen Klammern automatisch, z.B. LoRA-Aufrufe <lora:name:0.8>, <hypernet:…>, Embeddings. Gilt zusätzlich zu den Änderungen."><input type="checkbox" id="pe-angle"> &lt;…&gt; (LoRA) entfernen</label>
        <label style="font-size:13px;margin-left:auto" data-tip="Aus: nur das Ergebnis anzeigen, ohne entfernte Teile"><input type="checkbox" id="pe-diff" checked> Änderungen zeigen</label>
      </div>
      <div class="pe-legend"><span><del style="color:var(--pe-del)">entfernt</del></span><span><ins style="color:var(--pe-ins);text-decoration:none">neu</ins></span><span><mark style="background:#f1c40f55;color:inherit;border-radius:3px;padding:0 2px">markiert</mark></span><span id="pe-lang"></span></div>
      <div id="pe-warn" class="pe-warn"></div>
    </div>
  </div>
  <div id="pe-tip" class="pe-tip" role="tooltip"></div>
  <div id="pe-toast" class="pe-toast">In die Zwischenablage kopiert ✓</div>
</div>
"""

PAGE_JS = r"""
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const KIND = {remove: 'Entfernen', replace: 'Ersetzen', add: 'Hinzufügen', highlight: 'Markieren', unknown: '?'};
const STORE = 'pe_state_' + PE_CFG.type;
let timer = null, seq = 0, last = null;

function save() {
  try { localStorage.setItem(STORE, JSON.stringify({p: $('pe-prompt').value, c: $('pe-changes').value, a: $('pe-angle').checked})); } catch (e) {}
}
function load() {
  try { const s = JSON.parse(localStorage.getItem(STORE) || 'null'); if (s) { $('pe-prompt').value = s.p; $('pe-changes').value = s.c; if (typeof s.a === 'boolean') $('pe-angle').checked = s.a; return true; } } catch (e) {}
  return false;
}
function schedule() { clearTimeout(timer); timer = setTimeout(run, 220); save(); }

function run() {
  const my = ++seq;
  $('pe-stats').textContent = '…';
  fetch('/', {method: 'POST', body: new URLSearchParams({type: PE_CFG.type, prompt: $('pe-prompt').value, changes: $('pe-changes').value, strip_angle: $('pe-angle').checked ? '1' : '0'})})
    .then((r) => r.json())
    .then((d) => { if (my === seq) render(d); })
    .catch((e) => { if (my === seq) { $('pe-stats').textContent = 'Fehler'; $('pe-warn').textContent = String(e); } });
}

function render(d) {
  if (!d || d.success === false) { $('pe-stats').textContent = 'Fehler'; $('pe-warn').textContent = (d && d.message) || ''; return; }
  last = d;
  let h = '';
  for (const s of d.segments) {
    if (s.t === 'eq') { h += esc(s.text); continue; }
    if (s.t === 'hl') { h += '<mark class="pe-mark" data-line="' + s.line + '" data-tip="' + esc(s.tip) + '">' + esc(s.text) + '</mark>'; continue; }
    const tip = esc(s.tip);
    if (s.old) h += '<del class="pe-mark" data-line="' + s.line + '" data-tip="' + tip + '">' + esc(s.old) + '</del>';
    if (s.new) h += '<ins class="pe-mark" data-line="' + s.line + '" data-tip="' + tip + '">' + esc(s.new) + '</ins>';
  }
  const out = $('pe-out');
  out.innerHTML = h || '<span style="opacity:.5">Noch kein Prompt</span>';
  out.dataset.clean = d.result;
  const n = d.changes;
  $('pe-stats').textContent = (n === 0 ? 'keine Änderung' : (n === 1 ? '1 Änderung' : n + ' Änderungen')) +
    (d.highlights ? ', ' + d.highlights + ' markiert' : '') + (d.auto_removed ? ' (davon ' + d.auto_removed + '× <…>)' : '');
  $('pe-lang').textContent = 'Prompt-Sprache: ' + (d.language === 'de' ? 'Deutsch' : 'Englisch');
  $('pe-warn').innerHTML = (d.warnings || []).map(esc).join('<br>');
  $('pe-insts').innerHTML = d.instructions.map((i) => {
    const cls = i.kind === 'unknown' ? 'pe-bad' : (!i.matches ? 'pe-miss' : (i.kind === 'highlight' ? 'pe-hl-inst' : 'pe-hit'));
    const dsc = i.kind === 'highlight' && i.description ? '<div class="pe-desc">' + esc(i.description.key
        ? i.description.text + ' · Aliase: ' + i.description.aliases.join(', ')
        : i.description.words.map((w) => w.word + (w.alternatives.length ? ' → ' + w.alternatives.join('/') : '')).join(' · ')) + '</div>' : '';
    let tip = i.info || '';
    if (i.expansion) tip += ' — u.a.: ' + PE_CFG.groups.filter((g) => g.key === i.expansion.key).map((g) => g.terms)[0];
    let what = i.kind === 'replace' ? esc(i.source) + ' → ' + esc(i.target) : esc(i.source || i.target || i.raw);
    let sugg = i.suggestion ? ' <span class="pe-sugg" data-line="' + i.line + '" data-s="' + esc(i.suggestion) + '" data-tip="Klick: Anweisung auf diesen Teil des Prompts ändern">Meintest du »' + esc(i.suggestion) + '«?</span>' : '';
    return '<li class="' + cls + '" data-line="' + i.line + '" data-tip="' + esc(tip) + '"><span class="pe-kind">' + KIND[i.kind] + '</span>' +
           '<span>' + what + sugg + dsc + '</span><span class="pe-n">' + (i.kind === 'unknown' ? 'nicht verstanden' : (i.matches ? i.matches + '×' : 'nicht gefunden')) + '</span></li>';
  }).join('');
}

function toast(msg) {
  const t = $('pe-toast'); t.textContent = msg; t.classList.add('pe-show');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('pe-show'), 1400);
}
function copyText(text, btn) {
  const done = () => {
    toast('In die Zwischenablage kopiert ✓');
    if (btn) { const old = btn.textContent; btn.textContent = 'Kopiert ✓'; btn.classList.add('pe-ok');
      setTimeout(() => { btn.textContent = old; btn.classList.remove('pe-ok'); }, 1300); }
  };
  const fallback = () => {
    const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { toast('Kopieren nicht möglich'); }
    ta.remove();
  };
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
}
function addLine(line) {
  const ta = $('pe-changes');
  const v = ta.value.replace(/\s+$/, '');
  if (v.split('\n').some((l) => l.trim().toLowerCase() === line.toLowerCase())) return;
  ta.value = (v ? v + '\n' : '') + line; schedule();
}

$('pe-examples').innerHTML = PE_CFG.examples.map((e, i) => '<button class="pe-chip" data-ex="' + i + '" data-tip="Lädt Beispiel-Prompt und -Änderungen">' + esc(e.title) + '</button>').join('');
$('pe-groups').innerHTML = PE_CFG.groups.map((g) => '<button class="pe-chip" data-alias="' + esc(g.alias) + '" data-tip="' + esc('Erkennt u.a.: ' + g.terms) + '">' + esc(g.label) + '</button>').join('');

document.querySelector('.pe').addEventListener('click', (e) => {
  const ex = e.target.closest('[data-ex]');
  if (ex) { const x = PE_CFG.examples[+ex.dataset.ex]; $('pe-prompt').value = x.prompt; $('pe-changes').value = x.changes; schedule(); return; }
  const g = e.target.closest('[data-alias]');
  if (g) { addLine(e.shiftKey ? '? ' + g.dataset.alias : g.dataset.alias + ' entfernen'); return; }
  const s = e.target.closest('.pe-sugg');
  if (s) {
    const lines = $('pe-changes').value.split('\n'); const n = +s.dataset.line - 1;
    const inst = last.instructions.find((i) => i.line === +s.dataset.line);
    lines[n] = inst.kind === 'replace' ? s.dataset.s + ' -> ' + inst.target
      : (inst.kind === 'highlight' ? '? ' + s.dataset.s : s.dataset.s + ' entfernen');
    $('pe-changes').value = lines.join('\n'); schedule(); return;
  }
  const m = e.target.closest('.pe-mark, .pe-insts li');
  if (m) {
    const line = m.dataset.line;
    document.querySelectorAll('.pe-mark').forEach((x) => x.classList.toggle('pe-hl', x.dataset.line === line));
    if (m.tagName === 'LI' || m.classList.contains('pe-mark')) {
      const ta = $('pe-changes'), lines = ta.value.split('\n'); let pos = 0;
      for (let i = 0; i < +line - 1; i++) pos += lines[i].length + 1;
      if (m.classList.contains('pe-mark') && +line > 0) { ta.focus(); ta.setSelectionRange(pos, pos + (lines[+line - 1] || '').length); }
    }
  }
});
$('pe-prompt').addEventListener('input', schedule);
$('pe-changes').addEventListener('input', schedule);
$('pe-copy').addEventListener('click', (e) => copyText($('pe-out').dataset.clean || '', e.currentTarget));
$('pe-copy-orig').addEventListener('click', (e) => copyText($('pe-prompt').value, e.currentTarget));
$('pe-paste').addEventListener('click', () => {
  const ta = $('pe-prompt');
  // Ohne HTTPS (z.B. LAN-IP) oder ohne Berechtigung: Feld markieren, Nutzer fügt mit Strg+V ein
  const manual = () => { ta.focus(); ta.select(); toast('Strg+V drücken, um einzufügen'); };
  if (!navigator.clipboard || !navigator.clipboard.readText || !window.isSecureContext) { manual(); return; }
  navigator.clipboard.readText().then((text) => {
    if (!text.trim()) { toast('Zwischenablage ist leer'); return; }
    ta.value = text.trim(); schedule(); toast('Prompt eingefügt');
  }, manual);
});
$('pe-apply').addEventListener('click', () => { if (!last) return; $('pe-prompt').value = last.result; $('pe-changes').value = ''; schedule(); toast('Ergebnis übernommen'); });
$('pe-angle').addEventListener('change', schedule);
$('pe-diff').addEventListener('change', (e) => $('pe-out').classList.toggle('pe-clean', !e.target.checked));
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter' && document.querySelector('.pe').contains(document.activeElement)) { e.preventDefault(); $('pe-copy').click(); }
});
// Tooltips: ein Element, an der Maus/Fokus ausgerichtet und im Fenster gehalten
const tipEl = $('pe-tip');
function showTip(el) {
  const text = el.getAttribute('data-tip'); if (!text) return;
  tipEl.textContent = text; tipEl.classList.add('pe-show');
  const r = el.getBoundingClientRect(), t = tipEl.getBoundingClientRect(), m = 8;
  let x = Math.min(Math.max(m, r.left), window.innerWidth - t.width - m);
  let y = r.bottom + 6; if (y + t.height > window.innerHeight - m) y = r.top - t.height - 6;
  tipEl.style.left = x + 'px'; tipEl.style.top = Math.max(m, y) + 'px';
}
const hideTip = () => tipEl.classList.remove('pe-show');
const root = document.querySelector('.pe');
root.addEventListener('mouseover', (e) => { const el = e.target.closest('[data-tip]'); if (el) showTip(el); else hideTip(); });
root.addEventListener('mouseleave', hideTip);
root.addEventListener('focusin', (e) => { const el = e.target.closest('[data-tip]'); if (el) showTip(el); });
root.addEventListener('focusout', hideTip);
root.addEventListener('scroll', hideTip, true);
$('pe-angle').checked = PE_CFG.strip_angle;
if (!load()) { $('pe-prompt').value = PE_CFG.examples[0].prompt; $('pe-changes').value = PE_CFG.examples[0].changes; }
run();
"""
