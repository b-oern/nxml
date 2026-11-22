from nxml.prompt_edit import edit_prompt, parse_instructions, split_chunks


def res(prompt, changes):
    return edit_prompt(prompt, changes)['result']


def test_parse_instructions():
    kinds = [(i.kind, i.source, i.target) for i in parse_instructions(
        "comic style entfernen\n- crowd\nrot haar -> blonde haare\nersetze sunset durch night\n+ film grain\n# x\nblabla")]
    assert kinds == [('remove', 'comic style', ''), ('remove', 'crowd', ''), ('replace', 'rot haar', 'blonde haare'),
                     ('replace', 'sunset', 'night'), ('add', '', 'film grain'), ('unknown', '', '')]


def test_changes_as_list_or_json():
    expected = "a woman, blonde hair"
    assert res("a woman, red hair, comic style", ["comic style entfernen", "rot haar -> blonde haare"]) == expected
    assert res("a woman, red hair, comic style", '["comic style entfernen", "rot haar -> blonde haare"]') == expected
    assert res("a woman, red hair, comic style", ("comic style entfernen\nrot haar -> blonde haare",)) == expected


def test_parse_highlight():
    kinds = [(i.kind, i.source) for i in parse_instructions("? comic\nmarkiere leute im hintergrund\nred hair hervorheben")]
    assert kinds == [('highlight', 'comic'), ('highlight', 'leute im hintergrund'), ('highlight', 'red hair')]


def test_highlight_group_keeps_prompt():
    p = "a woman, comic style, (halftone dots:1.2), crowd, night"
    d = edit_prompt(p, "? comic")
    assert d['result'] == p and d['changes'] == 0 and d['highlights'] == 2
    hl = [s for s in d['segments'] if s['t'] == 'hl']
    assert [s['text'] for s in hl] == ['comic style', 'halftone dots']
    assert 'Comic-Look' in hl[0]['tip'] and 'Aliase' in hl[0]['tip']
    desc = d['instructions'][0]['description']
    assert desc['key'] == 'comic' and 'cartoon' in desc['aliases'] and desc['text']


def test_highlight_free_phrase_with_synonyms():
    d = edit_prompt("woman with long ginger hair", "markiere rote haare")
    assert [s['text'] for s in d['segments'] if s['t'] == 'hl'] == ['ginger hair']
    words = d['instructions'][0]['description']['words']
    assert words[0]['kind'] == 'Farbe red' and 'ginger' in words[0]['alternatives']


def test_highlight_with_edits():
    d = edit_prompt("a woman, red hair, comic style, crowd", "comic entfernen\n? leute im hintergrund")
    assert d['result'] == "a woman, red hair, crowd"
    assert [s['text'] for s in d['segments'] if s['t'] == 'hl'] == ['crowd']


def test_strip_angle_default():
    d = edit_prompt("<lora:comic:0.8>, a woman, (<lora:x:1>:1.2), night <hypernet:y:0.5>, <lora:z:1>", "")
    assert d['result'] == "a woman, night"
    assert d['auto_removed'] == 4 and d['strip_angle'] is True
    assert all('Automatisch entfernt' in s['tip'] for s in d['segments'] if s['t'] == 'del')


def test_strip_angle_off():
    p = "a woman, <lora:comic:0.8>, comic style"
    d = edit_prompt(p, "- comic\n? comic", strip_angle=False)
    assert d['result'] == "a woman, <lora:comic:0.8>"
    assert [s['text'] for s in d['segments'] if s['t'] == 'hl'] == []


def test_strip_angle_with_other_edits():
    assert res("a woman, red hair, <lora:a:1>, comic style, <lora:b:1>", "- comic\nrot haar -> blonde haare") == \
           "a woman, blonde hair"


def test_runner_strip_angle_param():
    from nxml.prompt_edit import PromptEditor
    pe = PromptEditor()
    assert pe.execute({'prompt': 'a cat, <lora:x:1>', 'changes': ''})['result'] == 'a cat'
    assert pe.execute({'prompt': 'a cat, <lora:x:1>', 'changes': '', 'strip_angle': '0'})['result'] == 'a cat, <lora:x:1>'
    assert PromptEditor(strip_angle=False).execute({'prompt': 'a cat, <lora:x:1>'})['result'] == 'a cat, <lora:x:1>'


def test_split_chunks_keeps_weights_and_lora():
    chunks = [c.text for c in split_chunks("a, (b, c:1.2), <lora:x:0.5>, d")]
    assert chunks == ['a', '(b, c:1.2)', '<lora:x:0.5>', 'd']


def test_remove_comic_tags():
    assert res("a woman, comic style, (halftone dots:1.2), bold black outlines, cel shading, night",
               "comic style entfernen") == "a woman, night"


def test_remove_first_chunks():
    assert res("comic style, halftone, a woman, night", "comic style entfernen") == "a woman, night"


def test_remove_inline_free_text():
    assert res("a knight drawn in a comic book style with bold outlines and halftone dots, riding a horse",
               "- comic") == "a knight, riding a horse"


def test_remove_typo_alias():
    assert res("a cat, comic style", "comix entfernen") == "a cat"


def test_remove_people_background():
    assert res("street, people in the background, crowd, night", "leute im hintergrund entfernen") == "street, night"


def test_remove_free_phrase():
    assert res("a cat, watermark, text, night", "ohne wasserzeichen") == "a cat, text, night"


def test_replace_hair_color_with_synonyms():
    assert res("woman with long ginger hair, red dress", "rot haar -> blonde haare") == \
           "woman with long blonde hair, red dress"
    assert res("red-haired girl, redhead", "rote haare -> blonde haare") == "blonde-haired girl, blonde"


def test_replace_german_inflection():
    assert res("Eine Frau mit roten Haaren, Nacht", "rot haar -> blonde haare") == "Eine Frau mit blonden Haaren, Nacht"


def test_replace_phrase_and_case():
    assert res("Sunset over the sea", "sonnenuntergang -> night") == "Night over the sea"


def test_add():
    assert res("a cat", "+ film grain") == "a cat, film grain"
    d = edit_prompt("a cat, film grain", "+ film grain")
    assert d['changes'] == 0


def test_not_found_suggestion():
    d = edit_prompt("a cat, golden retriever", "golden retrievr entfernen")
    assert d['instructions'][0]['matches'] == 0
    assert d['instructions'][0]['suggestion'] == 'golden retriever'


def test_segments_rebuild_result():
    d = edit_prompt("a woman, red hair, comic style", "comic style entfernen\nrot haar -> blonde haare")
    kinds = [s['t'] for s in d['segments']]
    assert 'del' in kinds and 'rep' in kinds
    assert all(s.get('tip') for s in d['segments'] if s['t'] != 'eq')
