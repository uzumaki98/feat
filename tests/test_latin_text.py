"""latin_text: Latin-letters-only captions, hook and copy. No network: Gemini
is a fake client or switched off."""
import json

import pytest

import latin_text as lt


class TestTransliterate:
    @pytest.mark.parametrize("src,want", [
        ("बहुत", "bahut"), ("गंदा", "ganda"), ("मार", "mar"), ("दिया", "diya"),
        ("भाई", "bhai"), ("क्या", "kya"), ("मुझे", "mujhe"), ("ज़िंदगी", "zindagi"),
        ("है।", "hai."), ("Hello भाई 123", "Hello bhai 123"), ("१२३", "123"),
    ])
    def test_devanagari(self, src, want):
        assert lt.transliterate(src) == want

    def test_nukta_as_separate_sign(self):
        assert lt.transliterate("ज़रा") == "zara"

    def test_other_scripts_are_dropped_and_latin_kept(self):
        assert lt.transliterate("gg مرحبا ok") == "gg ok"
        assert lt.transliterate("café #Shorts 🎥") == "café #Shorts 🎥"

    def test_is_latin(self):
        assert lt.is_latin("Bahut ganda maar diya! #GTA6 🎥 123")
        assert not lt.is_latin("बहुत")
        assert not lt.is_latin("ok ि")  # a lone Devanagari vowel sign
        assert lt.is_latin("") and lt.is_latin(None)

    def test_emoji_marks_are_not_a_script(self):
        assert lt.is_latin("Bike escape 🏍️ ❤️ 1️⃣")
        assert lt.is_latin("café")


class _Resp:
    def __init__(self, text):
        self.text = text
        self.candidates = []
        self.prompt_feedback = None


class _Models:
    def __init__(self, answers, calls):
        self.answers, self.calls = answers, calls

    def generate_content(self, model, contents, config):
        self.calls.append(contents[0])
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return _Resp(json.dumps(a))


class _Client:
    def __init__(self, *answers):
        self.calls = []
        self.models = _Models(list(answers), self.calls)


@pytest.fixture(autouse=True)
def _plain_parse(monkeypatch):
    import gemini_worker
    monkeypatch.setattr(gemini_worker, "raise_if_blocked", lambda r: None)
    monkeypatch.setattr(gemini_worker, "_get_response_text", lambda r: r.text)


class TestLatinizer:
    def test_latin_words_never_call_gemini(self):
        c = _Client()
        assert lt.Latinizer(client=c).words(["hi", "GTA", "6"]) == ["hi", "GTA", "6"]
        assert c.calls == []

    def test_gemini_spellings_used_only_for_the_non_latin_words(self):
        c = _Client({"words": ["bahut", "maar"]})
        got = lt.Latinizer(client=c).words(["bro", "बहुत", "मार"])
        assert got == ["bro", "bahut", "maar"]
        assert len(c.calls) == 1 and "बहुत" in c.calls[0] and "bro" not in c.calls[0]

    @pytest.mark.parametrize("answer", [
        {"words": ["bahut"]},                  # wrong count
        {"nope": 1},                           # wrong shape
        RuntimeError("503 unavailable"),      # call failed
    ])
    def test_bad_answers_fall_back_to_the_table(self, answer):
        got = lt.Latinizer(client=_Client(answer), say=lambda *_: None).words(["बहुत", "मार"])
        assert got == ["bahut", "mar"]

    def test_a_non_latin_answer_word_falls_back(self):
        c = _Client({"words": ["बहुत", "maar"]})
        assert lt.Latinizer(client=c).words(["बहुत", "मार"]) == ["bahut", "maar"]

    def test_no_client_is_rule_based(self):
        assert lt.Latinizer(client=False).text("बहुत गंदा मार दिया!") == "bahut ganda mar diya!"

    def test_caption_words_keep_timing_and_drop_empty(self):
        words = [{"word": "बहुत", "start": 1.0, "end": 1.3},
                 {"word": "مرحبا", "start": 1.3, "end": 1.6},
                 {"word": "bro", "start": 1.6, "end": 1.9}]
        got = lt.Latinizer(client=False).caption_words(words)
        assert got == [{"word": "bahut", "start": 1.0, "end": 1.3},
                       {"word": "bro", "start": 1.6, "end": 1.9}]

    def test_caption_words_keep_the_leading_space(self):
        c = _Client({"words": ["bahut", "maar"]})
        words = [{"word": " बहुत", "start": 0, "end": 1}, {"word": " मार", "start": 1, "end": 2},
                 {"word": " bro", "start": 2, "end": 3}]
        got = lt.Latinizer(client=c).caption_words(words)
        assert [w["word"] for w in got] == [" bahut", " maar", " bro"]
        assert "बहुत" in c.calls[0] and " बहुत" not in c.calls[0]

    def test_english_copy_rewrites_non_latin_copy(self):
        c = _Client({"title": "He wrecked them!", "description": "Pure chaos.",
                     "hashtags": ["#GTA6", "#गेमिंग", "#Funny"]})
        t, d, h = lt.Latinizer(client=c).english_copy(
            "बहुत गंदा मार दिया!", "लूट का मजा", ["#GTA5", "#Shorts"], "Source: GTA 6 stream")
        assert (t, d) == ("He wrecked them!", "Pure chaos.")
        assert h == ["#GTA6", "#geming", "#Funny"]
        assert "Source: GTA 6 stream" in c.calls[0]

    def test_english_copy_leaves_latin_copy_alone(self):
        c = _Client()
        assert lt.Latinizer(client=c).english_copy("Big win", "d", ["#A"]) == \
            ("Big win", "d", ["#A"])
        assert c.calls == []

    def test_english_copy_without_gemini_romanises(self):
        t, d, h = lt.Latinizer(client=False).english_copy("बहुत गंदा", "", ["#मार"])
        assert (t, d, h) == ("bahut ganda", "", ["#mar"])
        assert lt.is_latin(t + d + "".join(h))

    def test_english_copy_rejects_a_non_latin_answer(self):
        c = _Client({"title": "बहुत", "description": "x", "hashtags": []})
        t, _, _ = lt.Latinizer(client=c).english_copy("बहुत गंदा", "", [])
        assert t == "bahut ganda"
