"""English-letters-only text for the Shorts runner.

The speech may be Hindi (or Hinglish); what the viewer reads (captions, the
hook, the title, the description, the hashtags) is written in Latin letters
only. Captions and the hook are romanised word by word, so a caption still sits
on the word that was said ("bahut ganda maar diya"); the title and description
are rewritten in English for the channel's copy.

Gemini does the romanising when a client and key are there (natural Hinglish
spellings); every answer is checked in code (same number of words, Latin
letters only) and anything that fails the check, or any call that fails, falls
back to ``transliterate``: a fixed Devanagari table, so the result is always
Latin. Letters of other scripts that the table does not know are dropped.
"""
import json
import os
import re
import unicodedata

DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_TOKENS_PER_CALL = 400

# --- Latin check ---------------------------------------------------------------

def _is_latin_letter(ch):
    return "LATIN" in unicodedata.name(ch, "")


_SCRIPTLESS_MARKS = ("COMBINING", "VARIATION SELECTOR", "ZERO WIDTH")


def _script_char(ch):
    """A letter, or a mark that belongs to a script (a Devanagari vowel sign);
    emoji variation selectors and generic combining marks are not."""
    if ch.isalpha():
        return True
    if unicodedata.category(ch).startswith("M"):
        return not unicodedata.name(ch, "").startswith(_SCRIPTLESS_MARKS)
    return False


def is_latin(text):
    """True when every letter in ``text`` is a Latin letter (digits, marks of
    punctuation, spaces and emoji are fine)."""
    return all(_is_latin_letter(ch) for ch in str(text or "") if _script_char(ch))


# --- rule-based Devanagari -> Latin -------------------------------------------

_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
           "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e",
           "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
_CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
               "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
               "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
               "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
               "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
               "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh",
               "फ़": "f", "य़": "y", "ऱ": "r", "ऴ": "l", "ऩ": "n"}
_NUKTA = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़", "फ": "फ़", "य": "य़"}
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_VIRAMA, _NUKTA_SIGN = "्", "़"
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
_DANDA = {"।": ".", "॥": "."}


def transliterate(text):
    """Devanagari in ``text`` to plain Hinglish-style Latin letters; Latin
    text passes through; letters of any other script are dropped. A
    consonant's inherent "a" is dropped at the end of a word ("मार" ->
    "mar", "दिया" -> "diya")."""
    text = unicodedata.normalize("NFC", str(text or ""))
    # Base consonant + nukta sign -> the precomposed consonant.
    chars = []
    for ch in text:
        if ch == _NUKTA_SIGN and chars and chars[-1] in _NUKTA:
            chars[-1] = _NUKTA[chars[-1]]
        else:
            chars.append(ch)
    out = []
    i, n = 0, len(chars)
    while i < n:
        ch = chars[i]
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                i += 2
                continue
            if nxt == _VIRAMA:
                i += 2
                continue
            # Inherent "a", except at the end of a word.
            end_of_word = not nxt or not (nxt in _CONSONANTS or nxt in _SIGNS
                                          or nxt in _VOWELS)
            if not end_of_word or nxt in _SIGNS:
                out.append("a")
            i += 1
            continue
        if ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch in _DANDA:
            out.append(_DANDA[ch])
        elif ch in _MATRAS or ch in (_VIRAMA, _NUKTA_SIGN):
            out.append(_MATRAS.get(ch, ""))
        elif ch.isalpha() and not _is_latin_letter(ch):
            pass  # another script: no table for it
        elif unicodedata.category(ch).startswith("M") and not _is_latin_letter(ch):
            pass
        else:
            out.append(ch)
        i += 1
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


# --- Gemini --------------------------------------------------------------------

def _client():
    if not os.environ.get("GEMINI_API_KEY"):
        try:  # the key lives in .env, like the rest of the config
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except ImportError:
            pass
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


def _ask_json(client, prompt, model=None):
    from google.genai import types as genai_types
    import gemini_worker
    response = client.models.generate_content(
        model=model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        contents=[prompt],
        config=genai_types.GenerateContentConfig(response_mime_type="application/json"))
    gemini_worker.raise_if_blocked(response)
    return gemini_worker._parse_json_response_text(gemini_worker._get_response_text(response))


_ROMANISE_PROMPT = """You romanise spoken Hindi/Hinglish for YouTube Shorts captions.
Write each token below in Latin letters the way Indian viewers type Hinglish
(e.g. "bahut", "kya", "bhai", "maar diya", "chal"). Do not translate. Keep
English words, names and numbers as they are. No diacritics, no Devanagari.
Return JSON {"words": [...]} with exactly %d strings, one per token, same order.
Tokens (JSON): %s"""


class Latinizer:
    """Romanises caption words and short texts, and rewrites copy in English.
    ``client``: a google-genai client (None: made from GEMINI_API_KEY; False:
    never call Gemini, rule-based only)."""

    def __init__(self, client=None, model=None, say=None):
        self.client = _client() if client is None else (client or None)
        self.model = model
        self.say = say or (lambda m: None)

    def _gemini_words(self, tokens):
        if not self.client or not tokens:
            return None
        out = []
        for k in range(0, len(tokens), MAX_TOKENS_PER_CALL):
            part = tokens[k:k + MAX_TOKENS_PER_CALL]
            try:
                data = _ask_json(self.client, _ROMANISE_PROMPT % (
                    len(part), json.dumps(part, ensure_ascii=False)), self.model)
            except Exception as e:
                self.say(f"latin: Gemini romanising failed ({type(e).__name__}); "
                         f"using the table")
                return None
            words = data.get("words") if isinstance(data, dict) else None
            if not isinstance(words, list) or len(words) != len(part):
                self.say("latin: Gemini returned a different word count; using the table")
                return None
            out.extend(words)
        return out

    def words(self, tokens):
        """Each token in Latin letters, one for one. Latin tokens are kept."""
        tokens = [str(t or "") for t in tokens]
        todo = [i for i, t in enumerate(tokens) if not is_latin(t)]
        if not todo:
            return list(tokens)
        got = self._gemini_words([tokens[i] for i in todo]) or [None] * len(todo)
        out = list(tokens)
        for i, g in zip(todo, got):
            g = str(g).strip() if isinstance(g, str) else ""
            out[i] = g if g and is_latin(g) and len(g) <= 4 * len(tokens[i]) + 8 \
                else transliterate(tokens[i])
        return out

    def caption_words(self, words):
        """Caption word dicts ({"word", "start", "end"}) with Latin text;
        a word that comes out empty is dropped."""
        words = [w for w in (words or []) if isinstance(w, dict)]
        texts = self.words([str(w.get("word") or "").strip() for w in words])
        out = []
        for w, t in zip(words, texts):
            raw = str(w.get("word") or "")
            # Whisper words carry their leading space (" bahut"); the caption
            # builder separates words by it, so keep the original spacing.
            if t.strip():
                out.append(dict(w, word=raw[:len(raw) - len(raw.lstrip())] + t.strip()))
        return out

    def text(self, s):
        """A short text (the hook) in Latin letters, word by word."""
        s = str(s or "")
        if is_latin(s):
            return s
        parts = s.split()
        return " ".join(p for p in self.words(parts) if p.strip())

    def english_copy(self, title, description, hashtags, context=""):
        """(title, description, hashtags) with every letter Latin: Gemini
        rewrites non-English copy in English; failing that, it is romanised.
        Copy that is already Latin is returned as it was."""
        title, description = str(title or ""), str(description or "")
        tags = [str(t) for t in (hashtags or [])]
        if is_latin(title) and is_latin(description) and all(is_latin(t) for t in tags):
            return title, description, tags
        got = None
        if self.client:
            prompt = (
                "Rewrite this YouTube Short's copy in natural English (Latin letters only, "
                "no Devanagari). Keep the meaning and the energy; do not invent facts; "
                "name a game only if the copy or context names it.\n"
                'Return JSON {"title": str (<= 90 chars), "description": str, '
                '"hashtags": [str, ...] (each starting with #, Latin letters)}.\n'
                f"Context: {context[:600]}\n"
                + json.dumps({"title": title, "description": description,
                              "hashtags": tags}, ensure_ascii=False))
            try:
                got = _ask_json(self.client, prompt, self.model)
            except Exception as e:
                self.say(f"latin: Gemini copy failed ({type(e).__name__}); romanising")
        if isinstance(got, dict):
            t, d = str(got.get("title") or "").strip(), str(got.get("description") or "")
            h = [str(x) for x in got.get("hashtags") or [] if isinstance(x, str)]
            if t and is_latin(t) and is_latin(d):
                title, description, tags = t, d.strip() or t, h or tags
        title = title if is_latin(title) else transliterate(title)
        description = description if is_latin(description) else transliterate(description)
        clean = []
        for t in tags:
            t = t if is_latin(t) else transliterate(t)
            t = "#" + re.sub(r"[^0-9A-Za-z_]", "", t.lstrip("#"))
            if len(t) > 1 and t.lower() not in (c.lower() for c in clean):
                clean.append(t)
        return title.strip(), description.strip(), clean
