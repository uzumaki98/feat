"""Checks that a rendered short is publishable: 9:16, the right length, both
streams, decodable end to end, sane loudness, and captions that line up with
the words.

Pure and importable (``check_short``) for shorts_pipeline.py, and a CLI:

    python verify_shorts.py clip.mp4 [--captions autosubs_<ts>_<hex>.ass] [--json]

Everything read from a job directory (caption sidecars, transcripts) is data:
it is parsed as numbers and text, never handed to a shell or a filter graph.
ffmpeg/ffprobe come from FFMPEG_BIN / FFPROBE_BIN or PATH; when one is missing
the checks that need it report ``ok: None`` instead of failing the run.
"""
import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys

TARGET_ASPECT = 9 / 16
ASPECT_TOLERANCE = 0.01
TARGET_LUFS = -14.0
LUFS_TOLERANCE = 3.0
CAPTION_MEDIAN_MAX = 0.25
CAPTION_OFFSET_MAX = 0.6
# A caption may end a hair past the last frame (centisecond rounding in ASS).
CAPTION_EDGE_SLACK = 0.05

_SUBPROCESS_TIMEOUT = 600


# --- binaries -----------------------------------------------------------------

def _binary(env_var, name):
    """Path of an executable from ``env_var`` or PATH, or None."""
    configured = (os.environ.get(env_var) or "").strip()
    if configured:
        if os.path.isfile(configured):
            return configured
        found = shutil.which(configured)
        if found:
            return found
    return shutil.which(name)


def ffmpeg_bin():
    return _binary("FFMPEG_BIN", "ffmpeg")


def ffprobe_bin():
    return _binary("FFPROBE_BIN", "ffprobe")


def _run(args, timeout=_SUBPROCESS_TIMEOUT):
    """(returncode, stdout, stderr) of an argv list; never via a shell."""
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)
    return (proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


def _num(value, default=None):
    """A finite float from anything, else ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _rate(text):
    """'30000/1001' -> 29.97; '0/0' or junk -> None."""
    if not isinstance(text, str) or not text:
        return None
    if "/" in text:
        num, _, den = text.partition("/")
        n, d = _num(num), _num(den)
        if n is None or not d:
            return None
        return n / d
    return _num(text)


# --- probe --------------------------------------------------------------------

def parse_probe(data):
    """The fields check_short needs from ffprobe's JSON (already parsed)."""
    streams = data.get("streams") if isinstance(data, dict) else None
    streams = streams if isinstance(streams, list) else []
    fmt = data.get("format") if isinstance(data, dict) else None
    fmt = fmt if isinstance(fmt, dict) else {}
    video = next((s for s in streams if isinstance(s, dict)
                  and s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if isinstance(s, dict)
                  and s.get("codec_type") == "audio"), None)
    duration = _num(fmt.get("duration"))
    if duration is None and video:
        duration = _num(video.get("duration"))
    fps = None
    if video:
        fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    width = int(_num(video.get("width"), 0)) if video else 0
    height = int(_num(video.get("height"), 0)) if video else 0
    if video:
        # A rotated phone recording reports landscape dimensions plus a
        # rotation; what plays is the swapped size.
        rotation = _num((video.get("tags") or {}).get("rotate"), 0) or 0
        for side in video.get("side_data_list") or []:
            if isinstance(side, dict) and "rotation" in side:
                rotation = _num(side.get("rotation"), rotation) or 0
        if int(abs(rotation)) % 180 == 90:
            width, height = height, width
    return {
        "width": width,
        "height": height,
        "duration": duration,
        "fps": round(fps, 3) if fps else None,
        "video_codec": video.get("codec_name") if video else None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "has_video": video is not None,
        "has_audio": audio is not None,
    }


def probe(path):
    """ffprobe summary of ``path``; raises RuntimeError when it cannot run."""
    exe = ffprobe_bin()
    if not exe:
        raise RuntimeError("ffprobe not available")
    code, out, err = _run([exe, "-v", "error", "-print_format", "json",
                           "-show_streams", "-show_format", path])
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()[:300]}")
    try:
        data = json.loads(out or "{}")
    except ValueError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}")
    return parse_probe(data)


# --- pure checks --------------------------------------------------------------

def _check(name, ok, detail, **extra):
    out = {"name": name, "ok": ok, "detail": detail}
    out.update(extra)
    return out


def aspect_check(width, height):
    if not width or not height:
        return _check("aspect_9_16", False, "no video dimensions")
    ratio = width / height
    ok = abs(ratio - TARGET_ASPECT) < ASPECT_TOLERANCE
    return _check("aspect_9_16", ok, f"{width}x{height} (ratio {ratio:.4f}, want 0.5625)")


def duration_check(duration, min_seconds=5.0, max_seconds=60.0):
    if duration is None:
        return _check("duration", False, "duration unknown")
    ok = min_seconds <= duration <= max_seconds
    return _check("duration", ok,
                  f"{duration:.2f}s (allowed {min_seconds:g}-{max_seconds:g}s)")


def streams_check(info):
    ok = bool(info.get("has_video")) and bool(info.get("has_audio"))
    detail = (f"video={info.get('video_codec') or 'missing'}, "
              f"audio={info.get('audio_codec') or 'missing'}")
    if info.get("fps"):
        detail += f", {info['fps']} fps"
    return _check("streams", ok, detail)


def parse_lufs(stderr_text):
    """Integrated loudness from ebur128's summary (the last ``I:`` line)."""
    found = re.findall(r"\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS", stderr_text or "")
    return float(found[-1]) if found else None


def loudness_check_from(lufs):
    if lufs is None:
        return _check("loudness", None, "integrated loudness not measured",
                      informational=True)
    ok = abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE
    return _check("loudness", ok,
                  f"{lufs:.1f} LUFS integrated (target {TARGET_LUFS:g} ±{LUFS_TOLERANCE:g})",
                  informational=True, lufs=lufs)


# --- captions -----------------------------------------------------------------

def _ass_seconds(text):
    m = re.fullmatch(r"\s*(\d+):(\d{1,2}):(\d{1,2})(?:[.,](\d+))?\s*", text or "")
    if not m:
        raise ValueError(f"bad ASS time {text!r}")
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + (float("0." + frac) if frac else 0.0)


def _srt_seconds(text):
    m = re.fullmatch(r"\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*", text or "")
    if not m:
        raise ValueError(f"bad SRT time {text!r}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


_ASS_TAG = re.compile(r"\{[^}]*\}")


def parse_ass(text):
    """Dialogue events of an ASS script as [{start, end, text}] (file order)."""
    events = []
    fields = None
    in_events = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fields = [f.strip().lower() for f in line[7:].split(",")]
            continue
        if not line.lower().startswith("dialogue:"):
            continue
        cols = fields or ["layer", "start", "end", "style", "name", "marginl",
                          "marginr", "marginv", "effect", "text"]
        parts = line[9:].split(",", len(cols) - 1)
        if len(parts) < len(cols):
            raise ValueError("Dialogue line with too few fields")
        row = dict(zip(cols, parts))
        events.append({
            "start": _ass_seconds(row.get("start")),
            "end": _ass_seconds(row.get("end")),
            "text": _ASS_TAG.sub("", row.get("text", "")).strip(),
        })
    return events


def parse_srt(text):
    events = []
    arrow = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = arrow.match(lines[i])
        if m:
            start, end = _srt_seconds(m.group(1)), _srt_seconds(m.group(2))
            body = []
            i += 1
            while i < len(lines) and lines[i].strip():
                body.append(lines[i].strip())
                i += 1
            events.append({"start": start, "end": end, "text": " ".join(body)})
        i += 1
    return events


def parse_caption_json(data):
    """[{start, end, text|word}] or {"events"|"captions"|"words": [...]}."""
    if isinstance(data, dict):
        for key in ("events", "captions", "words", "segments"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("caption JSON is not a list of events")
    events = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("caption JSON event is not an object")
        start, end = _num(item.get("start")), _num(item.get("end"))
        if start is None or end is None:
            raise ValueError("caption JSON event without numeric start/end")
        events.append({"start": start, "end": end,
                       "text": str(item.get("text", item.get("word", "")))})
    return events


def parse_captions(path):
    """Events of a caption sidecar (.ass / .ssa / .srt / .json)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ass", ".ssa"):
        return parse_ass(text)
    if ext == ".srt":
        return parse_srt(text)
    if ext == ".json":
        return parse_caption_json(json.loads(text))
    if "[Events]" in text:
        return parse_ass(text)
    if "-->" in text:
        return parse_srt(text)
    return parse_caption_json(json.loads(text))


def transcript_words(transcript):
    """Flat [{word, start, end}] (absolute seconds) of a transcript dict."""
    out = []
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    for seg in segments if isinstance(segments, list) else []:
        words = seg.get("words") if isinstance(seg, dict) else None
        for w in words if isinstance(words, list) else []:
            if not isinstance(w, dict):
                continue
            start, end = _num(w.get("start")), _num(w.get("end"))
            if start is None or end is None:
                continue
            out.append({"word": str(w.get("word", "")), "start": start, "end": end})
    return out


def relative_word_starts(words, clip_start, clip_end=None):
    """Start of each word of the clip, in clip seconds (as subtitles.py
    computes them: clamped at 0 for a word that straddles the cut)."""
    starts = []
    for w in words or []:
        start, end = _num(w.get("start")), _num(w.get("end"))
        if start is None or end is None:
            continue
        if end <= clip_start:
            continue
        if clip_end is not None and start >= clip_end:
            continue
        if not str(w.get("word", "")).strip():
            continue
        starts.append(max(0.0, start - clip_start))
    return sorted(starts)


def caption_offsets(event_starts, word_starts):
    """Absolute distance of each caption start to its nearest word start."""
    if not word_starts:
        return []
    import bisect
    ws = sorted(word_starts)
    out = []
    for t in event_starts:
        i = bisect.bisect_left(ws, t)
        best = min(abs(ws[j] - t) for j in (i - 1, i) if 0 <= j < len(ws))
        out.append(best)
    return out


def caption_checks(events, duration=None, word_starts=None):
    """Checks over parsed caption events; ``word_starts`` in clip seconds."""
    checks = []
    if not events:
        return [_check("captions_parse", False, "no caption events")]
    checks.append(_check("captions_parse", True, f"{len(events)} event(s)"))

    bad = [i for i, ev in enumerate(events) if ev["end"] < ev["start"]]
    backwards = [i for i in range(1, len(events))
                 if events[i]["start"] < events[i - 1]["start"]]
    ok = not bad and not backwards
    detail = "starts non-decreasing, every end >= start"
    if not ok:
        detail = (f"{len(backwards)} event(s) start before the previous one, "
                  f"{len(bad)} end before they start")
    checks.append(_check("captions_monotonic", ok, detail))

    if duration is not None:
        outside = [i for i, ev in enumerate(events)
                   if ev["start"] < 0 or ev["end"] > duration + CAPTION_EDGE_SLACK]
        last = max(ev["end"] for ev in events)
        checks.append(_check(
            "captions_in_bounds", not outside,
            f"last caption ends {last:.2f}s of {duration:.2f}s"
            + (f"; {len(outside)} event(s) outside" if outside else "")))
    else:
        checks.append(_check("captions_in_bounds", None, "clip duration unknown"))

    if word_starts is not None:
        offsets = caption_offsets([ev["start"] for ev in events], word_starts)
        if not offsets:
            checks.append(_check("captions_aligned", False,
                                 "no transcript words inside the clip"))
        else:
            med = statistics.median(offsets)
            mx = max(offsets)
            ok = med <= CAPTION_MEDIAN_MAX and mx <= CAPTION_OFFSET_MAX
            checks.append(_check(
                "captions_aligned", ok,
                f"median offset {med:.3f}s (<= {CAPTION_MEDIAN_MAX}), "
                f"max {mx:.3f}s (<= {CAPTION_OFFSET_MAX}) over {len(offsets)} event(s)",
                median_offset=round(med, 4), max_offset=round(mx, 4),
                offsets=[round(o, 4) for o in offsets]))
    return checks


# --- ffmpeg checks ------------------------------------------------------------

def decode_check(path):
    exe = ffmpeg_bin()
    if not exe:
        return _check("decodable", None, "ffmpeg not available")
    try:
        code, _, err = _run([exe, "-hide_banner", "-nostdin", "-v", "error",
                             "-i", path, "-f", "null", "-"])
    except (OSError, subprocess.TimeoutExpired) as e:
        return _check("decodable", False, f"decode did not finish: {e}")
    errors = [ln for ln in err.splitlines() if ln.strip()]
    ok = code == 0 and not errors
    detail = "decoded cleanly" if ok else (
        f"exit {code}; {len(errors)} error line(s): " + " / ".join(errors[:3])[:300])
    return _check("decodable", ok, detail)


def loudness_check(path):
    exe = ffmpeg_bin()
    if not exe:
        return _check("loudness", None, "ffmpeg not available", informational=True)
    try:
        _, _, err = _run([exe, "-hide_banner", "-nostdin", "-nostats", "-i", path,
                          "-map", "0:a:0", "-af", "ebur128", "-f", "null", "-"])
    except (OSError, subprocess.TimeoutExpired) as e:
        return _check("loudness", None, f"loudness not measured: {e}",
                      informational=True)
    return loudness_check_from(parse_lufs(err))


# --- entry point --------------------------------------------------------------

def overall_ok(checks):
    """False if any blocking check failed; None if nothing could be checked."""
    blocking = [c for c in checks if not c.get("informational")]
    if any(c["ok"] is False for c in blocking):
        return False
    if any(c["ok"] is True for c in blocking):
        return True
    return None


def check_short(path, *, max_seconds=60.0, min_seconds=5.0, captions_path=None,
                transcript_words=None, clip_start=None):
    """Every check for one rendered short -> {"file", "ok", "checks", "probe"}."""
    checks = []
    info = None
    if not os.path.isfile(path):
        checks.append(_check("file_exists", False, "file not found"))
        return {"file": path, "ok": False, "checks": checks, "probe": None}
    try:
        info = probe(path)
    except RuntimeError as e:
        reason = str(e)
        na = None if reason == "ffprobe not available" else False
        for name in ("aspect_9_16", "duration", "streams"):
            checks.append(_check(name, na, reason))
    except (OSError, subprocess.TimeoutExpired) as e:
        for name in ("aspect_9_16", "duration", "streams"):
            checks.append(_check(name, False, f"ffprobe did not run: {e}"))
    if info is not None:
        checks.append(aspect_check(info["width"], info["height"]))
        checks.append(duration_check(info["duration"], min_seconds, max_seconds))
        checks.append(streams_check(info))

    checks.append(decode_check(path))
    if info is None or info.get("has_audio"):
        checks.append(loudness_check(path))
    else:
        checks.append(_check("loudness", None, "no audio stream", informational=True))

    if captions_path:
        try:
            events = parse_captions(captions_path)
        except (OSError, ValueError) as e:
            checks.append(_check("captions_parse", False,
                                 f"sidecar did not parse: {type(e).__name__}: {e}"[:300]))
        else:
            duration = info["duration"] if info else None
            word_starts = None
            if transcript_words is not None and clip_start is not None:
                start = _num(clip_start, 0.0)
                end = start + duration if duration is not None else None
                word_starts = relative_word_starts(transcript_words, start, end)
            checks.extend(caption_checks(events, duration, word_starts))
    else:
        checks.append(_check("captions_parse", None, "no caption sidecar given",
                             informational=True))

    return {"file": path, "ok": overall_ok(checks), "checks": checks, "probe": info}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a rendered short.")
    parser.add_argument("file")
    parser.add_argument("--captions", help="caption sidecar (.ass/.srt/.json)")
    parser.add_argument("--transcript", help="transcript JSON or *_metadata.json, "
                                             "to measure caption alignment")
    parser.add_argument("--clip-start", type=float,
                        help="clip start in source seconds (with --transcript)")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    words = None
    if args.transcript:
        with open(args.transcript, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("transcript"), dict):
            data = data["transcript"]
        words = transcript_words(data)

    result = check_short(args.file, max_seconds=args.max_seconds,
                         min_seconds=args.min_seconds, captions_path=args.captions,
                         transcript_words=words, clip_start=args.clip_start)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = {True: "PASS", False: "FAIL", None: "N/A"}
        print(f"{status[result['ok']]}  {result['file']}")
        for c in result["checks"]:
            print(f"  [{status[c['ok']]}] {c['name']}: {c['detail']}")
    return 0 if result["ok"] is not False else 1


if __name__ == "__main__":
    sys.exit(main())
