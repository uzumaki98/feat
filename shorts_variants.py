"""Side-by-side edit variants of one finished Short, for judging effects.

    python shorts_variants.py --clip <vertical uncaptioned clip> --hook "<text>" \
        --out output/variants_<name> [--published <the posted short>]

Experimental and standalone: it only reads the clip and writes into ``--out``;
nothing in the job pipeline imports it or changes because of it.

  a_current.mp4   the published short, copied as-is (when --published is given)
  b_tight.mp4     dead air trimmed, punch-ins on the beats, one-word captions,
                  hook, -14 LUFS
  c_full.mp4      b + a cold open (the payoff flashed first) and synthetic
                  sound effects (whoosh on each punch-in and the cut back,
                  a low hit under the cold open)

Every sound effect is generated here with ffmpeg (noise/sine), so nothing in a
variant comes from a sound library or carries anyone's rights.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import punch_in
import subtitles as subs
from ffmpeg_utils import LOUDNORM_FILTER

# Silence trim: a pause longer than GAP between two words is cut down to PAD
# on each side, which keeps a breath without the dead air.
GAP = 0.45
PAD = 0.15
COLD_OPEN = 1.6          # seconds of payoff flashed first
SFX_GAIN_DB = -14        # effects sit well under the speech
# Livelier than the pipeline's punch-in defaults (one every 18 s): this is a
# 40-50 s short and a variant meant to be seen.
PUNCH_GAP = 6.0
PUNCH_PROMINENCE = 0.55
CAPTION_PRESET = "oneword"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-1500:]}")
    return r.stdout


def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate:format=duration", "-of", "json", path])
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), float(d["format"]["duration"])


def words_of(transcript):
    flat = []
    for seg in transcript.get("segments", []):
        flat.extend(seg.get("words", []))
    flat = subs.merge_continuation_words(flat)
    return [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
            for w in flat if w.get("word", "").strip() and w.get("end", 0) > w.get("start", 0)]


ENV_WINDOW = 0.1
# A wordless window is dead air only when it is also quiet: on gameplay the
# payoff is often loud and wordless (a crash, a scream, gunfire), and a
# speech-only trim cut 21 s of exactly that off the first test clip.
QUIET_OF_MEDIAN = 0.5
QUIET_FLOOR = 0.06


def keep_ranges(words, duration, env=None, window=ENV_WINDOW, gap=GAP, pad=PAD):
    """[(start, end)] of the clip to keep: everything except runs longer than
    ``gap`` with no word and quiet audio (``env``: per-``window`` loudness,
    0-1), each cut down to ``pad`` either side. No ``env``: every wordless
    stretch counts as quiet."""
    n = max(1, int(round(duration / window)))
    active = [False] * n
    for w in words:
        for i in range(int(w["start"] / window), min(n, int(w["end"] / window) + 1)):
            active[i] = True
    if env is not None and len(env):
        v = np.asarray(env, dtype=float)
        quiet = max(QUIET_FLOOR, float(np.median(v)) * QUIET_OF_MEDIAN)
        for i in range(min(n, len(v))):
            if v[i] >= quiet:
                active[i] = True
    cuts, i = [], 0
    while i < n:
        if active[i]:
            i += 1
            continue
        j = i
        while j < n and not active[j]:
            j += 1
        a, b = i * window, min(duration, j * window)
        if b - a > gap:
            lo = a + (pad if i > 0 else 0.0)
            hi = b - (pad if j < n else 0.0)
            if hi > lo:
                cuts.append((lo, hi))
        i = j
    ranges, t = [], 0.0
    for lo, hi in cuts:
        if lo > t:
            ranges.append((t, lo))
        t = hi
    if t < duration:
        ranges.append((t, duration))
    out = [(round(x, 3), round(y, 3)) for x, y in ranges if y - x > 0.05]
    return out or [(0.0, duration)]


def remap(t, ranges):
    """Time ``t`` of the source on the trimmed timeline (None if cut out)."""
    off = 0.0
    for a, b in ranges:
        if a <= t <= b:
            return off + (t - a)
        off += b - a
    return None


def as_transcript(words):
    return {"segments": [{"start": words[0]["start"] if words else 0,
                          "end": words[-1]["end"] if words else 0,
                          "text": " ".join(w["word"] for w in words),
                          "words": words}]}


def cut(src, ranges, out, fps):
    """Concatenate ``ranges`` of ``src`` in the order given (short fades so
    cuts do not click).

    One seeked input per range, not trim on one input: ffmpeg splits a single
    input to every trim and buffers decoded frames for the later ones, which
    for a cold open (a late range first) is ~30 s of raw 1080p60 per job;
    three jobs at once ran out of memory and wrote a truncated file. Seeked
    inputs stream. The output length is checked: a short cut raises."""
    inputs, parts, labels = [], [], []
    for i, (a, b) in enumerate(ranges):
        inputs += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src]
        fade = min(0.01, (b - a) / 4)
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[{i}:a]asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={fade},afade=t=out:st={b - a - fade:.3f}:d={fade}[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(ranges)}:v=1:a=1[v][a]"
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph,
         "-map", "[v]", "-map", "[a]", "-r", f"{fps:g}", "-c:v", "libx264", "-crf", "16",
         "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k", out])
    want = sum(b - a for a, b in ranges)
    got = probe(out)[3]
    if abs(got - want) > 0.5:
        raise RuntimeError(f"cut came out {got:.2f}s, expected {want:.2f}s")


def payoff_time(path, duration, length=COLD_OPEN):
    """Start of the loudest ``length`` window in the back 60% of the clip:
    the moment most likely to be the payoff."""
    import active_speaker
    win = 0.1
    env = active_speaker.audio_envelope(path, 0.0, duration, window_s=win)
    if not env:
        return max(0.0, duration * 0.6)
    v = np.asarray(env, dtype=float)
    n = max(1, int(length / win))
    sums = np.convolve(v, np.ones(n), "valid")
    first = int(duration * 0.4 / win)
    last = max(first + 1, int((duration - length - 0.3) / win))
    i = first + int(np.argmax(sums[first:last])) if last <= len(sums) else int(np.argmax(sums))
    return round(i * win, 2)


def zoom_expr(beats, max_zoom=1.12):
    """ffmpeg expression of t: 1.0, rising to ``max_zoom`` on each beat with
    punch_in's envelope (smoothstep in, hold, smoothstep out).

    Why an expression and not punch_in.sendcmd_lines: a crop whose w/h change
    per frame followed by scale segfaults this ffmpeg (seen 19-nov-2026, on
    the CPU path; the GPU render reads the boxes itself). Scaling up per frame
    and cropping a fixed window gives the same push without resizing a link."""
    r, h, f = punch_in.RISE_SECONDS, punch_in.HOLD_SECONDS, punch_in.FALL_SECONDS
    terms = []
    for b in beats:
        b = float(b)
        terms.append(
            f"(st(0,clip((t-{b:.3f})/{r},0,1))*0+ld(0)*ld(0)*(3-2*ld(0)))"
            f"*(1-(st(1,clip((t-{b + r + h:.3f})/{f},0,1))*0+ld(1)*ld(1)*(3-2*ld(1))))")
    if not terms:
        return "1"
    m = terms[0]
    for t in terms[1:]:
        m = f"max({m},{t})"
    return f"(1+{max_zoom - 1:.3f}*{m})"


def punch(src, out, w, h, fps, duration, beats, zoom_from=0):
    """Punch-ins on ``beats``. ``zoom_from``: only the band below this row
    zooms (the gameplay of the streamer template); the camera above stays."""
    z = zoom_expr(beats)
    zoom_from = int(zoom_from or 0)
    zoom_from -= zoom_from % 2
    if 0 < zoom_from < h - 2:
        bh = h - zoom_from
        graph = (f"[0:v]split=2[t][b];[t]crop={w}:{zoom_from}:0:0[top];"
                 f"[b]crop={w}:{bh}:0:{zoom_from},"
                 f"scale=w='2*trunc({w}*{z}/2)':h='2*trunc({bh}*{z}/2)':eval=frame:"
                 f"flags=bicubic,crop={w}:{bh}:(iw-{w})/2:(ih-{bh})/2[bot];"
                 f"[top][bot]vstack=inputs=2,setsar=1[v]")
    else:
        graph = (f"[0:v]scale=w='2*trunc({w}*{z}/2)':h='2*trunc({h}*{z}/2)':eval=frame:"
                 f"flags=bicubic,crop={w}:{h}:(iw-{w})/2:(ih-{h})/2,setsar=1[v]")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", graph, "-map", "[v]",
         "-map", "0:a", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
         "-c:a", "copy", out])


def captions(src, out, words, duration):
    p = subs.CAPTION_PRESETS[CAPTION_PRESET]
    ass = os.path.join(os.path.dirname(out), "captions.ass")
    if not subs.generate_ass(
            as_transcript(words), 0.0, duration, ass, max_chars=p["max_chars"],
            max_duration=p["max_duration"], alignment="bottom", fontsize=p["font_size"],
            font_name=p["font_name"], font_color=p["font_color"],
            border_color="#000000", border_width=p["border_width"],
            highlight_color=p["highlight_color"], effect=p["effect"],
            base_opacity=p["base_opacity"], uppercase=p["uppercase"],
            reveal=p["reveal"], shadow=p["shadow"]):
        shutil.copy(src, out)
        return None
    subs.burn_subtitles(src, ass, out, font_name=p["font_name"], fontsize=p["font_size"],
                        border_width=p["border_width"])
    return ass


def hook(src, out, text):
    import hooks
    hooks.add_hook_to_video(src, text, out, position="auto", duration=5.0,
                            style="pill", has_captions=True)


def make_sfx(dirname):
    """whoosh.wav and hit.wav, synthesised (no third-party audio)."""
    whoosh = os.path.join(dirname, "whoosh.wav")
    hit = os.path.join(dirname, "hit.wav")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "anoisesrc=d=0.45:c=pink:a=0.8", "-af",
         "highpass=f=400,lowpass=f=6000,afade=t=in:d=0.3:curve=exp,"
         "afade=t=out:st=0.3:d=0.15", "-ar", "48000", "-ac", "2", whoosh])
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "sine=f=55:d=0.7", "-af", "volume=2.5,afade=t=out:st=0.05:d=0.65:curve=exp",
         "-ar", "48000", "-ac", "2", hit])
    return whoosh, hit


def mix_sfx(src, out, cues):
    """Lay ``cues`` [(path, seconds)] under ``src``'s audio, then -14 LUFS."""
    inputs, chains, labels = ["-i", src], [], ["[0:a]"]
    for i, (path, t) in enumerate(cues, start=1):
        inputs += ["-i", path]
        ms = max(0, int(t * 1000))
        chains.append(f"[{i}:a]adelay={ms}|{ms},volume={SFX_GAIN_DB}dB[s{i}]")
        labels.append(f"[s{i}]")
    graph = ";".join(chains + [f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:"
                               f"normalize=0,{LOUDNORM_FILTER}[a]"])
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph,
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-movflags", "+faststart", out])


def loudnorm(src, out):
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", LOUDNORM_FILTER, "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", out])


def full_edit(clip, words, hook_text, out, work, *, cold_open=True, sfx=True,
              max_seconds=59.5, zoom_from=0):
    """Style C on one vertical clip without captions or hook: quiet pauses
    trimmed, cold open (when it still fits ``max_seconds``), punch-ins,
    one-word captions, hook, sound effects, -14 LUFS. ``words`` are the clip's
    words in clip seconds. Returns what was done, with ``captions`` (the
    burned .ass, None without words) and ``words`` on the final timeline."""
    os.makedirs(work, exist_ok=True)
    w, h, fps, duration = probe(clip)
    import active_speaker
    env = active_speaker.audio_envelope(clip, 0.0, duration, window_s=ENV_WINDOW)
    ranges = keep_ranges(words, duration, env)
    trimmed = os.path.join(work, "trim.mp4")
    cut(clip, ranges, trimmed, fps)
    t_words = []
    for wd in words:
        s, e = remap(wd["start"], ranges), remap(wd["end"], ranges)
        if s is not None and e is not None and e > s:
            t_words.append(dict(wd, start=round(s, 3), end=round(e, 3)))
    _, _, _, t_dur = probe(trimmed)
    beats = punch_in.emphasis_times(trimmed, t_dur, min_gap=PUNCH_GAP,
                                    prominence=PUNCH_PROMINENCE)
    report = {"source_seconds": round(duration, 2), "cuts": len(ranges) - 1,
              "trimmed_seconds": round(duration - t_dur, 2), "cold_open_from": None}
    src, f_words, lead = trimmed, t_words, 0.0
    if cold_open and t_dur + COLD_OPEN <= max_seconds:
        p = payoff_time(trimmed, t_dur)
        span = (p, min(t_dur, p + COLD_OPEN))
        lead = span[1] - span[0]
        src = os.path.join(work, "cold.mp4")
        cut(trimmed, [span, (0.0, t_dur)], src, fps)
        f_words = [dict(wd, start=round(wd["start"] - p, 3), end=round(wd["end"] - p, 3))
                   for wd in t_words if wd["start"] >= p and wd["end"] <= p + lead]
        f_words += [dict(wd, start=round(wd["start"] + lead, 3),
                         end=round(wd["end"] + lead, 3)) for wd in t_words]
        report["cold_open_from"] = p
    _, _, _, f_dur = probe(src)
    f_beats = [round(b + lead, 3) for b in beats]
    cues = None
    if sfx:
        whoosh, hit = make_sfx(work)
        cues = [(whoosh, b - 0.15) for b in f_beats]
        if lead:
            cues = [(hit, 0.0), (whoosh, lead - 0.3)] + cues
    report.update(seconds=round(f_dur, 2), punch_ins=f_beats, sfx=len(cues or []))
    report["captions"] = _finish(src, f_words, f_dur, f_beats, w, h, fps, hook_text,
                                 os.path.join(work, "edit"), out, sfx=cues,
                                 zoom_from=zoom_from)
    report["words"] = f_words
    return report


def build(clip, hook_text, out_dir, published=None):
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "work")
    report = {"clip": os.path.basename(clip)}
    if published:
        shutil.copy(published, os.path.join(out_dir, "a_current.mp4"))
    from transcribe_backends import release_models
    try:
        words = words_of(subs.transcribe_audio(clip))
    finally:
        release_models()
    for name, kw in (("b_tight", dict(cold_open=False, sfx=False)), ("c_full", {})):
        r = full_edit(clip, words, hook_text, os.path.join(out_dir, name + ".mp4"),
                      os.path.join(work, name), **kw)
        r.pop("words")
        report[name] = r
    with open(os.path.join(out_dir, "variants.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _finish(src, words, duration, beats, w, h, fps, hook_text, stem, out, sfx,
            zoom_from=0):
    """Punch, caption, hook, then effects or loudness. Returns the .ass burned."""
    punched = stem + "_punch.mp4"
    punch(src, punched, w, h, fps, duration, beats, zoom_from=zoom_from)
    captioned = stem + "_caps.mp4"
    ass = captions(punched, captioned, words, duration)
    hooked = stem + "_hook.mp4"
    if hook_text:
        hook(captioned, hooked, hook_text)
    else:
        hooked = captioned
    if sfx:
        mix_sfx(hooked, out, sfx)
    else:
        loudnorm(hooked, out)
    return ass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", required=True, help="vertical clip without captions or hook")
    ap.add_argument("--hook", required=True, help="hook text for the first seconds")
    ap.add_argument("--out", required=True, help="new directory for the variants")
    ap.add_argument("--published", help="the posted short, copied as variant a")
    args = ap.parse_args(argv)
    report = build(args.clip, args.hook, args.out, args.published)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
