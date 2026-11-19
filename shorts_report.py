"""Ranked report of a headless shorts run (shorts_pipeline.py).

    python shorts_report.py <job_dir>     # writes <job_dir>/report.md + report.json

``build_report(job_dir)`` reads what the run left in the job directory:

- ``<title>_metadata.json`` (main.py): ``shorts`` (rendered, best first),
  ``candidates`` (every rated pick, when RENDER_TOP_N is on), ``transcript``,
  ``cost_analysis``, ``source_video``;
- ``verification.json`` (shorts_pipeline verify stage);
- ``stages.json`` or, failing that, ``pipeline.log`` (stage timings).

Every value read from those files is data from a model or another process:
numbers are coerced to numbers, text is escaped before it lands in a table or
a heading, and file names are only ever matched as plain basenames inside the
job directory. Any key may be missing (older metadata): the report degrades
and lists what it could not find instead of failing.

The helpers that locate the served file of a clip and its caption sidecar, and
the log-marker parsing, live here too so the pipeline and the report agree.
"""
import glob
import html
import json
import math
import os
import re
import sys
import time

CRITERIA = ("hook_strength", "energy_pacing", "narrative_completeness",
            "av_clarity", "retention_potential")
CRITERIA_SHORT = {"hook_strength": "Hook", "energy_pacing": "Energy",
                  "narrative_completeness": "Story", "av_clarity": "A/V",
                  "retention_potential": "Retention"}

METADATA_SUFFIX = "_metadata.json"
VERIFICATION_FILE = "verification.json"
STAGES_FILE = "stages.json"
LOG_FILE = "pipeline.log"
RUN_FILE = "run.json"


# --- coercion and escaping ----------------------------------------------------

def num(value, default=None):
    """Finite float from anything (bools excluded), else ``default``."""
    if isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def as_int(value, default=None):
    out = num(value)
    return int(out) if out is not None else default


def text(value, limit=None):
    """A plain one-line string from anything."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) \
            else str(value)
    out = re.sub(r"[\r\n\t\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", value).strip()
    if limit and len(out) > limit:
        out = out[:limit - 1].rstrip() + "…"
    return out


def cell(value, limit=120):
    """Safe Markdown table cell: one line, no pipes, no raw HTML."""
    out = html.escape(text(value, limit), quote=False)
    return out.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def para(value, limit=5000):
    """Free text for a Markdown block: HTML escaped, every line quoted so a
    leading '#', '---' or list marker cannot restructure the report."""
    if value is None:
        return "> _(none)_"
    raw = value if isinstance(value, str) else text(value)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")[:limit]
    lines = [html.escape(ln, quote=False).replace("`", "\\`") for ln in raw.split("\n")]
    body = "\n".join("> " + ln if ln.strip() else ">" for ln in lines).strip()
    return body if body.strip("> \n") else "> _(none)_"


def hms(seconds):
    s = num(seconds)
    if s is None or s < 0:
        return "?"
    s = int(s)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def hashtags(value):
    """List of '#tag' strings from a list or a whitespace/comma string."""
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r"[\s,]+", text(value))
    out = []
    for item in items:
        tag = re.sub(r"\s+", "", text(item))
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag
        if tag not in out:
            out.append(tag)
    return out


def load_json(path):
    """Parsed JSON of ``path`` or None (missing, unreadable, invalid)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- files in the job directory -----------------------------------------------

def find_metadata(job_dir):
    """Newest ``*_metadata.json`` in job_dir, or None."""
    found = [p for p in glob.glob(os.path.join(glob.escape(job_dir), "*" + METADATA_SUFFIX))
             if os.path.isfile(p)]
    if not found:
        return None
    return max(found, key=lambda p: os.path.getmtime(p))


def video_title_of(metadata_path):
    base = os.path.basename(metadata_path or "")
    return base[:-len(METADATA_SUFFIX)] if base.endswith(METADATA_SUFFIX) else None


def safe_basename(name):
    """``name`` if it is a plain file name (no directory, not hidden), else None."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if (not name or name in (".", "..") or name.startswith(".")
            or "/" in name or "\\" in name or "\x00" in name
            or os.path.basename(name) != name):
        return None
    return name


def resolve_in(job_dir, name):
    """Absolute path of plain file ``name`` inside job_dir if it is a regular
    file there (not a link), else None."""
    name = safe_basename(name)
    if not name:
        return None
    path = os.path.join(job_dir, name)
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    root = os.path.realpath(job_dir)
    if os.path.dirname(os.path.realpath(path)) != root:
        return None
    return path


_LAYER = r"(?:(?:subtitled|hooked)_\d+_)"


def find_served_file(job_dir, video_title, index):
    """Most finished file of clip ``index`` (0-based): the captioned/hooked
    derivative of ``<title>_clip_<index+1>.mp4`` when one exists (main.py
    writes ``subtitled_<ts>_hooked_<ts>_<canonical>``), else the canonical.
    A ``wm_`` copy is used only when nothing else exists."""
    if not video_title:
        return None
    canonical = f"{video_title}_clip_{int(index) + 1}.mp4"
    pattern = re.compile(r"^(wm_)?(" + _LAYER + r"*)" + re.escape(canonical) + r"$")
    best = None
    try:
        names = os.listdir(job_dir)
    except OSError:
        return None
    for name in names:
        m = pattern.match(name)
        path = resolve_in(job_dir, name) if m else None
        if not path:
            continue
        layers = len(re.findall(_LAYER, m.group(2)))
        key = (0 if m.group(1) else 1, layers, os.path.getmtime(path))
        if best is None or key > best[0]:
            best = (key, name)
    return best[1] if best else None


def caption_sidecars(job_dir, served_name):
    """``autosubs_<gid>_<hex8>.ass`` files that belong to a served file named
    ``[wm_]subtitled_<gid>_...`` (main.auto_caption_clip uses one generation
    id for both names). Several clips captioned in the same second share the
    gid, so this can return more than one."""
    name = safe_basename(served_name) or ""
    m = re.match(r"^(?:wm_)?subtitled_(\d+)_", name)
    if not m:
        return []
    pat = re.compile(r"^autosubs_" + m.group(1) + r"_[0-9a-f]{8}\.ass$")
    try:
        names = sorted(os.listdir(job_dir))
    except OSError:
        return []
    return [n for n in names if pat.match(n) and resolve_in(job_dir, n)]


def find_caption_sidecar(job_dir, served_name, words=None, clip_start=None, clip_end=None):
    """The caption sidecar of ``served_name`` (basename) or None. When several
    share its generation id, the one whose events sit on this clip's words."""
    found = caption_sidecars(job_dir, served_name)
    if len(found) <= 1 or not words or clip_start is None:
        return found[0] if found else None
    import verify_shorts
    starts = verify_shorts.relative_word_starts(words, num(clip_start, 0.0), num(clip_end))
    best = None
    for name in found:
        try:
            events = verify_shorts.parse_captions(os.path.join(job_dir, name))
        except (OSError, ValueError):
            continue
        offs = verify_shorts.caption_offsets([e["start"] for e in events], starts)
        if not offs:
            continue
        score = sorted(offs)[len(offs) // 2]
        if best is None or score < best[0]:
            best = (score, name)
    return best[1] if best else found[0]


# --- pipeline log markers -----------------------------------------------------

# (stage, regex) in pipeline order. A stage starts at the first line matching
# its marker and ends when a LATER stage starts (stages never go backwards:
# the download retries after transcription has started are still download
# noise, not a new stage). Markers are main.py's own log lines.
STAGE_MARKERS = (
    ("download", re.compile(r"Downloading video|Download attempt")),
    ("layout", re.compile(r"Choosing a layout for this video|already vertical, no layout")),
    ("transcription", re.compile(r"Transcribing video|Reusing precomputed transcript"
                                 r"|Reusing the transcript|Using the transcript made")),
    ("clip_selection", re.compile(r"Analyzing with |switching to visual analysis"
                                  r"|Using the clips Gemini picked|Analyzing video with Gemini")),
    ("rating", re.compile(r"(?i)^\W*rating\b|\bRATE_CLIPS\b|\bclip_rating\b")),
    ("render", re.compile(r"Processing Clip \d+")),
    ("done", re.compile(r"Total execution time")),
)
STAGE_ORDER = [name for name, _ in STAGE_MARKERS]

_CLIP_READY = re.compile(r"^CLIP_READY\s+(\d+)\s+(\S.*?)\s*$")
_LOG_LINE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)s\]\s?(.*)$")


def parse_clip_ready(line):
    """(index, basename) of a ``CLIP_READY <index> <basename>`` line, else None.
    A name that is not a plain basename is refused (it is data)."""
    m = _CLIP_READY.match((line or "").strip())
    if not m:
        return None
    name = safe_basename(m.group(2))
    if not name:
        return None
    return int(m.group(1)), name


def match_stage(line):
    """Name of the stage whose marker ``line`` carries, or None."""
    stripped = (line or "").strip()
    for name, rx in STAGE_MARKERS:
        if rx.search(stripped):
            return name
    return None


class StageTracker:
    """Stage boundaries and CLIP_READY records from a stream of log lines."""

    def __init__(self):
        self.stages = []          # [{"name", "start_s", "end_s"}]
        self.clip_ready = {}      # index -> basename
        self.metadata_file = None
        self.clips_found = None

    @property
    def current(self):
        return self.stages[-1]["name"] if self.stages else None

    def feed(self, line, elapsed):
        line = line or ""
        ready = parse_clip_ready(line)
        if ready:
            self.clip_ready[ready[0]] = ready[1]
            if self.current != "render" and "render" not in [s["name"] for s in self.stages]:
                self._enter("render", elapsed)
            return "clip_ready"
        m = re.search(r"Saved metadata to (.+?)\s*$", line)
        if m:
            self.metadata_file = safe_basename(os.path.basename(m.group(1).strip()))
        m = re.search(r"Found (\d+) clips", line)
        if m:
            self.clips_found = int(m.group(1))
        stage = match_stage(line)
        if stage and self._is_forward(stage):
            self._enter(stage, elapsed)
            return stage
        return None

    def _is_forward(self, stage):
        if self.current is None:
            return True
        return STAGE_ORDER.index(stage) > STAGE_ORDER.index(self.current)

    def _enter(self, stage, elapsed):
        if self.stages and self.stages[-1].get("end_s") is None:
            self.stages[-1]["end_s"] = round(elapsed, 2)
        self.stages.append({"name": stage, "start_s": round(elapsed, 2), "end_s": None})

    def finish(self, elapsed):
        if self.stages and self.stages[-1].get("end_s") is None:
            self.stages[-1]["end_s"] = round(elapsed, 2)

    def summary(self):
        out = []
        for s in self.stages:
            end = s.get("end_s")
            out.append({"name": s["name"], "start_s": s["start_s"], "end_s": end,
                        "duration_s": round(end - s["start_s"], 2) if end is not None else None})
        return out


def stages_from_log(path):
    """StageTracker fed from a pipeline.log (``[  12.3s] line``), or None."""
    try:
        f = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    tracker = StageTracker()
    last = 0.0
    with f:
        for raw in f:
            m = _LOG_LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            last = float(m.group(1))
            tracker.feed(m.group(2), last)
    tracker.finish(last)
    return tracker


# --- building the report ------------------------------------------------------

def _rating(item):
    r = item.get("rating") if isinstance(item, dict) else None
    return r if isinstance(r, dict) else None


def normalise_candidate(item, position):
    """Plain, typed view of one candidate/short dict from metadata."""
    item = item if isinstance(item, dict) else {}
    rating = _rating(item) or {}
    scores_in = rating.get("scores") if isinstance(rating.get("scores"), dict) else {}
    scores = {k: num(scores_in.get(k)) for k in CRITERIA}
    start, end = num(item.get("start")), num(item.get("end"))
    overall = num(rating.get("overall"))
    return {
        "position": position,
        "rank": as_int(item.get("rank")),
        "candidate_index": as_int(item.get("candidate_index")),
        "start": start,
        "end": end,
        "duration": round(end - start, 2) if start is not None and end is not None else None,
        "overall": overall,
        "scores": scores,
        "summary": text(rating.get("summary"), 400),
        "rationale": {text(k, 60): text(v, 400)
                      for k, v in (rating.get("rationale") or {}).items()}
        if isinstance(rating.get("rationale"), dict) else {},
        "rated": bool(rating) and overall is not None,
        "title": text(item.get("video_title_for_youtube_short"), 300),
        "description": item.get("video_description_for_youtube_short")
        if isinstance(item.get("video_description_for_youtube_short"), str)
        else text(item.get("video_description_for_youtube_short")),
        "hashtags": hashtags(item.get("youtube_hashtags")),
        "hook": text(item.get("viral_hook_text"), 300),
        "virality_score": num(item.get("virality_score", item.get("score"))),
    }


def _same_clip(a, b):
    if a.get("candidate_index") is not None and b.get("candidate_index") is not None:
        return a["candidate_index"] == b["candidate_index"]
    if None not in (a.get("start"), a.get("end"), b.get("start"), b.get("end")):
        return abs(a["start"] - b["start"]) < 0.05 and abs(a["end"] - b["end"]) < 0.05
    return False


def rendered_files(job_dir, shorts, video_title):
    """short index -> served basename, from verification.json, stages.json
    CLIP_READY records, then the files on disk."""
    files = {}
    ver = load_json(os.path.join(job_dir, VERIFICATION_FILE))
    for row in (ver or {}).get("shorts", []) if isinstance(ver, dict) else []:
        if not isinstance(row, dict):
            continue
        i, name = as_int(row.get("short_index")), safe_basename(row.get("file"))
        if i is not None and name:
            files[i] = name
    st = load_json(os.path.join(job_dir, STAGES_FILE))
    ready = st.get("clip_ready") if isinstance(st, dict) else None
    for k, v in (ready or {}).items() if isinstance(ready, dict) else []:
        i, name = as_int(k), safe_basename(v)
        if i is not None and name and i not in files:
            files[i] = name
    for i in range(len(shorts)):
        if i not in files:
            name = find_served_file(job_dir, video_title, i)
            if name:
                files[i] = name
    return files


def cost_summary(metadata):
    """{"total_usd", "per_stage": {stage: usd}, "model", "tokens"} or None."""
    if not isinstance(metadata, dict):
        return None
    per_stage = {}
    model = None
    tokens = {}
    ca = metadata.get("cost_analysis")
    if isinstance(ca, dict):
        model = text(ca.get("model"), 80) or None
        for k in ("input_tokens", "output_tokens", "thinking_tokens"):
            if as_int(ca.get(k)) is not None:
                tokens[k] = as_int(ca.get(k))
        stages = ca.get("stages") if isinstance(ca.get("stages"), dict) else None
        if stages:
            for name, val in stages.items():
                usd = num(val.get("total_cost")) if isinstance(val, dict) else num(val)
                if usd is not None:
                    per_stage[text(name, 60)] = usd
        elif num(ca.get("total_cost")) is not None:
            per_stage["clip_selection"] = num(ca.get("total_cost"))
    # Other stages may record their own cost next to it:
    # "<stage>_cost_analysis": {"total_cost": ...}.
    for key, val in metadata.items():
        m = re.fullmatch(r"([a-z0-9_]+)_cost_analysis", str(key))
        if m and isinstance(val, dict) and num(val.get("total_cost")) is not None:
            per_stage.setdefault(m.group(1), num(val["total_cost"]))
    if not per_stage:
        return None
    total = sum(per_stage.values())
    if isinstance(ca, dict) and isinstance(ca.get("stages"), dict) \
            and num(ca.get("total_cost")) is not None:
        total = max(total, num(ca["total_cost"]))
    return {"total_usd": round(total, 6),
            "per_stage": {k: round(v, 6) for k, v in per_stage.items()},
            "model": model, "tokens": tokens}


def build_report(job_dir):
    """(markdown, report_dict) for the run in ``job_dir``."""
    issues = []
    meta_path = find_metadata(job_dir)
    metadata = load_json(meta_path) if meta_path else None
    if not meta_path:
        issues.append("No *_metadata.json in the job directory (the run did not get "
                      "past clip selection).")
    elif not isinstance(metadata, dict):
        issues.append(f"{os.path.basename(meta_path)} is not valid JSON.")
        metadata = None
    metadata = metadata or {}
    title = video_title_of(meta_path)

    raw_shorts = metadata.get("shorts") if isinstance(metadata.get("shorts"), list) else []
    raw_cands = metadata.get("candidates") if isinstance(metadata.get("candidates"), list) \
        else None
    shorts = [normalise_candidate(s, i) for i, s in enumerate(raw_shorts)]
    candidates = ([normalise_candidate(c, i) for i, c in enumerate(raw_cands)]
                  if raw_cands is not None else [dict(s) for s in shorts])
    if raw_cands is None and raw_shorts:
        issues.append("Metadata has no `candidates` list (RENDER_TOP_N not applied): "
                      "the table shows the rendered shorts only.")

    files = rendered_files(job_dir, raw_shorts, title)
    ver = load_json(os.path.join(job_dir, VERIFICATION_FILE))
    ver_rows = {}
    for row in (ver or {}).get("shorts", []) if isinstance(ver, dict) else []:
        if isinstance(row, dict) and as_int(row.get("short_index")) is not None:
            ver_rows[as_int(row["short_index"])] = row

    for i, s in enumerate(shorts):
        if s["rank"] is None:
            s["rank"] = i + 1
        s["file"] = files.get(i)
        s["file_exists"] = bool(s["file"] and resolve_in(job_dir, s["file"]))
        row = ver_rows.get(i) or {}
        result = row.get("result") if isinstance(row.get("result"), dict) else None
        s["captions_file"] = safe_basename(row.get("captions"))
        s["verification"] = result
        s["verified_ok"] = result.get("ok") if result else None

    # Which candidates were rendered.
    for c in candidates:
        c["rendered"] = False
        c["file"] = None
        for s in shorts:
            if _same_clip(c, s) or (raw_cands is None and c["position"] == s["position"]):
                c["rendered"] = bool(s["file"])
                c["file"] = s["file"]
                if c["rank"] is None:
                    c["rank"] = s["rank"]
                break

    def sort_key(c):
        return (c["rank"] if c["rank"] is not None else 10 ** 6,
                -(c["overall"] if c["overall"] is not None else -1),
                c["position"])
    candidates.sort(key=sort_key)

    # Edge cases.
    for c in candidates:
        label = f"candidate #{c['candidate_index'] if c['candidate_index'] is not None else c['position']}"
        if not c["rated"]:
            issues.append(f"{label} has no rating (or no overall score).")
        if c["duration"] is not None and c["duration"] > 60:
            issues.append(f"{label} is {c['duration']:.1f}s, longer than a 60 s Short.")
        if c["start"] is None or c["end"] is None:
            issues.append(f"{label} has no numeric start/end.")
    ranks = [c["rank"] for c in candidates if c["rank"] is not None]
    if len(ranks) != len(set(ranks)):
        issues.append("Duplicate ranks among candidates.")
    for s in shorts:
        tag = f"short rank {s['rank']}"
        if not s["file"]:
            issues.append(f"{tag}: no rendered file found (no CLIP_READY and nothing on disk).")
        elif not s["file_exists"]:
            issues.append(f"{tag}: file {s['file']} is missing from the job directory.")
        if s["verification"] is None:
            issues.append(f"{tag}: not verified (run the verify stage).")
        else:
            for chk in s["verification"].get("checks") or []:
                if isinstance(chk, dict) and chk.get("ok") is False:
                    sev = "warning" if chk.get("informational") else "FAILED"
                    issues.append(f"{tag}: {sev} {text(chk.get('name'), 40)}: "
                                  f"{text(chk.get('detail'), 200)}")
        if not s["title"]:
            issues.append(f"{tag}: no YouTube title.")
        if not text(s["description"]):
            issues.append(f"{tag}: no YouTube description.")
        if not s["hashtags"]:
            issues.append(f"{tag}: no hashtags.")

    cost = cost_summary(metadata)
    if metadata and cost is None:
        issues.append("No Gemini cost recorded in metadata (cost_analysis missing).")

    st = load_json(os.path.join(job_dir, STAGES_FILE))
    stages, exit_code = [], None
    if isinstance(st, dict) and isinstance(st.get("stages"), list):
        for s in st["stages"]:
            if isinstance(s, dict):
                stages.append({"name": text(s.get("name"), 40),
                               "start_s": num(s.get("start_s")),
                               "end_s": num(s.get("end_s")),
                               "duration_s": num(s.get("duration_s"))})
        exit_code = as_int(st.get("exit_code"))
    else:
        tracker = stages_from_log(os.path.join(job_dir, LOG_FILE))
        if tracker:
            stages = tracker.summary()
    if exit_code not in (None, 0):
        issues.append(f"main.py exited with code {exit_code}.")
    if isinstance(st, dict):
        for err in st.get("errors") or []:
            issues.append(text(err, 300))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_dir": os.path.basename(os.path.abspath(job_dir)),
        "metadata_file": os.path.basename(meta_path) if meta_path else None,
        "video_title": title,
        "source_video": safe_basename(metadata.get("source_video")),
        "candidates": candidates,
        "shorts": shorts,
        "cost": cost,
        "stages": stages,
        "exit_code": exit_code,
        "issues": issues,
    }
    return render_markdown(report), report


def _fmt(value, digits=0):
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def render_markdown(r):
    out = [f"# Shorts report: {cell(r.get('video_title') or r.get('job_dir'), 200)}", ""]
    out.append(f"Generated {r['generated_at']}. Metadata: "
               f"`{cell(r.get('metadata_file') or 'none')}`. "
               f"{len(r['candidates'])} candidate(s), {len(r['shorts'])} rendered.")
    out.append("")
    out.append("Nothing in this run is uploaded: every short waits for manual approval "
               "(`publish/approval.json`).")
    out.append("")

    out += ["## Ranked candidates", ""]
    head = (["Rank", "Cand #", "Source", "Dur (s)", "Overall"]
            + [CRITERIA_SHORT[k] for k in CRITERIA] + ["Summary", "Rendered"])
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join("---" for _ in head) + "|")
    for c in r["candidates"]:
        row = [_fmt(c["rank"]) if c["rank"] is not None else "–",
               str(c["candidate_index"]) if c["candidate_index"] is not None else "–",
               f"{hms(c['start'])}–{hms(c['end'])}",
               _fmt(c["duration"], 1),
               _fmt(c["overall"])]
        row += [_fmt(c["scores"].get(k), 1) for k in CRITERIA]
        row.append(cell(c["summary"] or "(unrated)", 140))
        row.append(cell(c["file"]) if c["rendered"] else "no")
        out.append("| " + " | ".join(row) + " |")
    if not r["candidates"]:
        out.append("| " + " | ".join("–" for _ in head) + " |")
    out.append("")

    out += ["## Rendered shorts", ""]
    if not r["shorts"]:
        out += ["_No short was rendered._", ""]
    for s in r["shorts"]:
        out.append(f"### Rank {s['rank']}: {cell(s['title'] or '(untitled)', 200)}")
        out.append("")
        out.append(f"- Source: {hms(s['start'])}–{hms(s['end'])} "
                   f"({_fmt(s['duration'], 1)} s), overall {_fmt(s['overall'])}")
        out.append(f"- File: `{cell(s['file'] or 'missing')}`"
                   + ("" if s["file_exists"] or not s["file"] else " (not on disk)"))
        if s.get("captions_file"):
            out.append(f"- Captions: `{cell(s['captions_file'])}`")
        out.append(f"- Hook text: {cell(s['hook'] or '(none)', 300)}")
        out.append(f"- Hashtags: {cell(' '.join(s['hashtags']) or '(none)', 400)}")
        out.append("")
        out.append("Description:")
        out.append("")
        out.append(para(s["description"]))
        out.append("")
        if s["summary"]:
            out.append(f"Rating summary: {cell(s['summary'], 400)}")
            out.append("")
        ver = s.get("verification")
        if ver:
            status = {True: "PASS", False: "FAIL", None: "N/A"}
            out.append(f"Verification: **{status.get(ver.get('ok'), 'N/A')}**")
            out.append("")
            out.append("| Check | Result | Detail |")
            out.append("|---|---|---|")
            for chk in ver.get("checks") or []:
                if not isinstance(chk, dict):
                    continue
                res = status.get(chk.get("ok"), "N/A")
                if chk.get("informational") and chk.get("ok") is False:
                    res = "WARN"
                out.append(f"| {cell(chk.get('name'), 40)} | {res} | "
                           f"{cell(chk.get('detail'), 200)} |")
            out.append("")
        else:
            out += ["_Not verified._", ""]

    out += ["## Gemini cost", ""]
    cost = r.get("cost")
    if cost:
        out.append(f"Total: **${cost['total_usd']:.4f}**"
                   + (f" ({cell(cost['model'], 80)})" if cost.get("model") else ""))
        out.append("")
        out.append("| Stage | USD |")
        out.append("|---|---|")
        for k, v in cost["per_stage"].items():
            out.append(f"| {cell(k, 60)} | {v:.4f} |")
        if cost.get("tokens"):
            out.append("")
            out.append("Tokens: " + ", ".join(f"{k} {v}" for k, v in cost["tokens"].items()))
        out.append("")
    else:
        out += ["_No cost recorded._", ""]

    out += ["## Stage timings", ""]
    if r.get("stages"):
        out.append("| Stage | Start (s) | Duration (s) |")
        out.append("|---|---|---|")
        for s in r["stages"]:
            out.append(f"| {cell(s['name'], 40)} | {_fmt(s.get('start_s'), 1)} | "
                       f"{_fmt(s.get('duration_s'), 1)} |")
        out.append("")
    else:
        out += ["_No stage timings (no stages.json or pipeline.log)._", ""]

    out += ["## Errors and edge cases", ""]
    if r["issues"]:
        out += [f"- {cell(i, 400)}" for i in r["issues"]]
    else:
        out.append("_None._")
    out.append("")
    return "\n".join(out)


def write_report(job_dir):
    md, data = build_report(job_dir)
    with open(os.path.join(job_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(job_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return md, data


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python shorts_report.py <job_dir>", file=sys.stderr)
        return 2
    job_dir = argv[0]
    if not os.path.isdir(job_dir):
        print(f"not a directory: {job_dir}", file=sys.stderr)
        return 2
    _, data = write_report(job_dir)
    print(f"report.md / report.json written: {len(data['candidates'])} candidate(s), "
          f"{len(data['shorts'])} short(s), {len(data['issues'])} issue(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
