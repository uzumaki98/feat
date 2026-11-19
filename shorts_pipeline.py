"""One command from a source URL to YouTube Shorts awaiting manual approval.

    python shorts_pipeline.py <url> --minutes 90 --top 3 [--out output/<slug>]
                              [--stage all|run|edit|verify|report|package] [--model ...]
                              [--edit full|none]

Stages (``all`` runs them in order; each one re-runs alone on an existing
out dir):

- run:     main.py with layout picker, hook, captions, rating and top-N render
           on; its output is streamed to the console and to ``pipeline.log``
           (each line stamped with elapsed seconds), stage boundaries and the
           CLIP_READY records go to ``stages.json``.
- edit:    the house edit (shorts_variants.full_edit, ``--edit full``, the
           default) on every rendered short, from its clip before captions
           and hook: quiet pauses trimmed, a cold open, punch-ins, one-word
           captions, hook, sound effects, -14 LUFS -> ``edited_clip_<n>.mp4``
           recorded in ``edits.json``; verify and package then use it. A short
           the edit fails on keeps main.py's render.
- verify:  verify_shorts.check_short on every rendered short ->
           ``verification.json``.
- report:  shorts_report.build_report -> ``report.md`` / ``report.json``.
- package: ``publish/short_<rank>/`` (video, title, description, hashtags,
           metadata) and ``publish/approval.json`` with every short
           ``pending_approval``.

Nothing is posted by these stages. Posting is separate and only on request
(shorts_publish.py): ``--stage approve --ranks 1,2`` records a person's
decision, ``--stage publish`` posts the approved shorts to YouTube through
Upload-Post (``UPLOAD_POST_API_KEY``), ``--stage post-status`` asks how each
post ended. ``all`` never runs them.
Paths go through os.path only, so it runs the same inside the Linux backend
container (cwd /app) as on a dev machine.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

import shorts_report as sr
import verify_shorts

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("run", "edit", "verify", "report", "package")
# Run only when asked by name, never by "all": a person decides, then posts.
MANUAL_STAGES = ("approve", "publish", "post-status")
PUBLISH_RECORD = ("decided_at", "posted_at", "hashtags_sent", "credit_sent", "profile", "privacy", "request_id",
                  "scheduled_for", "timezone", "error", "tried_at", "upload_status",
                  "youtube_ok", "youtube_url", "youtube_message", "checked_at")
PUBLISH_DIR = "publish"
APPROVAL_FILE = "approval.json"
EDITS_FILE = "edits.json"
EDIT_STYLES = ("full", "none")
# main.py's served name around the clip it was made from:
# subtitled_<gen>_hooked_<gen>_<clip>.mp4 (either prefix optional).
_SERVED = re.compile(r"(?:subtitled_\d+_)?(?:hooked_\d+_)?(?P<clip>[^/\\]+\.mp4)")


def _say(line):
    """Print a line on any console (a cp1252 one included)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def slug_for(url):
    """Directory name for a URL: the YouTube id when there is one."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        parsed = None
    vid = None
    if parsed and parsed.hostname:
        host = parsed.hostname.lower()
        if host.endswith("youtu.be"):
            vid = parsed.path.strip("/").split("/")[0]
        elif "youtube" in host:
            vid = (parse_qs(parsed.query).get("v") or [None])[0]
            if not vid:
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) >= 2 and parts[0] in ("shorts", "live", "embed"):
                    vid = parts[1]
    if vid and re.fullmatch(r"[A-Za-z0-9_-]{6,20}", vid):
        return vid
    base = re.sub(r"[^A-Za-z0-9]+", "-", (parsed.path if parsed else "") or "").strip("-")[-40:]
    digest = hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:10]
    return f"{base}-{digest}" if base else digest


# --- stage: run ---------------------------------------------------------------

def child_env(minutes, top, model=None, clip_workers=None, base=None):
    env = dict(os.environ if base is None else base)
    env.update({
        "AUTO_LAYOUT": "1",
        "AUTO_HOOK": "1",
        "AUTO_CAPTIONS": "1",
        "RATE_CLIPS": "1",
        "RENDER_TOP_N": str(int(top)),
        "MAX_SOURCE_MINUTES": f"{float(minutes):g}",
    })
    env["CLIP_WORKERS"] = (str(clip_workers) if clip_workers
                           else env.get("CLIP_WORKERS") or default_clip_workers())
    if model:
        env["GEMINI_MODEL"] = model
    # main.py prints emoji from its first line; on a cp1252 console the child
    # would die before rendering anything (same fix app.py applies).
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def default_clip_workers(which=shutil.which):
    """Clips rendered at once. One on a GPU host (a small card runs out of
    memory with several renders); three on a CPU-only host, where each render
    is mostly single-threaded ffmpeg work and the cores are otherwise idle."""
    return "1" if which("nvidia-smi") else "3"


def main_command(url, out_dir, input_path=None):
    """main.py on the URL, or on a local file already on disk (``input_path``:
    an upload, or the source a stopped run left behind; the URL is then only
    recorded)."""
    source = ["-i", input_path] if input_path else ["-u", url]
    return [sys.executable, "-u", os.path.join(HERE, "main.py"), *source,
            "--keep-original", "-o", out_dir, "--format", "vertical"]


def run_stage(url, out_dir, minutes, top, model=None, clip_workers=None,
              popen=subprocess.Popen, clock=time.monotonic, input_path=None):
    """Run main.py, stream and stamp its output, record stages. Returns the
    StageTracker summary dict written to stages.json."""
    os.makedirs(out_dir, exist_ok=True)
    tracker = sr.StageTracker()
    started = clock()
    state = {"url": url, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "exit_code": None, "stages": [], "clip_ready": {}, "errors": []}
    stages_path = os.path.join(out_dir, sr.STAGES_FILE)

    def save():
        state["stages"] = tracker.summary()
        state["clip_ready"] = {str(k): v for k, v in sorted(tracker.clip_ready.items())}
        state["metadata_file"] = tracker.metadata_file
        state["clips_found"] = tracker.clips_found
        _write_json(stages_path, state)

    _write_json(os.path.join(out_dir, sr.RUN_FILE), {
        "url": url, "minutes": minutes, "top": top, "model": model,
        "clip_workers": clip_workers, "started_at": state["started_at"]})

    cmd = main_command(url, os.path.abspath(out_dir),
                       os.path.abspath(input_path) if input_path else None)
    env = child_env(minutes, top, model, clip_workers)
    log_path = os.path.join(out_dir, sr.LOG_FILE)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"[{0.0:8.1f}s] === shorts_pipeline run: main.py, "
                  f"{minutes:g} min, top {top} ===\n")
        try:
            proc = popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
        except OSError as e:
            state["errors"].append(f"run: could not start main.py: {e}")
            state["exit_code"] = -1
            save()
            return state
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                elapsed = clock() - started
                stamped = f"[{elapsed:8.1f}s] {line}"
                log.write(stamped + "\n")
                log.flush()
                _say(stamped)
                if tracker.feed(line, elapsed):
                    save()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            state["errors"].append("run: interrupted")
            code = -2
        tracker.finish(clock() - started)
        state["exit_code"] = code
        if code != 0:
            msg = f"run: main.py exited with code {code}"
            state["errors"].append(msg)
            log.write(f"[{clock() - started:8.1f}s] {msg}\n")
            _say(msg)
        save()
    return state


# --- shared lookups -----------------------------------------------------------

def load_job(out_dir):
    """(metadata dict or {}, video_title or None, metadata basename or None)."""
    path = sr.find_metadata(out_dir)
    data = sr.load_json(path) if path else None
    return (data if isinstance(data, dict) else {}), sr.video_title_of(path), \
        (os.path.basename(path) if path else None)


def rendered_shorts(out_dir, metadata, video_title):
    """[{short_index, rank, short, file}] for every short in metadata, file
    from CLIP_READY (stages.json) else from the files on disk (None if none)."""
    shorts = metadata.get("shorts") if isinstance(metadata.get("shorts"), list) else []
    st = sr.load_json(os.path.join(out_dir, sr.STAGES_FILE))
    ready = st.get("clip_ready") if isinstance(st, dict) else None
    ready = ready if isinstance(ready, dict) else {}
    edits = sr.load_json(os.path.join(out_dir, EDITS_FILE))
    edits = edits.get("shorts") if isinstance(edits, dict) else None
    edits = edits if isinstance(edits, dict) else {}
    out = []
    for i, short in enumerate(shorts):
        short = short if isinstance(short, dict) else {}
        name = sr.safe_basename(ready.get(str(i)))
        if name and not sr.resolve_in(out_dir, name):
            name = None
        if not name:
            name = sr.find_served_file(out_dir, video_title, i)
        item = {"short_index": i, "rank": sr.as_int(short.get("rank"), i + 1),
                "short": short, "file": name, "edit": None}
        edit = edits.get(str(i)) if isinstance(edits.get(str(i)), dict) else None
        if edit and name and edit.get("source_file") == name:
            edited = sr.safe_basename(edit.get("file"))
            if edited and sr.resolve_in(out_dir, edited):
                item.update(file=edited, edit=edit)
        out.append(item)
    return out


# --- stage: edit --------------------------------------------------------------

def base_clip(out_dir, served):
    """The clip ``served`` was made from (before captions and hook), or None."""
    m = _SERVED.fullmatch(served or "")
    name = sr.safe_basename(m.group("clip")) if m else None
    return sr.resolve_in(out_dir, name) if name else None


def clip_words(words, start, end):
    """``words`` (source seconds) inside [start, end], in clip seconds."""
    if start is None:
        return []
    out = []
    for w in words:
        if w["end"] > start and (end is None or w["start"] < end):
            s, e = max(0.0, w["start"] - start), w["end"] - start
            if end is not None:
                e = min(e, end - start)
            if e > s:
                out.append({"word": w["word"], "start": round(s, 3), "end": round(e, 3)})
    return out


def edit_words(out_dir, edit):
    """The words an edit placed on its own timeline (verify's format), or None."""
    name = sr.safe_basename(edit.get("words"))
    data = sr.load_json(os.path.join(out_dir, name)) if name else None
    if not isinstance(data, list):
        return None
    return verify_shorts.transcript_words({"segments": [{"words": data}]})


def edit_stage(out_dir, style="full", editor=None, say=None):
    """The house edit on every rendered short (see the module docstring).
    ``editor`` is shorts_variants.full_edit (a test passes a fake)."""
    say = say or _say
    record = {"style": style, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "shorts": {}, "errors": []}
    path = os.path.join(out_dir, EDITS_FILE)
    # Cleared first: rendered_shorts must list main.py's files, not old edits.
    _write_json(path, record)
    if style == "none":
        return record
    if editor is None:
        import shorts_variants
        editor = shorts_variants.full_edit
    metadata, title, _ = load_job(out_dir)
    words = verify_shorts.transcript_words(metadata.get("transcript") or {})
    for item in rendered_shorts(out_dir, metadata, title):
        n = item["short_index"] + 1
        if not item["file"]:
            continue
        clip = base_clip(out_dir, item["file"])
        if not clip:
            record["errors"].append(f"edit: short rank {item['rank']}: no source clip "
                                    f"for {item['file']}; kept as rendered")
            continue
        cand = sr.normalise_candidate(item["short"], item["short_index"])
        name = f"edited_clip_{n}.mp4"
        work = os.path.join(out_dir, f".edit_work_{n}")
        try:
            done = editor(clip, clip_words(words, cand["start"], cand["end"]),
                          cand["hook"] or "", os.path.join(out_dir, name), work)
        except Exception as e:  # keep main.py's render for this one
            record["errors"].append(f"edit: short rank {item['rank']}: "
                                    f"{type(e).__name__}: {e}"[:400])
            say(f"edit: rank {item['rank']} failed, kept as rendered ({type(e).__name__})")
            continue
        entry = {"file": name, "source_file": item["file"], "captions": None,
                 "words": f"edited_clip_{n}.words.json"}
        if done.get("captions") and os.path.isfile(done["captions"]):
            entry["captions"] = f"edited_clip_{n}.ass"
            shutil.copyfile(done["captions"], os.path.join(out_dir, entry["captions"]))
        _write_json(os.path.join(out_dir, entry["words"]), done.get("words") or [])
        entry.update({k: done.get(k) for k in ("seconds", "source_seconds", "cuts",
                                               "trimmed_seconds", "cold_open_from",
                                               "punch_ins", "sfx")})
        record["shorts"][str(item["short_index"])] = entry
        _write_json(path, record)
        shutil.rmtree(work, ignore_errors=True)
        say(f"edit: rank {item['rank']} -> {name} ({entry['seconds']}s)")
    _write_json(path, record)
    return record


# --- stage: verify ------------------------------------------------------------

def verify_stage(out_dir, max_seconds=60.0, min_seconds=5.0, check=None):
    check = check or verify_shorts.check_short
    metadata, title, meta_name = load_job(out_dir)
    words = verify_shorts.transcript_words(metadata.get("transcript") or {})
    rows, errors = [], []
    if not meta_name:
        errors.append("verify: no *_metadata.json in the out dir")
    for item in rendered_shorts(out_dir, metadata, title):
        short = item["short"]
        start, end = sr.num(short.get("start")), sr.num(short.get("end"))
        row = {"short_index": item["short_index"], "rank": item["rank"],
               "candidate_index": sr.as_int(short.get("candidate_index")),
               "file": item["file"], "captions": None, "result": None}
        if not item["file"]:
            errors.append(f"verify: short rank {item['rank']} has no rendered file")
            rows.append(row)
            continue
        clip_words_, clip_start = words, start
        if item["edit"]:
            # The edit has its own timeline: its captions and the words it
            # placed on it (clip seconds), not the source transcript.
            captions = sr.safe_basename(item["edit"].get("captions"))
            if captions and not sr.resolve_in(out_dir, captions):
                captions = None
            clip_words_ = edit_words(out_dir, item["edit"])
            clip_start = 0.0
        else:
            captions = sr.find_caption_sidecar(out_dir, item["file"], words, start, end)
        row["captions"] = captions
        try:
            result = check(os.path.join(out_dir, item["file"]),
                           max_seconds=max_seconds, min_seconds=min_seconds,
                           captions_path=os.path.join(out_dir, captions) if captions else None,
                           transcript_words=clip_words_ if clip_words_ else None,
                           clip_start=clip_start)
            result = dict(result)
            result["file"] = item["file"]
        except Exception as e:  # one bad file must not stop the others
            result = {"file": item["file"], "ok": False,
                      "checks": [{"name": "verify", "ok": False,
                                  "detail": f"{type(e).__name__}: {e}"[:300]}]}
        row["result"] = result
        rows.append(row)
        status = {True: "PASS", False: "FAIL", None: "N/A"}.get(result.get("ok"), "N/A")
        _say(f"verify: rank {item['rank']} {status} {item['file']}")
    data = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metadata_file": meta_name, "max_seconds": max_seconds,
            "min_seconds": min_seconds, "shorts": rows, "errors": errors}
    _write_json(os.path.join(out_dir, sr.VERIFICATION_FILE), data)
    return data


# --- stage: report ------------------------------------------------------------

def report_stage(out_dir):
    md, data = sr.write_report(out_dir)
    _say(f"report: {os.path.join(out_dir, 'report.md')} "
         f"({len(data['issues'])} issue(s))")
    return data


# --- stage: package -----------------------------------------------------------

def _place(src, dst):
    """Copy ``src`` to ``dst``. A copy, not a hard link: a re-run that
    rewrites the render in place (ffmpeg -y keeps the inode) must never
    change a short someone already reviewed."""
    if os.path.lexists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verification_summary(row):
    result = (row or {}).get("result") if isinstance(row, dict) else None
    if not isinstance(result, dict):
        return {"ok": None, "failed": [], "warnings": [], "checks": []}
    checks = [c for c in result.get("checks") or [] if isinstance(c, dict)]
    return {
        "ok": result.get("ok"),
        "failed": [sr.text(c.get("name"), 40) for c in checks
                   if c.get("ok") is False and not c.get("informational")],
        "warnings": [sr.text(c.get("name"), 40) for c in checks
                     if c.get("ok") is False and c.get("informational")],
        "checks": [{"name": sr.text(c.get("name"), 40), "ok": c.get("ok"),
                    "detail": sr.text(c.get("detail"), 300)} for c in checks],
    }


def package_stage(out_dir):
    metadata, title, meta_name = load_job(out_dir)
    ver = sr.load_json(os.path.join(out_dir, sr.VERIFICATION_FILE))
    ver_rows = {}
    for row in (ver or {}).get("shorts", []) if isinstance(ver, dict) else []:
        if isinstance(row, dict) and sr.as_int(row.get("short_index")) is not None:
            ver_rows[sr.as_int(row["short_index"])] = row
    run = sr.load_json(os.path.join(out_dir, sr.RUN_FILE))
    run = run if isinstance(run, dict) else {}

    publish = os.path.join(out_dir, PUBLISH_DIR)
    os.makedirs(publish, exist_ok=True)
    approval_path = os.path.join(publish, APPROVAL_FILE)
    previous = sr.load_json(approval_path)
    prev_status = {}
    for e in (previous or {}).get("shorts", []) if isinstance(previous, dict) else []:
        if isinstance(e, dict):
            # A review belongs to the exact video that was reviewed, not to a
            # rank or a file name that a re-run may give to another clip.
            if isinstance(e.get("sha256"), str):
                prev_status[e["sha256"]] = e

    entries, errors, used_ranks = [], [], set()
    for item in rendered_shorts(out_dir, metadata, title):
        short = item["short"]
        rank = item["rank"]
        if rank in used_ranks or rank is None or rank < 1:
            rank = max(used_ranks | {0}) + 1
        used_ranks.add(rank)
        src = sr.resolve_in(out_dir, item["file"]) if item["file"] else None
        if not src:
            errors.append(f"package: short rank {rank} has no rendered file; skipped")
            continue
        cand = sr.normalise_candidate(short, item["short_index"])
        name = f"short_{rank}"
        dest = os.path.join(publish, name)
        os.makedirs(dest, exist_ok=True)
        video = f"{name}.mp4"
        _place(src, os.path.join(dest, video))
        digest = _sha256(os.path.join(dest, video))
        description = cand["description"] if isinstance(cand["description"], str) else ""
        with open(os.path.join(dest, "title.txt"), "w", encoding="utf-8") as f:
            f.write(cand["title"] + "\n")
        with open(os.path.join(dest, "description.txt"), "w", encoding="utf-8") as f:
            f.write(description.replace("\r\n", "\n").strip() + "\n")
        with open(os.path.join(dest, "hashtags.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(cand["hashtags"]) + "\n")
        summary = _verification_summary(ver_rows.get(item["short_index"]))
        meta = {
            "rank": rank,
            "candidate_index": cand["candidate_index"],
            "title": cand["title"],
            "description": description,
            "hashtags": cand["hashtags"],
            "hook_text": cand["hook"],
            "rating": short.get("rating") if isinstance(short.get("rating"), dict) else None,
            "source": {
                "url": sr.text(run.get("url"), 2000) or None,
                "video": sr.safe_basename(metadata.get("source_video")),
                "start": cand["start"], "end": cand["end"],
                "start_hms": sr.hms(cand["start"]), "end_hms": sr.hms(cand["end"]),
                "duration": cand["duration"],
            },
            "source_file": item["file"],
            "captions_file": sr.safe_basename((ver_rows.get(item["short_index"]) or {})
                                              .get("captions")),
            "verification": summary,
        }
        _write_json(os.path.join(dest, "metadata.json"), meta)
        prev = prev_status.get(digest) or {}
        status = prev.get("status") if prev.get("status") in (
            "pending_approval", "approved", "rejected", "posted",
            "publish_failed") else "pending_approval"
        entry = {
            "rank": rank, "dir": name, "video": f"{name}/{video}",
            "title": cand["title"], "overall": cand["overall"],
            "source_file": item["file"], "sha256": digest,
            "verification_ok": summary["ok"], "status": status,
        }
        # A decision and a post belong to these bytes: carry their record over,
        # so a re-run of package can never make a posted short postable again.
        for k in PUBLISH_RECORD:
            if k in prev:
                entry[k] = prev[k]
        entries.append(entry)
        _say(f"package: {name} <- {item['file']} ({status})")

    entries.sort(key=lambda e: e["rank"])
    keep = {e["dir"] for e in entries}
    for stale in os.listdir(publish):
        if re.fullmatch(r"short_\d+", stale) and stale not in keep \
                and os.path.isdir(os.path.join(publish, stale)) \
                and not os.path.islink(os.path.join(publish, stale)):
            shutil.rmtree(os.path.join(publish, stale))
    approval = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_url": sr.text(run.get("url"), 2000) or None,
                "metadata_file": meta_name,
                "note": "Nothing has been uploaded. Set status to approved or "
                        "rejected per short after review.",
                "shorts": entries, "errors": errors}
    _write_json(approval_path, approval)
    return approval


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="URL -> verified, packaged YouTube Shorts "
                                            "awaiting manual approval (no upload).")
    p.add_argument("url", nargs="?", help="source URL (needed for the run stage)")
    p.add_argument("--minutes", type=float, default=90.0,
                   help="process only the first N minutes of the source")
    p.add_argument("--top", type=int, default=3, help="render the N best candidates")
    p.add_argument("--out", help="output directory (default output/<slug>)")
    p.add_argument("--stage", choices=("all",) + STAGES + MANUAL_STAGES, default="all")
    p.add_argument("--ranks", help="approve: comma list of short ranks, e.g. 1,2,3")
    p.add_argument("--decision", default="approved",
                   choices=("approved", "rejected", "pending_approval"))
    p.add_argument("--profile", help="publish: Upload-Post profile (default: the one "
                                     "with YouTube connected)")
    p.add_argument("--privacy", default="public", choices=("public", "unlisted", "private"))
    p.add_argument("--start", help="publish: schedule the first post at this ISO time; "
                                   "omit to post now")
    p.add_argument("--every-hours", type=float, default=24.0,
                   help="publish: hours between scheduled posts")
    p.add_argument("--timezone", default="UTC", help="publish: IANA timezone of --start")
    p.add_argument("--hashtags", help="publish: hashtags put first on every post, e.g. "
                                      "'#GameName,#SomeStreamer' (the clip's own follow, #Shorts last)")
    p.add_argument("--credit", metavar="NAME",
                   help="publish: creator named in the credit line (default: the source's "
                        "channel when known); the line links the moment in the stream")
    p.add_argument("--no-credit", action="store_true",
                   help="publish: leave the credit line out")
    p.add_argument("--dry-run", action="store_true",
                   help="publish: show what would be sent, send nothing")
    p.add_argument("--model", help="GEMINI_MODEL override")
    p.add_argument("--clip-workers", type=int, default=None,
                   help="CLIP_WORKERS for main.py (default: the env value, else 1 "
                        "with a GPU and 3 without)")
    p.add_argument("--input", help="run on this local video file instead of "
                                   "downloading the URL (the URL is still recorded)")
    p.add_argument("--edit", choices=EDIT_STYLES, default="full",
                   help="edit stage: the house edit (full) or main.py's render as is (none)")
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--min-seconds", type=float, default=5.0)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.top < 1:
        p.error("--top must be at least 1")
    if args.minutes <= 0:
        p.error("--minutes must be positive")
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.stage in MANUAL_STAGES:
        if not args.out:
            p.error(f"--stage {args.stage} needs --out (the job directory)")
        return manual_stage(args)
    if "run" in stages and not (args.url or args.input):
        p.error("the run stage needs a URL or --input")
    if args.input and not os.path.isfile(args.input):
        p.error(f"--input file not found: {args.input}")
    if not args.out and not args.url:
        p.error("give --out (or a URL to derive it from)")

    out_dir = args.out or os.path.join("output", slug_for(args.url))
    os.makedirs(out_dir, exist_ok=True)

    failed = []
    for stage in stages:
        _say(f"=== stage {stage} ===")
        try:
            if stage == "run":
                state = run_stage(args.url, out_dir, args.minutes, args.top,
                                  args.model, args.clip_workers, input_path=args.input)
                if state.get("exit_code") != 0:
                    failed.append(stage)
            elif stage == "edit":
                if edit_stage(out_dir, args.edit)["errors"]:
                    failed.append(stage)
            elif stage == "verify":
                data = verify_stage(out_dir, args.max_seconds, args.min_seconds)
                if data["errors"] or any((r.get("result") or {}).get("ok") is False
                                         for r in data["shorts"]):
                    failed.append(stage)
            elif stage == "report":
                report_stage(out_dir)
            elif stage == "package":
                approval = package_stage(out_dir)
                if approval["errors"] or not approval["shorts"]:
                    failed.append(stage)
        except Exception as e:  # later stages still run on whatever exists
            _say(f"stage {stage} failed: {type(e).__name__}: {e}")
            failed.append(stage)
    _say(f"done: {out_dir}" + (f" (problems in: {', '.join(failed)})" if failed else ""))
    return 1 if failed else 0


def manual_stage(args):
    import httpx
    import shorts_publish as pub
    try:
        # The key lives in .env like the rest of the config (main.py and app.py
        # load it the same way); never on the command line.
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, ".env"))
    except ImportError:
        pass
    try:
        if args.stage == "approve":
            if not args.ranks:
                _say("approve needs --ranks, e.g. --ranks 1,2,3")
                return 2
            ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
            for rank, status in pub.approve(args.out, ranks, args.decision):
                _say(f"approve: rank {rank} -> {status}")
            return 0
        with httpx.Client(timeout=60.0) as http:
            if args.stage == "publish":
                acted = pub.publish(args.out, http=http, profile=args.profile,
                                    privacy=args.privacy, start=args.start,
                                    every_hours=args.every_hours, timezone=args.timezone,
                                    dry_run=args.dry_run, say=_say,
                                    pinned_tags=[t for t in (args.hashtags or "").replace(" ", ",")
                                                 .split(",") if t.strip()],
                                    credit=not args.no_credit, credit_name=args.credit)
                return 1 if any(e.get("status") == "publish_failed" for e in acted) else 0
            pub.post_status(args.out, http=http, say=_say)
            return 0
    except (pub.PublishError, ValueError, OSError) as e:
        _say(f"{args.stage}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
