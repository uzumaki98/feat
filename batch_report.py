"""Global report and publish package of a chunked long-stream run.

    python batch_report.py --out <dir> [--top 10]

A long live stream is processed in chunks (stream_batch.py): ``<out>/fetch_plan.json``
lists the chunks (``index``, ``dir``, ``stream_start_seconds``, ``seconds``) and
every ``<out>/chunk_NN/`` is an ordinary shorts_pipeline job directory. This
module ranks every candidate of every chunk on one stream timeline, picks the
global top N among the rendered shorts that passed verification, packages
them into ``<out>/publish/`` (``short_<rank>/`` + ``approval.json``, the shape
shorts_publish.py reads) and writes ``<out>/report.md`` + ``report.json``.

Everything read from a chunk directory is data from another process (or
another machine): numbers are coerced to numbers, text is escaped before it
lands in Markdown, and files are only ever taken as plain names inside the
chunk directory. A failed chunk (no metadata, no package) degrades the report
and is listed under errors; it never stops the others.
"""
import argparse
import os
import re
import shutil
import sys
import time

import shorts_report as sr
from shorts_pipeline import PUBLISH_RECORD, _sha256, _verification_summary, _write_json

PLAN_FILE = "fetch_plan.json"
FETCHED_FILE = "fetched.json"
DONE_FILE = "batch_done.json"
PUBLISH_DIR = "publish"
APPROVAL_FILE = "approval.json"
STATUSES = ("pending_approval", "approved", "rejected", "posted", "publish_failed")
_SHORT_DIR = re.compile(r"short_(\d+)")
_CHUNK_DIR = re.compile(r"chunk_\d+")


def _say(line):
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


# --- chunks -------------------------------------------------------------------

def _inside(root, path):
    """True when ``path`` (not a link) resolves inside ``root``."""
    if os.path.islink(path):
        return False
    real, base = os.path.realpath(path), os.path.realpath(root)
    try:
        return os.path.commonpath([real, base]) == base and real != base
    except ValueError:
        return False


def chunk_dirs(out_dir, plan, issues):
    """[{index, name, path, stream_start, seconds}] from fetch_plan.json, else
    from the ``chunk_NN`` directories on disk (stream offsets then unknown)."""
    chunks = []
    if isinstance(plan, dict) and isinstance(plan.get("chunks"), list):
        for pos, c in enumerate(plan["chunks"]):
            c = _dict(c)
            index = sr.as_int(c.get("index"), pos)
            name = sr.safe_basename(c.get("dir"))
            if not name:
                issues.append(f"chunk {index}: fetch_plan dir "
                              f"{sr.text(c.get('dir'), 80)!r} is not a plain name; skipped.")
                continue
            chunks.append({"index": index, "name": name,
                           "path": os.path.join(out_dir, name),
                           "stream_start": sr.num(c.get("stream_start_seconds")),
                           "seconds": sr.num(c.get("seconds"))})
    else:
        issues.append(f"No readable {PLAN_FILE}: chunks taken from disk, stream "
                      "offsets unknown.")
        try:
            names = sorted(n for n in os.listdir(out_dir) if _CHUNK_DIR.fullmatch(n))
        except OSError:
            names = []
        for name in names:
            chunks.append({"index": int(name.split("_")[1]), "name": name,
                           "path": os.path.join(out_dir, name),
                           "stream_start": None, "seconds": None})
    out = []
    for c in chunks:
        if not _inside(out_dir, c["path"]) or not os.path.isdir(c["path"]):
            issues.append(f"chunk {c['index']}: directory {sr.text(c['name'], 80)} missing.")
            c["missing"] = True
        out.append(c)
    return out


def _chunk_short_file(chunk_path, rank, approval_entry):
    """Packaged mp4 of chunk short ``rank``, only inside ``<chunk>/publish/``."""
    publish = os.path.join(chunk_path, PUBLISH_DIR)
    names = []
    if approval_entry:
        rel = sr.text(approval_entry.get("video"), 200)
        parts = rel.split("/")
        if len(parts) == 2 and _SHORT_DIR.fullmatch(parts[0]) \
                and parts[1] == parts[0] + ".mp4":
            names.append(parts[0])
    if rank is not None:
        names.append(f"short_{rank}")
    for name in names:
        d = os.path.join(publish, name)
        if not _inside(chunk_path, d) or not os.path.isdir(d):
            continue
        path = sr.resolve_in(d, name + ".mp4")
        if path:
            return d, path
    return None, None


def _read_file(path, limit=5000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:limit]
    except OSError:
        return None


def load_chunk(chunk):
    """Candidates, shorts, cost and timings of one chunk directory."""
    path = chunk["path"]
    out = {"index": chunk["index"], "dir": chunk["name"],
           "stream_start": chunk["stream_start"], "seconds": chunk["seconds"],
           "candidates": [], "shorts": [], "cost": None, "stages": [],
           "fetch_seconds": None, "pipeline_seconds": None, "exit_code": None,
           "metadata_file": None, "errors": []}
    if chunk.get("missing"):
        out["errors"].append("chunk directory missing")
        return out
    fetched = _dict(sr.load_json(os.path.join(path, FETCHED_FILE)))
    out["fetch_seconds"] = sr.num(fetched.get("fetch_seconds"))
    done = sr.load_json(os.path.join(path, DONE_FILE))
    if isinstance(done, dict):
        out["exit_code"] = sr.as_int(done.get("exit_code"))
        out["pipeline_seconds"] = sr.num(done.get("seconds"))
        if out["exit_code"] not in (None, 0):
            out["errors"].append(f"pipeline exited with code {out['exit_code']}")
    else:
        out["errors"].append(f"no {DONE_FILE} (chunk not finished?)")
    st = _dict(sr.load_json(os.path.join(path, sr.STAGES_FILE)))
    for s in _list(st.get("stages")):
        if isinstance(s, dict):
            out["stages"].append({"name": sr.text(s.get("name"), 40),
                                  "duration_s": sr.num(s.get("duration_s"))})

    meta_path = sr.find_metadata(path)
    metadata = sr.load_json(meta_path) if meta_path else None
    if not isinstance(metadata, dict):
        out["errors"].append("no readable *_metadata.json" if not meta_path
                             else "metadata is not valid JSON")
        return out
    out["metadata_file"] = os.path.basename(meta_path)
    out["cost"] = sr.cost_summary(metadata)

    raw_shorts = _list(metadata.get("shorts"))
    raw_cands = metadata.get("candidates") if isinstance(metadata.get("candidates"), list) \
        else raw_shorts
    ver = _dict(sr.load_json(os.path.join(path, sr.VERIFICATION_FILE)))
    ver_rows = {sr.as_int(r.get("short_index")): r for r in _list(ver.get("shorts"))
                if isinstance(r, dict) and sr.as_int(r.get("short_index")) is not None}
    appr = _dict(sr.load_json(os.path.join(path, PUBLISH_DIR, APPROVAL_FILE)))
    appr_rows = {sr.as_int(e.get("rank")): e for e in _list(appr.get("shorts"))
                 if isinstance(e, dict) and sr.as_int(e.get("rank")) is not None}

    for i, raw in enumerate(raw_shorts):
        s = sr.normalise_candidate(raw, i)
        rank = s["rank"] if s["rank"] is not None else i + 1
        s["rank"] = rank
        pkg_dir, video = _chunk_short_file(path, rank, appr_rows.get(rank))
        s["package_dir"], s["video_path"] = pkg_dir, video
        row = ver_rows.get(i)
        if row is not None:
            summary = _verification_summary(row)
            verified = isinstance(row.get("result"), dict)
        else:
            # No verification.json row: fall back on the chunk's package record.
            pkg_meta = _dict(sr.load_json(os.path.join(pkg_dir, "metadata.json"))) \
                if pkg_dir else {}
            v = _dict(pkg_meta.get("verification"))
            summary = {"ok": v.get("ok") if v.get("ok") in (True, False, None) else None,
                       "failed": [sr.text(x, 40) for x in _list(v.get("failed"))],
                       "warnings": [sr.text(x, 40) for x in _list(v.get("warnings"))],
                       "checks": [{"name": sr.text(_dict(c).get("name"), 40),
                                   "ok": _dict(c).get("ok"),
                                   "detail": sr.text(_dict(c).get("detail"), 300)}
                                  for c in _list(v.get("checks"))]}
            verified = bool(v.get("checks"))
        s["verification"] = summary
        s["verified"] = verified
        # ok None (a tool was missing) counts as passed; False does not.
        s["passed"] = verified and summary["ok"] is not False
        out["shorts"].append(s)

    for i, raw in enumerate(raw_cands):
        c = sr.normalise_candidate(raw, i)
        c["predicted_score"] = sr.num(_dict(raw).get("predicted_score"))
        c["short"] = None
        for s in out["shorts"]:
            if sr._same_clip(c, s) or (raw_cands is raw_shorts and s["position"] == i):
                c["short"] = s
                break
        out["candidates"].append(c)
    return out


# --- ranking ------------------------------------------------------------------

def _plus(a, b):
    return round(a + b, 3) if a is not None and b is not None else None


def global_candidates(chunks):
    """Every candidate of every chunk on the stream timeline, best first."""
    rows = []
    for ch in chunks:
        for c in ch["candidates"]:
            s = c["short"]
            rows.append(dict(
                c, chunk_index=ch["index"], chunk_dir=ch["dir"],
                chunk_rank=s["rank"] if s else None,
                stream_start=_plus(ch["stream_start"], c["start"]),
                stream_end=_plus(ch["stream_start"], c["end"]),
                rendered=bool(s and s["video_path"]),
                passed=bool(s and s["video_path"] and s["passed"]),
                selected=False, publish_rank=None))

    def key(r):
        return (-(r["overall"] if r["overall"] is not None else float("-inf")),
                -(r["predicted_score"] if r["predicted_score"] is not None
                  else float("-inf")),
                r["stream_start"] if r["stream_start"] is not None else float("inf"),
                r["chunk_index"], r["position"])
    rows.sort(key=key)
    for i, r in enumerate(rows):
        r["global_rank"] = i + 1
    return rows


# --- package ------------------------------------------------------------------

def _copy(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)


def _write_text(path, value):
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def package(out_dir, selected, source_url, channel=None):
    """Copy the selected shorts into ``<out>/publish/`` and write approval.json.
    A review or a post belongs to the exact bytes (sha256): carried over on a
    re-run, and a posted short is never dropped, overwritten or re-opened."""
    publish = os.path.join(out_dir, PUBLISH_DIR)
    os.makedirs(publish, exist_ok=True)
    approval_path = os.path.join(publish, APPROVAL_FILE)
    previous = _dict(sr.load_json(approval_path))
    prev_by_sha = {}
    for e in _list(previous.get("shorts")):
        if isinstance(e, dict) and isinstance(e.get("sha256"), str):
            prev_by_sha.setdefault(e["sha256"], e)

    digests = [_sha256(r["short"]["video_path"]) for r in selected]
    chosen = set(digests)
    # Posted entries that are no longer selected stay, with their rank and dir.
    kept_posted = [e for sha, e in prev_by_sha.items()
                   if e.get("status") == "posted" and sha not in chosen]
    taken = {sr.as_int(e.get("rank")) for e in kept_posted}
    for e in kept_posted:
        m = _SHORT_DIR.fullmatch(str(e.get("dir") or ""))
        if m:
            taken.add(int(m.group(1)))

    entries, errors, next_rank = [], [], 1
    for r, digest in zip(selected, digests):
        prev = prev_by_sha.get(digest) or {}
        while next_rank in taken:
            next_rank += 1
        rank = next_rank
        next_rank += 1
        r["publish_rank"] = rank
        name = f"short_{rank}"
        dest = os.path.join(publish, name)
        if os.path.islink(dest):
            os.remove(dest)
        os.makedirs(dest, exist_ok=True)
        video = f"{name}.mp4"
        s = r["short"]
        _copy(s["video_path"], os.path.join(dest, video))
        digest = _sha256(os.path.join(dest, video))
        pkg = s["package_dir"]
        title = _read_file(os.path.join(pkg, "title.txt"), 1000)
        description = _read_file(os.path.join(pkg, "description.txt"), 5000)
        tags = _read_file(os.path.join(pkg, "hashtags.txt"), 2000)
        title = sr.text(title if title is not None else s["title"], 300)
        description = (description if description is not None
                       else (s["description"] if isinstance(s["description"], str) else ""))
        description = description.replace("\r\n", "\n").strip()
        tag_list = sr.hashtags(tags) if tags is not None else s["hashtags"]
        _write_text(os.path.join(dest, "title.txt"), title + "\n")
        _write_text(os.path.join(dest, "description.txt"), description + "\n")
        _write_text(os.path.join(dest, "hashtags.txt"), " ".join(tag_list) + "\n")
        r.update(publish_title=title, publish_description=description,
                 publish_hashtags=tag_list, publish_video=f"{name}/{video}")
        meta = {
            "rank": rank,
            "global_rank": r["global_rank"],
            "chunk_index": r["chunk_index"],
            "chunk_dir": r["chunk_dir"],
            "chunk_rank": r["chunk_rank"],
            "candidate_index": r["candidate_index"],
            "title": title, "description": description, "hashtags": tag_list,
            "hook_text": r["hook"],
            "rating": {"overall": r["overall"], "scores": r["scores"],
                       "summary": r["summary"], "rationale": r["rationale"]},
            "predicted_score": r["predicted_score"],
            "source": {
                "url": source_url,
                "channel": channel,
                "chunk_start": r["start"], "chunk_end": r["end"],
                "stream_start": r["stream_start"], "stream_end": r["stream_end"],
                "stream_start_hms": sr.hms(r["stream_start"]),
                "stream_end_hms": sr.hms(r["stream_end"]),
                "duration": r["duration"],
            },
            "verification": s["verification"],
        }
        _write_json(os.path.join(dest, "metadata.json"), meta)
        status = prev.get("status") if prev.get("status") in STATUSES else "pending_approval"
        entry = {"rank": rank, "dir": name, "video": f"{name}/{video}", "title": title,
                 "overall": r["overall"],
                 "source_file": f"{r['chunk_dir']}/{PUBLISH_DIR}/"
                                f"{os.path.basename(s['package_dir'])}/"
                                f"{os.path.basename(s['video_path'])}",
                 "sha256": digest, "verification_ok": s["verification"]["ok"],
                 "status": status, "global_rank": r["global_rank"],
                 "chunk_index": r["chunk_index"]}
        for k in PUBLISH_RECORD:
            if k in prev:
                entry[k] = prev[k]
        entries.append(entry)
        _say(f"package: {name} <- chunk {r['chunk_index']} rank {r['chunk_rank']} ({status})")

    entries.extend(kept_posted)
    entries.sort(key=lambda e: sr.as_int(e.get("rank"), 10 ** 6))
    keep = {e["dir"] for e in entries if isinstance(e.get("dir"), str)}
    for stale in os.listdir(publish):
        p = os.path.join(publish, stale)
        if _SHORT_DIR.fullmatch(stale) and stale not in keep \
                and os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
    approval = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_url": source_url,
                "note": "Nothing has been uploaded. Set status to approved or "
                        "rejected per short after review.",
                "shorts": entries, "errors": errors}
    _write_json(approval_path, approval)
    return approval


# --- report -------------------------------------------------------------------

def build(out_dir, top=10):
    """Rank, package and return the report dict."""
    issues = []
    plan = sr.load_json(os.path.join(out_dir, PLAN_FILE))
    plan_d = _dict(plan)
    source_url = sr.text(plan_d.get("url"), 2000) or None
    chunks = [load_chunk(c) for c in chunk_dirs(out_dir, plan, issues)]
    for ch in chunks:
        for err in ch["errors"]:
            issues.append(f"chunk {ch['index']} ({sr.text(ch['dir'], 40)}): {err}")
        if ch["stream_start"] is None and ch["candidates"]:
            issues.append(f"chunk {ch['index']}: no stream_start_seconds, stream times unknown.")

    rows = global_candidates(chunks)
    selectable = [r for r in rows if r["passed"]]
    selected = selectable[:max(0, int(top))]
    for r in selected:
        r["selected"] = True
    if len(selected) < top:
        issues.append(f"Only {len(selected)} rendered short(s) passed verification; "
                      f"asked for the top {top}.")
    for r in rows:
        label = f"chunk {r['chunk_index']} candidate #{r['candidate_index'] if r['candidate_index'] is not None else r['position']}"
        if not r["rated"]:
            issues.append(f"{label}: no rating (or no overall score).")
    for ch in chunks:
        for s in ch["shorts"]:
            tag = f"chunk {ch['index']} short rank {s['rank']}"
            if not s["video_path"]:
                issues.append(f"{tag}: no packaged file in {PUBLISH_DIR}/.")
            if not s["verified"]:
                issues.append(f"{tag}: not verified.")
            elif s["verification"]["ok"] is False:
                issues.append(f"{tag}: verification FAILED "
                              f"({', '.join(s['verification']['failed']) or 'see checks'}).")

    approval = package(out_dir, selected, source_url,
                       sr.text(plan_d.get("channel"), 100) or None)

    per_chunk_cost = {}
    for ch in chunks:
        if ch["cost"]:
            per_chunk_cost[str(ch["index"])] = ch["cost"]["total_usd"]
    total_cost = round(sum(per_chunk_cost.values()), 6)

    def plain(r):
        return {k: r[k] for k in (
            "global_rank", "chunk_index", "chunk_dir", "chunk_rank", "candidate_index",
            "start", "end", "stream_start", "stream_end", "duration", "overall", "scores",
            "summary", "predicted_score", "rated", "rendered", "passed", "selected",
            "publish_rank", "title", "hook")}

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_url": source_url,
        "title": sr.text(plan_d.get("title"), 300) or None,
        "top": top,
        "candidates": [plain(r) for r in rows],
        "selected": [dict(plain(r), description=r.get("publish_description"),
                          hashtags=r.get("publish_hashtags"), video=r.get("publish_video"),
                          publish_title=r.get("publish_title"),
                          verification=r["short"]["verification"]) for r in selected],
        "cost": {"total_usd": total_cost, "per_chunk": per_chunk_cost},
        "chunks": [{"index": ch["index"], "dir": ch["dir"],
                    "stream_start": ch["stream_start"], "seconds": ch["seconds"],
                    "fetch_seconds": ch["fetch_seconds"],
                    "pipeline_seconds": ch["pipeline_seconds"],
                    "exit_code": ch["exit_code"], "stages": ch["stages"],
                    "candidates": len(ch["candidates"]), "shorts": len(ch["shorts"]),
                    "cost_usd": ch["cost"]["total_usd"] if ch["cost"] else None,
                    "errors": ch["errors"]} for ch in chunks],
        "approval": {"shorts": len(approval["shorts"])},
        "issues": issues,
    }
    return report


def _fmt(value, digits=0):
    v = sr.num(value)
    return "–" if v is None else f"{v:.{digits}f}"


def render_markdown(r):
    lines = [f"# Batch report: {sr.cell(r['title'] or 'stream', 200)}", ""]
    if r["source_url"]:
        lines += [f"Source: {sr.cell(r['source_url'], 300)}", ""]
    lines += [f"{len(r['candidates'])} candidate(s) across {len(r['chunks'])} chunk(s); "
              f"{len(r['selected'])} selected (top {r['top']}).", "",
              "## All candidates", ""]
    crit = [sr.CRITERIA_SHORT[k] for k in sr.CRITERIA]
    lines.append("| # | Chunk | Stream | Dur | Overall | " + " | ".join(crit)
                 + " | Summary | Rendered | Selected |")
    lines.append("|" + "---|" * (9 + len(crit)))
    for c in r["candidates"]:
        scores = [_fmt(c["scores"].get(k)) for k in sr.CRITERIA]
        lines.append(
            f"| {c['global_rank']} | {c['chunk_index']} | "
            f"{sr.hms(c['stream_start'])}–{sr.hms(c['stream_end'])} | "
            f"{_fmt(c['duration'], 1)} | {_fmt(c['overall'])} | " + " | ".join(scores)
            + f" | {sr.cell(c['summary'], 160)} | {'yes' if c['rendered'] else 'no'} | "
            + (f"#{c['publish_rank']}" if c["selected"] else "no") + " |")
    lines += ["", "## Selected shorts", ""]
    if not r["selected"]:
        lines += ["_None._", ""]
    for s in r["selected"]:
        v = s["verification"] or {}
        lines += [f"### #{s['publish_rank']}: {sr.cell(s['publish_title'], 200)}", "",
                  f"- Stream: {sr.hms(s['stream_start'])}–{sr.hms(s['stream_end'])} "
                  f"(chunk {s['chunk_index']}, chunk rank {s['chunk_rank']}, "
                  f"global candidate #{s['global_rank']})",
                  f"- Overall: {_fmt(s['overall'])}",
                  f"- File: {PUBLISH_DIR}/{sr.cell(s['video'], 200)}",
                  f"- Hook: {sr.cell(s['hook'], 300)}",
                  f"- Hashtags: {sr.cell(' '.join(s['hashtags'] or []), 500)}",
                  "- Description:", "", sr.para(s["description"]), "",
                  f"- Verification: {sr.cell({True: 'passed', False: 'FAILED', None: 'n/a (tool missing)'}.get(v.get('ok'), 'n/a'))}"]
        for chk in v.get("checks") or []:
            ok = {True: "ok", False: "FAIL", None: "n/a"}.get(chk.get("ok"), "n/a")
            lines.append(f"  - {sr.cell(chk.get('name'), 40)}: {ok} "
                         f"{sr.cell(chk.get('detail'), 200)}")
        lines.append("")
    lines += ["## Gemini cost", "",
              f"Total: ${r['cost']['total_usd']:.4f}", "",
              "| Chunk | USD |", "|---|---|"]
    for ch in r["chunks"]:
        lines.append(f"| {ch['index']} | "
                     + ("–" if ch["cost_usd"] is None else f"{ch['cost_usd']:.4f}") + " |")
    lines += ["", "## Timings", "",
              "| Chunk | Fetch s | Pipeline s | Exit | Stages |", "|---|---|---|---|---|"]
    for ch in r["chunks"]:
        stages = ", ".join(f"{s['name']} {_fmt(s['duration_s'], 1)}s" for s in ch["stages"])
        lines.append(f"| {ch['index']} | {_fmt(ch['fetch_seconds'], 1)} | "
                     f"{_fmt(ch['pipeline_seconds'], 1)} | "
                     f"{'–' if ch['exit_code'] is None else ch['exit_code']} | "
                     f"{sr.cell(stages, 400)} |")
    lines += ["", "## Errors and edge cases", ""]
    lines += [f"- {sr.cell(i, 400)}" for i in r["issues"]] or ["_None._"]
    return "\n".join(lines) + "\n"


def write_report(out_dir, top=10):
    report = build(out_dir, top)
    md = render_markdown(report)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    _write_json(os.path.join(out_dir, "report.json"), report)
    return md, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="batch output directory")
    ap.add_argument("--top", type=int, default=10, help="shorts to package (default 10)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.out):
        ap.error(f"{args.out} is not a directory")
    _, report = write_report(args.out, max(0, args.top))
    _say(f"batch report: {os.path.join(args.out, 'report.md')} "
         f"({len(report['selected'])} selected, {len(report['issues'])} issue(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
