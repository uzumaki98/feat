"""Fetch a stretch of a YouTube live broadcast in fixed-length chunks, fast.

    python stream_fetch.py <url> --out output/<dir> --from-minutes 90
                           [--to-minutes N] [--chunk-minutes 90] [--threads 12]

A broadcast that is still live is served as numbered 2-second DASH fragments
(``&sq=N`` on the format URL, from the first one on). yt-dlp's own from-start
download fetches them one at a time (~4x real time on a home connection) and
cannot start in the middle; this fetches any range of them with a thread pool
(~15x real time measured 19-nov-2026), one chunk after another, and writes
``chunk_NN/source.mp4`` (video + audio, timestamps from 0) with a
``chunk_NN/fetched.json`` marker once the chunk is complete, so a scheduler can
start on it while the next chunk downloads. Re-running skips finished chunks.

Fragments are self-contained fMP4 segments, so concatenating them in order is
a valid stream. The format URLs expire after some hours: a 403/404 refreshes
them once for everyone and the fragment is fetched again.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import threading
import time

import httpx
import yt_dlp

FORMAT = "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]"
POT_ARGS = {"youtubepot-bgutilscript": {"script_path": [
    os.environ.get("BGUTIL_SCRIPT_PATH") or "/opt/bgutil-provider/server/build/generate_once.js"]}}


def log(out_dir, line):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {line}", flush=True)
    with open(os.path.join(out_dir, "fetch.log"), "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {line}\n")


class Source:
    """The broadcast's video and audio format URLs, refreshable."""

    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.generation = 0
        self.refresh()

    def refresh(self, seen_generation=None):
        with self.lock:
            if seen_generation is not None and seen_generation != self.generation:
                return  # someone else refreshed meanwhile
            opts = {"quiet": True, "no_warnings": True, "cachedir": False,
                    "live_from_start": True, "format": FORMAT, "noplaylist": True,
                    "extractor_args": POT_ARGS}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
            if info.get("live_status") != "is_live":
                raise SystemExit("This broadcast is not live any more: use shorts_pipeline.py "
                                 "on the recording (its sections download normally).")
            fmts = info.get("requested_formats") or []
            self.video = next(f for f in fmts if (f.get("vcodec") or "none") != "none")
            self.audio = next(f for f in fmts if (f.get("vcodec") or "none") == "none")
            self.step = float(self.video.get("target_duration") or 2.0)
            self.headers = dict(self.video.get("http_headers") or {})
            self.title = info.get("title") or ""
            self.channel = info.get("channel") or info.get("uploader") or ""
            self.generation += 1

    def base(self, kind):
        return (self.video if kind == "video" else self.audio)["url"]

    def head_seq(self, client):
        r = client.head(self.base("video"))
        return int(r.headers["X-Head-Seqnum"])


def fetch_fragment(client, src, kind, sq, tries=5):
    for attempt in range(tries):
        gen = src.generation
        url = yt_dlp.utils.update_url_query(src.base(kind), {"sq": str(sq)})
        try:
            r = client.get(url)
            if r.status_code in (403, 404, 410):
                src.refresh(gen)
                continue
            r.raise_for_status()
            if r.content:
                return r.content
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{kind} fragment {sq} failed {tries} times")


def fetch_stream(client, src, kind, first, last, path, threads):
    """Fragments [first, last) of one stream, in order, into path."""
    tmp = path + ".part"
    with open(tmp, "wb") as out, cf.ThreadPoolExecutor(threads) as pool:
        for data in pool.map(lambda sq: fetch_fragment(client, src, kind, sq),
                             range(first, last)):
            out.write(data)
    os.replace(tmp, path)


def chunk_plan(first_sq, end_sq, per_chunk):
    """[(index 1-based, first, last)] covering [first_sq, end_sq)."""
    plan, i, sq = [], 1, first_sq
    while sq < end_sq:
        plan.append((i, sq, min(sq + per_chunk, end_sq)))
        sq += per_chunk
        i += 1
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("url")
    p.add_argument("--out", required=True)
    p.add_argument("--from-minutes", type=float, default=0.0)
    p.add_argument("--to-minutes", type=float, default=None,
                   help="stop here (default: the live point when the fetch starts)")
    p.add_argument("--chunk-minutes", type=float, default=90.0)
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    src = Source(args.url)
    with httpx.Client(timeout=30.0, headers=src.headers, follow_redirects=True) as client:
        head = src.head_seq(client)
        first = int(math.ceil(args.from_minutes * 60 / src.step))
        end = head + 1 if args.to_minutes is None else min(
            head + 1, int(args.to_minutes * 60 / src.step))
        per = int(round(args.chunk_minutes * 60 / src.step))
        plan = chunk_plan(first, end, per)
        log(args.out, f"{src.title!r}: fragments {first}-{end - 1} of {head} "
                      f"({(end - first) * src.step / 3600:.1f} h) in {len(plan)} chunk(s) "
                      f"of {args.chunk_minutes:g} min, {args.threads} threads")
        with open(os.path.join(args.out, "fetch_plan.json"), "w", encoding="utf-8") as f:
            json.dump({"url": args.url, "title": src.title, "channel": src.channel,
                       "step": src.step,
                       "chunks": [{"index": i, "first_sq": a, "end_sq": b,
                                   "stream_start_seconds": a * src.step,
                                   "seconds": (b - a) * src.step,
                                   "dir": f"chunk_{i:02d}"} for i, a, b in plan]},
                      f, indent=2, ensure_ascii=False)
        for i, a, b in plan:
            d = os.path.join(args.out, f"chunk_{i:02d}")
            marker = os.path.join(d, "fetched.json")
            if os.path.exists(marker):
                continue
            os.makedirs(d, exist_ok=True)
            t0 = time.time()
            v, au = os.path.join(d, "video.mp4"), os.path.join(d, "audio.m4a")
            fetch_stream(client, src, "audio", a, b, au, args.threads)
            fetch_stream(client, src, "video", a, b, v, args.threads)
            target = os.path.join(d, "source.mp4")
            res = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", v, "-i", au,
                                  "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                                  "-movflags", "+faststart", target],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                log(args.out, f"chunk {i}: merge failed: {res.stderr[-300:]}")
                return 1
            os.remove(v)
            os.remove(au)
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"index": i, "first_sq": a, "end_sq": b,
                           "stream_start_seconds": a * src.step,
                           "seconds": (b - a) * src.step,
                           "fetch_seconds": round(time.time() - t0, 1)}, f, indent=2)
            log(args.out, f"chunk {i}/{len(plan)} ready: {(b - a) * src.step / 60:.0f} min "
                          f"from {a * src.step / 3600:.2f} h, fetched in {time.time() - t0:.0f}s")
    log(args.out, "all chunks fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
