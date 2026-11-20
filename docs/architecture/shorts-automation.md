# Headless Shorts runner: rating, top-N render, checks, approval package

`shorts_pipeline.py` takes a URL (or a local file) to YouTube Shorts that wait
for a person's approval. Nothing is ever uploaded.

```bash
python shorts_pipeline.py <url> --minutes 90 --top 3 [--out output/<slug>]
python shorts_pipeline.py <url> --input <local.mp4> ...   # skip the download
python shorts_pipeline.py --stage report --out output/<slug>   # re-run one stage
```

Stages, each re-runnable on an existing out dir:

- **run**: `main.py` as a subprocess with `AUTO_LAYOUT=1`, `AUTO_HOOK=1`,
  `AUTO_CAPTIONS=1`, `RATE_CLIPS=1`, `RENDER_TOP_N=<top>`,
  `MAX_SOURCE_MINUTES=<minutes>`, and `CLIP_WORKERS` (the env value, else 1 with
  a GPU and 3 without). Every output line goes to `pipeline.log` with elapsed
  seconds; stage boundaries and `CLIP_READY` lines go to `stages.json`.
- **edit** (`shorts_variants.full_edit`, `--edit full`, the default; `--edit
  none` keeps main.py's render): each short is re-made from its clip before
  captions and hook (`subtitled_<g>_hooked_<g>_<clip>.mp4` -> `<clip>.mp4`), with
  the job transcript's words for that clip. Stretches with no word AND quiet
  audio (below half the clip's median loudness) longer than 0.45 s are cut to
  0.15 s each side; loud wordless action (a crash, a scream) is never cut. Then
  a 1.6 s cold open (the loudest window in the back 60%, skipped when it would
  pass 59.5 s), punch-ins on the audio beats (one per 6 s at most), one-word
  captions (`CAPTION_PRESETS["oneword"]`), the hook, synthetic sound effects
  (ffmpeg noise/sine, no third-party audio) and -14 LUFS. Output
  `edited_clip_<n>.mp4` + `.ass` + `.words.json`, listed in `edits.json` against
  the render it replaces; `rendered_shorts` uses an edit only while that render
  is still the one on record. A failed edit keeps the render and marks the
  stage. The zoom is a per-frame `scale` then a fixed `crop`: a crop whose size
  changes per frame (punch_in's sendcmd) segfaults the image's ffmpeg on CPU.
  `python shorts_variants.py --clip ... --hook ... --out ...` renders the plain
  (B) and full (C) edit of one clip side by side.
  **Streamer template** (`facecam_template.py`, `--template facecam`, the
  default; `none` keeps main.py's framing), **only for a YouTube live stream or
  its recording** (`is_youtube_stream`: a YouTube URL in `run.json`, and either
  `run.json["stream"]`, which `stream_batch` sets with `--stream` on every
  chunk, or `metadata["source_live_status"]` in is_live / was_live /
  post_live, which main.py records from its download's extraction). Any other
  source, a non-stream YouTube video or another site's stream, keeps main.py's
  framing even with a webcam in it. For a stream, before the edit, each short's span
  is cut again from the horizontal source (`metadata["source_video"]`), and
  when `camera_inset.detect` finds a webcam the clip is re-framed as the
  stream-clip look: the webcam enlarged in the top 38% (`CAM_RATIO`), the
  gameplay below at full height, centred on the side away from the webcam.
  The detector grows a 16:9 box from the face, which on a stream overlay takes
  in the counters and menus around the camera, so the box is snapped to the
  camera's own border (`snap_box`: the time-averaged gradient peaks on the one
  edge that stays put; no clear peak keeps the estimate), and the band is the
  largest window inside it. Punch-ins then zoom only the gameplay band
  (`zoom_from`). No webcam, an already vertical source or any failure: the
  clip keeps main.py's framing; `edits.json` records `layout` per short.
- **English letters only** (`latin_text.py`, `--script latin`, the default;
  `any` keeps the speech's script): the speech can be Hindi, but what the
  viewer reads is in Latin letters. The edit stage romanises the caption words
  one for one (timings kept) and the hook, as Hinglish ("bahut ganda maar
  diya"); the package stage rewrites a title, description or hashtag that is
  not Latin in English. Gemini does both when a key is there; every answer is
  checked in code (same word count, Latin letters only) and anything failing
  falls back to a fixed Devanagari table, so the output is always Latin. Copy
  and words already in Latin letters never call Gemini. The dashboard path is
  unchanged.
- **verify** (`verify_shorts.py`): per rendered short (the edit when there is
  one, checked against its own captions and words), 9:16, 5-60 s, video and
  audio streams, a clean full decode, loudness (informational, -14 LUFS ±3),
  and the caption sidecar (parses, monotonic, in bounds, aligned to the clip's
  transcript words) -> `verification.json`. The alignment check compares the
  sidecar with the transcript it was made from; it cannot see a cut whose audio
  is offset. ffmpeg/ffprobe missing = that check is `ok: None`, not a crash.
- **report** (`shorts_report.py`): `report.md` / `report.json` with every
  candidate ranked (timestamps, the five scores, summary), the rendered shorts
  with their copy and checks, the Gemini cost and the stage timings.
- **package**: `publish/short_<rank>/` with a copy of the video, `title.txt`,
  `description.txt`, `hashtags.txt`, `metadata.json`, and
  `publish/approval.json` with every short `pending_approval`. A status a
  reviewer sets is kept across re-runs only for the same video bytes (sha256);
  a re-render goes back to pending. Copies, never hard links: ffmpeg `-y`
  rewrites a render in place and would change a reviewed file.

## Posting (`shorts_publish.py`, never part of `all`)

```bash
python shorts_pipeline.py --stage approve --ranks 1,2,3 --out output/<slug>
python shorts_pipeline.py --stage publish --out output/<slug> [--privacy public|unlisted|private]
       [--start 2026-11-20T18:00 --every-hours 24 --timezone Asia/Kolkata] [--profile NAME] [--dry-run]
python shorts_pipeline.py --stage post-status --out output/<slug>
```

- `approve` records the decision and the sha256 of the video reviewed.
  `publish` posts only `approved` entries whose packaged video still has those
  bytes; a changed video goes back to `pending_approval`.
- YouTube only, through Upload-Post with the same fields as the dashboard
  (`social_platforms.upload_fields`). Key from `UPLOAD_POST_API_KEY` only (env /
  `.env`, never an argument, never printed); profile from `--profile` or the
  one profile with YouTube connected.
- Each result is written to `approval.json` right after its upload (`posted` +
  request id, or `publish_failed` + reason), so no short is posted twice; the
  package stage carries that record across re-runs. `post-status` records
  Upload-Post's final status and the YouTube link when it gives one.
- `--hashtags '#A,#B'` puts those first (YouTube shows the first three above
  the title), then the clip's own (at most 5 topical), `#Shorts` last.
- Every post credits its source above the tags: `🎥 Credit: <name> · full
  stream: <link to the moment>` (`credit_line`). The name is `--credit NAME`,
  else the channel `stream_fetch` recorded; the link is rebuilt from the
  package's `metadata.json` `source` (YouTube, Twitch or Kick over https only,
  with the stream offset), never pasted from it. `--no-credit` leaves it out;
  `approval.json` keeps the line sent (`credit_sent`).

## Rating (`clip_rating.py`, `RATE_CLIPS=1`)

After the detail pass, every candidate gets one Gemini call with three frames
(<= 768 px) from inside the clip and the clip's own words with relative times,
never the video. Rubric, 0-10 each: `hook_strength` (first 2 s),
`energy_pacing`, `narrative_completeness` (stands alone, setup and payoff),
`av_clarity`, `retention_potential`, each with a <= 20-word rationale and a
summary. `overall` (0-100) is computed in code from fixed weights (hook and
retention 0.30, energy and story 0.15, clarity 0.10); the model's numbers are
clamped and its text truncated. A transient error is retried (`RETRY_WAITS`), a
policy block or any other failure leaves that candidate's `rating` None and it
ranks by `predicted_score`.

`RENDER_TOP_N=n` keeps every candidate under `metadata["candidates"]` (with
`candidate_index`, 1-based, and `rank`) and renders only the n best:
`metadata["shorts"]` is that ranked list, so `CLIP_READY i`, `_clip_{i+1}` and
`shorts[i]` agree as before. Neither flag is a remote-worker job option, and
`remote_worker.clean_metadata` drops a `candidates` list a worker sends. The
dashboard path sets neither.

## YouTube copy (`gemini_worker.normalize_youtube_copy`)

The detail and vision prompts also ask for `video_description_for_youtube_short`
and `youtube_hashtags` (both optional in the schema). Every clip goes through
`normalize_youtube_copy` in code: title <= 100 chars, no `<` / `>`, description
falling back to the title, at most 5 topical tags plus `#Shorts` (always last,
always in the description), within YouTube's 5000-byte limit. Run again after
hook grounding rewrites a title.

## A stream that is still live

With a cap on the minutes, `main._live_head` fetches the broadcast's first N
minutes from its start (yt-dlp `live_from_start`) instead of recording the live
edge; see `proxy-routing.md`.
