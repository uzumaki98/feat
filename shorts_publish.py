"""Post approved Shorts to YouTube through Upload-Post, after a person said yes.

The gate is ``publish/approval.json`` (written by ``shorts_pipeline.py``'s
package stage). Nothing here runs on its own: ``shorts_pipeline.py --stage
approve`` records a person's decision, ``--stage publish`` posts, ``--stage
post-status`` asks Upload-Post how each post ended.

- Only entries a person set to ``approved`` are posted, and only while the
  packaged video still has the bytes that were reviewed (sha256). A re-render
  is never posted on an old approval.
- Each result is written back to approval.json right after its upload
  (``posted`` with Upload-Post's request id, or ``publish_failed`` with the
  reason), so a re-run never posts the same short twice.
- The Upload-Post key comes from ``UPLOAD_POST_API_KEY`` only (never a
  command-line argument, never printed). The profile is ``--profile``, or the
  one profile on the account that has YouTube connected.
- Same vendor fields as the dashboard's posting path
  (``social_platforms.upload_fields``), YouTube only.
"""
import datetime as _dt
import hashlib
import json
import os
import re
from urllib.parse import parse_qs, urlparse

import social_platforms

UPLOAD_URL = "https://api.upload-post.com/api/upload"
STATUS_URL = "https://api.upload-post.com/api/uploadposts/status"
USERS_URL = "https://api.upload-post.com/api/uploadposts/users"
PRIVACY = ("public", "unlisted", "private")
DECISIONS = ("approved", "rejected", "pending_approval")
DONE = ("posted",)
UPLOAD_TIMEOUT = 300.0


class PublishError(RuntimeError):
    pass


TOPICAL_MAX = 5           # YouTube ignores every hashtag past 15; we keep it short
SHORTS_TAG = "#Shorts"
_TAG = re.compile(r"#[^\W_][\w]{0,49}", re.UNICODE)


def clean_tag(tag):
    """``#Word`` (letters, digits, underscores, no spaces), or None."""
    t = str(tag or "").strip()
    if not t:
        return None
    if not t.startswith("#"):
        t = "#" + t
    m = _TAG.fullmatch(t)
    if not m or t[1:].replace("_", "").isdigit():
        return None
    return SHORTS_TAG if t.lower() == "#shorts" else t


def split_description(text):
    """(prose, [tags]) of a description whose hashtags trail its text."""
    words = str(text or "").split()
    tags = []
    while words and clean_tag(words[-1]) and words[-1].startswith("#"):
        tags.insert(0, words.pop())
    return " ".join(words), tags


# Hosts a credit link may point at: the source's own page, never anything a
# metadata file could smuggle in (the file is data).
CREDIT_HOSTS = {"youtube.com": "yt", "www.youtube.com": "yt", "m.youtube.com": "yt",
                "youtu.be": "yt", "twitch.tv": "twitch", "www.twitch.tv": "twitch",
                "kick.com": "plain", "www.kick.com": "plain"}


def _seconds(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(v) if 0 <= v < 10 ** 7 else None


def source_link(source):
    """Link to the moment in the original broadcast, from a package's
    ``metadata.json`` ``source`` block; None when there is no usable URL."""
    if not isinstance(source, dict):
        return None
    url = str(source.get("url") or "").strip()
    if not url or len(url) > 300 or any(c.isspace() for c in url):
        return None
    parts = urlparse(url)
    kind = CREDIT_HOSTS.get((parts.hostname or "").lower())
    if parts.scheme != "https" or not kind:
        return None
    t = _seconds(source.get("stream_start"))
    if t is None:
        t = _seconds(source.get("start"))
    if kind == "yt":
        vid = (parse_qs(parts.query).get("v") or [""])[0]
        if parts.hostname == "youtu.be":
            vid = parts.path.strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid or ""):
            return None
        return f"https://www.youtube.com/watch?v={vid}" + (f"&t={t}s" if t else "")
    if kind == "twitch":
        m = re.fullmatch(r"/videos/(\d+)/?", parts.path)
        if m:
            stamp = f"{t // 3600}h{t % 3600 // 60}m{t % 60}s" if t else ""
            return f"https://www.twitch.tv/videos/{m.group(1)}" + (f"?t={stamp}" if stamp else "")
        m = re.fullmatch(r"/([A-Za-z0-9_]{3,25})/?", parts.path)
        return f"https://www.twitch.tv/{m.group(1)}" if m else None
    m = re.fullmatch(r"/[A-Za-z0-9_./-]{1,200}", parts.path)
    return f"https://{parts.hostname}{parts.path}" if m else None


def credit_line(source, name=None):
    """'🎥 Credit: <name> · full stream: <link>' (name from ``name``, else the
    source's channel), or None without a usable link."""
    link = source_link(source)
    if not link:
        return None
    who = " ".join(str(name or (source or {}).get("channel") or "").split())
    who = re.sub(r"[#@<>\[\]{}]", "", who)[:60].strip()
    return (f"🎥 Credit: {who} · full stream: {link}" if who
            else f"🎥 Full stream: {link}")


def with_pinned_tags(description, model_tags, pinned, credit=None):
    """Description and hashtag list with ``pinned`` first (in order), then the
    clip's own topical tags, de-duplicated case-insensitively, at most
    TOPICAL_MAX topical tags, and #Shorts last. YouTube shows the first three
    hashtags of the description above the title, so the pinned ones are
    the ones people see."""
    prose, trailing = split_description(description)
    seen, out = set(), []
    for t in list(pinned or []) + list(model_tags or []) + trailing:
        c = clean_tag(t)
        if not c or c == SHORTS_TAG or c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
    tags = out[:TOPICAL_MAX] + [SHORTS_TAG]
    if credit and credit not in prose:
        prose = (prose + "\n\n" + credit).strip()
    return (prose + "\n\n" + " ".join(tags)).strip(), tags


def api_key(env=None):
    return ((env if env is not None else os.environ).get("UPLOAD_POST_API_KEY") or "").strip()


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("shorts"), list):
        raise PublishError(f"{path} is not an approval file")
    return data


def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _paths(out_dir):
    publish = os.path.join(out_dir, "publish")
    return publish, os.path.join(publish, "approval.json")


def _entry_video(publish, entry):
    """The packaged video of an entry, only inside publish/ (the entry is data)."""
    rel = str(entry.get("video") or "")
    path = os.path.realpath(os.path.join(publish, rel))
    root = os.path.realpath(publish)
    if not rel or os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        return None
    return path


def _load_meta(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path, limit):
    try:
        with open(path, encoding="utf-8") as f:
            return " ".join(f.read().split())[:limit] if limit < 200 else f.read().strip()[:limit]
    except OSError:
        return ""


def approve(out_dir, ranks, decision="approved"):
    """Record a person's decision for the given ranks. A posted short keeps
    its state. Returns the list of (rank, new status)."""
    if decision not in DECISIONS:
        raise PublishError(f"decision must be one of {', '.join(DECISIONS)}")
    publish, path = _paths(out_dir)
    data = _load(path)
    wanted = {int(r) for r in ranks}
    changed = []
    for e in data["shorts"]:
        if not isinstance(e, dict) or e.get("rank") not in wanted:
            continue
        if e.get("status") in DONE:
            changed.append((e["rank"], e["status"]))
            continue
        video = _entry_video(publish, e)
        if not video:
            raise PublishError(f"short rank {e.get('rank')}: its video is missing")
        # The decision is about these exact bytes.
        e["sha256"] = _sha256(video)
        e["status"] = decision
        e["decided_at"] = _now()
        changed.append((e["rank"], decision))
    missing = wanted - {r for r, _ in changed}
    if missing:
        raise PublishError(f"no short with rank {', '.join(map(str, sorted(missing)))}")
    _save(path, data)
    return changed


def schedule(n, start=None, every_hours=24.0):
    """ISO times for n posts: ``start`` (ISO-8601) then every ``every_hours``.
    No start: None for each (post now)."""
    if not start:
        return [None] * n
    try:
        t0 = _dt.datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except ValueError:
        raise PublishError(f"--start is not an ISO date/time: {start}")
    step = _dt.timedelta(hours=float(every_hours))
    return [(t0 + i * step).isoformat(timespec="minutes") for i in range(n)]


def youtube_profile(http, key, wanted=None):
    """The Upload-Post profile to post as: ``wanted`` if it exists with YouTube
    connected, else the only such profile on the account."""
    resp = http.get(USERS_URL, headers={"Authorization": f"Apikey {key}"})
    if resp.status_code != 200:
        raise PublishError(f"Upload-Post refused the key or is unavailable ({resp.status_code})")
    profiles = (resp.json() or {}).get("profiles") or []
    usable = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        socials = p.get("social_accounts") or {}
        if isinstance(socials.get("youtube"), dict) and p.get("username"):
            usable.append(str(p["username"]))
    if wanted:
        if wanted in usable:
            return wanted
        raise PublishError(f"profile {wanted!r} has no YouTube account connected "
                           f"(profiles with YouTube: {', '.join(usable) or 'none'})")
    if len(usable) == 1:
        return usable[0]
    if not usable:
        raise PublishError("no Upload-Post profile has a YouTube account connected; "
                           "connect YouTube at app.upload-post.com first")
    raise PublishError(f"several profiles have YouTube connected ({', '.join(usable)}); "
                       f"choose one with --profile")


def publish(out_dir, *, http, key=None, profile=None, privacy="public", start=None,
            every_hours=24.0, timezone="UTC", dry_run=False, say=print, pinned_tags=None,
            credit=True, credit_name=None):
    """Post every approved, unposted short in rank order. Returns the entries
    acted on. ``http`` is an httpx.Client-like object (get/post). ``credit``
    adds a line naming the creator (``credit_name``, else the source's
    channel) and linking the moment in the original stream."""
    if privacy not in PRIVACY:
        raise PublishError(f"privacy must be one of {', '.join(PRIVACY)}")
    key = key if key is not None else api_key()
    if not key:
        raise PublishError("UPLOAD_POST_API_KEY is not set (put it in .env, not on the command line)")
    publish_dir, path = _paths(out_dir)
    data = _load(path)
    todo = sorted((e for e in data["shorts"]
                   if isinstance(e, dict) and e.get("status") == "approved"),
                  key=lambda e: e.get("rank") or 0)
    if not todo:
        say("publish: no approved short waiting (set them with --stage approve)")
        return []
    user = youtube_profile(http, key, profile)
    times = schedule(len(todo), start, every_hours)
    acted = []
    for entry, when in zip(todo, times):
        rank = entry.get("rank")
        video = _entry_video(publish_dir, entry)
        if not video:
            entry.update(status="publish_failed", error="video missing", tried_at=_now())
            _save(path, data)
            acted.append(entry)
            continue
        if _sha256(video) != entry.get("sha256"):
            # Approved bytes and packaged bytes differ: review again.
            entry.update(status="pending_approval", error="video changed after approval",
                         tried_at=_now())
            _save(path, data)
            say(f"publish: rank {rank} changed after it was approved; back to pending")
            acted.append(entry)
            continue
        short_dir = os.path.dirname(video)
        title = _read_text(os.path.join(short_dir, "title.txt"), 100) or str(entry.get("title") or "")
        description = _read_text(os.path.join(short_dir, "description.txt"), 5000)
        model_tags = _read_text(os.path.join(short_dir, "hashtags.txt"), 199).split()
        credit_text = None
        if credit:
            meta = _load_meta(os.path.join(short_dir, "metadata.json"))
            credit_text = credit_line(meta.get("source"), credit_name)
            if not credit_text:
                say(f"publish: rank {rank} has no source link to credit")
        description, sent_tags = with_pinned_tags(description, model_tags, pinned_tags,
                                                  credit_text)
        payload = {"user": user, "title": title, "platform[]": ["youtube"],
                   "async_upload": "true"}
        payload.update(social_platforms.upload_fields(
            ["youtube"], title=title, description=description, yt_title=title))
        payload["privacyStatus"] = privacy
        if when:
            payload["scheduled_date"] = when
            payload["timezone"] = timezone
        if dry_run:
            say(f"publish (dry run): rank {rank} -> {user} YouTube, {privacy}"
                + (f", scheduled {when} {timezone}" if when else ", now")
                + f": {social_platforms.youtube_title(title)} | {' '.join(sent_tags)}")
            acted.append(dict(entry, would_send={k: v for k, v in payload.items()}))
            continue
        with open(video, "rb") as fh:
            resp = http.post(UPLOAD_URL, headers={"Authorization": f"Apikey {key}"},
                             data=payload,
                             files={"video": (os.path.basename(video), fh, "video/mp4")},
                             timeout=UPLOAD_TIMEOUT)
        if resp.status_code in (200, 201, 202):
            body = {}
            try:
                body = resp.json() or {}
            except ValueError:
                pass
            entry.update(status="posted", posted_at=_now(), profile=user, privacy=privacy,
                         hashtags_sent=sent_tags, credit_sent=credit_text,
                         request_id=str(body.get("request_id") or "") or None,
                         scheduled_for=when, timezone=timezone if when else None)
            entry.pop("error", None)
            say(f"publish: rank {rank} sent to YouTube via Upload-Post"
                + (f" (scheduled {when} {timezone})" if when else "")
                + (f", request {entry['request_id']}" if entry.get("request_id") else ""))
        else:
            reason = " ".join(str(resp.text or "").split())[:300]
            entry.update(status="publish_failed", tried_at=_now(),
                         error=f"Upload-Post {resp.status_code}: {reason}")
            say(f"publish: rank {rank} failed: Upload-Post {resp.status_code}")
        # Written after every upload: a crash mid-run never re-posts the ones done.
        _save(path, data)
        acted.append(entry)
    return acted


def post_status(out_dir, *, http, key=None, say=print):
    """Ask Upload-Post where each posted short ended; recorded per entry."""
    key = key if key is not None else api_key()
    if not key:
        raise PublishError("UPLOAD_POST_API_KEY is not set")
    _, path = _paths(out_dir)
    data = _load(path)
    out = []
    for e in data["shorts"]:
        if not isinstance(e, dict) or e.get("status") != "posted" or not e.get("request_id"):
            continue
        resp = http.get(STATUS_URL, headers={"Authorization": f"Apikey {key}"},
                        params={"request_id": e["request_id"]})
        try:
            body = resp.json() or {}
        except ValueError:
            body = {}
        results = [r for r in (body.get("results") or []) if isinstance(r, dict)]
        yt = next((r for r in results if r.get("platform") == "youtube"), {})
        e["upload_status"] = str(body.get("status") or "unknown")[:40]
        e["youtube_ok"] = yt.get("success")
        link = yt.get("url") or yt.get("post_url") or yt.get("video_url")
        if isinstance(link, str) and link.startswith("https://"):
            e["youtube_url"] = link[:300]
        msg = yt.get("message") or yt.get("error")
        if msg:
            e["youtube_message"] = " ".join(str(msg).split())[:300]
        e["checked_at"] = _now()
        say(f"status: rank {e.get('rank')}: {e['upload_status']}"
            + (f", {e['youtube_url']}" if e.get("youtube_url") else "")
            + (f" ({e['youtube_message']})" if e.get("youtube_message") else ""))
        out.append(e)
    _save(path, data)
    return out
