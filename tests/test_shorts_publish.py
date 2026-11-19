"""shorts_publish: the approval gate in front of every post. Fake HTTP only."""
import hashlib
import json

import pytest

import shorts_publish as pub

KEY = "test-key"


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text or json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _Http:
    def __init__(self, profiles=None, post_status=200, post_body=None, status_body=None):
        self.profiles = profiles if profiles is not None else [
            {"username": "me", "social_accounts": {"youtube": {"handle": "x"}}}]
        self.post_status = post_status
        self.post_body = post_body if post_body is not None else {"request_id": "req-1"}
        self.status_body = status_body or {}
        self.posts = []
        self.gets = []

    def get(self, url, headers=None, params=None):
        self.gets.append((url, params))
        if url == pub.USERS_URL:
            return _Resp(200, {"profiles": self.profiles})
        return _Resp(200, self.status_body)

    def post(self, url, headers=None, data=None, files=None, timeout=None):
        assert headers["Authorization"] == f"Apikey {KEY}"
        name, fh, mime = files["video"]
        self.posts.append({"url": url, "data": dict(data), "bytes": fh.read(), "name": name})
        return _Resp(self.post_status, self.post_body, text="nope")


@pytest.fixture
def job(tmp_path):
    publish = tmp_path / "publish"
    shorts = []
    for rank in (1, 2, 3):
        d = publish / f"short_{rank}"
        d.mkdir(parents=True)
        (d / f"short_{rank}.mp4").write_bytes(f"video-{rank}".encode())
        (d / "title.txt").write_text(f"Title {rank}\n", encoding="utf-8")
        (d / "description.txt").write_text(f"Desc {rank} #gaming #Shorts\n", encoding="utf-8")
        shorts.append({"rank": rank, "dir": f"short_{rank}",
                       "video": f"short_{rank}/short_{rank}.mp4",
                       "title": f"Title {rank}", "status": "pending_approval"})
    (publish / "approval.json").write_text(json.dumps({"shorts": shorts}), encoding="utf-8")
    return tmp_path


def _approval(job):
    return json.loads((job / "publish" / "approval.json").read_text(encoding="utf-8"))


def test_nothing_is_posted_without_approval(job):
    http = _Http()
    assert pub.publish(str(job), http=http, key=KEY, say=lambda *_: None) == []
    assert http.posts == []


def test_approved_shorts_are_posted_once(job):
    pub.approve(str(job), [1, 3])
    http = _Http()
    acted = pub.publish(str(job), http=http, key=KEY, say=lambda *_: None)
    assert [e["rank"] for e in acted] == [1, 3]
    assert [p["bytes"] for p in http.posts] == [b"video-1", b"video-3"]
    sent = http.posts[0]["data"]
    assert sent["user"] == "me" and sent["platform[]"] == ["youtube"]
    assert sent["youtube_title"] == "Title 1" and sent["privacyStatus"] == "public"
    assert sent["youtube_description"] == "Desc 1\n\n#gaming #Shorts"
    assert "scheduled_date" not in sent
    data = _approval(job)
    assert [e["status"] for e in data["shorts"]] == ["posted", "pending_approval", "posted"]
    assert data["shorts"][0]["request_id"] == "req-1"
    # A second run posts nothing again.
    again = _Http()
    assert pub.publish(str(job), http=again, key=KEY, say=lambda *_: None) == []
    assert again.posts == []


def test_a_video_changed_after_approval_is_not_posted(job):
    pub.approve(str(job), [1])
    (job / "publish" / "short_1" / "short_1.mp4").write_bytes(b"re-rendered")
    http = _Http()
    pub.publish(str(job), http=http, key=KEY, say=lambda *_: None)
    assert http.posts == []
    assert _approval(job)["shorts"][0]["status"] == "pending_approval"


def test_failure_is_recorded_and_retryable_by_approving_again(job):
    pub.approve(str(job), [2])
    http = _Http(post_status=401)
    pub.publish(str(job), http=http, key=KEY, say=lambda *_: None)
    e = _approval(job)["shorts"][1]
    assert e["status"] == "publish_failed" and "401" in e["error"]
    assert KEY not in json.dumps(_approval(job))


def test_scheduled_one_a_day_with_privacy(job):
    pub.approve(str(job), [1, 2, 3])
    http = _Http()
    pub.publish(str(job), http=http, key=KEY, privacy="unlisted",
                start="2026-11-20T18:00", every_hours=24, timezone="Asia/Kolkata",
                say=lambda *_: None)
    dates = [p["data"]["scheduled_date"] for p in http.posts]
    assert dates == ["2026-11-20T18:00", "2026-11-21T18:00", "2026-11-22T18:00"]
    assert all(p["data"]["timezone"] == "Asia/Kolkata" for p in http.posts)
    assert all(p["data"]["privacyStatus"] == "unlisted" for p in http.posts)


def test_dry_run_sends_nothing(job):
    pub.approve(str(job), [1])
    http = _Http()
    acted = pub.publish(str(job), http=http, key=KEY, dry_run=True, say=lambda *_: None)
    assert http.posts == [] and acted[0]["would_send"]["youtube_title"] == "Title 1"
    assert _approval(job)["shorts"][0]["status"] == "approved"


def test_profile_choice(job):
    two = [{"username": "a", "social_accounts": {"youtube": {}}},
           {"username": "b", "social_accounts": {"youtube": {}}},
           {"username": "c", "social_accounts": {"tiktok": {}}}]
    with pytest.raises(pub.PublishError, match="several"):
        pub.youtube_profile(_Http(profiles=two), KEY)
    assert pub.youtube_profile(_Http(profiles=two), KEY, "b") == "b"
    with pytest.raises(pub.PublishError, match="no YouTube"):
        pub.youtube_profile(_Http(profiles=two), KEY, "c")
    with pytest.raises(pub.PublishError, match="no Upload-Post profile"):
        pub.youtube_profile(_Http(profiles=[two[2]]), KEY)


def test_no_key_no_post(job, monkeypatch):
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)
    pub.approve(str(job), [1])
    with pytest.raises(pub.PublishError, match="UPLOAD_POST_API_KEY"):
        pub.publish(str(job), http=_Http(), say=lambda *_: None)


def test_entry_video_outside_publish_is_refused(job):
    data = _approval(job)
    data["shorts"][0]["video"] = "../../etc/passwd"
    (job / "publish" / "approval.json").write_text(json.dumps(data))
    with pytest.raises(pub.PublishError, match="missing"):
        pub.approve(str(job), [1])


def test_approve_records_the_reviewed_bytes(job):
    pub.approve(str(job), [2])
    e = _approval(job)["shorts"][1]
    assert e["sha256"] == hashlib.sha256(b"video-2").hexdigest()
    assert e["status"] == "approved" and e["decided_at"]
    with pytest.raises(pub.PublishError, match="no short with rank 9"):
        pub.approve(str(job), [9])


def test_post_status_records_the_link(job):
    pub.approve(str(job), [1])
    pub.publish(str(job), http=_Http(), key=KEY, say=lambda *_: None)
    http = _Http(status_body={"status": "completed", "results": [
        {"platform": "youtube", "success": True, "url": "https://youtube.com/shorts/abc"}]})
    pub.post_status(str(job), http=http, key=KEY, say=lambda *_: None)
    e = _approval(job)["shorts"][0]
    assert e["upload_status"] == "completed" and e["youtube_ok"] is True
    assert e["youtube_url"] == "https://youtube.com/shorts/abc"


def test_pinned_tags_come_first_and_shorts_last():
    desc, tags = pub.with_pinned_tags(
        "Total chaos on the road. #gaming #funny #roadrage #Shorts",
        ["#gaming", "#funny", "#roadrage", "#Shorts"], ["#GameName", "SomeStreamer", "#gaming"])
    assert tags == ["#GameName", "#SomeStreamer", "#gaming", "#funny", "#roadrage", "#Shorts"]
    assert desc == "Total chaos on the road.\n\n#GameName #SomeStreamer #gaming #funny #roadrage #Shorts"
    # Idempotent, capped, malformed dropped.
    again, tags2 = pub.with_pinned_tags(desc, tags, ["#GameName"])
    assert again == desc and tags2 == tags
    _, capped = pub.with_pinned_tags("x", ["#a", "#b", "#c", "#d", "#e", "#f"], ["#bad tag", "#123"])
    assert capped == ["#a", "#b", "#c", "#d", "#e", "#Shorts"]


def test_credit_links_the_moment_in_the_stream():
    src = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "stream_start": 53818.16,
           "start": 12.0, "channel": "SomeStreamer"}
    assert pub.source_link(src) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=53818s"
    assert pub.credit_line(src) == ("🎥 Credit: SomeStreamer · full stream: "
                                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=53818s")
    assert pub.credit_line({"url": "https://youtu.be/dQw4w9WgXcQ", "start": 0}, "Speed") \
        == "🎥 Credit: Speed · full stream: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert pub.source_link({"url": "https://www.twitch.tv/videos/123", "start": 3725}) \
        == "https://www.twitch.tv/videos/123?t=1h2m5s"
    # Data, not instructions: other hosts, http and odd ids give no link.
    for bad in ("https://evil.example/watch?v=dQw4w9WgXcQ", "http://youtube.com/watch?v=dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=x", "javascript:alert(1)", ""):
        assert pub.credit_line({"url": bad}) is None
    assert pub.credit_line(src, "#Speed <b>") == pub.credit_line(src, "Speed b")


def test_publish_adds_the_credit_above_the_tags(job):
    pub.approve(str(job), [1])
    meta = {"source": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "start": 90}}
    (job / "publish" / "short_1" / "metadata.json").write_text(json.dumps(meta))
    http = _Http()
    pub.publish(str(job), http=http, key=KEY, credit_name="SomeStreamer", say=lambda *_: None)
    sent = http.posts[0]["data"]["youtube_description"]
    assert sent == ("Desc 1\n\n🎥 Credit: SomeStreamer · full stream: "
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s\n\n#gaming #Shorts")
    assert _approval(job)["shorts"][0]["credit_sent"].startswith("🎥 Credit: SomeStreamer")
    # Opt-out leaves the description as it was.
    pub.approve(str(job), [2])
    http2 = _Http()
    pub.publish(str(job), http=http2, key=KEY, credit=False, say=lambda *_: None)
    assert "Credit" not in http2.posts[0]["data"]["youtube_description"]


def test_publish_sends_the_pinned_tags(job):
    pub.approve(str(job), [1])
    (job / "publish" / "short_1" / "hashtags.txt").write_text("#gaming #Shorts\n")
    http = _Http()
    pub.publish(str(job), http=http, key=KEY, pinned_tags=["#GameName"], say=lambda *_: None)
    sent = http.posts[0]["data"]["youtube_description"]
    assert sent.endswith("#GameName #gaming #Shorts")
    assert _approval(job)["shorts"][0]["hashtags_sent"] == ["#GameName", "#gaming", "#Shorts"]
