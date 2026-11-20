"""shorts_pipeline: run-stage streaming with a fake main.py process, verify
and package stages on fake files. No ffmpeg, no network, no upload."""
import io
import json
import os
import subprocess

import pytest

import shorts_pipeline as sp

TITLE = "My_Talk"


def _metadata():
    return {
        "shorts": [
            {"start": 100.0, "end": 130.0, "rank": 1, "candidate_index": 4,
             "rating": {"overall": 90, "scores": {}, "summary": "great"},
             "video_title_for_youtube_short": "Best one",
             "video_description_for_youtube_short": "Desc\r\nline 2",
             "youtube_hashtags": ["shorts", "#ai"], "viral_hook_text": "Look"},
            {"start": 10.0, "end": 35.0, "rank": 2, "candidate_index": 1},
            {"start": 50.0, "end": 70.0, "rank": 3, "candidate_index": 0},
        ],
        "transcript": {"segments": [{"words": [
            {"word": "hi", "start": 100.2, "end": 100.5},
            {"word": "there", "start": 100.6, "end": 101.0}]}]},
        "cost_analysis": {"total_cost": 0.01},
        "source_video": "My_Talk.mp4",
    }


@pytest.fixture
def job(tmp_path):
    (tmp_path / f"{TITLE}_metadata.json").write_text(json.dumps(_metadata()))
    (tmp_path / f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4").write_bytes(b"one")
    (tmp_path / "autosubs_7_abcdef01.ass").write_text("[Events]\n")
    (tmp_path / f"{TITLE}_clip_2.mp4").write_bytes(b"two")
    return tmp_path


class TestSlugAndEnv:
    def test_slug(self):
        assert sp.slug_for("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5") == "dQw4w9WgXcQ"
        assert sp.slug_for("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert sp.slug_for("https://www.youtube.com/shorts/abcDEF12345") == "abcDEF12345"
        s = sp.slug_for("https://example.com/videos/talk.mp4")
        assert s.startswith("videos-talk-mp4-") and "/" not in s
        assert "/" not in sp.slug_for("not a url")

    def test_child_env(self):
        env = sp.child_env(90, 3, "gemini-x", None, base={"PATH": "p"})
        assert env["AUTO_LAYOUT"] == env["AUTO_HOOK"] == env["AUTO_CAPTIONS"] == "1"
        assert env["RATE_CLIPS"] == "1" and env["RENDER_TOP_N"] == "3"
        assert env["MAX_SOURCE_MINUTES"] == "90"
        assert env["CLIP_WORKERS"] == sp.default_clip_workers() and env["GEMINI_MODEL"] == "gemini-x"
        assert env["PATH"] == "p"
        env = sp.child_env(1.5, 2, None, None, base={"CLIP_WORKERS": "4"})
        assert env["CLIP_WORKERS"] == "4" and "GEMINI_MODEL" not in env
        assert env["MAX_SOURCE_MINUTES"] == "1.5"
        assert sp.child_env(1, 1, None, 2, base={})["CLIP_WORKERS"] == "2"

    def test_default_workers_follow_the_gpu(self):
        assert sp.default_clip_workers(lambda name: "/usr/bin/nvidia-smi") == "1"
        assert sp.default_clip_workers(lambda name: None) == "3"

    def test_main_command_on_a_local_file(self):
        cmd = sp.main_command("https://x/y", "/out", "/out/source.mp4")
        assert cmd[cmd.index("-i") + 1] == "/out/source.mp4"
        assert "-u" not in cmd[3:] and "https://x/y" not in cmd

    def test_main_command(self):
        cmd = sp.main_command("https://x/y", "/out")
        assert cmd[1:2] == ["-u"] and cmd[2].endswith("main.py")
        assert cmd[3:] == ["-u", "https://x/y", "--keep-original", "-o", "/out",
                           "--format", "vertical"]


class _FakeProc:
    def __init__(self, lines, code):
        self.stdout = io.BytesIO("".join(l + "\n" for l in lines).encode("utf-8"))
        self._code = code

    def wait(self):
        return self._code

    def terminate(self):
        pass


def _fake_popen(lines, code, seen):
    def popen(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return _FakeProc(lines, code)
    return popen


def _clock():
    t = {"v": 0.0}

    def tick():
        t["v"] += 1.0
        return t["v"]
    return tick


class TestRunStage:
    LINES = ["📥 Downloading video from YouTube...",
             "🎙️  Transcribing video...",
             "🤖  Analyzing with Gemini (2-pass: score → detail)...",
             "🔥 Found 3 clips!",
             "🎬 Processing Clip 1: 100.0s - 130.0s",
             f"CLIP_READY 0 subtitled_7_hooked_6_{TITLE}_clip_1.mp4",
             "CLIP_READY 1 ../escape.mp4",
             "⏱️  Total execution time: 9.00s"]

    def test_streams_and_records(self, tmp_path, capsys):
        seen = {}
        state = sp.run_stage("https://youtu.be/abcdefghijk", str(tmp_path), 90, 3,
                             popen=_fake_popen(self.LINES, 0, seen), clock=_clock())
        assert state["exit_code"] == 0
        assert seen["kw"]["env"]["RENDER_TOP_N"] == "3"
        assert seen["kw"]["stderr"] == subprocess.STDOUT
        assert "shell" not in seen["kw"]
        stages = json.loads((tmp_path / "stages.json").read_text(encoding="utf-8"))
        assert [s["name"] for s in stages["stages"]] == [
            "download", "transcription", "clip_selection", "render", "done"]
        assert stages["clip_ready"] == {"0": f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4"}
        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert "s] 🎙️  Transcribing video..." in log
        assert "CLIP_READY 0" in capsys.readouterr().out
        run = json.loads((tmp_path / "run.json").read_text())
        assert run["top"] == 3 and run["minutes"] == 90

    def test_nonzero_exit_is_recorded(self, tmp_path):
        state = sp.run_stage("u", str(tmp_path), 5, 1,
                             popen=_fake_popen(["boom"], 3, {}), clock=_clock())
        assert state["exit_code"] == 3
        stages = json.loads((tmp_path / "stages.json").read_text())
        assert any("exited with code 3" in e for e in stages["errors"])

    def test_popen_failure(self, tmp_path):
        def bad(*a, **k):
            raise OSError("no python")
        state = sp.run_stage("u", str(tmp_path), 5, 1, popen=bad, clock=_clock())
        assert state["exit_code"] == -1


class TestVerifyStage:
    def test_uses_clip_ready_sidecar_and_words(self, job):
        (job / "stages.json").write_text(json.dumps(
            {"clip_ready": {"0": f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4",
                            "2": "../../etc/passwd"}}))
        calls = []

        def fake_check(path, **kw):
            calls.append((os.path.basename(path), kw))
            return {"file": path, "ok": True, "checks": []}

        data = sp.verify_stage(str(job), check=fake_check)
        assert [c[0] for c in calls] == [f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4",
                                         f"{TITLE}_clip_2.mp4"]
        first = calls[0][1]
        assert os.path.basename(first["captions_path"]) == "autosubs_7_abcdef01.ass"
        assert first["clip_start"] == 100.0 and len(first["transcript_words"]) == 2
        assert first["max_seconds"] == 60.0
        assert calls[1][1]["captions_path"] is None
        rows = data["shorts"]
        assert rows[0]["result"]["file"] == f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4"
        assert rows[2]["file"] is None and rows[2]["result"] is None
        assert any("rank 3 has no rendered file" in e for e in data["errors"])
        assert (job / "verification.json").exists()

    def test_check_exception_is_contained(self, job):
        def boom(path, **kw):
            raise RuntimeError("bad")
        data = sp.verify_stage(str(job), check=boom)
        assert data["shorts"][0]["result"]["ok"] is False

    def test_no_metadata(self, tmp_path):
        data = sp.verify_stage(str(tmp_path), check=lambda p, **k: {})
        assert data["shorts"] == [] and data["errors"]


class TestPackageStage:
    def test_package(self, job):
        (job / "verification.json").write_text(json.dumps({"shorts": [
            {"short_index": 0, "file": f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4",
             "captions": "autosubs_7_abcdef01.ass",
             "result": {"ok": False, "checks": [
                 {"name": "duration", "ok": False, "detail": "x"},
                 {"name": "loudness", "ok": False, "detail": "y", "informational": True}]}}]}))
        (job / "run.json").write_text(json.dumps({"url": "https://youtu.be/abcdefghijk"}))
        approval = sp.package_stage(str(job))

        pub = job / "publish"
        d1 = pub / "short_1"
        assert (d1 / "short_1.mp4").read_bytes() == b"one"
        assert (d1 / "title.txt").read_text(encoding="utf-8") == "Best one\n"
        assert (d1 / "description.txt").read_text(encoding="utf-8") == "Desc\nline 2\n"
        assert (d1 / "hashtags.txt").read_text(encoding="utf-8") == "#shorts #ai\n"
        meta = json.loads((d1 / "metadata.json").read_text(encoding="utf-8"))
        assert meta["rating"]["overall"] == 90
        assert meta["source"]["start_hms"] == "0:01:40" and meta["source"]["duration"] == 30.0
        assert meta["source"]["url"] == "https://youtu.be/abcdefghijk"
        assert meta["verification"]["failed"] == ["duration"]
        assert meta["verification"]["warnings"] == ["loudness"]
        assert meta["captions_file"] == "autosubs_7_abcdef01.ass"
        assert (pub / "short_2" / "short_2.mp4").read_bytes() == b"two"
        assert not (pub / "short_3").exists()

        assert [e["rank"] for e in approval["shorts"]] == [1, 2]
        assert all(e["status"] == "pending_approval" for e in approval["shorts"])
        assert any("rank 3" in e for e in approval["errors"])
        on_disk = json.loads((pub / "approval.json").read_text(encoding="utf-8"))
        assert on_disk["shorts"][0]["video"] == "short_1/short_1.mp4"

    def test_rerun_keeps_decisions_and_drops_stale(self, job):
        sp.package_stage(str(job))
        path = job / "publish" / "approval.json"
        data = json.loads(path.read_text())
        data["shorts"][0]["status"] = "approved"
        data["shorts"][1]["status"] = "bogus"
        path.write_text(json.dumps(data))
        (job / "publish" / "short_9").mkdir()
        approval = sp.package_stage(str(job))
        assert approval["shorts"][0]["status"] == "approved"
        assert approval["shorts"][1]["status"] == "pending_approval"
        assert not (job / "publish" / "short_9").exists()

    def test_a_changed_video_loses_its_approval(self, job):
        # The review belongs to the exact video reviewed: a re-render with the
        # same rank and file name must come back as pending.
        sp.package_stage(str(job))
        path = job / "publish" / "approval.json"
        data = json.loads(path.read_text())
        data["shorts"][0]["status"] = "approved"
        path.write_text(json.dumps(data))
        src = job / data["shorts"][0]["source_file"]
        src.write_bytes(b"re-rendered")
        approval = sp.package_stage(str(job))
        assert approval["shorts"][0]["status"] == "pending_approval"
        # And the packaged copy is not the same inode as the render.
        packaged = job / "publish" / "short_1" / "short_1.mp4"
        assert packaged.read_bytes() == b"re-rendered"
        src.write_bytes(b"changed again")
        assert packaged.read_bytes() == b"re-rendered"

    def test_duplicate_or_missing_ranks(self, tmp_path):
        meta = {"shorts": [{"start": 0, "end": 20, "rank": 1},
                           {"start": 30, "end": 50, "rank": 1},
                           {"start": 60, "end": 80}]}
        (tmp_path / f"{TITLE}_metadata.json").write_text(json.dumps(meta))
        for i in (1, 2, 3):
            (tmp_path / f"{TITLE}_clip_{i}.mp4").write_bytes(b"x")
        approval = sp.package_stage(str(tmp_path))
        assert sorted(e["rank"] for e in approval["shorts"]) == [1, 2, 3]


class TestCli:
    def test_needs_url_for_run(self, tmp_path):
        with pytest.raises(SystemExit):
            sp.main(["--stage", "run", "--out", str(tmp_path)])

    def test_report_and_package_on_existing_dir(self, job):
        code = sp.main(["--stage", "report", "--out", str(job)])
        assert code == 0
        assert (job / "report.md").exists() and (job / "report.json").exists()
        # rank 3 has no file -> package reports a problem but still writes
        assert sp.main(["--stage", "package", "--out", str(job)]) == 1
        assert (job / "publish" / "approval.json").exists()

    def test_all_never_publishes(self, monkeypatch, job):
        # Posting lives in shorts_publish and runs only as --stage publish.
        assert "publish" not in sp.STAGES and "publish" in sp.MANUAL_STAGES
        called = []
        monkeypatch.setattr(sp, "manual_stage", lambda args: called.append(args) or 0)
        sp.main(["--stage", "report", "--out", str(job)])
        assert called == []
        src = open(sp.__file__, encoding="utf-8").read().lower()
        for word in ("api.upload-post.com", "requests.", "urllib.request", "http.client"):
            assert word not in src

    def test_package_rerun_keeps_a_posted_record(self, job):
        sp.package_stage(str(job))
        path = job / "publish" / "approval.json"
        data = json.loads(path.read_text())
        data["shorts"][0].update(status="posted", request_id="r1", posted_at="t")
        path.write_text(json.dumps(data))
        again = sp.package_stage(str(job))
        assert again["shorts"][0]["status"] == "posted"
        assert again["shorts"][0]["request_id"] == "r1"


class TestEdit:
    """The house edit stage, with a fake editor (no ffmpeg)."""

    @staticmethod
    def _editor(calls, fail_on=None):
        def edit(clip, words, hook, out, work, **kw):
            calls.append({"clip": os.path.basename(clip), "words": words, "hook": hook,
                          "kw": kw})
            if fail_on and fail_on in clip:
                raise RuntimeError("ffmpeg failed")
            os.makedirs(work, exist_ok=True)
            with open(out, "wb") as f:
                f.write(b"edited " + os.path.basename(clip).encode())
            ass = os.path.join(work, "captions.ass")
            with open(ass, "w") as f:
                f.write("[Events]\n")
            return {"captions": ass, "seconds": 25.0, "cuts": 2, "cold_open_from": 3.0,
                    "words": [{"word": "hi", "start": 1.8, "end": 2.1}]}
        return edit

    def test_edits_every_short_from_its_clean_clip(self, job):
        (job / f"{TITLE}_clip_1.mp4").write_bytes(b"clean one")
        calls = []
        rec = sp.edit_stage(str(job), editor=self._editor(calls), say=lambda *_: None)
        assert rec["errors"] == []
        assert [c["clip"] for c in calls] == [f"{TITLE}_clip_1.mp4", f"{TITLE}_clip_2.mp4"]
        assert calls[0]["hook"] == "Look"
        assert calls[0]["words"] == [{"word": "hi", "start": 0.2, "end": 0.5},
                                     {"word": "there", "start": 0.6, "end": 1.0}]
        assert not any((job / ".edit_work_1").glob("*"))
        items = sp.rendered_shorts(str(job), *sp.load_job(str(job))[:2])
        assert [i["file"] for i in items[:2]] == ["edited_clip_1.mp4", "edited_clip_2.mp4"]

        seen = []

        def fake_check(path, **kw):
            seen.append((os.path.basename(path), kw["captions_path"], kw["transcript_words"],
                         kw["clip_start"]))
            return {"ok": True, "checks": []}
        sp.verify_stage(str(job), check=fake_check)
        name, caps, words, start = seen[0]
        assert name == "edited_clip_1.mp4" and caps.endswith("edited_clip_1.ass")
        assert words == [{"word": "hi", "start": 1.8, "end": 2.1}] and start == 0.0

        approval = sp.package_stage(str(job))
        assert approval["shorts"][0]["source_file"] == "edited_clip_1.mp4"
        assert (job / "publish" / "short_1" / "short_1.mp4").read_bytes() \
            == b"edited " + f"{TITLE}_clip_1.mp4".encode()

    def test_a_failed_edit_keeps_the_render(self, job):
        (job / f"{TITLE}_clip_1.mp4").write_bytes(b"clean one")
        rec = sp.edit_stage(str(job), editor=self._editor([], fail_on="clip_2"),
                            say=lambda *_: None)
        assert len(rec["errors"]) == 1 and "rank 2" in rec["errors"][0]
        items = sp.rendered_shorts(str(job), *sp.load_job(str(job))[:2])
        assert items[0]["file"] == "edited_clip_1.mp4"
        assert items[1]["file"] == f"{TITLE}_clip_2.mp4" and items[1]["edit"] is None

    def test_no_clean_clip_and_style_none(self, job):
        calls = []
        rec = sp.edit_stage(str(job), editor=self._editor(calls), say=lambda *_: None)
        assert "no source clip" in rec["errors"][0]  # clip_1's clean cut is missing
        sp.edit_stage(str(job), style="none", editor=self._editor(calls), say=lambda *_: None)
        items = sp.rendered_shorts(str(job), *sp.load_job(str(job))[:2])
        assert items[1]["file"] == f"{TITLE}_clip_2.mp4"

    def test_an_edit_of_another_render_is_ignored(self, job):
        (job / f"{TITLE}_clip_1.mp4").write_bytes(b"clean one")
        sp.edit_stage(str(job), editor=self._editor([]), say=lambda *_: None)
        data = json.loads((job / sp.EDITS_FILE).read_text())
        data["shorts"]["0"]["source_file"] = "subtitled_9_hooked_9_other.mp4"
        (job / sp.EDITS_FILE).write_text(json.dumps(data))
        items = sp.rendered_shorts(str(job), *sp.load_job(str(job))[:2])
        assert items[0]["file"] == f"subtitled_7_hooked_6_{TITLE}_clip_1.mp4"

    def test_base_clip_and_words(self, job):
        assert sp.base_clip(str(job), "subtitled_7_hooked_6_x/../../etc.mp4") is None
        assert sp.clip_words([{"word": "a", "start": 9.0, "end": 10.5},
                              {"word": "b", "start": 20.0, "end": 21.0}], 10.0, 20.0) \
            == [{"word": "a", "start": 0.0, "end": 0.5}]

    @staticmethod
    def _run(job, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", stream=True):
        (job / "run.json").write_text(json.dumps({"url": url, "stream": stream}))

    def test_facecam_template_when_the_source_has_a_webcam(self, job):
        (job / f"{TITLE}_clip_1.mp4").write_bytes(b"clean one")
        (job / "My_Talk.mp4").write_bytes(b"source")
        self._run(job)
        framed = []

        def framer(source, start, end, work):
            framed.append((os.path.basename(source), start, end))
            if start == 10.0:
                return None  # no webcam in this stretch
            os.makedirs(work, exist_ok=True)
            path = os.path.join(work, "template.mp4")
            with open(path, "wb") as f:
                f.write(b"stacked")
            return path, 730
        calls = []
        rec = sp.edit_stage(str(job), editor=self._editor(calls), framer=framer,
                            say=lambda *_: None)
        assert framed == [("My_Talk.mp4", 100.0, 130.0), ("My_Talk.mp4", 10.0, 35.0)]
        assert calls[0]["clip"] == "template.mp4" and calls[0]["kw"] == {"zoom_from": 730}
        assert calls[1]["clip"] == f"{TITLE}_clip_2.mp4" and calls[1]["kw"] == {}
        assert [rec["shorts"][k]["layout"] for k in ("0", "1")] == ["facecam", "main"]

    def test_a_failing_template_keeps_main_framing(self, job):
        (job / f"{TITLE}_clip_1.mp4").write_bytes(b"clean one")
        (job / "My_Talk.mp4").write_bytes(b"source")
        self._run(job)

        def framer(*a):
            raise RuntimeError("no decoder")
        calls = []
        rec = sp.edit_stage(str(job), editor=self._editor(calls), framer=framer,
                            say=lambda *_: None)
        assert rec["errors"] == [] and calls[0]["clip"] == f"{TITLE}_clip_1.mp4"
        sp.edit_stage(str(job), editor=self._editor(calls), framer=framer, template="none",
                      say=lambda *_: None)
        assert calls[-1]["kw"] == {}

    def test_template_only_for_a_youtube_stream(self, job):
        (job / f"{TITLE}_clip_1.mp4").write_bytes(b"clean one")
        (job / "My_Talk.mp4").write_bytes(b"source")
        asked = []

        def framer(source, start, end, work):
            asked.append(start)
            return None

        def run_edit():
            asked.clear()
            sp.edit_stage(str(job), editor=self._editor([]), framer=framer,
                          say=lambda *_: None)
            return bool(asked)
        assert run_edit() is False                       # no run.json: unknown source
        self._run(job, stream=False)
        assert run_edit() is False                       # a YouTube video, not a stream
        self._run(job, url="https://www.twitch.tv/videos/123", stream=True)
        assert run_edit() is False                       # a stream, not on YouTube
        self._run(job, stream=True)
        assert run_edit() is True                        # stream_batch's chunks
        self._run(job, url="https://youtu.be/dQw4w9WgXcQ", stream=False)
        meta = json.loads((job / f"{TITLE}_metadata.json").read_text())
        meta["source_live_status"] = "was_live"
        (job / f"{TITLE}_metadata.json").write_text(json.dumps(meta))
        assert run_edit() is True                        # a URL run of a stream VOD
