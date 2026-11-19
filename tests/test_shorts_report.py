"""shorts_report: the ranked report from a fixture job directory, with missing
keys and hostile strings, and the log-marker / file-discovery helpers."""
import json
import os

import shorts_report as sr

TITLE = "My_Talk"


def _rating(overall, summary="Strong open"):
    return {"scores": {"hook_strength": 8, "energy_pacing": 7,
                       "narrative_completeness": 9, "av_clarity": 6,
                       "retention_potential": 8},
            "rationale": {"hook_strength": "asks a question"},
            "summary": summary, "overall": overall}


def _metadata():
    c0 = {"start": 10.0, "end": 40.0, "candidate_index": 0, "rank": 2,
          "rating": _rating(71), "video_title_for_youtube_short": "Second best",
          "video_description_for_youtube_short": "Line one\nLine two",
          "youtube_hashtags": ["#shorts", "talk"], "viral_hook_text": "Wait for it"}
    c1 = {"start": 3700.0, "end": 3745.5, "candidate_index": 1, "rank": 1,
          "rating": _rating(88, "Bad | pipe\nand newline <b>x</b>"),
          "video_title_for_youtube_short": "Best | one\n# not a heading",
          "video_description_for_youtube_short": "# heading?\n---\n<script>x</script>",
          "youtube_hashtags": "#a #b, c", "viral_hook_text": "Hook | here"}
    c2 = {"start": "bogus", "end": None, "candidate_index": 2, "rank": 3}  # unrated
    return {"shorts": [dict(c1), dict(c0)], "candidates": [c0, c1, c2],
            "transcript": {"segments": []},
            "cost_analysis": {"input_tokens": 1000, "output_tokens": 200,
                              "total_cost": 0.0123, "model": "gemini-x"},
            "rating_cost_analysis": {"total_cost": 0.002},
            "source_video": "My_Talk.mp4"}


def _job(tmp_path, metadata):
    (tmp_path / f"{TITLE}_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return str(tmp_path)


class TestHelpers:
    def test_cell_escapes(self):
        out = sr.cell("a | b\nc <x> `y`")
        assert "\n" not in out
        assert "\\|" in out and "&lt;x&gt;" in out and "\\`" in out
        assert sr.cell(None) == ""
        assert sr.cell({"k": 1}) == '{"k": 1}'

    def test_hms(self):
        assert sr.hms(0) == "0:00:00"
        assert sr.hms(3745.9) == "1:02:25"
        assert sr.hms("x") == "?" and sr.hms(-1) == "?"

    def test_hashtags(self):
        assert sr.hashtags("#a #b, c") == ["#a", "#b", "#c"]
        assert sr.hashtags(["#x", "x", " y z"]) == ["#x", "#yz"]
        assert sr.hashtags(None) == []

    def test_num_rejects_bools_and_nan(self):
        assert sr.num(True) is None
        assert sr.num(float("nan")) is None
        assert sr.num("3.5") == 3.5

    def test_safe_basename(self):
        assert sr.safe_basename("a.mp4") == "a.mp4"
        for bad in ("../a.mp4", "/etc/passwd", "a\\b.mp4", ".hidden", "..", "", None, 5):
            assert sr.safe_basename(bad) is None

    def test_find_served_file_prefers_most_layers(self, tmp_path):
        for name in (f"{TITLE}_clip_1.mp4", f"hooked_100_{TITLE}_clip_1.mp4",
                     f"subtitled_101_hooked_100_{TITLE}_clip_1.mp4",
                     f"wm_subtitled_101_hooked_100_{TITLE}_clip_1.mp4",
                     f"temp_{TITLE}_clip_1.mp4", f"{TITLE}_clip_11.mp4"):
            (tmp_path / name).write_bytes(b"x")
        assert sr.find_served_file(str(tmp_path), TITLE, 0) == \
            f"subtitled_101_hooked_100_{TITLE}_clip_1.mp4"
        assert sr.find_served_file(str(tmp_path), TITLE, 10) == f"{TITLE}_clip_11.mp4"
        assert sr.find_served_file(str(tmp_path), TITLE, 4) is None
        assert sr.find_served_file(str(tmp_path), None, 0) is None

    def test_title_with_regex_chars(self, tmp_path):
        title = "a.b+(c)"
        (tmp_path / f"{title}_clip_1.mp4").write_bytes(b"x")
        (tmp_path / "aXb+(c)_clip_1.mp4").write_bytes(b"x")
        assert sr.find_served_file(str(tmp_path), title, 0) == f"{title}_clip_1.mp4"

    def test_caption_sidecar_by_generation_id(self, tmp_path):
        served = f"subtitled_1700000001_hooked_1700000000_{TITLE}_clip_1.mp4"
        (tmp_path / served).write_bytes(b"x")
        (tmp_path / "autosubs_1700000001_0123abcd.ass").write_text("x")
        (tmp_path / "autosubs_1700000002_0123abcd.ass").write_text("x")
        assert sr.find_caption_sidecar(str(tmp_path), served) == \
            "autosubs_1700000001_0123abcd.ass"
        assert sr.find_caption_sidecar(str(tmp_path), f"{TITLE}_clip_1.mp4") is None
        assert sr.find_caption_sidecar(str(tmp_path), "wm_" + served) == \
            "autosubs_1700000001_0123abcd.ass"

    def test_caption_sidecar_disambiguates_by_words(self, tmp_path):
        head = ("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
                "MarginV, Effect, Text\n")
        (tmp_path / "autosubs_5_aaaaaaaa.ass").write_text(
            head + "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,X\n")
        (tmp_path / "autosubs_5_bbbbbbbb.ass").write_text(
            head + "Dialogue: 0,0:00:00.50,0:00:01.00,Default,,0,0,0,,X\n")
        words = [{"word": "w", "start": 100.5, "end": 100.9}]
        assert sr.find_caption_sidecar(str(tmp_path), "subtitled_5_x.mp4", words,
                                       100.0, 110.0) == "autosubs_5_bbbbbbbb.ass"


class TestMarkers:
    def test_clip_ready(self):
        assert sr.parse_clip_ready("CLIP_READY 0 subtitled_1_My_Talk_clip_1.mp4") == \
            (0, "subtitled_1_My_Talk_clip_1.mp4")
        assert sr.parse_clip_ready("CLIP_READY 2 ../../etc/passwd") is None
        assert sr.parse_clip_ready("  CLIP_READY x y") is None
        assert sr.parse_clip_ready("something CLIP_READY 0 a.mp4") is None

    def test_stage_tracker_forward_only(self):
        t = sr.StageTracker()
        lines = [(0.5, "📥 Downloading video from YouTube..."),
                 (1.0, "📥 Download attempt: direct"),
                 (20.0, "🎛️  Choosing a layout for this video…"),
                 (25.0, "🎙️  Transcribing video..."),
                 (26.0, "📥 Download attempt: static (retry 1)"),  # not backwards
                 (60.0, "🤖  Analyzing with Gemini (2-pass: score → detail)..."),
                 (70.0, "🔥 Found 7 clips!"),
                 (71.0, "   Saved metadata to /some/dir/My_Talk_metadata.json"),
                 (72.0, "⭐ Rating 7 candidates"),
                 (80.0, "🎬 Processing Clip 1: 10.0s - 40.0s"),
                 (120.0, "CLIP_READY 0 subtitled_1_My_Talk_clip_1.mp4"),
                 (125.0, "🎬 Processing Clip 2: 1.0s - 30.0s"),
                 (160.0, "CLIP_READY 1 My_Talk_clip_2.mp4"),
                 (161.0, "⏱️  Total execution time: 161.00s")]
        for ts, line in lines:
            t.feed(line, ts)
        t.finish(162.0)
        names = [s["name"] for s in t.summary()]
        assert names == ["download", "layout", "transcription", "clip_selection",
                         "rating", "render", "done"]
        by = {s["name"]: s for s in t.summary()}
        assert by["download"]["duration_s"] == 19.5
        assert by["render"]["start_s"] == 80.0 and by["render"]["end_s"] == 161.0
        assert t.clip_ready == {0: "subtitled_1_My_Talk_clip_1.mp4", 1: "My_Talk_clip_2.mp4"}
        assert t.metadata_file == "My_Talk_metadata.json"
        assert t.clips_found == 7

    def test_clip_ready_opens_render_stage(self):
        t = sr.StageTracker()
        t.feed("CLIP_READY 0 a.mp4", 5.0)
        assert t.current == "render"

    def test_stages_from_log(self, tmp_path):
        log = tmp_path / "pipeline.log"
        log.write_text("[     0.0s] === start ===\n"
                       "[     1.0s] 🎙️  Transcribing video...\n"
                       "garbage line\n"
                       "[    10.0s] 🎬 Processing Clip 1: 1s - 2s\n"
                       "[    30.0s] CLIP_READY 0 x.mp4\n", encoding="utf-8")
        t = sr.stages_from_log(str(log))
        assert [s["name"] for s in t.summary()] == ["transcription", "render"]
        assert t.summary()[-1]["end_s"] == 30.0
        assert sr.stages_from_log(str(tmp_path / "none.log")) is None


class TestBuildReport:
    def test_full_report(self, tmp_path):
        job = _job(tmp_path, _metadata())
        (tmp_path / f"subtitled_9_hooked_8_{TITLE}_clip_1.mp4").write_bytes(b"x")
        (tmp_path / "stages.json").write_text(json.dumps({
            "exit_code": 0, "clip_ready": {"0": f"subtitled_9_hooked_8_{TITLE}_clip_1.mp4"},
            "stages": [{"name": "download", "start_s": 0, "end_s": 5, "duration_s": 5}]}))
        (tmp_path / "verification.json").write_text(json.dumps({"shorts": [
            {"short_index": 0, "file": f"subtitled_9_hooked_8_{TITLE}_clip_1.mp4",
             "captions": "autosubs_9_aaaaaaaa.ass",
             "result": {"ok": False, "checks": [
                 {"name": "aspect_9_16", "ok": True, "detail": "1080x1920"},
                 {"name": "duration", "ok": False, "detail": "61s | too long"},
                 {"name": "loudness", "ok": False, "detail": "-25 LUFS",
                  "informational": True}]}}]}))
        md, rep = sr.build_report(job)

        # Ranked: rank 1 first, then 2, then the unrated rank 3.
        assert [c["candidate_index"] for c in rep["candidates"]] == [1, 0, 2]
        top = rep["candidates"][0]
        assert top["overall"] == 88 and top["rendered"] is True
        assert top["file"] == f"subtitled_9_hooked_8_{TITLE}_clip_1.mp4"
        assert rep["candidates"][1]["rendered"] is False  # clip_2 not on disk
        assert rep["candidates"][2]["rated"] is False

        table = [ln for ln in md.splitlines() if ln.startswith("| 1 |")]
        assert len(table) == 1
        row = table[0]
        assert "1:01:40–1:02:25" in row and "45.5" in row
        assert "Bad \\| pipe and newline &lt;b&gt;x&lt;/b&gt;" in row
        # Each table row keeps its column count despite hostile text.
        header = next(ln for ln in md.splitlines() if ln.startswith("| Rank |"))
        assert row.replace("\\|", "").count("|") == header.count("|")

        # Hostile title / description cannot add headings or raw HTML.
        assert "# not a heading" not in [ln.strip() for ln in md.splitlines()]
        assert not any(ln.startswith("# heading?") for ln in md.splitlines())
        assert "> # heading?" in md and "&lt;script&gt;" in md
        assert "<script>" not in md

        assert rep["cost"]["total_usd"] == 0.0143
        assert rep["cost"]["per_stage"] == {"clip_selection": 0.0123, "rating": 0.002}
        assert "$0.0143" in md
        assert rep["stages"][0]["name"] == "download"

        issues = "\n".join(rep["issues"])
        assert "candidate #2 has no rating" in issues
        assert "FAILED duration" in issues and "warning loudness" in issues
        assert "short rank 2: no rendered file found" in issues
        assert "short rank 2: not verified" in issues
        assert "| WARN |" in md

    def test_missing_metadata(self, tmp_path):
        md, rep = sr.build_report(str(tmp_path))
        assert rep["candidates"] == [] and rep["shorts"] == []
        assert any("No *_metadata.json" in i for i in rep["issues"])
        assert "## Errors and edge cases" in md

    def test_invalid_metadata_json(self, tmp_path):
        (tmp_path / f"{TITLE}_metadata.json").write_text("{not json")
        _, rep = sr.build_report(str(tmp_path))
        assert any("not valid JSON" in i for i in rep["issues"])

    def test_old_metadata_without_new_keys(self, tmp_path):
        meta = {"shorts": [{"start": 1, "end": 31, "video_title_for_youtube_short": "Old"},
                           "garbage", {"start": 40, "end": 70}],
                "transcript": {"segments": []}}
        job = _job(tmp_path, meta)
        (tmp_path / f"{TITLE}_clip_1.mp4").write_bytes(b"x")
        md, rep = sr.build_report(job)
        assert len(rep["candidates"]) == 3
        assert rep["shorts"][0]["rank"] == 1
        assert rep["shorts"][0]["file"] == f"{TITLE}_clip_1.mp4"
        assert rep["candidates"][0]["rendered"] is True
        assert rep["cost"] is None
        issues = "\n".join(rep["issues"])
        assert "no `candidates` list" in issues
        assert "No Gemini cost recorded" in issues
        assert "no rating" in issues

    def test_stage_timings_from_log_and_failed_exit(self, tmp_path):
        job = _job(tmp_path, {"shorts": []})
        (tmp_path / "pipeline.log").write_text(
            "[     2.0s] 🎙️  Transcribing video...\n[     9.0s] 🔥 Found 0 clips!\n",
            encoding="utf-8")
        _, rep = sr.build_report(job)
        assert rep["stages"][0]["name"] == "transcription"
        (tmp_path / "stages.json").write_text(json.dumps({"exit_code": 1, "stages": [],
                                                          "errors": ["run: boom"]}))
        _, rep = sr.build_report(job)
        assert any("exited with code 1" in i for i in rep["issues"])
        assert "run: boom" in rep["issues"]

    def test_cost_with_stage_breakdown(self):
        c = sr.cost_summary({"cost_analysis": {"total_cost": 0.05, "stages": {
            "selection": {"total_cost": 0.03}, "rating": 0.01}}})
        assert c["per_stage"] == {"selection": 0.03, "rating": 0.01}
        assert c["total_usd"] == 0.05
        assert sr.cost_summary({"cost_analysis": {"total_cost": "NaN"}}) is None

    def test_cli_writes_files(self, tmp_path):
        job = _job(tmp_path, _metadata())
        assert sr.main([job]) == 0
        assert os.path.isfile(os.path.join(job, "report.md"))
        data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert data["metadata_file"] == f"{TITLE}_metadata.json"
        assert sr.main([str(tmp_path / "nope")]) == 2
