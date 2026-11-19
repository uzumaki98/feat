"""verify_shorts: pure checks run without ffmpeg; the real-file checks skip
when ffmpeg is not installed."""
import json
import shutil
import subprocess

import pytest

import verify_shorts as vs


def _ass(events):
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResY: 288\n\n[V4+ Styles]\n"
            "Format: Name, Fontname\nStyle: Default,Anton\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    body = "\n".join(f"Dialogue: 0,{vs_time(a)},{vs_time(b)},Default,,0,0,0,,{t}"
                     for a, b, t in events)
    return head + body + "\n"


def vs_time(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


class TestPureChecks:
    def test_aspect(self):
        assert vs.aspect_check(1080, 1920)["ok"] is True
        assert vs.aspect_check(720, 1280)["ok"] is True
        assert vs.aspect_check(1920, 1080)["ok"] is False
        assert vs.aspect_check(1080, 1080)["ok"] is False
        assert vs.aspect_check(0, 0)["ok"] is False
        # within the 0.01 tolerance (odd encoder sizes)
        assert vs.aspect_check(1088, 1920)["ok"] is True

    def test_duration(self):
        assert vs.duration_check(30.0)["ok"] is True
        assert vs.duration_check(60.0)["ok"] is True
        assert vs.duration_check(60.5)["ok"] is False
        assert vs.duration_check(4.0)["ok"] is False
        assert vs.duration_check(None)["ok"] is False
        assert vs.duration_check(90, max_seconds=120)["ok"] is True

    def test_parse_probe(self):
        data = {"streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1080,
             "height": 1920, "avg_frame_rate": "30000/1001", "r_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac"}],
            "format": {"duration": "31.5"}}
        info = vs.parse_probe(data)
        assert info["width"] == 1080 and info["height"] == 1920
        assert info["duration"] == 31.5
        assert info["fps"] == pytest.approx(29.97, abs=0.01)
        assert info["has_video"] and info["has_audio"]
        assert vs.streams_check(info)["ok"] is True

    def test_parse_probe_rotation_and_missing_audio(self):
        data = {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                             "height": 1080, "avg_frame_rate": "0/0",
                             "side_data_list": [{"rotation": -90}]}],
                "format": {}}
        info = vs.parse_probe(data)
        assert (info["width"], info["height"]) == (1080, 1920)
        assert info["duration"] is None and info["fps"] is None
        assert vs.streams_check(info)["ok"] is False

    def test_parse_probe_garbage(self):
        info = vs.parse_probe({"streams": "nope", "format": None})
        assert info["has_video"] is False and info["width"] == 0

    def test_lufs(self):
        log = ("[Parsed_ebur128_0 @ x] t: 1.0 M: -20 S: -20 I: -30.0 LUFS\n"
               "Summary:\n  Integrated loudness:\n    I:         -15.2 LUFS\n")
        assert vs.parse_lufs(log) == -15.2
        assert vs.parse_lufs("nothing") is None
        c = vs.loudness_check_from(-15.2)
        assert c["ok"] is True and c["informational"]
        assert vs.loudness_check_from(-23.0)["ok"] is False
        assert vs.loudness_check_from(None)["ok"] is None

    def test_overall_ignores_informational(self):
        checks = [{"name": "a", "ok": True, "detail": ""},
                  {"name": "loud", "ok": False, "detail": "", "informational": True}]
        assert vs.overall_ok(checks) is True
        checks.append({"name": "b", "ok": False, "detail": ""})
        assert vs.overall_ok(checks) is False
        assert vs.overall_ok([{"name": "x", "ok": None, "detail": ""}]) is None


class TestCaptions:
    def test_parse_ass_events_and_tags(self, tmp_path):
        p = tmp_path / "autosubs_1700000000_deadbeef.ass"
        p.write_text("﻿" + _ass([(0.0, 0.5, "{\\c&H00E5FF&}HELLO{\\r} WORLD"),
                                      (0.5, 1.1, "HELLO {\\an5}WORLD, again")]),
                     encoding="utf-8")
        ev = vs.parse_captions(str(p))
        assert len(ev) == 2
        assert ev[0]["start"] == 0.0 and ev[0]["end"] == 0.5
        assert ev[0]["text"] == "HELLO WORLD"
        assert ev[1]["text"] == "HELLO WORLD, again"  # commas in text survive

    def test_parse_srt(self, tmp_path):
        p = tmp_path / "x.srt"
        p.write_text("1\n00:00:00,000 --> 00:00:01,250\nhi there\n\n"
                     "2\n00:00:01,250 --> 00:00:02,000\nagain\n", encoding="utf-8-sig")
        ev = vs.parse_captions(str(p))
        assert [(e["start"], e["end"]) for e in ev] == [(0.0, 1.25), (1.25, 2.0)]

    def test_parse_json(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text(json.dumps({"events": [{"start": 0, "end": 1, "text": "a"}]}))
        assert vs.parse_captions(str(p))[0]["end"] == 1.0
        p.write_text(json.dumps([{"start": "x", "end": 1}]))
        with pytest.raises(ValueError):
            vs.parse_captions(str(p))

    def test_bad_ass_time_raises(self, tmp_path):
        p = tmp_path / "x.ass"
        p.write_text("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
                     "MarginV, Effect, Text\nDialogue: 0,abc,0:00:01.00,Default,,0,0,0,,x\n")
        with pytest.raises(ValueError):
            vs.parse_captions(str(p))

    def test_monotonic_and_bounds(self):
        good = [{"start": 0, "end": 1, "text": ""}, {"start": 1, "end": 2, "text": ""}]
        checks = {c["name"]: c for c in vs.caption_checks(good, duration=2.02)}
        assert checks["captions_monotonic"]["ok"] is True
        assert checks["captions_in_bounds"]["ok"] is True
        bad = [{"start": 1, "end": 2, "text": ""}, {"start": 0.5, "end": 0.4, "text": ""},
               {"start": 2, "end": 5, "text": ""}]
        checks = {c["name"]: c for c in vs.caption_checks(bad, duration=3.0)}
        assert checks["captions_monotonic"]["ok"] is False
        assert checks["captions_in_bounds"]["ok"] is False
        assert vs.caption_checks([], 3.0)[0]["ok"] is False

    def test_alignment_offsets(self):
        words = [{"word": "a", "start": 100.0, "end": 100.4},
                 {"word": "b", "start": 100.5, "end": 100.9},
                 {"word": "c", "start": 101.2, "end": 101.6},
                 {"word": "out", "start": 150.0, "end": 150.5}]
        starts = vs.relative_word_starts(words, 100.0, 110.0)
        assert starts == pytest.approx([0.0, 0.5, 1.2])
        events = [{"start": 0.0, "end": 0.5, "text": ""},
                  {"start": 0.52, "end": 1.2, "text": ""},
                  {"start": 1.2, "end": 1.6, "text": ""}]
        aligned = {c["name"]: c for c in vs.caption_checks(events, 10.0, starts)}
        c = aligned["captions_aligned"]
        assert c["ok"] is True
        assert c["max_offset"] == pytest.approx(0.02)
        assert c["offsets"] == pytest.approx([0.0, 0.02, 0.0])

    def test_alignment_fails_when_shifted(self):
        words = [{"word": w, "start": 10 + i * 1.5, "end": 10 + i * 1.5 + 0.4}
                 for i, w in enumerate("abcdef")]
        starts = vs.relative_word_starts(words, 10.0)
        shifted = [{"start": s + 0.7, "end": s + 1.0, "text": ""} for s in starts]
        c = {x["name"]: x for x in vs.caption_checks(shifted, 20.0, starts)}["captions_aligned"]
        assert c["ok"] is False
        assert c["median_offset"] == pytest.approx(0.7)

    def test_straddling_word_clamps_to_zero(self):
        words = [{"word": "x", "start": 9.8, "end": 10.3}]
        assert vs.relative_word_starts(words, 10.0) == [0.0]

    def test_transcript_words_tolerates_junk(self):
        t = {"segments": [{"words": [{"word": "a", "start": 1, "end": 2},
                                     {"word": "b", "start": None, "end": 2}, "junk"]},
                          "junk", {"no": "words"}]}
        assert vs.transcript_words(t) == [{"word": "a", "start": 1.0, "end": 2.0}]
        assert vs.transcript_words(None) == []


class TestMissingBinaries:
    def test_no_ffprobe_means_not_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs, "ffprobe_bin", lambda: None)
        monkeypatch.setattr(vs, "ffmpeg_bin", lambda: None)
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        cap = tmp_path / "c.ass"
        cap.write_text(_ass([(0.0, 0.5, "A"), (0.5, 1.0, "B")]))
        res = vs.check_short(str(f), captions_path=str(cap),
                             transcript_words=[{"word": "a", "start": 5, "end": 5.4},
                                               {"word": "b", "start": 5.5, "end": 6}],
                             clip_start=5.0)
        by = {c["name"]: c for c in res["checks"]}
        assert by["aspect_9_16"]["ok"] is None
        assert by["aspect_9_16"]["detail"] == "ffprobe not available"
        assert by["decodable"]["ok"] is None
        assert by["captions_parse"]["ok"] is True
        assert by["captions_in_bounds"]["ok"] is None  # duration unknown
        assert by["captions_aligned"]["ok"] is True
        assert res["ok"] is True  # only the caption checks could run and passed

    def test_missing_file(self, tmp_path):
        res = vs.check_short(str(tmp_path / "nope.mp4"))
        assert res["ok"] is False

    def test_env_binary_override(self, monkeypatch, tmp_path):
        fake = tmp_path / "myffprobe"
        fake.write_text("")
        monkeypatch.setenv("FFPROBE_BIN", str(fake))
        assert vs.ffprobe_bin() == str(fake)
        monkeypatch.setenv("FFPROBE_BIN", str(tmp_path / "missing"))
        monkeypatch.setattr(vs.shutil, "which", lambda name: None)
        assert vs.ffprobe_bin() is None


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


@needs_ffmpeg
class TestRealFile:
    def _make(self, path, w, h, seconds):
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=30:duration={seconds}",
                        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(path)], check=True)

    def test_vertical_short_passes(self, tmp_path):
        f = tmp_path / "v.mp4"
        self._make(f, 360, 640, 6)
        res = vs.check_short(str(f))
        by = {c["name"]: c for c in res["checks"]}
        assert by["aspect_9_16"]["ok"] and by["duration"]["ok"]
        assert by["streams"]["ok"] and by["decodable"]["ok"]
        assert "lufs" in by["loudness"]
        assert res["ok"] is True

    def test_horizontal_fails(self, tmp_path):
        f = tmp_path / "h.mp4"
        self._make(f, 640, 360, 6)
        res = vs.check_short(str(f))
        assert res["ok"] is False
