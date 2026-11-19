"""batch_report: global ranking across chunks, packaging of the top N and the
batch report, on fake chunk directories. No ffmpeg, no network, no upload."""
import json
import os

import pytest

import batch_report as br
import shorts_publish


def _w(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data if isinstance(data, str) else json.dumps(data))


def _cand(ci, start, end, overall, pred=None, rank=None, title="t", **extra):
    c = {"candidate_index": ci, "start": start, "end": end,
         "rating": {"overall": overall, "summary": f"sum {ci}",
                    "scores": {"hook_strength": 8, "energy_pacing": 7,
                               "narrative_completeness": 6, "av_clarity": 9,
                               "retention_potential": 5}},
         "predicted_score": pred, "video_title_for_youtube_short": title,
         "video_description_for_youtube_short": "desc", "youtube_hashtags": ["a"],
         "viral_hook_text": "hook"}
    if rank is not None:
        c["rank"] = rank
    c.update(extra)
    return c


def _chunk(out, name, cands, shorts, ok_by_index, cost, packaged=True):
    d = out / name
    meta = {"candidates": cands, "shorts": shorts,
            "cost_analysis": {"total_cost": cost},
            "rating_cost_analysis": {"total_cost": cost / 10}}
    _w(str(d / "Talk_metadata.json"), meta)
    _w(str(d / "verification.json"), {"shorts": [
        {"short_index": i, "file": f"x{i}.mp4",
         "result": {"ok": ok, "checks": [{"name": "duration", "ok": ok is not False,
                                          "detail": "d"}]}}
        for i, ok in ok_by_index.items()]})
    _w(str(d / "stages.json"), {"stages": [{"name": "render", "duration_s": 12.5}]})
    _w(str(d / "fetched.json"), {"fetch_seconds": 30})
    _w(str(d / "batch_done.json"), {"exit_code": 0, "seconds": 600})
    entries = []
    for s in shorts:
        r = s["rank"]
        if packaged:
            p = d / "publish" / f"short_{r}"
            os.makedirs(p, exist_ok=True)
            (p / f"short_{r}.mp4").write_bytes(f"{name}-{r}".encode())
            (p / "title.txt").write_text(s["video_title_for_youtube_short"] + "\n",
                                         encoding="utf-8")
            (p / "description.txt").write_text("pkg desc\n", encoding="utf-8")
            (p / "hashtags.txt").write_text("#pkg #tag\n", encoding="utf-8")
        entries.append({"rank": r, "dir": f"short_{r}", "video": f"short_{r}/short_{r}.mp4"})
    _w(str(d / "publish" / "approval.json"), {"shorts": entries})
    return d


@pytest.fixture
def out(tmp_path):
    _w(str(tmp_path / "fetch_plan.json"), {
        "url": "https://www.youtube.com/watch?v=abcdefghijk", "title": "Stream",
        "step": 5400, "chunks": [
            {"index": 0, "dir": "chunk_00", "stream_start_seconds": 0, "seconds": 5400},
            {"index": 1, "dir": "chunk_01", "stream_start_seconds": 5400, "seconds": 5400},
            {"index": 2, "dir": "chunk_02", "stream_start_seconds": 10800, "seconds": 5400},
        ]})
    # chunk 0: two rendered (one failed verification), one unrendered.
    c0 = [_cand(0, 10, 40, 80, rank=1, title="c0 best"),
          _cand(1, 100, 130, 95, rank=2, title="c0 failed"),
          _cand(2, 200, 230, 50)]
    _chunk(tmp_path, "chunk_00", c0, [c0[0], c0[1]], {0: True, 1: False}, 0.02)
    # chunk 1: ok None counts as passed; hostile strings.
    c1 = [_cand(0, 60, 90, 90, pred=7, rank=1,
                title="x | y\n## injected ../../etc"),
          _cand(1, 300, 330, 80, pred=9, rank=2, title="c1 second")]
    _chunk(tmp_path, "chunk_01", c1, list(c1), {0: None, 1: True}, 0.03)
    # chunk 2: failed, nothing but a done marker.
    os.makedirs(tmp_path / "chunk_02")
    _w(str(tmp_path / "chunk_02" / "batch_done.json"), {"exit_code": 1, "seconds": 5})
    return tmp_path


def test_global_order_and_stream_offsets(out):
    _, r = br.write_report(str(out), top=10)
    order = [(c["chunk_index"], c["candidate_index"]) for c in r["candidates"]]
    # 95, 90, 80 (pred 9, chunk 1), 80 (no pred, chunk 0), 50
    assert order == [(0, 1), (1, 0), (1, 1), (0, 0), (0, 2)]
    first_c1 = r["candidates"][1]
    assert first_c1["stream_start"] == 5460 and first_c1["stream_end"] == 5490
    assert [c["global_rank"] for c in r["candidates"]] == [1, 2, 3, 4, 5]


def test_only_rendered_and_passed_selected(out):
    _, r = br.write_report(str(out), top=10)
    sel = [(s["chunk_index"], s["candidate_index"]) for s in r["selected"]]
    assert sel == [(1, 0), (1, 1), (0, 0)]  # failed and unrendered left out
    assert [s["publish_rank"] for s in r["selected"]] == [1, 2, 3]
    assert any("Only 3" in i for i in r["issues"])
    assert any("chunk 2" in i and "metadata" in i for i in r["issues"])
    assert any("FAILED" in i for i in r["issues"])


def test_top_bound(out):
    _, r = br.write_report(str(out), top=2)
    assert len(r["selected"]) == 2
    assert sorted(os.listdir(out / "publish")) == ["approval.json", "short_1", "short_2"]
    assert not any("Only" in i for i in r["issues"])


def test_package_and_approval_readable_by_shorts_publish(out):
    br.write_report(str(out), top=10)
    pub = out / "publish"
    d = pub / "short_3"
    assert (d / "short_3.mp4").read_bytes() == b"chunk_00-1"
    assert not os.path.islink(d / "short_3.mp4")
    assert (d / "title.txt").read_text(encoding="utf-8").strip() == "c0 best"
    assert (d / "hashtags.txt").read_text(encoding="utf-8").strip() == "#pkg #tag"
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["chunk_index"] == 0 and meta["chunk_rank"] == 1 and meta["rank"] == 3
    assert meta["source"]["url"].startswith("https://www.youtube.com/")
    assert meta["source"]["stream_start_hms"] == "0:00:10"
    assert meta["verification"]["ok"] is True
    m1 = json.loads((pub / "short_1" / "metadata.json").read_text(encoding="utf-8"))
    assert m1["source"]["stream_start_hms"] == "1:31:00"
    appr = json.loads((pub / "approval.json").read_text(encoding="utf-8"))
    for e in appr["shorts"]:
        assert e["status"] == "pending_approval"
        assert e["video"] == f"short_{e['rank']}/short_{e['rank']}.mp4"
        assert {"rank", "dir", "title", "overall", "source_file", "sha256",
                "verification_ok"} <= set(e)
    assert shorts_publish.approve(str(out), [1, 3]) == [(1, "approved"), (3, "approved")]


def test_rerun_keeps_posted_and_never_reopens(out):
    br.write_report(str(out), top=10)
    path = out / "publish" / "approval.json"
    appr = json.loads(path.read_text(encoding="utf-8"))
    appr["shorts"][0].update(status="posted", posted_at="2026-01-01T00:00:00Z",
                             request_id="req1", youtube_url="https://youtu.be/x")
    posted_sha = appr["shorts"][0]["sha256"]
    path.write_text(json.dumps(appr), encoding="utf-8")

    br.write_report(str(out), top=10)
    again = json.loads(path.read_text(encoding="utf-8"))
    e = next(x for x in again["shorts"] if x["sha256"] == posted_sha)
    assert e["status"] == "posted" and e["request_id"] == "req1"
    assert e["youtube_url"] == "https://youtu.be/x"
    assert shorts_publish.approve(str(out), [e["rank"]]) == [(e["rank"], "posted")]

    # Next run with top 1 drops that short from the selection? No: it is the
    # best one. Ask for 0: the posted entry and its dir still stay.
    br.write_report(str(out), top=0)
    last = json.loads(path.read_text(encoding="utf-8"))
    assert [x["status"] for x in last["shorts"]] == ["posted"]
    assert os.path.isfile(out / "publish" / e["dir"] / f"{e['dir']}.mp4")
    assert sorted(os.listdir(out / "publish")) == ["approval.json", e["dir"]]


def test_posted_dir_never_overwritten(out):
    br.write_report(str(out), top=10)
    path = out / "publish" / "approval.json"
    appr = json.loads(path.read_text(encoding="utf-8"))
    # Rank 3 posted; then its chunk short disappears from the selection.
    e3 = next(x for x in appr["shorts"] if x["rank"] == 3)
    e3["status"] = "posted"
    path.write_text(json.dumps(appr), encoding="utf-8")
    ver = out / "chunk_00" / "verification.json"
    v = json.loads(ver.read_text(encoding="utf-8"))
    v["shorts"][0]["result"]["ok"] = False
    ver.write_text(json.dumps(v), encoding="utf-8")
    _, r = br.write_report(str(out), top=10)
    again = json.loads(path.read_text(encoding="utf-8"))
    by_rank = {x["rank"]: x for x in again["shorts"]}
    assert by_rank[3]["status"] == "posted" and by_rank[3]["sha256"] == e3["sha256"]
    assert (out / "publish" / "short_3" / "short_3.mp4").read_bytes() == b"chunk_00-1"


def test_hostile_strings_stay_data(out):
    plan = json.loads((out / "fetch_plan.json").read_text(encoding="utf-8"))
    plan["chunks"].append({"index": 3, "dir": "../evil", "stream_start_seconds": 1})
    (out / "fetch_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    ap = out / "chunk_01" / "publish" / "approval.json"
    a = json.loads(ap.read_text(encoding="utf-8"))
    a["shorts"][0]["video"] = "../../../chunk_00/publish/short_1/short_1.mp4"
    ap.write_text(json.dumps(a), encoding="utf-8")
    md, r = br.write_report(str(out), top=10)
    assert any("not a plain name" in i for i in r["issues"])
    title_row = next(line for line in md.splitlines() if "injected" in line)
    assert "\n## injected" not in md
    assert "x \\| y" in md and title_row.startswith("### ")
    # the hostile video path was ignored: the chunk's own rank-1 file was used
    assert (out / "publish" / "short_1" / "short_1.mp4").read_bytes() == b"chunk_01-1"
    title = (out / "publish" / "short_1" / "title.txt").read_text(encoding="utf-8")
    assert title.strip() == "x | y ## injected ../../etc"
    assert not os.path.exists(out.parent / "evil")


def test_cost_and_timings(out):
    md, r = br.write_report(str(out), top=10)
    assert r["cost"]["per_chunk"] == {"0": pytest.approx(0.022), "1": pytest.approx(0.033)}
    assert r["cost"]["total_usd"] == pytest.approx(0.055)
    ch0 = r["chunks"][0]
    assert ch0["fetch_seconds"] == 30 and ch0["pipeline_seconds"] == 600
    assert ch0["stages"] == [{"name": "render", "duration_s": 12.5}]
    assert r["chunks"][2]["exit_code"] == 1
    assert "Total: $0.0550" in md
    assert os.path.isfile(out / "report.json") and os.path.isfile(out / "report.md")


def test_no_plan_degrades(tmp_path):
    os.makedirs(tmp_path / "chunk_00")
    assert br.main(["--out", str(tmp_path)]) == 0
    r = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert r["selected"] == [] and any("fetch_plan" in i for i in r["issues"])
