"""stream_batch: chunk scheduling with fake children (no network, ffmpeg or real processes)."""
import json
import os

import stream_batch as sb


def make_plan(out, n, seconds=5400.0):
    os.makedirs(out, exist_ok=True)
    chunks = [{"index": i, "dir": f"chunk_{i:02d}", "first_sq": 0, "end_sq": 1,
               "stream_start_seconds": (i - 1) * seconds, "seconds": seconds}
              for i in range(1, n + 1)]
    with open(os.path.join(out, "fetch_plan.json"), "w", encoding="utf-8") as f:
        json.dump({"url": "u", "title": "t", "step": 2.0, "chunks": chunks}, f)
    for c in chunks:
        os.makedirs(os.path.join(out, c["dir"]), exist_ok=True)


def fetch(out, i):
    with open(os.path.join(out, f"chunk_{i:02d}", "fetched.json"), "w") as f:
        json.dump({"index": i}, f)


def fetch_all_done(out):
    with open(os.path.join(out, "fetch.log"), "a", encoding="utf-8") as f:
        f.write("[00:00:00] all chunks fetched\n")


def write_done(out, i, code, attempts=1):
    with open(os.path.join(out, f"chunk_{i:02d}", "batch_done.json"), "w") as f:
        json.dump({"exit_code": code, "attempts": attempts}, f)


class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.code = None

    def poll(self):
        return self.code

    def wait(self):
        return self.code if self.code is not None else 0


class FakePopen:
    def __init__(self, codes=None):
        self.calls = []
        self.procs = []
        self.codes = codes or {}  # chunk dir name -> list of exit codes per attempt

    def __call__(self, cmd, cwd=None, env=None, stdout=None, stderr=None):
        p = FakeProc(1000 + len(self.calls))
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        self.procs.append(p)
        return p

    def chunk_of(self, proc):
        cmd = self.calls[self.procs.index(proc)]["cmd"]
        return os.path.basename(cmd[cmd.index("--out") + 1])

    def finish_all(self):
        for p in self.procs:
            if p.code is None:
                seq = self.codes.get(self.chunk_of(p), [0])
                p.code = seq.pop(0) if len(seq) > 1 else seq[0]


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def batch(out, popen, clock=None, **kw):
    kw.setdefault("jobs", 2)
    return sb.Batch("https://youtu.be/x", str(out), popen=popen, clock=clock or Clock(),
                    sleep=lambda s: None, alive=lambda pid: False, quiet=True, **kw)


def started_dirs(popen):
    return [os.path.basename(c["cmd"][c["cmd"].index("--out") + 1]) for c in popen.calls]


def test_ready_chunk_detection(tmp_path):
    out = str(tmp_path)
    make_plan(out, 3)
    fetch(out, 1)
    fetch(out, 3)
    write_done(out, 3, 0)
    plan = sb.read_plan(out)
    assert [c["index"] for c in sb.ready_chunks(out, plan, alive=lambda p: False)] == [1]
    assert sb.read_plan(str(tmp_path / "nope")) is None


def test_respects_jobs_and_chunk_order(tmp_path):
    out = str(tmp_path)
    make_plan(out, 5)
    for i in (5, 3, 1, 2, 4):
        fetch(out, i)
    fp = FakePopen()
    b = batch(out, fp, jobs=2)
    assert b.step() is False
    assert started_dirs(fp) == ["chunk_01", "chunk_02"]
    assert b.step() is False
    assert len(fp.calls) == 2  # nothing finished, nothing more started
    fp.procs[0].code = 0
    b.step()
    assert started_dirs(fp) == ["chunk_01", "chunk_02", "chunk_03"]
    assert sb.to_start([1, 2, 3], 1, 3) == [1, 2]
    assert sb.to_start([1, 2], 3, 3) == []


def test_skips_done_chunks(tmp_path):
    out = str(tmp_path)
    make_plan(out, 2)
    fetch(out, 1)
    fetch(out, 2)
    write_done(out, 1, 0)
    write_done(out, 2, 1, attempts=2)  # failed twice: final
    fetch_all_done(out)
    fp = FakePopen()
    b = batch(out, fp)
    assert b.step() is True
    assert fp.calls == []
    state = json.load(open(os.path.join(out, "batch_state.json")))
    assert state["chunks"]["1"]["status"] == "done"
    assert state["chunks"]["2"]["status"] == "failed"


def test_retries_failed_chunk_once(tmp_path):
    out = str(tmp_path)
    make_plan(out, 2)
    fetch(out, 1)
    fetch(out, 2)
    fetch_all_done(out)
    fp = FakePopen(codes={"chunk_01": [1, 1], "chunk_02": [0]})
    b = batch(out, fp, jobs=2)
    for _ in range(10):
        fp.finish_all()
        if b.step():
            break
    assert started_dirs(fp) == ["chunk_01", "chunk_02", "chunk_01"]
    done = json.load(open(os.path.join(out, "chunk_01", "batch_done.json")))
    assert done["exit_code"] == 1 and done["attempts"] == 2
    assert {"started_at", "finished_at", "seconds"} <= set(done)
    assert not os.path.exists(os.path.join(out, "chunk_01", "batch_running.json"))
    state = json.load(open(os.path.join(out, "batch_state.json")))
    assert state["chunks"]["1"]["status"] == "failed"
    assert state["chunks"]["1"]["attempts"] == 2
    assert state["chunks"]["2"]["status"] == "done"


def test_finishes_only_after_final_fetch_line(tmp_path):
    out = str(tmp_path)
    make_plan(out, 2)
    fetch(out, 1)
    fp = FakePopen()
    b = batch(out, fp)
    b.step()
    fp.finish_all()
    assert b.step() is False  # chunk 2 not fetched yet
    fetch(out, 2)
    assert b.step() is False  # chunk 2 now running
    fp.finish_all()
    assert b.step() is False  # all processed, but the fetcher has not said it is done
    fetch_all_done(out)
    assert b.step() is True
    assert b.reason == "all chunks processed"


def test_stall_detection(tmp_path):
    out = str(tmp_path)
    make_plan(out, 3)
    fetch(out, 1)
    clock = Clock()
    fp = FakePopen()
    b = batch(out, fp, clock=clock, stall_minutes=30)
    b.step()
    fp.finish_all()
    clock.t += 29 * 60
    assert b.step() is False
    clock.t += 2 * 60
    assert b.step() is True
    assert "stalled" in b.reason
    assert sb.is_stalled(False, 0, 31 * 60, 30) is True
    assert sb.is_stalled(True, 0, 999 * 60, 30) is False
    assert b.run() == 1  # chunks 2-3 never fetched
    assert "never fetched" in open(os.path.join(out, "batch.log"), encoding="utf-8").read()


def test_stall_waits_for_running_child(tmp_path):
    out = str(tmp_path)
    make_plan(out, 2)
    fetch(out, 1)
    clock = Clock()
    fp = FakePopen()
    b = batch(out, fp, clock=clock, stall_minutes=1)
    b.step()
    clock.t += 3600
    assert b.step() is False  # stalled, but chunk 1 still running
    fp.finish_all()
    assert b.step() is True


def test_resumes_after_dead_running_marker(tmp_path):
    out = str(tmp_path)
    make_plan(out, 2)
    fetch(out, 1)
    fetch(out, 2)
    for i, pid in ((1, 111), (2, 222)):
        with open(os.path.join(out, f"chunk_{i:02d}", "batch_running.json"), "w") as f:
            json.dump({"batch_pid": 9, "child_pid": pid}, f)
    fp = FakePopen()
    b = sb.Batch("u", out, jobs=2, popen=fp, clock=Clock(), sleep=lambda s: None,
                 alive=lambda pid: pid == 222, quiet=True)
    b.step()
    assert started_dirs(fp) == ["chunk_01"]  # 222 still alive: left alone
    running = json.load(open(os.path.join(out, "chunk_01", "batch_running.json")))
    assert running["child_pid"] == fp.procs[0].pid


def test_child_command_and_env(tmp_path):
    out = str(tmp_path)
    make_plan(out, 1, seconds=5401.0)
    fetch(out, 1)
    fp = FakePopen()
    b = batch(out, fp, jobs=3, top_per_chunk=4, clip_workers=2)
    b.step()
    call = fp.calls[0]
    cmd = call["cmd"]
    chunk_dir = os.path.join(os.path.abspath(out), "chunk_01")
    assert cmd[0] == sb.sys.executable
    assert cmd[1].endswith("shorts_pipeline.py") and cmd[2] == "https://youtu.be/x"
    assert cmd[cmd.index("--input") + 1] == os.path.join(chunk_dir, "source.mp4")
    assert cmd[cmd.index("--minutes") + 1] == "92"  # ceil(5401/60) + 1
    assert cmd[cmd.index("--top") + 1] == "4"
    assert cmd[cmd.index("--out") + 1] == chunk_dir
    assert cmd[cmd.index("--clip-workers") + 1] == "2"
    assert "--stream" in cmd  # every chunk is a piece of a live stream
    assert call["env"]["ASR_HOST_SLOTS"] == "3"
    assert call["cwd"] == sb.REPO_DIR
    assert os.path.exists(os.path.join(chunk_dir, "batch_child.log"))
    assert sb.minutes_for(5400) == 91


def test_then_report_runs_batch_report(tmp_path):
    out = str(tmp_path)
    make_plan(out, 1)
    fetch(out, 1)
    fetch_all_done(out)
    fp = FakePopen()
    b = batch(out, fp, then_report=True, final_top=7)
    orig = b.step

    def step():
        fp.finish_all()
        return orig()
    b.step = step
    assert b.run() == 0
    cmd = fp.calls[-1]["cmd"]
    assert cmd[1].endswith("batch_report.py")
    assert cmd[cmd.index("--top") + 1] == "7"
    assert "batch_report.py exit 0" in open(os.path.join(out, "batch.log"),
                                            encoding="utf-8").read()


def test_pid_alive_self_and_bogus():
    assert sb.pid_alive(os.getpid()) is True
    assert sb.pid_alive(None) is False
    assert sb.pid_alive(-5) is False


def test_exit_1_with_a_package_is_done_not_retried(tmp_path):
    # shorts_pipeline exits 1 when one short fails a check: the chunk produced
    # its package, so it is final (a retry would only repeat the transcription).
    import json as _json
    import stream_batch as sb
    d = tmp_path / "chunk_01" / "publish"
    d.mkdir(parents=True)
    (d / "approval.json").write_text(_json.dumps({"shorts": [{"rank": 1}]}))
    chunk = {"index": 1, "dir": "chunk_01", "seconds": 60}
    assert sb.produced(str(tmp_path), chunk)
    assert sb.is_final({"exit_code": 1, "completed": True, "attempts": 1})
    assert not sb.is_final({"exit_code": 1, "attempts": 1})


def test_finished_orphan_is_adopted(tmp_path):
    import json as _json
    import time as _time
    import stream_batch as sb
    c = tmp_path / "chunk_02"
    (c / "publish").mkdir(parents=True)
    (c / "fetched.json").write_text("{}")
    started = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(_time.time() - 60))
    (c / "batch_running.json").write_text(_json.dumps(
        {"child_pid": 999999, "attempt": 1, "started_at": started}))
    (c / "publish" / "approval.json").write_text(_json.dumps({"shorts": [{"rank": 1}]}))
    chunk = {"index": 2, "dir": "chunk_02", "seconds": 60}
    assert sb.ready_chunks(str(tmp_path), [chunk], alive=lambda pid: False) == []
    done = _json.loads((c / "batch_done.json").read_text())
    assert done["completed"] and done["adopted"]
    assert not (c / "batch_running.json").exists()
