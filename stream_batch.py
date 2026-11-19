"""Run shorts_pipeline.py on each chunk that stream_fetch.py finishes, as it lands.

    python stream_batch.py <url> --out output/<dir> [--top-per-chunk 4] [--jobs 3]
                           [--clip-workers 2] [--poll 30] [--stall-minutes 30]
                           [--final-top 10] [--then-report]

Reads what stream_fetch.py writes into ``<out>``: ``fetch_plan.json`` (every
chunk), ``chunk_NN/fetched.json`` (that chunk's ``source.mp4`` is complete) and
``fetch.log`` (``all chunks fetched`` at the end). Each fetched chunk gets one
``shorts_pipeline.py --input chunk_NN/source.mp4 --out chunk_NN`` child, at most
``--jobs`` at once, in chunk order, with ``ASR_HOST_SLOTS=<jobs>`` so every
child may transcribe at the same time.

Markers in each chunk dir: ``batch_running.json`` while a child runs (batch pid
and child pid; one left by a dead earlier run is started again),
``batch_done.json`` when it ends (exit code, attempts, times), and the child's
output in ``batch_child.log``. A chunk is final once it exited 0 or failed
``MAX_ATTEMPTS`` times; a failed chunk is retried once, then left failed and the
batch goes on. Re-running skips final chunks.

The batch ends when every chunk in the plan is final and the fetch log has its
last line, or, if the fetcher died (no new ``fetched.json`` for
``--stall-minutes`` and no last line), with what was fetched. Events go to
``<out>/batch.log`` and stdout; ``<out>/batch_state.json`` holds per-chunk
status. ``--then-report`` then runs ``batch_report.py --out <out> --top N``.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_ATTEMPTS = 2
FETCH_DONE_LINE = "all chunks fetched"


# --------------------------------------------------------------------------- io

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _stamp(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else time.time()))


def pid_alive(pid):
    """True if a process with this pid exists (never signals it)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) terminates the process on Windows: ask the kernel instead.
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            k32.CloseHandle(h)
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ------------------------------------------------------------- pure-ish logic

def read_plan(out):
    """The plan's chunks (sorted by index), or None while there is no readable plan."""
    plan = _read_json(os.path.join(out, "fetch_plan.json"))
    if not isinstance(plan, dict) or not isinstance(plan.get("chunks"), list):
        return None
    chunks = []
    for c in plan["chunks"]:
        try:
            idx = int(c["index"])
        except (KeyError, TypeError, ValueError):
            continue
        d = str(c.get("dir") or f"chunk_{idx:02d}")
        if os.path.basename(d) != d or d in ("", ".", ".."):
            continue  # plain dir names only
        chunks.append({"index": idx, "dir": d,
                       "seconds": float(c.get("seconds") or 0),
                       "stream_start_seconds": float(c.get("stream_start_seconds") or 0)})
    return sorted(chunks, key=lambda c: c["index"])


def fetch_complete(out):
    try:
        with open(os.path.join(out, "fetch.log"), encoding="utf-8", errors="replace") as f:
            return any(line.rstrip().endswith(FETCH_DONE_LINE) for line in f)
    except OSError:
        return False


def is_fetched(out, chunk):
    return os.path.exists(os.path.join(out, chunk["dir"], "fetched.json"))


def done_info(out, chunk):
    return _read_json(os.path.join(out, chunk["dir"], "batch_done.json"))


def is_final(done):
    """A batch_done.json that needs no further attempt: exit 0, a run that
    produced its package (shorts_pipeline exits 1 when one short fails a check,
    and repeating ~45 min of transcription would not change that), or the
    attempts used up."""
    if not isinstance(done, dict):
        return False
    return (done.get("exit_code") == 0 or bool(done.get("completed"))
            or int(done.get("attempts") or 1) >= MAX_ATTEMPTS)


def produced(out, chunk, since=None):
    """True if the chunk's pipeline wrote its package (publish/approval.json
    with at least one short), and, with ``since`` (epoch s), wrote it after."""
    path = os.path.join(out, chunk["dir"], "publish", "approval.json")
    data = _read_json(path)
    if not (isinstance(data, dict) and isinstance(data.get("shorts"), list) and data["shorts"]):
        return False
    try:
        return since is None or os.path.getmtime(path) >= since
    except OSError:
        return False


def adopt_orphan(out, chunk, alive=pid_alive, clock=time.time):
    """A child left by an earlier batch run that has since ended: if it
    produced its package after it started, record it done instead of running
    the chunk again. Returns True when adopted."""
    d = os.path.join(out, chunk["dir"])
    info = _read_json(os.path.join(d, "batch_running.json"))
    if not isinstance(info, dict) or alive(info.get("child_pid")):
        return False
    try:
        started = time.mktime(time.strptime(str(info.get("started_at")), "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return False
    if not produced(out, chunk, since=started):
        return False
    _write_json(os.path.join(d, "batch_done.json"),
                {"exit_code": None, "completed": True, "adopted": True,
                 "attempts": info.get("attempt") or 1,
                 "started_at": info.get("started_at"), "finished_at": _stamp(clock())})
    try:
        os.remove(os.path.join(d, "batch_running.json"))
    except OSError:
        pass
    return True


def held_elsewhere(out, chunk, alive=pid_alive):
    """True if a batch_running.json from another run names a child still alive.
    (Container pids repeat across restarts, so our own pid never counts.)"""
    info = _read_json(os.path.join(out, chunk["dir"], "batch_running.json"))
    if not isinstance(info, dict):
        return False
    pid = info.get("child_pid")
    return pid != os.getpid() and alive(pid)


def ready_chunks(out, plan, running=(), alive=pid_alive):
    """Fetched chunks that still need a child, in chunk order."""
    ready = []
    for c in plan or []:
        if c["index"] in running or not is_fetched(out, c):
            continue
        if is_final(done_info(out, c)):
            continue
        if adopt_orphan(out, c, alive):
            continue
        if held_elsewhere(out, c, alive):
            continue
        ready.append(c)
    return ready


def to_start(ready, running_count, jobs):
    """The first chunks to start now, keeping at most `jobs` running."""
    return list(ready[:max(0, jobs - running_count)])


def minutes_for(seconds):
    return int(math.ceil(float(seconds) / 60.0)) + 1


def child_command(url, out, chunk, top, clip_workers=None, python=None):
    d = os.path.join(os.path.abspath(out), chunk["dir"])
    cmd = [python or sys.executable, os.path.join(REPO_DIR, "shorts_pipeline.py"), url,
           "--input", os.path.join(d, "source.mp4"),
           "--minutes", str(minutes_for(chunk["seconds"])),
           "--top", str(int(top)), "--out", d]
    if clip_workers:
        cmd += ["--clip-workers", str(int(clip_workers))]
    return cmd


def child_env(jobs, base=None):
    env = dict(os.environ if base is None else base)
    env["ASR_HOST_SLOTS"] = str(int(jobs))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def is_stalled(fetch_done, last_progress, now, stall_minutes):
    return (not fetch_done) and (now - last_progress) > stall_minutes * 60


def batch_finished(out, plan, running_count, fetch_done, stalled):
    """(finished, reason). Finished only with no child running."""
    if running_count:
        return False, None
    if plan is None:
        return (True, "stalled: no fetch plan") if stalled else (False, None)
    if fetch_done:
        if all(is_final(done_info(out, c)) for c in plan):
            return True, "all chunks processed"
        return False, None
    if stalled:
        fetched = [c for c in plan if is_fetched(out, c)]
        if all(is_final(done_info(out, c)) for c in fetched):
            return True, (f"fetcher stalled: finishing with {len(fetched)}/{len(plan)} "
                          f"fetched chunk(s)")
    return False, None


# ---------------------------------------------------------------------- batch

class Batch:
    def __init__(self, url, out, top_per_chunk=4, jobs=3, clip_workers=2, poll=30.0,
                 stall_minutes=30.0, final_top=10, then_report=False,
                 popen=subprocess.Popen, clock=time.time, sleep=time.sleep,
                 alive=pid_alive, python=None, quiet=False):
        self.url, self.out = url, os.path.abspath(out)
        self.top, self.jobs = top_per_chunk, max(1, int(jobs))
        self.clip_workers, self.poll = clip_workers, poll
        self.stall_minutes, self.final_top = stall_minutes, final_top
        self.then_report = then_report
        self.popen, self.clock, self.sleep, self.alive = popen, clock, sleep, alive
        self.python = python or sys.executable
        self.quiet = quiet
        self.running = {}   # index -> dict(proc, chunk, started, log, attempts)
        self.state = {}     # index -> per-chunk state
        self.last_progress = clock()
        self.fetched_seen = -1
        self.reason = None
        os.makedirs(self.out, exist_ok=True)

    def log(self, line):
        stamp = _stamp(self.clock())
        if not self.quiet:
            print(f"[{stamp}] {line}", flush=True)
        with open(os.path.join(self.out, "batch.log"), "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {line}\n")

    def save_state(self, plan):
        chunks = {}
        for c in plan or []:
            st = dict(self.state.get(c["index"], {}))
            if not st:
                done = done_info(self.out, c)
                if is_final(done):
                    st = {"status": "done" if (done.get("exit_code") == 0
                                               or done.get("completed")) else "failed",
                          "exit_code": done.get("exit_code"),
                          "attempts": done.get("attempts"), "seconds": done.get("seconds")}
                else:
                    st = {"status": "queued" if is_fetched(self.out, c) else "waiting_fetch"}
            st["dir"] = c["dir"]
            chunks[str(c["index"])] = st
        _write_json(os.path.join(self.out, "batch_state.json"),
                    {"updated_at": _stamp(self.clock()), "jobs": self.jobs,
                     "finished": self.reason, "chunks": chunks})

    def start(self, chunk):
        d = os.path.join(self.out, chunk["dir"])
        prev = done_info(self.out, chunk)
        attempts = (int(prev.get("attempts") or 1) if isinstance(prev, dict) else 0) + 1
        cmd = child_command(self.url, self.out, chunk, self.top, self.clip_workers, self.python)
        logf = open(os.path.join(d, "batch_child.log"), "a", encoding="utf-8")
        logf.write(f"\n===== attempt {attempts} at {_stamp(self.clock())}: "
                   f"{' '.join(cmd)}\n")
        logf.flush()
        try:
            proc = self.popen(cmd, cwd=REPO_DIR, env=child_env(self.jobs),
                              stdout=logf, stderr=subprocess.STDOUT)
        except Exception as e:
            logf.close()
            self.log(f"chunk {chunk['index']}: could not start: {e}")
            self.finish(chunk, {"proc": None, "started": self.clock(), "log": None,
                                "attempts": attempts}, 127)
            return
        started = self.clock()
        _write_json(os.path.join(d, "batch_running.json"),
                    {"batch_pid": os.getpid(), "child_pid": getattr(proc, "pid", None),
                     "attempt": attempts, "started_at": _stamp(started)})
        self.running[chunk["index"]] = {"proc": proc, "chunk": chunk, "started": started,
                                        "log": logf, "attempts": attempts}
        self.state[chunk["index"]] = {"status": "running", "attempts": attempts,
                                      "started_at": _stamp(started)}
        self.log(f"chunk {chunk['index']}: started (attempt {attempts}, "
                 f"{minutes_for(chunk['seconds'])} min, pid {getattr(proc, 'pid', '?')})")

    def finish(self, chunk, run, code):
        d = os.path.join(self.out, chunk["dir"])
        now = self.clock()
        secs = round(now - run["started"], 1)
        completed = code == 0 or (code == 1 and produced(self.out, chunk, since=run["started"]))
        _write_json(os.path.join(d, "batch_done.json"),
                    {"exit_code": code, "completed": completed, "attempts": run["attempts"],
                     "started_at": _stamp(run["started"]), "finished_at": _stamp(now),
                     "seconds": secs})
        try:
            os.remove(os.path.join(d, "batch_running.json"))
        except OSError:
            pass
        if run.get("log"):
            run["log"].close()
        final = completed or run["attempts"] >= MAX_ATTEMPTS
        status = "done" if completed else ("failed" if final else "retry")
        self.state[chunk["index"]] = {"status": status if status != "retry" else "queued",
                                      "exit_code": code, "attempts": run["attempts"],
                                      "seconds": secs}
        if code == 0:
            self.log(f"chunk {chunk['index']}: done in {secs:.0f}s")
        elif completed:
            self.log(f"chunk {chunk['index']}: done in {secs:.0f}s with problems "
                     f"(exit {code}; see its report)")
        elif final:
            self.log(f"chunk {chunk['index']}: FAILED (exit {code}) after "
                     f"{run['attempts']} attempt(s); going on")
        else:
            self.log(f"chunk {chunk['index']}: exit {code} after {secs:.0f}s; will retry")

    def reap(self):
        for idx in list(self.running):
            run = self.running[idx]
            code = run["proc"].poll()
            if code is None:
                continue
            del self.running[idx]
            self.finish(run["chunk"], run, code)

    def step(self):
        """One scheduling pass. Returns True when the batch is finished."""
        self.reap()
        plan = read_plan(self.out)
        fdone = fetch_complete(self.out)
        if plan is not None:
            n = sum(1 for c in plan if is_fetched(self.out, c))
            if n != self.fetched_seen:
                if n > self.fetched_seen >= 0:
                    self.log(f"{n}/{len(plan)} chunk(s) fetched")
                self.fetched_seen = n
                self.last_progress = self.clock()
        for c in to_start(ready_chunks(self.out, plan, self.running, self.alive),
                          len(self.running), self.jobs):
            self.start(c)
        stalled = is_stalled(fdone, self.last_progress, self.clock(), self.stall_minutes)
        finished, reason = batch_finished(self.out, plan, len(self.running), fdone, stalled)
        if finished:
            self.reason = reason
        self.save_state(plan)
        return finished

    def report(self):
        cmd = [self.python, os.path.join(REPO_DIR, "batch_report.py"),
               "--out", self.out, "--top", str(int(self.final_top))]
        self.log("running batch_report.py")
        with open(os.path.join(self.out, "batch_report.log"), "a", encoding="utf-8") as logf:
            try:
                proc = self.popen(cmd, cwd=REPO_DIR, env=dict(os.environ),
                                  stdout=logf, stderr=subprocess.STDOUT)
                code = proc.wait()
            except Exception as e:
                self.log(f"batch_report.py could not run: {e}")
                return 127
        self.log(f"batch_report.py exit {code}")
        return code

    def run(self):
        self.log(f"batch start: {self.out}, jobs {self.jobs}, top {self.top}/chunk, "
                 f"clip workers {self.clip_workers}")
        while not self.step():
            self.sleep(self.poll)
        self.log(f"batch finished: {self.reason}")
        plan = read_plan(self.out) or []
        failed = [c["index"] for c in plan
                  if isinstance(done_info(self.out, c), dict)
                  and done_info(self.out, c).get("exit_code") != 0]
        missing = [c["index"] for c in plan if not is_fetched(self.out, c)]
        if failed:
            self.log(f"chunks with problems: {failed}")
        if missing:
            self.log(f"chunks never fetched: {missing}")
        code = 1 if (failed or missing) else 0
        if self.then_report:
            if self.report() != 0:
                code = 1
        return code


def main(argv=None, **kw):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("url")
    p.add_argument("--out", required=True)
    p.add_argument("--top-per-chunk", type=int, default=4)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--clip-workers", type=int, default=2)
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--stall-minutes", type=float, default=30.0)
    p.add_argument("--final-top", type=int, default=10)
    p.add_argument("--then-report", action="store_true")
    a = p.parse_args(argv)
    return Batch(a.url, a.out, a.top_per_chunk, a.jobs, a.clip_workers, a.poll,
                 a.stall_minutes, a.final_top, a.then_report, **kw).run()


if __name__ == "__main__":
    sys.exit(main())
