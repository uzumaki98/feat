"""Retry behaviour of the Gemini scoring/detail stage.

Prod (22-jul-2026) lost 3 jobs to Gemini answering 200 with an empty body.
That raises while parsing, not while calling, so it used to escape the retry
loop and kill the job on the first blip.
"""
import types
import pytest

# main pulls in cv2/torch/mediapipe at import time; the minimal CI env lacks
# them, so skip there. Runs fully in the container/local where deps exist.
main = pytest.importorskip("main")


class _FakeResponse:
    def __init__(self, parsed=None):
        self.parsed = parsed
        self.candidates = []
        self.usage_metadata = None

    @property
    def text(self):
        return "" if self.parsed is None else "{}"


class _FakeModels:
    """Returns an empty body for the first `blips` calls, then a good one."""

    def __init__(self, blips, payload=None):
        self.blips = blips
        self.calls = 0
        self.payload = payload if payload is not None else {"windows": [{"id": "w0", "score": 90}]}

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.blips:
            return _FakeResponse(parsed=None)
        return _FakeResponse(parsed=self.payload)


def _client(models):
    return types.SimpleNamespace(models=models)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)


def test_recovers_from_a_single_empty_body(monkeypatch):
    models = _FakeModels(blips=1)
    parsed, _cost = main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 2
    assert parsed["windows"][0]["score"] == 90


def test_recovers_from_two_consecutive_blips():
    models = _FakeModels(blips=2)
    parsed, _cost = main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 3
    assert parsed["windows"]


def test_gives_up_after_five_attempts():
    models = _FakeModels(blips=99)
    with pytest.raises(Exception) as exc:
        main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 5
    assert "empty response body" in str(exc.value)


def test_non_transient_errors_are_not_retried():
    class _Boom:
        calls = 0

        def generate_content(self, **kwargs):
            _Boom.calls += 1
            raise ValueError("400 INVALID_ARGUMENT: bad request")

    with pytest.raises(ValueError):
        main._run_gemini_stage(_client(_Boom()), "m", "prompt", object)
    assert _Boom.calls == 1


def test_succeeds_without_retrying_when_the_first_call_is_fine():
    models = _FakeModels(blips=0)
    main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 1


class _BlockedResponse:
    """Mimics the real prod shape: 200, empty candidates, block_reason set."""
    text = ""
    candidates = []
    usage_metadata = None

    class _PF:
        class _Reason:
            name = "PROHIBITED_CONTENT"
        block_reason = _Reason()

    prompt_feedback = _PF()


def test_policy_block_fails_fast_without_retrying():
    # Prod 23-jul-2026: PROHIBITED_CONTENT is deterministic — 3 retries just
    # burned quota and reported a misleading "empty response body".
    class _Models:
        calls = 0

        def generate_content(self, **kwargs):
            _Models.calls += 1
            return _BlockedResponse()

    import gemini_worker
    with pytest.raises(gemini_worker.GeminiBlockedError) as exc:
        main._run_gemini_stage(_client(_Models()), "m", "prompt", object)
    assert _Models.calls == 1
    assert "PROHIBITED_CONTENT" in str(exc.value)


def test_blocked_finish_reason_also_raises():
    import gemini_worker

    class _Cand:
        class _FR:
            name = "SAFETY"
        finish_reason = _FR()
        content = None

    class _Resp:
        prompt_feedback = None
        candidates = [_Cand()]

    with pytest.raises(gemini_worker.GeminiBlockedError):
        gemini_worker.raise_if_blocked(_Resp())


def test_clean_response_is_not_flagged_as_blocked():
    import gemini_worker

    class _Resp:
        prompt_feedback = None
        candidates = []

    gemini_worker.raise_if_blocked(_Resp())  # must not raise


# --- the vision pass (silent footage) -------------------------------------
# 16-nov-2026: a single 503 "high demand" on the one stretch of an 8-minute
# video failed the job after its download and transcription had worked.

class _VisionModels:
    """Raises ``errors`` in order, one per call, then answers ``text``."""

    def __init__(self, errors, text='{"shorts": [{"start": 1, "end": 31}]}'):
        self.errors = list(errors)
        self.text = text
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.errors:
            err = self.errors.pop(0)
            if isinstance(err, Exception):
                raise err
            return types.SimpleNamespace(text=err, candidates=[], usage_metadata=None,
                                         prompt_feedback=None)
        return types.SimpleNamespace(text=self.text, candidates=[], usage_metadata=None,
                                     prompt_feedback=None)


class _VisionFiles:
    def __init__(self):
        self.deleted = 0

    def get(self, name):
        return types.SimpleNamespace(state=types.SimpleNamespace(name="ACTIVE"))

    def delete(self, name):
        self.deleted += 1


def _vision(monkeypatch, models):
    uploads = []
    monkeypatch.setattr(main.gemini_worker, "upload_media",
                        lambda client, path: uploads.append(path) or types.SimpleNamespace(name="files/x"))
    monkeypatch.setattr(main.gemini_worker, "raise_if_blocked", lambda response: None)
    monkeypatch.setattr(main.gemini_worker, "_calculate_cost_analysis", lambda *a: None)
    files = _VisionFiles()
    client = types.SimpleNamespace(models=models, files=files)
    run = lambda: main._visual_pass(client, "m", "part0.mp4", 60.0, "en", 1, 3, 15, 60)
    return run, uploads, files


BUSY = RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': "
                    "'This model is currently experiencing high demand.'}}")


def test_vision_pass_gets_past_a_busy_model(monkeypatch):
    models = _VisionModels([BUSY, BUSY])
    run, uploads, files = _vision(monkeypatch, models)
    shorts, _cost = run()
    assert models.calls == 3
    assert shorts == [{"start": 1.0, "end": 31.0}]
    # The video is uploaded once and removed once, however many tries.
    assert len(uploads) == 1 and files.deleted == 1


def test_vision_pass_retries_an_empty_body(monkeypatch):
    models = _VisionModels([""])
    run, _uploads, _files = _vision(monkeypatch, models)
    shorts, _cost = run()
    assert models.calls == 2 and len(shorts) == 1


def test_vision_pass_gives_up_after_its_tries(monkeypatch):
    models = _VisionModels([BUSY] * 10)
    run, _uploads, files = _vision(monkeypatch, models)
    with pytest.raises(RuntimeError, match="503"):
        run()
    assert models.calls == len(main.VISION_RETRY_WAITS) + 1
    assert files.deleted == 1


def test_vision_pass_does_not_retry_a_request_error(monkeypatch):
    models = _VisionModels([RuntimeError("400 INVALID_ARGUMENT")])
    run, _uploads, _files = _vision(monkeypatch, models)
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        run()
    assert models.calls == 1


def test_vision_pass_never_retries_a_policy_block(monkeypatch):
    models = _VisionModels([])
    run, _uploads, _files = _vision(monkeypatch, models)

    def blocked(response):
        raise main.gemini_worker.GeminiBlockedError("PROHIBITED_CONTENT")
    monkeypatch.setattr(main.gemini_worker, "raise_if_blocked", blocked)
    with pytest.raises(main.gemini_worker.GeminiBlockedError):
        run()
    assert models.calls == 1


def test_rate_limit_waits_out_the_minute():
    gw = main.gemini_worker
    assert gw.is_rate_limited("429 RESOURCE_EXHAUSTED. You exceeded your current quota")
    assert gw.retry_wait("429 RESOURCE_EXHAUSTED", 0) == 20
    assert gw.retry_wait("429 RESOURCE_EXHAUSTED", 5) == 60
    assert gw.retry_wait("429 RESOURCE_EXHAUSTED ... 'retryDelay': '23s'", 0) == 25
    assert gw.retry_wait("503 UNAVAILABLE", 0) == 5
    assert gw.retry_wait("503 UNAVAILABLE", 3) == 40


def test_stage_gets_past_four_busy_answers():
    class Models:
        calls = 0

        def generate_content(self, **kw):
            Models.calls += 1
            if Models.calls <= 4:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return _FakeResponse(parsed={"windows": []})
    parsed, _ = main._run_gemini_stage(_client(Models()), "m", "p", object)
    assert Models.calls == 5 and parsed == {"windows": []}
