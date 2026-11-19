import json
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import clip_rating  # noqa: E402
from gemini_worker import GeminiBlockedError  # noqa: E402


def _payload(score=7, summary="Strong opener and clean payoff."):
    data = {c: {"score": score, "rationale": f"{c} looks fine"} for c in clip_rating.CRITERIA}
    data["summary"] = summary
    return data


def _response(data=None, text=None, blocked=None):
    usage = SimpleNamespace(prompt_token_count=1000, candidates_token_count=100,
                            thoughts_token_count=0)
    pf = SimpleNamespace(block_reason=blocked) if blocked else None
    return SimpleNamespace(text=text if text is not None else json.dumps(data),
                           usage_metadata=usage, prompt_feedback=pf, candidates=[])


class FakeModels:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.lock = threading.Lock()

    def generate_content(self, model, contents, config):
        with self.lock:
            self.calls.append({"model": model, "contents": contents, "config": config})
        prompt = contents[-1]
        return self.handler(prompt, contents)


class FakeClient:
    def __init__(self, handler):
        self.models = FakeModels(handler)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(clip_rating.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def video(tmp_path):
    import cv2

    path = str(tmp_path / "src.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720))
    assert w.isOpened()
    for i in range(100):  # 10 s
        frame = np.full((720, 1280, 3), (i * 2) % 255, dtype=np.uint8)
        w.write(frame)
    w.release()
    return path


def _transcript():
    return {"language": "en", "segments": [
        {"start": 0.0, "end": 4.0, "text": "hello there friends",
         "words": [{"word": " hello", "start": 0.0, "end": 0.5},
                   {"word": " there", "start": 1.0, "end": 1.5},
                   {"word": " friends", "start": 3.0, "end": 3.5}]},
        {"start": 5.0, "end": 9.0, "text": "the big reveal",
         "words": [{"word": " the", "start": 5.0, "end": 5.2},
                   {"word": " big", "start": 6.0, "end": 6.3},
                   {"word": " reveal", "start": 8.0, "end": 8.5}]},
    ]}


def _shorts():
    return [
        {"start": 0.0, "end": 4.0, "predicted_score": 60,
         "video_title_for_youtube_short": "Hi", "viral_hook_text": "Wait for it"},
        {"start": 5.0, "end": 9.0, "predicted_score": 80,
         "video_title_for_youtube_short": "Reveal", "viral_hook_text": "The reveal"},
        {"start": 2.0, "end": 7.0, "predicted_score": 70,
         "video_title_for_youtube_short": "Mid", "viral_hook_text": "Middle"},
    ]


def test_weights_sum_to_one():
    assert set(clip_rating.WEIGHTS) == set(clip_rating.CRITERIA)
    assert abs(sum(clip_rating.WEIGHTS.values()) - 1.0) < 1e-9
    top = max(clip_rating.WEIGHTS.values())
    assert clip_rating.WEIGHTS["hook_strength"] == top
    assert clip_rating.WEIGHTS["retention_potential"] == top


def test_ratings_set_in_place_with_right_shape(video):
    client = FakeClient(lambda prompt, contents: _response(_payload(8)))
    shorts = _shorts()
    costs = clip_rating.rate_candidates(shorts, _transcript(), video, client=client,
                                        model_name="m-test", frames_per_clip=3)
    assert len(costs) == 3
    assert all(c["model"] == "m-test" and c["input_tokens"] == 1000 for c in costs)
    for s in shorts:
        r = s["rating"]
        assert set(r) == {"scores", "rationale", "summary", "overall", "model"}
        assert set(r["scores"]) == set(clip_rating.CRITERIA)
        assert all(isinstance(v, int) and v == 8 for v in r["scores"].values())
        assert set(r["rationale"]) == set(clip_rating.CRITERIA)
        assert r["overall"] == 80.0
        assert r["model"] == "m-test"
        assert r["summary"]
    # frames sent (3 images + prompt), never the video; images <= 768 px
    import cv2
    for call in client.models.calls:
        assert len(call["contents"]) == 4
        for part in call["contents"][:3]:
            data = part.inline_data.data
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert max(img.shape[:2]) <= 768
    # transcript restricted to the clip, relative timestamps
    prompts = [c["contents"][-1] for c in client.models.calls]
    reveal = next(p for p in prompts if "Reveal" in p)
    assert "reveal" in reveal and "hello" not in reveal
    assert "[0.0s] the big reveal" in reveal


def test_silent_footage_says_no_speech_and_runs_without_frames(tmp_path):
    client = FakeClient(lambda prompt, contents: _response(_payload(5)))
    shorts = [{"start": 0, "end": 10, "predicted_score": 50}]
    clip_rating.rate_candidates(shorts, {"language": "none", "segments": []},
                                str(tmp_path / "missing.mp4"), client=client, model_name="m")
    assert shorts[0]["rating"]["overall"] == 50.0
    contents = client.models.calls[0]["contents"]
    assert len(contents) == 1  # zero frames: prompt only
    assert "(no speech)" in contents[0]


def test_overall_score_weighting_and_clamping():
    perfect = {c: 10 for c in clip_rating.CRITERIA}
    assert clip_rating.overall_score(perfect) == 100.0
    assert clip_rating.overall_score({}) == 0.0
    only_hook = {"hook_strength": 10}
    assert clip_rating.overall_score(only_hook) == round(clip_rating.WEIGHTS["hook_strength"] * 100, 1)
    wild = {c: 99 for c in clip_rating.CRITERIA}
    wild["av_clarity"] = -5
    expected = round(100 - clip_rating.WEIGHTS["av_clarity"] * 100, 1)
    assert clip_rating.overall_score(wild) == expected
    assert clip_rating.overall_score({"hook_strength": "junk", "energy_pacing": None}) == 0.0


def test_model_output_is_clamped_and_truncated(video):
    data = _payload(7)
    data["hook_strength"] = {"score": 42, "rationale": "word " * 100}
    data["av_clarity"] = {"score": -3, "rationale": "x"}
    data["energy_pacing"] = "nonsense"
    data["summary"] = "s " * 200
    client = FakeClient(lambda prompt, contents: _response(data))
    shorts = [_shorts()[0]]
    clip_rating.rate_candidates(shorts, _transcript(), video, client=client, model_name="m")
    r = shorts[0]["rating"]
    assert r["scores"]["hook_strength"] == 10
    assert r["scores"]["av_clarity"] == 0
    assert r["scores"]["energy_pacing"] == 0
    assert len(r["rationale"]["hook_strength"].split()) <= clip_rating.RATIONALE_MAX_WORDS
    assert len(r["summary"].split()) <= clip_rating.SUMMARY_MAX_WORDS
    assert r["overall"] == clip_rating.overall_score(r["scores"])


def test_select_top_ordering_fallback_and_stable_ties():
    shorts = [
        {"predicted_score": 90},                                   # unrated -> 90
        {"predicted_score": 10, "rating": {"overall": 75.0}},      # 75
        {"predicted_score": 99, "rating": {"overall": 95.5}},      # 95.5
        {"predicted_score": 75},                                   # unrated -> 75, ties with #1
        {"predicted_score": 50, "rating": None},                   # 50
    ]
    assert clip_rating.select_top(shorts, 10) == [2, 0, 1, 3, 4]
    assert clip_rating.select_top(shorts, 2) == [2, 0]
    assert clip_rating.select_top(shorts, 0) == []
    assert clip_rating.select_top([], 3) == []


def test_one_candidate_exception_leaves_only_that_rating_none(video):
    def handler(prompt, contents):
        if "Reveal" in prompt:
            raise ValueError("400 INVALID_ARGUMENT something broke")
        return _response(_payload(6))

    client = FakeClient(handler)
    shorts = _shorts()
    costs = clip_rating.rate_candidates(shorts, _transcript(), video, client=client, model_name="m")
    assert shorts[1]["rating"] is None
    assert shorts[0]["rating"]["overall"] == 60.0
    assert shorts[2]["rating"]["overall"] == 60.0
    assert len(costs) == 2


def test_transient_error_retried_then_succeeds(video, no_sleep):
    state = {"n": 0}

    def handler(prompt, contents):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("503 UNAVAILABLE: model overloaded")
        if state["n"] == 2:
            return _response(text="")  # empty body: transient too
        return _response(_payload(9))

    client = FakeClient(handler)
    shorts = [_shorts()[0]]
    costs = clip_rating.rate_candidates(shorts, _transcript(), video, client=client, model_name="m")
    assert state["n"] == 3
    assert no_sleep == list(clip_rating.RETRY_WAITS[:2])
    assert shorts[0]["rating"]["overall"] == 90.0
    assert len(costs) == 2  # the empty-body response was billed too


def test_transient_error_gives_up_after_retries(video, no_sleep):
    client = FakeClient(lambda p, c: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")))
    shorts = [_shorts()[0]]
    clip_rating.rate_candidates(shorts, _transcript(), video, client=client, model_name="m")
    assert shorts[0]["rating"] is None
    assert len(client.models.calls) == len(clip_rating.RETRY_WAITS) + 1


def test_non_transient_error_not_retried(video, no_sleep):
    def handler(prompt, contents):
        raise ValueError("400 INVALID_ARGUMENT: bad request")

    client = FakeClient(handler)
    shorts = [_shorts()[0]]
    clip_rating.rate_candidates(shorts, _transcript(), video, client=client, model_name="m")
    assert shorts[0]["rating"] is None
    assert len(client.models.calls) == 1
    assert no_sleep == []


def test_policy_block_leaves_none_without_retry(video, no_sleep):
    blocked = SimpleNamespace(name="PROHIBITED_CONTENT")
    client = FakeClient(lambda p, c: _response(_payload(5), blocked=blocked))
    shorts = _shorts()[:2]
    costs = clip_rating.rate_candidates(shorts, _transcript(), video, client=client, model_name="m")
    assert all(s["rating"] is None for s in shorts)
    assert len(client.models.calls) == 2
    assert no_sleep == []
    assert len(costs) == 2  # the refused calls were still made
    # sanity: raise_if_blocked is what classifies it
    with pytest.raises(GeminiBlockedError):
        import gemini_worker
        gemini_worker.raise_if_blocked(_response(_payload(5), blocked=blocked))


def test_no_api_key_leaves_all_none(monkeypatch, video):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    shorts = _shorts()
    shorts[0]["rating"] = {"stale": True}
    assert clip_rating.rate_candidates(shorts, _transcript(), video) == []
    assert all(s["rating"] is None for s in shorts)


def test_default_model_from_env(monkeypatch):
    monkeypatch.delenv("RATING_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert clip_rating._default_model() == "gemini-3.1-flash-lite"
    monkeypatch.setenv("GEMINI_MODEL", "g-model")
    assert clip_rating._default_model() == "g-model"
    monkeypatch.setenv("RATING_MODEL", "r-model")
    assert clip_rating._default_model() == "r-model"


def test_sample_frames_even_and_bounded(video):
    times = clip_rating.sample_times(2.0, 8.0, 3)
    assert times == pytest.approx([3.0, 5.0, 7.0])
    frames = clip_rating.sample_frames(video, 2.0, 8.0, 3)
    assert len(frames) == 3
    assert all(f[:2] == b"\xff\xd8" for f in frames)
    assert clip_rating.sample_frames(video, 2.0, 8.0, 0) == []
