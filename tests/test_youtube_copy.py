"""YouTube Shorts copy on every picked clip: the schema fields (defaulted so a
model that skips them never loses the clip), the prompt templates that ask for
them, and normalize_youtube_copy(), which makes the copy safe in code."""
import copy
import re

import pytest

gw = pytest.importorskip("gemini_worker")


def _detail_clip(**extra):
    clip = {
        "start": 1.0, "end": 20.0, "source_window_id": "w1", "predicted_score": 80,
        "video_description_for_tiktok": "t", "video_description_for_instagram": "i",
        "video_title_for_youtube_short": "Title", "viral_hook_text": "Hook",
    }
    clip.update(extra)
    return clip


class TestModels:
    def test_detail_defaults_when_fields_missing(self):
        m = gw.DetailClipModel.model_validate(_detail_clip())
        assert m.video_description_for_youtube_short == ""
        assert m.youtube_hashtags == []

    def test_visual_defaults_when_fields_missing(self):
        clip = _detail_clip()
        clip.pop("source_window_id")
        m = gw.VisualClipModel.model_validate(clip)
        assert m.video_description_for_youtube_short == ""
        assert m.youtube_hashtags == []

    def test_fields_are_kept_when_present(self):
        clip = _detail_clip(video_description_for_youtube_short="Watch this #AI #Shorts",
                            youtube_hashtags=["#AI", "#Shorts"])
        dumped = gw.DetailResponse.model_validate({"shorts": [clip]}).model_dump()
        out = dumped["shorts"][0]
        assert out["video_description_for_youtube_short"] == "Watch this #AI #Shorts"
        assert out["youtube_hashtags"] == ["#AI", "#Shorts"]

    def test_defaults_are_not_shared_between_instances(self):
        a = gw.DetailClipModel.model_validate(_detail_clip())
        a.youtube_hashtags.append("#x")
        b = gw.DetailClipModel.model_validate(_detail_clip())
        assert b.youtube_hashtags == []

    def test_new_fields_are_optional_in_json_schema(self):
        # llm_backend sends model_json_schema() to local servers that enforce it.
        schema = gw.DetailResponse.model_json_schema()
        item = schema["$defs"]["DetailClipModel"]
        assert "video_description_for_youtube_short" in item["properties"]
        assert "youtube_hashtags" in item["properties"]
        assert "video_description_for_youtube_short" not in item.get("required", [])
        assert "youtube_hashtags" not in item.get("required", [])


class TestTemplates:
    def test_detail_template_formats_and_asks_for_the_fields(self):
        text = gw.DETAIL_PROMPT_TEMPLATE.format(
            video_duration=100, language="es", min_clips=2, max_clips=4,
            min_secs=15.0, max_secs=60.0, windows_json="[]")
        assert '"video_description_for_youtube_short"' in text
        assert '"youtube_hashtags"' in text
        assert "#Shorts" in text
        assert re.findall(r"\{[a-z_]+\}", text) == []

    def test_visual_template_formats_and_asks_for_the_fields(self):
        text = gw.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=100, language="es", min_clips=2, max_clips=4,
            min_secs=15.0, max_secs=60.0)
        assert "video_description_for_youtube_short" in text
        assert "youtube_hashtags" in text
        assert "#Shorts" in text
        assert re.findall(r"\{[a-z_]+\}", text) == []


def _norm(**fields):
    return gw.normalize_youtube_copy(dict(fields))


class TestNormalizeYoutubeCopy:
    def test_empty_description_falls_back_to_title_with_shorts(self):
        out = _norm(video_title_for_youtube_short="Big reveal")
        assert out["video_description_for_youtube_short"] == "Big reveal #Shorts"
        assert out["youtube_hashtags"] == ["#Shorts"]

    def test_empty_title_falls_back_to_hook(self):
        out = _norm(viral_hook_text="  Stop doing   this  ")
        assert out["video_title_for_youtube_short"] == "Stop doing this"
        assert out["video_description_for_youtube_short"] == "Stop doing this #Shorts"

    def test_everything_missing_still_yields_shorts(self):
        out = _norm()
        assert out["video_title_for_youtube_short"] == ""
        assert out["video_description_for_youtube_short"] == "#Shorts"
        assert out["youtube_hashtags"] == ["#Shorts"]

    def test_hashtags_derived_from_description(self):
        out = _norm(video_title_for_youtube_short="T",
                    video_description_for_youtube_short="It works. #AI #Productivity #shorts")
        assert out["youtube_hashtags"] == ["#AI", "#Productivity", "#Shorts"]
        # #shorts already present (any case): not appended again.
        assert out["video_description_for_youtube_short"] == "It works. #AI #Productivity #shorts"

    def test_list_tags_missing_from_description_are_appended(self):
        out = _norm(video_title_for_youtube_short="T",
                    video_description_for_youtube_short="It works.",
                    youtube_hashtags=["#AI", "Coding"])
        assert out["youtube_hashtags"] == ["#AI", "#Coding", "#Shorts"]
        assert out["video_description_for_youtube_short"] == "It works. #AI #Coding #Shorts"

    def test_dedupe_case_insensitive_keeps_first_spelling(self):
        out = _norm(video_description_for_youtube_short="x",
                    youtube_hashtags=["#AI", "#ai", "#Ai", "#SHORTS", "#Shorts", "#Tech"])
        assert out["youtube_hashtags"] == ["#AI", "#Tech", "#Shorts"]

    def test_malformed_tags_dropped(self):
        out = _norm(video_description_for_youtube_short="x",
                    youtube_hashtags=["#two words", "#", "", None, "#bad-tag", "#2024",
                                      "#" + "a" * 60, "#ok_tag", "#Café", 42])
        assert out["youtube_hashtags"] == ["#ok_tag", "#Café", "#Shorts"]

    def test_capped_at_five_topical_plus_shorts(self):
        tags = [f"#tag{c}" for c in "abcdefgh"]
        out = _norm(video_description_for_youtube_short="x", youtube_hashtags=tags)
        assert out["youtube_hashtags"] == ["#taga", "#tagb", "#tagc", "#tagd", "#tage", "#Shorts"]
        assert len(out["youtube_hashtags"]) == 6

    def test_shorts_is_always_last(self):
        out = _norm(video_description_for_youtube_short="x",
                    youtube_hashtags=["#Shorts", "#AI"])
        assert out["youtube_hashtags"] == ["#AI", "#Shorts"]

    def test_string_hashtags_are_split(self):
        out = _norm(video_description_for_youtube_short="x",
                    youtube_hashtags="#AI, #Tech #Shorts")
        assert out["youtube_hashtags"] == ["#AI", "#Tech", "#Shorts"]

    def test_all_invalid_list_falls_back_to_description(self):
        out = _norm(video_description_for_youtube_short="Look #Gaming",
                    youtube_hashtags=["not valid", "#"])
        assert out["youtube_hashtags"] == ["#Gaming", "#Shorts"]

    def test_non_string_input_coerced(self):
        out = _norm(video_title_for_youtube_short=12345,
                    video_description_for_youtube_short=None,
                    youtube_hashtags=None)
        assert out["video_title_for_youtube_short"] == "12345"
        assert out["video_description_for_youtube_short"] == "12345 #Shorts"
        assert out["youtube_hashtags"] == ["#Shorts"]

    def test_title_trimmed_and_angle_brackets_removed(self):
        out = _norm(video_title_for_youtube_short="<b>" + "x" * 200,
                    video_description_for_youtube_short="a <script> b")
        assert out["video_title_for_youtube_short"] == "b" + "x" * 99
        assert len(out["video_title_for_youtube_short"]) == 100
        assert "<" not in out["video_description_for_youtube_short"]
        assert ">" not in out["video_description_for_youtube_short"]

    def test_long_description_fits_youtube_limit_and_keeps_tags(self):
        out = _norm(video_description_for_youtube_short="é" * 6000,
                    youtube_hashtags=["#AI"])
        desc = out["video_description_for_youtube_short"]
        assert len(desc.encode("utf-8")) <= gw.YT_DESCRIPTION_MAX_BYTES
        assert desc.endswith("#AI #Shorts")

    def test_appending_tags_never_pushes_past_the_limit(self):
        out = _norm(video_description_for_youtube_short="a" * 4990 + " #Shorts",
                    youtube_hashtags=["#Long"])
        desc = out["video_description_for_youtube_short"]
        assert len(desc.encode("utf-8")) <= gw.YT_DESCRIPTION_MAX_BYTES
        assert "#Long" in desc and "#Shorts" in desc

    def test_short_description_is_not_trimmed_to_target(self):
        body = "word " * 100  # ~500 chars: over the ~300 target, under the limit
        out = _norm(video_description_for_youtube_short=body)
        assert out["video_description_for_youtube_short"].startswith(body.strip())

    @pytest.mark.parametrize("clip", [
        {},
        {"video_title_for_youtube_short": "T"},
        {"viral_hook_text": "H"},
        {"video_description_for_youtube_short": "It works. #AI #Productivity"},
        {"video_description_for_youtube_short": "x", "youtube_hashtags": [f"#t{i}a" for i in range(9)]},
        {"video_description_for_youtube_short": "é" * 6000, "youtube_hashtags": ["#AI"]},
        {"video_description_for_youtube_short": "a" * 4990 + " #Shorts", "youtube_hashtags": ["#Long"]},
        {"video_title_for_youtube_short": "<x>" * 50, "youtube_hashtags": "#a #b,#a"},
        {"video_description_for_youtube_short": "Look #Gaming", "youtube_hashtags": ["bad tag"]},
    ])
    def test_idempotent(self, clip):
        once = gw.normalize_youtube_copy(copy.deepcopy(clip))
        twice = gw.normalize_youtube_copy(copy.deepcopy(once))
        assert twice == once

    def test_mutates_and_returns_same_dict_preserving_other_keys(self):
        clip = {"start": 1.0, "video_title_for_youtube_short": "T"}
        out = gw.normalize_youtube_copy(clip)
        assert out is clip
        assert clip["start"] == 1.0
        assert clip["youtube_hashtags"] == ["#Shorts"]

    def test_non_dict_returned_untouched(self):
        assert gw.normalize_youtube_copy(None) is None
        assert gw.normalize_youtube_copy("x") == "x"
