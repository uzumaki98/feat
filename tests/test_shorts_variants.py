"""shorts_variants: the dead-air trim and its timeline remap (pure helpers)."""
import shorts_variants as sv


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def _r(ranges):
    return [(round(a, 2), round(b, 2)) for a, b in ranges]


def test_long_quiet_pauses_are_cut_to_a_breath():
    words = [_w("one", 1.0, 1.4), _w("two", 1.5, 1.9), _w("three", 4.0, 4.5)]
    # No loudness given: every wordless stretch counts as quiet.
    assert _r(sv.keep_ranges(words, 5.0)) == [(0.85, 2.05), (3.85, 5.0)]


def test_loud_wordless_action_is_kept():
    words = [_w("look", 0.0, 0.5)]
    env = [0.5] * 10 + [0.01] * 20 + [0.9] * 30  # talk, silence, loud payoff
    assert _r(sv.keep_ranges(words, 6.0, env)) == [(0.0, 1.15), (2.85, 6.0)]


def test_short_pauses_and_silent_clips():
    words = [_w("a", 0.0, 0.5), _w("b", 0.8, 1.2)]
    assert _r(sv.keep_ranges(words, 1.2)) == [(0.0, 1.2)]
    assert _r(sv.keep_ranges([], 12.5, [0.4] * 125)) == [(0.0, 12.5)]


def test_zoom_expr_is_flat_without_beats_and_numeric_only():
    assert sv.zoom_expr([]) == "1"
    expr = sv.zoom_expr([1.5, 9.0])
    assert expr.startswith("(1+0.120*max(") and "t-1.500" in expr and "t-9.000" in expr
    assert not any(c in expr for c in "';[]")  # nothing that could break the graph


def test_remap_follows_the_cuts():
    ranges = [(0.85, 2.05), (3.85, 4.85)]
    assert round(sv.remap(1.0, ranges), 3) == 0.15
    assert round(sv.remap(4.0, ranges), 3) == 1.35
    assert sv.remap(3.0, ranges) is None  # inside a cut
