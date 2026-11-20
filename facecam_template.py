"""Streamer template: the webcam on top, the gameplay below.

The look of most stream-clip channels: the streamer's camera, enlarged, fills
the top band; the game fills the rest; the hook sits on the seam. It needs the
horizontal source, because the vertical render has already thrown the webcam
box away, so the Shorts runner's edit stage cuts the clip from the source
again, finds the webcam (camera_inset.detect) and stacks the two here. A clip
without a usable webcam (a menu, a full-screen IRL camera) keeps main.py's
framing.

Only crop, scale and vstack, so the graph is the kind the rest of the
renderer accepts.
"""
import subprocess

# Share of the height the camera band takes. Stream-clip channels sit around
# 35-40%: the face reads, and the game keeps the larger half.
CAM_RATIO = 0.38


def _even(v):
    v = int(round(v))
    return max(2, v - v % 2)


def cam_crop(box, orig_w, orig_h, aspect):
    """The largest ``aspect`` window INSIDE the webcam box (2% in from its
    border), so the band is scaled uniformly, the face is never stretched and
    nothing around the camera (counters, menus) leaks into the band."""
    x, y, w, h = box
    x, y, w, h = x + 0.02 * w, y + 0.02 * h, 0.96 * w, 0.96 * h
    if w / float(h) > aspect:
        want = h * aspect
        x, w = x + w / 2.0 - want / 2.0, want
    else:
        want = w / aspect
        y, h = y + h / 2.0 - want / 2.0, want
    w, h = _even(min(w, orig_w)), _even(min(h, orig_h))
    x = _even(max(0, min(x, orig_w - w))) if x > 1 else 0
    y = _even(max(0, min(y, orig_h - h))) if y > 1 else 0
    return w, h, min(x, orig_w - w), min(y, orig_h - h)


def game_crop(box, orig_w, orig_h, aspect):
    """Full-height gameplay window of ``aspect``, centred on the part of the
    frame the webcam does not cover, and kept off the webcam when it fits."""
    h = orig_h
    w = _even(h * aspect)
    if w > orig_w:
        w = orig_w
        h = _even(w / aspect)
    bx, _, bw, _ = box
    if bx + bw / 2.0 < orig_w / 2.0:      # webcam on the left
        lo, hi = bx + bw, orig_w
    else:                                  # webcam on the right
        lo, hi = 0, bx
    centre = (lo + hi) / 2.0 if hi - lo >= w else orig_w / 2.0
    x = int(round(centre - w / 2.0))
    if hi - lo >= w:
        x = max(int(lo), min(x, int(hi) - w))
    x = max(0, min(x, orig_w - w))
    x -= x % 2
    return w, h, x, (orig_h - h) // 2 // 2 * 2


def stack_graph(orig_w, orig_h, box, out_w=1080, out_h=1920, cam_ratio=CAM_RATIO):
    """Filter graph: camera band on top, gameplay below, ``[v]`` out."""
    cam_h = _even(out_h * cam_ratio)
    game_h = out_h - cam_h
    cw, ch, cx, cy = cam_crop(box, orig_w, orig_h, out_w / float(cam_h))
    gw, gh, gx, gy = game_crop(box, orig_w, orig_h, out_w / float(game_h))
    return (f"[0:v]split=2[c][g];"
            f"[c]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={out_w}:{cam_h}:flags=lanczos[cam];"
            f"[g]crop=w={gw}:h={gh}:x={gx}:y={gy},scale={out_w}:{game_h}:flags=lanczos[game];"
            f"[cam][game]vstack=inputs=2,setsar=1[v]")


def edges(profile, lo, hi):
    """Index of the strongest value of ``profile`` in [lo, hi), or None when
    nothing stands out (a peak under twice the window's median is texture,
    not a border)."""
    lo, hi = max(0, int(lo)), min(len(profile), int(hi))
    if hi - lo < 2:
        return None
    window = sorted(profile[lo:hi])
    best = lo + int(max(range(hi - lo), key=lambda i: profile[lo + i]))
    median = window[len(window) // 2]
    if profile[best] <= 0 or profile[best] < 2.0 * median:
        return None
    return best


def snap_box(box, row_edge, col_edge, frame_w, frame_h):
    """``box`` moved onto the webcam's own border.

    camera_inset grows a 16:9 box from the face, which on a stream overlay
    takes in what sits around the camera (a follower counter above it, the
    game's menu beside it). The camera's border is the one edge that stays put
    while the picture on both sides changes, so the time-averaged gradient
    peaks on it: ``row_edge[r]`` is that average between rows r-1 and r over
    the box's columns, ``col_edge[c]`` the same across columns. A border that
    cannot be found, or a result far from the estimate, keeps the estimate."""
    x, y, w, h = box
    top = edges(row_edge, y - 0.6 * h, y + 0.4 * h)
    bottom = edges(row_edge, y + 0.6 * h, y + h + 0.6 * h)
    left = 0 if x <= 2 else edges(col_edge, x - 0.4 * w, x + 0.4 * w)
    right = frame_w if x + w >= frame_w - 2 else edges(col_edge, x + 0.6 * w, x + w + 0.6 * w)
    if None in (top, bottom, left, right):
        return box
    nw, nh = right - left, bottom - top
    if not (0.6 * w <= nw <= 1.8 * w and 0.6 * h <= nh <= 1.8 * h and 1.1 <= nw / nh <= 2.4):
        return box
    return (int(left), int(top), int(nw), int(nh))


def edge_profiles(video_path, box, samples=12):
    """Time-averaged row and column gradients around ``box`` (see snap_box)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y, w, h = box
        c0, c1 = max(0, int(x + 0.15 * w)), min(fw, int(x + 0.85 * w))
        r0, r1 = max(0, int(y + 0.15 * h)), min(fh, int(y + 0.85 * h))
        rows, cols, n = np.zeros(fh), np.zeros(fw), 0
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            rows[1:] += np.abs(np.diff(g[:, c0:c1], axis=0)).mean(axis=1)
            cols[1:] += np.abs(np.diff(g[r0:r1, :], axis=1)).mean(axis=0)
            n += 1
    finally:
        cap.release()
    if not n:
        return None
    return rows / n, cols / n, fw, fh


def detect(clip_path):
    """The webcam box (x, y, w, h) in a horizontal clip, snapped to its
    border, or None when there is no usable webcam."""
    import camera_inset
    box = camera_inset.detect(clip_path)
    if not box:
        return None
    prof = edge_profiles(clip_path, box)
    return snap_box(box, *prof) if prof else box


def frame(source, start, end, work, out_w=1080, out_h=1920):
    """The streamer-template clip of [start, end] of ``source``, or None when
    the stretch has no usable webcam. Returns (path, camera band height)."""
    import os
    import shorts_variants
    os.makedirs(work, exist_ok=True)
    w, h, fps, duration = shorts_variants.probe(source)
    if w <= h or start is None or end is None or end <= start:
        return None  # already vertical, or no span: nothing to stack
    horiz = os.path.join(work, "horizontal.mp4")
    shorts_variants.cut(source, [(max(0.0, start), min(duration, end))], horiz, fps)
    box = detect(horiz)
    if not box:
        return None
    out = os.path.join(work, "template.mp4")
    render(horiz, out, box, w, h, out_w, out_h)
    return out, _even(out_h * CAM_RATIO)


def render(src, out, box, orig_w, orig_h, out_w=1080, out_h=1920):
    """Write the stacked vertical clip (audio copied)."""
    graph = stack_graph(orig_w, orig_h, box, out_w, out_h)
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", graph,
                        "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "16",
                        "-preset", "veryfast", "-c:a", "copy", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")
