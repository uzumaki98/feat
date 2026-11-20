"""facecam_template: the streamer layout's geometry (pure helpers, no ffmpeg)."""
import facecam_template as ft

W, H = 1920, 1080
CAM = (0, 460, 390, 240)  # webcam pinned to the left edge, mid-height


def test_camera_window_stays_inside_the_webcam():
    w, h, x, y = ft.cam_crop(CAM, W, H, 1080 / 730)
    assert abs(w / h - 1080 / 730) < 0.03
    assert x >= CAM[0] and y >= CAM[1]
    assert x + w <= CAM[0] + CAM[2] and y + h <= CAM[1] + CAM[3]


def test_gameplay_window_avoids_the_webcam():
    w, h, x, y = ft.game_crop(CAM, W, H, 1080 / 1190)
    assert h == H and abs(w / h - 1080 / 1190) < 0.01
    assert x >= CAM[0] + CAM[2] and x + w <= W
    right = (1500, 400, 400, 250)
    w, h, x, y = ft.game_crop(right, W, H, 1080 / 1190)
    assert x + w <= right[0]


def test_graph_stacks_camera_over_game_at_full_height():
    g = ft.stack_graph(W, H, CAM)
    cam_h = ft._even(1920 * ft.CAM_RATIO)
    assert f"scale=1080:{cam_h}" in g and f"scale=1080:{1920 - cam_h}" in g
    assert "vstack=inputs=2" in g and g.endswith("[v]")


def test_snap_box_moves_onto_the_border_or_keeps_the_estimate():
    rows = [0.0] * H
    cols = [0.0] * W
    rows[470], rows[690], cols[400] = 9.0, 9.0, 9.0
    assert ft.snap_box(CAM, rows, cols, W, H) == (0, 470, 400, 220)
    # Nothing found: the estimate stands.
    assert ft.snap_box(CAM, [0.0] * H, [0.0] * W, W, H) == CAM
    # A "border" that would make an absurd box is ignored.
    rows2 = [0.0] * H
    rows2[300], rows2[310] = 9.0, 9.0
    assert ft.snap_box(CAM, rows2, cols, W, H) == CAM
