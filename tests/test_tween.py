import math

from stickman.rig.poses import POSES
from stickman.rig.skeleton import BONES, forward, ground_offset, ik_two_bone
from stickman.tween import (Track, clamp01, ease_in_out_cubic, ease_out_back, interpolate_keyframes, lerp,
                            lerp_angle)


def test_easings_endpoints():
    for f in (ease_in_out_cubic, ease_out_back):
        assert abs(f(0.0)) < 1e-9 and abs(f(1.0) - 1) < 1e-9
    assert ease_out_back(0.6) > 1.0  # rebasa (efecto "pop")


def test_lerp_angle_shortest_path():
    assert abs(lerp_angle(170, -170, 0.5)) == 180
    assert abs(lerp_angle(-170, 170, 0.25) - (-175)) < 1e-9
    assert lerp_angle(10, 30, 0.5) == 20


def test_clamp_and_lerp():
    assert clamp01(-1) == 0 and clamp01(2) == 1
    assert lerp(0, 10, 0.3) == 3


def test_track_reaches_targets_and_is_continuous():
    tr = Track([(0, {"a": 0.0}, 0), (1, {"a": 10.0}, 0.5), (1.2, {"a": -10.0}, 0.5)])
    assert tr(0.5)["a"] == 0
    assert tr(1.0)["a"] == 0
    # la transición 2 empieza desde el valor real en t=1.2 (no desde 10): sin saltos
    v_at_12 = tr(1.2)["a"]
    assert 0 < v_at_12 < 10
    assert abs(tr(1.2 + 1e-4)["a"] - v_at_12) < 0.01
    assert tr(5)["a"] == -10
    prev = tr(0)["a"]
    for i in range(1, 400):
        v = tr(i / 100)["a"]
        assert abs(v - prev) < 1.5
        prev = v


def test_track_progress():
    tr = Track([(0, {"a": 0.0}, 0), (1, {"a": 1.0}, 0.5)])
    assert tr.progress(0.5) == (0, 1.0)
    idx, u = tr.progress(1.25)
    assert idx == 1 and abs(u - 0.5) < 1e-9


def test_angular_track_takes_short_way():
    tr = Track([(0, {"a": 170.0}, 0), (1, {"a": -170.0}, 1.0)], angular=True)
    mid = tr(1.5)["a"]
    assert abs(abs(mid) - 180) < 1e-6


def test_camera_keyframes_interpolation():
    keys = [dict(t=0, zoom=1, x=0, y=0, ease="linear"), dict(t=2, zoom=2, x=1, y=0, ease="linear")]
    assert interpolate_keyframes(keys, -1, ("zoom",))["zoom"] == 1
    assert interpolate_keyframes(keys, 1, ("zoom", "x")) == {"zoom": 1.5, "x": 0.5}
    assert interpolate_keyframes(keys, 9, ("zoom",))["zoom"] == 2


def test_bones_never_stretch():
    for name, pose in POSES.items():
        p = forward(pose)
        for side in ("l", "r"):
            ua = math.dist(p[f"{side}_shoulder"], p[f"{side}_elbow"])
            fa = math.dist(p[f"{side}_elbow"], p[f"{side}_wrist"])
            th = math.dist(p["hip"], p[f"{side}_knee"])
            assert abs(ua - BONES["upper_arm"]) < 1e-6 and abs(fa - BONES["forearm"]) < 1e-6, name
            assert abs(th - BONES["thigh"]) < 1e-6, name


def test_grounding_puts_lowest_foot_on_floor():
    for pose in POSES.values():
        p = forward(pose)
        dy = ground_offset(p)
        assert abs(max(p["l_ankle"][1], p["r_ankle"][1]) + dy) < 1e-9


def test_ik_reaches_target():
    for target in ((50, 60), (36, -30), (-40, 52), (120, 10)):
        for bend in (1, -1):
            sh, el = ik_two_bone(target, BONES["upper_arm"], BONES["forearm"], bend)
            ex = math.cos(math.radians(sh)) * BONES["upper_arm"]
            ey = math.sin(math.radians(sh)) * BONES["upper_arm"]
            wx = ex + math.cos(math.radians(sh + el)) * BONES["forearm"]
            wy = ey + math.sin(math.radians(sh + el)) * BONES["forearm"]
            assert math.dist((wx, wy), target) < 1e-6
