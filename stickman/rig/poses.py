"""Biblioteca de poses (ángulos de articulación en grados).

Idea original (pose = diccionario de ángulos con valores por defecto de pie)
de src/rig/poses.js de stickman-explainers (MIT, Copyright (c) 2026 21cabbagee).
Valores y poses nuevas (think, shrug, cheer, arms_crossed, present, sad) propios.
El brazo/pierna "r" es el delantero cuando el personaje mira a la derecha.
"""
from __future__ import annotations

import math
from typing import Dict

from .skeleton import BONES, DEFAULT_JOINTS, ik_two_bone

Joints = Dict[str, float]


def _arm(side: str, target, elbow: str = "down") -> Joints:
    """IK hacia ``target`` (relativo al hombro). ``elbow`` = 'down' u 'out'."""
    best = None
    for bend in (1, -1):
        sh, el = ik_two_bone(target, BONES["upper_arm"], BONES["forearm"], bend)
        ex = math.cos(math.radians(sh)) * BONES["upper_arm"]
        ey = math.sin(math.radians(sh)) * BONES["upper_arm"]
        score = ey if elbow == "down" else (ex if side == "r" else -ex)
        if best is None or score > best[0]:
            best = (score, sh, el)
    return {f"{side}_sh": best[1], f"{side}_el": best[2]}


def _pose(**kw: float) -> Joints:
    j = dict(DEFAULT_JOINTS)
    j.update(kw)
    return j


POSES: Dict[str, Joints] = {
    "idle": _pose(),
    "talk": _pose(r_sh=62, r_el=-58, l_sh=108, l_el=14),
    "wave": _pose(r_sh=-28, r_el=-62, l_sh=106, l_el=10, neck=-4),
    "point": _pose(r_sh=-12, r_el=-4, l_sh=108, l_el=12),
    "think": _pose(neck=6, **_arm("r", (36, -30), "down"), **_arm("l", (34, 70), "out")),
    "shrug": _pose(neck=-7, l_sh=148, l_el=78, r_sh=32, r_el=-78),
    "cheer": _pose(l_sh=-134, l_el=14, r_sh=-46, r_el=-14, spine=-91),
    "arms_crossed": _pose(**_arm("r", (-40, 52), "out"), **_arm("l", (44, 60), "out")),
    "present": _pose(r_sh=18, r_el=-48, l_sh=110, l_el=12),
    "sad": _pose(spine=-88, neck=20, l_sh=96, l_el=2, r_sh=84, r_el=-2, l_hip=95, r_hip=85),
    "sit": _pose(spine=-94, l_sh=74, l_el=-30, r_sh=64, r_el=-40,
                 l_hip=-6, l_knee=92, r_hip=4, r_knee=84),
}

# Fases del salto (crouch -> air -> land); se mezclan en scene.py
JUMP_CROUCH = _pose(spine=-80, l_sh=128, l_el=-16, r_sh=52, r_el=18,
                    l_hip=62, l_knee=78, r_hip=48, r_knee=80)
JUMP_AIR = _pose(spine=-92, l_sh=-118, l_el=12, r_sh=-62, r_el=-12,
                 l_hip=112, l_knee=26, r_hip=70, r_knee=34)


def get_pose(name: str) -> Joints:
    return POSES.get(name, POSES["idle"])


def walk_overlay(phase: float) -> Joints:
    """Ciclo de andar procedimental (offset sobre la pose base)."""
    s = math.sin(phase)
    c = math.cos(phase)
    return {
        "l_hip": 90 + 26 * s, "r_hip": 90 - 26 * s,
        "l_knee": max(0.0, -c) * 38 * (1 if s < 0.3 else 0.6),
        "r_knee": max(0.0, c) * 38 * (1 if s > -0.3 else 0.6),
        "l_sh": 90 - 24 * s, "r_sh": 90 + 24 * s,
        "l_el": -18, "r_el": -18,
        "spine": -87.0,
    }
