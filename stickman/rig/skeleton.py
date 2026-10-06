"""Cinemática directa/inversa del muñeco.

Portado y adaptado de src/rig/skeleton.js de stickman-explainers (MIT,
Copyright (c) 2026 21cabbagee): ángulos en grados en espacio de pantalla
(0 = derecha, 90 = abajo, -90 = arriba); solo se animan ángulos, las longitudes
de hueso son fijas, así las extremidades nunca se deforman.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

Point = Tuple[float, float]

BONES = {
    "torso": 178.0,
    "neck": 20.0,
    "head_r": 60.0,
    "upper_arm": 98.0,
    "forearm": 92.0,
    "thigh": 118.0,
    "shin": 112.0,
    "shoulder_drop": 16.0,  # los brazos nacen un poco por debajo del final del torso
    "foot": 26.0,
    "shoulder_w": 12.0,  # separación lateral de los hombros respecto al eje
}
LINE = 11.0  # grosor de trazo del personaje (unidades del rig)

DEFAULT_JOINTS = {
    "spine": -90.0, "neck": 0.0,
    "l_sh": 104.0, "l_el": 8.0, "r_sh": 76.0, "r_el": -8.0,
    "l_hip": 97.0, "l_knee": 0.0, "r_hip": 83.0, "r_knee": 0.0,
}

DEG = math.pi / 180.0


def _proj(o: Point, ang_deg: float, length: float) -> Point:
    a = ang_deg * DEG
    return (o[0] + math.cos(a) * length, o[1] + math.sin(a) * length)


def forward(j: Dict[str, float], bones: Dict[str, float] = BONES) -> Dict[str, Point]:
    """Puntos del esqueleto con la cadera en (0,0), mirando a la derecha."""
    g = lambda k: j.get(k, DEFAULT_JOINTS[k])  # noqa: E731
    hip = (0.0, 0.0)
    spine = g("spine")
    chest = _proj(hip, spine, bones["torso"])
    shoulder = _proj(hip, spine, bones["torso"] - bones["shoulder_drop"])
    neck_a = spine + g("neck")
    head_base = _proj(chest, neck_a, bones["neck"])
    head = _proj(head_base, neck_a, bones["head_r"])
    pts = {"hip": hip, "chest": chest, "shoulder": shoulder, "head_base": head_base, "head": head}
    ux, uy = math.cos(spine * DEG), math.sin(spine * DEG)
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        sp = (shoulder[0] - uy * bones["shoulder_w"] * sgn, shoulder[1] + ux * bones["shoulder_w"] * sgn)
        pts[f"{side}_shoulder"] = sp
        sh = g(f"{side}_sh")
        el = _proj(sp, sh, bones["upper_arm"])
        wr = _proj(el, sh + g(f"{side}_el"), bones["forearm"])
        hp = g(f"{side}_hip")
        kn = _proj(hip, hp, bones["thigh"])
        an = _proj(kn, hp + g(f"{side}_knee"), bones["shin"])
        pts[f"{side}_elbow"], pts[f"{side}_wrist"] = el, wr
        pts[f"{side}_knee"], pts[f"{side}_ankle"] = kn, an
    pts["head_angle"] = (neck_a + 90.0, 0.0)  # inclinación de la cabeza (0 = recta)
    return pts


def ground_offset(pts: Dict[str, Point]) -> float:
    """Desplazamiento vertical para que el pie más bajo toque el suelo (y=0)."""
    return -max(pts["l_ankle"][1], pts["r_ankle"][1])


def ik_two_bone(target: Point, l1: float, l2: float, bend: int = 1) -> Tuple[float, float]:
    """IK analítica de 2 huesos. Devuelve (ángulo absoluto hombro, ángulo relativo codo).

    ``bend`` = +1 dobla el codo en sentido horario (pantalla), -1 antihorario.
    """
    tx, ty = target
    d = math.hypot(tx, ty)
    d = max(1e-6, min(d, l1 + l2 - 1e-6, max(abs(l1 - l2) + 1e-6, d)))
    base = math.atan2(ty, tx)
    cos_a = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    sh = base - bend * a
    el_pt = (math.cos(sh) * l1, math.sin(sh) * l1)
    fa = math.atan2(ty - el_pt[1], tx - el_pt[0])
    rel = (fa - sh) / DEG
    rel = (rel + 180) % 360 - 180
    return sh / DEG, rel
