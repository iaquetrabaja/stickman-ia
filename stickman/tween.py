"""Easing, interpolación y pistas de keyframes.

Funciones de easing y ``lerp_angle`` portadas de stickman-explainers
(src/rig/tween.js, MIT, Copyright (c) 2026 21cabbagee).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple


def linear(t: float) -> float:
    return t


def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 2) / 2


def ease_in_out_cubic(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


EASINGS: Dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in_out": ease_in_out_cubic,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_out": ease_out_cubic,
    "ease_out_back": ease_out_back,
}


def get_easing(name: str | None) -> Callable[[float], float]:
    return EASINGS.get(name or "ease_in_out", ease_in_out_cubic)


def clamp01(t: float) -> float:
    return 0.0 if t < 0 else 1.0 if t > 1 else t


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolación por el camino más corto (grados)."""
    diff = ((b - a + 180) % 360 + 360) % 360 - 180
    return a + diff * t


def lerp_dict(a: Dict[str, float], b: Dict[str, float], t: float, angular: bool = False) -> Dict[str, float]:
    f = lerp_angle if angular else lerp
    out = {}
    for k in a.keys() | b.keys():
        out[k] = f(a.get(k, b.get(k, 0.0)), b.get(k, a.get(k, 0.0)), t)
    return out


class Track:
    """Pista de keyframes de diccionarios de floats.

    En el instante ``t_i`` empieza una transición de ``dur_i`` segundos desde el
    estado *real* en ese momento (aunque la transición anterior no hubiera
    terminado) hacia el valor ``v_i``. Así nunca hay saltos.
    """

    def __init__(self, keys: Sequence[Tuple[float, Dict[str, float], float]], angular: bool = False,
                 easing: Callable[[float], float] = ease_in_out_cubic):
        self.keys: List[Tuple[float, Dict[str, float], float]] = sorted(keys, key=lambda k: k[0])
        if not self.keys:
            raise ValueError("Track sin keyframes")
        self.angular = angular
        self.easing = easing
        self._start_cache: Dict[int, Dict[str, float]] = {}

    def _start_state(self, i: int) -> Dict[str, float]:
        if i == 0:
            return self.keys[0][1]
        if i not in self._start_cache:
            self._start_cache[i] = self._eval_segment(i - 1, self.keys[i][0])
        return self._start_cache[i]

    def _eval_segment(self, i: int, t: float) -> Dict[str, float]:
        t_i, target, dur = self.keys[i]
        if i == 0:
            return target
        start = self._start_state(i)
        if dur <= 0:
            return target
        u = clamp01((t - t_i) / dur)
        if u >= 1:
            return target
        return lerp_dict(start, target, self.easing(u), self.angular)

    def progress(self, t: float) -> Tuple[int, float]:
        """(índice del keyframe activo, progreso 0-1 de su transición)."""
        idx = 0
        for i, (ti, _, _) in enumerate(self.keys):
            if ti <= t:
                idx = i
            else:
                break
        ti, _, dur = self.keys[idx]
        if idx == 0 or dur <= 0:
            return idx, 1.0
        return idx, clamp01((t - ti) / dur)

    def __call__(self, t: float) -> Dict[str, float]:
        idx = 0
        for i, (ti, _, _) in enumerate(self.keys):
            if ti <= t:
                idx = i
            else:
                break
        return self._eval_segment(idx, t)


def interpolate_keyframes(keys: Sequence[dict], t: float, fields: Sequence[str]) -> Dict[str, float]:
    """Interpolación clásica (cámara): en ``keys[i]['t']`` vale exactamente keys[i]."""
    if not keys:
        return {f: 0.0 for f in fields}
    if t <= keys[0]["t"]:
        return {f: keys[0][f] for f in fields}
    if t >= keys[-1]["t"]:
        return {f: keys[-1][f] for f in fields}
    for a, b in zip(keys, keys[1:]):
        if a["t"] <= t <= b["t"]:
            span = b["t"] - a["t"]
            u = 1.0 if span == 0 else (t - a["t"]) / span
            e = get_easing(b.get("ease"))(clamp01(u))
            return {f: lerp(a[f], b[f], e) for f in fields}
    return {f: keys[-1][f] for f in fields}


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / (e1 - e0)) if e1 != e0 else (1.0 if x >= e1 else 0.0)
    return t * t * (3 - 2 * t)


def hash01(*vals: float) -> float:
    """Pseudo-aleatorio determinista en [0,1)."""
    x = math.sin(sum(v * (12.9898 + 7.233 * i) for i, v in enumerate(vals)) + 0.5) * 43758.5453
    return x - math.floor(x)
