"""Renderizador determinista: guion + audio -> fotogramas -> ffmpeg (MP4).

La arquitectura (timeline por escenas con duración real del audio, cámara por
keyframes, subtítulos en espacio de pantalla, boca por amplitud, fotogramas
enviados por tubería a ffmpeg) sigue a stickman-explainers (MIT, Copyright (c)
2026 21cabbagee), reimplementada en Python con skia.
"""
from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import skia

from .audio import Cue
from .props import BUBBLES, DESK_TOP, GROUND_PROPS, PropPainter
from .rig.draw import CharacterFrame, CharacterLook, CharacterPainter, Theme, hex_color, stroke_paint
from .rig.face import get_expression
from .rig.poses import JUMP_AIR, JUMP_CROUCH, get_pose, walk_overlay
from .rig.skeleton import BONES
from .schema import Script, background_hex
from .text import draw_image_centered, fit_font, font, render_outlined, text_width
from .tween import (Track, clamp01, ease_in_out_cubic, ease_out_back, ease_out_cubic, hash01,
                    interpolate_keyframes, lerp, lerp_dict, smoothstep)

CHAR_H = 575.0  # altura aproximada del muñeco en unidades de rig
WALK_SPEED = 1.25  # unidades de escenario por segundo
JUMP_DUR = 0.95
POSE_BLEND = 0.38
EXPR_BLEND = 0.25
ACCENT = "#D24D1E"


@dataclass
class Layout:
    W: int
    H: int
    scale: float          # px por unidad de rig
    ground_y: float
    stage_half: float     # px que equivalen a x=1
    keyword_y: float
    keyword_size: float
    caption_y: float
    caption_size: float
    caption_maxw: float
    caption_words: int
    caption_chars: int


FORMATS: Dict[str, Layout] = {
    "9:16": Layout(1080, 1920, 1.15, 1290, 400, 330, 92, 1530, 86, 940, 4, 22),
    "16:9": Layout(1920, 1080, 0.90, 850, 640, 150, 76, 960, 66, 1500, 5, 34),
}


# --------------------------------------------------------------------------- #
@dataclass
class Move:
    t0: float
    x0: float
    x1: float
    dur: float


@dataclass
class CharRuntime:
    id: str
    painter: CharacterPainter
    x0: float
    moves: List[Move]
    pose_track: Track
    pose_names: List[str]
    expr_track: Track
    facing_keys: List[Tuple[float, int]]
    jumps: List[float]
    pop_in: bool
    seed: float
    speaker: bool
    last_x: float = 0.0
    hands: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def x_at(self, t: float) -> Tuple[float, float, int]:
        """(x, distancia recorrida acumulada, dirección de movimiento o 0)."""
        x = self.x0
        dist = 0.0
        moving = 0
        for m in self.moves:
            if t < m.t0:
                break
            u = clamp01((t - m.t0) / m.dur) if m.dur > 0 else 1.0
            x = lerp(m.x0, m.x1, u)
            dist += abs(m.x1 - m.x0) * u
            if 0 < u < 1:
                moving = 1 if m.x1 > m.x0 else -1
        return x, dist, moving

    def walk_weight(self, t: float) -> float:
        w = 0.0
        for m in self.moves:
            if m.t0 - 0.12 <= t <= m.t0 + m.dur + 0.12:
                w = max(w, smoothstep(m.t0 - 0.12, m.t0 + 0.08, t) * (1 - smoothstep(m.t0 + m.dur - 0.08, m.t0 + m.dur + 0.12, t)))
        return w


@dataclass
class PropRuntime:
    kind: str
    x: float
    y: Optional[float]
    t0: float
    t1: Optional[float]
    holder: Optional[str]
    text: Optional[str]
    trend: str
    on_desk: Optional[float] = None  # x de la mesa


@dataclass
class SceneRuntime:
    index: int
    start: float
    duration: float
    bg: str
    theme: Theme
    chars: List[CharRuntime]
    props: List[PropRuntime]
    keywords: List[Tuple[float, float, skia.Image]]
    camera: List[dict]
    prop_painter: PropPainter


# --------------------------------------------------------------------------- #
class VideoRenderer:
    def __init__(self, script: Script, fmt: str, scene_durations: List[float], envelope: np.ndarray,
                 cues: List[Cue], fps: int = 30, render_scale: float = 1.0):
        self.script = script
        self.L = FORMATS[fmt]
        self.fps = fps
        self.rs = render_scale
        self.out_w = int(round(self.L.W * render_scale / 2) * 2)
        self.out_h = int(round(self.L.H * render_scale / 2) * 2)
        self.env = envelope
        self.cues = cues
        self.cast = {c.id: c for c in script.cast}
        self._painters: Dict[Tuple[str, str], CharacterPainter] = {}
        self._caption_cache: Dict[str, skia.Image] = {}
        self.scenes: List[SceneRuntime] = []
        t = 0.0
        frame_cursor = 0
        self.scene_frames: List[Tuple[int, int]] = []
        for i, (sc, d) in enumerate(zip(script.scenes, scene_durations)):
            n = int(round(d * fps))
            self.scene_frames.append((frame_cursor, n))
            self.scenes.append(self._build_scene(i, sc, frame_cursor / fps, n / fps))
            frame_cursor += n
            t += d
        self.total_frames = frame_cursor
        currency = "€" if script.language in ("es", "fr", "it", "de", "pt") else "$"
        for s in self.scenes:
            s.prop_painter.currency = currency

    # ------------------------------------------------------------------ #
    def _painter(self, cid: str, theme: Theme) -> CharacterPainter:
        key = (cid, theme.ink)
        if key not in self._painters:
            m = self.cast.get(cid)
            look = CharacterLook(m.preset.color, m.preset.hair, m.preset.hair_color, m.preset.accessory) if m \
                else CharacterLook()
            self._painters[key] = CharacterPainter(look, theme)
        return self._painters[key]

    def _keyword_image(self, text: str, theme: Theme) -> skia.Image:
        L = self.L
        txt = text.upper()
        f, lines = fit_font(txt, L.keyword_size, L.W * 0.82, 2, "Black", 36)
        fs = f.getSize()
        m = f.getMetrics()
        asc, desc = -m.fAscent, m.fDescent
        lh = fs * 1.08
        pad = 22
        w = max(text_width(f, ln) for ln in lines) + pad * 2
        h = lh * (len(lines) - 1) + asc + desc + pad * 2
        surf = skia.Surface(int(w) + 4, int(h) + 4)
        c = surf.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        ink = skia.Paint(AntiAlias=True, Color=hex_color(theme.ink))
        hl = skia.Paint(AntiAlias=True, Color=hex_color(ACCENT, 0.9 if theme.ink != Theme().ink else 0.85))
        y = pad + asc
        for ln in lines:
            tw = text_width(f, ln)
            x = (w - tw) / 2
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x - 12, y - asc * 0.42, x + tw + 12, y + desc * 0.9),
                                              6, 6), hl)
            c.drawString(ln, x, y, f, ink)
            y += lh
        return surf.makeImageSnapshot()

    def _build_scene(self, idx: int, sc, start: float, D: float) -> SceneRuntime:
        bg = background_hex(sc.background)
        theme = Theme.for_background(bg)
        chars: List[CharRuntime] = []
        n_chars = len(sc.characters)
        for ci, ch in enumerate(sc.characters):
            others = [o.x for o in sc.characters if o is not ch]
            if ch.facing:
                face0 = 1 if ch.facing == "right" else -1
            elif others:
                face0 = 1 if sum(others) / len(others) >= ch.x else -1
            else:
                face0 = 1 if ch.x <= 0.05 else -1
            moves: List[Move] = []
            x_cur = ch.x
            off = 1.65 * (self.L.W / 2) / self.L.stage_half
            if ch.enter in ("left", "right"):
                sx = -off if ch.enter == "left" else off
                moves.append(Move(0.0, sx, ch.x, abs(ch.x - sx) / WALK_SPEED))
                x_start = sx
            else:
                x_start = ch.x
            t_free = moves[-1].t0 + moves[-1].dur if moves else 0.0
            pose_keys = [(0.0, get_pose(ch.pose), 0.0)]
            pose_names = [ch.pose]
            expr_keys = [(0.0, get_expression(ch.expression), 0.0)]
            facing_keys: List[Tuple[float, int]] = [(0.0, face0)]
            jumps: List[float] = []
            last_pose = ch.pose
            for a in sorted(ch.actions, key=lambda a: a.at):
                ta = a.at * D
                if a.pose:
                    if a.pose == "jump":
                        jumps.append(ta)
                    else:
                        pose_keys.append((ta, get_pose(a.pose), POSE_BLEND))
                        pose_names.append(a.pose)
                        last_pose = a.pose
                if a.expression:
                    expr_keys.append((ta, get_expression(a.expression), EXPR_BLEND))
                if a.facing:
                    facing_keys.append((ta, 1 if a.facing == "right" else -1))
                if a.move_to is not None and abs(a.move_to - x_cur) > 0.02:
                    t0 = max(ta, t_free)
                    dur = abs(a.move_to - x_cur) / WALK_SPEED
                    moves.append(Move(t0, x_cur, a.move_to, dur))
                    x_cur = a.move_to
                    t_free = t0 + dur
            if ch.pose == "jump":
                jumps.append(0.2)
            if ch.exit in ("left", "right"):
                ex = -off if ch.exit == "left" else off
                dur = abs(ex - x_cur) / WALK_SPEED
                t0 = max(t_free, D - dur - 0.05)
                moves.append(Move(t0, x_cur, ex, dur))
            chars.append(CharRuntime(
                id=ch.id, painter=self._painter(ch.id, theme), x0=x_start, moves=moves,
                pose_track=Track(pose_keys, angular=True), pose_names=pose_names,
                expr_track=Track(expr_keys), facing_keys=facing_keys, jumps=jumps,
                pop_in=ch.enter == "pop", seed=idx * 7.13 + ci * 3.71,
                speaker=(sc.speaker == ch.id), last_x=x_start))
        props: List[PropRuntime] = []
        desks = [p.x for p in sc.props if p.type == "desk"]
        for p in sc.props:
            pr = PropRuntime(p.type, p.x, p.y, p.at * D, p.until * D if p.until is not None else None,
                             p.holder, p.text, p.trend)
            if p.type not in GROUND_PROPS and p.type not in BUBBLES and not p.holder and p.y is None and desks:
                near = min(desks, key=lambda dx: abs(dx - p.x))
                if abs(near - p.x) < 0.45 and p.type in ("laptop", "coffee", "phone", "money", "clock", "lightbulb"):
                    pr.on_desk = near
            props.append(pr)
        # ordenar: suelo primero, luego objetos, bocadillos al final
        order = {"desk": 0, "chair": 0}
        props.sort(key=lambda p: (order.get(p.kind, 1), 2 if p.kind in BUBBLES else 1))
        keywords = []
        kws = sorted(sc.keywords, key=lambda k: k.at)
        for ki, k in enumerate(kws):
            t0 = k.at * D
            t1 = k.until * D if k.until is not None else (kws[ki + 1].at * D if ki + 1 < len(kws) else D + 1)
            keywords.append((t0, t1, self._keyword_image(k.text, theme)))
        camera = self._camera_keys(sc, D)
        return SceneRuntime(idx, start, D, bg, theme, chars, props, keywords, camera, PropPainter(theme))

    def _camera_keys(self, sc, D: float) -> List[dict]:
        cam = sc.camera
        spk_x = 0.0
        for ch in sc.characters:
            if ch.id == sc.speaker:
                spk_x = ch.x
        if isinstance(cam, list):
            keys = [dict(t=k.at * D, zoom=k.zoom, x=k.x, y=k.y, ease=k.ease) for k in cam]
            return sorted(keys, key=lambda k: k["t"])
        k = lambda t, z, x=0.0, y=0.0, e="ease_in_out": dict(t=t, zoom=z, x=x, y=y, ease=e)  # noqa: E731
        if cam == "zoom_in":
            return [k(0, 1.0), k(D, 1.2, spk_x * 0.6, -0.35)]
        if cam == "zoom_out":
            return [k(0, 1.22, spk_x * 0.6, -0.35), k(D, 1.0)]
        if cam == "pan_left":
            return [k(0, 1.1, 0.18), k(D, 1.1, -0.18)]
        if cam == "pan_right":
            return [k(0, 1.1, -0.18), k(D, 1.1, 0.18)]
        if cam == "punch_in":
            tp = D * 0.45
            return [k(0, 1.0), k(tp, 1.0), k(tp + 0.3, 1.28, spk_x * 0.7, -0.45, "ease_out")]
        return [k(0, 1.0)]

    # ------------------------------------------------------------------ #
    def _char_frame(self, cr: CharRuntime, t: float, frame: int, D: float) -> CharacterFrame:
        L = self.L
        x, dist, moving = cr.x_at(t)
        cr.last_x = x
        joints = dict(cr.pose_track(t))
        idx, u = cr.pose_track.progress(t)
        cur_name = cr.pose_names[idx]
        # respiración / balanceo
        ph = cr.seed
        br = math.sin(2 * math.pi * t / 3.4 + ph)
        joints["spine"] += 1.1 * br
        joints["l_sh"] += 2.2 * math.sin(2 * math.pi * t / 3.1 + ph + 1)
        joints["r_sh"] -= 2.2 * math.sin(2 * math.pi * t / 2.9 + ph + 2)
        amp = float(self.env[frame]) if cr.speaker and frame < len(self.env) else 0.0
        if cr.speaker:
            sm = float(np.mean(self.env[max(0, frame - 8):frame + 1])) if frame < len(self.env) else 0.0
            if cur_name in ("talk", "present", "idle", "point"):
                joints["r_el"] += 14 * sm * math.sin(2 * math.pi * 1.25 * t + ph)
                joints["r_sh"] += 7 * sm * math.sin(2 * math.pi * 0.8 * t + ph + 1.3)
            if cur_name in ("talk", "shrug"):
                joints["l_el"] += 10 * sm * math.sin(2 * math.pi * 1.05 * t + ph + 2.1)
            joints["neck"] += 3.5 * sm * math.sin(2 * math.pi * 1.6 * t + ph)
        if cur_name == "wave":
            joints["r_el"] += 24 * math.sin(2 * math.pi * 2.3 * t) * u
        # andar
        lift = 0.0
        ww = cr.walk_weight(t)
        if ww > 0:
            phase = dist * L.stage_half / (L.scale * 150.0) * math.pi
            wj = walk_overlay(phase)
            joints = lerp_dict(joints, {**joints, **wj}, ww, angular=True)
            lift += abs(math.sin(phase)) * 7 * ww
        # salto
        for tj in cr.jumps:
            if tj <= t <= tj + JUMP_DUR:
                p = (t - tj) / JUMP_DUR
                if p < 0.25:
                    joints = lerp_dict(joints, JUMP_CROUCH, ease_in_out_cubic(p / 0.25), True)
                elif p < 0.7:
                    q = (p - 0.25) / 0.45
                    joints = lerp_dict(JUMP_CROUCH, JUMP_AIR, ease_out_cubic(min(1, q * 2.2)), True)
                    lift += 4 * 120 * q * (1 - q)
                else:
                    q = (p - 0.7) / 0.3
                    joints = lerp_dict(JUMP_CROUCH, joints, ease_in_out_cubic(q), True)
        # orientación
        facing = cr.facing_keys[0][1]
        for tk, fk in cr.facing_keys:
            if tk <= t:
                facing = fk
        if moving:
            facing = moving
        # expresión, parpadeo y boca
        face = cr.expr_track(t)
        k = int(t / 3.0)
        blink_t = k * 3.0 + 0.6 + 2.0 * hash01(cr.seed, k)
        eyes = 1.0
        if blink_t <= t <= blink_t + 0.16:
            eyes = abs((t - blink_t) / 0.08 - 1.0)
        mouth = min(0.85, amp * 0.8) if amp > 0.08 else 0.0
        pop = 1.0
        if cr.pop_in:
            pop = ease_out_back(clamp01(t / 0.4)) if t < 0.4 else 1.0
        seat = 0.0
        if cur_name == "sit":
            seat = u if idx > 0 else 1.0
        elif idx > 0 and cr.pose_names[idx - 1] == "sit":
            seat = 1 - u
        return CharacterFrame(x=L.W / 2 + x * L.stage_half, ground_y=L.ground_y, scale=L.scale,
                              facing=facing, joints=joints, face=face, lift=lift, mouth=mouth,
                              eyes_open=eyes, pop=pop, seat=seat)

    def _apply_camera(self, c: skia.Canvas, sc: SceneRuntime, t: float) -> None:
        L = self.L
        cam = interpolate_keyframes(sc.camera, t, ("zoom", "x", "y"))
        fx, fy = L.W / 2, L.ground_y - CHAR_H * L.scale * 0.55
        cx = fx + cam["x"] * L.stage_half
        cy = fy + cam["y"] * CHAR_H * L.scale * 0.5
        zoom = cam["zoom"]
        # Con zoom el suelo baja: se limita para que los pies nunca queden bajo los subtítulos.
        limit = L.caption_y - L.caption_size * 1.1
        if L.ground_y > cy and zoom > 1.0:
            zoom = min(zoom, max(1.0, (limit - fy) / (L.ground_y - cy)))
        c.translate(fx, fy)
        c.scale(zoom, zoom)
        c.translate(-cx, -cy)

    def _prop_alpha_scale(self, p: PropRuntime, t: float) -> Tuple[float, float, float]:
        if t < p.t0:
            return 0.0, 0.0, 0.0
        prog = clamp01((t - p.t0) / 0.38)
        sc = ease_out_back(prog)
        alpha = 1.0
        if p.t1 is not None and t > p.t1:
            q = clamp01((t - p.t1) / 0.25)
            alpha = 1 - q
            sc *= 1 - 0.3 * q
        return alpha, sc, clamp01((t - p.t0) / 1.0)

    def _draw_prop(self, c: skia.Canvas, sc: SceneRuntime, p: PropRuntime, t: float,
                   chars: Dict[str, Tuple[CharRuntime, CharacterFrame]]) -> None:
        L = self.L
        alpha, s, prog = self._prop_alpha_scale(p, t)
        if alpha <= 0.01 or s <= 0.01:
            return
        S = L.scale
        tail = 0.0
        holder = chars.get(p.holder) if p.holder else None
        if holder and p.kind in ("phone", "coffee", "money", "laptop"):
            cr, cf = holder
            hx, hy = cf.hand_points.get("r", (cf.x, L.ground_y - 300))
            anchor = {"phone": 40, "coffee": 30, "money": 40, "laptop": 60}[p.kind]
            px, py, k = hx, hy + anchor * S * 0.8, 0.8
        elif holder:
            cr, cf = holder
            hx, hy = cf.hand_points.get("head", (cf.x, L.ground_y - 500))
            if p.kind == "lightbulb":
                px, py, k = hx, hy - BONES["head_r"] * S - 24 * S, 1.0
            elif p.kind in BUBBLES:
                k = 1.0
                px = hx + cf.facing * 120 * S
                py = hy - 200 * S
                tail = hx - px
            else:
                k = 1.0
                px = hx + cf.facing * 250 * S
                py = hy + 70 * S
        elif p.on_desk is not None:
            k = 1.0
            px = L.W / 2 + (p.on_desk + max(-0.22, min(0.22, p.x - p.on_desk))) * L.stage_half
            py = L.ground_y - DESK_TOP * S
        elif p.kind in GROUND_PROPS:
            k = 1.0
            px, py = L.W / 2 + p.x * L.stage_half, L.ground_y
        else:
            k = 1.35 if p.kind not in BUBBLES else 1.0
            y = p.y if p.y is not None else (1.12 if p.kind in BUBBLES else 0.62)
            px = L.W / 2 + p.x * L.stage_half
            py = L.ground_y - y * CHAR_H * S + 6 * S * math.sin(t * 2.0 + p.x * 3)
            if p.kind not in BUBBLES:
                py += 60 * S * k  # la base del objeto queda por debajo del centro
        if p.kind in BUBBLES:
            # mantener dentro de la pantalla
            px = max(L.W * 0.2, min(L.W * 0.8, px))
        c.save()
        if alpha < 1:
            c.saveLayerAlpha(None, int(alpha * 255))
        c.translate(px, py)
        c.scale(S * k * s, S * k * s)
        sc.prop_painter.draw(c, p.kind, t, prog, p.text, p.trend, tail)
        if alpha < 1:
            c.restore()
        c.restore()

    def _caption_image(self, text: str) -> skia.Image:
        img = self._caption_cache.get(text)
        if img is None:
            L = self.L
            img = render_outlined(text, L.caption_size, L.caption_maxw, 2)
            if len(self._caption_cache) > 64:
                self._caption_cache.clear()
            self._caption_cache[text] = img
        return img

    # ------------------------------------------------------------------ #
    def draw_frame(self, c: skia.Canvas, frame: int) -> None:
        L = self.L
        gt = frame / self.fps
        si = 0
        for i, (f0, n) in enumerate(self.scene_frames):
            if frame >= f0:
                si = i
        sc = self.scenes[si]
        t = (frame - self.scene_frames[si][0]) / self.fps
        c.save()
        if self.rs != 1.0:
            c.scale(self.rs, self.rs)
        c.clear(hex_color(sc.bg))
        # escenario con cámara
        c.save()
        self._apply_camera(c, sc, t)
        gl = stroke_paint(hex_color(sc.theme.ink, 0.10), 4)
        c.drawLine(-L.W, L.ground_y + 2, L.W * 2, L.ground_y + 2, gl)
        frames: Dict[str, Tuple[CharRuntime, CharacterFrame]] = {}
        for cr in sc.chars:
            frames[cr.id] = (cr, self._char_frame(cr, t, frame, sc.duration))
        ground = [p for p in sc.props if p.kind in GROUND_PROPS]
        rest = [p for p in sc.props if p.kind not in GROUND_PROPS]
        for p in ground:
            self._draw_prop(c, sc, p, t, frames)
        # detrás de los personajes: objetos sobre la mesa e iconos flotantes
        behind = [p for p in rest if p.on_desk is not None or (not p.holder and p.kind not in BUBBLES)]
        for p in behind:
            self._draw_prop(c, sc, p, t, frames)
        for cr, cf in frames.values():
            cr.painter.draw(c, cf)
        # delante: objetos en la mano y bocadillos
        for p in rest:
            if p not in behind:
                self._draw_prop(c, sc, p, t, frames)
        c.restore()
        # textos clave (pantalla)
        for t0, t1, img in sc.keywords:
            if t0 <= t <= t1 + 0.25:
                a_in = clamp01((t - t0) / 0.3)
                a_out = 1 - clamp01((t - t1) / 0.25)
                s = 0.85 + 0.15 * ease_out_back(a_in)
                draw_image_centered(c, img, L.W / 2, L.keyword_y, s, min(a_in * 1.4, 1) * a_out)
        # subtítulos
        cue = None
        for q in self.cues:
            if q.start <= gt < q.end:
                cue = q
                break
        if cue:
            img = self._caption_image(cue.text)
            age = gt - cue.start
            s = 1.0 + 0.07 * (1 - ease_out_cubic(clamp01(age / 0.14)))
            draw_image_centered(c, img, L.W / 2, L.caption_y, s)
        c.restore()

    # ------------------------------------------------------------------ #
    def render(self, out_mp4: str, audio_wav: str, progress: Optional[Callable[[float], None]] = None,
               threads: int = 2, crf: int = 21, preset: str = "veryfast") -> float:
        W, H = self.out_w, self.out_h
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        surf = skia.Surface(arr)
        canvas = surf.getCanvas()
        pix = "bgra" if surf.imageInfo().colorType() == skia.ColorType.kBGRA_8888_ColorType else "rgba"
        vf = []
        if (W, H) != (self.L.W, self.L.H):
            vf = ["-vf", f"scale={self.L.W}:{self.L.H}:flags=lanczos"]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-thread_queue_size", "4", "-f", "rawvideo", "-pix_fmt", pix,
               "-s", f"{W}x{H}", "-r", str(self.fps), "-i", "-",
               "-i", audio_wav, *vf,
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-threads", str(threads), "-x264-params", "rc-lookahead=10:sync-lookahead=0",
               # sin -shortest: en ffmpeg 7+ hace que se acumulen fotogramas en memoria (>1 GB)
               "-c:a", "aac", "-b:a", "160k", "-t", f"{self.total_frames / self.fps:.3f}",
               "-movflags", "+faststart", out_mp4]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        t0 = time.time()
        try:
            for f in range(self.total_frames):
                self.draw_frame(canvas, f)
                proc.stdin.write(arr.data)
                if progress and (f % 15 == 0 or f == self.total_frames - 1):
                    progress((f + 1) / self.total_frames)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg falló ({code}): {err[-800:]}")
        return time.time() - t0

    def snapshot(self, frame: int, path: str) -> None:
        surf = skia.Surface(self.out_w, self.out_h)
        self.draw_frame(surf.getCanvas(), frame)
        surf.makeImageSnapshot().save(path)
