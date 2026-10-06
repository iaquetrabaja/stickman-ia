"""Dibujo vectorial del muñeco con skia (antialias, trazos redondeados).

Estilo propio: trazo de tinta uniforme, torso como "píldora" del color del
personaje, cabeza blanca con contorno, cara paramétrica y peinados simples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import skia

from .skeleton import BONES, LINE, forward, ground_offset


def hex_color(h: str, alpha: float = 1.0) -> int:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return skia.ColorSetARGB(int(round(alpha * 255)), r, g, b)


def luminance(h: str) -> float:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def stroke_paint(color: int, width: float) -> skia.Paint:
    return skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width,
                      StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join, Color=color)


def fill_paint(color: int) -> skia.Paint:
    return skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style, Color=color)


@dataclass
class Theme:
    ink: str = "#1F1D1B"
    head_fill: str = "#FFFFFF"
    blush: str = "#F08F7A"

    @staticmethod
    def for_background(bg: str) -> "Theme":
        if luminance(bg) < 0.35:
            return Theme(ink="#F3F0EA", head_fill="#2A3038", blush="#F08F7A")
        return Theme()


@dataclass
class CharacterLook:
    color: str = "#D24D1E"
    hair: str = "short"
    hair_color: str = "#22201E"
    accessory: str = "none"


@dataclass
class CharacterFrame:
    """Estado de un personaje en un fotograma (ya resuelto por la escena)."""
    x: float                      # px pantalla/mundo (pies)
    ground_y: float               # px
    scale: float                  # px por unidad de rig
    facing: int                   # +1 derecha, -1 izquierda
    joints: Dict[str, float]
    face: Dict[str, float]
    lift: float = 0.0             # unidades rig (salto)
    mouth: float = 0.0            # apertura extra por voz (0-1)
    eyes_open: float = 1.0        # parpadeo
    alpha: float = 1.0
    pop: float = 1.0              # escala de aparición
    seat: float = 0.0             # taburete (pose sentada), 0-1
    hand_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # salida


class CharacterPainter:
    def __init__(self, look: CharacterLook, theme: Theme):
        self.look = look
        self.theme = theme
        ink = hex_color(theme.ink)
        self.p_line = stroke_paint(ink, LINE)
        self.p_ink = fill_paint(ink)
        self.p_head = fill_paint(hex_color(theme.head_fill))
        self.p_body = stroke_paint(hex_color(look.color), 44)
        self.p_color_fill = fill_paint(hex_color(look.color))
        self.p_hair = fill_paint(hex_color(look.hair_color if theme.ink == Theme().ink else
                                           (look.hair_color if luminance(look.hair_color) > 0.25 else "#D9D4CC")))
        self.p_brow = stroke_paint(ink, 6.5)
        self.p_mouth = stroke_paint(ink, 6.5)
        self.p_white = fill_paint(hex_color("#FFFFFF"))
        self.p_blush = fill_paint(hex_color(theme.blush, 0.45))
        self.p_glasses = stroke_paint(ink, 5)
        self.p_shadow = fill_paint(hex_color("#000000", 0.10))
        self.p_tie = fill_paint(hex_color("#2B2F3A" if luminance(look.color) > 0.2 else "#E8E2D8"))

    # ------------------------------------------------------------------ #
    def draw(self, c: skia.Canvas, st: CharacterFrame) -> None:
        pts = forward(st.joints)
        dy = ground_offset(pts) - st.lift
        s = st.scale * st.pop
        # sombra en el suelo (se encoge al saltar)
        sh_w = 70 * st.scale * max(0.45, 1 - st.lift / 260)
        c.save()
        if st.alpha < 1:
            c.saveLayerAlpha(None, int(255 * st.alpha))
        c.drawOval(skia.Rect.MakeLTRB(st.x - sh_w, st.ground_y - 9 * st.scale,
                                      st.x + sh_w, st.ground_y + 9 * st.scale), self.p_shadow)
        c.translate(st.x, st.ground_y)
        c.scale(s * st.facing, s)
        c.translate(0, dy)
        P = lambda k: pts[k]  # noqa: E731

        def seg(a, b, paint=self.p_line):
            c.drawLine(a[0], a[1], b[0], b[1], paint)

        if st.seat > 0.02:
            gy = -dy  # el suelo en coordenadas locales
            sp = stroke_paint(self.p_line.getColor(), LINE)
            sp.setAlphaf(min(1.0, st.seat) * sp.getAlphaf())
            c.drawLine(-52, 26, 34, 26, sp)
            c.drawLine(-42, 26, -48, gy, sp)
            c.drawLine(24, 26, 30, gy, sp)
            c.drawLine(-44, gy - 46, 26, gy - 46, sp)
        # piernas y pies
        for side in ("l", "r"):
            path = skia.Path()
            path.moveTo(*P("hip"))
            path.lineTo(*P(f"{side}_knee"))
            path.lineTo(*P(f"{side}_ankle"))
            c.drawPath(path, self.p_line)
            ax, ay = P(f"{side}_ankle")
            seg((ax, ay), (ax + BONES["foot"], ay + 1))
        # pelo largo/coleta/moño detrás de la cabeza
        hx, hy = P("head")
        tilt = pts["head_angle"][0]
        c.save()
        c.translate(hx, hy)
        c.rotate(tilt)
        self._hair_back(c)
        c.restore()
        # torso
        hip, chest = P("hip"), P("chest")
        vx, vy = chest[0] - hip[0], chest[1] - hip[1]
        n = math.hypot(vx, vy) or 1
        ux, uy = vx / n, vy / n
        seg(P("chest"), P("head_base"))
        seg((hip[0] + ux * 4, hip[1] + uy * 4), (chest[0] - ux * 12, chest[1] - uy * 12), self.p_body)
        if self.look.accessory == "tie":
            self._tie(c, chest, ux, uy)
        # cabeza
        c.save()
        c.translate(hx, hy)
        c.rotate(tilt)
        self._head(c, st)
        c.restore()
        # brazos (delante del cuerpo)
        for side in ("l", "r"):
            path = skia.Path()
            path.moveTo(*P(f"{side}_shoulder"))
            path.lineTo(*P(f"{side}_elbow"))
            path.lineTo(*P(f"{side}_wrist"))
            c.drawPath(path, self.p_line)
            wx, wy = P(f"{side}_wrist")
            c.drawCircle(wx, wy, 9.5, self.p_ink)
            # posición en pantalla de las manos (para props sujetos)
            st.hand_points[side] = (st.x + st.facing * s * wx, st.ground_y + s * (wy + dy))
        st.hand_points["head"] = (st.x + st.facing * s * hx, st.ground_y + s * (hy + dy))
        if st.alpha < 1:
            c.restore()
        c.restore()

    # ------------------------------------------------------------------ #
    def _head(self, c: skia.Canvas, st: CharacterFrame) -> None:
        R = BONES["head_r"]
        f = st.face
        c.drawCircle(0, 0, R, self.p_head)
        c.drawCircle(0, 0, R, self.p_line)
        self._hair_front(c, R)
        fx = 9.0  # desplazamiento 3/4 hacia donde mira
        # ojos
        eye_s = f.get("eye", 1.0)
        open_ = max(0.08, st.eyes_open)
        ex, ey = 19.0, -2.0
        for sx in (-1, 1):
            cx = fx + sx * ex
            rx, ry = 6.6 * eye_s, 8.8 * eye_s * open_
            c.drawOval(skia.Rect.MakeLTRB(cx - rx, ey - ry, cx + rx, ey + ry), self.p_ink)
            if open_ > 0.5:
                hl = 2.3 * eye_s
                c.drawCircle(cx + 2.2 + f.get("px", 0) * 1.5, ey - 3.0 + f.get("py", 0) * 1.5, hl, self.p_white)
        # cejas
        by = -23.0 + f.get("brow_y", 0)
        ang = f.get("brow_in", 0)
        tilt = f.get("brow_tilt", 0)
        for sx in (-1, 1):
            cx = fx + sx * ex
            a = math.radians(ang + (tilt if sx == 1 else 0))
            half = 9.0
            # extremo exterior / interior
            ox, oy = cx + sx * half, by
            ix, iy = cx - sx * half, by
            # rotar alrededor del centro: interior sube si ang>0
            dyv = math.sin(a) * half
            c.drawLine(ox, oy + dyv, ix, iy - dyv - (tilt * 0.35 if sx == 1 else 0), self.p_brow)
        # rubor
        bl = f.get("blush", 0)
        if bl > 0.02:
            self.p_blush.setAlphaf(0.40 * min(1, bl))
            for sx in (-1, 1):
                cx = fx + sx * 31
                c.drawOval(skia.Rect.MakeLTRB(cx - 8, 10, cx + 8, 17), self.p_blush)
        # boca
        self._mouth(c, fx + f.get("mdx", 0), 25.0, f, st.mouth)
        # accesorios de cara/cabeza
        acc = self.look.accessory
        if acc == "glasses":
            for sx in (-1, 1):
                c.drawCircle(fx + sx * ex, ey, 13.5, self.p_glasses)
            c.drawLine(fx - ex + 13.5, ey - 1, fx + ex - 13.5, ey - 1, self.p_glasses)
            c.drawLine(fx + ex + 13.5, ey - 3, R - 4, ey - 6, self.p_glasses)
        elif acc == "cap":
            self._cap(c, R)
        elif acc == "headphones":
            self._headphones(c, R)
        elif acc == "bow":
            self._bow(c, R)

    def _mouth(self, c: skia.Canvas, mx: float, my: float, f: Dict[str, float], talk: float) -> None:
        curve = f.get("curve", 0.0)
        w = f.get("mw", 22.0)
        o = min(1.0, f.get("mo", 0.0) + talk)
        w = w * (1.0 - 0.25 * min(1.0, talk))
        hw = w / 2
        cy = curve * 11.0
        if o < 0.06:
            path = skia.Path()
            path.moveTo(mx - hw, my - cy * 0.5)
            path.quadTo(mx, my + cy, mx + hw, my - cy * 0.5)
            c.drawPath(path, self.p_mouth)
            return
        depth = 6 + o * 26
        path = skia.Path()
        path.moveTo(mx - hw, my - cy * 0.5)
        path.quadTo(mx, my + cy * 0.55, mx + hw, my - cy * 0.5)
        path.cubicTo(mx + hw, my + depth * 0.9, mx - hw, my + depth * 0.9, mx - hw, my - cy * 0.5)
        path.close()
        c.drawPath(path, self.p_ink)
        c.drawPath(path, self.p_mouth)

    # ------------------------------------------------------------------ #
    def _hair_front(self, c: skia.Canvas, R: float) -> None:
        h = self.look.hair
        if h in ("none",) or self.look.accessory == "cap" and h in ("short", "spiky", "curly"):
            return
        if h == "curly":
            for i in range(9):
                a = math.radians(192 + i * 19.5)
                c.drawCircle(math.cos(a) * R * 0.9, math.sin(a) * R * 0.9 - 2, R * 0.30, self.p_hair)
            return
        oval = skia.Rect.MakeLTRB(-R - 5, -R - 5, R + 5, R + 5)
        path = skia.Path()
        path.arcTo(oval, 188, 164, True)
        # flequillo de lado
        path.cubicTo(R * 0.55, -R * 0.62, R * 0.05, -R * 0.38, -R * 0.42, -R * 0.52)
        path.quadTo(-R * 0.8, -R * 0.45, -R - 4, -R * 0.12)
        path.close()
        c.drawPath(path, self.p_hair)
        if h == "spiky":
            path = skia.Path()
            for i in range(5):
                a0 = math.radians(205 + i * 27)
                am = math.radians(205 + i * 27 + 13)
                a1 = math.radians(205 + i * 27 + 27)
                path.moveTo(math.cos(a0) * (R + 2), math.sin(a0) * (R + 2))
                path.lineTo(math.cos(am) * (R + 24), math.sin(am) * (R + 24))
                path.lineTo(math.cos(a1) * (R + 2), math.sin(a1) * (R + 2))
                path.close()
            c.drawPath(path, self.p_hair)

    def _hair_back(self, c: skia.Canvas) -> None:
        R = BONES["head_r"]
        h = self.look.hair
        if h == "long":
            path = skia.Path()
            path.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-R - 9, -R - 6, R + 9, R * 1.45), R, R * 0.6))
            c.drawPath(path, self.p_hair)
        elif h == "bun":
            c.drawCircle(-R * 0.25, -R - R * 0.22, R * 0.38, self.p_hair)
        elif h == "ponytail":
            c.save()
            c.translate(-R * 0.95, -R * 0.15)
            c.rotate(25)
            c.drawOval(skia.Rect.MakeLTRB(-R * 0.32, -R * 0.2, R * 0.32, R * 0.95), self.p_hair)
            c.restore()

    def _cap(self, c: skia.Canvas, R: float) -> None:
        oval = skia.Rect.MakeLTRB(-R - 4, -R - 6, R + 4, R + 2)
        path = skia.Path()
        path.arcTo(oval, 182, 176, True)
        path.quadTo(0, -R * 0.42, -R - 3, -R * 0.24)
        path.close()
        c.drawPath(path, self.p_color_fill)
        c.drawPath(path, self.p_glasses)
        brim = skia.Path()
        brim.moveTo(R * 0.2, -R * 0.36)
        brim.quadTo(R * 1.1, -R * 0.42, R * 1.42, -R * 0.2)
        brim.quadTo(R * 1.0, -R * 0.12, R * 0.2, -R * 0.2)
        brim.close()
        c.drawPath(brim, self.p_color_fill)
        c.drawPath(brim, self.p_glasses)

    def _headphones(self, c: skia.Canvas, R: float) -> None:
        oval = skia.Rect.MakeLTRB(-R - 9, -R - 11, R + 9, R + 7)
        band = skia.Path()
        band.arcTo(oval, 190, 160, True)
        c.drawPath(band, stroke_paint(self.p_line.getColor(), 9))
        for sx in (-1, 1):
            r = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(sx * (R + 2) - 10, -20, sx * (R + 2) + 10, 18), 8, 8)
            c.drawRRect(r, self.p_color_fill)
            c.drawRRect(r, self.p_glasses)

    def _bow(self, c: skia.Canvas, R: float) -> None:
        c.save()
        c.translate(-R * 0.45, -R * 0.9)
        c.rotate(-20)
        for sx in (-1, 1):
            p = skia.Path()
            p.moveTo(0, 0)
            p.lineTo(sx * 20, -12)
            p.lineTo(sx * 20, 12)
            p.close()
            c.drawPath(p, self.p_color_fill)
            c.drawPath(p, self.p_glasses)
        c.drawCircle(0, 0, 6, self.p_color_fill)
        c.drawCircle(0, 0, 6, self.p_glasses)
        c.restore()

    def _tie(self, c: skia.Canvas, chest, ux: float, uy: float) -> None:
        # perpendicular
        px, py = -uy, ux
        top = (chest[0] - ux * 22, chest[1] - uy * 22)
        p = skia.Path()
        p.moveTo(top[0] + px * 6, top[1] + py * 6)
        p.lineTo(top[0] - px * 6, top[1] - py * 6)
        p.lineTo(top[0] - ux * 62 - px * 9, top[1] - uy * 62 - py * 9)
        p.lineTo(top[0] - ux * 74, top[1] - uy * 74)
        p.lineTo(top[0] - ux * 62 + px * 9, top[1] - uy * 62 + py * 9)
        p.close()
        c.drawPath(p, self.p_tie)
