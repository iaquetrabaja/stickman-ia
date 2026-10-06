"""Props vectoriales propios (dibujados a mano en código, estilo plano).

Cada función dibuja en coordenadas locales (unidades de rig) con la base del
objeto en y=0 y centrado en x=0, salvo los bocadillos (centrados).
"""
from __future__ import annotations

import math
from typing import Optional

import skia

from .rig.draw import Theme, fill_paint, hex_color, stroke_paint
from .text import draw_centered_lines, fit_font, text_width

PLW = 8.0  # grosor de trazo de props

WOOD = "#C9A27C"
SCREEN = "#2B2F3A"
SCREEN_GLOW = "#7FB7E6"
PAPER = "#FFFFFF"
GREEN = "#7DBF8E"
YELLOW = "#F7C948"
GRAY = "#CFC8BC"
ACCENT = "#D24D1E"


class PropPainter:
    def __init__(self, theme: Theme, currency: str = "€"):
        self.theme = theme
        ink = hex_color(theme.ink)
        self.ink = ink
        self.line = stroke_paint(ink, PLW)
        self.thin = stroke_paint(ink, PLW * 0.7)
        self.fill_ink = fill_paint(ink)
        self.text_paint = fill_paint(hex_color(Theme().ink))  # texto sobre rellenos claros
        self.currency = currency
        self._fills = {}

    def fill(self, hx: str, alpha: float = 1.0) -> skia.Paint:
        key = (hx, round(alpha, 3))
        p = self._fills.get(key)
        if p is None:
            p = fill_paint(hex_color(hx, alpha))
            self._fills[key] = p
        return p

    def shape(self, c: skia.Canvas, path_or_rrect, fill_hex: Optional[str], line: bool = True) -> None:
        draw = c.drawRRect if isinstance(path_or_rrect, skia.RRect) else (
            c.drawRect if isinstance(path_or_rrect, skia.Rect) else c.drawPath)
        if fill_hex:
            draw(path_or_rrect, self.fill(fill_hex))
        if line:
            draw(path_or_rrect, self.line)

    # ------------------------------------------------------------------ #
    def draw(self, c: skia.Canvas, kind: str, t: float, prog: float, text: Optional[str] = None,
             trend: str = "up", tail_dx: float = 0.0) -> None:
        fn = getattr(self, f"_{kind}", None)
        if fn is None:
            return
        fn(c, t=t, prog=prog, text=text, trend=trend, tail_dx=tail_dx)

    def _laptop(self, c, t, **_):
        # pantalla (vista 3/4 sencilla)
        scr = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-62, -96, 52, -14), 8, 8)
        self.shape(c, scr, SCREEN)
        glow = self.fill(SCREEN_GLOW, 0.9)
        for i, w in enumerate((60, 44, 70)):
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-46, -80 + i * 18, w, 8), 4, 4), glow)
        base = skia.Path()
        base.moveTo(-74, -14)
        base.lineTo(64, -14)
        base.lineTo(78, 0)
        base.lineTo(-84, 0)
        base.close()
        self.shape(c, base, GRAY)

    def _phone(self, c, t, **_):
        body = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-19, -66, 19, 0), 8, 8)
        self.shape(c, body, SCREEN)
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-12, -58, 12, -10), 4, 4), self.fill(SCREEN_GLOW))

    def _coffee(self, c, t, **_):
        handle = skia.Path()
        handle.addCircle(24, -26, 12)
        c.drawPath(handle, self.line)
        cup = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-24, -52, 26, 0), 8, 8)
        self.shape(c, cup, PAPER)
        c.drawRect(skia.Rect.MakeLTRB(-20, -34, 22, -24), self.fill(ACCENT))
        steam = stroke_paint(self.ink, 5)
        for i, x in enumerate((-10, 8)):
            ph = t * 2.2 + i * 1.7
            a = 0.25 + 0.35 * (0.5 + 0.5 * math.sin(ph))
            steam.setAlphaf(a)
            p = skia.Path()
            p.moveTo(x, -62)
            p.cubicTo(x + 8, -72, x - 8, -82, x + 2, -94 - 4 * math.sin(ph))
            c.drawPath(p, steam)

    def _desk(self, c, t, **_):
        for x in (-92, 92):
            c.drawLine(x, -140, x, 0, self.line)
        top = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-118, -156, 118, -138), 5, 5)
        self.shape(c, top, WOOD)

    def _chair(self, c, t, **_):
        c.drawLine(-36, -100, 36, -100, self.line)
        c.drawLine(-30, -100, -36, 0, self.line)
        c.drawLine(30, -100, 36, 0, self.line)
        c.drawLine(-30, -100, -34, -190, self.line)

    def _chart(self, c, t, prog, trend="up", **_):
        card = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-100, -170, 100, 0), 14, 14)
        self.shape(c, card, PAPER)
        c.drawLine(-74, -30, 76, -30, self.thin)
        vals = [0.35, 0.55, 0.7, 1.0] if trend == "up" else [1.0, 0.72, 0.5, 0.3]
        g = min(1.0, prog * 1.4)
        for i, v in enumerate(vals):
            gi = max(0.0, min(1.0, g * 4 - i * 0.6))
            gi = 1 - (1 - gi) ** 3
            h = 100 * v * gi
            x = -66 + i * 36
            col = ACCENT if (i == 3 and trend == "up") or (i == 0 and trend == "down") else "#9C958B"
            if h > 1:
                c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x, -30 - h, x + 24, -30), 4, 4),
                            self.fill(col))
        # flecha de tendencia
        if g >= 0.95:
            ap = stroke_paint(hex_color(ACCENT), 7)
            if trend == "up":
                c.drawLine(-70, -70, 50, -150, ap)
                p = skia.Path()
                p.moveTo(64, -158)
                p.lineTo(38, -158)
                p.lineTo(56, -134)
            else:
                c.drawLine(-70, -150, 50, -70, ap)
                p = skia.Path()
                p.moveTo(64, -62)
                p.lineTo(38, -62)
                p.lineTo(56, -86)
            p.close()
            c.drawPath(p, self.fill(ACCENT))

    def _money(self, c, t, **_):
        for i, (dx, rot) in enumerate(((-10, -8), (8, 6))):
            c.save()
            c.translate(dx, -34 - i * 8)
            c.rotate(rot)
            bill = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-60, -30, 60, 30), 6, 6)
            self.shape(c, bill, GREEN)
            c.drawCircle(0, 0, 17, self.thin)
            if i == 1:
                f = fit_font(self.currency, 26, 40, 1, "Black")[0]
                draw_centered_lines(c, [self.currency], f, 0, 1, self.text_paint)
            c.restore()

    def _clock(self, c, t, **_):
        cy = -62
        c.drawCircle(0, cy, 58, self.fill(PAPER))
        c.drawCircle(0, cy, 58, self.line)
        for i in range(12):
            a = math.radians(i * 30)
            r0 = 44 if i % 3 else 40
            c.drawLine(math.cos(a) * r0, cy + math.sin(a) * r0, math.cos(a) * 49, cy + math.sin(a) * 49, self.thin)
        am = math.radians(-90 + t * 160)
        ah = math.radians(-90 + 120 + t * 14)
        c.drawLine(0, cy, math.cos(am) * 38, cy + math.sin(am) * 38, self.thin)
        c.drawLine(0, cy, math.cos(ah) * 24, cy + math.sin(ah) * 24, self.line)
        c.drawCircle(0, cy, 6, self.fill(ACCENT))

    def _lightbulb(self, c, t, prog, **_):
        cy = -98
        on = min(1.0, prog * 2.5)
        if on > 0.05:
            rp = stroke_paint(hex_color(YELLOW if self.theme.ink != Theme().ink else "#E2A72E"), 7)
            rp.setAlphaf(on)
            pulse = 1 + 0.06 * math.sin(t * 6)
            for i in range(7):
                a = math.radians(-180 + i * 30)
                r0, r1 = 58 * pulse, 80 * pulse
                c.drawLine(math.cos(a) * r0, cy + math.sin(a) * r0, math.cos(a) * r1, cy + math.sin(a) * r1, rp)
        bulb = skia.Path()
        bulb.addCircle(0, cy, 44)
        c.drawPath(bulb, self.fill(YELLOW if on > 0.5 else PAPER))
        neck = skia.Path()
        neck.moveTo(-22, cy + 38)
        neck.lineTo(22, cy + 38)
        neck.lineTo(18, cy + 52)
        neck.lineTo(-18, cy + 52)
        neck.close()
        c.drawPath(neck, self.fill(YELLOW if on > 0.5 else PAPER))
        c.drawCircle(0, cy, 44, self.line)
        base = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-20, cy + 50, 20, cy + 76), 6, 6)
        self.shape(c, base, GRAY)
        c.drawLine(-19, cy + 63, 19, cy + 63, self.thin)
        fp = stroke_paint(self.ink, 5)
        p = skia.Path()
        p.moveTo(-10, cy + 38)
        p.lineTo(-10, cy + 6)
        p.quadTo(0, cy - 8, 10, cy + 6)
        p.lineTo(10, cy + 38)
        c.drawPath(p, fp)

    # bocadillos: centrados en (0,0); tail_dx = dirección de la cola (hacia el personaje)
    def _bubble_box(self, text: Optional[str]):
        txt = text or "…"
        f, lines = fit_font(txt, 46, 300, 2, "Bold", 24)
        w = max(110, max(text_width(f, ln) for ln in lines) + 56)
        h = f.getSize() * 1.12 * len(lines) + 44
        return f, lines, w, h

    def _speech_bubble(self, c, t, text=None, tail_dx=0.0, **_):
        f, lines, w, h = self._bubble_box(text)
        body = skia.Path()
        body.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-w / 2, -h / 2, w / 2, h / 2), 26, 26))
        tail = skia.Path()
        sx = -1 if tail_dx < 0 else 1
        bx = sx * w * 0.16
        tail.moveTo(bx - 18, h / 2 - 6)
        tail.lineTo(bx + sx * 34, h / 2 + 40)
        tail.lineTo(bx + 18, h / 2 - 6)
        tail.close()
        full = skia.Op(body, tail, skia.PathOp.kUnion_PathOp) or body
        self.shape(c, full, PAPER)
        draw_centered_lines(c, lines, f, 0, 0, self.text_paint)

    def _thought_bubble(self, c, t, text=None, tail_dx=0.0, **_):
        f, lines, w, h = self._bubble_box(text)
        body = skia.Path()
        body.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-w / 2, -h / 2, w / 2, h / 2), h / 2, h / 2))
        n = max(4, int(w / 52))
        for i in range(n):
            x = -w / 2 + 30 + i * (w - 60) / max(1, n - 1)
            body = skia.Op(body, skia.Path.Circle(x, -h / 2 + 6, 24), skia.PathOp.kUnion_PathOp) or body
            body = skia.Op(body, skia.Path.Circle(x + 14, h / 2 - 6, 22), skia.PathOp.kUnion_PathOp) or body
        self.shape(c, body, PAPER)
        sx = -1 if tail_dx < 0 else 1
        for i, r in enumerate((13, 8)):
            cx, cy = sx * (w * 0.15 + 18 * i), h / 2 + 30 + i * 30
            c.drawCircle(cx, cy, r, self.fill(PAPER))
            c.drawCircle(cx, cy, r, self.thin)
        draw_centered_lines(c, lines, f, 0, 0, self.text_paint)


BUBBLES = ("speech_bubble", "thought_bubble")
GROUND_PROPS = ("desk", "chair")
DESK_TOP = 156.0
