"""Texto con skia: fuentes empaquetadas (Roboto, Apache 2.0), ajuste de líneas y caché."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import skia

FONT_DIR = Path(__file__).parent / "assets" / "fonts"


@lru_cache(maxsize=8)
def typeface(weight: str = "Black") -> skia.Typeface:
    p = FONT_DIR / f"Roboto-{weight}.ttf"
    tf = skia.Typeface.MakeFromFile(str(p)) if p.exists() else None
    if tf is None:
        tf = skia.Typeface.MakeFromName("Arial", skia.FontStyle.Bold())
    return tf


def font(size: float, weight: str = "Black") -> skia.Font:
    f = skia.Font(typeface(weight), size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    return f


WORD_GAP = 0.30  # espacio entre palabras (en "em"), más ancho que el de la fuente


def text_width(f: skia.Font, s: str, spaced: bool = False) -> float:
    if not spaced:
        return f.measureText(s)
    words = s.split(" ")
    return sum(f.measureText(w) for w in words) + f.getSize() * WORD_GAP * (len(words) - 1)


def draw_spaced(c: skia.Canvas, s: str, x: float, y: float, f: skia.Font, paint: skia.Paint) -> None:
    for w in s.split(" "):
        c.drawString(w, x, y, f, paint)
        x += f.measureText(w) + f.getSize() * WORD_GAP


def wrap(f: skia.Font, text: str, max_w: float, max_lines: int = 3) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and text_width(f, cand) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return lines


def fit_font(text: str, size: float, max_w: float, max_lines: int, weight: str = "Black",
             min_size: float = 20) -> Tuple[skia.Font, List[str]]:
    s = size
    while True:
        f = font(s, weight)
        lines = wrap(f, text, max_w, 99)
        if (len(lines) <= max_lines and all(text_width(f, ln) <= max_w for ln in lines)) or s <= min_size:
            return f, lines[:max_lines] if len(lines) <= max_lines else wrap(f, text, max_w, max_lines)
        s *= 0.92


def render_outlined(text: str, size: float, max_w: float, max_lines: int = 2,
                    fill: int = skia.ColorWHITE, outline: int = skia.ColorBLACK,
                    outline_w: float = 0.0, weight: str = "Black", line_gap: float = 1.08,
                    shadow: bool = True) -> skia.Image:
    """Bloque de texto centrado con contorno, devuelto como imagen (para cachear)."""
    f, lines = fit_font(text, size, max_w, max_lines, weight)
    fs = f.getSize()
    ow = outline_w or fs * 0.095
    metrics = f.getMetrics()
    asc, desc = -metrics.fAscent, metrics.fDescent
    lh = fs * line_gap
    width = max(text_width(f, ln, True) for ln in lines) + ow * 2 + 12
    height = lh * (len(lines) - 1) + asc + desc + ow * 2 + 14
    surf = skia.Surface(int(width + 2), int(height + 2))
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    p_out = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=ow * 2,
                       StrokeJoin=skia.Paint.kRound_Join, Color=outline)
    p_fill = skia.Paint(AntiAlias=True, Color=fill)
    p_sh = skia.Paint(AntiAlias=True, Style=skia.Paint.kStrokeAndFill_Style, StrokeWidth=ow * 2,
                      StrokeJoin=skia.Paint.kRound_Join, Color=skia.ColorSetARGB(60, 0, 0, 0))
    y = ow + 6 + asc
    for ln in lines:
        x = (width - text_width(f, ln, True)) / 2
        if shadow:
            draw_spaced(c, ln, x, y + fs * 0.06, f, p_sh)
        draw_spaced(c, ln, x, y, f, p_out)
        draw_spaced(c, ln, x, y, f, p_fill)
        y += lh
    return surf.makeImageSnapshot()


def draw_image_centered(c: skia.Canvas, img: skia.Image, cx: float, cy: float, scale: float = 1.0,
                        alpha: float = 1.0) -> None:
    w, h = img.width() * scale, img.height() * scale
    paint: Optional[skia.Paint] = None
    if alpha < 0.999:
        paint = skia.Paint(AntiAlias=True)
        paint.setAlphaf(max(0.0, alpha))
    c.drawImageRect(img, skia.Rect.MakeXYWH(cx - w / 2, cy - h / 2, w, h),
                    skia.SamplingOptions(skia.FilterMode.kLinear), paint)


def draw_centered_lines(c: skia.Canvas, lines: List[str], f: skia.Font, cx: float, cy: float,
                        paint: skia.Paint, line_gap: float = 1.12) -> None:
    fs = f.getSize()
    m = f.getMetrics()
    asc, desc = -m.fAscent, m.fDescent
    total = fs * line_gap * (len(lines) - 1) + asc * 0.75
    y = cy - total / 2 + asc * 0.75
    for ln in lines:
        c.drawString(ln, cx - text_width(f, ln) / 2, y, f, paint)
        y += fs * line_gap
