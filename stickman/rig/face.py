"""Expresiones faciales paramétricas (interpolables entre sí)."""
from __future__ import annotations

from typing import Dict

# eye: escala de ojos, brow_in: ángulo de ceja (+ = parte interior arriba, triste/preocupado;
# - = interior abajo, enfado), brow_y: altura de cejas, curve: curvatura de boca (+ sonrisa),
# mw: ancho de boca, mo: apertura base de boca, mdx: desplazamiento lateral de boca,
# px/py: dirección de la mirada, blush: rubor, brow_tilt: asimetría (ceja levantada).
EXPRESSIONS: Dict[str, Dict[str, float]] = {
    "neutral":   dict(eye=1.0, brow_in=0, brow_y=0, curve=0.25, mw=22, mo=0.0, mdx=0, px=0.0, py=0.0, blush=0, brow_tilt=0),
    "happy":     dict(eye=0.92, brow_in=6, brow_y=-3, curve=1.0, mw=28, mo=0.18, mdx=0, px=0.0, py=0.0, blush=1, brow_tilt=0),
    "sad":       dict(eye=0.9, brow_in=22, brow_y=2, curve=-0.85, mw=20, mo=0.0, mdx=0, px=0.0, py=0.35, blush=0, brow_tilt=0),
    "surprised": dict(eye=1.28, brow_in=4, brow_y=-9, curve=0.0, mw=13, mo=0.55, mdx=0, px=0.0, py=-0.1, blush=0, brow_tilt=0),
    "thinking":  dict(eye=0.95, brow_in=0, brow_y=-2, curve=-0.15, mw=16, mo=0.0, mdx=6, px=0.55, py=-0.75, blush=0, brow_tilt=9),
    "angry":     dict(eye=0.95, brow_in=-26, brow_y=4, curve=-0.6, mw=22, mo=0.0, mdx=0, px=0.0, py=0.0, blush=0, brow_tilt=0),
}


def get_expression(name: str) -> Dict[str, float]:
    return EXPRESSIONS.get(name, EXPRESSIONS["neutral"])
