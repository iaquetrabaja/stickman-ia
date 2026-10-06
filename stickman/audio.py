"""Audio: WAV, remuestreo, envolvente de amplitud (boca), silencios, subtítulos y SRT.

La envolvente RMS por fotograma y el reparto de tiempos de palabra por número de
caracteres están portados de src/audio/amplitude.js y captionTiming.js de
stickman-explainers (MIT, Copyright (c) 2026 21cabbagee). Añadido: alineación
de frases con los silencios detectados.
"""
from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

SR = 24000  # frecuencia interna


def read_wav(path: str | Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("solo WAV PCM de 16 bits")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def write_wav(path: str | Path, samples: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def pcm16_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def resample(a: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out or len(a) == 0:
        return a.astype(np.float32)
    n_out = int(round(len(a) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, a).astype(np.float32)


def trim_silence(a: np.ndarray, sr: int = SR, thresh: float = 0.012, keep: float = 0.06) -> np.ndarray:
    if len(a) == 0:
        return a
    win = max(1, int(sr * 0.01))
    n = len(a) // win
    if n == 0:
        return a
    rms = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
    idx = np.where(rms > thresh)[0]
    if len(idx) == 0:
        return a
    k = int(keep * sr)
    s = max(0, idx[0] * win - k)
    e = min(len(a), (idx[-1] + 1) * win + k)
    return a[s:e]


def normalize(a: np.ndarray, peak: float = 0.89) -> np.ndarray:
    m = float(np.max(np.abs(a))) if len(a) else 0.0
    return a * (peak / m) if m > 1e-4 else a


def amplitude_envelope(samples: np.ndarray, sr: int, fps: int, total_frames: int) -> np.ndarray:
    """RMS por fotograma, normalizado a 0-1 y suavizado (ataque rápido, caída lenta)."""
    spf = sr / fps
    env = np.zeros(total_frames, dtype=np.float32)
    for f in range(total_frames):
        s = int(f * spf)
        e = min(len(samples), int((f + 1) * spf))
        if e > s:
            seg = samples[s:e]
            env[f] = float(np.sqrt(np.mean(seg * seg)))
    ref = float(np.percentile(env[env > 0.01], 90)) if np.any(env > 0.01) else 1.0
    env = np.clip(env / max(ref, 1e-4), 0, 1.3)
    out = np.zeros_like(env)
    v = 0.0
    for i, x in enumerate(env):
        v = x if x > v else v * 0.55 + x * 0.45
        out[i] = v
    return out


def silence_gaps(a: np.ndarray, sr: int = SR, thresh: float = 0.015, min_len: float = 0.12) -> List[Tuple[float, float]]:
    """Intervalos (inicio, fin) en segundos de silencios de al menos ``min_len``."""
    win = max(1, int(sr * 0.01))
    n = len(a) // win
    if n == 0:
        return []
    rms = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
    quiet = rms < thresh
    gaps = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= min_len and i > 0 and j < n:
                gaps.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return gaps


def split_by_gaps(a: np.ndarray, weights: Sequence[float], sr: int = SR) -> List[np.ndarray]:
    """Divide un audio continuo en N trozos usando los silencios más próximos al
    reparto proporcional esperado (``weights`` ~ nº de caracteres de cada escena)."""
    n = len(weights)
    if n == 1:
        return [a]
    dur = len(a) / sr
    total = float(sum(weights)) or 1.0
    expected, acc = [], 0.0
    for w in weights[:-1]:
        acc += w
        expected.append(dur * acc / total)
    gaps = silence_gaps(a, sr, min_len=0.18) or silence_gaps(a, sr, min_len=0.08)
    cuts: List[float] = []
    last = 0.0
    for k, e in enumerate(expected):
        best, best_d = None, None
        for g0, g1 in gaps:
            mid = (g0 + g1) / 2
            if mid <= last + 0.3:
                continue
            # penaliza silencios cortos frente a largos
            d = abs(mid - e) - min(0.6, (g1 - g0)) * 1.5
            if best_d is None or d < best_d:
                best, best_d = mid, d
        if best is None or abs(best - e) > max(2.5, dur * 0.12):
            best = max(e, last + 0.3)
        cuts.append(best)
        last = best
    pieces, prev = [], 0
    for cpos in cuts:
        idx = int(cpos * sr)
        pieces.append(a[prev:idx])
        prev = idx
    pieces.append(a[prev:])
    return pieces


# --------------------------------------------------------------------------- #
# Subtítulos
# --------------------------------------------------------------------------- #
@dataclass
class Cue:
    text: str
    start: float
    end: float


_PAUSE = re.compile(r"[,;:.!?…]$")


def word_timings(text: str, start: float, end: float, gaps: Sequence[Tuple[float, float]] = ()) -> List[Cue]:
    """Tiempos de palabra proporcionales a su longitud; las pausas de puntuación se
    alinean con silencios reales si los hay."""
    words = text.split()
    if not words:
        return []
    # dividir en frases por puntuación
    phrases: List[List[str]] = [[]]
    for w in words:
        phrases[-1].append(w)
        if _PAUSE.search(w):
            phrases.append([])
    phrases = [p for p in phrases if p]
    weight = lambda ws: sum(len(w) + 1.5 for w in ws)  # noqa: E731
    total = sum(weight(p) for p in phrases)
    dur = max(0.01, end - start)
    # límites esperados
    bounds = [start]
    acc = 0.0
    for p in phrases[:-1]:
        acc += weight(p)
        exp = start + dur * acc / total
        cand = [((g0 + g1) / 2, g0, g1) for g0, g1 in gaps if start < g0 and g1 < end]
        best = min(cand, key=lambda g: abs(g[0] - exp), default=None)
        if best and abs(best[0] - exp) < 0.9 and best[0] > bounds[-1] + 0.15:
            bounds.append(best[1])  # la frase termina al empezar el silencio
            bounds.append(best[2])  # la siguiente empieza al acabar
        else:
            bounds.append(exp)
            bounds.append(exp)
    bounds.append(end)
    cues: List[Cue] = []
    for i, p in enumerate(phrases):
        s, e = bounds[2 * i], bounds[2 * i + 1]
        tw = weight(p)
        cur = s
        for w in p:
            d = (e - s) * (len(w) + 1.5) / tw
            cues.append(Cue(w, cur, cur + d))
            cur += d
    return cues


_SENT_END = re.compile(r"[.!?…]+[»\"')]*$")
_SOFT = re.compile(r"[,;:—–]$")

CAPTION_LEAD = 0.08   # el subtítulo aparece un poco antes de que empiece la voz
CAPTION_HOLD = 0.15   # y se mantiene un poco tras la última palabra
CAPTION_BRIDGE = 0.35  # huecos más cortos que esto se cierran (sin parpadeo)


def _chunk_sentence(ws: List[Cue], max_words: int, max_chars: int, pause: float = 0.45) -> List[List[Cue]]:
    chunks: List[List[Cue]] = []
    cur: List[Cue] = []
    for w in ws:
        if cur:
            cand = " ".join(x.text for x in cur + [w])
            long_pause = w.start - cur[-1].end > pause
            if len(cur) >= max_words or len(cand) > max_chars or long_pause:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if _SOFT.search(w.text) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    # sin palabras huérfanas al final de la frase
    if len(chunks) >= 2 and len(chunks[-1]) == 1:
        prev, last = chunks[-2], chunks[-1]
        joined = " ".join(x.text for x in prev + last)
        if len(prev) + 1 <= max_words + 1 and len(joined) <= max_chars + 4:
            chunks[-2:] = [prev + last]
        elif len(prev) >= 3:
            chunks[-2:] = [prev[:-1], [prev[-1]] + last]
    return chunks


def group_captions(words: Sequence[Cue], max_words: int = 4, max_chars: int = 22,
                   lead: float = CAPTION_LEAD, hold: float = CAPTION_HOLD) -> List[Cue]:
    """Agrupa palabras cronometradas en subtítulos de 2-``max_words`` palabras sin cruzar
    frases. Inicio = primera palabra - ``lead``; fin = última palabra + ``hold``; sin solapes."""
    sentences: List[List[Cue]] = [[]]
    for w in words:
        sentences[-1].append(w)
        if _SENT_END.search(w.text):
            sentences.append([])
    out: List[Cue] = []
    for sent in sentences:
        for ch in _chunk_sentence(sent, max_words, max_chars):
            out.append(Cue(" ".join(x.text for x in ch), max(0.0, ch[0].start - lead), ch[-1].end + hold))
    for a, b in zip(out, out[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.05, b.start)
            b.start = max(b.start, a.end)
        elif b.start - a.end < CAPTION_BRIDGE:
            a.end = b.start
    return out


def _ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: Sequence[Cue]) -> str:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{_ts(c.start)} --> {_ts(c.end)}", c.text, ""]
    return "\n".join(lines)
