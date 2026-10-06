"""Orquestación: tema -> guion (Gemini) -> voz -> render -> MP4 + SRT."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from . import gemini
from . import align
from .audio import (SR, Cue, amplitude_envelope, group_captions, pcm16_to_float, silence_gaps, split_by_gaps,
                    to_srt, word_timings, write_wav)
from .prompt import build_prompt, build_repair_prompt
from .render import FORMATS, VideoRenderer
from .schema import Script, ScriptError, parse_text, validate_script
from .tts import synthesize

MAX_SECONDS = 120
LEAD = 0.15
TAIL = 0.35

Log = Callable[[str], None]


@dataclass
class Stats:
    api_calls: dict = field(default_factory=lambda: {"list_models": 0, "script": 0, "tts": 0})
    tts_engine: str = ""
    duration: float = 0.0
    render_seconds: float = 0.0
    align_seconds: float = 0.0
    caption_timing: str = ""
    total_seconds: float = 0.0
    frames: int = 0


def write_script(key: str, topic: str, language: str = "es", duration: int = 45, fmt: str = "9:16",
                 model: Optional[str] = None, log: Log = print, stats: Optional[Stats] = None):
    """Pide el guion a Gemini y lo valida (con un reintento de reparación)."""
    stats = stats or Stats()
    if not model:
        stats.api_calls["list_models"] += 1
        models = gemini.list_models(key)
        model = models.best_text
        if not model:
            raise gemini.GeminiError("Tu clave no tiene ningún modelo 'flash' de texto disponible.")
    log(f"Escribiendo guion con {model}...")
    prompt = build_prompt(topic, language, duration, fmt)
    stats.api_calls["script"] += 1
    text = gemini.generate_json(key, model, prompt)
    try:
        script, warnings = validate_script(parse_text(text))
    except ScriptError as e:
        log("El guion tenía errores, pido una corrección...")
        stats.api_calls["script"] += 1
        text = gemini.generate_json(key, model, build_repair_prompt(text, e.errors), temperature=0.3)
        script, warnings = validate_script(parse_text(text))
    for w in warnings:
        log(f"aviso: {w}")
    return script, warnings, model


def render_script(script: Script, out_dir: str | Path, fmt: str = "9:16", engine: str = "auto",
                  key: Optional[str] = None, tts_model: Optional[str] = None, voice: str = "Puck",
                  fps: int = 30, render_scale: float = 1.0, max_seconds: float = MAX_SECONDS,
                  log: Log = print, progress: Optional[Callable[[str, float], None]] = None,
                  basename: str = "video", stats: Optional[Stats] = None, narration: Optional[str] = None,
                  word_align: bool = True) -> dict:
    """Voz -> línea de tiempo -> subtítulos alineados -> render.

    ``narration``: archivo de audio (WAV, MP4...) con la narración ya grabada; se usa en lugar
    de sintetizar la voz (se trocea por escenas con la alineación). ``word_align``: subtítulos
    con tiempos reales por palabra (faster-whisper); si no, estimados por longitud."""
    stats = stats or Stats()
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt not in FORMATS:
        raise ValueError(f"formato no soportado: {fmt} (usa 9:16 o 16:9)")
    prog = progress or (lambda stage, p: None)
    # 1) voz
    prog("voz", 0.0)
    texts = [s.narration for s in script.scenes]
    if engine in ("auto", "gemini") and key:
        stats.api_calls["tts"] += 1
        if not tts_model and stats.api_calls["list_models"] == 0:
            stats.api_calls["list_models"] += 1  # (en caché si ya se pidió para el guion)
    lead, tail, end_pad, min_d = LEAD, TAIL, 0.6, 2.2
    if narration:
        # la narración se coloca tal cual (ya trae sus pausas): el vídeo dura lo mismo que el audio
        audios = load_narration(narration, texts, script.language, log, fps=fps)
        used = f"audio:{Path(narration).name}"
        lead, tail, end_pad, min_d = 0.0, 0.0, 0.0, 0.0
    else:
        audios, used = synthesize(texts, script.language, engine, key, tts_model, voice, log)
    stats.tts_engine = used
    log(f"Voz: {used}")
    prog("voz", 1.0)
    # 2) línea de tiempo
    durs: List[float] = []
    for i, a in enumerate(audios):
        d = lead + len(a) / SR + tail
        if i == len(audios) - 1:
            d += end_pad
        durs.append(max(min_d, d))
    total = sum(durs)
    if total > max_seconds + 5:
        raise ValueError(f"El vídeo saldría de {total:.0f} s y el máximo es {max_seconds:.0f} s. "
                         "Acorta la narración.")
    fps_durs = [round(d * fps) / fps for d in durs]
    track = np.zeros(int(sum(fps_durs) * SR) + SR, dtype=np.float32)
    L = FORMATS[fmt]
    t0 = 0.0
    ranges = []
    for a, d in zip(audios, fps_durs):
        s = int((t0 + lead) * SR)
        track[s:s + len(a)] = a[: max(0, len(track) - s)]
        v0, v1 = _voiced_extent(a)
        ranges.append((t0 + lead + v0, t0 + lead + v1))
        t0 += d
    track = track[: int(t0 * SR)]
    # 3) subtítulos: tiempos reales por palabra (ASR + guion) o, si no se puede, estimados
    prog("subtitulos", 0.0)
    ta = time.time()
    cues, stats.caption_timing = timed_captions(texts, ranges, track, script.language, L.caption_words,
                                                L.caption_chars, word_align, log)
    stats.align_seconds = time.time() - ta
    prog("subtitulos", 1.0)
    wav = out / f"{basename}.wav"
    write_wav(wav, track)
    total_frames = int(round(t0 * fps))
    env = amplitude_envelope(track, SR, fps, total_frames)
    srt = out / f"{basename}.srt"
    srt.write_text(to_srt(cues), encoding="utf-8")
    (out / f"{basename}.json").write_text(script.to_json(), encoding="utf-8")
    # 4) render
    log(f"Renderizando {total_frames} fotogramas ({t0:.1f} s, {fmt}, {fps} fps)...")
    r = VideoRenderer(script, fmt, fps_durs, env, cues, fps=fps, render_scale=render_scale)
    mp4 = out / f"{basename}.mp4"
    stats.render_seconds = r.render(str(mp4), str(wav), progress=lambda p: prog("render", p))
    try:
        wav.unlink()
    except OSError:
        pass
    stats.duration = t0
    stats.frames = total_frames
    stats.total_seconds = time.time() - t_start
    log(f"Listo: {mp4} ({t0:.1f} s de vídeo, subtítulos {stats.caption_timing} en {stats.align_seconds:.1f} s, "
        f"render {stats.render_seconds:.1f} s, total {stats.total_seconds:.1f} s)")
    return {"mp4": str(mp4), "srt": str(srt), "script": str(out / f"{basename}.json"),
            "duration": t0, "stats": stats, "renderer": r}


def timed_captions(texts: List[str], ranges: List[tuple], track: np.ndarray, language: str = "es",
                   max_words: int = 4, max_chars: int = 22, word_align: bool = True,
                   log: Log = print) -> tuple:
    """Subtítulos de toda la pista. ``ranges``: (inicio, fin) de la voz de cada escena en ``track``.
    Devuelve (cues, "asr" | "estimado"). Los subtítulos nunca cruzan de una escena a otra."""
    aligned = align.align_script(texts, ranges, track, SR, language, log) if word_align else None
    cues: List[Cue] = []
    for i, text in enumerate(texts):
        lo, hi = ranges[i]
        if aligned:
            words = [Cue(w.text, w.start, w.end) for w in aligned[i]]
        else:
            seg = track[int(lo * SR):int(hi * SR)]
            gaps = [(lo + g0, lo + g1) for g0, g1 in silence_gaps(seg)]
            words = word_timings(text, lo, hi, gaps)
        cues += group_captions(words, max_words, max_chars)
    return cues, ("asr" if aligned else "estimado")


def decode_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Cualquier audio/vídeo -> float32 mono a ``sr`` Hz (con ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return pcm16_to_float(raw)


def _voiced_extent(a: np.ndarray) -> tuple:
    """(inicio, fin) en segundos de la parte con voz de un audio."""
    win = int(SR * 0.01)
    n = len(a) // win
    if n == 0:
        return 0.0, len(a) / SR
    rms = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
    idx = np.nonzero(rms > 0.012)[0]
    if len(idx) == 0:
        return 0.0, len(a) / SR
    return idx[0] * 0.01, min(len(a) / SR, (idx[-1] + 1) * 0.01)


def load_narration(path: str | Path, texts: List[str], language: str = "es", log: Log = print,
                   fps: int = 30) -> List[np.ndarray]:
    """Trocea una narración ya grabada en un audio por escena, SIN recortarla: corta dentro del
    silencio entre la última palabra de una escena y la primera de la siguiente (alineación con
    el guion; sin ASR, en los silencios más próximos al reparto por longitud). Los cortes caen
    en fotogramas exactos para que el audio no se desplace."""
    a = decode_audio(path)
    dur = len(a) / SR
    log(f"Narración: {Path(path).name} ({dur:.1f} s)")
    pieces = None
    if len(texts) > 1:
        words = align.align_script(texts, [(0.0, dur)] * len(texts), a, SR, language, log)
        if words and all(words):
            cuts = []
            for prev, nxt in zip(words, words[1:]):
                e, s = prev[-1].end, nxt[0].start
                gap = [(g0, g1) for g0, g1 in silence_gaps(a) if g0 < s + 0.2 and g1 > e - 0.2]
                g = max(gap, key=lambda x: x[1] - x[0]) if gap else (e, s)
                # el corte, más cerca del final del silencio (la escena nueva arranca justo antes de hablar)
                cuts.append(g[0] + 0.7 * (g[1] - g[0]))
            if all(c2 > c1 for c1, c2 in zip(cuts, cuts[1:])):
                pieces = cuts
    if pieces is None:
        log("Narración: troceo por silencios (sin alineación).")
        acc, pieces = 0.0, []
        for p in split_by_gaps(a, [len(t) for t in texts])[:-1]:
            acc += len(p) / SR
            pieces.append(acc)
    idx = [0] + [int(round(round(c * fps) / fps * SR)) for c in pieces]
    end = int(np.ceil(len(a) / SR * fps) / fps * SR)
    a = np.concatenate([a, np.zeros(max(0, end - len(a)), np.float32)])
    idx.append(len(a))
    return [a[i:j] for i, j in zip(idx, idx[1:])]


def script_from_file(path: str | Path) -> Script:
    script, warnings = validate_script(parse_text(Path(path).read_text(encoding="utf-8")))
    for w in warnings:
        print(f"aviso: {w}")
    return script


def dump(script: Script, path: str | Path) -> None:
    Path(path).write_text(json.dumps(json.loads(script.to_json()), ensure_ascii=False, indent=2), encoding="utf-8")
