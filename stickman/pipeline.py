"""Orquestación: tema -> guion (Gemini) -> voz -> render -> MP4 + SRT."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from . import gemini
from .audio import SR, Cue, amplitude_envelope, group_captions, silence_gaps, to_srt, word_timings, write_wav
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
                  basename: str = "video", stats: Optional[Stats] = None) -> dict:
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
    audios, used = synthesize(texts, script.language, engine, key, tts_model, voice, log)
    stats.tts_engine = used
    log(f"Voz: {used}")
    prog("voz", 1.0)
    # 2) línea de tiempo
    durs: List[float] = []
    for i, a in enumerate(audios):
        d = LEAD + len(a) / SR + TAIL
        if i == len(audios) - 1:
            d += 0.6
        durs.append(max(2.2, d))
    total = sum(durs)
    if total > max_seconds + 5:
        raise ValueError(f"El vídeo saldría de {total:.0f} s y el máximo es {max_seconds:.0f} s. "
                         "Acorta la narración.")
    fps_durs = [round(d * fps) / fps for d in durs]
    track = np.zeros(int(sum(fps_durs) * SR) + SR, dtype=np.float32)
    cues: List[Cue] = []
    L = FORMATS[fmt]
    t0 = 0.0
    for sc, a, d in zip(script.scenes, audios, fps_durs):
        s = int((t0 + LEAD) * SR)
        track[s:s + len(a)] = a[: max(0, len(track) - s)]
        gaps = [(t0 + LEAD + g0, t0 + LEAD + g1) for g0, g1 in silence_gaps(a)]
        words = word_timings(sc.narration, t0 + LEAD, t0 + LEAD + len(a) / SR, gaps)
        cues += group_captions(words, L.caption_words, L.caption_chars)
        t0 += d
    track = track[: int(t0 * SR)]
    wav = out / f"{basename}.wav"
    write_wav(wav, track)
    total_frames = int(round(t0 * fps))
    env = amplitude_envelope(track, SR, fps, total_frames)
    srt = out / f"{basename}.srt"
    srt.write_text(to_srt(cues), encoding="utf-8")
    (out / f"{basename}.json").write_text(script.to_json(), encoding="utf-8")
    # 3) render
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
    log(f"Listo: {mp4} ({t0:.1f} s de vídeo, render {stats.render_seconds:.1f} s, total {stats.total_seconds:.1f} s)")
    return {"mp4": str(mp4), "srt": str(srt), "script": str(out / f"{basename}.json"),
            "duration": t0, "stats": stats, "renderer": r}


def script_from_file(path: str | Path) -> Script:
    script, warnings = validate_script(parse_text(Path(path).read_text(encoding="utf-8")))
    for w in warnings:
        print(f"aviso: {w}")
    return script


def dump(script: Script, path: str | Path) -> None:
    Path(path).write_text(json.dumps(json.loads(script.to_json()), ensure_ascii=False, indent=2), encoding="utf-8")
