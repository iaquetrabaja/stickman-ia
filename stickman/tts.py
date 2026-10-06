"""Síntesis de voz: Gemini TTS (con clave) o Piper local (descarga automática).

Con Gemini se hace UNA sola llamada para toda la narración (ahorra cuota) y se
trocea por escenas usando los silencios. Si falla (cuota, red...), se usa Piper.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import gemini
from .audio import SR, normalize, pcm16_to_float, read_wav, resample, split_by_gaps, trim_silence

PIPER_VOICES = {
    "es": "es_ES-davefx-medium",
    "en": "en_US-lessac-medium",
    "pt": "pt_BR-faber-medium",
    "fr": "fr_FR-siwis-medium",
    "it": "it_IT-paola-medium",
    "de": "de_DE-thorsten-medium",
}
HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

STYLE = {
    "es": "Lee este guion en español de España, con tono cercano, enérgico y natural, como un "
          "creador de contenido de vídeos cortos: ritmo ágil, sin pausas largas, y solo una pausa "
          "breve entre párrafos:",
    "en": "Read this script in a warm, energetic, natural tone, like a short-form video creator: "
          "brisk pace, no long pauses, only a brief pause between paragraphs:",
}

Progress = Callable[[str], None]


def data_dir() -> Path:
    d = os.environ.get("STICKMAN_DATA")
    p = Path(d) if d else Path.home() / ".cache" / "stickman-ia"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_piper_voice(lang: str, log: Progress = print) -> Path:
    name = PIPER_VOICES.get(lang, PIPER_VOICES["es"])
    env = os.environ.get("STICKMAN_PIPER_DIR")
    folder = Path(env) if env else data_dir() / "piper"
    folder.mkdir(parents=True, exist_ok=True)
    onnx = folder / f"{name}.onnx"
    cfg = folder / f"{name}.onnx.json"
    loc, speaker, quality = name.split("-")
    base = f"{HF}/{loc.split('_')[0]}/{loc}/{speaker}/{quality}/{name}"
    for path, url in ((cfg, base + ".onnx.json"), (onnx, base + ".onnx")):
        if not path.exists() or path.stat().st_size == 0:
            log(f"Descargando voz Piper {path.name}...")
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
    return onnx


_PIPER_CACHE: dict = {}


def piper_synth(texts: List[str], lang: str, log: Progress = print, rate: float = 1.0) -> List[np.ndarray]:
    from piper import PiperVoice, SynthesisConfig  # import perezoso (onnxruntime pesa)

    model = ensure_piper_voice(lang, log)
    voice = _PIPER_CACHE.get(str(model))
    if voice is None:
        voice = PiperVoice.load(str(model))
        _PIPER_CACHE[str(model)] = voice
    cfg = SynthesisConfig(length_scale=1.0 / max(0.5, rate))
    out = []
    for t in texts:
        chunks = []
        sr = voice.config.sample_rate
        for ch in voice.synthesize(t, syn_config=cfg):
            chunks.append(ch.audio_float_array.astype(np.float32))
            sr = ch.sample_rate
            chunks.append(np.zeros(int(sr * 0.18), np.float32))
        a = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        out.append(trim_silence(resample(a, sr, SR)))
    return out


def gemini_synth(texts: List[str], lang: str, key: str, model: str, voice: str,
                 log: Progress = print) -> List[np.ndarray]:
    style = STYLE.get(lang, STYLE["en"])
    full = "\n\n".join(texts)
    log(f"Generando voz con {model} ({voice})...")
    pcm, sr = gemini.tts(key, model, full, voice=voice, style=style,
                         on_wait=lambda s: log(f"Gemini ocupado, reintento en {s:.0f} s..."))
    a = trim_silence(resample(pcm16_to_float(pcm), sr, SR))
    pieces = split_by_gaps(a, [len(t) for t in texts])
    return [trim_silence(p) for p in pieces]


def synthesize(texts: List[str], lang: str = "es", engine: str = "auto", key: Optional[str] = None,
               tts_model: Optional[str] = None, voice: str = "Puck", log: Progress = print,
               rate: float = 1.0) -> Tuple[List[np.ndarray], str]:
    """Devuelve (audios por escena a 24 kHz, motor usado)."""
    if engine in ("auto", "gemini") and key:
        try:
            model = tts_model
            if not model:
                model = gemini.list_models(key).best_tts
            if model:
                parts = gemini_synth(texts, lang, key, model, voice, log)
                return [normalize(p) for p in parts], f"gemini:{model}:{voice}"
            log("Tu clave no tiene modelos TTS de Gemini; uso Piper local.")
        except gemini.GeminiError as e:
            if engine == "gemini":
                raise
            log(f"{e} → uso la voz local Piper.")
    parts = piper_synth(texts, lang, log, rate)
    return [normalize(p) for p in parts], f"piper:{PIPER_VOICES.get(lang, PIPER_VOICES['es'])}"


def load_wav_resampled(path: str) -> np.ndarray:
    a, sr = read_wav(path)
    return resample(a, sr, SR)
