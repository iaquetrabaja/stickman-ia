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


# Segundos de habla esperables por carácter (voces de Gemini en español: ~14-16 caracteres/s).
CHARS_PER_SEC = 14.0
MAX_RATIO = 1.45   # si el audio dura más que esto x lo esperado, ha leído algo que no es el guion


def _too_long(audio: np.ndarray, text: str) -> bool:
    expected = max(1.5, len(text) / CHARS_PER_SEC)
    return len(audio) / SR > expected * MAX_RATIO + 1.0


def gemini_synth(texts: List[str], lang: str, key: str, model: str, voice: str,
                 log: Progress = print) -> List[np.ndarray]:
    """Voz con Gemini. NO se envía ninguna instrucción de estilo: los modelos TTS a veces la leen en voz alta.
    El tono lo da la voz elegida. Se comprueba la duración; si sale larga se repite y después se va escena a escena."""
    wait = lambda s: log(f"Gemini ocupado, reintento en {s:.0f} s...")  # noqa: E731
    full = "\n\n".join(texts)
    log(f"Generando voz con {model} ({voice})...")
    for intento in range(2):
        pcm, sr = gemini.tts(key, model, full, voice=voice, style="", on_wait=wait)
        a = trim_silence(resample(pcm16_to_float(pcm), sr, SR))
        if not _too_long(a, full):
            pieces = split_by_gaps(a, [len(t) for t in texts])
            return [trim_silence(p) for p in pieces]
        log("La voz ha salido más larga de lo normal; se repite" + (" escena a escena" if intento else "") + "...")
    out = []   # último recurso: una llamada por escena, cada audio validado por separado
    for t in texts:
        best = None
        for _ in range(2):
            pcm, sr = gemini.tts(key, model, t, voice=voice, style="", on_wait=wait)
            best = trim_silence(resample(pcm16_to_float(pcm), sr, SR))
            if not _too_long(best, t):
                break
        out.append(best)
    return out


VOICE_SAMPLE = {
    "es": "Hola, soy la voz {nombre}. Así sonará la narración de tu vídeo.",
    "en": "Hi, I'm the voice {nombre}. This is how your video's narration will sound.",
    "pt": "Olá, eu sou a voz {nombre}. É assim que vai soar a narração do seu vídeo.",
    "fr": "Bonjour, je suis la voix {nombre}. Voici comment sonnera la narration de ta vidéo.",
    "it": "Ciao, sono la voce {nombre}. Così suonerà la narrazione del tuo video.",
    "de": "Hallo, ich bin die Stimme {nombre}. So wird die Erzählung deines Videos klingen.",
}


def voice_sample(nombre: str, lang: str = "es") -> str:
    return VOICE_SAMPLE.get(lang, VOICE_SAMPLE["es"]).format(nombre=nombre)


def preview(lang: str = "es", engine: str = "gemini", key: Optional[str] = None, model: Optional[str] = None,
            voice: str = "Puck") -> np.ndarray:
    """Una frase corta con la voz elegida (24 kHz), para oírla antes de generar el vídeo."""
    if engine == "piper":
        audio = piper_synth([voice_sample("local", lang)], lang, log=lambda _m: None)[0]
    else:
        model = model or gemini.list_models(key).best_tts
        if not model:
            raise gemini.GeminiError("Tu clave no tiene ningún modelo de voz de Gemini. Usa la voz local.")
        pcm, sr = gemini.tts(key, model, voice_sample(voice, lang), voice=voice, style="", retries=1)
        audio = trim_silence(resample(pcm16_to_float(pcm), sr, SR))
    return normalize(audio)


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
