"""Cliente mínimo de la API de Gemini (BYOK) con la librería estándar.

La clave viaja solo en la cabecera ``x-goog-api-key`` (nunca en la URL, para que
no acabe en logs) y no se guarda en disco.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

API = "https://generativelanguage.googleapis.com/v1beta"

EXCLUDE = ("tts", "image", "audio", "live", "embedding", "vision", "robotics", "computer-use",
           "gemma", "aqa", "imagen", "veo", "nano", "lyria", "omni", "transcribe", "antigravity",
           "deep-research", "customtools", "learnlm", "thinking-exp")

# Voces precompiladas de Gemini TTS con su carácter según Google
# (https://ai.google.dev/gemini-api/docs/speech-generation#voices), traducido.
TTS_VOICE_INFO = {
    "Zephyr": "brillante", "Puck": "animada", "Charon": "informativa", "Kore": "firme",
    "Fenrir": "enérgica", "Leda": "juvenil", "Orus": "firme", "Aoede": "desenfadada",
    "Callirrhoe": "tranquila", "Autonoe": "brillante", "Enceladus": "suave, aspirada", "Iapetus": "clara",
    "Umbriel": "tranquila", "Algieba": "suave", "Despina": "suave", "Erinome": "clara",
    "Algenib": "grave, rasgada", "Rasalgethi": "informativa", "Laomedeia": "animada", "Achernar": "suave",
    "Alnilam": "firme", "Schedar": "equilibrada", "Gacrux": "madura", "Pulcherrima": "directa",
    "Achird": "cercana", "Zubenelgenubi": "informal", "Vindemiatrix": "amable", "Sadachbia": "viva",
    "Sadaltager": "experta", "Sulafat": "cálida",
}
TTS_VOICES = list(TTS_VOICE_INFO)
DEFAULT_VOICE = "Puck"


def voice_label(voice: str) -> str:
    """'Charon — informativa' (para los desplegables)."""
    info = TTS_VOICE_INFO.get(voice)
    return f"{voice} — {info}" if info else voice


class GeminiError(RuntimeError):
    """Error con mensaje en español apto para mostrar al usuario."""

    def __init__(self, msg: str, status: int = 0, quota: bool = False):
        super().__init__(msg)
        self.status = status
        self.quota = quota


def _request(method: str, path: str, key: str, body: Optional[dict] = None, timeout: float = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}/{path}", data=data, method=method,
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001
            detail = ""
        detail = detail.replace(key, "***") if key else detail
        if e.code == 429:
            raise GeminiError("Has superado la cuota gratuita de Gemini para este modelo (error 429). "
                              "Espera unos minutos o usa la voz local (Piper).", 429, quota=True) from None
        if e.code in (400, 401, 403) and ("API key" in detail or "API_KEY" in detail or e.code in (401, 403)):
            raise GeminiError("La clave de Gemini no es válida o no tiene permisos. Revísala en "
                              "https://aistudio.google.com/apikey", e.code) from None
        if e.code == 404:
            raise GeminiError(f"Modelo no disponible para tu clave ({detail[:160]})", 404) from None
        if e.code >= 500:
            raise GeminiError(f"Gemini no responde ahora mismo (error {e.code}). Inténtalo de nuevo.", e.code) from None
        raise GeminiError(f"Error de Gemini {e.code}: {detail[:300]}", e.code) from None
    except urllib.error.URLError as e:
        raise GeminiError(f"No se pudo conectar con Gemini: {e.reason}") from None
    except TimeoutError:
        raise GeminiError("Gemini tardó demasiado en responder.") from None


@dataclass
class Models:
    text: List[str]
    tts: List[str]

    @property
    def best_text(self) -> Optional[str]:
        return self.text[0] if self.text else None

    @property
    def best_tts(self) -> Optional[str]:
        return self.tts[0] if self.tts else None


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def _score_text(name: str) -> float:
    s = _version(name) * 10
    if name.endswith("flash-latest"):
        s += 22  # alias estable recomendado (equivale a ~2.x)
    if "lite" in name:
        s -= 15
    if "preview" in name or "exp" in name:
        s -= 4
    return s


def _score_tts(name: str) -> float:
    s = _version(name) * 10
    if "lite" in name:
        s -= 8
    if "pro" in name:
        s -= 3  # más lento y con cuota menor
    if "preview" in name:
        s -= 2
    return s


def rank_models(names: List[Tuple[str, List[str]]]) -> Models:
    text, tts = [], []
    for full, methods in names:
        n = full.split("/")[-1]
        if "generateContent" not in methods:
            continue
        low = n.lower()
        if "tts" in low and low.startswith("gemini"):
            tts.append(n)
            continue
        if not low.startswith("gemini") or "flash" not in low:
            continue
        if any(x in low for x in EXCLUDE):
            continue
        text.append(n)
    text.sort(key=_score_text, reverse=True)
    tts.sort(key=_score_tts, reverse=True)
    return Models(text, tts)


_MODELS_CACHE: dict = {}


def list_models(key: str) -> Models:
    """ListModels con caché en memoria de 10 min (indexada por hash de la clave)."""
    if not key or len(key) < 20:
        raise GeminiError("Falta la clave de Gemini (o es demasiado corta).")
    import hashlib

    h = hashlib.sha256(key.encode()).hexdigest()
    hit = _MODELS_CACHE.get(h)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    models = _list_models(key)
    _MODELS_CACHE[h] = (time.time(), models)
    return models


def _list_models(key: str) -> Models:
    out: List[Tuple[str, List[str]]] = []
    token = ""
    for _ in range(5):
        path = "models?pageSize=200" + (f"&pageToken={token}" if token else "")
        d = _request("GET", path, key, timeout=30)
        out += [(m["name"], m.get("supportedGenerationMethods", [])) for m in d.get("models", [])]
        token = d.get("nextPageToken", "")
        if not token:
            break
    return rank_models(out)


def generate_json(key: str, model: str, prompt: str, temperature: float = 0.9) -> str:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    d = _request("POST", f"models/{model}:generateContent", key, body, timeout=180)
    try:
        parts = d["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError):
        reason = d.get("promptFeedback", {}).get("blockReason") or d.get("candidates", [{}])[0].get("finishReason")
        raise GeminiError(f"Gemini no devolvió texto (motivo: {reason}).") from None


def tts(key: str, model: str, text: str, voice: str = "Puck", style: str = "",
        retries: int = 2, on_wait: Optional[Callable[[float], None]] = None) -> Tuple[bytes, int]:
    """Devuelve (PCM 16 bits mono, frecuencia).

    Usa ``style=""`` (lo normal): los modelos TTS a veces leen en voz alta las instrucciones de estilo."""
    prompt = f"{style.strip()}\n\n{text}" if style else text
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    last: Optional[GeminiError] = None
    for attempt in range(retries + 1):
        try:
            d = _request("POST", f"models/{model}:generateContent", key, body, timeout=300)
            part = d["candidates"][0]["content"]["parts"][0]["inlineData"]
            mime = part.get("mimeType", "")
            m = re.search(r"rate=(\d+)", mime)
            return base64.b64decode(part["data"]), int(m.group(1)) if m else 24000
        except GeminiError as e:
            last = e
            if e.status in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 8.0 * (attempt + 1) if e.status != 429 else 20.0 * (attempt + 1)
                if on_wait:
                    on_wait(wait)
                time.sleep(wait)
                continue
            raise
        except (KeyError, IndexError):
            last = GeminiError("Gemini TTS no devolvió audio.")
            if attempt < retries:
                time.sleep(3)
                continue
    raise last or GeminiError("Gemini TTS falló.")
