import importlib

import numpy as np
import pytest

from stickman import gemini, tts
from stickman.audio import SR


def tone(sec: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def pcm(sec: float) -> bytes:
    return (tone(sec) * 32767).astype("<i2").tobytes()


def test_voice_catalog():
    assert len(gemini.TTS_VOICES) == 30 and gemini.DEFAULT_VOICE == "Puck"
    assert all(gemini.TTS_VOICE_INFO[v] for v in gemini.TTS_VOICES)
    assert gemini.voice_label("Charon") == "Charon — informativa"
    assert tts.voice_sample("Kore") == "Hola, soy la voz Kore. Así sonará la narración de tu vídeo."
    assert tts.voice_sample("Kore", "zz").startswith("Hola")


def test_too_long():
    text = "a" * 140  # ~10 s esperados
    assert not tts._too_long(tone(12), text)
    assert tts._too_long(tone(25), text)


def test_gemini_synth_sends_no_style_and_retries(monkeypatch):
    texts = ["Primera escena con una frase.", "Segunda escena con otra frase."]
    calls = []
    durations = [30.0, 4.0]

    def fake_tts(key, model, text, voice="Puck", style="", **kw):
        calls.append((text, style))
        return pcm(durations.pop(0) if durations else 2.0), SR

    monkeypatch.setattr(gemini, "tts", fake_tts)
    monkeypatch.setattr(tts, "split_by_gaps", lambda a, w: [a[: len(a) // 2], a[len(a) // 2:]])
    out = tts.gemini_synth(texts, "es", "k" * 20, "m", "Puck", log=lambda m: None)
    assert len(out) == 2 and len(calls) == 2
    assert all(style == "" for _, style in calls)
    assert calls[0][0] == "Primera escena con una frase.\n\nSegunda escena con otra frase."


def test_preview_uses_only_sample_sentence(monkeypatch):
    seen = {}

    def fake_tts(key, model, text, voice="Puck", style="", **kw):
        seen.update(text=text, voice=voice, style=style, model=model)
        return pcm(1.0), SR

    monkeypatch.setattr(gemini, "tts", fake_tts)
    a = tts.preview("es", "gemini", "k" * 20, "modelo-tts", "Sulafat")
    assert len(a) > SR * 0.5
    assert seen == {"text": "Hola, soy la voz Sulafat. Así sonará la narración de tu vídeo.",
                    "voice": "Sulafat", "style": "", "model": "modelo-tts"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STICKMAN_DATA", str(tmp_path))
    import stickman.web.app as appmod
    appmod = importlib.reload(appmod)
    return TestClient(appmod.app), appmod


def test_estado_lists_voices(client):
    c, _ = client
    d = c.get("/api/estado").json()
    assert len(d["voces"]) == 30 and d["voz_defecto"] == "Puck" and d["voces_info"]["Puck"] == "animada"


def test_probar_voz_endpoint(client, monkeypatch):
    c, appmod = client
    r = c.post("/api/probar-voz", json={"voz": "Puck"})
    assert r.status_code == 400 and "clave" in r.json()["detail"]
    r = c.post("/api/probar-voz", json={"clave": "x" * 20, "voz": "Nadie"})
    assert r.status_code == 400

    got = {}

    def fake_preview(lang, engine, key, model, voice):
        got.update(lang=lang, engine=engine, key=key, voice=voice)
        return tone(0.5)

    monkeypatch.setattr(tts, "preview", fake_preview)
    r = c.post("/api/probar-voz", json={"clave": "AIzaSECRET-123456", "voz": "Charon"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert b"SECRET" not in r.content and got["voice"] == "Charon" and got["engine"] == "gemini"

    monkeypatch.setattr(appmod.preview_limit, "per_hour", 2)
    r = c.post("/api/probar-voz", json={"motor": "piper"})
    assert r.status_code == 200 and got["engine"] == "piper" and got["key"] is None
    r = c.post("/api/probar-voz", json={"motor": "piper"})
    assert r.status_code == 429
