import numpy as np

from stickman.audio import (SR, amplitude_envelope, group_captions, silence_gaps, split_by_gaps, to_srt,
                            word_timings)
from stickman.gemini import rank_models
from stickman.render import VideoRenderer
from stickman.schema import validate_script


def tone(sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * SR), np.float32)


def test_envelope_and_gaps():
    a = np.concatenate([tone(1), silence(0.5), tone(1)])
    env = amplitude_envelope(a, SR, 30, 75)
    assert env[10] > 0.5 and env[40] < 0.05
    gaps = silence_gaps(a)
    assert len(gaps) == 1 and abs(gaps[0][0] - 1.0) < 0.05


def test_split_by_gaps_uses_silences():
    a = np.concatenate([tone(2.0), silence(0.4), tone(1.0), silence(0.4), tone(1.0)])
    parts = split_by_gaps(a, [20, 10, 10])
    assert len(parts) == 3
    assert abs(len(parts[0]) / SR - 2.2) < 0.15
    assert sum(len(p) for p in parts) == len(a)


def test_word_timings_and_captions():
    words = word_timings("Hola, esto es una prueba de subtítulos. Fin.", 1.0, 5.0)
    assert words[0].start == 1.0 and abs(words[-1].end - 5.0) < 1e-6
    assert all(w.end >= w.start for w in words)
    cues = group_captions(words, 4, 22)
    assert all(len(c.text.split()) <= 4 for c in cues)
    assert to_srt(cues).startswith("1\n00:00:01,000 --> ")


def test_word_timings_snap_to_silence():
    words = word_timings("Uno dos tres, cuatro cinco seis.", 0.0, 4.0, gaps=[(1.9, 2.3)])
    tres = [w for w in words if w.text == "tres,"][0]
    cuatro = [w for w in words if w.text == "cuatro"][0]
    assert abs(tres.end - 1.9) < 1e-6 and abs(cuatro.start - 2.3) < 1e-6


def test_rank_models():
    gc = ["generateContent"]
    names = [("models/gemini-2.5-flash", gc), ("models/gemini-3.8-flash", gc),
             ("models/gemini-3.8-flash-lite", gc), ("models/gemini-3.1-flash-image", gc),
             ("models/gemini-flash-latest", gc), ("models/gemini-3.8-flash-tts", gc),
             ("models/gemini-3.8-flash-lite-tts", gc), ("models/gemini-2.5-flash-preview-tts", gc),
             ("models/gemini-3.8-live", ["bidiGenerateContent"]), ("models/gemini-embedding-2", ["embedContent"])]
    m = rank_models(names)
    assert m.best_text == "gemini-3.8-flash"
    assert m.best_tts == "gemini-3.8-flash-tts"
    assert "gemini-3.1-flash-image" not in m.text and "gemini-3.8-live" not in m.text


def test_tts_falls_back_to_piper_on_quota(monkeypatch):
    from stickman import gemini, tts

    def boom(*a, **k):
        raise gemini.GeminiError("cuota", 429, quota=True)

    monkeypatch.setattr(tts, "gemini_synth", boom)
    monkeypatch.setattr(tts, "piper_synth", lambda texts, lang, log, rate=1.0: [tone(0.5) for _ in texts])
    logs = []
    audios, used = tts.synthesize(["a", "b"], "es", "auto", "x" * 39, "modelo-tts", log=logs.append)
    assert used.startswith("piper:") and len(audios) == 2
    assert any("Piper" in m for m in logs)


def test_render_frames_smoke(tmp_path):
    raw = {"title": "t", "cast": [{"id": "a"}], "scenes": [
        {"narration": "Uno dos tres.", "camera": "zoom_in", "keywords": ["CLAVE"],
         "characters": [{"id": "a", "enter": "left", "actions": [{"at": 0.5, "pose": "jump"}]}],
         "props": [{"type": "speech_bubble", "holder": "a", "text": "Hola"}, {"type": "chart"}]},
        {"narration": "Cuatro.", "background": "noche", "characters": [{"id": "a", "pose": "sit"}],
         "props": [{"type": "desk"}, {"type": "laptop"}, {"type": "money", "holder": "a"}]}]}
    script, _ = validate_script(raw)
    for fmt in ("9:16", "16:9"):
        env = np.full(90, 0.5, np.float32)
        r = VideoRenderer(script, fmt, [1.5, 1.5], env, [], fps=30)
        assert r.total_frames == 90
        for f in (0, 20, 45, 89):
            r.snapshot(f, str(tmp_path / f"{fmt.replace(':', 'x')}_{f}.png"))
    assert len(list(tmp_path.glob("*.png"))) == 8
