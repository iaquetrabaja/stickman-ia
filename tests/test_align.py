"""Alineación de subtítulos: guion <-> ASR, interpolación, agrupado y prueba real con Piper + Whisper."""
import numpy as np
import pytest

from stickman.align import Word, align_tokens, fill_missing, norm_token, refine_with_energy
from stickman.audio import SR, Cue, group_captions, to_srt


def asr(*items):
    return [Word(t, s, e) for t, s, e in items]


def test_norm_token_accents_case_punct():
    assert norm_token("¿Estrés?") == "estres"
    assert norm_token("Comprométete,") == "comprometete"
    assert norm_token("Año") == norm_token("año") == "ano"
    assert norm_token("—") == ""


def test_align_exact_and_accents():
    script = "Tu cerebro no posterga tareas.".split()
    rec = asr(("tu", 0.1, 0.3), ("Cerebro", 0.3, 0.7), ("no", 0.7, 0.9), ("postergá", 0.9, 1.4), ("tareas", 1.5, 2.0))
    t = align_tokens(script, rec)
    assert [x[0] for x in t] == [0.1, 0.3, 0.7, 0.9, 1.5]


def test_align_insertions_deletions_and_numbers():
    # guion: "Aplica la regla de los 2 minutos hoy mismo"
    # ASR: dice "dos" en vez de "2", se inventa "eh", y no oye "mismo"
    script = "Aplica la regla de los 2 minutos hoy mismo.".split()
    rec = asr(("aplica", 0.0, 0.4), ("eh", 0.4, 0.5), ("la", 0.5, 0.6), ("regla", 0.6, 0.9), ("de", 0.9, 1.0),
              ("los", 1.0, 1.1), ("dos", 1.1, 1.3), ("minutos", 1.3, 1.8), ("hoy", 1.9, 2.1))
    t = align_tokens(script, rec)
    assert t[0] == (0.0, 0.4)
    assert t[2] == (0.6, 0.9)                 # "eh" insertado no desplaza nada
    assert t[5] == (1.1, 1.3)                 # "2" <-> "dos" (sustitución 1 a 1)
    assert t[6] == (1.3, 1.8) and t[7] == (1.9, 2.1)
    assert t[8] is None                       # "mismo." no reconocida
    words = fill_missing(script, t, 0.0, 2.6)
    assert words[8].start >= 2.1 - 1e-9 and words[8].end <= 2.6 + 1e-9
    assert all(b.start >= a.start for a, b in zip(words, words[1:]))
    assert all(a.end <= b.start + 1e-9 for a, b in zip(words, words[1:]))


def test_align_unequal_replacement_spreads_time():
    # "20%" leído como "veinte por ciento": el tramo del ASR se reparte
    script = "Sube un 20% este año.".split()
    rec = asr(("sube", 0.0, 0.3), ("un", 0.3, 0.4), ("veinte", 0.4, 0.7), ("por", 0.7, 0.8), ("ciento", 0.8, 1.1),
              ("este", 1.2, 1.4), ("año", 1.4, 1.7))
    t = align_tokens(script, rec)
    assert t[2] == pytest.approx((0.4, 1.1))
    assert t[3] == (1.2, 1.4)


def test_align_ignores_spurious_far_match():
    script = "uno dos tres cuatro".split()
    rec = asr(("uno", 0.0, 0.2), ("dos", 0.2, 0.4), ("tres", 0.4, 0.6), ("cuatro", 0.6, 0.8), ("uno", 5.0, 5.2))
    t = align_tokens(script, rec)
    assert t[0] == (0.0, 0.2)


def test_fill_missing_all_unmatched_is_proportional():
    script = ["aa", "bbbbbb"]
    w = fill_missing(script, [None, None], 1.0, 2.0)
    assert w[0].start == 1.0 and w[1].end == pytest.approx(2.0)
    assert (w[0].end - w[0].start) < (w[1].end - w[1].start)


def test_refine_moves_start_out_of_silence():
    sil, tone = np.zeros(int(0.5 * SR), np.float32), 0.3 * np.ones(int(0.5 * SR), np.float32)
    a = np.concatenate([sil, tone])
    w = refine_with_energy([Word("hola", 0.2, 0.9)], a, SR)
    assert w[0].start == pytest.approx(0.5, abs=0.02)


def test_refine_moves_sentence_start_after_pause():
    # "...fin. La acción": Whisper pone "La" pegada a la frase anterior, antes de una pausa de 0.8 s
    t = lambda s: 0.3 * np.ones(int(s * SR), np.float32)  # noqa: E731
    a = np.concatenate([t(1.0), np.zeros(int(0.8 * SR), np.float32), t(1.0)])
    ws = [Word("fin.", 0.6, 0.98), Word("La", 0.98, 1.1), Word("acción", 1.82, 2.4)]
    refine_with_energy(ws, a, SR)
    assert ws[1].start == pytest.approx(1.8, abs=0.02)
    assert ws[1].start <= ws[2].start


def test_group_captions_sentences_lead_hold_no_overlap():
    text = "Tu cerebro no posterga tareas por pereza. Al evitar ese informe pendiente, sientes alivio."
    ws, t = [], 0.0
    for w in text.split():
        ws.append(Cue(w, t, t + 0.3))
        t += 0.35
    cues = group_captions(ws, 4, 22)
    # nunca cruza el punto
    assert any(c.text.endswith("pereza.") for c in cues)
    assert all(2 <= len(c.text.split()) <= 4 for c in cues)
    assert cues[0].start == pytest.approx(0.0)  # no negativo
    second = [c for c in cues if c.text.startswith("Al ")][0]
    assert second.start == pytest.approx(ws[7].start - 0.08)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(cues, cues[1:]))
    assert to_srt(cues).startswith("1\n00:00:00,000 --> ")


# --------------------------------------------------------------------------- #
# Integración: voz Piper real + Whisper real
# --------------------------------------------------------------------------- #
def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not (_have("faster_whisper") and _have("piper")), reason="faltan faster-whisper o piper")
def test_piper_captions_follow_asr_words():
    from stickman import align
    from stickman.pipeline import _voiced_extent, timed_captions
    from stickman.tts import piper_synth

    texts = ["Tu cerebro no posterga tareas por pereza. Lo hace para protegerte del estrés inmediato.",
             "Aplica la regla de los dos minutos: abre el archivo y escribe una sola frase."]
    try:
        audios = piper_synth(texts, "es", log=lambda m: None)
    except Exception as e:  # noqa: BLE001 (sin red para descargar la voz)
        pytest.skip(f"Piper no disponible: {e}")
    lead, tail = 0.15, 0.35
    track, ranges, t0 = [], [], 0.0
    for a in audios:
        track += [np.zeros(int(lead * SR), np.float32), a, np.zeros(int(tail * SR), np.float32)]
        v0, v1 = _voiced_extent(a)
        ranges.append((t0 + lead + v0, t0 + lead + v1))
        t0 += lead + len(a) / SR + tail
    track = np.concatenate(track)
    try:
        cues, method = timed_captions(texts, ranges, track, "es", 4, 22, True, log=lambda m: None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Whisper no disponible: {e}")
    if method != "asr":
        pytest.skip("Whisper no disponible (sin red para descargar el modelo)")
    # referencia: palabras del ASR (con el inicio sacado del silencio previo, error típico de Whisper)
    rec = refine_with_energy(align.transcribe(track, SR, "es", log=lambda m: None), track, SR)
    words = [w for c in cues for w in c.text.split()]
    times = align_tokens(words, rec)
    k, checked = 0, 0
    for c in cues:
        if times[k] is not None:
            assert abs(c.start - times[k][0]) <= 0.15, (c.text, c.start, times[k][0])
            checked += 1
        k += len(c.text.split())
    assert checked >= 0.8 * len(cues)
    # el primer subtítulo de cada escena aparece con la voz (inicio real de la escena, ±150 ms)
    for lo, _hi in ranges:
        first = min((c for c in cues if c.start >= lo - 0.5), key=lambda c: c.start)
        assert abs(first.start - lo) <= 0.15
    # el SRT sale de los mismos tiempos
    srt = to_srt(cues)
    assert srt.count("-->") == len(cues)


def test_refine_word_swallowing_a_pause():
    # "nueva, predice": Whisper hace empezar "predice" en la cola de "nueva", antes de 0,25 s de pausa
    t = lambda s: 0.3 * np.ones(int(s * SR), np.float32)  # noqa: E731
    a = np.concatenate([t(1.0), np.zeros(int(0.25 * SR), np.float32), t(0.6)])
    ws = [Word("nueva,", 0.5, 0.92), Word("predice", 0.92, 1.8)]
    refine_with_energy(ws, a, SR)
    assert ws[1].start == pytest.approx(1.25, abs=0.02)
