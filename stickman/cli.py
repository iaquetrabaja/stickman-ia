"""Línea de comandos.

Ejemplos:
  python -m stickman "Por qué procrastinamos" --formato 9:16 --duracion 45 --clave TU_CLAVE
  python -m stickman --offline                      # sin clave: guion de ejemplo + voz Piper
  python -m stickman --guion examples/regla-2-minutos.yaml --formato 16:9
  python -m stickman "tema" --review                # revisa/edita el guion antes de renderizar
  python -m stickman --guion g.json --audio voz.wav  # usa una narración ya grabada (sin TTS)
  python -m stickman web --puerto 8000              # interfaz web
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFLINE_EXAMPLE = Path(__file__).resolve().parent / "assets" / "ejemplo-offline.json"


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50] or "video"


def _review(path: Path) -> None:
    from .schema import ScriptError, load_script

    while True:
        print(f"\nRevisa y edita el guion: {path}")
        editor = os.environ.get("EDITOR")
        try:
            if editor:
                subprocess.call([editor, str(path)])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["notepad", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-t", str(path)])
        except OSError:
            pass
        input("Guarda el archivo y pulsa Enter para continuar (Ctrl+C para cancelar)... ")
        try:
            _, warnings = load_script(path)
            for w in warnings:
                print(f"aviso: {w}")
            return
        except ScriptError as e:
            print(e)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "web":
        return _web(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m stickman", description="Stickman IA: vídeos explicativos con muñecos de palo.")
    ap.add_argument("tema", nargs="?", help="tema del vídeo (lo escribe Gemini)")
    ap.add_argument("--formato", default="9:16", choices=["9:16", "16:9"])
    ap.add_argument("--duracion", type=int, default=45, help="duración objetivo en segundos (10-120)")
    ap.add_argument("--clave", default=None, help="clave de Gemini (o variable GEMINI_API_KEY)")
    ap.add_argument("--idioma", default="es", help="es, en, pt, fr, it, de")
    ap.add_argument("--voz", default="Puck", help="voz de Gemini TTS (Puck, Kore, Charon, Aoede...)")
    ap.add_argument("--motor-voz", default="auto", choices=["auto", "gemini", "piper"])
    ap.add_argument("--modelo", default=None, help="modelo de texto (por defecto, el mejor 'flash')")
    ap.add_argument("--modelo-tts", default=None, help="modelo TTS (por defecto, el mejor disponible)")
    ap.add_argument("--guion", default=None, help="usar un guion JSON/YAML existente (sin LLM)")
    ap.add_argument("--solo-guion", action="store_true", help="solo generar el guion y salir")
    ap.add_argument("--review", action="store_true", help="pausar para revisar/editar el guion")
    ap.add_argument("--offline", action="store_true", help="sin clave: guion de ejemplo + Piper")
    ap.add_argument("--salida", default="salida", help="carpeta de salida")
    ap.add_argument("--nombre", default=None, help="nombre base de los archivos")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--escala", type=float, default=1.0,
                    help="escala de render (0.667 = 720p reescalado con ffmpeg, más rápido)")
    ap.add_argument("--max-segundos", type=float, default=120)
    ap.add_argument("--audio", default=None,
                    help="narración ya grabada (WAV, MP3, MP4...) en lugar de sintetizar la voz; requiere --guion")
    ap.add_argument("--sin-alineacion", action="store_true",
                    help="no alinear los subtítulos con la voz (tiempos estimados por longitud)")
    a = ap.parse_args(argv)

    from . import gemini
    from .pipeline import Stats, dump, render_script, script_from_file, write_script
    from .schema import ScriptError

    key = a.clave or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    a.duracion = max(10, min(int(a.max_segundos), a.duracion))
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    if a.audio and not (a.guion or a.offline):
        ap.error("--audio necesita el guion de esa narración (--guion)")
    try:
        if a.offline:
            script = script_from_file(a.guion or OFFLINE_EXAMPLE)
            key = None
            a.motor_voz = "piper"
        elif a.guion:
            script = script_from_file(a.guion)
        else:
            if not a.tema:
                ap.error("indica un tema, --guion o --offline")
            if not key:
                ap.error("falta la clave de Gemini (--clave o GEMINI_API_KEY). Consíguela gratis en "
                         "https://aistudio.google.com/apikey  — o prueba con --offline")
            script, _, _ = write_script(key, a.tema, a.idioma, a.duracion, a.formato, a.modelo, stats=stats)
        name = a.nombre or _slug(script.title)
        if a.review or a.solo_guion:
            p = out / f"{name}.guion.json"
            dump(script, p)
            print(f"Guion guardado en {p}")
            if a.solo_guion:
                return 0
            _review(p)
            script = script_from_file(p)
        res = render_script(script, out, a.formato, a.motor_voz, key, a.modelo_tts, a.voz, a.fps, a.escala,
                            a.max_segundos, basename=name, stats=stats, narration=a.audio,
                            word_align=not a.sin_alineacion)
        s = res["stats"]
        print(f"MP4: {res['mp4']}\nSRT: {res['srt']}\nLlamadas a la API: {s.api_calls}")
        return 0
    except (gemini.GeminiError, ScriptError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _web(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m stickman web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--puerto", type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn

    uvicorn.run("stickman.web.app:asgi", host=a.host, port=a.puerto, proxy_headers=True,
                forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"))
    return 0
