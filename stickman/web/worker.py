"""Proceso hijo que renderiza un trabajo de la web.

Se lanza como ``python -m stickman.web.worker <carpeta>``. La clave de Gemini
(si la hay) llega por stdin y solo vive en memoria de este proceso.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    job_dir = Path(sys.argv[1])
    key = sys.stdin.readline().strip() or None
    spec = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    prog_file = job_dir / "progress.json"
    state = {"etapa": "voz", "progreso": 0.0, "mensaje": "Preparando...", "estado": "renderizando"}
    last = [0.0]

    def write(force: bool = False) -> None:
        now = time.time()
        if not force and now - last[0] < 0.5:
            return
        last[0] = now
        tmp = prog_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, prog_file)

    def log(msg: str) -> None:
        state["mensaje"] = msg
        write(True)

    def progress(stage: str, p: float) -> None:
        state["etapa"] = stage
        # voz = 0-12 %, subtítulos (alineación) = 12-18 %, render = 18-100 %
        base, span = {"voz": (0.0, 0.12), "subtitulos": (0.12, 0.06)}.get(stage, (0.18, 0.82))
        state["progreso"] = round(base + span * p, 4)
        write()

    try:
        from stickman.pipeline import render_script
        from stickman.schema import validate_script

        script, _ = validate_script(spec["script"])
        res = render_script(script, job_dir, spec.get("formato", "9:16"), spec.get("motor", "auto"), key,
                            spec.get("modelo_tts"), spec.get("voz", "Puck"), 30, float(spec.get("escala", 1.0)),
                            float(spec.get("max_segundos", 120)), log=log, progress=progress, basename="video")
        state.update(estado="terminado", progreso=1.0, etapa="listo",
                     mensaje=f"Vídeo de {res['duration']:.0f} s listo", voz=res["stats"].tts_engine,
                     duracion=res["duration"], render_s=round(res["stats"].render_seconds, 1))
        write(True)
        return 0
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if key:
            msg = msg.replace(key, "***")
        state.update(estado="error", mensaje=msg[:500])
        write(True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
