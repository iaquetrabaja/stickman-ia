"""App web (FastAPI): cola de UN render a la vez, límite por IP, borrado a las 24 h.

Variables de entorno:
  ROOT_PATH      prefijo público (p. ej. /stickman) si va detrás de un proxy
  STICKMAN_DATA  carpeta de datos (trabajos, voces Piper). Por defecto ~/.cache/stickman-ia
  LIMITE_DIARIO  renders por IP y día (3)
  MAX_SEGUNDOS   duración máxima del vídeo (120)
  MAX_COLA       trabajos máximos en cola (20)
  HORAS_BORRADO  horas que se guardan los archivos (24)
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import gemini
from ..prompt import WORDS_PER_SEC
from ..schema import LANGUAGES, ScriptError, validate_script
from ..tts import data_dir

ROOT_PATH = os.environ.get("ROOT_PATH", "").rstrip("/")
LIMIT = int(os.environ.get("LIMITE_DIARIO", "3"))
SCRIPT_LIMIT = int(os.environ.get("LIMITE_GUIONES", "20"))
MAX_SECONDS = int(os.environ.get("MAX_SEGUNDOS", "120"))
MAX_QUEUE = int(os.environ.get("MAX_COLA", "20"))
KEEP_HOURS = float(os.environ.get("HORAS_BORRADO", "24"))
RENDER_SCALE = float(os.environ.get("ESCALA_RENDER", "1.0"))

DATA = data_dir()
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
LIMITS_FILE = DATA / "limites.json"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Stickman IA", docs_url=None, redoc_url=None, openapi_url=None)


class StripPrefix:
    """Acepta rutas con o sin el prefijo ROOT_PATH (funciona tanto si el proxy lo
    quita como si no). El HTML usa URLs relativas, así que no necesita conocerlo."""

    def __init__(self, inner, prefix: str):
        self.inner = inner
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if self.prefix and scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path == self.prefix and scope["type"] == "http":
                # /stickman -> /stickman/ (las URLs relativas necesitan la barra final)
                await send({"type": "http.response.start", "status": 308,
                            "headers": [(b"location", (self.prefix + "/").encode())]})
                await send({"type": "http.response.body", "body": b""})
                return
            if path.startswith(self.prefix + "/"):
                scope = dict(scope, path=path[len(self.prefix):], raw_path=path[len(self.prefix):].encode())
        await self.inner(scope, receive, send)


# --------------------------------------------------------------------------- #
class Limits:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        try:
            self.data: Dict[str, Dict[str, int]] = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            self.data = {}

    def _today(self) -> Dict[str, int]:
        d = date.today().isoformat()
        if d not in self.data:
            self.data = {d: {}}
        return self.data[d]

    def used(self, key: str) -> int:
        with self.lock:
            return self._today().get(key, 0)

    def take(self, key: str, limit: int) -> bool:
        with self.lock:
            t = self._today()
            if t.get(key, 0) >= limit:
                return False
            t[key] = t.get(key, 0) + 1
            try:
                self.path.write_text(json.dumps(self.data))
            except OSError:
                pass
            return True

    def give_back(self, key: str) -> None:
        with self.lock:
            t = self._today()
            if t.get(key, 0) > 0:
                t[key] -= 1


limits = Limits(LIMITS_FILE)


class Job:
    def __init__(self, jid: str, ip: str, key: Optional[str]):
        self.id = jid
        self.ip = ip
        self.key = key  # solo en memoria; se borra al lanzar el render
        self.created = time.time()
        self.status = "en_cola"
        self.message = "En cola"
        self.dir = JOBS / jid


class Queue:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.pending: Deque[str] = deque()
        self.current: Optional[str] = None
        self.lock = threading.Lock()
        self.event = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def add(self, job: Job) -> int:
        with self.lock:
            self.jobs[job.id] = job
            self.pending.append(job.id)
            pos = len(self.pending) + (1 if self.current else 0)
        self.event.set()
        return pos

    def position(self, jid: str) -> int:
        with self.lock:
            if self.current == jid:
                return 0
            try:
                return list(self.pending).index(jid) + 1 + (1 if self.current else 0)
            except ValueError:
                return -1

    def _loop(self) -> None:
        while True:
            self.event.wait(2)
            self.event.clear()
            while True:
                with self.lock:
                    if not self.pending:
                        break
                    jid = self.pending.popleft()
                    self.current = jid
                job = self.jobs.get(jid)
                if job:
                    self._run(job)
                with self.lock:
                    self.current = None

    def _run(self, job: Job) -> None:
        job.status = "renderizando"
        key, job.key = job.key, None
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "2")
        try:
            with open(job.dir / "worker.log", "wb") as errlog:
                proc = subprocess.Popen([sys.executable, "-m", "stickman.web.worker", str(job.dir)],
                                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errlog,
                                        env=env, cwd=str(Path(__file__).resolve().parents[2]))
            proc.stdin.write(((key or "") + "\n").encode())
            proc.stdin.close()
            del key
            try:
                proc.wait(timeout=15 * 60)
            except subprocess.TimeoutExpired:
                proc.kill()
                self._set_error(job, "El render tardó demasiado y se canceló.")
                return
            st = read_progress(job)
            job.status = st.get("estado", "error") if proc.returncode == 0 else "error"
            if job.status == "error":
                job.message = st.get("mensaje") or "Error al renderizar"
                limits.give_back(job.ip)  # un fallo no gasta cupo
        except Exception as e:  # noqa: BLE001
            self._set_error(job, f"Error interno: {e}")

    def _set_error(self, job: Job, msg: str) -> None:
        job.status = "error"
        job.message = msg
        (job.dir / "progress.json").write_text(json.dumps({"estado": "error", "mensaje": msg}), encoding="utf-8")
        limits.give_back(job.ip)

    def _cleanup_loop(self) -> None:
        while True:
            cutoff = time.time() - KEEP_HOURS * 3600
            for d in JOBS.iterdir() if JOBS.exists() else []:
                try:
                    if d.is_dir() and d.stat().st_mtime < cutoff and d.name != self.current:
                        shutil.rmtree(d, ignore_errors=True)
                        with self.lock:
                            self.jobs.pop(d.name, None)
                except OSError:
                    pass
            time.sleep(900)


queue = Queue()


def read_progress(job: Job) -> Dict[str, Any]:
    try:
        return json.loads((job.dir / "progress.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def client_ip(req: Request) -> str:
    # uvicorn con --proxy-headers ya pone la IP real (X-Forwarded-For) en req.client
    return req.client.host if req.client else "desconocida"


# --------------------------------------------------------------------------- #
class KeyIn(BaseModel):
    clave: str = Field(..., min_length=10, max_length=200)


class ScriptIn(BaseModel):
    clave: str = Field(..., min_length=10, max_length=200)
    tema: str = Field(..., min_length=3, max_length=300)
    idioma: str = "es"
    formato: str = "9:16"
    duracion: int = Field(45, ge=10, le=120)
    modelo: Optional[str] = None


class ValidateIn(BaseModel):
    script: Any


class RenderIn(BaseModel):
    script: Any
    formato: str = "9:16"
    motor: str = "auto"
    voz: str = "Puck"
    clave: Optional[str] = Field(None, max_length=200)
    modelo_tts: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace("__MAX__", str(MAX_SECONDS)).replace("__LIMITE__", str(LIMIT))
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/api/estado")
def estado(req: Request) -> dict:
    ip = client_ip(req)
    with queue.lock:
        cola = len(queue.pending) + (1 if queue.current else 0)
    return {"cola": cola, "usados": limits.used(ip), "limite": LIMIT, "max_segundos": MAX_SECONDS,
            "idiomas": LANGUAGES, "voces": gemini.TTS_VOICES}


@app.post("/api/modelos")
def modelos(body: KeyIn) -> dict:
    try:
        m = gemini.list_models(body.clave.strip())
    except gemini.GeminiError as e:
        raise HTTPException(400, str(e))
    return {"texto": m.text, "tts": m.tts}


@app.get("/api/ejemplo")
def ejemplo() -> JSONResponse:
    p = Path(__file__).resolve().parents[1] / "assets" / "ejemplo-offline.json"
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.post("/api/guion")
def guion(body: ScriptIn, req: Request) -> dict:
    from ..pipeline import write_script

    if body.formato not in ("9:16", "16:9"):
        raise HTTPException(400, "Formato no válido")
    if not limits.take("guion:" + client_ip(req), SCRIPT_LIMIT):
        raise HTTPException(429, f"Has alcanzado el límite de {SCRIPT_LIMIT} guiones por día.")
    logs: list = []
    try:
        script, warnings, model = write_script(body.clave.strip(), body.tema.strip(), body.idioma,
                                               min(body.duracion, MAX_SECONDS), body.formato, body.modelo,
                                               log=logs.append)
    except gemini.GeminiError as e:
        raise HTTPException(400 if not e.quota else 429, str(e))
    except ScriptError as e:
        raise HTTPException(422, str(e))
    return {"script": json.loads(script.to_json()), "avisos": warnings, "modelo": model}


@app.post("/api/validar")
def validar(body: ValidateIn) -> dict:
    try:
        script, warnings = validate_script(body.script)
    except ScriptError as e:
        return {"ok": False, "errores": e.errors}
    return {"ok": True, "script": json.loads(script.to_json()), "avisos": warnings,
            "segundos_estimados": round(estimate_seconds(script), 1)}


def estimate_seconds(script) -> float:
    wps = WORDS_PER_SEC.get(script.language, 2.5)
    return script.word_count() / wps + 0.6 * len(script.scenes) + 0.6


@app.post("/api/render")
def render(body: RenderIn, req: Request) -> dict:
    try:
        script, _ = validate_script(body.script)
    except ScriptError as e:
        raise HTTPException(422, str(e))
    if body.formato not in ("9:16", "16:9"):
        raise HTTPException(400, "Formato no válido")
    if body.motor not in ("auto", "gemini", "piper"):
        raise HTTPException(400, "Motor de voz no válido")
    if body.voz not in gemini.TTS_VOICES:
        raise HTTPException(400, "Voz no válida")
    est = estimate_seconds(script)
    if est > MAX_SECONDS * 1.1:
        raise HTTPException(400, f"El guion daría unos {est:.0f} s de vídeo; el máximo es {MAX_SECONDS} s. Acórtalo.")
    with queue.lock:
        if len(queue.pending) >= MAX_QUEUE:
            raise HTTPException(503, "La cola está llena. Inténtalo en unos minutos.")
    ip = client_ip(req)
    if not limits.take(ip, LIMIT):
        raise HTTPException(429, f"Has alcanzado el límite de {LIMIT} vídeos por día. Vuelve mañana o "
                                 "instala Stickman IA en tu ordenador (es gratis y open source).")
    jid = secrets.token_urlsafe(12)
    job = Job(jid, ip, (body.clave or "").strip() or None)
    job.dir.mkdir(parents=True, exist_ok=True)
    spec = {"script": json.loads(script.to_json()), "formato": body.formato, "motor": body.motor,
            "voz": body.voz, "modelo_tts": body.modelo_tts, "max_segundos": MAX_SECONDS, "escala": RENDER_SCALE}
    (job.dir / "job.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    pos = queue.add(job)
    return {"id": jid, "posicion": pos}


def _job(jid: str) -> Job:
    job = queue.jobs.get(jid)
    if job is None and jid.replace("-", "").replace("_", "").isalnum() and (JOBS / jid / "progress.json").exists():
        # el servidor se reinició: recuperamos el trabajo terminado desde disco
        job = Job(jid, "", None)
        job.created = (JOBS / jid).stat().st_mtime
        job.status = "terminado"
        st = read_progress(job)
        if st.get("estado") != "terminado":
            job.status, job.message = "error", st.get("mensaje") or "Trabajo interrumpido"
        queue.jobs[jid] = job
    if not job or not job.dir.exists():
        raise HTTPException(404, "Trabajo no encontrado (los archivos se borran a las 24 h).")
    return job


@app.get("/api/trabajo/{jid}")
def trabajo(jid: str) -> dict:
    job = _job(jid)
    st = read_progress(job)
    estado_ = job.status if job.status in ("en_cola", "error") else st.get("estado", job.status)
    out = {"id": jid, "estado": estado_, "posicion": queue.position(jid), "etapa": st.get("etapa", ""),
           "progreso": st.get("progreso", 0.0), "mensaje": st.get("mensaje", job.message),
           "expira": job.created + KEEP_HOURS * 3600}
    if job.status == "error":
        out["mensaje"] = job.message
    if estado_ == "terminado":
        out.update(video=f"api/trabajo/{jid}/video.mp4", srt=f"api/trabajo/{jid}/subtitulos.srt",
                   duracion=st.get("duracion"), voz=st.get("voz"), render_s=st.get("render_s"))
    return out


@app.get("/api/trabajo/{jid}/video.mp4")
def video(jid: str) -> FileResponse:
    p = _job(jid).dir / "video.mp4"
    if not p.exists():
        raise HTTPException(404, "Aún no está listo")
    return FileResponse(p, media_type="video/mp4", filename="stickman-ia.mp4")


@app.get("/api/trabajo/{jid}/subtitulos.srt")
def subtitulos(jid: str) -> FileResponse:
    p = _job(jid).dir / "video.srt"
    if not p.exists():
        raise HTTPException(404, "Aún no está listo")
    return FileResponse(p, media_type="application/x-subrip", filename="stickman-ia.srt")


# aplicación ASGI final (uvicorn stickman.web.app:asgi)
asgi = StripPrefix(app, ROOT_PATH)
