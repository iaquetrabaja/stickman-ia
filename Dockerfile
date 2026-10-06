# Stickman IA — imagen para el servidor web (un render a la vez, ~1 GB RAM, 2 CPU)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STICKMAN_DATA=/data \
    STICKMAN_PIPER_DIR=/opt/piper \
    STICKMAN_WHISPER_DIR=/opt/whisper \
    STICKMAN_WHISPER_MODEL=base \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 \
    ROOT_PATH=""

# ffmpeg + librerías que necesita skia-python en Linux + fuentes de respaldo
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libfontconfig1 libfreetype6 libgl1 libegl1 fonts-roboto ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Modelo Whisper para alinear los subtítulos con la voz, dentro de la imagen
# (no depende de la red ni retrasa la primera petición)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', download_root='/opt/whisper')"
ENV HF_HUB_OFFLINE=1

COPY stickman ./stickman
COPY examples ./examples

# Voz Piper es_ES descargada en la imagen (no depende de la red al arrancar)
RUN python -c "from stickman.tts import ensure_piper_voice; ensure_piper_voice('es')" \
    && useradd -m -u 1000 stickman && mkdir -p /data && chown -R stickman /data /opt/piper /opt/whisper

USER stickman
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/estado', timeout=4)" || exit 1

# Un único worker: la cola de render vive en memoria del proceso
CMD ["uvicorn", "stickman.web.app:asgi", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
