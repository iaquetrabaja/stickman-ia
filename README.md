# Stickman IA

Generador open source de **vídeos explicativos con muñecos de palo** (stickman) a partir de un tema.
Escribes el tema, Gemini escribe el guion y la puesta en escena, tú lo revisas, y Stickman IA pone la voz,
anima los personajes y te devuelve un **MP4 con subtítulos incrustados + archivo SRT**.

<table>
<tr>
<td align="center"><a href="examples/demo-procrastinacion.mp4"><img src="examples/demo-procrastinacion.gif" width="260" alt="Ejemplo: por qué procrastinamos"></a><br><sub><b>Con Gemini</b> · «Por qué procrastinamos y cómo evitarlo»<br><a href="examples/demo-procrastinacion.mp4">ver vídeo con sonido</a></sub></td>
<td align="center"><a href="examples/demo-offline.mp4"><img src="examples/demo-offline.gif" width="260" alt="Ejemplo sin clave"></a><br><sub><b>Modo sin clave</b> (guion de ejemplo + voz local)<br><a href="examples/demo-offline.mp4">ver vídeo con sonido</a></sub></td>
</tr>
</table>

Pruébalo online gratis en **[tools.iaquetrabaja.com/stickman](https://tools.iaquetrabaja.com/stickman/)** (cuenta gratuita).

### Coste y tiempo aproximados (por 1 minuto de vídeo)

| | Con clave de pago | Con clave gratuita |
|---|---|---|
| Guion y puesta en escena (gemini-3.8-flash) | ~0,01 $ | 0 $ (cuota gratis) |
| Voz, 60 s (gemini-3.8-flash-tts) | ~0,02 $ | 0 $ (cuota gratis) o voz local Piper |
| **Total** | **≈ 0,03 $ (3 céntimos)** | **0 €** |
| **Tiempo de generación** | **≈ 1–2 min** (≈ 40 s de IA + ≈ 1 min de render en 2 núcleos de CPU) | ≈ 1–2 min |

Los muñecos se dibujan en tu ordenador: no se usa ningún modelo de imagen. Solo 3 llamadas a la API por vídeo. Precios de la API de Gemini a octubre de 2026 (Google duplica las tarifas de texto y voz desde el 1 de enero de 2027).

- Vertical **9:16 (1080x1920)** por defecto o horizontal **16:9 (1920x1080)**, 30 fps.
- Español por defecto (también inglés, portugués, francés, italiano y alemán).
- **Trae tu propia clave (BYOK)** de Gemini, gratuita. La clave no se guarda en ningún servidor.
- Sin clave también funciona: modo `--offline` con un guion de ejemplo y voz local **Piper**.
- Render determinista y ligero: un vídeo de 45 s tarda menos de 1 minuto con 2 CPU y ~600 MB de RAM.

Demo generada con una clave real (tema: *"Por qué procrastinamos y cómo evitarlo"*):
[`examples/demo-procrastinacion.mp4`](examples/demo-procrastinacion.mp4)

## Cómo funciona

```
tema ──► Gemini (texto) ──► guion JSON validado ──► revisión/edición ──► voz (Gemini TTS o Piper)
                                                                              │
          MP4 + SRT ◄── ffmpeg (fotogramas por tubería) ◄── render vectorial (skia) ◄┘
```

1. **Guion**: Gemini (se elige automáticamente el mejor modelo *flash* disponible para tu clave con
   `ListModels`) escribe un guion estructurado: escenas, narración, personajes con aspecto (pelo, accesorio,
   color), poses, expresiones, posiciones, entradas andando, objetos, texto clave en pantalla y cámara.
2. **Validación**: el guion se valida contra un esquema (pydantic). Los errores pequeños del LLM se corrigen
   solos (p. ej. `"pointing"` → `"point"`) y se avisa; si hay errores graves se pide una corrección a Gemini.
3. **Revisión**: puedes editar el guion antes de renderizar (paso de la web o `--review` en la CLI).
4. **Voz**: Gemini TTS en **una sola llamada** para toda la narración (ahorra cuota) que luego se trocea por
   escenas usando los silencios. Si no hay modelo TTS, se agota la cuota (429) o falla la red, se usa
   **Piper** en local (la voz `es_ES-davefx-medium` se descarga sola la primera vez, ~60 MB).
5. **Subtítulos sincronizados**: tras la voz, un reconocimiento de voz local
   ([faster-whisper](https://github.com/SYSTRAN/faster-whisper) `base`, int8, CPU) da el instante de cada
   palabra; esas marcas se alinean con las palabras del **guion** (que es lo que se muestra, sin faltas del
   reconocedor) y las que no se reconocen se interpolan. Los subtítulos (2-5 palabras, sin cruzar frases)
   aparecen 80 ms antes de que empiece la voz y el SRT usa los mismos tiempos. El modelo (~140 MB) se
   descarga la primera vez; si no está disponible, los tiempos se estiman por longitud del texto.
6. **Render**: muñecos vectoriales con antialias (skia), poses interpoladas con *easing*, respiración,
   parpadeo, boca sincronizada con la amplitud del audio, gestos al hablar, ciclo de andar, saltos,
   objetos que aparecen con rebote, zoom/paneo de cámara, texto clave con subrayado naranja y subtítulos
   grandes centrados (blanco con contorno negro). Los fotogramas se envían a ffmpeg por tubería
   (no se guardan en disco ni en RAM).

Llamadas a la API por vídeo con clave: **3** (ListModels + guion + TTS). Si el guion no valida, 1 más.

### Lo que puede usar el guion

| Elemento | Valores |
|---|---|
| Poses | `idle`, `talk`, `wave`, `point`, `think`, `shrug`, `cheer`, `arms_crossed`, `present`, `sad`, `sit`, `jump` |
| Expresiones | `neutral`, `happy`, `sad`, `surprised`, `thinking`, `angry` |
| Pelo | `none`, `short`, `long`, `bun`, `ponytail`, `curly`, `spiky` |
| Accesorios | `none`, `glasses`, `cap`, `headphones`, `tie`, `bow` |
| Objetos | `laptop`, `phone`, `coffee`, `desk`, `chair`, `speech_bubble`, `thought_bubble`, `chart`, `money`, `clock`, `lightbulb` |
| Cámara | `static`, `zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `punch_in` o keyframes `{at, zoom, x, y}` |
| Fondos | `papel`, `blanco`, `arena`, `menta`, `cielo`, `melocoton`, `lavanda`, `noche` o `#RRGGBB` |

Los tiempos (`at`) son fracciones de la escena (0 = inicio, 1 = final), porque la duración real la marca la
voz. Mira [`examples/ia-en-tu-trabajo.json`](examples/ia-en-tu-trabajo.json) y
[`examples/regla-2-minutos.yaml`](examples/regla-2-minutos.yaml) (JSON y YAML valen igual).

## Instalación local

Necesitas **Python 3.10+** (probado con 3.12) y **ffmpeg** en el PATH.

### Windows

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
git clone <url-del-repo> stickman-ia
cd stickman-ia
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS

```bash
brew install python@3.12 ffmpeg
git clone <url-del-repo> stickman-ia && cd stickman-ia
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Linux (Debian/Ubuntu)

```bash
sudo apt install python3 python3-venv ffmpeg libgl1 libegl1 libfontconfig1
git clone <url-del-repo> stickman-ia && cd stickman-ia
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Comprueba que todo va bien sin clave:

```bash
python -m stickman --offline
# -> salida/la-ia-no-te-quita-el-trabajo-todavia.mp4 y .srt
```

## Clave gratuita de Gemini

1. Entra en [Google AI Studio → API keys](https://aistudio.google.com/apikey) con tu cuenta de Google.
2. Pulsa **Create API key** y cópiala.
3. Úsala con `--clave` o en la variable de entorno `GEMINI_API_KEY` (recomendado, así no queda en el
   historial de la terminal).

El nivel gratuito tiene límites por minuto y por día, sobre todo en los modelos de voz. Si se agotan,
Stickman IA te lo dice y usa la voz local Piper automáticamente.

## Uso por línea de comandos

```bash
# Windows PowerShell: $env:GEMINI_API_KEY="tu_clave"   ·   macOS/Linux: export GEMINI_API_KEY=tu_clave
python -m stickman "Por qué procrastinamos y cómo evitarlo" --formato 9:16 --duracion 45

# Revisar y editar el guion antes de renderizar (abre el editor y espera a que pulses Enter)
python -m stickman "Cómo funciona el interés compuesto" --review

# Solo el guion (para editarlo con calma) y luego renderizarlo
python -m stickman "Tema" --solo-guion --salida mis-guiones
python -m stickman --guion mis-guiones/tema.guion.json --formato 16:9

# Sin clave: guion de ejemplo + Piper (o cualquier guion propio con --motor-voz piper)
python -m stickman --offline
python -m stickman --guion examples/regla-2-minutos.yaml --formato 16:9 --motor-voz piper

# Con una narración ya grabada (WAV, MP3 o el MP4 de un vídeo anterior): sin TTS
python -m stickman --guion mi-guion.json --audio narracion.wav
```

Opciones útiles: `--idioma en`, `--voz Kore` (30 voces de Gemini; en la web se pueden probar antes de generar: Puck (animada), Charon (informativa), Kore (firme), Sulafat (cálida), Achird (cercana),
Leda, Zephyr...), `--motor-voz auto|gemini|piper`, `--modelo` / `--modelo-tts` para forzar modelos,
`--escala 0.667` (renderiza a 720p y reescala con ffmpeg, más rápido en máquinas lentas), `--salida`,
`--nombre`, `--sin-alineacion` (subtítulos con tiempos estimados, sin Whisper). La duración máxima es 120 s. La duración final es aproximada (±20 %): depende del ritmo de la voz.

## Interfaz web

```bash
python -m stickman web --puerto 8000
# abre http://127.0.0.1:8000/
```

Pasos: clave (opcionalmente recordada **solo en tu navegador**) → tema, idioma, formato y duración →
revisar/editar el guion (narración, texto en pantalla, fondo, cámara o el JSON completo) → voz → crear vídeo.
Muestra la posición en la cola y el progreso, y permite descargar MP4 y SRT.

Reglas del servidor (configurables con variables de entorno):

| Variable | Por defecto | Qué hace |
|---|---|---|
| `ROOT_PATH` | vacío | Prefijo público, p. ej. `/stickman`. Funciona tanto si el proxy quita el prefijo como si no. |
| `STICKMAN_DATA` | `~/.cache/stickman-ia` | Trabajos, límites y voces Piper |
| `LIMITE_DIARIO` | `3` | Vídeos por IP y día (un render fallido no gasta cupo) |
| `LIMITE_GUIONES` | `20` | Guiones por IP y día |
| `MAX_SEGUNDOS` | `120` | Duración máxima del vídeo |
| `MAX_COLA` | `20` | Trabajos máximos en cola |
| `HORAS_BORRADO` | `24` | Los archivos se borran pasado este tiempo |
| `ESCALA_RENDER` | `1.0` | `0.667` para renderizar a 720p y reescalar |
| `STICKMAN_WHISPER_MODEL` | `base` | Modelo de alineación de subtítulos (`base` o `small`) |
| `STICKMAN_WHISPER_DIR` | `$STICKMAN_DATA/whisper` | Dónde se guarda el modelo (en Docker va dentro de la imagen) |

Se renderiza **un vídeo a la vez** (en un proceso aparte); la clave del usuario pasa al proceso de render
por stdin, solo vive en memoria y nunca se escribe en disco ni en los logs.

### Docker (servidor)

```bash
docker build -t stickman-ia .
docker run -d --name stickman -p 127.0.0.1:8000:8000 --cpus 2 --memory 1g \
  -e ROOT_PATH=/stickman -v stickman-data:/data stickman-ia
```

Detrás de un proxy (nginx, Caddy, Traefik) sirviendo `https://tu-dominio/stickman/`. Ejemplo nginx:

```nginx
location /stickman/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_read_timeout 180s;   # escribir el guion puede tardar 30-60 s
    client_max_body_size 1m;
}
```

Importante: el contenedor confía en `X-Forwarded-For` para el límite por IP, así que publícalo solo
detrás del proxy (como en el ejemplo, en `127.0.0.1`). Usa **un único worker** de uvicorn (la cola vive en
memoria).

## Rendimiento medido

Windows, CPU limitada a 2 núcleos (afinidad), 9:16 1080x1920 a 30 fps:

| Prueba | Vídeo | Render (fotogramas + ffmpeg) | Total |
|---|---|---|---|
| `--offline` (Piper, incluye cargar la voz) | 24,9 s | 28 s | 40 s |
| Guion de 6 escenas, Piper | 38,8 s | 44 s | 57 s |
| Clave real (Gemini 3.8 flash + 3.8 flash TTS) | 45,2 s | 47 s | 92 s (incluye ~40 s de Gemini) |

Memoria pico: ~350 MB Python (con Piper cargado) + ~300 MB ffmpeg. La alineación de subtítulos añade
~3-6 s por vídeo y ~150 MB (modelo Whisper `base` int8).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Cubren la validación/saneado del esquema, el tweening (continuidad, ángulos por el camino corto, cámara),
la cinemática (huesos que no se estiran, pies en el suelo, IK), el audio (envolvente, silencios, troceado,
subtítulos, SRT), la selección de modelos, el *fallback* a Piper y un render de fotogramas en ambos formatos.

## Estructura

```
stickman/
  cli.py, __main__.py   línea de comandos
  pipeline.py           tema → guion → voz → render
  schema.py             esquema del guion + saneado tolerante
  prompt.py             prompt para Gemini
  gemini.py             cliente REST mínimo (ListModels, texto JSON, TTS)
  tts.py                Gemini TTS / Piper
  audio.py              WAV, envolvente, silencios, subtítulos, SRT
  align.py              subtítulos alineados palabra a palabra (faster-whisper + guion)
  render.py             escenas, cámara, props, textos y salida a ffmpeg
  props.py              objetos vectoriales
  text.py               texto (Roboto incluida)
  tween.py              easing y pistas de keyframes
  rig/                  esqueleto, poses, caras y dibujo del muñeco
  web/                  FastAPI + interfaz + proceso de render
examples/               guiones de ejemplo y vídeo demo
tests/
```

## Créditos

- Inspirado y parcialmente portado de **[stickman-explainers](https://github.com/21cabbagee/stickman-explainers)**
  de 21cabbagee (MIT): rig de cinemática directa, poses, tweening, cámara, boca por amplitud, reparto de
  tiempos de subtítulos, línea de tiempo y las pautas de retención del prompt. El aviso de copyright original
  está en [`LICENSE`](LICENSE).
- Ideas de diseño (presets de personaje, expresiones, objetos, enfoque en dos fases: el LLM produce datos
  estructurados y un renderizador determinista dibuja) tomadas de
  [stickman-animation-agent](https://github.com/chrisaswain/stickman-animation-agent). No se ha copiado su
  código ni sus SVG: todos los dibujos de este proyecto son propios.
- Alineación de subtítulos: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) con los modelos
  Whisper de OpenAI (MIT).
- Voz local: [Piper](https://github.com/rhasspy/piper) y sus voces (cada una con su licencia).
- Fuente: Roboto (Apache 2.0).

## Licencia

MIT. Consulta [`LICENSE`](LICENSE).
