"""Prompt para que Gemini escriba el guion de escenas.

Las técnicas de retención (gancho, especificidad, bucle abierto, una idea por
escena, segunda persona, payoff y cierre) proceden de src/script/promptBuilder.js
de stickman-explainers (MIT, Copyright (c) 2026 21cabbagee); el formato de
salida (escenas con personajes, poses, props, cámara) es propio.
"""
from __future__ import annotations

from .schema import (ACCESSORIES, BACKGROUNDS, CAMERA_PRESETS, EXPRESSIONS, HAIR, LANGUAGES, POSES,
                     PROPS)

WORDS_PER_SEC = {"es": 2.1, "en": 2.3, "pt": 2.1, "fr": 2.2, "it": 2.1, "de": 1.9}


def scene_count(duration: int) -> int:
    return max(3, min(14, round(duration / 6.5)))


def build_prompt(topic: str, language: str = "es", duration: int = 45, fmt: str = "9:16") -> str:
    lang = LANGUAGES.get(language, language)
    words = int(duration * WORDS_PER_SEC.get(language, 2.5))
    n = scene_count(duration)
    return f"""Eres guionista experto de vídeos cortos explicativos (TikTok, Shorts, Reels) y director de animación
de un estilo minimalista de muñecos de palo ("stickman"). Escribe el guion y la puesta en escena.

TEMA: {topic}
IDIOMA DE LA NARRACIÓN Y TEXTOS: {lang} (código "{language}")
DURACIÓN OBJETIVO: unos {duration} segundos de voz => en total {words - 10}-{words + 5} palabras de narración.
ESCENAS: exactamente {n}. FORMATO: {fmt}.

Técnicas de retención:
- GANCHO en la primera frase (afirmación sorprendente, dato concreto o pregunta que abre curiosidad). Nunca
  empieces con "¿Alguna vez te has preguntado?" ni "Hoy vamos a hablar de".
- ESPECIFICIDAD: números, ejemplos y detalles concretos, sin inventar datos ni estadísticas dudosas.
- BUCLE ABIERTO al principio que se resuelve más adelante.
- UNA IDEA POR ESCENA, frases cortas y naturales para leer en voz alta, en segunda persona ("tú").
- PAYOFF o momento "ajá" a mitad, y CIERRE con una frase memorable (no "gracias por ver").
- Contenido apto para todos los públicos y riguroso.

Puesta en escena (deja que sea variada pero sencilla y legible):
- "cast": 1 o 2 personajes recurrentes (máx 3). Cada uno con id corto en minúsculas, name y preset:
  hair: {HAIR}; accessory: {ACCESSORIES}; color: color de camiseta #RRGGBB (usa colores sobrios y distintos
  entre personajes, por ejemplo #D24D1E, #2F6FB0, #2E8B6A, #7A4FB5, #C9932B).
- En cada escena: "characters" (1-2 normalmente) con x entre -1 (izquierda) y 1 (derecha); si hay 2 personajes
  usa x de -0.5 y 0.5 aprox; uno solo, x entre -0.45 y 0. "facing": "left"/"right" (opcional).
  "enter": none/left/right/pop (entrar andando desde un lado o aparecer), "exit": none/left/right.
  "pose" inicial y "expression" inicial. "actions": lista de cambios con "at" (fracción 0-1 de la escena),
  y opcionalmente "pose", "expression", "move_to" (x destino, camina), "facing".
- poses: {POSES}. ("jump" es un salto puntual; "sit" se sienta en un taburete; "think" mano en la barbilla.)
- expressions: {EXPRESSIONS}. Cambia expresión y pose 1-3 veces por escena para que haya vida.
- "speaker": id del personaje que habla en la escena (su boca se moverá).
- "props" (0-3 por escena): type en {PROPS}. Campos: x (-1..1), at (0-1, cuándo aparece), until (opcional),
  holder (id del personaje que lo sostiene: phone/coffee/money en la mano; speech_bubble, thought_bubble,
  lightbulb, chart y clock junto a su cabeza), text (máx 4 palabras, para bocadillos), trend (up/down para chart).
  Sin holder, los objetos flotan como iconos en x (deja libre el sitio de los personajes). desk/chair van en el suelo;
  laptop/coffee cerca de una desk se colocan encima.
- "keywords": 1 texto clave por escena (2-4 palabras, impactante, NO repitas la frase entera) con "at".
- "background": uno de {list(BACKGROUNDS)} (varía con criterio, "noche" solo para momentos dramáticos).
- "camera": uno de {CAMERA_PRESETS} (usa zoom_in o punch_in en los momentos clave, static en otros).

Responde SOLO con JSON válido (sin markdown) con esta forma exacta:
{{
  "title": "título corto y potente",
  "language": "{language}",
  "description": "1-2 frases + 3-5 hashtags para publicar",
  "cast": [{{"id": "ana", "name": "Ana", "preset": {{"hair": "long", "accessory": "glasses", "color": "#2F6FB0"}}}}],
  "scenes": [
    {{
      "id": "s1",
      "narration": "texto que se lee en voz alta en esta escena",
      "speaker": "ana",
      "background": "papel",
      "camera": "static",
      "keywords": [{{"text": "TEXTO CLAVE", "at": 0.1}}],
      "characters": [{{"id": "ana", "x": -0.3, "enter": "left", "pose": "wave", "expression": "happy",
                      "actions": [{{"at": 0.4, "pose": "talk"}}, {{"at": 0.7, "expression": "surprised", "pose": "point"}}]}}],
      "props": [{{"type": "lightbulb", "x": 0.55, "at": 0.5}}]
    }}
  ]
}}
"scenes" debe tener exactamente {n} elementos en orden de narración."""


def build_repair_prompt(original: str, errors: list) -> str:
    return ("El JSON anterior tiene estos errores de esquema:\n- " + "\n- ".join(errors[:20]) +
            "\n\nDevuélvelo corregido, completo, solo JSON válido sin comentarios:\n" + original[:20000])
