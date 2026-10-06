"""Esquema del guion de escenas (JSON/YAML) y validación.

Inspirado en src/sceneScript/schema.js de stickman-explainers (MIT,
Copyright (c) 2026 21cabbagee), ampliado con reparto, expresiones, accesorios,
props, textos clave y cámara por presets. Los tiempos de acciones se expresan
como fracción (0-1) de la duración de la escena, porque la duración real la
decide el TTS.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

POSES = ["idle", "talk", "wave", "point", "think", "shrug", "cheer", "arms_crossed",
         "present", "sad", "sit", "jump"]
EXPRESSIONS = ["neutral", "happy", "sad", "surprised", "thinking", "angry"]
HAIR = ["none", "short", "long", "bun", "ponytail", "curly", "spiky"]
ACCESSORIES = ["none", "glasses", "cap", "headphones", "tie", "bow"]
PROPS = ["laptop", "phone", "coffee", "desk", "chair", "speech_bubble", "thought_bubble",
         "chart", "money", "clock", "lightbulb"]
CAMERA_PRESETS = ["static", "zoom_in", "zoom_out", "pan_left", "pan_right", "punch_in"]
EASES = ["linear", "ease_in_out", "ease_out", "ease_out_back"]
ENTER = ["none", "left", "right", "pop"]
EXIT = ["none", "left", "right"]
BACKGROUNDS = {
    "papel": "#F2EEE6",
    "blanco": "#FAFAF8",
    "arena": "#EFE4D2",
    "menta": "#DDEEE5",
    "cielo": "#DCE7F2",
    "melocoton": "#F6DFD2",
    "lavanda": "#E6E1F0",
    "noche": "#1F242B",
}
CHAR_COLORS = ["#D24D1E", "#2F6FB0", "#2E8B6A", "#7A4FB5", "#C9932B", "#3A3A3A"]
LANGUAGES = {"es": "español", "en": "English", "pt": "português", "fr": "français",
             "it": "italiano", "de": "Deutsch"}

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
ID_RE = re.compile(r"^[a-z0-9_\-]{1,24}$")


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Preset(_Base):
    hair: Literal[tuple(HAIR)] = "short"  # type: ignore[valid-type]
    accessory: Literal[tuple(ACCESSORIES)] = "none"  # type: ignore[valid-type]
    color: str = "#D24D1E"
    hair_color: str = "#22201E"

    @field_validator("color", "hair_color")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not HEX_RE.match(v):
            raise ValueError(f"color hexadecimal inválido: {v!r} (usa #RRGGBB)")
        return v.upper()


class CastMember(_Base):
    id: str
    name: str = ""
    preset: Preset = Field(default_factory=Preset)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError(f"id inválido {v!r}: minúsculas, números, _ o - (máx 24)")
        return v


class Action(_Base):
    at: float = Field(0.0, ge=0.0, le=1.0)
    pose: Optional[Literal[tuple(POSES)]] = None  # type: ignore[valid-type]
    expression: Optional[Literal[tuple(EXPRESSIONS)]] = None  # type: ignore[valid-type]
    move_to: Optional[float] = Field(None, ge=-1.0, le=1.0)
    facing: Optional[Literal["left", "right"]] = None


class SceneCharacter(_Base):
    id: str
    x: float = Field(0.0, ge=-1.0, le=1.0)
    facing: Optional[Literal["left", "right"]] = None
    enter: Literal[tuple(ENTER)] = "none"  # type: ignore[valid-type]
    exit: Literal[tuple(EXIT)] = "none"  # type: ignore[valid-type]
    pose: Literal[tuple(POSES)] = "idle"  # type: ignore[valid-type]
    expression: Literal[tuple(EXPRESSIONS)] = "neutral"  # type: ignore[valid-type]
    actions: List[Action] = Field(default_factory=list, max_length=12)


class Prop(_Base):
    type: Literal[tuple(PROPS)]  # type: ignore[valid-type]
    x: float = Field(0.55, ge=-1.2, le=1.2)
    y: Optional[float] = Field(None, ge=0.0, le=1.6)
    at: float = Field(0.0, ge=0.0, le=1.0)
    until: Optional[float] = Field(None, ge=0.0, le=1.0)
    holder: Optional[str] = None
    text: Optional[str] = Field(None, max_length=48)
    trend: Literal["up", "down"] = "up"


class Keyword(_Base):
    text: str = Field(..., min_length=1, max_length=48)
    at: float = Field(0.0, ge=0.0, le=1.0)
    until: Optional[float] = Field(None, ge=0.0, le=1.0)


class CameraKey(_Base):
    at: float = Field(0.0, ge=0.0, le=1.0)
    zoom: float = Field(1.0, ge=0.7, le=2.2)
    x: float = Field(0.0, ge=-1.0, le=1.0)
    y: float = Field(0.0, ge=-1.0, le=1.0)
    ease: Literal[tuple(EASES)] = "ease_in_out"  # type: ignore[valid-type]


class Scene(_Base):
    id: str = ""
    narration: str = Field(..., min_length=1, max_length=700)
    speaker: Optional[str] = None
    background: str = "papel"
    characters: List[SceneCharacter] = Field(default_factory=list, max_length=3)
    props: List[Prop] = Field(default_factory=list, max_length=6)
    keywords: List[Keyword] = Field(default_factory=list, max_length=3)
    camera: Union[Literal[tuple(CAMERA_PRESETS)], List[CameraKey]] = "static"  # type: ignore[valid-type]

    @field_validator("background")
    @classmethod
    def _bg(cls, v: str) -> str:
        if v in BACKGROUNDS:
            return v
        if HEX_RE.match(v):
            return v.upper()
        raise ValueError(f"fondo inválido {v!r}: usa {', '.join(BACKGROUNDS)} o #RRGGBB")


class Script(_Base):
    title: str = Field("Vídeo", max_length=120)
    language: str = "es"
    cast: List[CastMember] = Field(default_factory=list, max_length=4)
    scenes: List[Scene] = Field(..., min_length=1, max_length=24)
    description: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)

    def word_count(self) -> int:
        return sum(len(s.narration.split()) for s in self.scenes)

    def to_json(self, **kw: Any) -> str:
        return self.model_dump_json(indent=2, exclude_none=True, **kw)


class ScriptError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("Guion inválido:\n- " + "\n- ".join(errors))
        self.errors = errors


# --------------------------------------------------------------------------- #
# Saneado tolerante (los LLM cometen errores pequeños): corrige lo corregible y
# deja avisos; lo estructural se valida después con pydantic.
# --------------------------------------------------------------------------- #
_POSE_ALIAS = {"talking": "talk", "speak": "talk", "standing": "idle", "stand": "idle",
               "pointing": "point", "waving": "wave", "thinking": "think", "celebrate": "cheer",
               "celebrating": "cheer", "crossed": "arms_crossed", "arms-crossed": "arms_crossed",
               "presenting": "present", "explain": "present", "sitting": "sit", "jumping": "jump",
               "walk": "idle", "walking": "idle", "shrugging": "shrug", "sadness": "sad"}
_EXPR_ALIAS = {"smile": "happy", "smiling": "happy", "excited": "happy", "joy": "happy",
               "confused": "thinking", "curious": "thinking", "worried": "sad", "shocked": "surprised",
               "surprise": "surprised", "mad": "angry", "frustrated": "angry", "calm": "neutral",
               "think": "thinking", "focused": "neutral"}


def _num(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(lo, min(hi, f))


def _enum(v: Any, allowed: List[str], alias: dict, default: Optional[str], where: str,
          warnings: List[str]) -> Optional[str]:
    if v is None:
        return default
    s = str(v).strip().lower().replace(" ", "_")
    s = alias.get(s, s)
    if s in allowed:
        return s
    warnings.append(f"{where}: valor {v!r} desconocido, uso {default!r}")
    return default


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9_\-]", "", str(s).lower().replace(" ", "_"))[:24]
    return s or "p"


def sanitize(raw: Any) -> Tuple[dict, List[str]]:
    w: List[str] = []
    if not isinstance(raw, dict):
        raise ScriptError(["el guion debe ser un objeto JSON con 'scenes'"])
    d = copy.deepcopy(raw)
    scenes = d.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ScriptError(["'scenes' debe ser una lista con al menos una escena"])
    d["language"] = str(d.get("language") or "es")[:5]

    cast = d.get("cast") if isinstance(d.get("cast"), list) else []
    cast_ids = []
    for i, c in enumerate(cast):
        if not isinstance(c, dict):
            continue
        c["id"] = _slug(c.get("id") or c.get("name") or f"p{i + 1}")
        p = c.get("preset") if isinstance(c.get("preset"), dict) else {}
        p["hair"] = _enum(p.get("hair"), HAIR, {"bald": "none"}, "short", f"cast[{i}].hair", w)
        p["accessory"] = _enum(p.get("accessory"), ACCESSORIES, {"hat": "cap", "headset": "headphones"},
                               "none", f"cast[{i}].accessory", w)
        for key, dflt in (("color", CHAR_COLORS[i % len(CHAR_COLORS)]), ("hair_color", "#22201E")):
            if not HEX_RE.match(str(p.get(key, ""))):
                if key in p:
                    w.append(f"cast[{i}].{key}: {p.get(key)!r} no es #RRGGBB, uso {dflt}")
                p[key] = dflt
        c["preset"] = p
        cast_ids.append(c["id"])
    d["cast"] = cast

    for si, s in enumerate(scenes):
        if not isinstance(s, dict):
            raise ScriptError([f"scenes[{si}] debe ser un objeto"])
        where = f"scenes[{si}]"
        s["id"] = str(s.get("id") or f"s{si + 1}")
        if isinstance(s.get("narration"), str):
            s["narration"] = re.sub(r"\s+", " ", s["narration"]).strip()
        bg = str(s.get("background") or "papel").strip()
        if bg.lower() in BACKGROUNDS:
            bg = bg.lower()
        elif not HEX_RE.match(bg):
            w.append(f"{where}.background: {bg!r} no válido, uso 'papel'")
            bg = "papel"
        s["background"] = bg
        # keywords: acepta string, lista de strings o lista de objetos
        kws = s.get("keywords", s.pop("keyword", None))
        if isinstance(kws, str):
            kws = [{"text": kws}]
        if isinstance(kws, list):
            kws = [k if isinstance(k, dict) else {"text": str(k)} for k in kws]
            for k in kws:
                k["text"] = str(k.get("text", "")).strip()[:48]
                k["at"] = _num(k.get("at"), 0, 1, 0.0)
                if k.get("until") is not None:
                    k["until"] = _num(k.get("until"), 0, 1, 1.0)
            kws = [k for k in kws if k["text"]][:3]
        s["keywords"] = kws or []
        # cámara
        cam = s.get("camera", "static")
        if isinstance(cam, str):
            cam = _enum(cam, CAMERA_PRESETS, {"zoom": "zoom_in", "punch": "punch_in", "none": "static"},
                        "static", f"{where}.camera", w)
        elif isinstance(cam, list):
            cam = [{"at": _num(k.get("at"), 0, 1, 0), "zoom": _num(k.get("zoom", 1), 0.7, 2.2, 1),
                    "x": _num(k.get("x", 0), -1, 1, 0), "y": _num(k.get("y", 0), -1, 1, 0),
                    "ease": _enum(k.get("ease"), EASES, {}, "ease_in_out", f"{where}.camera.ease", w)}
                   for k in cam if isinstance(k, dict)] or "static"
        else:
            cam = "static"
        s["camera"] = cam
        # personajes
        chars = s.get("characters") if isinstance(s.get("characters"), list) else []
        if len(chars) > 3:
            w.append(f"{where}: máximo 3 personajes por escena, recorto")
            chars = chars[:3]
        for ci, ch in enumerate(chars):
            cw = f"{where}.characters[{ci}]"
            if isinstance(ch, str):
                ch = {"id": ch}
                chars[ci] = ch
            ch["id"] = _slug(ch.get("id", f"p{ci + 1}"))
            if ch["id"] not in cast_ids:
                w.append(f"{cw}: '{ch['id']}' no está en el reparto, lo añado con aspecto por defecto")
                n = len(cast)
                cast.append({"id": ch["id"], "name": ch["id"].capitalize(),
                             "preset": {"hair": "short", "accessory": "none",
                                        "color": CHAR_COLORS[n % len(CHAR_COLORS)], "hair_color": "#22201E"}})
                cast_ids.append(ch["id"])
            ch["x"] = _num(ch.get("x", 0), -1, 1, 0)
            if ch.get("facing") not in (None, "left", "right"):
                ch["facing"] = None
            ch["enter"] = _enum(ch.get("enter"), ENTER, {"walk_left": "left", "walk_right": "right"},
                                "none", f"{cw}.enter", w)
            ch["exit"] = _enum(ch.get("exit"), EXIT, {"walk_left": "left", "walk_right": "right"},
                               "none", f"{cw}.exit", w)
            ch["pose"] = _enum(ch.get("pose"), POSES, _POSE_ALIAS, "idle", f"{cw}.pose", w)
            ch["expression"] = _enum(ch.get("expression"), EXPRESSIONS, _EXPR_ALIAS, "neutral",
                                     f"{cw}.expression", w)
            acts = ch.get("actions") if isinstance(ch.get("actions"), list) else []
            clean = []
            for ai, a in enumerate(acts[:12]):
                if not isinstance(a, dict):
                    continue
                aw = f"{cw}.actions[{ai}]"
                na = {"at": _num(a.get("at", 0), 0, 1, 0)}
                if a.get("pose") is not None:
                    na["pose"] = _enum(a["pose"], POSES, _POSE_ALIAS, None, aw + ".pose", w)
                if a.get("expression") is not None:
                    na["expression"] = _enum(a["expression"], EXPRESSIONS, _EXPR_ALIAS, None,
                                             aw + ".expression", w)
                if a.get("move_to") is not None:
                    na["move_to"] = _num(a["move_to"], -1, 1, ch["x"])
                if a.get("facing") in ("left", "right"):
                    na["facing"] = a["facing"]
                clean.append(na)
            ch["actions"] = clean
        s["characters"] = chars
        ids_here = [c["id"] for c in chars]
        spk = s.get("speaker")
        if spk is not None:
            spk = _slug(spk)
            if spk not in ids_here:
                if spk not in ("narrador", "narrator", "none", "ninguno"):
                    w.append(f"{where}.speaker: '{spk}' no está en la escena")
                spk = None
        elif ids_here:
            spk = ids_here[0]
        s["speaker"] = spk
        # props
        props = s.get("props") if isinstance(s.get("props"), list) else []
        clean_p = []
        for pi, p in enumerate(props):
            if isinstance(p, str):
                p = {"type": p}
            if not isinstance(p, dict):
                continue
            pw = f"{where}.props[{pi}]"
            t = _enum(p.get("type"), PROPS, {"bubble": "speech_bubble", "thought": "thought_bubble",
                                             "graph": "chart", "computer": "laptop", "mobile": "phone",
                                             "cash": "money", "idea": "lightbulb", "bulb": "lightbulb",
                                             "table": "desk", "mug": "coffee", "cup": "coffee"},
                      None, pw + ".type", w)
            if t is None:
                continue
            np_ = {"type": t, "x": _num(p.get("x", 0.55), -1.2, 1.2, 0.55),
                   "at": _num(p.get("at", 0), 0, 1, 0), "trend": "down" if p.get("trend") == "down" else "up"}
            if p.get("y") is not None:
                np_["y"] = _num(p["y"], 0, 1.6, 0.8)
            if p.get("until") is not None:
                np_["until"] = _num(p["until"], 0, 1, 1)
            if p.get("text"):
                np_["text"] = str(p["text"]).strip()[:48]
            h = p.get("holder")
            if h:
                h = _slug(h)
                if h in ids_here:
                    np_["holder"] = h
                else:
                    w.append(f"{pw}.holder: '{h}' no está en la escena, lo ignoro")
            clean_p.append(np_)
        if len(clean_p) > 6:
            w.append(f"{where}: máximo 6 props, recorto")
        s["props"] = clean_p[:6]
    if not cast:
        d["cast"] = []
    return d, w


def validate_script(raw: Any) -> Tuple[Script, List[str]]:
    """Sanea y valida. Devuelve (Script, avisos). Lanza ScriptError si no se puede."""
    d, warnings = sanitize(raw)
    try:
        script = Script.model_validate(d)
    except ValidationError as e:
        errs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errs.append(f"{loc}: {err['msg']}")
        raise ScriptError(errs) from None
    return script, warnings


def parse_text(text: str) -> Any:
    """Acepta JSON o YAML (y quita vallas ``` si las hay)."""
    t = text.strip()
    t = re.sub(r"^```(?:json|yaml|yml)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        import yaml
        try:
            return yaml.safe_load(t)
        except yaml.YAMLError as e:
            raise ScriptError([f"no es JSON ni YAML válido: {e}"]) from None


def load_script(path: str | Path) -> Tuple[Script, List[str]]:
    return validate_script(parse_text(Path(path).read_text(encoding="utf-8")))


def background_hex(name: str) -> str:
    return BACKGROUNDS.get(name, name)
