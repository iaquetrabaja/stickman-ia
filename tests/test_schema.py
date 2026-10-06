import json
from pathlib import Path

import pytest

from stickman.schema import ScriptError, parse_text, validate_script

ROOT = Path(__file__).resolve().parents[1]


def minimal(**scene):
    s = {"narration": "Hola mundo."}
    s.update(scene)
    return {"title": "t", "scenes": [s]}


def test_examples_are_valid():
    for p in (ROOT / "examples").glob("*.json"):
        script, warnings = validate_script(json.loads(p.read_text(encoding="utf-8")))
        assert script.scenes and not warnings, (p, warnings)
    text = (ROOT / "examples" / "regla-2-minutos.yaml").read_text(encoding="utf-8")
    script, warnings = validate_script(parse_text(text))
    assert len(script.scenes) == 3 and not warnings


def test_minimal_defaults():
    script, warnings = validate_script(minimal())
    sc = script.scenes[0]
    assert sc.background == "papel" and sc.camera == "static" and sc.characters == []
    assert script.language == "es"


def test_empty_scenes_rejected():
    with pytest.raises(ScriptError):
        validate_script({"title": "x", "scenes": []})
    with pytest.raises(ScriptError):
        validate_script({"title": "x"})
    with pytest.raises(ScriptError):
        validate_script("no soy un objeto")


def test_empty_narration_rejected():
    with pytest.raises(ScriptError) as e:
        validate_script(minimal(narration=""))
    assert any("narration" in m for m in e.value.errors)


def test_aliases_and_unknown_values_are_repaired():
    raw = minimal(characters=[{"id": "Ana", "pose": "pointing", "expression": "smiling",
                               "actions": [{"at": 3, "pose": "breakdance"}]}],
                  props=[{"type": "graph"}, {"type": "unicorn"}], camera="zoom", background="morado",
                  keyword="IDEA")
    script, warnings = validate_script(raw)
    ch = script.scenes[0].characters[0]
    assert ch.id == "ana" and ch.pose == "point" and ch.expression == "happy"
    assert ch.actions[0].at == 1.0 and ch.actions[0].pose is None
    assert [p.type for p in script.scenes[0].props] == ["chart"]
    assert script.scenes[0].camera == "zoom_in"
    assert script.scenes[0].background == "papel"
    assert script.scenes[0].keywords[0].text == "IDEA"
    # el personaje desconocido se añade al reparto y pasa a ser quien habla
    assert [c.id for c in script.cast] == ["ana"]
    assert script.scenes[0].speaker == "ana"
    assert len(warnings) >= 3


def test_numbers_are_clamped():
    script, _ = validate_script(minimal(characters=[{"id": "a", "x": 7}],
                                       props=[{"type": "clock", "x": -9, "at": -1}]))
    assert script.scenes[0].characters[0].x == 1.0
    assert script.scenes[0].props[0].x == -1.2 and script.scenes[0].props[0].at == 0.0


def test_bad_hex_color_repaired():
    raw = minimal()
    raw["cast"] = [{"id": "a", "preset": {"color": "red", "hair": "mohawk"}}]
    script, warnings = validate_script(raw)
    assert script.cast[0].preset.color.startswith("#") and script.cast[0].preset.hair == "short"
    assert warnings


def test_holder_must_be_in_scene():
    script, warnings = validate_script(minimal(characters=["a"], props=[{"type": "phone", "holder": "zz"}]))
    assert script.scenes[0].props[0].holder is None
    assert any("holder" in w for w in warnings)


def test_too_many_characters_trimmed():
    script, warnings = validate_script(minimal(characters=["a", "b", "c", "d"]))
    assert len(script.scenes[0].characters) == 3


def test_parse_text_json_yaml_and_fences():
    assert parse_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_text("a: 1\nb: [1, 2]") == {"a": 1, "b": [1, 2]}
    with pytest.raises(ScriptError):
        parse_text("{: [")


def test_camera_keyframes_clamped():
    script, _ = validate_script(minimal(camera=[{"at": 0, "zoom": 1}, {"at": 1, "zoom": 9, "x": 0.2}]))
    cam = script.scenes[0].camera
    assert isinstance(cam, list) and cam[1].zoom == 2.2
