"""Staging (D-063): from where does the show look at its people, and from where do the drawn frames?

  python -m src.v3.staging train              # label ~600 ordinary training frames with people (Gemini Flash) -> data/processed/v3/staging_train.json
  python -m src.v3.staging run --run ep22_v3e # label a run's first frames the same way, print both mixes and their distance
  python -m src.v3.staging direct --run ep22_v3f   # camera pass: every shot gets an angle, a view and (sometimes) a foreground

Blind raters name "flat, frontal, evenly spaced staging" as a giveaway (RESULTS M6): the show cuts to over-the-shoulder views,
profiles, backs, high and low angles; the painter puts everybody side by side, facing the camera, at eye level. This module
measures that with one labeller on both sides, so that the shot plan can ask for the show's mix and the painter be told the
angle in words. Training episodes only; the held-out episode is never labelled for this.
"""
from __future__ import annotations

import argparse
import collections
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.llm.client import vision, writer

TRAIN = P("processed") / "v3" / "staging_train.json"
ANGLES = ["eye level", "high angle", "low angle", "top-down"]
VIEWS = ["frontal", "three-quarter", "profile", "from behind", "over the shoulder"]
SYS = ("You label the camera staging of anime frames. For each numbered frame give: angle (the height of the camera: one of the list), "
       "view (how the MAIN person is seen: frontal = facing the camera; three-quarter; profile = side view; from behind; over the shoulder "
       "= past the back or shoulder of a person in the foreground toward someone or something else), foreground (true when a person or "
       "object sits large and close in front of the main subject, usually cut by the frame edge), soft_background (true when the "
       "background is clearly out of focus or blown out behind the people), and people (how many people are clearly visible).\n"
       'Answer with one JSON object: {"frames": [{"n": 1, "angle": "...", "view": "...", "foreground": false, "soft_background": false, "people": 1}, ...]}')
SCHEMA = {"type": "object", "required": ["frames"], "properties": {"frames": {"type": "array", "items": {
    "type": "object", "required": ["n", "angle", "view", "foreground", "soft_background", "people"], "properties": {
        "n": {"type": "integer"}, "angle": {"type": "string", "enum": ANGLES}, "view": {"type": "string", "enum": VIEWS},
        "foreground": {"type": "boolean"}, "soft_background": {"type": "boolean"}, "people": {"type": "integer"}}}}}}


def label_frames(paths: list[Path], tag: str, batch: int = 8) -> list[dict]:
    def one(part: list[Path]) -> list[dict]:
        try:
            r = vision("staging").chat([{"role": "system", "content": SYS},
                                        {"role": "user", "content": f"Angles: {'; '.join(ANGLES)}. Views: {'; '.join(VIEWS)}.\n{len(part)} frames follow, numbered 1-{len(part)}."}],
                                       schema=SCHEMA, images=[str(p) for p in part], max_tokens=2000, effort="low", media_resolution="medium",
                                       tag=f"staging:{tag}:{part[0].parent.name}")
        except Exception as e:                                             # noqa: BLE001
            log(f"staging batch failed: {str(e)[:120]}")
            return []
        return [{"path": str(part[x["n"] - 1]), **{k: x[k] for k in ("angle", "view", "foreground", "soft_background", "people")}}
                for x in r.get("frames", []) if isinstance(x.get("n"), int) and 1 <= x["n"] <= len(part)]

    parts = [paths[i : i + batch] for i in range(0, len(paths), batch)]
    with ThreadPoolExecutor(8) as ex:
        return [x for res in ex.map(one, parts) for x in res]


def mix(rows: list[dict]) -> dict:
    rows = [r for r in rows if int(r.get("people") or 0) >= 1]
    n = max(1, len(rows))
    return {"frames": len(rows),
            "angle": {a: round(sum(r["angle"] == a for r in rows) / n, 3) for a in ANGLES},
            "view": {v: round(sum(r["view"] == v for r in rows) / n, 3) for v in VIEWS},
            "foreground": round(sum(bool(r["foreground"]) for r in rows) / n, 3),
            "soft_background": round(sum(bool(r["soft_background"]) for r in rows) / n, 3)}


def tv(a: dict, b: dict) -> float:
    return round(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)), 3)


def train(n: int = 600, seed: int = 3) -> dict:
    have = read_json(TRAIN)
    if have:
        return have
    bank = [b for b in (read_json(P("processed") / "v3" / "style_bank.json") or [])
            if b.get("clean") and b.get("plain") and not b.get("letterbox") and b["people"] >= 1 and Path(b["path"]).exists()]
    assert all(int(b["episode"]) <= 21 for b in bank), "held-out episodes must never be labelled for staging"
    pick = random.Random(seed).sample(bank, min(n, len(bank)))
    size = {b["path"]: b["shot_size"] for b in pick}
    rows = [{**r, "shot_size": size.get(r["path"])} for r in label_frames([Path(b["path"]) for b in pick], "train")]
    out = {"rows": rows, "mix": mix(rows),
           "by_size": {s: mix([r for r in rows if r["shot_size"] == s]) for s in ("wide", "medium wide", "medium", "medium close-up", "close-up")}}
    write_json(TRAIN, out)
    log(f"staging, training frames: {out['mix']}")
    return out


def run(name: str) -> dict:
    rd = P("runs") / name
    shots = [s for s in read_json(rd / "shotlist.json")["shots"] if s["start"] in ("panel", "new") and (rd / "shots" / s["shot_id"] / "kf.png").exists()]
    rows = label_frames([rd / "shots" / s["shot_id"] / "kf.png" for s in shots], name)
    t = train()
    m = mix(rows)
    out = {"run": name, "mix": m, "train_mix": t["mix"], "tv_angle": tv(m["angle"], t["mix"]["angle"]), "tv_view": tv(m["view"], t["mix"]["view"]),
           "planned_angle": dict(collections.Counter(str(s.get("angle") or "") for s in shots)), "rows": rows}
    (rd / "eval").mkdir(exist_ok=True)
    write_json(rd / "eval" / "staging.json", out)
    log(f"staging, {name}: {m}")
    log(f"staging, show  : {t['mix']}")
    log(f"staging, {name}: distance to the show's mix - angle {out['tv_angle']}, view {out['tv_view']}")
    return out


DP_SYS = """You are the director of photography of the TV anime "Komi Can't Communicate". The storyboard of one scene is finished:
every shot has its size, the people in it, what they do and the lines spoken. Storyboards drawn from manga panels come out
flat: everybody faces the camera, side by side, at eye level. Your job is to decide FROM WHERE the camera looks in each shot,
so that the scene cuts like the show.

How the show stages its shots with people (measured on its episodes):
{mix}

For every shot give:
- angle: eye level | high angle | low angle | top-down
- view: frontal | three-quarter | profile | from behind | over the shoulder
  (over the shoulder = the camera looks past the back of the head and shoulder of one person, large, soft and cut by the frame
  edge, at the person or thing the shot is about)
- over: for "over the shoulder" only, the id of the person whose back is in the foreground. It must be one of the people of
  the shot, and never the one who speaks or reacts in it.
- foreground: "" or a few words naming something large and out of focus at a frame edge (the corner of a desk, a chair back,
  a blurred classmate's shoulder, a curtain, a shoe locker door). Not for over-the-shoulder shots, which already have one.
  Together with the over-the-shoulder shots, about one shot in three has a foreground - not more.
- soft_background: true when the background should be out of focus or blown out behind the people (typical for close shots).

Rules of the craft:
1. Read the scene as a whole and vary: two consecutive shots are never taken from the same position unless the second is a
   push-in on the same person. Over the scene, come close to the show's proportions above - do not put every shot at eye level
   and frontal, and do not overdo the unusual either.
2. Two people talking: alternate over-the-shoulder and its reverse, or singles in three-quarter; the listener's reaction in
   profile or three-quarter. Keep everybody on their side of the screen (the positions are fixed by the storyboard).
3. Frontal is right for a gag face, a stare into the lens, a reveal or an entrance. From behind is right when someone looks at
   something, walks away, or the scene hides a face. Low angle for someone imposing or triumphant, high angle for someone
   small, embarrassed or seen by a crowd. Inserts of objects on a desk or in hands: high angle or top-down.
4. A shot with one person cannot be "over the shoulder"; give it a foreground object instead when it helps.
5. Do not change who is in the shot, what they do or the shot size. Give every shot of the list, with its id exactly as given.
Return ONE JSON object: {{"shots": [{{"shot": "...", "angle": "...", "view": "...", "over": "", "foreground": "", "soft_background": false}}]}}"""
DP_SCHEMA = {"type": "object", "required": ["shots"], "properties": {"shots": {"type": "array", "minItems": 1, "items": {
    "type": "object", "required": ["shot", "angle", "view"], "properties": {
        "shot": {"type": "string"}, "angle": {"type": "string"}, "view": {"type": "string"}, "over": {"type": "string"},
        "foreground": {"type": "string"}, "soft_background": {"type": "boolean"}}}}}}


def _mix_text(m: dict) -> str:
    pc = lambda d: ", ".join(f"{k} {v:.0%}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))      # noqa: E731
    return (f"- camera height: {pc(m['angle'])}\n- how the main person is seen: {pc(m['view'])}\n"
            f"- something large and soft in the foreground: {m['foreground']:.0%} of shots\n- background out of focus: {m['soft_background']:.0%} of shots")


def direct(name: str) -> dict:
    """Camera pass over a finished shot list (after the editor pass, before the frames): shot["staging"] for every drawn shot."""
    rd = P("runs") / name
    sl = read_json(rd / "shotlist.json")
    shots = sl["shots"]
    drawn = [s for s in shots if s["start"] in ("panel", "new")]
    if sl.get("staged") and all(s.get("staging") for s in drawn):
        return read_json(rd / "staging_plan.json") or {}
    t = train()
    by_scene: dict[str, list[dict]] = collections.OrderedDict()
    for s in drawn:
        by_scene.setdefault(str(s["scene_id"]), []).append(s)
    # a long scene is staged in runs of 16 shots (a 70-shot scene in one call did not come back as valid JSON); shots staged before are kept
    scenes: dict[str, list[dict]] = collections.OrderedDict()
    for sid, part in by_scene.items():
        todo = [s for s in part if not s.get("staging")]
        for k in range(0, len(todo), 16):
            scenes[f"{sid}#{k // 16}"] = todo[k : k + 16]

    def one(item: tuple[str, list[dict]]) -> dict[str, dict]:
        sid, part = item[0].split("#")[0], item[1]
        view = [{"shot": s["shot_id"], "size": s.get("framing"), "shows": s.get("image_sentence"),
                 "people": [f"{v['id']} ({v.get('position', 'center')}): {str(v.get('pose', ''))[:80]}" for v in s.get("visible", []) if v["id"] != "Classmates"],
                 "crowd": any(v["id"] == "Classmates" for v in s.get("visible", [])),
                 "lines": [f"{d['speaker']}: {str(d['text'])[:70]}" for d in s.get("dialogue", [])][:3]} for s in part]
        user = (f"SCENE {sid}: {part[0].get('slug', '')}" + (" (a run of consecutive shots from a longer scene)" if len(by_scene[sid]) > len(part) else "")
                + "\n" + __import__("json").dumps(view, ensure_ascii=False))
        try:
            r = writer("staging").chat([{"role": "system", "content": DP_SYS.format(mix=_mix_text(t["mix"]))}, {"role": "user", "content": user}],
                                       schema=DP_SCHEMA, max_tokens=16000, thinking=True, effort="medium", tag=f"staging:{name}:{item[0]}")
        except Exception as e:                                             # noqa: BLE001
            log(f"staging {sid}: {str(e)[:160]}")
            return {}
        by = {s["shot_id"]: s for s in part}
        out = {}
        for x in r.get("shots", []):
            s = by.get(str(x.get("shot")))
            if s is None:
                continue
            people = [v["id"] for v in s.get("visible", []) if v["id"] != "Classmates"]
            angle = str(x.get("angle") or "").lower().strip()
            v = str(x.get("view") or "").lower().strip()
            over = str(x.get("over") or "").strip()
            if v == "over the shoulder" and (over not in people or len(people) < 2):
                v, over = "three-quarter", ""
            out[s["shot_id"]] = {"angle": angle if angle in ANGLES else "eye level", "view": v if v in VIEWS else "three-quarter",
                                 "over": over if v == "over the shoulder" else "",
                                 "foreground": "" if v == "over the shoulder" else " ".join(str(x.get("foreground") or "").split())[:90],
                                 "soft_background": bool(x.get("soft_background"))}
        return out

    with ThreadPoolExecutor(6) as ex:
        new = {k: v for part in ex.map(one, scenes.items()) for k, v in part.items()}
    for s in shots:
        if s["shot_id"] in new:
            s["staging"] = new[s["shot_id"]]
    plan = {s["shot_id"]: s["staging"] for s in drawn if s.get("staging")}
    write_json(rd / "staging_plan.json", plan)
    write_json(rd / "shotlist.json", {**sl, "shots": shots, "staged": True})
    rows = [{"people": 1, **p, "foreground": bool(p["foreground"]) or p["view"] == "over the shoulder"} for p in plan.values()]
    m = mix(rows)
    log(f"staging {name}: {len(plan)} of {len(drawn)} drawn shots staged ({len(new)} now, in {len(scenes)} calls); planned mix {m}; "
        f"distance to the show's: angle {tv(m['angle'], t['mix']['angle'])}, view {tv(m['view'], t['mix']['view'])}")
    return plan


def camera_line(shot: dict) -> str:
    """The camera position in words, for the painter and for the frame check ('' when the shot was not staged)."""
    st = shot.get("staging") or {}
    if not st:
        return ""
    people = [v["id"] for v in shot.get("visible", []) if v["id"] != "Classmates"]
    main = next((p for p in people if p != st.get("over")), people[0] if people else "the subject")
    view = {"frontal": f"{main} faces the camera",
            "three-quarter": f"{main} is seen in three-quarter view, turned partly away from the camera",
            "profile": f"{main} is seen in profile, from the side",
            "from behind": f"{main} is seen from behind (back of the head and shoulders toward the camera)",
            "over the shoulder": f"over the shoulder of {st.get('over')}: the back of {st.get('over')}'s head and one shoulder fill a side of the frame, large, "
                                 f"soft and cut by the frame edge, and the camera looks past them at {main}"}.get(st.get("view"), "")
    angle = {"eye level": "camera at eye level", "high angle": "HIGH ANGLE - the camera is above head height and looks down",
             "low angle": "LOW ANGLE - the camera is below chest height and looks up", "top-down": "TOP-DOWN - the camera looks straight down"}.get(st.get("angle"), "")
    bits = [b for b in (angle, view) if b]
    if st.get("foreground"):
        bits.append(f"in the foreground, large and out of focus at a frame edge: {st['foreground']}")
    if st.get("soft_background"):
        bits.append("the background behind the people is out of focus")
    return "; ".join(bits) + "."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "run", "direct"])
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    {"train": train, "run": lambda: run(a.run), "direct": lambda: direct(a.run)}[a.cmd]()
