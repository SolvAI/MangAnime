"""All-ages screen of every finished clip of a run (D-054).

The frame check looks at first frames and the clip check at deformation; neither stopped the image-to-video model from
moving two characters into a cheek-to-cheek pose in a chained clip of ep22_v3. This pass looks at four frames of every clip
(strips, several clips per request, so the whole episode costs a few cents) and asks one question only.

  python -m src.v3.screen --run ep22_v3 [--apply]
--apply replaces each flagged clip by a held frame that passed the frame check (the first frame of the last drawn, unflagged
shot before it), keeps the original as clip_unsuitable.mp4 and marks clips_state.json.
Re-assemble afterwards: python -m src.v3.pipeline --run <run> --stages assemble,report
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.eval.ab_judge import strip
from src.llm.client import encode_image, vision

SCHEMA = {"type": "object", "required": ["flagged"], "properties": {"flagged": {"type": "array", "items": {
    "type": "object", "required": ["clip", "reason"], "properties": {"clip": {"type": "integer"}, "reason": {"type": "string"}}}}}}
SYS = """You screen clips of an all-ages TV anime whose characters are school students. Each image is ONE clip shown as four
frames side by side in time order. Flag a clip when ANY of its frames shows:
- a kiss, or a face or mouth pressed against another person's face or body;
- a romantic or sexualised embrace, or one person lying on another;
- a hand on someone's chest, buttocks or thigh;
- underwear, nudity, a view up a skirt, or framing that dwells on chest, hips or thighs;
- a sexualised pose or expression.
Do NOT flag: people standing or sitting close, a hand on a shoulder or arm, a pat on the head, children hugging an adult's
waist in play, comedy faces, blushing, crying, or ordinary school and home scenes.
Return ONE JSON object: {"flagged": [{"clip": <number as labelled>, "reason": "what is shown, in a few words"}]} - an empty
list when nothing is to flag."""


def screen(run: str, batch: int = 8) -> dict:
    rd = P("runs") / run
    shots = [s for s in read_json(rd / "shotlist.json")["shots"] if s["start"] != "title" and (rd / "shots" / s["shot_id"] / "clip.mp4").exists()]
    strips = []
    for s in shots:
        f = strip(rd / "shots" / s["shot_id"] / "clip.mp4", rd / "eval" / "screen_strips" / f"{s['shot_id']}.jpg")
        if f:
            strips.append((s["shot_id"], f))

    def one(part):
        content = []
        for k, (sid, f) in enumerate(part):
            content.append({"type": "text", "text": f"Clip {k + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": encode_image(f, 1536, 85)}})
        content.append({"type": "text", "text": f"{len(part)} clips. Return the JSON object."})
        try:
            r = vision("screen").chat([{"role": "system", "content": SYS}, {"role": "user", "content": content}], schema=SCHEMA,
                                      max_tokens=3000, effort="low", media_resolution="medium", tag=f"screen:{run}:{part[0][0]}")
        except Exception as e:                                             # noqa: BLE001
            log(f"screen {part[0][0]}: {str(e)[:140]}")
            return [{"shot_id": sid, "reason": "not screened: " + str(e)[:80], "error": True} for sid, _ in part]
        return [{"shot_id": part[x["clip"] - 1][0], "reason": str(x.get("reason", ""))[:160]} for x in r.get("flagged", [])
                if isinstance(x.get("clip"), int) and 1 <= x["clip"] <= len(part)]

    parts = [strips[i : i + batch] for i in range(0, len(strips), batch)]
    with ThreadPoolExecutor(6) as ex:
        flagged = [x for res in ex.map(one, parts) for x in res]
    out = {"run": run, "clips_screened": len(strips), "flagged": [x for x in flagged if not x.get("error")], "not_screened": [x for x in flagged if x.get("error")]}
    write_json(rd / "eval" / "screen.json", out)
    log(f"screen {run}: {len(out['flagged'])} of {len(strips)} clips flagged" + (f", {len(out['not_screened'])} not screened" if out["not_screened"] else ""))
    for x in out["flagged"]:
        log(f"   {x['shot_id']}: {x['reason']}")
    return out


def apply(run: str) -> list[str]:
    from src.generate.video import still_clip

    rd = P("runs") / run
    shots = read_json(rd / "shotlist.json")["shots"]
    flagged = {x["shot_id"] for x in (read_json(rd / "eval" / "screen.json") or {}).get("flagged", [])}
    state = read_json(rd / "clips_state.json") or {}
    done = []
    for i, s in enumerate(shots):
        if s["shot_id"] not in flagged:
            continue
        sd = rd / "shots" / s["shot_id"]
        # the picture that is held instead: the first frame of the last drawn shot before this one that was not flagged itself
        # (the shot's own first frame is not used: the problem is often already in it)
        src = next((p for p in reversed(shots[:i]) if p["start"] in ("panel", "new") and p["shot_id"] not in flagged), s)
        kf = rd / "shots" / src["shot_id"] / "kf.png"
        if not (sd / "clip_unsuitable.mp4").exists():
            shutil.copyfile(sd / "clip.mp4", sd / "clip_unsuitable.mp4")
        still_clip(kf, sd / "clip.mp4", s["duration"], s.get("camera", "static"))
        state.setdefault(s["shot_id"], {})
        state[s["shot_id"]].update({"final": "still", "why": "all-ages screen", "still_from": src["shot_id"]})
        done.append(s["shot_id"])
    write_json(rd / "clips_state.json", state)
    log(f"screen {run}: {len(done)} clips replaced by a held frame: {done}")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    screen(a.run)
    if a.apply:
        apply(a.run)
