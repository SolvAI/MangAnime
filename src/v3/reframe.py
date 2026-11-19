"""Reframing pass (D-060): crop a finished first frame to the shot size the plan asked for.

  python -m src.v3.reframe --run ep22_v3e            # writes shots/<id>/kf_reframed.png, prints the shot-size mix before / after
  python -m src.v3.reframe --run ep22_v3e --apply    # ... and makes it the first frame (the uncropped picture stays as kf_wide.png)

Measured with one labeller (Gemini Flash) on both sides: the show's training frames are 19 % close-ups, 20 % medium close-ups
and 9 % inserts; the plan asks for about that; the painter delivers 3 %, 19 % and 2 % - every close framing comes out one
or two sizes wide, with a room full of desks behind it. The picture itself is right (people, clothes, place), so this pass
does not redraw anything: the vision model locates the heads of the people the shot names (or the object of an insert), and
the frame is cropped around them until the head has the size that framing means, then scaled back to frame size.
"""
from __future__ import annotations

import argparse
import collections
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.llm.client import vision

# Head height as a share of the frame height that a framing means IN THIS SHOW: medians over ~35 training frames per size, boxes
# by the same locator (data/processed/v3/head_size_by_framing.json): close-up 0.92, medium close-up 0.61, medium 0.44,
# medium wide 0.23, wide 0.12. The show frames a size tighter than the painter does.
HEAD = {"extreme close-up": 1.0, "close-up": 0.90, "medium close-up": 0.60, "medium": 0.43, "medium wide": 0.23}
TOP = {"extreme close-up": 0.5, "close-up": 0.5, "medium close-up": 0.45, "medium": 0.38, "medium wide": 0.34}     # where the heads sit in the crop
AIR = {"extreme close-up": 0.0, "close-up": 0.02, "medium close-up": 0.10}                                          # air kept above + below the heads
INSERT_FILL = 0.6          # an insert: its object spans this share of the frame's shorter relative side
MAX_ZOOM = 2.2             # 1024 px wide frames: a 2.2x crop still has 465 px for Wan's 832
SYS = ("You locate things in one anime frame. Boxes are [ymin, xmin, ymax, xmax] in thousandths of the picture height and width "
       "(0-1000), tight around the thing. Answer with one JSON object.")
SCHEMA = {"type": "object", "required": ["boxes"], "properties": {"boxes": {"type": "array", "items": {
    "type": "object", "required": ["id", "box"], "properties": {"id": {"type": "string"}, "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}}}}


def locate(frame: Path, shot: dict) -> list[dict]:
    vis = [v for v in shot.get("visible", []) if v.get("id") != "Classmates"]
    if vis:
        ask = ("Give the box of the HEAD (face and hair, not the body) of each of these people: "
               + "; ".join(f"{v['id']} ({v.get('position', 'center')}; {v.get('pose', '')})" for v in vis)
               + '. {"boxes": [{"id": "<name as given>", "box": [ymin, xmin, ymax, xmax]}]} - leave out a person you cannot find. '
               + f'The frame shows: "{shot.get("image_sentence", "")}" If that sentence is about something a person holds, hands over, writes on or '
               'reaches for, or about a gesture of the hands, add one more box around that object or those hands with the id "keep".')
    else:
        ask = (f"The frame is an insert: {shot.get('image_sentence', '')} Give the box of the object the shot is about. "
               '{"boxes": [{"id": "object", "box": [ymin, xmin, ymax, xmax]}]}')
    try:
        r = vision("reframe").chat([{"role": "system", "content": SYS}, {"role": "user", "content": ask}], schema=SCHEMA, images=[str(frame)],
                                   max_tokens=1200, effort="low", media_resolution="medium", tag=f"reframe:{shot['shot_id']}")
    except Exception as e:                                             # noqa: BLE001
        log(f"reframe {shot['shot_id']}: {str(e)[:100]}")
        return []
    out = []
    for b in r.get("boxes", []):
        try:
            y0, x0, y1, x1 = [min(1000.0, max(0.0, float(v))) / 1000 for v in b["box"]]
        except Exception:
            continue
        if y1 - y0 > 0.02 and x1 - x0 > 0.02:
            out.append({"id": str(b.get("id")), "box": [x0, y0, x1, y1]})
    return out


def window(shot: dict, boxes: list[dict]) -> tuple[float, float, float, float] | None:
    """Crop window (x0, y0, x1, y1) as shares of the frame, 16:9 like the frame, or None when the frame is already as tight as asked."""
    if not boxes:
        return None
    framing = shot.get("framing") or "medium"
    x0, y0 = min(b["box"][0] for b in boxes), min(b["box"][1] for b in boxes)
    x1, y1 = max(b["box"][2] for b in boxes), max(b["box"][3] for b in boxes)
    heads = [b for b in boxes if b["id"] != "keep"] or boxes       # "keep": the held object or the hands the shot is about - stays in frame, does not set the size
    if framing == "insert" or not shot.get("visible"):
        zoom = INSERT_FILL / max(x1 - x0, y1 - y0)
        top = 0.5
    else:
        head = max(b["box"][3] - b["box"][1] for b in heads)                      # the largest head sets the size
        zoom = HEAD.get(framing, 0.43) / head
        top = TOP.get(framing, 0.4)
    # everything asked for must stay inside, with air around it
    air = AIR.get(framing, 0.12)
    zoom = min(zoom, MAX_ZOOM, 1 / (x1 - x0 + 0.08), 1 / max(0.05, y1 - y0 + air))
    if zoom < 1.12:
        return None
    w = h = 1 / zoom
    cx = (x0 + x1) / 2
    hy0, hy1 = min(b["box"][1] for b in heads), max(b["box"][3] for b in heads)
    cy = (hy0 + hy1) / 2
    wx0 = min(max(cx - w / 2, 0.0), 1 - w)
    wy0 = min(max(cy - top * h, 0.0), 1 - h)
    wy0 = min(wy0, max(0.0, y0 - 0.01))                 # never cut the top of a head that fits
    if y1 - y0 <= h:
        wy0 = max(wy0, y1 - h)                          # ... nor a chin
    return (wx0, wy0, wx0 + w, wy0 + h)


def run(run: str, apply: bool = False, only: set[str] | None = None, threads: int = 8, lock: set[str] | None = None) -> dict:
    """lock: shots whose clip is already rendered or rendering - their crop is computed and listed in reframe_deferred.json
    but kf.png is left alone until `deferred` is called (the clip is then set aside and rendered again from the cropped frame)."""
    rd = P("runs") / run
    shots = [s for s in read_json(rd / "shotlist.json")["shots"] if s["start"] in ("panel", "new") and (only is None or s["shot_id"] in only)]
    todo = [s for s in shots if s.get("framing") in ("extreme close-up", "close-up", "medium close-up", "medium", "medium wide", "insert")
            and (s.get("framing") == "insert" or any(v.get("id") != "Classmates" for v in s.get("visible", [])))
            and (s.get("staging") or {}).get("view") != "over the shoulder"]        # composed around a foreground figure by the painter: left as drawn
    state = read_json(rd / "reframe_state.json") or {}

    def one(s: dict) -> tuple[str, dict]:
        sd = rd / "shots" / s["shot_id"]
        src = sd / "kf_wide.png" if (sd / "kf_wide.png").exists() else sd / "kf.png"
        if not src.exists():
            return s["shot_id"], {"error": "no frame"}
        boxes = (state.get(s["shot_id"]) or {}).get("boxes") or locate(src, s)       # boxes are kept; the window follows the current targets
        win = window(s, boxes)
        if win is None:
            (sd / "kf_reframed.png").unlink(missing_ok=True)
            return s["shot_id"], {"boxes": boxes, "window": None}
        im = Image.open(src).convert("RGB")
        W, H = im.size
        im.crop((round(win[0] * W), round(win[1] * H), round(win[2] * W), round(win[3] * H))).resize((W, H), Image.LANCZOS).save(sd / "kf_reframed.png")
        return s["shot_id"], {"boxes": boxes, "window": [round(v, 4) for v in win], "zoom": round(1 / (win[2] - win[0]), 2)}

    with ThreadPoolExecutor(threads) as ex:
        for sid, r in ex.map(one, todo):
            state[sid] = r
    write_json(rd / "reframe_state.json", state)
    done = [s for s in todo if (state.get(s["shot_id"]) or {}).get("window")]
    zs = [state[s["shot_id"]]["zoom"] for s in done]
    log(f"reframe {run}: {len(todo)} close framings and inserts looked at, {len(done)} cropped (zoom median {sorted(zs)[len(zs) // 2] if zs else 0}, "
        f"by framing {dict(collections.Counter(s['framing'] for s in done))}), {sum(1 for s in todo if not (state.get(s['shot_id']) or {}).get('boxes'))} without a box")
    if apply:
        later = [s["shot_id"] for s in done if lock and s["shot_id"] in lock]
        for s in done:
            if s["shot_id"] in later:
                continue
            _apply(rd / "shots" / s["shot_id"])
        write_json(rd / "reframe_deferred.json", later)
        log(f"reframe {run}: applied to {len(done) - len(later)} first frames (the uncropped pictures are kf_wide.png); {len(later)} wait for their clip to be redone")
    return state


def _apply(sd: Path) -> None:
    if not (sd / "kf_wide.png").exists():
        shutil.copyfile(sd / "kf.png", sd / "kf_wide.png")
    shutil.copyfile(sd / "kf_reframed.png", sd / "kf.png")


def deferred(run: str) -> list[str]:
    """Apply the crops that waited for a finished clip: the clip rendered from the wide frame is set aside (clip_wide.mp4) so
    that the video stage renders the shot again from the cropped frame."""
    rd = P("runs") / run
    ids = read_json(rd / "reframe_deferred.json") or []
    cs = read_json(rd / "clips_state.json") or {}
    shots = read_json(rd / "shotlist.json")["shots"]
    order = [s["shot_id"] for s in shots]
    start = {s["shot_id"]: s["start"] for s in shots}
    chained = []
    for sid in ids:                                    # a chained shot starts on the last frame of the clip before it: redo it too
        i = order.index(sid) + 1
        while i < len(order) and start[order[i]] == "chain":
            chained.append(order[i])
            i += 1
    for sid in ids + chained:
        sd = rd / "shots" / sid
        if sid in ids:
            if not (sd / "kf_reframed.png").exists():
                continue
            _apply(sd)
        else:
            (sd / "kf.png").unlink(missing_ok=True)    # taken again from the new clip before it
        if (sd / "clip.mp4").exists():
            (sd / "clip.mp4").rename(sd / "clip_wide.mp4")
        for f in list(sd.glob("raw_a*.mp4")) + list(sd.glob("clip_a*.mp4")) + list(sd.glob("chain_a*.png")):
            f.unlink()
        cs.pop(sid, None)
    write_json(rd / "clips_state.json", cs)
    write_json(rd / "reframe_deferred.json", [])
    log(f"reframe {run}: {len(ids)} deferred crops applied, their clips (and {len(chained)} chained after them) set aside for a new render")
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--lock", default="", help="file with a JSON list of shot ids whose clip exists or is rendering")
    ap.add_argument("--deferred", action="store_true", help="apply the crops that waited and set their clips aside")
    a = ap.parse_args()
    if a.deferred:
        deferred(a.run)
    else:
        run(a.run, a.apply, set(a.only.split(",")) if a.only else None, lock=set(read_json(a.lock) or []) if a.lock else None)
