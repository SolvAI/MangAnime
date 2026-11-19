"""Background pass (D-058): the image model repaints the BACKGROUND of a finished first frame the way the show paints it -
simple, soft, a little out of focus - and leaves the people as they are.

  python -m src.v3.soften --run ep22_v3d [--only id,id] [--apply]

Why: with style references the frames have the show's palette, but they still carry half again as many edges as real frames
(edge density 0.29 against 0.18-0.20, Laplacian variance 1,150 against 520-570; `src/eval/frame_style.py`), and "every part
equally sharp" is the first thing three blind raters named. Each frame is sent back with one instruction; the result is kept
as kf_soft.png next to kf.png and only replaces it with --apply after the frame check has accepted it.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image

from src.common.config import CFG, P
from src.common.io import append_jsonl, log, read_json, write_json
from src.llm.client import api_spend

INSTRUCTION = (
    "Edit this anime frame. Keep every person exactly as drawn: same faces, hair, clothes, poses, positions, sizes and outlines; keep the "
    "framing, the place, the light and the time of day; keep any object a person holds or the shot is about. Change ONLY the background: "
    "repaint it the way a TV anime background is painted - simple flat shapes, far fewer drawn lines and textures, no crisp outlines on "
    "furniture, walls, trees or floor, soft edges, gently out of focus behind the people (more so in close shots), a light bloom around "
    "windows and lamps. The people must stay sharp and stand out from the softer background. Do not add or remove people or objects, do not "
    "add any text. Output the single edited frame, 16:9."
)


FOCUS = {  # how far the background falls out of focus, by framing: the show keeps wide shots readable and throws close shots' backgrounds away
    "wide": "Wide shot: keep the place readable, but paint it simply and softly, without crisp outlines.",
    "medium wide": "Medium-wide shot: the background is soft and simple, slightly out of focus.",
    "medium": "Medium shot: the background is clearly out of focus - soft shapes and colours, hardly any lines.",
    "medium close-up": "Close shot: the background is strongly out of focus, like a long lens - only blurred shapes and colours remain.",
    "close-up": "Close-up: the background is strongly out of focus, like a long lens - only blurred shapes and colours remain.",
    "extreme close-up": "Close-up: the background is a blur of colour.",
    "insert": "Insert shot: the object stays sharp; whatever lies behind it is strongly out of focus.",
}


def soften_frame(src: Path, out: Path, tag: str, framing: str = "medium") -> Path | None:
    cfg = CFG.llm.gemini_image
    if out.exists():
        return out
    cap = min(float(CFG.budget.api_usd), float(CFG.budget.api_usd_effective))
    if api_spend() >= cap * float(CFG.budget.api_soft_fraction):
        raise RuntimeError(f"API budget: ${api_spend():.2f} spent")
    im = Image.open(src).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    body = {"contents": [{"role": "user", "parts": [{"text": INSTRUCTION + " " + FOCUS.get(framing, FOCUS["medium"])}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}]}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "16:9", "imageSize": str(cfg.image_size)}}}
    url = f"{CFG.llm.gemini.base_url}/models/{cfg.model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    for k in range(5):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=300)
        except httpx.HTTPError:
            time.sleep(min(60, 5 * 2**k))
            continue
        if r.status_code in (429, 500, 502, 503, 504, 529):
            time.sleep(min(90, 5 * 2**k))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        u = d.get("usageMetadata") or {}
        cost = (u.get("promptTokenCount", 0) * float(cfg.price_in) + u.get("candidatesTokenCount", 0) * float(cfg.price_out)) / 1e6
        append_jsonl(P("runs") / "api_cost.jsonl", {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "endpoint": "gemini", "model": cfg.model, "role": "frame_soften",
                                                    "tag": tag, "prompt_tokens": u.get("promptTokenCount", 0), "completion_tokens": u.get("candidatesTokenCount", 0),
                                                    "cost_usd": round(cost, 6)})
        cands = d.get("candidates") or []
        imgs = [p["inlineData"]["data"] for p in ((cands[0].get("content") or {}).get("parts") or []) if p.get("inlineData") and not p.get("thought")] if cands else []
        if not imgs:
            if k >= 1:
                return None
            continue
        res = Image.open(io.BytesIO(base64.b64decode(imgs[-1]))).convert("RGB")
        w, h = res.size
        t = im.width / im.height
        if w / h > t:
            x = int((w - h * t) / 2)
            res = res.crop((x, 0, w - x, h))
        elif w / h < t:
            y = int((h - w / t) / 2)
            res = res.crop((0, y, w, h - y))
        res.resize(im.size, Image.LANCZOS).save(out)
        return out
    return None


def run(run: str, only: set[str] | None = None, apply: bool = False, threads: int = 6) -> dict:
    from src.v3.frames import check_frame

    rd = P("runs") / run
    shots = [s for s in read_json(rd / "shotlist.json")["shots"] if s["start"] in ("panel", "new") and (only is None or s["shot_id"] in only)]
    state = read_json(rd / "soften_state.json") or {}

    def one(s: dict) -> tuple[str, dict]:
        sd = rd / "shots" / s["shot_id"]
        if state.get(s["shot_id"], {}).get("done"):
            return s["shot_id"], state[s["shot_id"]]
        if not (sd / "kf_sharp.png").exists():
            shutil.copyfile(sd / "kf.png", sd / "kf_sharp.png")
        try:
            f = soften_frame(sd / "kf_sharp.png", sd / "kf_soft.png", f"soften:{run}:{s['shot_id']}", str(s.get("framing") or s.get("shot_size") or "medium"))
        except Exception as e:                                             # noqa: BLE001
            return s["shot_id"], {"done": False, "error": str(e)[:120]}
        if f is None:
            return s["shot_id"], {"done": True, "kept": False, "why": "no image"}
        if not apply:
            return s["shot_id"], {"done": False, "drawn": True}
        # the softened frame must pass the same check as any first frame (people present, on-model, all-ages) before it replaces the sharp one
        shutil.copyfile(sd / "kf_soft.png", sd / "kf_a90.png")
        write_json(sd / "kf_a90.json", {"backend": "gemini", "soften": True, "from": "kf_sharp.png"})
        c = check_frame(rd, s, 90)
        ok = c.get("verdict") == "ok" and not c.get("unsuitable")
        if ok:
            shutil.copyfile(sd / "kf_soft.png", sd / "kf.png")
        return s["shot_id"], {"done": True, "kept": ok, "score": c.get("score"), "problems": (c.get("problems") or [])[:3]}

    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for sid, r in ex.map(one, shots):
            state[sid] = r
    write_json(rd / "soften_state.json", state)
    log(f"soften {run}: {len(shots)} frames in {(time.time() - t0) / 60:.1f} min; kept {sum(1 for r in state.values() if r.get('kept'))}, "
        f"not kept {sum(1 for r in state.values() if r.get('done') and not r.get('kept'))}; API spend ${api_spend():.2f}")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    run(a.run, set(a.only.split(",")) if a.only else None, a.apply)
