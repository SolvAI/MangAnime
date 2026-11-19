"""A second, independent blind read of "generated vs the real episode" (the Gemini judge also drew the frames, D-053).

  python -m src.eval.blind_set build --run ep22_v3b --n 40        # pair images + a key kept apart
  python -m src.eval.blind_set animatic --run ep22_v3b --n 40     # adds the generated-clip vs animatic pairs (same blind form)
  python -m src.eval.blind_set score --run ep22_v3b               # after raters wrote their answers

build writes <run>/eval/blind/frames/pair_XX.jpg (two stills side by side, L / R) and <run>/eval/blind/clips/pair_XX.jpg
(two 4-frame strips of a shot, TOP / BOTTOM), order randomised per pair, no names, and the key in <run>/eval/blind_key.json
(outside the folder the raters are given). A rater answers, per pair, in <run>/eval/blind/answers_<rater>.json:
  {"frames": {"pair_01": {"generated": "L", "better": "R", "note": "..."}, ...}, "clips": {"pair_01": {"generated": "TOP", "better": "tie"}, ...}}
The animatic set (<run>/eval/blind/animatic/, strips TOP / BOTTOM: the Wan clip and the animatic of the same shot, which share
the first frame) is answered under "animatic": {"pair_01": {"better": "TOP"}}.
score reports how often the generated side was recognised (50 % = cannot be told apart) and which side was preferred.
The real episode is used for evaluation only.
"""
from __future__ import annotations

import argparse
import random
from math import comb
from pathlib import Path

from PIL import Image, ImageDraw

from src.common.config import P
from src.common.io import log, read_json, write_json


def _label(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14 + 9 * len(text), 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return im


def _fit(path: str, w: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def build(run: str, n: int, seed: int = 7) -> dict:
    rd = P("runs") / run
    ev = rd / "eval"
    rng = random.Random(seed)
    key = {"run": run, "seed": seed, "frames": {}, "clips": {}}
    kf = [p for p in (read_json(ev / "ab_manifest_keyframes.json") or {}).get("pairs", []) if Path(p["a"]).exists() and Path(p["b"]).exists()]
    cl = [p for p in (read_json(ev / "ab_manifest_real.json") or {}).get("pairs", [])
          if (ev / "strips" / f"{p['pair_id']}_wan.jpg").exists() and (ev / "strips" / f"{p['pair_id']}_real.jpg").exists() and p.get("meta", {}).get("final") == "wan"]
    for kind, pool in (("frames", kf), ("clips", cl)):
        out = ev / "blind" / kind
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("pair_*.jpg"):
            old.unlink()
        pick = sorted(rng.sample(pool, min(n, len(pool))), key=lambda p: p["pair_id"])
        rng.shuffle(pick)
        for i, p in enumerate(pick, 1):
            flip = rng.random() < 0.5
            if kind == "frames":
                g, r = _fit(p["a"], 768), _fit(p["b"], 768)
                h = max(g.height, r.height)
                first, second = (r, g) if flip else (g, r)
                sheet = Image.new("RGB", (768 * 2 + 12, h), (40, 40, 40))
                sheet.paste(_label(first, "L"), (0, 0))
                sheet.paste(_label(second, "R"), (768 + 12, 0))
                gen_side = "R" if flip else "L"
            else:
                g, r = _fit(str(ev / "strips" / f"{p['pair_id']}_wan.jpg"), 1536), _fit(str(ev / "strips" / f"{p['pair_id']}_real.jpg"), 1536)
                first, second = (r, g) if flip else (g, r)
                sheet = Image.new("RGB", (1536, first.height + second.height + 12), (40, 40, 40))
                sheet.paste(_label(first, "TOP"), (0, 0))
                sheet.paste(_label(second, "BOTTOM"), (0, first.height + 12))
                gen_side = "BOTTOM" if flip else "TOP"
            name = f"pair_{i:02d}"
            sheet.save(out / f"{name}.jpg", quality=90)
            key[kind][name] = {"shot_id": p["pair_id"], "generated": gen_side}
    write_json(ev / "blind_key.json", key)
    log(f"blind set {run}: {len(key['frames'])} frame pairs, {len(key['clips'])} clip pairs in {ev / 'blind'}; key {ev / 'blind_key.json'}")
    return key


def build_animatic(run: str, n: int, seed: int = 11) -> dict:
    """Wan clip vs the animatic of the same shot, as strips; added to the key without touching the other two sets."""
    ev = P("runs") / run / "eval"
    rng = random.Random(seed)
    key = read_json(ev / "blind_key.json") or {"run": run, "frames": {}, "clips": {}}
    pool = [p for p in (read_json(ev / "ab_manifest_animatic.json") or {}).get("pairs", [])
            if (ev / "strips" / f"{p['pair_id']}_wan.jpg").exists() and (ev / "strips" / f"{p['pair_id']}_animatic.jpg").exists()]
    out = ev / "blind" / "animatic"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("pair_*.jpg"):
        old.unlink()
    pick = rng.sample(pool, min(n, len(pool)))
    key["animatic"] = {}
    for i, p in enumerate(pick, 1):
        flip = rng.random() < 0.5
        g, r = _fit(str(ev / "strips" / f"{p['pair_id']}_wan.jpg"), 1536), _fit(str(ev / "strips" / f"{p['pair_id']}_animatic.jpg"), 1536)
        first, second = (r, g) if flip else (g, r)
        sheet = Image.new("RGB", (1536, first.height + second.height + 12), (40, 40, 40))
        sheet.paste(_label(first, "TOP"), (0, 0))
        sheet.paste(_label(second, "BOTTOM"), (0, first.height + 12))
        sheet.save(out / f"pair_{i:02d}.jpg", quality=90)
        key["animatic"][f"pair_{i:02d}"] = {"shot_id": p["pair_id"], "generated": "BOTTOM" if flip else "TOP", "final": p.get("meta", {}).get("final")}
    write_json(ev / "blind_key.json", key)
    log(f"blind set {run}: {len(key['animatic'])} clip-vs-animatic pairs in {out}")
    return key


def build_pairs(items: list[tuple[str, str, str]], out_dir: Path, key_path: Path, labels: tuple[str, str] = ("a", "b"), seed: int = 5,
                reference: list[str] | None = None) -> dict:
    """Generic blind L / R sheets: items = [(id, picture a, picture b)]; the key says on which side picture b is.
    reference: real frames put on one sheet (real_reference.jpg) for the rater to calibrate on."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("pair_*.jpg"):
        old.unlink()
    items = list(items)
    rng.shuffle(items)
    key = {"labels": list(labels), "pairs": {}}
    for i, (sid, a, b) in enumerate(items, 1):
        flip = rng.random() < 0.5
        ia, ib = _fit(a, 768), _fit(b, 768)
        first, second = (ib, ia) if flip else (ia, ib)
        sheet = Image.new("RGB", (768 * 2 + 12, max(ia.height, ib.height)), (40, 40, 40))
        sheet.paste(_label(first, "L"), (0, 0))
        sheet.paste(_label(second, "R"), (780, 0))
        sheet.save(out_dir / f"pair_{i:02d}.jpg", quality=90)
        key["pairs"][f"pair_{i:02d}"] = {"id": sid, labels[1]: "L" if flip else "R"}
    if reference:
        ref = Image.new("RGB", (4 * 480, ((len(reference) + 3) // 4) * 270))
        for i, f in enumerate(reference):
            ref.paste(Image.open(f).convert("RGB").resize((480, 270)), ((i % 4) * 480, (i // 4) * 270))
        ref.save(out_dir / "real_reference.jpg", quality=90)
    write_json(key_path, key)
    return key


def score_pairs(out_dir: Path, key_path: Path, field: str = "closer") -> dict:
    key = read_json(key_path)
    a, b = key["labels"]
    res = {}
    for f in sorted(out_dir.glob("answers_*.json")):
        ans = read_json(f) or {}
        c = {a: 0, b: 0, "tie": 0}
        for n, k in key["pairs"].items():
            v = str((ans.get(n) or {}).get(field, "")).strip().upper()
            c["tie" if v not in ("L", "R") else (b if v == k[b] else a)] += 1
        res[f.stem.replace("answers_", "")] = c
        log(f"blind pairs {out_dir.name} {f.stem}: {field}: {c}")
    return res


def _p_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def score(run: str) -> dict:
    ev = P("runs") / run / "eval"
    key = read_json(ev / "blind_key.json")
    res = {"run": run, "raters": {}}
    for f in sorted((ev / "blind").glob("answers_*.json")):
        ans = read_json(f) or {}
        rr = {}
        for kind in ("frames", "clips", "animatic"):
            k = key.get(kind) or {}
            if not k or not ans.get(kind):
                continue
            got = {n: a for n, a in (ans.get(kind) or {}).items() if n in k}
            spotted = sum(1 for n, a in got.items() if str(a.get("generated", "")).strip().upper() == k[n]["generated"])
            pref = {"generated": 0, "real": 0, "tie": 0}
            for n, a in got.items():
                b = str(a.get("better", "")).strip().upper()
                pref["tie" if b not in ("L", "R", "TOP", "BOTTOM") else ("generated" if b == k[n]["generated"] else "real")] += 1
            if kind == "animatic":  # no "which is generated" question here: both sides are; the keyed side is the Wan clip
                dec = pref["generated"] + pref["real"]
                rr[kind] = {"pairs": len(got), "preferred": {"wan": pref["generated"], "animatic": pref["real"], "tie": pref["tie"]},
                            "p_sign": _p_two_sided(pref["generated"], dec)}
                continue
            rr[kind] = {"pairs": len(got), "generated_recognised": spotted, "recognised_share": round(spotted / max(1, len(got)), 3),
                        "p_vs_chance": _p_two_sided(spotted, len(got)), "preferred": pref,
                        "missed": sorted(k[n]["shot_id"] for n, a in got.items() if str(a.get("generated", "")).strip().upper() != k[n]["generated"])}
        res["raters"][f.stem.replace("answers_", "")] = rr
        log(f"blind {run} {f.stem}: " + "; ".join(
            (f"{kind}: preferred {v['preferred']} (p={v['p_sign']})" if kind == "animatic" else
             f"{kind}: generated recognised {v['generated_recognised']}/{v['pairs']} (p={v['p_vs_chance']}), preferred {v['preferred']}") for kind, v in rr.items()))
    write_json(ev / "blind_score.json", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "animatic", "score"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.run, a.n, a.seed)
    elif a.cmd == "animatic":
        build_animatic(a.run, a.n)
    else:
        score(a.run)
