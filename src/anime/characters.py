"""Character bank (brief §5.2): anime face/head detection -> CCIP embeddings -> named reference sets.

Auto-naming: votes come from the local VLM's roster-based tags on single-character shots (training episodes only);
CCIP then removes outliers (wrong tags) and picks medoid references. Low-agreement characters are flagged for the
review page (tools/review). `identify()` is the identity evidence used by QA (§8).

  python -m src.anime.characters build      # needs data/processed/train/wan/captions.jsonl (or shots captions)
Outputs: data/processed/characters/bank.json  {name: {refs:[paths], n:int, purity:float, feats:"<name>.npy"}}
         data/processed/characters/clusters.json (review-page schema)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json

OUT = P("processed") / "characters"
_feat_cache: dict[str, np.ndarray] = {}


def head_crops(img: Image.Image, min_frac: float = 0.06) -> list[tuple[tuple[int, int, int, int], float]]:
    """Head/upper-body crops around detected anime faces: [(box, face_height_fraction)]."""
    from imgutils.detect import detect_faces

    W, H = img.size
    out = []
    for (x0, y0, x1, y1), _, score in detect_faces(img):
        fw, fh = x1 - x0, y1 - y0
        if fh / H < min_frac or score < 0.4:
            continue
        cx = (x0 + x1) / 2
        box = (int(max(0, cx - 1.6 * fw)), int(max(0, y0 - 0.9 * fh)), int(min(W, cx + 1.6 * fw)), int(min(H, y1 + 2.2 * fh)))
        out.append((box, fh / H))
    out.sort(key=lambda b: -b[1])
    return out


def ccip(img: Image.Image) -> np.ndarray:
    from imgutils.metrics import ccip_extract_feature

    return np.asarray(ccip_extract_feature(img), dtype=np.float32)


def ccip_diff(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """CCIP difference between one feature and a stack (lower = same character)."""
    from imgutils.metrics import ccip_batch_differences

    d = ccip_batch_differences([a, *list(B)])
    return np.asarray(d)[0, 1:]


def roster_names() -> list[str]:
    return [c["name"] for c in CFG.characters.roster if c["role"] not in ("voice_only", "crowd")]


def build(max_per_char: int = 60, keep: int = 16) -> dict:
    from imgutils.metrics import ccip_batch_differences, ccip_default_threshold

    names = set(roster_names())
    caps = read_jsonl(P("processed") / "train" / "wan" / "captions.jsonl")
    for c in caps:
        c["_frame"] = str(P("processed") / "train" / "wan" / "frames" / c["id"] / "a.jpg")
    # captioned shots of TRAIN episodes add many more single-character examples (S2 cast in particular)
    for ep in CFG.split.train_episodes:
        for r in read_jsonl(P("processed") / "anime" / "shot_captions" / f"ep{ep:02d}.jsonl"):
            caps.append({**r, "ep": ep, "id": f"ep{ep:02d}_shot{r['shot']:04d}",
                         "_frame": str(P("interim") / "anime" / f"ep{ep:02d}" / "keyframes" / f"s{r['shot']:04d}_m.jpg")})
    votes: dict[str, list[dict]] = defaultdict(list)
    for c in caps:
        if c.get("ep") not in CFG.split.train_episodes:
            continue
        ch = [x for x in c.get("characters", []) if x in names]
        if len(ch) != 1 or len(c.get("characters", [])) != 1 or c.get("is_title_card") or c.get("is_chibi"):
            continue
        if c.get("shot_size") not in ("close-up", "medium close-up", "medium"):
            continue
        votes[ch[0]].append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    thr = float(ccip_default_threshold())
    bank, clusters = {}, []
    for ci, (name, items) in enumerate(sorted(votes.items(), key=lambda kv: -len(kv[1]))):
        cands = []
        for c in items[:max_per_char]:
            f = Path(c["_frame"])
            if not f.exists():
                continue
            img = Image.open(f).convert("RGB")
            crops = head_crops(img)
            if len(crops) != 1:      # exactly one visible face keeps the vote clean
                continue
            box, frac = crops[0]
            try:
                cands.append({"frame": str(f), "box": box, "frac": frac, "feat": ccip(img.crop(box)), "id": c["id"]})
            except Exception:
                continue
        if len(cands) < 3:
            continue
        F = np.stack([c["feat"] for c in cands])
        D = np.asarray(ccip_batch_differences(list(F)))
        med = int(np.argmin(np.median(D, axis=1)))
        d_med = D[med]
        inl = [i for i in np.argsort(d_med) if d_med[i] <= thr * 1.15]
        purity = len(inl) / len(cands)
        sel = inl[:keep] if len(inl) >= 3 else list(np.argsort(d_med)[: min(keep, len(cands))])
        np.save(OUT / f"{name.replace(' ', '_')}.npy", F[sel])
        # reference frames: close to the medoid AND a reasonably large face (good conditioning images)
        ref_idx = sorted(sel[:8], key=lambda i: (-(0.12 <= cands[i]["frac"] <= 0.5), d_med[i]))[:3]
        refs = []
        for k, i in enumerate(ref_idx):
            dst = OUT / "refs" / name.replace(" ", "_") / f"ref_{k:02d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.open(cands[i]["frame"]).convert("RGB").save(dst, quality=95)
            refs.append(str(dst))
        bank[name] = {"refs": refs, "n": len(sel), "n_votes": len(cands), "purity": round(purity, 3),
                      "feats": str(OUT / f"{name.replace(' ', '_')}.npy")}
        clusters.append({"id": ci, "name": name, "confidence": round(purity, 3), "size": len(cands),
                         "exemplars": [{"image": cands[i]["frame"]} for i in sel[:12]]})
        log(f"bank {name}: {len(sel)} feats from {len(cands)} votes, purity {purity:.2f}")
    write_json(OUT / "bank.json", bank)
    write_json(OUT / "clusters.json", {"clusters": clusters, "threshold": thr})
    return bank


def _bank_feats() -> dict[str, np.ndarray]:
    if not _feat_cache:
        for name, b in (read_json(OUT / "bank.json") or {}).items():
            if Path(b["feats"]).exists():
                _feat_cache[name] = np.load(b["feats"])
    return _feat_cache


def identify(image: str | Path | Image.Image, expected: list[str] | None = None) -> dict:
    """Identity evidence for one frame: every detected head is matched against the bank.

    Returns {"faces": [{"box", "best": name, "diff": d, "same": bool, "expected_diffs": {name: d}}], "threshold": t}
    """
    from imgutils.metrics import ccip_default_threshold

    bank = _bank_feats()
    thr = float(ccip_default_threshold())
    img = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
    faces = []
    for box, frac in head_crops(img, min_frac=0.05)[:5]:
        try:
            f = ccip(img.crop(box))
        except Exception:
            continue
        scores = {n: float(np.min(ccip_diff(f, B))) for n, B in bank.items()} if bank else {}
        best = min(scores, key=scores.get) if scores else None
        faces.append({"box": list(box), "face_frac": round(frac, 3), "best": best,
                      "diff": round(scores[best], 4) if best else None, "same": bool(best and scores[best] <= thr),
                      "expected_diffs": {n: round(scores[n], 4) for n in (expected or []) if n in scores}})
    return {"faces": faces, "threshold": round(thr, 4), "bank_size": len(bank)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    a = ap.parse_args()
    build()
