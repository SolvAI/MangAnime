"""BGM retrieval: pick cues from the separated-BGM library (TRAIN episodes only) for a mood / scene description.

    from src.generate.bgm import retrieve
    cues = retrieve("gentle_comedy", min_dur=12.0, k=5)
    cues = retrieve("tense, awkward silence", min_dur=8.0)
    # -> [{"id","ep","start","end","path","abs_path","dur","rms","emb_file","emb_idx","tags","dlg_ratio","sim","score",...}]

CLAP text -> audio retrieval (laion/clap-htsat-unfused, local copy in models/clap/). The library is built by
`python -m src.anime.audio_bgm`. Only numpy + transformers + torch are needed here (works in venvs/main and venvs/audio).
Text embeddings are cached on disk (cache/bgm_text/), so repeated moods never load the model again and the stage runs
with the network off. The text tower runs on CPU by default (set KOMI_BGM_DEVICE=cuda to change).

CLI:  python -m src.generate.bgm "tense, awkward silence" --min-dur 10 -k 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.anime.audio_common import ACFG
from src.common.config import CFG

BC = ACFG.bgm
LIB = Path(BC.out_dir)
CLAP_DIR = Path(CFG.paths.models) / "clap" / str(BC.clap_model).split("/")[-1]
TEXT_CACHE = Path(CFG.paths.cache) / "bgm_text"
TEMPLATES = ("{q}", "{q} background music", "anime soundtrack, {q}")
# director mood vocabulary -> a description CLAP understands better (free text also works; "_" is treated as a space)
MOOD_ALIASES = {
    "gentle_comedy": "light-hearted gentle comedy, playful pizzicato strings and woodwinds",
    "comedy": "funny playful comedic cue, bouncy and silly",
    "slapstick": "fast silly slapstick comedy cue",
    "everyday": "relaxed everyday school life, light acoustic guitar and piano",
    "slice_of_life": "relaxed everyday school life, light acoustic guitar and piano",
    "heartwarming": "warm heartwarming gentle piano and strings",
    "emotional": "emotional touching piano and strings",
    "romantic": "tender romantic soft piano",
    "sad": "sad melancholic slow piano",
    "lonely": "lonely quiet sparse piano",
    "tense": "tense suspenseful underscore, low drones",
    "awkward": "awkward quirky sparse comedic cue with pauses",
    "awkward_silence": "awkward quirky sparse cue, almost silent, hesitant",
    "nervous": "nervous anxious ticking staccato cue",
    "mysterious": "mysterious curious cue",
    "dramatic": "dramatic intense orchestral cue",
    "triumphant": "triumphant uplifting fanfare",
    "energetic": "energetic upbeat fast cue with drums",
    "calm": "calm peaceful ambient pads",
    "nostalgic": "nostalgic wistful gentle melody",
}


@lru_cache(maxsize=1)
def load_library() -> tuple[list[dict], np.ndarray]:
    """(cues, clap) - row i of clap.npy belongs to cues[i]."""
    cues = [json.loads(l) for l in (LIB / "cues.jsonl").read_text().splitlines() if l.strip()]
    E = np.load(LIB / "clap.npy").astype(np.float32)
    assert len(cues) == len(E), (len(cues), E.shape)
    for c in cues:
        c["abs_path"] = str(Path(CFG.paths.work) / c["path"])
    return cues, E


@lru_cache(maxsize=1)
def _text_model():
    import torch
    from transformers import AutoTokenizer, ClapTextModelWithProjection

    src = str(CLAP_DIR) if (CLAP_DIR / "config.json").exists() else str(BC.clap_model)
    dev = os.environ.get("KOMI_BGM_DEVICE", "cpu")
    model = ClapTextModelWithProjection.from_pretrained(src).eval().to(dev)
    return torch, AutoTokenizer.from_pretrained(src), model, dev


def embed_text(texts: list[str]) -> np.ndarray:
    """L2-normalised CLAP text embeddings [N, 512], cached on disk by content hash."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    out: list[np.ndarray | None] = []
    todo = []
    for t in texts:
        p = TEXT_CACHE / (hashlib.sha256(f"{BC.clap_model}|{t}".encode()).hexdigest()[:24] + ".npy")
        if p.exists():
            out.append(np.load(p))
        else:
            out.append(None)
            todo.append((len(out) - 1, t, p))
    if todo:
        torch, tok, model, dev = _text_model()
        inp = tok([t for _, t, _ in todo], padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            e = model(**{k: v.to(dev) for k, v in inp.items()}).text_embeds
        e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float32)
        for (i, _, p), v in zip(todo, e):
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, v)
            os.replace(tmp, p)
            out[i] = v
    return np.stack(out)


def query_embedding(text_query: str) -> np.ndarray:
    q = text_query.strip()
    q = MOOD_ALIASES.get(q.lower().replace(" ", "_"), q.replace("_", " "))
    e = embed_text([t.format(q=q) for t in TEMPLATES]).mean(axis=0)
    return e / max(float(np.linalg.norm(e)), 1e-9)


def retrieve(text_query: str, min_dur: float = 0.0, k: int = 5, max_dlg_ratio: float = 1.0, dialogue_penalty: float = 0.05,
             diverse: bool = True, exclude_ids: set[str] | None = None) -> list[dict]:
    """Top-k BGM cues for a mood string / free-text description.

    min_dur          only cues at least this long (s). If none is long enough the longest cues are returned with
                     "too_short": true (loop or cross-fade them).
    max_dlg_ratio    drop cues whose source span was covered by dialogue more than this (separation residue risk)
    dialogue_penalty score = cosine - penalty * dlg_ratio  (prefers spans that had little dialogue over them)
    diverse          skip cues that are near-duplicates (audio cosine > 0.92) of an already selected one - the same
                     track re-used in another episode
    """
    cues, E = load_library()
    if not cues:
        return []
    sim = E @ query_embedding(text_query)
    dlg = np.array([c.get("dlg_ratio", 0.0) for c in cues], np.float32)
    dur = np.array([c["dur"] for c in cues], np.float32)
    score = sim - dialogue_penalty * dlg
    ok = (dur >= min_dur) & (dlg <= max_dlg_ratio)
    if exclude_ids:
        ok &= np.array([c["id"] not in exclude_ids for c in cues])
    too_short = False
    if not ok.any():  # nothing long enough: fall back to the longest cues
        too_short = True
        ok = dur >= np.sort(dur)[-min(len(dur), max(k * 4, 20))]
    order = [i for i in np.argsort(-score) if ok[i]]
    picked: list[int] = []
    for i in order:
        if diverse and any(float(E[i] @ E[j]) > 0.92 for j in picked):
            continue
        picked.append(int(i))
        if len(picked) >= k:
            break
    out = []
    for i in picked:
        c = dict(cues[i])
        c["sim"] = round(float(sim[i]), 4)
        c["score"] = round(float(score[i]), 4)
        if too_short:
            c["too_short"] = True
        out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--min-dur", type=float, default=0.0)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    for c in retrieve(args.query, args.min_dur, args.k):
        print(f"{c['score']:.3f}  {c['id']}  {c['dur']:5.1f}s  dlg={c.get('dlg_ratio', 0):.2f}  tags={c.get('tags')}  {c['path']}")


if __name__ == "__main__":
    main()
