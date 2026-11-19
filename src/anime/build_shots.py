"""shots.jsonl (brief §5.2): shot boundaries + VLM captions + speaker-named dialogue.

  python -m src.anime.build_shots caption --episodes 22,23,24 [--spans "ep:start-end,..."]   (needs the local Qwen server)
  python -m src.anime.build_shots merge
Captions are stored per episode in data/processed/anime/shot_captions/epNN.jsonl (resumable); `merge` writes
data/processed/anime/shots.jsonl in the brief's schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.anime.oped import is_content, non_content
from src.common.config import P
from src.common.io import log, read_json, read_jsonl, write_jsonl

CAP_DIR = P("processed") / "anime" / "shot_captions"


def keyframes(ep: int, shot: int) -> list[str]:
    d = P("interim") / "anime" / f"ep{ep:02d}" / "keyframes"
    return [str(d / f"s{shot:04d}_{t}.jpg") for t in "amb"]


def caption_episode(ep: int, spans: list[tuple[float, float]] | None = None, batch: int = 32, max_per_span: int = 0) -> int:
    from src.anime.caption import caption_many

    shots = read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or []
    nc = non_content(ep)
    subs = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
    done = {r["shot"]: r for r in read_jsonl(CAP_DIR / f"ep{ep:02d}.jsonl")}
    todo = []
    for s in shots:
        if s["shot"] in done or not is_content(ep, s["start"], s["end"], nc):
            continue
        if spans and not any(a <= (s["start"] + s["end"]) / 2 <= b for a, b in spans):
            continue
        if not all(Path(k).exists() for k in keyframes(ep, s["shot"])):
            continue
        todo.append(s)
    if spans and max_per_span:      # few-shot examples only show the first N shots of a chapter
        keep = []
        for a, b in spans:
            keep += [s for s in todo if a <= (s["start"] + s["end"]) / 2 <= b][:max_per_span]
        todo = keep
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        items = []
        for s in chunk:
            line = " / ".join(e["text"] for e in subs if e["kind"] == "dialogue" and e["start"] < s["end"] and e["end"] > s["start"])
            fr = keyframes(ep, s["shot"])
            items.append({"frames": [fr[0], fr[2]] if s["end"] - s["start"] > 1.5 else fr[1:2], "subtitle": line})   # first + last frame
        for s, c in zip(chunk, caption_many(items)):
            if c and "error" not in c:
                done[s["shot"]] = {"episode": ep, "shot": s["shot"], **c}
        write_jsonl(CAP_DIR / f"ep{ep:02d}.jsonl", [done[k] for k in sorted(done)])
        log(f"ep{ep:02d}: captioned {min(i + batch, len(todo))}/{len(todo)} shots")
    return len(todo)


def merge() -> int:
    named = {}
    for l in read_jsonl(P("processed") / "anime" / "lines_named_all.jsonl") or read_jsonl(P("processed") / "anime" / "lines_named.jsonl"):
        named[(l["ep"], l["idx"])] = l["speaker"]
    rows = []
    for ep in range(1, 25):
        shots = read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or []
        caps = {r["shot"]: r for r in read_jsonl(CAP_DIR / f"ep{ep:02d}.jsonl")}
        subs = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        nc = non_content(ep)
        for s in shots:
            c = caps.get(s["shot"], {})
            dlg = [{"speaker": named.get((ep, i)), "text": e["text"], "start": e["start"], "end": e["end"], "kind": e["kind"]}
                   for i, e in enumerate(subs) if e["kind"] in ("dialogue", "notebook") and s["start"] <= (e["start"] + e["end"]) / 2 < s["end"]]
            rows.append({"episode": ep, "shot": s["shot"], "start": s["start"], "end": s["end"], "keyframes": keyframes(ep, s["shot"]),
                         "characters": c.get("characters", []), "dialogue": dlg, "shot_size": c.get("shot_size"), "camera": c.get("camera"),
                         "caption": c.get("caption"), "action": c.get("action"), "clip": None, "content": is_content(ep, s["start"], s["end"], nc)})
    write_jsonl(P("processed") / "anime" / "shots.jsonl", rows)
    log(f"shots.jsonl: {len(rows)} shots, {sum(1 for r in rows if r['caption'])} captioned")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["caption", "merge"])
    ap.add_argument("--episodes", default="")
    ap.add_argument("--spans", default="", help='restrict to time spans, e.g. "3:120-480,5:0-300"')
    ap.add_argument("--max-per-span", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "merge":
        merge()
    else:
        spans: dict[int, list[tuple[float, float]]] = {}
        for part in filter(None, a.spans.split(",")):
            e, r = part.split(":")
            lo, hi = r.split("-")
            spans.setdefault(int(e), []).append((float(lo), float(hi)))
        for ep in [int(x) for x in a.episodes.split(",") if x] or sorted(spans):
            caption_episode(ep, spans.get(ep), max_per_span=a.max_per_span)
