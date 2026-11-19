"""Auto-name subtitle lines and voice clusters (brief §5.2).

Votes for each voice cluster come from (a) manga speaker labels of aligned bubbles (Magi + VLM), and (b) the local
LLM's attribution of each subtitle line from the script context (text only). The majority name of a cluster then
labels all its lines; the review page (tools/review) shows low-confidence clusters. Never blocks on a human.

  python -m src.anime.name_speakers            # needs the local Qwen server; spk clusters optional
Outputs: data/processed/anime/lines_named.jsonl  {"ep","idx","start","end","text","speaker","conf","source"}
         names + confidence written into data/processed/anime/spk/clusters.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json, write_jsonl
from src.llm.client import LLM

SCHEMA = {"type": "object", "required": ["lines"], "properties": {"lines": {"type": "array", "items": {
    "type": "object", "required": ["i", "speaker"], "properties": {"i": {"type": "integer"}, "speaker": {"type": "string"}}}}}}

SYSTEM = (
    "You label who speaks each subtitle line of the anime \"Komi Can't Communicate\". You get consecutive lines of one "
    "episode with an index; some lines carry a hint `manga:<name>` taken from the matching manga speech bubble. "
    "Use the dialogue flow, how people address each other, speech style and the hints.\n"
    "Speakers (use exactly these names): {names}, Narrator (the calm female narrator explaining things), "
    "Classmates (any unnamed student or crowd), Other (adults, staff, strangers).\n"
    "Notes: Komi almost never speaks aloud (short stammers like \"I-I...\" can be hers); Tadano does most inner monologue "
    "and retorts; Najimi is loud and playful; lines in the style of rules/explanations are the Narrator.\n"
    "Return ONE JSON object {{\"lines\": [{{\"i\": <index>, \"speaker\": \"<name>\"}}]}} covering every index."
)


def llm_attribution(ep: int, lines: list[dict], hints: dict[int, str], llm: LLM, chunk: int = 50, overlap: int = 6) -> dict[int, str]:
    names = ", ".join(c["name"] for c in CFG.characters.roster if c["role"] in ("main", "recurring", "family", "minor"))
    valid = {c["name"] for c in CFG.characters.roster} | {"Narrator", "Classmates", "Other"}
    out: dict[int, str] = {}
    jobs = []
    for a in range(0, len(lines), chunk):
        part = lines[max(0, a - overlap) : a + chunk]
        txt = "\n".join(f"[{l['idx']}] ({l['start']:.0f}s) {l['text']}" + (f"   manga:{hints[l['idx']]}" if l["idx"] in hints else "") for l in part)
        jobs.append((a, part, txt))

    def one(job):
        a, part, txt = job
        try:
            res = llm.chat([{"role": "system", "content": SYSTEM.format(names=names)},
                            {"role": "user", "content": f"Episode {ep}, lines:\n{txt}"}], schema=SCHEMA, max_tokens=2500, temperature=0.0,
                           tag=f"speakers:ep{ep}")
            return a, part, res.get("lines", [])
        except Exception as e:
            log(f"ep{ep:02d} chunk {a}: {str(e)[:120]}")
            return a, part, []

    with ThreadPoolExecutor(int(CFG.llm.local.parallel)) as ex:
        for a, part, res in ex.map(one, jobs):
            core = {l["idx"] for l in lines[a : a + chunk]}
            for r in res:
                if r.get("i") in core and r.get("speaker") in valid:
                    out[int(r["i"])] = r["speaker"]
    return out


def run(episodes: list[int] | None = None) -> None:
    eps = episodes or list(range(1, 25))
    llm = LLM("local", role="speakers")
    anchors = read_jsonl(P("processed") / "align" / "anchors.jsonl")
    panels = {(r["chapter"], r["order"]): r for r in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    hint_by_ep: dict[int, dict[int, str]] = defaultdict(dict)
    for a in anchors:
        if a["sim"] < 0.7:
            continue
        p = panels.get((a["chapter"], a["panel_order"]))
        if p and a["bubble"] < len(p["dialogue"]) and p["dialogue"][a["bubble"]].get("speaker"):
            hint_by_ep[a["episode"]][a["line"]] = p["dialogue"][a["bubble"]]["speaker"]
    clusters = read_json(P("processed") / "anime" / "spk" / "clusters.json") or {}
    assign = clusters.get("assignments", {})
    rows = []
    for ep in eps:
        ev = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        lines = [{"idx": i, **e} for i, e in enumerate(ev) if e["kind"] == "dialogue"]
        if not lines:
            continue
        lab = llm_attribution(ep, lines, hint_by_ep.get(ep, {}), llm)
        for l in lines:
            rows.append({"ep": ep, "idx": l["idx"], "start": l["start"], "end": l["end"], "text": l["text"],
                         "llm": lab.get(l["idx"]), "manga": hint_by_ep.get(ep, {}).get(l["idx"]),
                         "cluster": assign.get(f"ep{ep:02d}:{l['idx']}")})
        log(f"ep{ep:02d}: {len(lines)} lines, {len(lab)} labelled by LLM, {len(hint_by_ep.get(ep, {}))} manga hints")
    # ---- cluster naming by votes (TRAIN episodes only build the names; held-out lines just receive them)
    train = set(CFG.split.train_episodes)
    votes: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        if r["cluster"] is None or r["ep"] not in train:
            continue
        if r["manga"]:
            votes[r["cluster"]][r["manga"]] += 2          # manga-aligned labels count double
        if r["llm"] and r["llm"] not in ("Other",):
            votes[r["cluster"]][r["llm"]] += 1
    cname: dict[int, tuple[str, float, int]] = {}
    for cid, cnt in votes.items():
        (name, n), total = cnt.most_common(1)[0], sum(cnt.values())
        cname[cid] = (name, n / total, total)
    for c in clusters.get("clusters", []):
        if c["id"] in cname:
            name, purity, total = cname[c["id"]]
            c["name"], c["confidence"], c["votes"] = (name if purity >= 0.5 and total >= 6 else None), round(purity, 3), total
    if clusters:
        write_json(P("processed") / "anime" / "spk" / "clusters.json", clusters)
    out = []
    for r in rows:
        name, conf, src = None, 0.0, "none"
        cn = cname.get(r["cluster"]) if r["cluster"] is not None else None
        if cn and cn[1] >= 0.6 and cn[2] >= 8 and cn[0] not in ("Classmates", "Other"):
            name, conf, src = cn[0], cn[1], "cluster"
            if r["manga"] and r["manga"] == r["llm"] and r["manga"] != name:   # two independent sources agree against the cluster
                name, conf, src = r["manga"], 0.7, "manga+llm"
        elif r["manga"] and r["manga"] == r["llm"]:
            name, conf, src = r["manga"], 0.8, "manga+llm"
        elif r["manga"]:
            name, conf, src = r["manga"], 0.55, "manga"
        elif r["llm"]:
            name, conf, src = r["llm"], 0.5, "llm"
        if name:
            out.append({"ep": r["ep"], "idx": r["idx"], "start": r["start"], "end": r["end"], "text": r["text"],
                        "speaker": name, "conf": round(conf, 3), "source": src})
    # voice references may only come from TRAIN episodes: the default file is train-only; the full file is for evaluation
    write_jsonl(P("processed") / "anime" / "lines_named.jsonl", [o for o in out if o["ep"] in train])
    write_jsonl(P("processed") / "anime" / "lines_named_all.jsonl", out)
    cnt = Counter(o["speaker"] for o in out if o["ep"] in train)
    log(f"lines named: {len(out)}; top speakers (train): {cnt.most_common(12)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=None)
    a = ap.parse_args()
    run([int(x) for x in a.episodes.split(",")] if a.episodes else None)
