"""Checks of the v3 reading (src/v3/read.py) against signals it did not see.

  python -m src.v3.eval_read --chapters 1,2,58,64
1. Text: does the reading contain the texts the OCR found in each panel (and the other way round)?
2. Speakers: for bubbles that the dialogue alignment ties to a subtitle line of a TRAINING episode, the anime tells us who
   says the line (voice cluster name). That label is noisy but independent of any manga reading, so it scores both the
   new reading and the old page-by-page pass (where that pass was run).
"""
from __future__ import annotations

import argparse
import collections
import difflib
import re

from src.common.config import CFG, P
from src.common.io import read_json, read_jsonl, write_json

GENERIC = {"classmates", "other", "crowd", "unknown", "", "none", "?"}


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return max(0.75, min(len(a), len(b)) / max(len(a), len(b)))
    return difflib.SequenceMatcher(None, a, b).ratio()


def who(name: str | None, cast: set[str]) -> str:
    """Canonical speaker: a cast name, 'Narrator', or 'generic' (unnamed students, crowds, strangers)."""
    n = (name or "").strip()
    if n in cast or n == "Narrator":
        return n
    return "generic"


def eval_chapter(chapter: int, sub: str = "reading") -> dict | None:
    r = read_json(P("processed") / "v3" / sub / f"c{chapter:03d}.json")
    if not r:
        return None
    cast = {c["name"] for c in CFG.characters.roster} | {c["id"] for c in (read_json(P("processed") / "cards" / "cards.json") or [])}
    panels = {p["order"]: p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] == chapter}
    rd = {p["n"]: p for p in r["panels"]}
    named = collections.defaultdict(dict)
    for l in read_jsonl(P("processed") / "anime" / "lines_named.jsonl"):
        named[l["ep"]][l["idx"]] = l["speaker"]
    silver = {}
    for a in read_jsonl(P("processed") / "align" / "anchors.jsonl"):
        if a["chapter"] == chapter and a["sim"] >= 0.6 and a["episode"] in CFG.split.train_episodes:
            sp = named.get(a["episode"], {}).get(a["line"])
            if sp:
                silver[(a["panel_order"], a["bubble"])] = sp
    n_ocr = n_ocr_hit = n_rd = n_rd_hit = 0
    spk = collections.Counter()
    rows = []
    for n, p in panels.items():
        lines = [l for l in (rd.get(n) or {}).get("lines", [])]
        used = set()
        for bi, d in enumerate(p["dialogue"]):
            essential = d["kind"] != "sfx" and len(norm(d["text"])) >= 4
            best, bj = 0.0, None
            for j, l in enumerate(lines):
                s = sim(d["text"], l["text"])
                if s > best:
                    best, bj = s, j
            hit = best >= 0.6
            if essential:
                n_ocr += 1
                n_ocr_hit += hit
            if hit:
                used.add(bj)
            sv = silver.get((n, bi))
            if sv and hit:
                gold = who(sv, cast)
                new = who(lines[bj]["speaker"], cast)
                old = who(d.get("speaker"), cast) if p.get("caption") else None
                spk["n"] += 1
                spk["new_ok"] += new == gold
                if gold != "generic":
                    spk["n_cast"] += 1
                    spk["new_ok_cast"] += new == gold
                if old is not None:
                    spk["n_old"] += 1
                    spk["old_ok"] += old == gold
                    spk["new_ok_where_old"] += new == gold
                rows.append({"panel": n, "text": d["text"][:60], "anime": sv, "new": lines[bj]["speaker"], "old": d.get("speaker") if p.get("caption") else None})
        for j, l in enumerate(lines):
            if l["kind"] in ("speech", "thought", "narration") and len(norm(l["text"])) >= 4:
                n_rd += 1
                n_rd_hit += j in used
    return {"chapter": chapter, "panels": len(panels), "panels_read": len(rd), "ocr_texts": n_ocr, "ocr_texts_found_in_reading": n_ocr_hit,
            "reading_lines": n_rd, "reading_lines_found_in_ocr": n_rd_hit, "speaker": dict(spk), "rows": rows,
            "with_reference_pictures": r.get("with_reference_pictures")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--show", type=int, default=0, help="print up to N speaker disagreements per chapter")
    ap.add_argument("--dir", default="reading", help="which reading to score: data/processed/v3/<dir>/")
    a = ap.parse_args()
    tot = collections.Counter()
    out = []
    for c in [int(x) for x in a.chapters.split(",") if x]:
        e = eval_chapter(c, a.dir)
        if not e:
            print(f"chapter {c}: no reading")
            continue
        out.append(e)
        s = e["speaker"]
        print(f"chapter {c}: panels {e['panels_read']}/{e['panels']} | OCR texts found in reading {e['ocr_texts_found_in_reading']}/{e['ocr_texts']}"
              f" | reading lines backed by OCR {e['reading_lines_found_in_ocr']}/{e['reading_lines']}"
              f" | speaker = anime's speaker: new {s.get('new_ok', 0)}/{s.get('n', 0)} (cast lines {s.get('new_ok_cast', 0)}/{s.get('n_cast', 0)})"
              + (f" | old pass {s.get('old_ok', 0)}/{s.get('n_old', 0)} vs new on the same lines {s.get('new_ok_where_old', 0)}/{s.get('n_old', 0)}" if s.get("n_old") else ""))
        for k in ("ocr_texts", "ocr_texts_found_in_reading", "reading_lines", "reading_lines_found_in_ocr"):
            tot[k] += e[k]
        tot.update(s)
        bad = [r for r in e["rows"] if r["new"] != r["anime"]][: a.show]
        for r in bad:
            print("    ", r)
    if out:
        print("TOTAL:", dict(tot))
        write_json(P("processed") / "v3" / a.dir / "eval.json", {"chapters": [{k: v for k, v in e.items() if k != "rows"} for e in out], "total": dict(tot)})


if __name__ == "__main__":
    main()
