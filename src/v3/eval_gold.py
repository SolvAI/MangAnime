"""Who is in the panel? Readings scored against a hand-labelled sample of held-out panels.

  python -m src.v3.eval_gold [--dirs reading,reading_pics,reading_cast32,reading_textonly]
Gold: data/processed/v3/gold/gold.json - 48 story panels drawn at random from chapters 114, 116, 117, 118 (12 each) and
labelled by the assistant from the panel crops and the cards' pictures BEFORE looking at any reading (2026-11-12; sheets
label_sheet_*.jpg and cast_*.jpg in the same folder). Per panel:
  required  people clearly drawn and identifiable (a reading must list them)
  optional  people it may list without penalty (tiny chibi faces, a figure seen from behind, a hand); "*" = the panel has
            figures nobody could name from the drawing, so extra names are not counted
Scores: found = required people the reading lists; extra = listed known characters that are neither required nor optional
(only on panels without "*"); exact = panels with every required person and no extra.
The old page-by-page pass (`panels.jsonl`, v1/v2) is scored the same way as "old_page_pass".
"""
from __future__ import annotations

import argparse

from src.common.config import P
from src.common.io import read_json, read_jsonl, write_json
from src.v3.cast import canon, is_known

G = P("processed") / "v3" / "gold"


def score(name: str, cast_of) -> dict:
    gold = read_json(G / "gold.json")["panels"]
    n_req = found = extra = exact = strict_panels = panels = 0
    rows = []
    for g in gold:
        got = cast_of(g["chapter"], g["panel"])
        if got is None:
            continue
        got = {canon(x) for x in got if is_known(x)}
        req, opt = set(g["required"]), set(g.get("optional", []))
        panels += 1
        n_req += len(req)
        f = len(req & got)
        found += f
        ex = set() if "*" in opt else got - req - opt
        if "*" not in opt:
            strict_panels += 1
            extra += len(ex)
        ok = f == len(req) and not ex
        exact += ok
        if not ok:
            rows.append({"k": g["k"], "chapter": g["chapter"], "panel": g["panel"], "missing": sorted(req - got), "extra": sorted(ex)})
    return {"reading": name, "panels": panels, "required_people": n_req, "found": found, "found_share": round(found / max(1, n_req), 3),
            "extra_people": extra, "panels_checked_for_extras": strict_panels, "exact_panels": exact, "exact_share": round(exact / max(1, panels), 3), "misses": rows}


def score_speakers(name: str, lines_of) -> dict:
    """Lines of the labelled panels whose speaker is certain from the drawing: does the reading give them to that person?"""
    from src.v3.eval_read import sim

    n = right = not_found = 0
    wrong = []
    for g in read_json(G / "gold.json")["panels"]:
        got = lines_of(g["chapter"], g["panel"])
        if got is None:
            continue
        for l in g.get("lines", []):
            n += 1
            best = max(got, key=lambda x: sim(l["text"], x[0]), default=None)
            if best is None or sim(l["text"], best[0]) < 0.5:
                not_found += 1
                continue
            if canon(best[1]) in l["speakers"]:
                right += 1
            else:
                wrong.append({"k": g["k"], "text": l["text"][:40], "gold": l["speakers"], "reading": best[1]})
    return {"reading": name, "lines": n, "right": right, "text_not_found": not_found, "wrong": wrong, "right_share": round(right / max(1, n), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="reading,reading_pics,reading_cast32,reading_textonly")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    res = []
    for sub in a.dirs.split(","):
        cache: dict[int, dict] = {}

        def cast_of(ch, n, sub=sub, cache=cache):
            if ch not in cache:
                r = read_json(P("processed") / "v3" / sub / f"c{ch:03d}.json")
                cache[ch] = {p["n"]: [v["id"] for v in p.get("visible", [])] for p in r["panels"]} if r else None
            return None if cache[ch] is None else cache[ch].get(n, [])

        res.append(score(sub, cast_of))
        lcache: dict[int, dict] = {}

        def lines_of(ch, n, sub=sub, lcache=lcache):
            if ch not in lcache:
                r = read_json(P("processed") / "v3" / sub / f"c{ch:03d}.json")
                lcache[ch] = {p["n"]: [(l["text"], l["speaker"]) for l in p.get("lines", [])] for p in r["panels"]} if r else None
            return None if lcache[ch] is None else lcache[ch].get(n, [])

        res[-1]["speakers"] = score_speakers(sub, lines_of)
    old = {(p["chapter"], p["order"]): p.get("characters") or [] for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p.get("caption")}
    if old:
        res.append(score("old_page_pass", lambda ch, n: old.get((ch, n))))
        oldl = {(p["chapter"], p["order"]): [(d["text"], d.get("speaker") or "") for d in p.get("dialogue", [])]
                for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p.get("caption")}
        res[-1]["speakers"] = score_speakers("old_page_pass", lambda ch, n: oldl.get((ch, n)))
    for s in res:
        print(f"{s['reading']:18s} panels {s['panels']:2d} | required people found {s['found']}/{s['required_people']} ({s['found_share']:.0%})"
              f" | extra names {s['extra_people']} on {s['panels_checked_for_extras']} panels | exact panels {s['exact_panels']}/{s['panels']} ({s['exact_share']:.0%})")
        sp = s.get("speakers")
        if sp:
            print(f"{'':18s} speaker of {sp['lines']} labelled lines: right {sp['right']} ({sp['right_share']:.0%}), wrong {len(sp['wrong'])}, line not in the reading {sp['text_not_found']}")
        if a.show:
            for r in s["misses"] + (sp or {}).get("wrong", []):
                print("     ", r)
    write_json(G / "scores.json", res)


if __name__ == "__main__":
    main()
