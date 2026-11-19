"""Episode-level continuity agent (brief §8): outfits, location, time of day and props across consecutive shots.

Evidence = the local VLM's descriptions of every keyframe (already produced by the keyframe QA round) + the shot specs.
The DeepSeek judge reads the whole sequence as text and flags shots that break continuity, with a repair hint that is
appended to the keyframe prompt when the shot is regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.io import log, read_json
from src.llm.client import LLM, BudgetExceeded, remote_or_local

SCHEMA = {"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array", "items": {
    "type": "object", "required": ["shot_id", "issue"],
    "properties": {"shot_id": {"type": "string"}, "issue": {"type": "string"}, "category": {"type": "string"},
                   "severity": {"type": "string"}, "fix_keyframe": {"type": "boolean"}, "hint": {"type": "string"}}}}}}

SYSTEM = """You are the continuity supervisor of an anime episode. You get the shot list in order. For each shot:
the director's spec (location, time of day, characters) and what a vision model actually saw in the generated keyframe.
Find continuity errors a viewer would notice between CONSECUTIVE shots of the same scene:
- outfit changes (uniform vs casual clothes, missing blazer, wrong colours), hair changes
- location or background that jumps although the scene did not change
- time of day / lighting that jumps (day -> sunset -> day)
- props that appear/disappear (bag, notebook, chocolate box), characters who vanish or multiply
Ignore deliberate scene changes (the spec's location changes), inserts, title cards, chibi/comedy style cut-ins.
Return ONE JSON object: {"issues": [{"shot_id": "...", "category": "outfit|location|time_of_day|prop|character",
"severity": "low|medium|high", "issue": "what is inconsistent with which neighbouring shot",
"fix_keyframe": true|false, "hint": "instruction to append to the keyframe prompt so it matches its neighbours"}]}
Only list real problems (medium/high get fix_keyframe=true). An empty list is a valid answer."""


def check_continuity(run_dir: Path, shots: list[dict], chunk: int = 60) -> dict:
    state = read_json(run_dir / "state.json") or {}
    rows = []
    for s in shots:
        st = state.get(s["shot_id"], {})
        qa = read_json(run_dir / "shots" / s["shot_id"] / f"qa_keyframe_a{st.get('kf_attempt', 0)}.json") or {}
        vis = (qa.get("evidence") or {}).get("visual") or {}
        rows.append({"shot_id": s["shot_id"], "spec": {k: s.get(k) for k in ("location", "time_of_day", "characters", "shot_size", "first_frame")},
                     "seen": {k: vis.get(k) for k in ("description", "characters", "setting")}})
    issues = []
    llm = remote_or_local("judge")
    for i in range(0, len(rows), chunk):
        part = rows[max(0, i - 2) : i + chunk]          # 2 shots of overlap for context
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "SHOTS IN ORDER:\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in part)}]
        try:
            res = llm.chat(msgs, schema=SCHEMA, max_tokens=3000, temperature=0.0, thinking=False, tag="continuity")
        except BudgetExceeded:
            llm = LLM("local", role="judge")
            res = llm.chat(msgs, schema=SCHEMA, max_tokens=3000, temperature=0.0, tag="continuity")
        except Exception as e:
            log(f"continuity chunk failed: {str(e)[:160]}")
            continue
        issues += res.get("issues", [])
    seen, uniq = set(), []
    for it in issues:
        if it.get("shot_id") not in seen:
            seen.add(it.get("shot_id"))
            uniq.append(it)
    return {"n_shots": len(shots), "issues": uniq}
