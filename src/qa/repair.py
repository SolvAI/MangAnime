"""Repair ladder (brief §8): escalate on each failure, max ~4 attempts per shot.

  rung 1  reseed            - same prompt, new seed
  rung 2  rewrite_prompt    - local Qwen rewrites the video prompt from the judge's failure_reasons + repair_hint
  rung 3  regen_keyframe    - new keyframe (new seed + hint), then video again
  rung 4  simplify          - fewer characters, less motion
  rung 5  fallback          - the keyframe with a camera pan/zoom (always succeeds technically)

The judge may suggest a rung; we never go *down* the ladder and always move at least one rung per failure.
Every failure is logged with its reason; `review_list()` produces the human-review list.
"""
from __future__ import annotations

from pathlib import Path

from src.common.config import P
from src.common.io import append_jsonl, read_jsonl, write_json

RUNGS = ["initial", "reseed", "rewrite_prompt", "regen_keyframe", "simplify", "fallback"]
MAX_ATTEMPTS = 4          # generated attempts before the pan/zoom fallback

REWRITE_SCHEMA = {"type": "object", "required": ["video_hint", "keyframe_hint"],
                  "properties": {"video_hint": {"type": "string"}, "keyframe_hint": {"type": "string"}}}

REWRITE_SYSTEM = (
    "You fix prompts for an anime video generator (image-to-video) and its keyframe editor. "
    "Given a shot spec, the previous prompt and the QA judge's failure reasons, write short corrective additions. "
    "video_hint: one or two sentences to append to the video prompt (what must stay stable, what motion to avoid or use). "
    "keyframe_hint: one or two sentences to append to the keyframe prompt (what must be drawn differently). "
    "Be concrete and visual; do not repeat the spec. Return ONE JSON object {\"video_hint\": ..., \"keyframe_hint\": ...}."
)


_MOTION_PAT = None


def motion_failure(verdict: dict) -> bool:
    """True when the failure is about what happens DURING the shot (morphing, flicker, deformation, chaotic motion).

    Only then does the pan/zoom fallback help: it shows the same keyframe as a clean still. A failure about the
    keyframe's content (wrong character, spec mismatch, manga leftovers) would survive the fallback unchanged, so the
    animated clip is kept and the shot is listed for review instead."""
    import re

    global _MOTION_PAT
    if _MOTION_PAT is None:
        _MOTION_PAT = re.compile(r"morph|melt|flicker|distort|deform|warp|jitter|shak|chaotic|unstable|glitch|drift|extra (limb|arm|finger|hand)|"
                                 r"multipl|smear|ghost|during the shot|between frames|from frame|mid-shot|changes? (shape|design|outfit) ", re.I)
    sc = (verdict.get("scores") or {}).get("motion")
    if isinstance(sc, (int, float)) and sc <= 5:
        return True
    return bool(_MOTION_PAT.search(" ".join(str(r) for r in verdict.get("failure_reasons", []))))


def next_rung(current: str, suggested: str | None, stage: str = "video") -> str:
    i = RUNGS.index(current) if current in RUNGS else 0
    j = RUNGS.index(suggested) if suggested in RUNGS else 0
    if stage == "keyframe":          # a bad first frame cannot be fixed by re-rolling the video
        j = max(j, RUNGS.index("regen_keyframe"))
    return RUNGS[min(len(RUNGS) - 1, max(i + 1, j))]


def rewrite_hints(llm, shot: dict, prev_prompt: str, verdict: dict) -> dict:
    """Rung 2: the local Qwen turns failure reasons into prompt additions."""
    import json

    user = ("SHOT SPEC:\n" + json.dumps({k: shot.get(k) for k in ("shot_size", "camera", "characters", "first_frame", "action", "location")}, ensure_ascii=False)
            + "\n\nPREVIOUS VIDEO PROMPT:\n" + prev_prompt
            + "\n\nJUDGE FAILURE REASONS:\n" + json.dumps(verdict.get("failure_reasons", []), ensure_ascii=False)
            + "\n\nJUDGE REPAIR HINT:\n" + str(verdict.get("repair_hint", "")))
    try:
        return llm.chat([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}], schema=REWRITE_SCHEMA,
                        max_tokens=300, temperature=0.3, tag="repair_rewrite")
    except Exception:
        hint = str(verdict.get("repair_hint", ""))[:300]
        return {"video_hint": hint, "keyframe_hint": hint}


def simplify(shot: dict) -> dict:
    """Rung 4: fewer characters, less motion."""
    s = dict(shot)
    speakers = [d["speaker"] for d in s.get("dialogue", []) if d["kind"] == "speech" and d["speaker"] in s.get("characters", [])]
    main = speakers[:1] or s.get("characters", [])[:1]
    s["characters"] = main
    s["action"] = "almost static held pose; only subtle blinking" + (" and mouth flaps while speaking" if speakers else "")
    s["camera"] = "static"
    s["first_frame"] = (f"{main[0]} alone in frame, " if main else "") + (s.get("first_frame") or "")
    s["simplified"] = True
    return s


def log_failure(run_dir: Path, shot_id: str, attempt: int, rung: str, verdict: dict) -> None:
    append_jsonl(run_dir / "qa_failures.jsonl", {"shot_id": shot_id, "attempt": attempt, "rung": rung, "stage": verdict.get("stage"),
                                                  "reasons": verdict.get("failure_reasons", []), "hint": verdict.get("repair_hint", ""),
                                                  "scores": verdict.get("scores", {})})


def review_list(run_dir: Path, state: dict) -> list[dict]:
    """Human-review list: shots that ended on the fallback or passed only after 3+ attempts."""
    fails = read_jsonl(run_dir / "qa_failures.jsonl")
    by = {}
    for f in fails:
        by.setdefault(f["shot_id"], []).append(f)
    rows = []
    for sid, st in state.items():
        if st.get("final") == "fallback" or st.get("attempts", 0) >= 3 or st.get("verdict") in ("fail", "fail_kept"):
            rows.append({"shot_id": sid, "final": st.get("final"), "attempts": st.get("attempts"), "last_verdict": st.get("verdict"),
                         "history": by.get(sid, [])})
    write_json(run_dir / "human_review.json", rows)
    return rows
