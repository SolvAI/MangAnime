"""QA judge (brief §8.2): DeepSeek V4.1 Flash decides pass/fail from the shot spec + local evidence bundle.

A metric alone never passes or fails a shot; the judge cites the evidence. DeepSeek V4.1 Flash accepts images, so the
bundle includes up to 3 downscaled frames (384 px) besides the text evidence. Calls are cached by hash(spec+evidence).
Disagreements between the verdict and any single metric are logged for calibration review.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.config import P
from src.common.io import append_jsonl
from src.llm.client import LLM, BudgetExceeded, remote_or_local

VERDICT_SCHEMA = {
    "type": "object",
    "required": ["verdict", "scores", "failure_reasons", "evidence_cited", "repair_hint"],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "scores": {"type": "object", "required": ["identity", "spec_match", "motion", "audio"]},
        "failure_reasons": {"type": "array"},
        "evidence_cited": {"type": "array"},
        "repair_hint": {"type": "string"},
        "suggested_repair": {"type": "string"},
        "metric_disagreement": {"type": "string"},
    },
}

SYSTEM = """You are the QA judge of an automatic anime production pipeline ("Komi Can't Communicate" style).
For ONE generated shot you receive: the shot specification, an evidence bundle produced by small local models, and
(when available) a few downscaled frames of the shot in time order. Decide whether the shot is good enough to ship.

How to read the evidence
- identity: CCIP differences per detected face against the character bank (LOWER is more similar; a face is the same
  character when diff <= threshold). `expected_diffs` are the differences to the characters the spec asks for.
  No face detected is normal for inserts, back views, wide shots and title cards.
- visual: captions and an artifact checklist written by a local vision model that looked at the frames.
- metrics.flicker / warp / motion: temporal stability (higher warp error and flicker = less stable; a held-frame
  ratio of 0.5-0.7 is expected because clips are animated "on twos" or "on threes" like TV anime). metrics.dover: no-reference quality scores.
- fidelity: line-art similarity between the source manga panel and the keyframe (0.3 is weak, 0.6+ is strong).
- audio: per-line ASR transcript, character error rate vs the Japanese script, speaker similarity, duration fit.
- metrics_vs_real: for the main metrics, this clip's value next to where REAL TV-anime clips fall (5th / 50th / 95th
  percentile) and a flag. "within real range" means the number alone says nothing. "BELOW THE REAL RANGE" on dover.* =
  blurred / noisy / degraded picture -> fail. "ABOVE THE REAL RANGE" or "high" on motion.global_motion_mag (the whole
  frame moves) while the spec's camera is static and the frames show no deliberate pan = camera shake / jitter -> fail.
  The same flags on flicker.* = flicker, on warp.* = the drawing deforms between frames: fail unless the frames clearly
  show an innocent cause (a fast pan, a scene with flashing lights) and say which.
- The local numbers are evidence, not truth. When metrics and captions (or the frames you see) disagree, say so in
  `metric_disagreement` and explain which one you trust.
- On-screen text: this anime itself shows clean typeset or hand-lettered JAPANESE text on screen (narration captions,
  sound words, notebook writing, floating labels, thought balloons). That is the show's style, NOT a manga leftover.
  Manga leftovers are: English lettering, empty white speech balloons, balloon tails covering characters, panel borders
  or gutters, screentone dot patterns, garbled / illegible glyphs.

Fail the shot when: the wrong character appears or the main character is unrecognisable; faces melt, limbs multiply,
the character morphs during the shot; manga leftovers (speech bubbles, panel borders, lettering) remain; the content
contradicts the spec in a way a viewer would notice (wrong number of people, wrong setting, wrong action); motion is
chaotic or the image visibly flickers; a spoken line is missing or unintelligible (CER > 0.5) or does not fit the shot.
Do NOT fail for: subtle or limited motion (TV anime is mostly held poses), small framing differences, a background
that differs from the manga, unknown background extras, minor colour differences.

Return ONE JSON object:
{"verdict": "pass" | "fail",
 "scores": {"identity": 0-10, "spec_match": 0-10, "motion": 0-10, "audio": 0-10},   // use 10 when not applicable
 "failure_reasons": ["..."],            // empty when pass
 "evidence_cited": ["identity.frames[0].faces[0].diff=0.31 > 0.18", "..."],  // which evidence supports each reason
 "repair_hint": "one concrete instruction for the next attempt (prompt wording, what to change)",
 "suggested_repair": "reseed" | "rewrite_prompt" | "regen_keyframe" | "simplify" | "fallback",
 "metric_disagreement": "" }"""


def spec_view(shot: dict) -> dict:
    keep = ("shot_id", "duration", "shot_size", "camera", "characters", "location", "time_of_day", "first_frame", "action", "notes")
    v = {k: shot.get(k) for k in keep}
    v["dialogue"] = [{k: d.get(k) for k in ("speaker", "kind", "text", "text_ja", "emotion")} for d in shot.get("dialogue", [])]
    return v


KEYFRAME_NOTE = (
    " (only the first frame exists yet; set motion and audio to 10). HOW THIS FRAME WAS MADE: it is a colourised 16:9 crop of "
    "the source manga panel described in evidence.source_panel. Judge it against THAT panel: are the panel's characters and "
    "scene shown as a clean colour frame, with on-model faces and no leftover speech balloons, lettering, screentone or "
    "panel borders? The director's first_frame / shot_size wording is an intention, not a requirement: do NOT fail for a "
    "framing, crowd size or background that differs from the spec text when it matches the source panel. Fail for: wrong or "
    "unrecognisable main character, distorted faces/anatomy, visible manga leftovers, an image that is still a "
    "black-and-white or multi-panel manga page, or content that contradicts the source panel.")


KEYFRAME_NOTE_REDRAW = (
    " (only the first frame exists yet; set motion and audio to 10). HOW THIS FRAME WAS MADE: an anime image model REDREW the "
    "shot from the source manga panel (used as a loose layout reference, described in evidence.source_panel) and from the shot "
    "spec. It does not have to follow the panel line for line; framing, crowd size and background may differ. Pass it when it is "
    "a clean, finished TV-anime frame that shows the spec's characters (recognisable, on-model, right hair and outfit) doing "
    "what the spec / panel describes. WHO MUST BE IN THE FRAME is given by evidence.intended_frame (written by a vision model "
    "that looked at the panel: people with name or 'unknown' + look, pose, expression; total head count; situation). The "
    "spec's character list comes from the script and sometimes names someone the panel does not show: never fail for a spec "
    "character that intended_frame does not list, and judge an 'unknown' person by the described look. Fail for: a person "
    "of intended_frame missing or unrecognisable, a named character who looks like someone else, the same character drawn "
    "twice (clones), clearly more or fewer people than intended, distorted faces / hands / anatomy, leftover speech balloons "
    "or lettering, a black-and-white or multi-panel image, and ANYTHING unsuitable for an all-ages show (start that reason "
    "with the word UNSUITABLE).")


def judge_shot(shot: dict, evidence: dict, *, stage: str = "video", send_frames: bool = True, llm: LLM | None = None) -> dict:
    """stage: 'keyframe' (first frame only; motion/audio not applicable) or 'video'."""
    import os

    redraw = os.environ.get("KOMI_KF_BACKEND", "qwen_edit") == "illustrious"
    frames = [f for f in evidence.get("frames", []) if Path(f).exists()][:3] if send_frames else []
    bundle = {k: v for k, v in evidence.items() if k not in ("frames",)}
    try:
        from src.qa.metric_ref import versus_real

        vr = versus_real(evidence.get("metrics"))
        if vr:
            bundle["metrics_vs_real"] = vr
    except Exception:
        pass
    user = (f"STAGE: {stage}" + ((KEYFRAME_NOTE_REDRAW if redraw else KEYFRAME_NOTE) if stage == "keyframe" else
                                 (" The first frame was redrawn by an anime image model from the source manga panel (layout reference, "
                                  "evidence.source_panel) and the spec. WHO must be in the shot is given by evidence.intended_frame (the "
                                  "people the panel actually shows; 'unknown' = judge by the described look): never fail for a spec "
                                  "character that intended_frame does not list. Judge the action and the dialogue against the spec, and "
                                  "fail anything unsuitable for an all-ages show (start that reason with UNSUITABLE)." if redraw else
                                  " The first frame is a colourised crop of the source manga panel (evidence.source_panel); judge the "
                                  "content against that panel and the action/dialogue against the spec."))
            + "\n\nSHOT SPEC:\n" + json.dumps(spec_view(shot), ensure_ascii=False)
            + "\n\nEVIDENCE BUNDLE:\n" + json.dumps(bundle, ensure_ascii=False)
            + (f"\n\n{len(frames)} downscaled frame(s) attached in time order." if frames else "\n\nNo frames attached."))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    llm = llm or remote_or_local("judge")
    try:
        v = llm.chat(msgs, schema=VERDICT_SCHEMA, images=frames or None, max_tokens=900, temperature=0.0, thinking=False,
                     image_max_side=384, tag=f"judge:{stage}:{shot['shot_id']}")
    except BudgetExceeded:
        llm = LLM("local", role="judge")
        v = llm.chat(msgs, schema=VERDICT_SCHEMA, images=frames or None, max_tokens=900, temperature=0.0, image_max_side=384,
                     tag=f"judge:{stage}:{shot['shot_id']}")
    v["judge_endpoint"] = llm.endpoint
    v["stage"] = stage
    _log_disagreements(shot, evidence, v)
    return v


def _log_disagreements(shot: dict, ev: dict, verdict: dict) -> None:
    """Record cases where a single metric points the other way (for later calibration review)."""
    flags = []
    faces = [f for fr in (ev.get("identity") or {}).get("frames", []) for f in fr.get("faces", [])]
    expected = (ev.get("identity") or {}).get("expected", [])
    if faces and expected:
        ok = any(f.get("same") and f.get("best") in expected for f in faces)
        if verdict["verdict"] == "pass" and not ok:
            flags.append("judge=pass but no detected face matches an expected character (CCIP)")
        if verdict["verdict"] == "fail" and ok and any("identity" in str(r).lower() or "character" in str(r).lower() for r in verdict.get("failure_reasons", [])):
            flags.append("judge=fail(identity) but CCIP matches an expected character")
    art = ((ev.get("visual") or {}).get("artifacts") or {})
    bad_art = [k for k, v in art.items() if v is True]
    if verdict["verdict"] == "pass" and bad_art:
        flags.append(f"judge=pass but the VLM checklist flags {bad_art}")
    dover = (((ev.get("metrics") or {}).get("dover")) or {})
    if isinstance(dover, dict) and dover.get("overall") is not None:
        if verdict["verdict"] == "pass" and dover["overall"] < 0.25:
            flags.append(f"judge=pass but DOVER overall is low ({dover['overall']})")
    for f in flags:
        append_jsonl(P("runs") / "judge_disagreements.jsonl", {"shot_id": shot["shot_id"], "stage": verdict.get("stage"), "flag": f,
                                                               "verdict": verdict["verdict"], "reasons": verdict.get("failure_reasons", [])})
