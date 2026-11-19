"""QA evidence collection (brief §8.1). Everything here is LOCAL and ends up as text for the judge.

Two phases so the GPU hosts one large model at a time:
  metrics_evidence(...)  - technical check, temporal stability, DOVER, identity (CCIP), panel fidelity   (small models)
  visual_evidence(...)   - Qwen 3.8 captions of sampled frames + artifact checklist                       (needs `qwen`)
Audio evidence comes from src.eval.voice_metrics (run once per TTS batch) and is attached per shot.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common.config import P
from src.common.io import log, read_json, read_jsonl, write_json
from src.generate.audio import OUT_FPS, frames_of

VISUAL_SCHEMA = {
    "type": "object",
    "required": ["description", "characters", "artifacts"],
    "properties": {
        "description": {"type": "string"},
        "characters": {"type": "array"},
        "shot_size": {"type": "string"},
        "setting": {"type": "string"},
        "motion": {"type": "string"},
        "artifacts": {"type": "object"},
        "looks_like_anime_frame": {"type": "boolean"},
        "leftover_manga_elements": {"type": "boolean"},
    },
}

VISUAL_SYSTEM = (
    "You are a meticulous quality inspector for AI-generated anime shots of \"Komi Can't Communicate\". "
    "You receive frames of ONE generated shot in time order. Describe only what you see; do not be generous.\n"
    "Roster for naming characters by look:\n{roster}\n"
    "Return ONE JSON object with keys: description (2 sentences: who/what is visible and what changes across the frames), "
    "characters (names from the roster, or 'unknown girl'/'unknown boy'), shot_size, setting, "
    "motion ('none', 'subtle', 'clear' or 'chaotic' + a few words), "
    "artifacts: an object with boolean fields extra_limbs, melted_or_distorted_face, garbled_text, wrong_or_changing_outfit, "
    "identity_drift_between_frames, flicker_or_morphing, duplicate_character, and a short string field notes, "
    "looks_like_anime_frame (false if it looks like a manga page, sketch, 3D render or photo), "
    "leftover_manga_elements (true if speech bubbles, panel borders, screentone or sound-effect lettering remain)."
)


def sample_frames(video: Path, out_dir: Path, n: int = 3, tag: str = "f") -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sorted({int(round(x)) for x in np.linspace(0, max(0, total - 1), n)})
    out = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            p = out_dir / f"{tag}{k}.jpg"
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out.append(p)
    cap.release()
    return out


def lineart(img_bgr: np.ndarray, size: int = 384) -> np.ndarray:
    """Cheap line-art proxy (grayscale + difference-of-Gaussians edges) so colour frames and manga panels are comparable."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s = size / max(g.shape)
    g = cv2.resize(g, (max(8, int(g.shape[1] * s)), max(8, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32) / 255.0
    dog = cv2.GaussianBlur(g, (0, 0), 0.8) - cv2.GaussianBlur(g, (0, 0), 1.6)
    edges = 1.0 - np.clip(np.abs(dog) * 12.0, 0, 1)
    return (np.stack([edges] * 3, -1) * 255).astype(np.uint8)


def metrics_evidence(run_dir: Path, shot: dict, *, clip: Path | None = None, device: str | None = None, with_dover: bool = True) -> dict:
    """Small-model evidence for the current clip (or only the keyframe if clip is None)."""
    from src.anime.characters import identify

    sdir = run_dir / "shots" / shot["shot_id"]
    ev: dict = {"shot_id": shot["shot_id"]}
    expected = [c for c in shot.get("characters", []) if c not in ("Narrator", "Classmates")]
    frames: list[Path] = []
    if clip is not None and clip.exists():
        from src.eval import video_metrics as vm

        want = frames_of(shot["duration"]) / OUT_FPS
        m: dict = {}
        # each metric group on its own: a missing optional model (e.g. DOVER) must not hide the others
        for group in (["technical"], ["flicker", "motion", "warp"], ["dover"] if with_dover else []):
            if not group:
                continue
            try:
                try:
                    r = vm.analyze_clip(clip, metrics=group, expected_dur=want, device=device, max_pairs=12, flow_height=288)
                except TypeError:
                    r = vm.analyze_clip(clip, metrics=group, expected_dur=want)
                m.update({k: v for k, v in r.items() if k in group or k not in m})
            except Exception as e:
                m[group[0] + "_error"] = str(e)[:160]
        ev["metrics"] = _compact(m)
        frames = sample_frames(clip, sdir / "qa_frames", 3)
    elif (sdir / "kf.png").exists():
        frames = [sdir / "kf.png"]
    # identity: every sampled frame against the character bank
    ident = []
    for f in frames:
        try:
            r = identify(f, expected)
            ident.append({"frame": f.name, "faces": [{k: x[k] for k in ("best", "diff", "same", "expected_diffs", "face_frac")} for x in r["faces"]],
                          "threshold": r["threshold"]})
        except Exception as e:
            ident.append({"frame": f.name, "error": str(e)[:120]})
    ev["identity"] = {"expected": expected, "frames": ident}
    # fidelity: keyframe vs source panel (line-art embedding similarity)
    try:
        panel = sdir / "panel.png"
        if panel.exists() and (sdir / "kf.png").exists():
            from src.align.visual import embed_images

            a = lineart(cv2.imread(str(panel)))
            b = lineart(cv2.imread(str(sdir / "kf.png")))
            e = embed_images([a, b])
            ev["fidelity"] = {"panel_vs_keyframe_lineart_cos": round(float(e[0] @ e[1]), 4)}
    except Exception as e:
        ev["fidelity"] = {"error": str(e)[:120]}
    ev["frames"] = [str(f) for f in frames]
    try:        # Illustrious backend: what the keyframe prompt intended (panel-aware reading), for the judge
        metas = sorted(sdir.glob("kf_a*.json"), key=lambda p: int(p.stem.split("_a")[-1]))
        km = read_json(metas[-1]) if metas else None
        if km and km.get("backend") == "illustrious":
            t = km.get("tags") or {}
            named = km.get("named") or []
            ev["intended_frame"] = {"drawn_by_name": named, "n_people_total": t.get("n_people_total"), "situation": t.get("situation"),
                                    "people": [{"name": (p.get("name") if p.get("name") in named else "unknown"), "position": p.get("position"),
                                                "look": p.get("look"), "pose": p.get("pose"), "expression": p.get("expression")}
                                               for p in (t.get("people") or [])[:4] if isinstance(p, dict)]}
    except Exception:
        pass
    try:        # what the source manga panel shows (VLM caption from the manga parse) = the reference for spec match
        pid = (shot.get("source_panels") or [None])[0]
        if pid is not None:
            p = _panel(shot["chapter"], pid)
            if p:
                ev["source_panel"] = {"caption": p.get("caption"), "characters": p.get("characters"), "shot_size": p.get("shot_size"),
                                      "location": p.get("location"), "layout": (read_json(sdir / f"kf_a{0}.json") or {}).get("layout")}
    except Exception:
        pass
    return ev


_PANELS: dict = {}


def _panel(chapter: int, order: int) -> dict | None:
    if not _PANELS:
        for r in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
            _PANELS[(r["chapter"], r["order"])] = r
    return _PANELS.get((chapter, order))


def _compact(m: dict, depth: int = 0):
    """Round floats and drop bulky arrays so the bundle stays small as text."""
    if isinstance(m, dict):
        return {k: _compact(v, depth + 1) for k, v in m.items() if not (isinstance(v, (list, tuple, np.ndarray)) and len(v) > 12)}
    if isinstance(m, (float, np.floating)):
        return round(float(m), 4)
    if isinstance(m, (np.integer,)):
        return int(m)
    if isinstance(m, (list, tuple)):
        return [_compact(v, depth + 1) for v in m]
    if isinstance(m, np.ndarray):
        return [round(float(x), 4) for x in m.tolist()]
    return m


def visual_evidence(llm, frames: list[str | Path], shot: dict) -> dict:
    """Qwen captions + artifact checklist for the sampled frames of one shot."""
    from src.anime.caption import roster_block

    n = len(frames)
    user = (f"{n} frame(s) of one generated shot, in time order. Inspect them carefully and fill in the JSON."
            if n > 1 else "One generated keyframe. Inspect it carefully and fill in the JSON (motion = 'n/a').")
    try:
        res = llm.chat([{"role": "system", "content": VISUAL_SYSTEM.format(roster=roster_block())}, {"role": "user", "content": user}],
                       schema=VISUAL_SCHEMA, images=list(frames), max_tokens=500, temperature=0.1, image_max_side=640, tag="qa_visual")
    except Exception as e:
        res = {"error": str(e)[:200]}
    return res


def audio_evidence(run_dir: Path, shot: dict) -> dict:
    """Per-line audio evidence from the voice metrics report (CER vs script, speaker similarity, duration fit)."""
    rep = {r["id"]: r for r in (read_json(run_dir / "tts" / "voice_metrics.json") or {}).get("items", [])}
    res = {r["id"]: r for r in read_jsonl(run_dir / "tts" / "tts_results.jsonl")}
    lines = []
    for k, d in enumerate(shot.get("dialogue", [])):
        lid = f"{shot['shot_id']}_l{k}"
        if d["kind"] not in ("speech", "thought", "narration"):
            continue
        r, t = rep.get(lid, {}), res.get(lid, {})
        lines.append({"line": k, "speaker": d["speaker"], "script_ja": d.get("text_ja", ""), "tts_ok": bool(t.get("ok")),
                      "dur": t.get("dur"), "fit_ok": t.get("fit_ok", True), "asr": r.get("asr_text"), "cer": r.get("cer"),
                      "speaker_sim": r.get("sim_bank") if r.get("sim_bank") is not None else r.get("sim_ref"), "flags": r.get("flags")})
    total = sum((l["dur"] or 0) for l in lines)
    return {"lines": lines, "spoken_sec": round(total, 2), "shot_sec": shot["duration"],
            "fits": bool(total <= shot["duration"] + 0.05), "missing": [l["line"] for l in lines if not l["tts_ok"]]}


def hard_fail(ev: dict) -> list[str]:
    """Hard technical failures fail automatically, without a judge call (brief §8.3)."""
    reasons = []
    tech = (ev.get("metrics") or {}).get("technical") or {}
    if tech and tech.get("ok") is False:
        reasons += [f"technical: {r}" for r in tech.get("reasons", ["failed"])]
    if (ev.get("metrics") or {}).get("technical_error"):      # the clip could not even be opened
        reasons.append("technical: clip unreadable (" + str(ev["metrics"]["technical_error"])[:80] + ")")
    return reasons
