"""v3 step 5b — VOICES (D-045): English lines in voices cloned from the anime's English dub, Qwen3-TTS 1.7B.

The shot plan gives every spoken line an acting instruction ("Flustered and too loud, words tumbling out"); the engine
(src/generate/tts_qwen.py) decides how the instruction reaches the model (see docs/tts_qwen.md for what was measured).
"""
from __future__ import annotations

from pathlib import Path

from src.common.io import log, read_jsonl, write_jsonl
from src.generate.audio import SPOKEN


# Speakers without a bank borrow a same-gender minor voice (tts_qwen.voice_refs_en). Two of them landing on the same voice
# inside one scene is fixed here by hand.
CAST_OVERRIDE = {"Onemine Nono": "Yadano"}          # her brother Kazuya also borrows Yamai's voice, and they share every scene


# Two ways to speak a line (measured in RESULTS M6 and docs/tts_qwen.md):
#   "cv_xvec"  CustomVoice weights driven by the character's cloned speaker embedding; the acting instruction is given to
#              the model (the published recipe for clone + instruction; temperature 0.7). Lower voice match.
#   "icl"      plain cloning from reference clip + transcript; the instruction only picks calm or lively reference clips.
CV_ARGS = ["--set", "tts_qwen.gen.temperature=0.7", "--set", "tts_qwen.gen.subtalker_temperature=0.7"]


def synth_lines(run_dir: Path, shots: list[dict], *, mode: str = "cv_xvec", sub: str = "tts", redo: set[str] | None = None) -> dict[str, dict]:
    """Synthesise every spoken line of the shot list into runs/<run>/<sub>/. Returns {line id: result row} for the lines that succeeded."""
    from src.generate.tts_qwen import synthesize, voice_refs_en

    tdir = run_dir / sub
    tdir.mkdir(parents=True, exist_ok=True)
    items, cast = [], {}
    for s in shots:
        for k, d in enumerate(s["dialogue"]):
            if d["kind"] not in SPOKEN or not str(d.get("text") or "").strip():
                continue
            refs, texts, who = voice_refs_en(CAST_OVERRIDE.get(d["speaker"], d["speaker"]), instruct=d.get("instruct"))
            if not refs:
                continue
            cast[d["speaker"]] = who
            lid = f"{s['shot_id']}_l{k}"
            it = {"id": lid, "speaker": who, "text": d["text"], "lang": "EN", "ref_wavs": refs, "ref_texts": texts,
                  "instruct": d.get("instruct") or None, "mode": mode, "target_dur": None, "max_dur": 13.0,
                  "meta": {"shot_id": s["shot_id"], "line": k, "script_speaker": d["speaker"], "kind": d["kind"]}}
            if redo and lid in redo:
                it["seed"] = 1000
            items.append(it)
    if not items:
        log("tts: no spoken lines")
        return {}
    write_jsonl(tdir / "lines.jsonl", items)
    borrowed = {k: v for k, v in cast.items() if k != v}
    log(f"tts [{mode} -> {sub}]: {len(items)} lines, {len(cast)} speakers" + (f"; voices borrowed for {borrowed}" if borrowed else ""))
    rows = synthesize(items, tdir, extra_args=CV_ARGS if mode.startswith("cv_") else ())
    ok = {r["id"]: r for r in rows if r.get("ok")}
    log(f"tts: {len(ok)}/{len(items)} lines synthesised")
    return ok


_VOWEL = {"a": "Ah", "e": "Eh", "o": "Oh", "u": "Uh", "n": "Hm"}


def speakable(text: str) -> str:
    """A line as an English speaker would say it. Manga translations keep Japanese interjections as bare letters ("A...",
    "E?", "N?"); the voice model reads those as letter names ("Aye, aye!" for Yamai's "A..."). A lone vowel between
    punctuation becomes Ah / Eh / Oh / Uh, a lone N or "Hn" becomes Hm, a shouted two-letter stump ("TA...") is lower-cased."""
    import re

    t = str(text or "")
    m = re.fullmatch(r"([\W_]*)([A-Za-z])([\W_]*)", t)
    if m and m.group(2).lower() in _VOWEL:
        t = m.group(1) + _VOWEL[m.group(2).lower()] + m.group(3)
    t = re.sub(r"\bHn\b", "Hm", t)
    if re.fullmatch(r"[\W_]*[A-Z]{2}[\W_]*", t) and "OK" not in t:
        t = re.sub(r"[A-Z]{2}", lambda x: x.group(0).capitalize(), t)
    return t


def revoice(run: str, replace: dict[str, str] | None = None) -> list[str]:
    """Rewrite the spoken lines that `speakable` changes (plus explicit {line id: new text}) in a run's shot list and voice
    them again in both takes. Only those lines are touched: new wav under the same line id, new line duration, and the shot
    grows if the new take needs more time. For a run whose clips are not rendered yet."""
    import shutil

    from src.common.config import P
    from src.common.io import read_json, write_json
    from src.generate.audio import GAP, LEAD, TAIL

    rd = P("runs") / run
    sl = read_json(rd / "shotlist.json")
    changed: dict[str, dict] = {}
    for s in sl["shots"]:
        for k, d in enumerate(s["dialogue"]):
            lid = f"{s['shot_id']}_l{k}"
            if d["kind"] not in SPOKEN or f"l{k}" not in (s.get("tts") or {}):
                continue
            new = (replace or {}).get(lid) or speakable(d["text"])
            if new != d["text"]:
                changed[lid] = {"shot": s, "k": k, "old": d["text"], "new": new}
                d["text"] = new
    if not changed:
        log(f"revoice {run}: nothing to change")
        return []
    part = [c["shot"] for c in {c["shot"]["shot_id"]: c for c in changed.values()}.values()]
    for mode, sub, key in (("cv_xvec", "tts", "tts"), ("icl", "tts_clone", "tts_clone")):
        res = synth_lines(rd, part, mode=mode, sub=sub + "_fix")
        for lid, c in changed.items():
            r = res.get(lid)
            if not r:
                log(f"revoice {run}: {lid} failed in {sub}; the old take stays")
                continue
            shutil.copyfile(r["path"], rd / sub / f"{lid}.wav")
            c["shot"].setdefault(key, {})[f"l{c['k']}"] = r["dur"]
    for s in part:
        need = 0.0
        for key in ("tts", "tts_clone"):
            durs = list((s.get(key) or {}).values())
            if durs:
                need = max(need, LEAD + sum(durs) + GAP * (len(durs) - 1) + TAIL)
        if need > float(s["duration"]):
            s["duration"] = round(min(14.0, need), 3)
    write_json(rd / "shotlist.json", sl)
    write_json(rd / "revoice.json", {lid: {"old": c["old"], "new": c["new"]} for lid, c in changed.items()})
    log(f"revoice {run}: {len(changed)} lines rewritten and voiced again: " + "; ".join(f"{lid} {c['old']!r} -> {c['new']!r}" for lid, c in changed.items()))
    return list(changed)


def metrics(run: str, device: str = "cuda") -> dict:
    """Voice stage numbers of a run (speaker similarity to the character's reference clips, word error rate of an ASR
    transcript against the script) for both takes, from the lines the shot list actually uses: <run>/<take>/voice_metrics.json.
    Runs src.eval.voice_metrics in the TTS environment (GPU through gpu_run.sh)."""
    import subprocess

    from src.common.config import CFG, P
    from src.common.io import read_json
    from src.generate.tts_qwen import voice_refs_en

    rd = P("runs") / run
    shots = read_json(rd / "shotlist.json")["shots"]
    out = {}
    for sub in ("tts", "tts_clone"):
        rows = []
        for s in shots:
            for k, d in enumerate(s["dialogue"]):
                wav = rd / sub / f"{s['shot_id']}_l{k}.wav"
                if d["kind"] not in SPOKEN or f"l{k}" not in (s.get(sub) or {}) or not wav.exists():
                    continue
                refs, _, who = voice_refs_en(CAST_OVERRIDE.get(d["speaker"], d["speaker"]), instruct=d.get("instruct"))
                rows.append({"id": f"{s['shot_id']}_l{k}", "speaker": who, "text": d["text"], "lang": "EN", "path": str(wav), "ok": True,
                             "dur": s[sub][f"l{k}"], "ref_wavs": refs, "target_dur": None, "max_dur": 13.0, "instruct": d.get("instruct"),
                             "mode": "cv_xvec" if sub == "tts" else "icl"})
        write_jsonl(rd / sub / "tts_results_used.jsonl", rows)
        cmd = [str(P("work") / "venvs" / "tts" / "bin" / "python"), "-m", "src.eval.voice_metrics", "--results", str(rd / sub / "tts_results_used.jsonl"),
               "--out", str(rd / sub / "voice_metrics.json"), "--device", device]
        if device.startswith("cuda"):
            cmd = [f"{CFG.paths.repo}/scripts/gpu_run.sh", "6"] + cmd
        r = subprocess.run(cmd, cwd=CFG.paths.repo, capture_output=True, text=True, timeout=5400, stdin=subprocess.DEVNULL)
        agg = (read_json(rd / sub / "voice_metrics.json") or {}).get("aggregate") or {}
        if r.returncode != 0 or not agg:
            log(f"voice metrics {run} {sub}: failed: {r.stderr.strip()[-200:]}")
            continue
        out[sub] = {"lines": agg.get("n"), "similarity": (agg.get("sim_ref") or {}).get("mean"), "wer_micro": agg.get("wer_micro")}
        log(f"voice metrics {run} [{sub}]: {agg.get('n')} lines, speaker similarity {out[sub]['similarity']}, WER {agg.get('wer_micro')}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["revoice", "metrics"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--replace", default="", help='JSON {"<line id>": "new text"}')
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "metrics":
        metrics(a.run, a.device)
    else:
        revoice(a.run, __import__("json").loads(a.replace) if a.replace else None)
