"""Audio (brief §7.4): TTS per line fitted to the shot, BGM retrieval, simple SFX, final mix.

tts_stage : shot dialogue -> runs/<run>/tts/<line_id>.wav (IndexTTS-2.5 via venvs/tts) and re-times shot durations
mix_stage : dialogue + ducked BGM (+ SFX) -> runs/<run>/audio/mix.wav (44.1 kHz stereo), aligned to the frame timeline
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json, write_jsonl

SR = 44100
OUT_FPS = 24000 / 1001
SPOKEN = ("speech", "thought", "narration")
LEAD, GAP, TAIL = 0.20, 0.18, 0.30     # seconds of air before / between / after lines inside a shot


def voice_refs(speaker: str) -> tuple[list[str], str]:
    """Reference wavs for a speaker, with fallbacks for unknown / crowd voices. Returns (refs, resolved_speaker)."""
    idx = (read_json(P("processed") / "voices" / "voices.json") or {}).get("speakers", {})
    if speaker in idx and idx[speaker].get("default_refs"):
        return list(idx[speaker]["default_refs"]), speaker
    # alias match ("Tadano-kun" -> Tadano)
    for c in CFG.characters.roster:
        if speaker == c["name"] or speaker in c.get("aliases", []) or speaker == c.get("full"):
            if c["name"] in idx and idx[c["name"]].get("default_refs"):
                return list(idx[c["name"]]["default_refs"]), c["name"]
    # deterministic fallback: any minor voice (never a main character's voice for a stranger if avoidable)
    pool = [n for n in sorted(idx) if n not in ("Komi", "Tadano", "Najimi", "Narrator") and idx[n].get("default_refs")] or sorted(idx)
    if not pool:
        return [], speaker
    pick = pool[sum(map(ord, speaker)) % len(pool)]
    return list(idx[pick]["default_refs"]), pick


def frames_of(duration: float) -> int:
    return max(1, int(round(duration * OUT_FPS)))


def timeline(shots: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of every shot on the frame grid (so audio stays in sync with the concatenated clips)."""
    out, f = [], 0
    for s in shots:
        n = frames_of(s["duration"])
        out.append((f / OUT_FPS, (f + n) / OUT_FPS))
        f += n
    return out


def tts_stage(run_dir: Path, shots: list[dict], *, redo: set[str] | None = None, seed_bump: int = 0) -> list[dict]:
    """Synthesize all spoken lines, then make every shot long enough for its lines. Returns the updated shots."""
    tdir = run_dir / "tts"
    tdir.mkdir(parents=True, exist_ok=True)
    items = []
    for s in shots:
        for k, d in enumerate(s["dialogue"]):
            if d["kind"] not in SPOKEN or not d.get("text_ja"):
                continue
            refs, resolved = voice_refs(d["speaker"])
            if not refs:
                continue
            lid = f"{s['shot_id']}_l{k}"
            it = {"id": lid, "speaker": resolved, "text": d["text_ja"], "lang": "JA", "ref_wavs": refs[:2],
                  "emotion": d.get("emotion", "neutral"), "target_dur": None, "max_dur": 7.0,
                  "meta": {"shot_id": s["shot_id"], "line": k, "script_speaker": d["speaker"]}}
            if redo and lid in redo:
                it["seed"] = 1000 + seed_bump
            items.append(it)
    if not items:
        log("tts: no spoken lines (or no voices available)")
        return shots
    write_jsonl(tdir / "lines.jsonl", items)
    py = P("work") / "venvs" / "tts" / "bin" / "python"
    cmd = [f"{CFG.paths.repo}/scripts/gpu_run.sh", "8", str(py), "-m", "src.generate.tts", "--in", str(tdir / "lines.jsonl"), "--out-dir", str(tdir)]
    log(f"tts: {len(items)} lines")
    r = subprocess.run(cmd, cwd=CFG.paths.repo, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        log("tts stage failed: " + (r.stderr or r.stdout)[-600:])
    res = {x["id"]: x for x in read_jsonl(tdir / "tts_results.jsonl") if x.get("ok")}
    # re-time shots: a shot must contain its lines (+ air); never shorter than what the director asked for unless silent
    for s in shots:
        durs = [res[f"{s['shot_id']}_l{k}"]["dur"] for k, d in enumerate(s["dialogue"]) if f"{s['shot_id']}_l{k}" in res]
        if durs:
            need = LEAD + sum(durs) + GAP * (len(durs) - 1) + TAIL
            s["duration"] = round(float(min(10.0, max(need, min(s["duration"], need + 1.2)))), 3)
        s["tts"] = {f"l{k}": res[f"{s['shot_id']}_l{k}"]["dur"] for k, _ in enumerate(s["dialogue"]) if f"{s['shot_id']}_l{k}" in res}
    return shots


def _read(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def _fade(x: np.ndarray, sec: float = 0.4) -> np.ndarray:
    n = min(len(x) // 2, int(sec * SR))
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


def line_schedule(shots: list[dict]) -> list[dict]:
    """Absolute timing of every dialogue line (spoken or not) -> used for the mix and for subtitles."""
    tl = timeline(shots)
    out = []
    for s, (t0, t1) in zip(shots, tl):
        t = t0 + LEAD
        silent = [d for d in s["dialogue"] if d["kind"] not in SPOKEN]
        for k, d in enumerate(s["dialogue"]):
            dur = (s.get("tts") or {}).get(f"l{k}")
            if d["kind"] in SPOKEN and dur:
                out.append({"shot_id": s["shot_id"], "line": k, "start": t, "end": min(t + dur, t1 + 0.5), "spoken": True, **d})
                t += dur + GAP
            else:  # notebook / on-screen text / unvoiced line: show for the rest of the shot
                st = max(t0 + 0.1, min(t, t1 - 0.6))
                out.append({"shot_id": s["shot_id"], "line": k, "start": st, "end": t1 - 0.05, "spoken": False, **d})
    return out


def mix_stage(run_dir: Path, shots: list[dict], *, bgm_gain_db: float = -15.0, duck_db: float = -9.0,
              tts_dir: str = "tts", out_dir: str = "audio") -> Path:
    adir = run_dir / out_dir
    adir.mkdir(parents=True, exist_ok=True)
    tl = timeline(shots)
    total = tl[-1][1] if tl else 0.0
    n = int(np.ceil(total * SR)) + SR // 10
    dlg = np.zeros(n, np.float32)
    sched = line_schedule(shots)
    for ln in sched:
        if not ln["spoken"]:
            continue
        f = run_dir / tts_dir / f"{ln['shot_id']}_l{ln['line']}.wav"
        if not f.exists():
            continue
        x = _read(f)
        i = int(ln["start"] * SR)
        x = x[: max(0, n - i)]
        dlg[i : i + len(x)] += x
    # ---- BGM: one cue per run of identical bgm_cue tags
    bgm = np.zeros(n, np.float32)
    used = []
    try:
        from src.generate.bgm import retrieve

        runs, cur = [], None
        for s, (t0, t1) in zip(shots, tl):
            cue = (s.get("bgm_cue") or "none").strip().lower()
            if cur and cur["cue"] == cue:
                cur["end"] = t1
            else:
                cur = {"cue": cue, "start": t0, "end": t1}
                runs.append(cur)
        recent: list[str] = []
        for r in runs:
            if r["cue"] in ("none", "silence", ""):
                continue
            need = r["end"] - r["start"]
            cands = retrieve(r["cue"], min_dur=min(need, 20.0), k=6)      # mood aliases (gentle_comedy, ...) are resolved by retrieve()
            cands = [c for c in cands if c["id"] not in recent[-4:]] or cands
            if not cands:
                continue
            c = cands[0]
            recent.append(c["id"])
            x = _read(c.get("abs_path") or c["path"])
            if len(x) < need * SR:  # loop with a short crossfade
                reps = int(np.ceil(need * SR / max(1, len(x) - SR // 2))) + 1
                x = np.concatenate([_fade(x.copy(), 0.25)] * reps)
            x = _fade(x[: int(need * SR)].copy(), 0.6)
            i = int(r["start"] * SR)
            x = x[: max(0, n - i)]
            bgm[i : i + len(x)] += x
            used.append({"cue": r["cue"], "start": round(r["start"], 2), "end": round(r["end"], 2), "id": c["id"], "sim": c.get("sim")})
    except Exception as e:  # BGM library not built yet -> dialogue-only mix
        log(f"mix: no BGM ({str(e)[:160]})")
    # ---- ducking: lower the music while someone speaks
    win = int(0.05 * SR)
    env = np.sqrt(np.convolve(dlg**2, np.ones(win, np.float32) / win, mode="same"))
    speak = (env > 0.01).astype(np.float32)
    k = int(0.35 * SR)
    speak = np.clip(np.convolve(speak, np.ones(k, np.float32) / k, mode="same") * 1.5, 0, 1)
    g_bgm = 10 ** (bgm_gain_db / 20) * (1 - speak * (1 - 10 ** (duck_db / 20)))
    if np.abs(dlg).max() > 0:
        dlg = dlg / max(1e-6, np.abs(dlg).max()) * 0.85
    mix = dlg + bgm * g_bgm
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        mix = mix / peak * 0.98
    sf.write(str(adir / "dialogue.wav"), dlg, SR, subtype="PCM_16")
    sf.write(str(adir / "bgm.wav"), (bgm * g_bgm).astype(np.float32), SR, subtype="PCM_16")
    sf.write(str(adir / "mix.wav"), np.stack([mix, mix], 1), SR, subtype="PCM_16")
    write_json(adir / "mix.json", {"duration": total, "bgm": used, "lines": sched})
    log(f"mix: {total:.1f}s, {sum(1 for l in sched if l['spoken'])} spoken lines, {len(used)} BGM cues")
    return adir / "mix.wav"
