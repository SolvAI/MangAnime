"""One-minute GPU probe of the non-separation stages: real peak VRAM + throughput of the speaker embedder, CLAP and
faster-whisper on small real inputs (raw mix of a train episode), each in its own process so the peaks are clean.

It also validates the GPU code paths (onnx2torch on CUDA == onnxruntime CPU, ctranslate2 CUDA libraries, CLAP on CUDA)
before the long batch is launched. Separation is measured by audio_sepcompare (per model / segment size).

    scripts/gpu_run.sh 4 $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_gpuprobe          # all stages, ~1 min
    -> runs/audio_sep_compare/gpu_probe.json   (shown by src.anime.audio_report)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import ACFG, VramMonitor, cap_torch_vram, dialogue_events, ep_dir, load_subs, log, read_wav
from src.common.config import CFG
from src.common.io import write_json

OUT = Path(CFG.paths.runs) / "audio_sep_compare" / "gpu_probe.json"
EP_SECONDS = 1488.0  # one episode (24.8 min)


def probe_spk(ep: int, n_lines: int) -> dict:
    import torch

    from src.anime.audio_embed import SR, SpeakerEmbedder, line_segments

    cap = cap_torch_vram(float(os.environ.get("AUDIO_SPK_VRAM_GB") or ACFG.spk.vram_gb), reserve_gb=0.5)
    x, sr = read_wav(ep_dir(ep) / "audio_jpn_16k.wav", mono=True)
    subs = load_subs(ep)
    segs = line_segments(subs, len(x) / sr)[:n_lines]
    wavs = [x[int(s["start"] * SR): int(s["end"] * SR)] for s in segs]
    with VramMonitor(0.2) as mon:
        emb = SpeakerEmbedder(device="cuda")
        emb.embed(wavs[:4])  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        E, _ = emb.embed(wavs)
        torch.cuda.synchronize()
        dt = time.time() - t0
    ref = SpeakerEmbedder(device="cpu", threads=4).embed(wavs[:3])[0]
    cos = float(np.min(np.sum(E[:3] * ref, axis=1)))
    lines_ep = len(line_segments(subs, len(x) / sr))
    return {"backend": emb.backend, "vram_cap_gb": cap, "lines": len(wavs), "seconds": round(dt, 2), "lines_per_s": round(len(wavs) / dt, 1),
            "est_seconds_per_episode": round(lines_ep / (len(wavs) / dt), 1), "min_cos_vs_onnxruntime_cpu": round(cos, 5), **mon.stats()}


def probe_clap(ep: int, n_cues: int) -> dict:
    import torch
    import torchaudio.functional as AF

    from src.anime.audio_bgm import CLAP_SR, Clap

    cap = cap_torch_vram(float(os.environ.get("AUDIO_BGM_VRAM_GB") or ACFG.bgm.vram_gb), reserve_gb=0.5)
    x, sr = read_wav(ep_dir(ep) / "audio_jpn.wav", start=300.0, dur=30.0 * n_cues, mono=True)
    x = AF.resample(torch.from_numpy(x), sr, CLAP_SR).numpy()
    cues = [x[i * 30 * CLAP_SR:(i + 1) * 30 * CLAP_SR] for i in range(n_cues)]
    with VramMonitor(0.2) as mon:
        clap = Clap("cuda")
        clap.audio(cues[0])  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        E = np.stack([clap.audio(c) for c in cues])
        T = clap.text(["gentle comedy background music", "tense suspenseful underscore"])
        torch.cuda.synchronize()
        dt = time.time() - t0
    return {"vram_cap_gb": cap, "cues": n_cues, "cue_seconds": 30, "seconds": round(dt, 2), "cues_per_s": round(n_cues / dt, 1),
            "est_seconds_per_1000_cues": round(1000 * dt / n_cues, 1), "emb_shape": list(E.shape), "text_shape": list(T.shape),
            "emb_norm": round(float(np.linalg.norm(E[0])), 4), **mon.stats()}


def probe_asr(ep: int, dur: float) -> dict:
    from src.anime.audio_asr import load_model, sanity, transcribe_file

    with VramMonitor(0.2) as mon:
        model = load_model("cuda")
        t0 = time.time()
        asr = transcribe_file(model, ep_dir(ep) / "audio_jpn_16k.wav", start=600.0, dur=dur)
        dt = time.time() - t0
    st = sanity(asr, load_subs(ep), 600.0, 600.0 + dur)
    return {"compute_type": getattr(model.model, "compute_type", None), "audio_seconds": dur, "seconds": round(dt, 2),
            "rtf": round(dur / dt, 1), "est_seconds_per_episode": round(EP_SECONDS / (dur / dt), 1), "source": "raw mix (probe only)",
            "line_recall": st["line_recall"], "word_in_dlg": st["word_in_dlg"], "mean_word_prob": st["mean_word_prob"], **mon.stats()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "spk", "clap", "asr"])
    ap.add_argument("--episode", type=int, default=2, help="a TRAIN episode whose raw mix is used as probe input")
    args = ap.parse_args()
    if args.stage == "all":
        res: dict = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "episode": args.episode}
        for st in ("spk", "asr", "clap"):
            r = subprocess.run([sys.executable, "-m", "src.anime.audio_gpuprobe", "--stage", st, "--episode", str(args.episode)],
                               capture_output=True, text=True, cwd=str(CFG.paths.repo))
            line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("PROBE_JSON ")), None)
            res[st] = json.loads(line[len("PROBE_JSON "):]) if line else {"error": (r.stderr or r.stdout)[-600:]}
            log(f"probe {st}: {res[st]}")
        write_json(OUT, res)
        log(f"wrote {OUT}")
        return
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    fn = {"spk": lambda: probe_spk(args.episode, 200), "clap": lambda: probe_clap(args.episode, 8), "asr": lambda: probe_asr(args.episode, 120.0)}[args.stage]
    print("PROBE_JSON " + json.dumps(fn()), flush=True)


if __name__ == "__main__":
    main()
