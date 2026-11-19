"""Speaker embeddings for every subtitle dialogue line, from the separated vocal stem.

Model: WeSpeaker ResNet293-LM (VoxCeleb2, 256-d, not gated) from its official ONNX export
(models/spk/wespeaker-voxceleb-resnet293-LM/voxceleb_resnet293_LM.onnx). On CUDA the ONNX graph is converted to a
torch module with onnx2torch (identical outputs, verified to 3e-7) because the onnxruntime-gpu wheel needs CUDA 13
while the torch wheel ships CUDA 12; on CPU onnxruntime is used directly.
Front end = WeSpeaker's: 80-dim Kaldi fbank (25 ms / 10 ms, hamming, no dither), energy-based frame selection, CMN.

Output per episode: data/processed/anime/spk/epNN.npz with
    idx        [N] int    index into data/processed/anime/subs/epNN.json (only `kind == "dialogue"` lines >= min_dur)
    emb        [N, 256]   L2-normalised embedding
    dur        [N]        subtitle duration (s)
    rms        [N]        RMS of the embedded vocal-stem segment (linear, full scale = 1.0)
    start,end  [N]        the (padded) segment that was actually embedded (s)
    speech_dur [N]        seconds of frames kept by the energy VAD
    overlap    [N] bool   the line overlaps another dialogue line by > 0.2 s (probably two voices)
    multi      [N] bool   the subtitle text is a two-speaker event ("- A  - B")
Lines whose vocal stem is silent (speech_dur < 0.25 s) are not embedded.

    . scripts/env.sh
    scripts/gpu_run.sh 2 $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_embed --episodes all
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import (ACFG, VramMonitor, cap_torch_vram, dialogue_events, ep_dir, ep_name, load_subs, log, parse_episodes,
                                    to_16k_mono)
from src.common.config import CFG
from src.common.io import read_json, write_json

SP = ACFG.spk
OUT_DIR = Path(SP.out_dir)
ONNX = Path(SP.model_dir) / "wespeaker-voxceleb-resnet293-LM" / "voxceleb_resnet293_LM.onnx"
MODEL_NAME = "Wespeaker/wespeaker-voxceleb-resnet293-LM"
SR = 16000
_MULTI = re.compile(r"^\s*-\s?\S.*\s-\s?\S")


class SpeakerEmbedder:
    """`embed(list_of_16k_mono_float_arrays) -> [N, 256]` L2-normalised embeddings (+ kept speech seconds)."""

    def __init__(self, onnx_path: str | Path = ONNX, device: str = "auto", threads: int = 8):
        import torch

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.backend = None
        if device == "cuda":
            try:
                import onnx
                import onnx2torch

                self.model = onnx2torch.convert(onnx.load(str(onnx_path))).eval().to("cuda")
                with torch.no_grad():  # self-test: a failure here (device mismatch, missing op) triggers the fallback
                    y = self.model(torch.zeros(1, 200, 80, device="cuda"))
                assert tuple(y.shape) == (1, 256), y.shape
                self.backend = "torch"
            except Exception as e:  # fall back to onnxruntime (CUDA if its libraries load, else CPU)
                log(f"onnx2torch on CUDA unavailable ({e!r}); using onnxruntime")
                self.model = None
        if self.backend is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.log_severity_level = 3
            prov = ["CPUExecutionProvider"]
            if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(onnx_path), so, providers=prov)
            self.inp = self.sess.get_inputs()[0].name
            self.backend = "ort:" + self.sess.get_providers()[0]

    def features(self, wav: np.ndarray) -> tuple[np.ndarray, float]:
        """fbank + energy VAD + CMN -> ([T, 80] float32, kept seconds)."""
        import torchaudio.compliance.kaldi as kaldi

        t = self.torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0) * (1 << 15)
        if t.shape[1] < int(0.03 * SR):
            return np.zeros((0, 80), np.float32), 0.0
        fb = kaldi.fbank(t, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=SR,
                         window_type="hamming", use_energy=False)
        # energy VAD on the waveform frames (10 ms): keep frames within 35 dB of the loud part and above -55 dBFS
        n = fb.shape[0]
        x = wav[: n * 160 + 240].astype(np.float64)
        p = np.concatenate([[0.0], np.cumsum(x ** 2)])
        i0 = np.arange(n) * 160
        e = 10 * np.log10(np.maximum((p[np.minimum(i0 + 400, len(x))] - p[i0]) / 400, 1e-12))
        keep = e > max(-55.0, np.percentile(e, 95) - 35.0)
        if keep.sum() >= 2:  # 50 ms hangover on both sides
            k = np.convolve(keep.astype(np.float32), np.ones(11, np.float32), mode="same") > 0
            fb = fb[self.torch.from_numpy(k)]
        else:
            return np.zeros((0, 80), np.float32), 0.0
        kept = float(fb.shape[0]) * 0.01
        fb = fb - fb.mean(dim=0, keepdim=True)
        return fb.numpy().astype(np.float32), kept

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            with self.torch.no_grad():
                return self.model(self.torch.from_numpy(feats).to(self.device)).float().cpu().numpy()
        return self.sess.run(None, {self.inp: feats})[0]

    def embed(self, wavs: list[np.ndarray], batch_frames: int = 6000) -> tuple[np.ndarray, np.ndarray]:
        """Returns (emb [N,256] L2-normalised; zero rows where there was no speech, speech_dur [N])."""
        feats = [self.features(w) for w in wavs]
        emb = np.zeros((len(wavs), 256), np.float32)
        sdur = np.array([f[1] for f in feats], np.float32)
        # bucket by length: utterances of the same frame count are batched together (no padding -> exact pooling)
        order = sorted((i for i in range(len(wavs)) if feats[i][0].shape[0] >= 20), key=lambda i: feats[i][0].shape[0])
        j = 0
        while j < len(order):
            T = feats[order[j]][0].shape[0]
            grp = [order[j]]
            j += 1
            while j < len(order) and feats[order[j]][0].shape[0] == T and (len(grp) + 1) * T <= batch_frames:
                grp.append(order[j])
                j += 1
            y = self._forward(np.stack([feats[i][0] for i in grp]))
            emb[grp] = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-9)
        return emb, sdur

    def embed_file(self, path: str | Path) -> np.ndarray:
        """Embedding of a whole audio file (any rate / channels) - for QA voice-similarity checks."""
        import soundfile as sf
        import torchaudio.functional as AF

        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = AF.resample(self.torch.from_numpy(x), sr, SR).numpy()
        return self.embed([x])[0][0]


def line_segments(subs: list[dict], total: float) -> list[dict]:
    """Dialogue lines -> padded segments to embed (see CFG.audio.spk)."""
    dl = dialogue_events(subs)
    out = []
    for n, (i, e) in enumerate(dl):
        s, en = float(e["start"]), float(e["end"])
        d = en - s
        if d < float(SP.min_dur):
            continue
        a, b = s - float(SP.pad), en + float(SP.pad)
        if b - a < float(SP.pad_to):
            ext = (float(SP.pad_to) - (b - a)) / 2
            a, b = a - ext, b + ext
        if b - a > float(SP.max_dur):
            c = (a + b) / 2
            a, b = c - float(SP.max_dur) / 2, c + float(SP.max_dur) / 2
        a, b = max(0.0, a), min(total, b)
        ov = any(min(en, o["end"]) - max(s, o["start"]) > 0.2 for m, (_, o) in enumerate(dl[max(0, n - 6): n + 7], max(0, n - 6)) if m != n)
        out.append({"idx": i, "start": a, "end": b, "dur": d, "overlap": ov, "multi": bool(_MULTI.search(e.get("text", "")))})
    return out


def embed_episode(emb: SpeakerEmbedder, ep: int, source: str = "vocals", out_dir: Path = OUT_DIR, force: bool = False) -> dict | None:
    import soundfile as sf

    out = out_dir / f"{ep_name(ep)}.npz"
    if out.exists() and not force:
        return None
    d = ep_dir(ep)
    if source == "vocals":
        if not (d / "vocals.wav").exists():
            raise FileNotFoundError(d / "vocals.wav")
        wav16 = to_16k_mono(d / "vocals.wav", d / "vocals_16k.wav")
    else:
        wav16 = d / "audio_jpn_16k.wav"
    x, sr = sf.read(str(wav16), dtype="float32")
    assert sr == SR and x.ndim == 1, (sr, x.shape)
    subs = load_subs(ep)
    segs = line_segments(subs, len(x) / SR)
    t0 = time.time()
    wavs = [x[int(s["start"] * SR): int(s["end"] * SR)] for s in segs]
    E, sdur = emb.embed(wavs)
    rms = np.array([float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) if len(w) else 0.0 for w in wavs], np.float32)
    keep = (sdur >= 0.25) & (np.linalg.norm(E, axis=1) > 0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp,
             idx=np.array([s["idx"] for s in segs], np.int32)[keep], emb=E[keep].astype(np.float32),
             dur=np.array([s["dur"] for s in segs], np.float32)[keep], rms=rms[keep],
             start=np.array([s["start"] for s in segs], np.float32)[keep], end=np.array([s["end"] for s in segs], np.float32)[keep],
             speech_dur=sdur[keep], overlap=np.array([s["overlap"] for s in segs], bool)[keep],
             multi=np.array([s["multi"] for s in segs], bool)[keep],
             model=np.array(MODEL_NAME), source=np.array(source))
    tmp.rename(out)
    return {"lines": len(dialogue_events(subs)), "candidates": len(segs), "embedded": int(keep.sum()), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--source", default="vocals", choices=["vocals", "mix"])
    ap.add_argument("--out-dir", default=None, help="default: data/processed/anime/spk (use another dir for --source mix tests)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.source == "mix" and out_dir == OUT_DIR:
        raise SystemExit("--source mix is a test mode: pass --out-dir so the real spk/ outputs are not polluted")
    eps = [ep for ep in parse_episodes(args.episodes) if args.force or not (out_dir / f"{ep_name(ep)}.npz").exists()]
    log(f"speaker embeddings: todo={eps} source={args.source}")
    if not eps:
        return
    import torch

    torch.set_num_threads(args.threads)
    if args.device != "cpu":
        cap_torch_vram(float(os.environ.get("AUDIO_SPK_VRAM_GB") or SP.vram_gb), reserve_gb=0.5)
    failed = []
    stats = read_json(out_dir / "embed_stats.json", default={}) or {}
    with VramMonitor() as mon:
        emb = SpeakerEmbedder(device=args.device, threads=args.threads)
        log(f"embedder backend: {emb.backend}")
        for ep in eps:
            try:
                st = embed_episode(emb, ep, args.source, out_dir, args.force)
                log(f"{ep_name(ep)}: {st}")
                if st:
                    stats[ep_name(ep)] = {**st, "backend": emb.backend}
            except Exception as e:
                log(f"{ep_name(ep)}: FAILED {e!r}")
                failed.append(ep)
    stats["_last_run"] = {"backend": emb.backend, **mon.stats(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(out_dir / "embed_stats.json", stats)
    log(f"speaker embeddings: peak VRAM {mon.peak_mb} MiB ({emb.backend})")
    if failed:
        raise SystemExit(f"speaker embeddings incomplete for episodes {failed}")


if __name__ == "__main__":
    main()
