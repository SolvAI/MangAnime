"""Media for the public repository: the same story moments across the sprints, next to the manga panel and the real shot.

  python scripts/build_public_media.py eval  --out <public repo>/eval/media     # one grid per sample: mp4 + animated webp + poster
  python scripts/build_public_media.py reels --out <public repo>/docs/media     # silent excerpts of the latest finished cut

Everything written here is SILENT. The research build's voices are cloned from the official English dub and its music is
taken from the show's soundtrack; neither is published. Excerpts of the real episode are short, silent and downscaled, and
serve as the reference of the evaluation only.

A sample is one shot of the real held-out episode (22) for which every sprint has a generated shot that adapts the same
moment (runs/<run>/eval/real_pairs.json, built by src/eval/real_pairs.py: dialogue and visual alignment, confidence >= 0.8).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json, read_jsonl, write_json  # noqa: E402

RUNS = [("ep22", "Sprint 1  colourised panel + Wan"), ("ep22_v2", "Sprint 2a  Illustrious + Wan LoRA"), ("ep22_v2b", "Sprint 2b  Illustrious + base Wan"),
        ("ep22_v3", "Sprint 3  script-first"), ("ep22_v3b", "Sprint 4  reference-drawn frames"), ("ep22_v3e", "Sprint 5  style refs, 16 fps, edit")]
SAMPLES = [9, 16, 18, 38, 84, 87, 193, 218, 259, 301, 336, 329]        # real shot numbers, picked by eye from the 88 that qualify
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TW, TH = 480, 270
BG = "0x16181d"


def ff(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])


def dur(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def best_pairs() -> dict[str, dict[int, dict]]:
    out = {}
    for run, _ in RUNS:
        d = read_json(P("runs") / run / "eval" / "real_pairs.json")
        best: dict[int, dict] = {}
        for x in d.get("pairs", d):
            if x["real_shot"] not in best or x["confidence"] > best[x["real_shot"]]["confidence"]:
                best[x["real_shot"]] = x
        out[run] = best
    return out


def label(text: str) -> str:
    t = text.replace(":", r"\:").replace("'", "")
    return f"drawtext=fontfile={FONT}:text='{t}':fontsize=15:fontcolor=white:x=8:y=8:box=1:boxcolor=black@0.62:boxborderw=6"


def eval_grids(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pairs = best_pairs()
    panels = {(p["chapter"], p["order"]): p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    shots = {run: {s["shot_id"]: s for s in read_json(P("runs") / run / "shotlist.json")["shots"]} for run, _ in RUNS}
    meta = []
    for n, k in enumerate(SAMPLES, 1):
        if any(k not in pairs[run] for run, _ in RUNS):
            print(f"sample {k}: not in every run, skipped")
            continue
        ids = {run: pairs[run][k]["shot_id"] for run, _ in RUNS}
        clips = [P("runs") / run / "shots" / ids[run] / "clip.mp4" for run, _ in RUNS]
        real = P("runs") / "ep22_v3e" / "shots" / ids["ep22_v3e"] / "real.mp4"
        s1 = shots["ep22"][ids["ep22"]]
        panel = panels.get((int(s1["chapter"]), (s1.get("source_panels") or [None])[0]))
        if not all(c.exists() for c in clips) or not real.exists():
            print(f"sample {k}: a clip is missing, skipped")
            continue
        T = round(min(6.0, max([dur(c) for c in clips] + [dur(real)])), 2)
        args, filt = [], []
        tiles = [(None, "Manga panel")] + list(zip(clips, [lab for _, lab in RUNS])) + [(real, "Real anime, episode 22 (reference)")]
        for i, (src, lab) in enumerate(tiles):
            if src is None:
                if panel and Path(panel["image"]).exists():
                    args += ["-loop", "1", "-t", str(T), "-i", str(panel["image"])]
                    filt.append(f"[{i}:v]fps=16,scale={TW}:{TH}:force_original_aspect_ratio=decrease,pad={TW}:{TH}:(ow-iw)/2:(oh-ih)/2:color={BG},setsar=1,{label(lab)}[v{i}]")
                else:
                    args += ["-f", "lavfi", "-t", str(T), "-i", f"color=c={BG}:s={TW}x{TH}:r=16"]
                    filt.append(f"[{i}:v]{label('Manga panel (none: shot invented by the script)')}[v{i}]")
            else:
                args += ["-i", str(src)]
                filt.append(f"[{i}:v]fps=16,scale={TW}:{TH}:force_original_aspect_ratio=increase,crop={TW}:{TH},setsar=1,tpad=stop_mode=clone:stop_duration=8,"
                            f"trim=duration={T},setpts=PTS-STARTPTS,{label(lab)}[v{i}]")
        lay = "|".join(f"{c * TW}_{r * TH}" for r in range(2) for c in range(4))
        filt.append("".join(f"[v{i}]" for i in range(8)) + f"xstack=inputs=8:layout={lay}:fill={BG}[grid]")
        name = f"sample_{n:02d}"
        ff(*args, "-filter_complex", ";".join(filt), "-map", "[grid]", "-t", str(T), "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / f"{name}.mp4"))
        ff("-i", str(out / f"{name}.mp4"), "-vf", "fps=10,scale=1440:-2:flags=lanczos", "-c:v", "libwebp_anim", "-loop", "0", "-q:v", "52", "-compression_level", "5",
           "-an", str(out / f"{name}.webp"))
        ff("-ss", f"{T * 0.45:.2f}", "-i", str(out / f"{name}.mp4"), "-frames:v", "1", "-q:v", "4", str(out / f"{name}.jpg"))
        line = next((d["text"] for d in shots["ep22_v3e"][ids["ep22_v3e"]].get("dialogue", []) if d.get("kind") in ("speech", "thought", "narration")), "")
        meta.append({"name": name, "real_shot": k, "chapter": int(s1["chapter"]), "seconds": T, "shots": ids, "line": line,
                     "shows": shots["ep22_v3e"][ids["ep22_v3e"]].get("image_sentence", ""),
                     "confidence": round(min(pairs[run][k]["confidence"] for run, _ in RUNS), 2),
                     "mb": {e: round((out / f"{name}.{e}").stat().st_size / 1e6, 2) for e in ("mp4", "webp")}})
        print(f"{name}: real shot {k}, {T} s, mp4 {meta[-1]['mb']['mp4']} MB, webp {meta[-1]['mb']['webp']} MB")
    write_json(out / "samples.json", meta)


def reels(out: Path, run: str, spans: list[tuple[str, str, str]]) -> None:
    """Silent excerpts of the finished cut, subtitles burned in: (name, first shot id, last shot id)."""
    out.mkdir(parents=True, exist_ok=True)
    rd = P("runs") / run
    shots = read_json(rd / "shotlist.json")["shots"]
    tl = read_json(rd / "assembly.json")["timeline"]
    idx = {s["shot_id"]: i for i, s in enumerate(shots)}
    for name, a, b in spans:
        t0, t1 = tl[idx[a]][0], tl[idx[b]][1]
        ff("-ss", f"{t0:.3f}", "-i", str(rd / "episode.mp4"), "-t", f"{t1 - t0:.3f}", "-an", "-vf", "scale=960:540:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
           "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / f"{name}.mp4"))
        ff("-ss", f"{(t1 - t0) * 0.4:.2f}", "-i", str(out / f"{name}.mp4"), "-frames:v", "1", "-q:v", "4", str(out / f"{name}.jpg"))
        print(f"{name}: {a}..{b}, {t1 - t0:.1f} s, {(out / (name + '.mp4')).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "reels"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="ep22_v3e")
    ap.add_argument("--spans", default="", help="reels: name:first_shot:last_shot,...")
    a = ap.parse_args()
    if a.cmd == "eval":
        eval_grids(Path(a.out))
    else:
        reels(Path(a.out), a.run, [tuple(x.split(":")) for x in a.spans.split(",") if x])
