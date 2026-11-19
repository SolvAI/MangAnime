"""Anime ingest: per episode -> 480p CFR proxy, Japanese audio (44.1k stereo + 16k mono), subtitle JSON.

Outputs (data/interim/anime/epNN/): proxy.mp4, audio_jpn.wav, audio_jpn_16k.wav
         (data/processed/anime/subs/epNN.json): [{start,end,text,kind,style}]
Idempotent: skips existing outputs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pysubs2

from src.common.config import CFG, P
from src.common.io import ffmpeg, ffprobe_json, log, media_duration, read_json, write_json


def episode_files() -> dict[int, Path]:
    out = {}
    for f in sorted((P("raw") / "anime").glob("ep*.mkv")):
        out[int(f.name[2:4])] = f
    return out


def ep_dir(ep: int) -> Path:
    d = P("interim") / "anime" / f"ep{ep:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_proxy(ep: int, src: Path) -> Path:
    out = ep_dir(ep) / "proxy.mp4"
    if out.exists() and abs(media_duration(out) - media_duration(src)) < 2.0:
        return out
    h = int(CFG.anime.work_height)
    w = int(round(h * 16 / 9 / 2) * 2)
    vf = f"scale={w}:{h}:flags=lanczos,fps=24000/1001,format=yuv420p"
    tmp = out.with_suffix(".tmp.mp4")
    try:
        ffmpeg("-i", str(src), "-map", "0:v:0", "-vf", vf, "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
               "-cq", "21", "-g", "12", "-an", str(tmp))
    except Exception as e:  # NVENC unavailable (GPU busy / session limit) -> CPU
        log(f"nvenc failed ({str(e)[:80]}), falling back to libx264")
        ffmpeg("-i", str(src), "-map", "0:v:0", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-g", "12", "-an", str(tmp))
    tmp.rename(out)
    return out


def extract_audio(ep: int, src: Path) -> tuple[Path, Path]:
    d = ep_dir(ep)
    a44, a16 = d / "audio_jpn.wav", d / "audio_jpn_16k.wav"
    lang = CFG.anime.audio_lang
    if not a44.exists():
        ffmpeg("-i", str(src), "-map", f"0:a:m:language:{lang}", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(a44))
    if not a16.exists():
        ffmpeg("-i", str(a44), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(a16))
    return a44, a16


# ----------------------------------------------------------------------------- subtitles
_TAG = re.compile(r"\{[^}]*\}")
_DRAW = re.compile(r"\\p[1-9]")  # ASS vector drawings (masks), not text


def _clean(text: str) -> str:
    t = _TAG.sub("", text).replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    t = re.sub(r"</?[a-zA-Z][^>]*>", "", t)
    return re.sub(r"\s+", " ", t).strip()


_SEG_TITLE = re.compile(r"^\s*(communication|komyu|mission)\s*#?\d+", re.I)
_DLG_STYLES = ("default", "main", "alt", "ital", "flashback", "dialog", "overlap", "top", "narrat", "thought", "internal")


def _kind_for(style: str, track_title: str, text: str, is_srt: bool, is_dialogue_track: bool) -> str:
    s = re.sub(r"^\d+\$", "", style).lower()   # merged fansub scripts prefix styles as "4$Default"
    if _SEG_TITLE.match(text):
        return "title"                            # anime segment card ("Communication 05 ...", "Komyu 69")
    if any(k in s for k in ("op_", "ed_", "ed2", "romaji", "kanji", "song", "karaoke", "insert", "op-", "ed-")) or s in ("op", "ed"):
        return "song"
    if "carnet" in s or "handw" in s or "notebook" in s:
        return "notebook"
    if is_srt:
        return "dialogue"
    if any(s.startswith(k) or k in s for k in _DLG_STYLES) and "sign" not in s and "title" not in s:
        return "dialogue" if (is_dialogue_track or "sign" not in track_title.lower()) else "sign"
    return "sign"


def parse_subs(ep: int, src: Path) -> list[dict]:
    """Merge all English subtitle tracks into one event list with a `kind`."""
    out_path = P("processed") / "anime" / "subs" / f"ep{ep:02d}.json"
    if out_path.exists():
        return read_json(out_path)
    probe = ffprobe_json(src, "stream=index,codec_type,codec_name:stream_tags=language,title")
    subs = [s for s in probe["streams"] if s["codec_type"] == "subtitle"]
    d = ep_dir(ep) / "subs"
    d.mkdir(exist_ok=True)
    tracks = []
    for s in subs:
        ext = "ass" if s["codec_name"] == "ass" else "srt"
        f = d / f"track{s['index']}.{ext}"
        if not f.exists():
            ffmpeg("-i", str(src), "-map", f"0:{s['index']}", str(f), check=False)
        if f.exists() and f.stat().st_size > 0:
            tracks.append({"index": s["index"], "title": (s.get("tags") or {}).get("title") or "", "path": f, "srt": ext == "srt"})
    # pick ONE dialogue source: prefer an ASS track titled "Dialogue…", else the SRT, else ASS Default-style lines
    dlg_track = next((t for t in tracks if t["title"].lower().startswith("dialogue")), None)
    if dlg_track is None:
        dlg_track = next((t for t in tracks if t["srt"]), None)
    events: list[dict] = []
    seen: set[tuple] = set()
    for t in tracks:
        try:
            sf = pysubs2.load(str(t["path"]))
        except Exception as e:
            log(f"ep{ep:02d}: cannot parse {t['path'].name}: {e}")
            continue
        for ev in sf:
            if ev.is_comment or _DRAW.search(ev.text):
                continue
            text = _clean(ev.text)
            if not text:
                continue
            kind = _kind_for(ev.style, t["title"], text, t["srt"], t is dlg_track)
            if kind == "dialogue" and dlg_track is not None and t is not dlg_track:
                continue  # avoid duplicated dialogue from alternate tracks
            if kind == "dialogue" and re.fullmatch(r"[♪\s~]+.*", text) :
                kind = "song"
            key = (kind, text, round(ev.start / 1000, 0)) if kind == "dialogue" else (kind, text)
            if key in seen:
                # typeset signs are split into many per-frame events: extend the existing one instead
                if kind != "dialogue":
                    for e0 in reversed(events):
                        if e0["kind"] == kind and e0["text"] == text:
                            if ev.start / 1000 - e0["end"] < 1.0:
                                e0["end"] = max(e0["end"], ev.end / 1000)
                            else:
                                events.append({"start": ev.start / 1000, "end": ev.end / 1000, "text": text, "kind": kind, "style": ev.style})
                            break
                continue
            seen.add(key)
            events.append({"start": ev.start / 1000, "end": ev.end / 1000, "text": text, "kind": kind, "style": ev.style})
    events.sort(key=lambda e: (e["start"], e["end"]))
    write_json(out_path, events)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--no-proxy", action="store_true")
    args = ap.parse_args()
    files = episode_files()
    eps = sorted(files) if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    for ep in eps:
        src = files[ep]
        ev = parse_subs(ep, src)
        kinds = {}
        for e in ev:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        extract_audio(ep, src)
        if not args.no_proxy:
            make_proxy(ep, src)
        log(f"ep{ep:02d} ingest ok: subs={kinds}")


if __name__ == "__main__":
    main()
