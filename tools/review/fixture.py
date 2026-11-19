"""Tiny synthetic fixture for the review page (no real assets needed).

    python -m tools.review.fixture --out /home/user/komi-poc-work/tmp/review_fixture
    python -m tools.review.app --processed /home/user/komi-poc-work/tmp/review_fixture --port 8301
"""
from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path


def _tone(path: Path, freq: float, dur: float = 0.8, sr: int = 16000) -> None:
    n = int(dur * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * freq * i / sr))) for i in range(n)))


def make_fixture(out: str | Path) -> Path:
    from PIL import Image, ImageDraw

    out = Path(out)
    cdir, sdir = out / "characters", out / "anime" / "spk"
    (cdir / "ex").mkdir(parents=True, exist_ok=True)
    (sdir / "ex").mkdir(parents=True, exist_ok=True)
    names = [("Komi", 0.93), ("Tadano", 0.81), (None, 0.22), ("Najimi", 0.48), ("Yamai", None)]
    clusters = []
    for cid, (name, conf) in enumerate(names):
        ex = []
        for k in range(4):
            p = cdir / "ex" / f"c{cid}_{k}.jpg"
            im = Image.new("RGB", (160, 160), (40 + 45 * cid, 90 + 20 * k, 200 - 30 * cid))
            d = ImageDraw.Draw(im)
            d.ellipse((40, 30, 120, 130), fill=(245, 225, 205))
            d.text((8, 8), f"cluster {cid} / {k}", fill=(255, 255, 255))
            im.save(p, quality=90)
            ex.append({"image": str(p)})
        c = {"id": cid, "name": name, "size": 100 * (cid + 1), "exemplars": ex, "votes": {"panel": 3 + cid}}
        if conf is not None:
            c["confidence"] = conf
        clusters.append(c)
    (cdir / "clusters.json").write_text(json.dumps({"clusters": clusters}, indent=1))
    spk = []
    for cid, (name, conf) in enumerate([("Tadano", 0.88), (None, 0.3), ("Najimi", 0.55)]):
        ex = []
        for k in range(3):
            p = sdir / "ex" / f"s{cid}_{k}.wav"
            _tone(p, 220 * (cid + 1) + 30 * k)
            ex.append({"audio": str(p), "text": f"line {k} of speaker {cid}"})
        spk.append({"id": cid, "name": name, "confidence": conf, "size": 40 + cid, "exemplars": ex})
    (sdir / "clusters.json").write_text(json.dumps({"clusters": spk}, indent=1))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    print(make_fixture(ap.parse_args().out))
