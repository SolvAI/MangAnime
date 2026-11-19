"""Contact sheet of a finished episode: N frames at even intervals, with their time, in one picture.

  python scripts/contact_sheet.py --video runs/best/ep22_v3e/episode_....mp4 --out runs/ep22_v3e/eval/episode_contact.jpg [--n 48 --cols 8]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def duration(video: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video], capture_output=True, text=True)
    return float(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--width", type=int, default=320)
    a = ap.parse_args()
    dur = duration(a.video)
    w, h = a.width, a.width * 9 // 16
    rows = -(-a.n // a.cols)
    sheet = Image.new("RGB", (a.cols * w, rows * h), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory() as td:
        for i in range(a.n):
            t = dur * (i + 0.5) / a.n
            f = Path(td) / f"{i:03d}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", a.video, "-frames:v", "1", "-vf", f"scale={w}:{h}", str(f)], check=False)
            if not f.exists():
                continue
            x, y = (i % a.cols) * w, (i // a.cols) * h
            sheet.paste(Image.open(f).convert("RGB"), (x, y))
            label = f"{int(t // 60)}:{int(t % 60):02d}"
            d.rectangle((x, y, x + 44, y + 14), fill=(0, 0, 0))
            d.text((x + 3, y + 1), label, fill=(255, 255, 255))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=88)
    print(f"contact sheet: {a.n} frames of {dur / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
