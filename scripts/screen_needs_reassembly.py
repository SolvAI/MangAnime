"""Exit 0 when the all-ages screen replaced a clip after the episode was assembled (so it has to be assembled again), else 1.

  python scripts/screen_needs_reassembly.py <run>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json  # noqa: E402

rd = P("runs") / sys.argv[1]
flagged = (read_json(rd / "eval" / "screen.json") or {}).get("flagged", [])
ep = rd / "episode.mp4"
newer = [x["shot_id"] for x in flagged if (rd / "shots" / x["shot_id"] / "clip.mp4").exists()
         and (not ep.exists() or (rd / "shots" / x["shot_id"] / "clip.mp4").stat().st_mtime > ep.stat().st_mtime)]
print(f"screen: {len(flagged)} flagged, {len(newer)} replaced since the last assembly")
sys.exit(0 if newer else 1)
