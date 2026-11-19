#!/usr/bin/env python
"""Mini end-to-end test with networking disabled (brief §3b).

Runs every local stage on a tiny slice and replays the director / judge from the disk cache:
  manga OCR cache -> text alignment -> director (cache) -> still/pan-zoom clips -> mix -> assembly -> judge (cache)
The network is blocked for this process and its children by pointing all proxies at a dead address and by
monkey-patching socket connections to non-loopback hosts (loopback stays open for the local model servers).

  python scripts/offline_check.py --run m1            # after the M1 slice has been generated online once
Exit code 0 = every stage ran offline.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def block_network() -> None:
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ[k] = "http://127.0.0.1:9"       # discard port: any outbound HTTP fails immediately
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["KOMI_OFFLINE"] = "1"               # LLM client: remote endpoints are cache-only
    real_connect = socket.socket.connect

    def guarded(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "localhost", "::1") and not str(host).startswith("/"):
            raise OSError(f"offline_check: network access blocked ({host})")
        return real_connect(self, address, *a, **kw)

    socket.socket.connect = guarded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m1")
    ap.add_argument("--chapter", type=int, default=None)
    a = ap.parse_args()
    block_network()
    from src.common.config import P
    from src.common.io import media_duration, read_json, read_jsonl

    results: list[tuple[str, bool, str]] = []

    def step(name, fn):
        t = time.time()
        try:
            info = fn()
            results.append((name, True, f"{info} ({time.time() - t:.1f}s)"))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {str(e)[:200]}"))

    rd = P("runs") / a.run
    sl = read_json(rd / "shotlist.json") or {}
    chapter = a.chapter or (sl.get("chapters") or [1])[0]

    def s_network():
        import httpx

        try:
            httpx.get("https://api.deepseek.com/models", timeout=5)
        except Exception:
            return "outbound HTTP is blocked"
        raise RuntimeError("network is still reachable")

    def s_manga():
        rows = [p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] == chapter]
        assert rows, "no parsed panels"
        return f"{len(rows)} panels of chapter {chapter}"

    def s_align():
        from src.align.text_align import embed

        e = embed(["Good morning, Komi-san.", "She is so beautiful."])
        return f"BGE-M3 embeddings {e.shape}"

    def s_director():
        from src.generate.director import direct_chapter

        res = direct_chapter(chapter)                 # replayed from the content-hash cache (CacheMiss otherwise)
        return f"{len(res['shots'])} shots replayed from cache"

    def s_video():
        from src.generate.assemble import assemble
        from src.generate.audio import mix_stage

        import shutil

        shots = sl["shots"][:6]
        scratch = rd.parent / f"{rd.name}_offline_check"      # never touch the real run's mix / episode files
        (scratch / "shots").mkdir(parents=True, exist_ok=True)
        if not (scratch / "tts").exists():
            (scratch / "tts").symlink_to(rd / "tts")
        for s in shots:
            d = scratch / "shots" / s["shot_id"]
            d.mkdir(exist_ok=True)
            shutil.copyfile(rd / "shots" / s["shot_id"] / "kf.png", d / "kf.png")
        mix_stage(scratch, shots)
        out = assemble(scratch, shots, variant="animatic", burn_subs=True)
        return f"animatic of 6 shots: {media_duration(out):.1f}s"

    def s_judge():
        from src.qa.judge import judge_shot

        state = read_json(rd / "state.json") or {}
        for s in sl["shots"]:
            n = (state.get(s["shot_id"]) or {}).get("vid_attempt", 0)
            for q in (rd / "shots" / s["shot_id"] / f"qa_video_a{n}.json", rd / "shots" / s["shot_id"] / "qa_video_a0.json"):
                qa = read_json(q)
                if qa and qa.get("verdict", {}).get("judge_endpoint") == "deepseek":
                    v = judge_shot(s, qa["evidence"], stage="video")
                    return f"judge replayed from cache for {s['shot_id']}: {v['verdict']}"
        raise RuntimeError("no judged shot found in this run")

    def s_local_models():
        import torch

        return f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"

    for name, fn in (("network blocked", s_network), ("manga parse (cached)", s_manga), ("text embeddings (local)", s_align),
                     ("director (cache replay)", s_director), ("mix + assembly (local)", s_video), ("judge (cache replay)", s_judge),
                     ("local runtime", s_local_models)):
        step(name, fn)
    ok = all(r[1] for r in results)
    for name, good, info in results:
        print(f"[{'OK ' if good else 'FAIL'}] {name}: {info}")
    print("OFFLINE_CHECK_PASSED" if ok else "OFFLINE_CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
