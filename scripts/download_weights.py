#!/usr/bin/env python
"""Fetch every pinned weight in configs/weights.yaml once, verify checksums, write models/manifest.lock.json.

Usage: python scripts/download_weights.py [--stage m1] [--only id1,id2]
Idempotent + resumable (huggingface_hub resumes partial files). Never uploads anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("KOMI_WORK", "/home/user/komi-poc-work"))
MODELS = WORK / "models"
LOCK = MODELS / "manifest.lock.json"


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of big files (faster)")
    args = ap.parse_args()
    load_env()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    entries = yaml.safe_load((REPO / "configs" / "weights.yaml").read_text())["weights"]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    only = set(args.only.split(",")) if args.only else None
    ok = True
    for e in entries:
        if only and e["id"] not in only:
            continue
        if args.stage and e.get("stage") not in (None, args.stage) and e["kind"] != "local":
            continue
        t0 = time.time()
        try:
            if e["kind"] == "local":
                p = Path(e["path"])
                lock[e["id"]] = {"kind": "local", "path": str(p), "exists": p.exists(),
                                 "size": p.stat().st_size if p.is_file() else None}
                print(f"[local] {e['id']}: {'ok' if p.exists() else 'MISSING'} {p}", flush=True)
                continue
            target = MODELS / e["target"]
            target.mkdir(parents=True, exist_ok=True)
            if e["kind"] == "file":
                dest = target / Path(e["file"]).name
                prev = lock.get(e["id"], {})
                if dest.exists() and prev.get("size") == dest.stat().st_size:
                    print(f"[skip] {e['id']} -> {dest}", flush=True)
                    continue
                staging = MODELS / "_hf" / e["repo"].replace("/", "__")
                got = Path(hf_hub_download(e["repo"], e["file"], revision=e["revision"], local_dir=staging))
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                os.replace(got, dest)
                rec = {"kind": "file", "repo": e["repo"], "revision": e["revision"], "file": e["file"],
                       "path": str(dest), "size": dest.stat().st_size}
                if not args.no_hash:
                    rec["sha256"] = sha256(dest)
                    try:
                        info = api.model_info(e["repo"], revision=e["revision"], files_metadata=True)
                        hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}.get(e["file"])
                        rec["sha256_matches_hub"] = (hub == rec["sha256"]) if hub else None
                        if hub and hub != rec["sha256"]:
                            ok = False
                            print(f"[CHECKSUM MISMATCH] {e['id']}", flush=True)
                    except Exception as ex:  # metadata lookup is best-effort
                        rec["sha256_matches_hub"] = f"unverified: {str(ex)[:80]}"
                lock[e["id"]] = rec
            else:  # repo snapshot
                done_marker = target / ".download_complete"
                if done_marker.exists():
                    print(f"[skip] {e['id']} -> {target}", flush=True)
                    continue
                snapshot_download(e["repo"], revision=e["revision"], local_dir=target,
                                  allow_patterns=e.get("allow"), ignore_patterns=e.get("ignore"))
                files = {str(p.relative_to(target)): p.stat().st_size for p in target.rglob("*")
                         if p.is_file() and ".cache" not in p.parts}
                done_marker.write_text(time.strftime("%F %T"))
                lock[e["id"]] = {"kind": "repo", "repo": e["repo"], "revision": e["revision"], "path": str(target),
                                 "n_files": len(files), "size": sum(files.values())}
            print(f"[ok] {e['id']} in {time.time() - t0:.0f}s", flush=True)
        except Exception as ex:
            ok = False
            print(f"[FAIL] {e['id']}: {str(ex)[:300]}", flush=True)
        finally:
            LOCK.parent.mkdir(parents=True, exist_ok=True)
            LOCK.write_text(json.dumps(lock, indent=1))
    print("ALL_OK" if ok else "SOME_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
