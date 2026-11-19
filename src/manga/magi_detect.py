"""Magi v3 page parsing (panels, texts, characters, tails, associations, reading order).

Runs in its own venv (`venvs/manga`, transformers 4.45.2 — the version Magi v3's remote code targets).
Standalone on purpose: no imports from src.* so the venv stays minimal.

  venvs/manga/bin/python src/manga/magi_detect.py --pages-jsonl data/interim/manga/pages.jsonl \
        --out-dir data/interim/manga/magi --volumes 1,2 [--ocr]

Output per page: <out>/vNNN/pNNN.json
  {panels, texts, characters, tails (xyxy, sorted: panels in reading order, texts in reading order),
   text_panel: [panel idx per text], char_panel: [...], text_char: {text idx: char idx},
   essential: [bool per text], char_clusters: [page-local cluster id per character], magi_ocr: [str per text] (if --ocr)}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_DIR = os.environ.get("MAGI_DIR", "/home/user/komi-poc-work/models/hf/magiv3")


def load_utils():
    spec = importlib.util.spec_from_file_location("magi_utils", Path(MODEL_DIR) / "utils.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_in(box, panel):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return panel[0] <= cx <= panel[2] and panel[1] <= cy <= panel[3]


def assign_to_panels(boxes, panels):
    out = []
    for b in boxes:
        best, best_v = -1, 0.0
        for i, p in enumerate(panels):
            x1, y1, x2, y2 = max(b[0], p[0]), max(b[1], p[1]), min(b[2], p[2]), min(b[3], p[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            v = inter / area
            if v > best_v:
                best, best_v = i, v
        if best < 0 and panels:  # nearest panel centre
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            best = int(np.argmin([np.linalg.norm(c - np.array([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])) for p in panels]))
        out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--volumes", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ocr", action="store_true", help="also run Magi's own OCR (fallback / comparison)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU (GPU busy)")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--pages", default=None, help="page range within the volume(s), e.g. 1-100")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    utils = load_utils()
    vols = {int(v) for v in args.volumes.split(",")} if args.volumes else None
    pages = [json.loads(l) for l in open(args.pages_jsonl) if l.strip()]
    pages = [p for p in pages if not vols or p["volume"] in vols]
    if args.pages:
        lo, hi = (int(x) for x in args.pages.split("-"))
        pages = [p for p in pages if lo <= p["page"] <= hi]
    out_dir = Path(args.out_dir)
    todo = []
    for p in pages:
        f = out_dir / f"v{p['volume']:03d}" / f"p{p['page']:03d}.json"
        if not f.exists():
            todo.append((p, f))
    if args.limit:
        todo = todo[: args.limit]
    print(f"magi: {len(todo)} pages to parse ({len(pages) - len(todo)} cached)", flush=True)
    if not todo:
        return
    use_gpu = torch.cuda.is_available() and not args.cpu
    dtype = torch.float16 if use_gpu else torch.float32
    if not use_gpu:
        torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype, trust_remote_code=True).eval()
    model = model.cuda() if use_gpu else model
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    t0 = time.time()
    for bi in range(0, len(todo), args.batch):
        chunk = todo[bi : bi + args.batch]
        images = [np.array(Image.open(p["path"]).convert("L").convert("RGB")) for p, _ in chunk]
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_gpu):
                dets = model.predict_detections_and_associations(images, processor)
                ocrs = model.predict_ocr(images, processor) if args.ocr else [None] * len(images)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            dets, ocrs = [], []
            for im in images:  # self-heal: retry one by one
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_gpu):
                    dets += model.predict_detections_and_associations([im], processor)
                    ocrs += model.predict_ocr([im], processor) if args.ocr else [None]
        for (p, f), det, ocr in zip(chunk, dets, ocrs):
            panels = [list(map(float, b)) for b in det["panels"]]
            texts = [list(map(float, b)) for b in det["texts"]]
            chars = [list(map(float, b)) for b in det["characters"]]
            tails = [list(map(float, b)) for b in det["tails"]]
            if not panels:  # full-page splash
                panels = [[0.0, 0.0, float(p["w"]), float(p["h"])]]
            # reading order (manga: right-to-left) using Magi's own utilities
            try:
                order = utils.sort_panels(panels)
            except Exception:
                order = sorted(range(len(panels)), key=lambda i: (panels[i][1], -panels[i][0]))
            panels_sorted = [panels[i] for i in order]
            try:
                t_order = utils.sort_text_boxes_in_reading_order(texts, panels_sorted) if texts else []
            except Exception:
                t_order = sorted(range(len(texts)), key=lambda i: (texts[i][1], -texts[i][0]))
            t_inv = {old: new for new, old in enumerate(t_order)}
            texts_sorted = [texts[i] for i in t_order]
            essential = [bool(det["is_essential_text"][i]) if i < len(det["is_essential_text"]) else True for i in t_order]
            text_char = {}
            for ti, ci in det["text_character_associations"]:
                if ti in t_inv and ci < len(chars):
                    text_char[str(t_inv[ti])] = int(ci)
            text_tail = {}
            for ti, tj in det["text_tail_associations"]:
                if ti in t_inv and tj < len(tails):
                    text_tail[str(t_inv[ti])] = int(tj)
            rec = {
                "volume": p["volume"], "page": p["page"], "w": p["w"], "h": p["h"],
                "panels": panels_sorted, "texts": texts_sorted, "characters": chars, "tails": tails,
                "text_panel": assign_to_panels(texts_sorted, panels_sorted),
                "char_panel": assign_to_panels(chars, panels_sorted),
                "text_char": text_char, "text_tail": text_tail, "essential": essential,
                "char_clusters": [int(x) for x in det["character_cluster_labels"]],
            }
            if ocr is not None:
                m_ocr = [""] * len(texts_sorted)
                for txt, bb in zip(ocr["ocr_texts"], ocr["bboxes"]):
                    if not texts_sorted:
                        break
                    j = int(np.argmax([iou(bb, t) for t in texts_sorted]))
                    if iou(bb, texts_sorted[j]) > 0.3:
                        m_ocr[j] = (m_ocr[j] + " " + txt.strip()).strip()
                rec["magi_ocr"] = m_ocr
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            os.replace(tmp, f)
        done = bi + len(chunk)
        if done % 20 < args.batch:
            print(f"magi {done}/{len(todo)}  {(time.time() - t0) / done:.2f}s/page", flush=True)
    print("MAGI_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
