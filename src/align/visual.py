"""Image embeddings (DINOv2) for panel<->frame matching and keyframe fidelity (brief §3b, §5.3 step 3).

embed_images(list of BGR/RGB uint8 arrays or paths) -> L2-normalised [N, D] float32 (CLS + mean patch token).
Runs on GPU when enough VRAM is free, else CPU. Results for file paths are cached on disk by file signature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.common.config import P
from src.common.io import file_sig

MODEL_ID = "facebook/dinov2-base"
_state: dict = {}
_CACHE = P("cache") / "emb_img"


def _load():
    if "model" not in _state:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        dev = "cpu"
        try:
            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3 * 2**30:
                dev = "cuda"
        except Exception:
            pass
        _state["proc"] = AutoImageProcessor.from_pretrained(MODEL_ID)
        _state["model"] = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
        _state["dev"] = dev
    return _state["proc"], _state["model"], _state["dev"]


def _to_pil(x) -> Image.Image:
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.fromarray(np.asarray(x)[..., :3].astype(np.uint8))


def embed_images(items: list, batch: int = 32) -> np.ndarray:
    import torch

    proc, model, dev = _load()
    out = []
    for i in range(0, len(items), batch):
        ims = [_to_pil(x) for x in items[i : i + batch]]
        inp = proc(images=ims, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        e = torch.nn.functional.normalize(e.float(), dim=1)
        out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1536), np.float32)


def embed_files(paths: list[str | Path], transform=None, tag: str = "raw") -> np.ndarray:
    """Cached embeddings for image files; `transform(bgr)->array` (e.g. line-art) is applied before embedding."""
    import cv2

    _CACHE.mkdir(parents=True, exist_ok=True)
    keys = [f"{tag}_{file_sig(p)}" for p in paths]
    out: list[np.ndarray | None] = [None] * len(paths)
    todo = []
    for i, k in enumerate(keys):
        f = _CACHE / k[-2:] / f"{k}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            todo.append(i)
    B = 64
    for j in range(0, len(todo), B):
        idx = todo[j : j + B]
        ims = []
        for i in idx:
            bgr = cv2.imread(str(paths[i]))
            ims.append(transform(bgr) if transform else bgr[..., ::-1])
        embs = embed_images(ims)
        for i, e in zip(idx, embs):
            f = _CACHE / keys[i][-2:] / f"{keys[i]}.npy"
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, e.astype(np.float16))
            out[i] = e
    return np.stack([np.asarray(e, np.float32) for e in out]) if out else np.zeros((0, 1536), np.float32)
