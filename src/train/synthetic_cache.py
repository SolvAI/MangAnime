"""Write a SYNTHETIC musubi-tuner cache (random tensors with the exact shapes / keys / file names of a real Wan 2.2
I2V cache), using musubi's own save functions.

Purpose: throughput / VRAM probes on rented GPUs. Seconds-per-step and peak VRAM depend only on tensor shapes, so a
probe does not need real data, and nothing derived from the private media leaves the machine. The loss of such a run
is meaningless. Runs on the CPU (GPU hidden).

    CUDA_VISIBLE_DEVICES="" venvs/musubi/bin/python -m src.train.synthetic_cache --out <dir> --n 8 \
        --resolution 832x480 --frames 81
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--resolution", default="832x480")
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--tokens", type=int, default=128, help="T5 sequence length of the fake captions")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.cuda.current_device = lambda: 0  # import-time default argument in musubi's wan/modules/t5.py
    from musubi_tuner.dataset.cache_io import save_latent_cache_wan, save_text_encoder_output_cache_wan
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    w, h = (int(x) for x in a.resolution.lower().split("x"))
    assert w % 16 == 0 and h % 16 == 0 and (a.frames - 1) % 4 == 0
    lat_f, lat_h, lat_w = (a.frames - 1) // 4 + 1, h // 8, w // 8
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(a.seed)
    dtype = torch.bfloat16  # what the GPU caching path writes (vae.dtype)
    for i in range(a.n):
        key = f"synthetic{i:03d}_{0:05d}-{a.frames:03d}"
        item = ItemInfo(f"{key}.mp4", "synthetic probe item (random tensors, no media)", (w, h), (w, h, a.frames), frame_count=a.frames)
        item.latent_cache_path = str(out / f"{key}_{w:04d}x{h:04d}_wan.safetensors")
        item.text_encoder_output_cache_path = str(out / f"synthetic{i:03d}_wan_te.safetensors")
        latent = torch.randn(16, lat_f, lat_h, lat_w, generator=g).to(dtype)
        # I2V conditioning y = [4 mask channels | 16 latent channels of (first frame + zero padding)]
        msk = torch.zeros(4, lat_f, lat_h, lat_w)
        msk[:, 0] = 1.0
        y_lat = torch.randn(16, lat_f, lat_h, lat_w, generator=g) * 0.3
        y_lat[:, 0] = latent[:, 0].float()
        y = torch.cat([msk, y_lat], dim=0).to(dtype)
        save_latent_cache_wan(item, latent, None, y, None)
        embed = (torch.randn(a.tokens, 4096, generator=g) * 0.15).to(dtype)
        save_text_encoder_output_cache_wan(item, embed)
    files = sorted(out.glob("*.safetensors"))
    print(f"synthetic cache: {a.n} items, {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB in {out}")
    print("example:", files[0].name, "|", files[-1].name)


if __name__ == "__main__":
    main()
