"""Run a musubi-tuner *caching* module on the CPU without ever touching the GPU.

    CUDA_VISIBLE_DEVICES="" venvs/musubi/bin/python -m src.train.musubi_cpu wan_cache_latents --device cpu ...

Why a shim: `musubi_tuner/wan/modules/t5.py` evaluates `torch.cuda.current_device()` in a default argument at import
time, so the Wan scripts cannot even be imported when no GPU is visible. The shim hides the GPU, stubs that call and
switches the text encoders to float32 (bf16 matmuls are extremely slow on CPUs without AVX512-BF16).
Only meant for small sets (smoke test) or when the local GPU is busy; the cached values are the same up to dtype.
"""
from __future__ import annotations

import os
import runpy
import sys

ALLOWED = {
    "wan_cache_latents",
    "wan_cache_text_encoder_outputs",
    "qwen_image_cache_latents",
    "qwen_image_cache_text_encoder_outputs",
}
# trainers: argument / timestep-distribution check only (`--show_timesteps console` returns before any model is loaded)
CHECK_ONLY = {"wan_train_network", "qwen_image_train_network"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ALLOWED | CHECK_ONLY:
        raise SystemExit(f"usage: python -m src.train.musubi_cpu {{{'|'.join(sorted(ALLOWED | CHECK_ONLY))}}} [musubi args...]")
    module = sys.argv[1]
    if module in CHECK_ONLY and "--show_timesteps" not in sys.argv:
        raise SystemExit(f"{module} can only be run on the CPU with --show_timesteps (argument check)")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard guarantee: nothing below can allocate on the GPU

    import torch

    assert not torch.cuda.is_available(), "GPU must be hidden for the CPU caching path"
    torch.cuda.current_device = lambda: 0  # import-time default argument in wan/modules/t5.py
    torch.set_num_threads(max(4, (os.cpu_count() or 8) // 2))

    if module == "wan_cache_text_encoder_outputs":
        from musubi_tuner.wan.configs import wan_t2v_14B

        wan_t2v_14B.t2v_14B.t5_dtype = torch.float32

    argv = [a for a in sys.argv[2:]]
    if module in ALLOWED and "--device" not in argv:
        argv += ["--device", "cpu"]
    sys.argv = [f"musubi_tuner.{module}"] + argv
    runpy.run_module(f"musubi_tuner.{module}", run_name="__main__")


if __name__ == "__main__":
    main()
