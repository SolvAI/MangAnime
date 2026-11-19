"""Style LoRA for the frame model (D-058): teach the SDXL checkpoint how THIS show draws, from frames of the training episodes.

Runs in its own environment (venvs/sdxl: diffusers + peft), on the local GPU:
  venvs/sdxl/bin/python src/train/style_lora.py dataset                      # style bank -> data/processed/train/style/meta.jsonl
  venvs/sdxl/bin/python src/train/style_lora.py train [--steps 1600]         # -> models/comfy/loras/komi_style_v1.safetensors

Why: the image model draws the right people in the right clothes, but not the show's sparse line and soft backgrounds
(edge density 0.29 against 0.18; RESULTS M6), and wording does not move it. A LoRA trained on ~1,000 ordinary frames of
episodes 1-21 is then used as a light image-to-image pass over each first frame (src/generate/restyle.py), which keeps the
layout and the people and changes how they are drawn. Episodes 22-24 are never in the dataset (asserted).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

WORK = Path("/home/user/komi-poc-work")
CKPT = WORK / "models" / "comfy" / "checkpoints" / "prefect_illustrious_v8.fp16.safetensors"
DATA = WORK / "data" / "processed" / "train" / "style"
BANK = WORK / "data" / "processed" / "v3" / "style_bank.json"
OUT = WORK / "models" / "comfy" / "loras"
TRIGGER = "komistyle"
RES = (1024, 576)            # 16:9, the size the first frames are kept at


def caption(b: dict) -> str:
    n = int(b.get("people") or 0)
    who = "no humans" if n == 0 else ("1 person" if n == 1 else f"{min(n, 6)} people")
    t = "" if b.get("time") in (None, "unclear") else f"{b['time']}, "
    return f"{TRIGGER}, anime screencap, {b['place']}, {t}{b['shot_size']} shot, {who}"


def dataset() -> None:
    bank = json.loads(BANK.read_text())
    rows = [b for b in bank if b.get("clean") and b.get("plain") and not b.get("letterbox") and b.get("black", 0) <= 0.35 and Path(b["path"]).exists()]
    assert rows and all(int(b["episode"]) <= 21 for b in rows), "held-out episodes must never be in the style dataset"
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "meta.jsonl", "w") as f:
        for b in rows:
            f.write(json.dumps({"path": b["path"], "episode": b["episode"], "shot": b["shot"], "caption": caption(b)}) + "\n")
    eps = sorted({b["episode"] for b in rows})
    print(f"style dataset: {len(rows)} frames of {len(eps)} training episodes ({eps[0]}-{eps[-1]}) -> {DATA / 'meta.jsonl'}")


def train(steps: int, rank: int, lr: float, batch: int, name: str, seed: int = 0, smoke: bool = False) -> None:
    import numpy as np
    import torch
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image

    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cpu" if smoke else "cuda"                 # --smoke: the whole path on the CPU, 4 small frames, 2 steps (the GPU is busy rendering)
    half = torch.float32 if smoke else torch.float16
    rows = [json.loads(l) for l in open(DATA / "meta.jsonl")]
    if smoke:
        rows = rows[:4]
    assert all(int(r["episode"]) <= 21 for r in rows)
    pipe = StableDiffusionXLPipeline.from_single_file(str(CKPT), torch_dtype=half, use_safetensors=True)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    print(f"{len(rows)} frames; prediction type {sched.config.prediction_type}; {steps} steps, rank {rank}, lr {lr}, batch {batch}")
    W, H = (256, 144) if smoke else RES

    # 1) latents and text embeddings once (VAE in float32: the SDXL VAE overflows in half precision)
    cache = DATA / (f"cache_{W}x{H}.pt" if not smoke else "cache_smoke.pt")
    if cache.exists():
        c = torch.load(cache)
    else:
        vae = pipe.vae.to(dev, torch.float32)
        lat = []
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                ims = []
                for r in rows[i : i + 8]:
                    im = Image.open(r["path"]).convert("RGB").resize((W, H), Image.LANCZOS)
                    ims.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1.0)
                x = torch.stack(ims).to(dev)
                lat.append((vae.encode(x).latent_dist.sample() * vae.config.scaling_factor).to(half).cpu())
        vae.to("cpu")
        pipe.text_encoder.to(dev)
        pipe.text_encoder_2.to(dev)
        caps = sorted({r["caption"] for r in rows} | {""})
        emb = {}
        with torch.no_grad():
            for cp in caps:
                pe, _, pp, _ = pipe.encode_prompt(prompt=cp, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
                emb[cp] = (pe[0].to(half).cpu(), pp[0].to(half).cpu())
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        c = {"latents": torch.cat(lat), "emb": emb}
        torch.save(c, cache)
    latents, emb = c["latents"], c["emb"]
    assert not torch.isnan(latents.float()).any(), "NaN in cached latents"

    # 2) LoRA on the UNet attention layers
    unet = pipe.unet.to(dev)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank // 2, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()                     # trainable weights in float32, the frozen net stays in half precision
    unet.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    lr_at = lambda s: lr * min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps))      # noqa: E731
    scaler = torch.amp.GradScaler("cuda", enabled=not smoke)
    time_ids = torch.tensor([[H, W, 0, 0, H, W]], dtype=half, device=dev)
    ac = sched.alphas_cumprod.to(dev)
    snr = ac / (1 - ac)
    order = []
    t0 = time.time()
    run_loss = 0.0
    unet.train()
    for step in range(steps):
        if len(order) < batch:
            order = random.sample(range(len(rows)), len(rows))
        idx = [order.pop() for _ in range(batch)]
        x0 = latents[idx].to(dev).float()
        caps_b = [("" if random.random() < 0.1 else rows[i]["caption"]) for i in idx]
        pe = torch.stack([emb[cp][0] for cp in caps_b]).to(dev)
        pp = torch.stack([emb[cp][1] for cp in caps_b]).to(dev)
        noise = torch.randn_like(x0)
        t = torch.randint(0, sched.config.num_train_timesteps, (batch,), device=dev).long()
        xt = sched.add_noise(x0, noise, t)
        target = noise if sched.config.prediction_type == "epsilon" else sched.get_velocity(x0, noise, t)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.float16, enabled=not smoke):
            pred = unet(xt.to(half), t, encoder_hidden_states=pe, added_cond_kwargs={"text_embeds": pp, "time_ids": time_ids.repeat(batch, 1)}).sample
        loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
        w = torch.clamp(snr[t], max=5.0) / (snr[t] if sched.config.prediction_type == "epsilon" else snr[t] + 1)      # min-SNR, gamma 5
        loss = (loss * w).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        run_loss += float(loss)
        if (step + 1) % 100 == 0 or smoke:
            print(f"step {step + 1}/{steps}  loss {run_loss / 100:.4f}  {(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
            run_loss = 0.0
        if (step + 1) % 400 == 0 or step + 1 == steps:
            sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            OUT.mkdir(parents=True, exist_ok=True)
            tag = name if step + 1 == steps else f"{name}_s{step + 1}"
            StableDiffusionXLPipeline.save_lora_weights(str(OUT), unet_lora_layers=sd, weight_name=f"{tag}.safetensors", safe_serialization=True)
    (OUT / f"{name}.json").write_text(json.dumps({"base": CKPT.name, "frames": len(rows), "episodes": sorted({r["episode"] for r in rows}), "steps": steps, "rank": rank,
                                                  "lr": lr, "batch": batch, "resolution": list(RES), "trigger": TRIGGER, "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"saved {OUT / (name + '.safetensors')} after {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dataset", "train"])
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="komi_style_v1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "dataset":
        dataset()
    elif a.smoke:
        train(2, 4, a.lr, 2, "smoke_style", smoke=True)
    else:
        train(a.steps, a.rank, a.lr, a.batch, a.name)
