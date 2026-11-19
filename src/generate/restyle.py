"""Light image-to-image pass that redraws a finished first frame in the show's own line and colour (style LoRA, D-058).

Standalone worker for venvs/sdxl (diffusers + peft), like ipa_redraw.py:
  venvs/sdxl/bin/python src/generate/restyle.py --jobs jobs.jsonl --lora komi_style_v1 [--strength 0.4] [--scale 1.0]
One job per line: {"id", "image", "prompt", "out", "seed"}. The frame keeps its layout and people (low strength); what
changes is how it is drawn. Jobs are written and the results checked by src/v3/restyle.py.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

MODELS = Path("/home/user/komi-poc-work/models")
CKPT = MODELS / "comfy" / "checkpoints" / "prefect_illustrious_v8.fp16.safetensors"
NEG = ("lowres, worst quality, bad anatomy, bad hands, extra fingers, text, watermark, signature, detailed background, intricate details, "
       "realistic, 3d, photo, glossy, shiny hair, gradient shading, thick outlines, heavy black lines")
W, H = 1024, 576


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--lora", default="komi_style_v1")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny pictures, 4 steps: checks the code path while the GPU is busy")
    a = ap.parse_args()
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline

    dev, dt = ("cpu", torch.float32) if a.smoke else ("cuda", torch.float16)
    w, h = (256, 144) if a.smoke else (W, H)
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(str(CKPT), torch_dtype=dt, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(MODELS / "comfy" / "loras"), weight_name=f"{a.lora}.safetensors", adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[a.scale])
    pipe.to(dev)
    if not a.smoke:
        pipe.vae.to(torch.float32)                  # the SDXL VAE overflows in half precision
    pipe.set_progress_bar_config(disable=True)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    t0 = time.time()
    done = 0
    for j in jobs:
        out = Path(j["out"])
        if out.exists():
            continue
        src = Image.open(j["image"]).convert("RGB")
        im = src.resize((w, h), Image.LANCZOS)
        g = torch.Generator(dev).manual_seed(int(j.get("seed", 0)))
        if a.smoke:
            res = pipe(prompt=j["prompt"], negative_prompt=NEG, image=im, strength=0.5, num_inference_steps=4, guidance_scale=a.cfg, generator=g).images[0]
        else:
            # img2img by hand so that the VAE runs in float32 and the UNet in half precision
            with torch.no_grad():
                x = pipe.image_processor.preprocess(im).to(dev, torch.float32)
                lat = (pipe.vae.encode(x).latent_dist.sample(g) * pipe.vae.config.scaling_factor).to(dt)
                res_lat = pipe(prompt=j["prompt"], negative_prompt=NEG, image=lat, strength=a.strength, num_inference_steps=a.steps, guidance_scale=a.cfg,
                               generator=g, output_type="latent").images
                img = pipe.vae.decode(res_lat.to(torch.float32) / pipe.vae.config.scaling_factor).sample
                res = pipe.image_processor.postprocess(img, output_type="pil")[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        res.resize(src.size, Image.LANCZOS).save(out)
        done += 1
    print(f"restyle: {done} of {len(jobs)} frames in {(time.time() - t0) / 60:.1f} min (lora {a.lora} x{a.scale}, strength {a.strength})")


if __name__ == "__main__":
    main()
