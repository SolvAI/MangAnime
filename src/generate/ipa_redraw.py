"""Redraw one person of a frame with a REFERENCE PICTURE of that character (IP-Adapter, D-051). Standalone worker.

Runs in its own environment (venvs/sdxl: diffusers + peft), because ComfyUI has no IP-Adapter node without third-party code:
  venvs/sdxl/bin/python src/generate/ipa_redraw.py --jobs jobs.jsonl
One job per line, processed in order (a job may read the output of an earlier one):
  {"id", "image", "mask", "prompt", "negative", "refs": [picture paths], "out", "seed", "denoise", "ip_scale", "ip_end"}
The white part of `mask` is redrawn from `prompt`; `refs` steer what the person looks like. Same checkpoint, LoRAs, sampler
and step count as the ComfyUI frames (src/generate/comfy.py: wf_illustrious_inpaint), so the redrawn part matches the rest.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

MODELS = Path("/home/user/komi-poc-work/models")
CKPT = MODELS / "comfy" / "checkpoints" / "prefect_illustrious_v8.fp16.safetensors"
LORAS = [("anime_screencap-IL-NOOB_v3.safetensors", 1.0), ("StS-Illustrious-Detail-Slider-v1.0.safetensors", 1.5)]
IPA_DIR = MODELS / "ipadapter_h94"                      # h94/IP-Adapter layout: sdxl_models/, models/image_encoder/
W, H = 1344, 768


def encode_long(pipe, prompt: str, negative: str):
    """SDXL text conditioning for prompts longer than 77 tokens: 75-token chunks encoded one by one and joined, as ComfyUI does."""
    def ids(tok, text):
        t = tok(text, truncation=False, add_special_tokens=False).input_ids
        return [t[i : i + 75] for i in range(0, max(len(t), 1), 75)]

    def run(text, n_chunks):
        embs, pooled = [], None
        for tok, enc in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
            chunks = ids(tok, text)
            chunks += [[] for _ in range(n_chunks - len(chunks))]
            outs = []
            for c in chunks:
                x = [tok.bos_token_id] + c + [tok.eos_token_id]
                x += [tok.pad_token_id if tok is pipe.tokenizer_2 else tok.eos_token_id] * (77 - len(x))
                o = enc(torch.tensor([x], device=enc.device), output_hidden_states=True)
                outs.append(o.hidden_states[-2])
                if enc is pipe.text_encoder_2 and pooled is None:
                    pooled = o[0]
            embs.append(torch.cat(outs, dim=1))
        return torch.cat(embs, dim=-1), pooled

    n = max(len(ids(pipe.tokenizer, prompt)), len(ids(pipe.tokenizer, negative)), len(ids(pipe.tokenizer_2, prompt)), len(ids(pipe.tokenizer_2, negative)))
    pe, pp = run(prompt, n)
    ne, npool = run(negative, n)
    return pe, pp, ne, npool


def reference(path: str, side: int = 448) -> Image.Image:
    """Reference picture on a white square, so the adapter takes the person and not the picture's format."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--adapter", default="ip-adapter-plus_sdxl_vit-h.safetensors")
    ap.add_argument("--steps", type=int, default=26)
    ap.add_argument("--cfg", type=float, default=5.5)
    a = ap.parse_args()
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLInpaintPipeline

    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    t0 = time.time()
    pipe = StableDiffusionXLInpaintPipeline.from_single_file(str(CKPT), torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    names = []
    for i, (f, w) in enumerate(LORAS):
        pipe.load_lora_weights(str(MODELS / "comfy" / "loras"), weight_name=f, adapter_name=f"l{i}")
        names.append(f"l{i}")
    pipe.set_adapters(names, [w for _, w in LORAS])
    pipe.load_ip_adapter(str(IPA_DIR), subfolder="sdxl_models", weight_name=a.adapter, image_encoder_folder="models/image_encoder")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"loaded in {time.time() - t0:.0f}s; {len(jobs)} jobs", flush=True)
    res = []
    for k, j in enumerate(jobs):
        try:
            img = Image.open(j["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
            mask = Image.open(j["mask"]).convert("L").resize((W, H), Image.LANCZOS)
            refs = [reference(r) for r in j.get("refs") or []]
            scale = float(j.get("ip_scale", 0.6)) if refs else 0.0
            pipe.set_ip_adapter_scale(scale)
            pe, pp, ne, npool = encode_long(pipe, j["prompt"], j.get("negative", ""))
            end = float(j.get("ip_end", 0.85))

            def cb(p, step, t, kw, _end=end, _n=a.steps):                 # the reference guides the first part of the steps only
                if step + 1 >= int(_end * _n):
                    p.set_ip_adapter_scale(0.0)
                return kw

            out = pipe(prompt_embeds=pe, pooled_prompt_embeds=pp, negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npool,
                       image=img, mask_image=mask, width=W, height=H, strength=float(j.get("denoise", 0.8)), num_inference_steps=a.steps,
                       guidance_scale=a.cfg, ip_adapter_image=[refs if refs else [Image.new("RGB", (448, 448), "white")]],
                       generator=torch.Generator("cuda").manual_seed(int(j.get("seed", 0))), callback_on_step_end=cb).images[0]
            Path(j["out"]).parent.mkdir(parents=True, exist_ok=True)
            if j.get("out_size"):
                out = out.resize(tuple(j["out_size"]), Image.LANCZOS)
            out.save(j["out"])
            res.append({"id": j["id"], "ok": True, "out": j["out"]})
        except Exception as e:                                             # noqa: BLE001
            res.append({"id": j["id"], "ok": False, "error": str(e)[:300]})
            print(f"job {j['id']} failed: {str(e)[:300]}", flush=True)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    Path(a.jobs).with_suffix(".results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    print(f"IPA_REDRAW_DONE {sum(r['ok'] for r in res)}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
