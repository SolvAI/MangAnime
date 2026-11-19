"""Headless ComfyUI client + workflow builders (API-format graphs, also dumped to workflows/*.json for versioning).

Workflows:
  wf_qwen_edit(...)  panel (+ up to 2 reference images) -> anime keyframe   (Qwen-Image-Edit-2511 + Lightning 4-step)
  wf_wan_i2v(...)    keyframe (+ optional last frame)   -> video clip       (Wan 2.2 I2V A14B high/low + lightx2v 4-step)

Sampler settings follow the official references: Wan lightx2v sigmas 1.0/0.9375/0.8333/0.625/0 == `simple` schedule
with shift 5 and 4 steps (2 high-noise + 2 low-noise, cfg 1); Qwen-Edit-2511 template: AuraFlow shift 3.1, CFGNorm 1,
`index_timestep_zero` reference method, 4 steps cfg 1 with the Lightning LoRA.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import httpx

from src.common.config import CFG, P

WAN_HIGH = "Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
WAN_LOW = "Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
WAN_T5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
LIGHT_HIGH = "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
LIGHT_LOW = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
QIE_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
QIE_TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QIE_VAE = "qwen_image_vae.safetensors"
QIE_LIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

ILL_CKPT = "prefect_illustrious_v8.fp16.safetensors"                  # civitai 1224788 (v8)
ILL_LORA_SCREENCAP = "anime_screencap-IL-NOOB_v3.safetensors"         # civitai 345962 (Illustrious v3.0); triggers: anime screencap, anime coloring
ILL_LORA_DETAIL = "StS-Illustrious-Detail-Slider-v1.0.safetensors"    # civitai 1001945 (slider; no trigger)
SDXL_CN_UNION = "xinsir_union_promax/diffusion_pytorch_model_promax.safetensors"   # HF xinsir/controlnet-union-sdxl-1.0 (promax)
WAN_ANIME_HIGH = "wan2.2_i2v_animestyle_v2_high.safetensors"          # civitai 2222779 v2.0; triggers: An1meStyl3, AnimeStyle
WAN_ANIME_LOW = "wan2.2_i2v_animestyle_v2_low.safetensors"

WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
           "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


class Comfy:
    def __init__(self, port: int | None = None):
        self.base = f"http://127.0.0.1:{port or CFG.comfy.port}"
        self.client_id = uuid.uuid4().hex
        self.io = P("work") / "comfy_io"

    # ------------------------------------------------------------------ files
    def put_input(self, path: str | Path, name: str | None = None) -> str:
        """Copy an image into ComfyUI's input directory (we own that directory). Returns the name for LoadImage."""
        src = Path(path)
        name = name or f"{uuid.uuid4().hex[:10]}_{src.name}"
        dst = self.io / "input" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return name

    # ------------------------------------------------------------------ queue
    def queue(self, graph: dict) -> str:
        r = httpx.post(f"{self.base}/prompt", json={"prompt": graph, "client_id": self.client_id}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:1500]}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 1800.0, poll: float = 1.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                h = httpx.get(f"{self.base}/history/{prompt_id}", timeout=30).json()
            except Exception:
                time.sleep(poll)
                continue
            if prompt_id in h:
                rec = h[prompt_id]
                st = rec.get("status", {})
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                    raise RuntimeError(f"ComfyUI execution error: {json.dumps(msgs)[:1500]}")
                if st.get("completed") or rec.get("outputs"):
                    return rec.get("outputs", {})
            time.sleep(poll)
        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")

    def run(self, graph: dict, timeout: float = 1800.0) -> list[Path]:
        """Queue, wait, and return output file paths (images / videos)."""
        outs = self.wait(self.queue(graph), timeout)
        files = []
        for node_out in outs.values():
            for key in ("images", "gifs", "videos", "latents"):
                for it in node_out.get(key, []) or []:
                    sub = it.get("subfolder", "")
                    base = self.io / ("output" if it.get("type", "output") == "output" else "temp")
                    files.append(base / sub / it["filename"])
        return files

    def free(self) -> None:
        try:
            httpx.post(f"{self.base}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except Exception:
            pass

    def interrupt(self) -> None:
        try:
            httpx.post(f"{self.base}/interrupt", timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------- workflows
def wf_qwen_edit(images: list[str], prompt: str, *, width: int = 1280, height: int = 720, seed: int = 0, steps: int = 4,
                 cfg: float = 1.0, loras: list[tuple[str, float]] | None = None, prefix: str = "kf",
                 negative: str = "", use_lightning: bool = True, shift: float = 3.1) -> dict:
    """`images`: names in ComfyUI's input dir; image1 = source panel, image2/3 = references."""
    assert 1 <= len(images) <= 3
    g: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": QIE_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QIE_TE, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QIE_VAE}},
    }
    model = ["unet", 0]
    chain = ([(QIE_LIGHT, 1.0)] if use_lightning else []) + list(loras or [])
    for i, (name, strength) in enumerate(chain):
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [f"lora{i}", 0]
    g["msa"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": shift}}
    g["cfgn"] = {"class_type": "CFGNorm", "inputs": {"model": ["msa", 0], "strength": 1.0}}
    img_in = {}
    for i, name in enumerate(images):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        img_in[f"image{i + 1}"] = [f"img{i}", 0]
    g["pos"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0], **img_in}}
    g["neg"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": negative, "vae": ["vae", 0], **img_in}}
    g["pos_m"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"}}
    g["neg_m"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}}
    g["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": ["cfgn", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
        "positive": ["pos_m", 0], "negative": ["neg_m", 0], "latent_image": ["latent", 0], "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}}
    return g


def wf_illustrious(image: str | None, prompt: str, negative: str, *, width: int = 1344, height: int = 768, seed: int = 0,
                   steps: int = 26, cfg: float = 5.5, denoise: float = 0.7, sampler: str = "euler_ancestral", scheduler: str = "normal",
                   loras: list[tuple[str, float]] | None = None, out_size: tuple[int, int] | None = None, prefix: str = "kf",
                   ckpt: str = ILL_CKPT, control_image: str | None = None, control_strength: float = 0.5, control_end: float = 0.6,
                   control_type: str = "canny/lineart/anime_lineart/mlsd",
                   regions: list[tuple[str, str]] | None = None, region_strength: float = 1.0) -> dict:
    """Illustrious-XL keyframe: img2img from `image` (the manga panel crop as layout reference) or txt2img when image is None.

    `regions` = [(mask image name, prompt)]: one prompt per person, applied only inside that person's part of the frame, so the
    features of two characters do not mix (the main `prompt` then carries what the whole frame shares).

    Settings from the model pages: Euler a, CFG 5-6, clip skip 1 (default); screencap LoRA at 1.0, detail slider as given."""
    g: dict = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    model, clip = ["ckpt", 0], ["ckpt", 1]
    for i, (name, strength) in enumerate(loras if loras is not None else [(ILL_LORA_SCREENCAP, 1.0), (ILL_LORA_DETAIL, 1.5)]):
        g[f"lora{i}"] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                                "strength_model": strength, "strength_clip": strength}}
        model, clip = [f"lora{i}", 0], [f"lora{i}", 1]
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": negative}}
    if image is not None:
        g["img"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g["fit"] = {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        g["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["fit", 0], "vae": ["ckpt", 2]}}
    else:
        g["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        denoise = 1.0
    pos, neg = ["pos", 0], ["neg", 0]
    for i, (mask_name, text) in enumerate(regions or []):
        g[f"rm{i}"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
        g[f"rt{i}"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": text}}
        g[f"rc{i}"] = {"class_type": "ConditioningSetMask", "inputs": {"conditioning": [f"rt{i}", 0], "mask": [f"rm{i}", 0],
                                                                         "strength": region_strength, "set_cond_area": "default"}}
        g[f"rj{i}"] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": pos, "conditioning_2": [f"rc{i}", 0]}}
        pos = [f"rj{i}", 0]
    if control_image is not None and control_strength > 0:
        # line-art ControlNet: the panel's line work steers who stands where and how, for the first part of the sampling only
        g["cn"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": SDXL_CN_UNION}}
        g["cnt"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["cn", 0], "type": control_type}}
        g["cimg"] = {"class_type": "LoadImage", "inputs": {"image": control_image}}
        g["cfit"] = {"class_type": "ImageScale", "inputs": {"image": ["cimg", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        g["cna"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": pos, "negative": neg, "control_net": ["cnt", 0], "image": ["cfit", 0], "strength": control_strength,
            "start_percent": 0.0, "end_percent": control_end, "vae": ["ckpt", 2]}}
        pos, neg = ["cna", 0], ["cna", 1]
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": model, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
        "positive": pos, "negative": neg, "latent_image": ["latent", 0], "denoise": denoise}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}}
    out = ["dec", 0]
    if out_size:
        g["out"] = {"class_type": "ImageScale", "inputs": {"image": out, "upscale_method": "lanczos", "width": out_size[0], "height": out_size[1], "crop": "center"}}
        out = ["out", 0]
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": prefix}}
    return g


def wf_illustrious_inpaint(image: str, mask: str, prompt: str, negative: str, *, width: int = 1344, height: int = 768, seed: int = 0,
                           steps: int = 26, cfg: float = 5.5, denoise: float = 0.8, loras: list[tuple[str, float]] | None = None,
                           out_size: tuple[int, int] | None = None, prefix: str = "kf_fix", ckpt: str = ILL_CKPT,
                           control_image: str | None = None, control_strength: float = 0.5, control_end: float = 0.6,
                           control_type: str = "canny/lineart/anime_lineart/mlsd") -> dict:
    """Redraw ONE part of an existing frame (white area of `mask`) with its own prompt: used to redraw one person of a
    multi-person frame from a single-character prompt, so the features of the other people cannot leak into it."""
    g: dict = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    model, clip = ["ckpt", 0], ["ckpt", 1]
    for i, (name, strength) in enumerate(loras if loras is not None else [(ILL_LORA_SCREENCAP, 1.0), (ILL_LORA_DETAIL, 1.5)]):
        g[f"lora{i}"] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                                "strength_model": strength, "strength_clip": strength}}
        model, clip = [f"lora{i}", 0], [f"lora{i}", 1]
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": negative}}
    g["img"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["fit"] = {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["fit", 0], "vae": ["ckpt", 2]}}
    g["m"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
    g["latent"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["m", 0]}}
    pos, neg = ["pos", 0], ["neg", 0]
    if control_image is not None and control_strength > 0:
        g["cn"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": SDXL_CN_UNION}}
        g["cnt"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["cn", 0], "type": control_type}}
        g["cimg"] = {"class_type": "LoadImage", "inputs": {"image": control_image}}
        g["cfit"] = {"class_type": "ImageScale", "inputs": {"image": ["cimg", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        g["cna"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": pos, "negative": neg, "control_net": ["cnt", 0], "image": ["cfit", 0], "strength": control_strength,
            "start_percent": 0.0, "end_percent": control_end, "vae": ["ckpt", 2]}}
        pos, neg = ["cna", 0], ["cna", 1]
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": model, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral", "scheduler": "normal",
        "positive": pos, "negative": neg, "latent_image": ["latent", 0], "denoise": denoise}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}}
    out = ["dec", 0]
    if out_size:
        g["out"] = {"class_type": "ImageScale", "inputs": {"image": out, "upscale_method": "lanczos", "width": out_size[0], "height": out_size[1], "crop": "center"}}
        out = ["out", 0]
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": prefix}}
    return g


def wf_wan_i2v(start_image: str, prompt: str, *, width: int = 832, height: int = 480, length: int = 81, seed: int = 0,
               steps: int = 4, split: int = 2, cfg: float = 1.0, shift: float = 5.0, negative: str = WAN_NEG,
               loras_high: list[tuple[str, float]] | None = None, loras_low: list[tuple[str, float]] | None = None,
               use_lightx2v: bool = True, last_image: str | None = None, prefix: str = "wan", fps: int = 16, crf: int = 10) -> dict:
    """Wan 2.2 I2V A14B mixture-of-experts: high-noise expert for steps [0, split), low-noise for [split, steps)."""
    assert (length - 1) % 4 == 0, "Wan length must be 4k+1"
    g: dict = {
        "unet_h": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN_HIGH, "weight_dtype": "default"}},
        "unet_l": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN_LOW, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_T5, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": start_image}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}},
    }

    def chain(tag: str, base: str, loras: list[tuple[str, float]]):
        model = [base, 0]
        for i, (name, strength) in enumerate(loras):
            g[f"lora_{tag}{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
            model = [f"lora_{tag}{i}", 0]
        g[f"ms_{tag}"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": shift}}
        return [f"ms_{tag}", 0]

    mh = chain("h", "unet_h", ([(LIGHT_HIGH, 1.0)] if use_lightx2v else []) + list(loras_high or []))
    ml = chain("l", "unet_l", ([(LIGHT_LOW, 1.0)] if use_lightx2v else []) + list(loras_low or []))
    if last_image:
        g["img_last"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
        g["i2v"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": width, "height": height,
            "length": length, "batch_size": 1, "start_image": ["img", 0], "end_image": ["img_last", 0]}}
    else:
        g["i2v"] = {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": width, "height": height,
            "length": length, "batch_size": 1, "start_image": ["img", 0]}}
    common = {"noise_seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["i2v", 0], "negative": ["i2v", 1]}
    g["ks_h"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": mh, "add_noise": "enable", "latent_image": ["i2v", 2],
                                                               "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}}
    g["ks_l"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ml, "add_noise": "disable", "latent_image": ["ks_h", 0],
                                                               "start_at_step": split, "end_at_step": steps, "return_with_leftover_noise": "disable"}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks_l", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["dec", 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": prefix, "format": "video/h264-mp4",
        "pix_fmt": "yuv420p", "crf": crf, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}}
    return g


def wf_wan_stage(stage: str, start_image: str, prompt: str, *, width: int = 832, height: int = 480, length: int = 81, seed: int = 0,
                 steps: int = 4, split: int = 2, cfg: float = 1.0, shift: float = 5.0, negative: str = WAN_NEG,
                 loras_high: list[tuple[str, float]] | None = None, loras_low: list[tuple[str, float]] | None = None,
                 use_lightx2v: bool = True, latent_name: str | None = None, prefix: str = "wan", fps: int = 16, crf: int = 10,
                 clip_device: str = "default", start_step: int | None = None, end_step: int | None = None) -> dict:
    """One half of the mixture-of-experts sampler, so a batch of shots can share one model load:
    stage "high": steps [0, split) with the high-noise expert -> SaveLatent
    stage "low" : LoadLatent(latent_name) -> steps [split, steps) with the low-noise expert -> decode -> mp4
    A "high" stage can be cut in two with start_step / end_step: a first pass [0, k) without the distill LoRA and with real
    guidance (cfg > 1), then a pass [k, split) that continues its latent (latent_name) with the LoRA - the usual remedy for
    the small, slow motion of the 4-step distilled sampler (motion test, D-057).
    Text encoder on the GPU next to the expert (6.4 + 13.9 GB fit; measured 17 s vs 20-35 s per 53-frame pass with the encoder on the CPU)."""
    assert stage in ("high", "low") and (length - 1) % 4 == 0
    g: dict = {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_T5, "type": "wan", "device": clip_device}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": start_image}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}},
        "i2v": {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": width, "height": height,
            "length": length, "batch_size": 1, "start_image": ["img", 0]}},
    }
    unet, light, extra = (WAN_HIGH, LIGHT_HIGH, loras_high) if stage == "high" else (WAN_LOW, LIGHT_LOW, loras_low)
    g["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    model = ["unet", 0]
    for i, (name, strength) in enumerate(([(light, 1.0)] if use_lightx2v else []) + list(extra or [])):
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [f"lora{i}", 0]
    g["ms"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": shift}}
    common = {"model": ["ms", 0], "noise_seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["i2v", 0], "negative": ["i2v", 1]}
    if stage == "high":
        a, b = (0 if start_step is None else start_step), (split if end_step is None else end_step)
        if latent_name:
            g["lat"] = {"class_type": "LoadLatent", "inputs": {"latent": latent_name}}
            g["ks"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "add_noise": "disable", "latent_image": ["lat", 0],
                                                                     "start_at_step": a, "end_at_step": b, "return_with_leftover_noise": "enable"}}
        else:
            g["ks"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "add_noise": "enable", "latent_image": ["i2v", 2],
                                                                     "start_at_step": a, "end_at_step": b, "return_with_leftover_noise": "enable"}}
        g["save"] = {"class_type": "SaveLatent", "inputs": {"samples": ["ks", 0], "filename_prefix": prefix}}
    else:
        g["lat"] = {"class_type": "LoadLatent", "inputs": {"latent": latent_name}}
        g["ks"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "add_noise": "disable", "latent_image": ["lat", 0],
                                                                 "start_at_step": split, "end_at_step": steps, "return_with_leftover_noise": "disable"}}
        g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
        g["save"] = {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["dec", 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": prefix, "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": crf, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}}
    return g


def dump_reference_workflows() -> None:
    """Write the current graphs to workflows/ so they are versioned in-repo."""
    out = Path(CFG.paths.repo) / "workflows"
    out.mkdir(exist_ok=True)
    (out / "qwen_edit_2511_keyframe.api.json").write_text(json.dumps(
        wf_qwen_edit(["PANEL.png", "REF1.png", "REF2.png"], "PROMPT", seed=0), indent=1))
    (out / "wan22_i2v_lightx2v.api.json").write_text(json.dumps(
        wf_wan_i2v("KEYFRAME.png", "PROMPT", seed=0, loras_high=[("KOMI_HIGH.safetensors", 1.0)], loras_low=[("KOMI_LOW.safetensors", 1.0)]),
        indent=1, ensure_ascii=False))
    (out / "wan22_flf2v_lightx2v.api.json").write_text(json.dumps(
        wf_wan_i2v("KEYFRAME.png", "PROMPT", seed=0, last_image="LAST.png"), indent=1, ensure_ascii=False))
    (out / "wan22_i2v_stage_high.api.json").write_text(json.dumps(
        wf_wan_stage("high", "KEYFRAME.png", "PROMPT", seed=0, prefix="latents/SHOT"), indent=1, ensure_ascii=False))
    (out / "wan22_i2v_stage_low.api.json").write_text(json.dumps(
        wf_wan_stage("low", "KEYFRAME.png", "PROMPT", seed=0, latent_name="SHOT.latent", prefix="SHOT"), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    dump_reference_workflows()
    print("workflows written")
