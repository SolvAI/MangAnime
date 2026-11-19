"""Single-speaker fine-tune of Qwen3-TTS 1.7B on one dub voice (the "custom voice" route to voice + `instruct`).

Adapted from QwenLM/Qwen3-TTS `finetuning/{prepare_data,dataset,sft_12hz}.py` (Apache-2.0; repo copy in
komi-poc-work/third_party/Qwen3-TTS, commit 022e286). Differences from the upstream script:
  * the training set is cut from the English voice data (lines_en.jsonl + vocals_en.wav) with the voice-bank
    cleanliness rules (one speaker, no neighbour line, SNR proxy, ECAPA consistency to the bank centroid);
  * ONE checkpoint is written at the end (upstream copies the whole 4.3 GB model directory after every epoch); the
    speech tokenizer is symlinked, not copied; a wall-clock budget stops training early;
  * no flash-attn (not installed), no tensorboard;
  * THE LOSS IS BUILT DIFFERENTLY. Measured on the released Base weights with this speaker's clips (qwen-tts 0.1.1,
    transformers 4.57.3): the published construction starts at a talker loss of 10.6 (chance level), because (1) it
    feeds raw text embeddings where inference feeds `talker.text_projection(...)`, (2) it shifts the labels by hand
    and the library loss (ForCausalLMLoss) shifts them again, so logits[t] are scored against token t+2, and (3) the
    sub-talker gets the hidden state of the frame's own position instead of the one that predicted the frame. With
    projected text embeddings, a plain next-token cross-entropy and the predicting hidden state the same weights
    start at talker CE 1.06 / sub-talker CE 5.5: that is the default here; `--upstream-objective` reproduces the
    published construction for comparison;
  * `--init custom_voice` fine-tunes the instruction-following CustomVoice weights instead of Base (experimental; the
    speaker embedding then comes from the Base speaker encoder via src.generate.tts_qwen.XVector).
What a "custom voice" is in this model: the speaker's x-vector written into row 3000 of the talker's codec embedding
table (config.talker_config.spk_id = {<speaker>: 3000}, tts_model_type = custom_voice) + the fine-tuned talker.

    . scripts/env.sh ; W=$KOMI_WORK
    # 1) data (TTS venv: ECAPA + Whisper stack; CPU ok)          -> runs/qwen_sft/<Speaker>/{clips/*.wav,train_raw.jsonl}
    $W/venvs/tts/bin/python -m src.train.qwen_sft data --speaker Tadano --max-min 12
    # 2) codec tokens (qwentts venv, ~2 GB)                       -> train_codes.jsonl
    scripts/gpu_run.sh 3 $W/venvs/qwentts/bin/python -m src.train.qwen_sft codes --speaker Tadano
    # 3) train (qwentts venv; full fine-tune in bf16: see docs/tts_qwen.md for the measured VRAM)
    scripts/gpu_run.sh 20 $W/venvs/qwentts/bin/python -m src.train.qwen_sft train --speaker Tadano --epochs 6 --max-minutes 20
    #    -> models/tts/qwen3/ft/<Speaker>/ ; register it under tts_qwen.finetuned.<Speaker> to use mode "custom"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from src.common.config import CFG
from src.common.io import log, read_json, read_jsonl, write_json, write_jsonl

Q = CFG.tts_qwen
SR = 24000


def work_dir(speaker: str) -> Path:
    return Path(CFG.paths.runs) / "qwen_sft" / speaker.replace(" ", "_").replace("'", "_")


# ============================================================================================ 1. data
def build_data(speaker: str, max_min: float = 12.0, min_sec: float = 1.5, min_conf: float = 0.6,
               min_consistency: float = 0.45, device: str = "cpu") -> dict:
    """Clean single-speaker clips of `speaker` from TRAIN episodes -> 24 kHz wavs + train_raw.jsonl."""
    import numpy as np

    from src.eval.voice_metrics import SpeakerEmbedder
    from src.generate.tts import fade, peak_norm, resample, trim_silence, write_wav
    from src.train import voice_bank as vb

    vb.use_lang("en")
    V = vb.V
    train = {int(e) for e in CFG.split.train_episodes}
    lines = [r for r in read_jsonl(V.lines) if int(r["ep"]) in train]
    idx = read_json(Path(V.out_dir) / "voices.json", default={}) or {}
    bank = (idx.get("speakers") or {}).get(speaker)
    if not bank:
        raise SystemExit(f"{speaker} is not in the English voice bank ({V.out_dir}); build it first")
    centroid = np.load(bank["centroid"])
    ref_files = set(bank["refs"])
    ref_meta = read_json(Path(bank["dir"]) / "meta.json")
    ref_spans = [(r["ep"], r["start"], r["end"]) for r in ref_meta["refs"]]
    # candidates: single lines and merged runs, as in the voice bank but with a lower floor and stricter labels
    old = (V.min_sec, V.min_conf)
    V.min_sec, V.min_conf = float(min_sec), float(min_conf)
    try:
        cands = vb.build_candidates(lines, vb._load_other_intervals({int(r["ep"]) for r in lines})).get(speaker, [])
    finally:
        V.min_sec, V.min_conf = old
    cands.sort(key=lambda c: (-c["n_lines"], -c["dur"]))                # longest runs first, then greedy non-overlap
    emb = SpeakerEmbedder(device)
    out = work_dir(speaker)
    clips = out / "clips"
    if clips.exists():
        shutil.rmtree(clips)
    clips.mkdir(parents=True)
    taken: list[tuple[int, float, float]] = []
    rows, total, rejected = [], 0.0, {"overlap": 0, "short": 0, "dirty": 0, "inconsistent": 0, "is_ref": 0}
    for c in cands:
        if total >= max_min * 60:
            break
        if any(e == c["ep"] and c["start"] < b and c["end"] > a for e, a, b in taken):
            rejected["overlap"] += 1
            continue
        if any(e == c["ep"] and c["start"] < b and c["end"] > a for e, a, b in ref_spans):
            rejected["is_ref"] += 1          # the prompt/reference clips stay out of the training set
            continue
        stem, kind = vb._stem_for(c["ep"], Path(V.audio_dir))
        if stem is None:
            continue
        x, sr = vb._read_segment(stem, c["start"] - float(V.edge_pad_sec), c["end"] + float(V.edge_pad_sec))
        if x is None:
            continue
        x = trim_silence(x, sr, top_db=35.0, pad_head_ms=40, pad_tail_ms=80)
        d = len(x) / sr
        if d < min_sec * 0.8:
            rejected["short"] += 1
            continue
        f = vb.audio_features(x, sr)
        if f["speech_ratio"] < float(V.min_speech_ratio) or f["snr_db"] < float(V.min_snr_db) or f["clip_frac"] > 0.002:
            rejected["dirty"] += 1
            continue
        cos = float(np.dot(emb.embed_wave(x, sr), centroid))
        if cos < min_consistency:
            rejected["inconsistent"] += 1
            continue
        path = clips / f"ep{c['ep']:02d}_{int(c['start'] * 100):07d}.wav"
        write_wav(path, fade(peak_norm(resample(x, sr, SR), -3.0), SR, 8, 30), SR)
        rows.append({"audio": str(path), "text": " ".join(str(c["text"]).split()), "ref_audio": bank["default_refs"][0],
                     "ep": c["ep"], "start": c["start"], "end": c["end"], "dur": round(d, 3), "consistency": round(cos, 4),
                     "snr_db": f["snr_db"]})
        taken.append((c["ep"], c["start"], c["end"]))
        total += d
    assert all(int(r["ep"]) in train for r in rows), "held-out episode leaked into the fine-tuning set"
    assert not any(r["audio"] in ref_files for r in rows)
    write_jsonl(out / "train_raw.jsonl", rows)
    st = {"speaker": speaker, "clips": len(rows), "minutes": round(total / 60, 2), "candidates": len(cands), "rejected": rejected,
          "episodes": sorted({r["ep"] for r in rows}), "mean_consistency": round(float(np.mean([r["consistency"] for r in rows])), 4) if rows else None,
          "ref_audio": bank["default_refs"][0], "xvec_refs": bank["default_refs"]}
    write_json(out / "data.json", st)
    log(f"fine-tuning data for {speaker}: {st}")
    return st


# ============================================================================================ 2. codec tokens
def encode_codes(speaker: str, device: str = "cuda:0", batch: int = 16) -> None:
    from qwen_tts import Qwen3TTSTokenizer

    out = work_dir(speaker)
    rows = read_jsonl(out / "train_raw.jsonl")
    tok = Qwen3TTSTokenizer.from_pretrained(str(Path(Q.models.base) / "speech_tokenizer"), device_map=device)
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        enc = tok.encode([r["audio"] for r in part])
        for r, code in zip(part, enc.audio_codes):
            r["audio_codes"] = code.cpu().tolist()
    write_jsonl(out / "train_codes.jsonl", rows)
    log(f"codes for {len(rows)} clips -> {out / 'train_codes.jsonl'}")


# ============================================================================================ 3. train
def collate(batch: list[dict], cfg) -> dict:
    """Upstream finetuning/dataset.py::TTSDataset.collate_fn (dual text / codec channel layout), unchanged logic."""
    import torch

    tc = cfg.talker_config
    lens = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
    b, t = len(batch), max(lens) + 8
    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
    text_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_emb_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attn = torch.zeros((b, t), dtype=torch.long)
    labels = torch.full((b, t), -100, dtype=torch.long)
    for i, d in enumerate(batch):
        text_ids, codes = d["text_ids"], d["audio_codes"]
        c0 = codes[:, 0]
        tl, cl = text_ids.shape[1], c0.shape[0]
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = cfg.tts_pad_token_id
        input_ids[i, 7, 0] = cfg.tts_bos_token_id
        input_ids[i, 8:8 + tl - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + tl - 3, 0] = cfg.tts_eos_token_id
        input_ids[i, 8 + tl - 2:8 + tl + cl, 0] = cfg.tts_pad_token_id
        text_mask[i, :8 + tl + cl] = True
        input_ids[i, 3:8, 1] = torch.tensor([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id, 0, tc.codec_pad_id])
        input_ids[i, 8:8 + tl - 3, 1] = tc.codec_pad_id
        input_ids[i, 8 + tl - 3, 1] = tc.codec_pad_id
        input_ids[i, 8 + tl - 2, 1] = tc.codec_bos_id
        input_ids[i, 8 + tl - 1:8 + tl - 1 + cl, 1] = c0
        input_ids[i, 8 + tl - 1 + cl, 1] = tc.codec_eos_token_id
        labels[i, 8 + tl - 1:8 + tl - 1 + cl] = c0
        labels[i, 8 + tl - 1 + cl] = tc.codec_eos_token_id
        codec_ids[i, 8 + tl - 1:8 + tl - 1 + cl, :] = codes
        codec_emb_mask[i, 3:8 + tl + cl] = True
        codec_emb_mask[i, 6] = False                  # position of the speaker embedding
        codec_mask[i, 8 + tl - 1:8 + tl - 1 + cl] = True
        attn[i, :8 + tl + cl] = True
    return {"input_ids": input_ids, "attention_mask": attn, "text_embedding_mask": text_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_emb_mask.unsqueeze(-1), "codec_0_labels": labels, "codec_ids": codec_ids,
            "codec_mask": codec_mask}


def train(speaker: str, init: str = "base", epochs: int = 6, lr: float = 2e-5, batch_size: int = 4, grad_accum: int = 2,
          max_minutes: float = 20.0, tag: str = "", seed: int = 0, freeze_text_embedding: bool = False,
          freeze_layers: int = 0, vram_cap_gb: float | None = None, upstream_objective: bool = False) -> dict:
    """freeze_text_embedding / freeze_layers (the lowest N of the 28 talker layers) shrink the optimizer state so the
    job fits beside other work on the shared card: a full fine-tune holds 1.93 B parameters x (2 weight + 2 gradient +
    4 AdamW) bytes = ~15.4 GB before activations. vram_cap_gb hard-caps the torch allocator of this process."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    from src.generate.tts_qwen import XVector

    torch.manual_seed(seed)
    if vram_cap_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, (float(vram_cap_gb) - 0.6) * 2**30 / total), 0)
    wd = work_dir(speaker)
    data = read_jsonl(wd / "train_codes.jsonl")
    info = read_json(wd / "data.json")
    init_path = Path(Q.models[init])
    out_dir = Path(Q.model_root) / "ft" / (speaker.replace(" ", "_").replace("'", "_") + (f"_{tag}" if tag else ""))
    t_start = time.time()
    acc = Accelerator(gradient_accumulation_steps=int(grad_accum), mixed_precision="bf16")
    tts = Qwen3TTSModel.from_pretrained(str(init_path), torch_dtype=torch.bfloat16)
    cfg = AutoConfig.from_pretrained(str(init_path))
    proc = tts.processor
    # one constant speaker embedding (mean x-vector of the bank's default refs, Base speaker encoder)
    xvec = torch.from_numpy(XVector("cpu").mean(list(info["xvec_refs"])))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            it = data[i]
            text = f"<|im_start|>assistant\n{it['text']}<|im_end|>\n<|im_start|>assistant\n"
            ids = proc(text=text, return_tensors="pt", padding=True)["input_ids"]
            ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
            return {"text_ids": ids[:, :-5], "audio_codes": torch.tensor(it["audio_codes"], dtype=torch.long)}

    dl = DataLoader(DS(), batch_size=int(batch_size), shuffle=True, collate_fn=lambda b: collate(b, cfg), drop_last=False)
    frozen = 0
    for name, prm in tts.model.named_parameters():
        low_layer = name.startswith("talker.model.layers.") and int(name.split(".")[3]) < int(freeze_layers)
        if name.startswith("speaker_encoder.") or (freeze_text_embedding and name.startswith("talker.model.text_embedding.")) or low_layer:
            prm.requires_grad_(False)
            frozen += prm.numel()
    trainable = [prm for prm in tts.model.parameters() if prm.requires_grad]
    n_train = sum(prm.numel() for prm in trainable)
    log(f"trainable parameters: {n_train / 1e6:.0f} M (frozen {frozen / 1e6:.0f} M: text embedding {freeze_text_embedding}, "
        f"lowest {freeze_layers} talker layers)")
    # foreach=False: the default multi-tensor AdamW step allocates temporaries as large as all trainable parameters
    # (measured: +3 GB at 800 M trainable parameters, which did not fit beside the pipeline's ComfyUI server)
    opt = torch.optim.AdamW(trainable, lr=float(lr), weight_decay=0.01, foreach=False)
    model, opt, dl = acc.prepare(tts.model, opt, dl)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    hist, step, stopped = [], 0, None
    for epoch in range(int(epochs)):
        ep_loss, ep_talker, ep_sub = [], [], []
        for batch in dl:
            with acc.accumulate(model):
                input_ids, codec_ids, codec_mask = batch["input_ids"], batch["codec_ids"], batch["codec_mask"]
                labels = batch["codec_0_labels"]
                text_emb = model.talker.model.text_embedding(input_ids[:, :, 0])
                if not upstream_objective:
                    text_emb = model.talker.text_projection(text_emb)     # inference feeds PROJECTED text embeddings
                text_emb = text_emb * batch["text_embedding_mask"]
                codec_emb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"]
                codec_emb[:, 6, :] = xvec.to(codec_emb.device).to(codec_emb.dtype)
                emb = text_emb + codec_emb
                for i in range(1, 16):
                    e_i = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    emb = emb + e_i * codec_mask.unsqueeze(-1)
                if freeze_layers > 0:
                    # the input embeddings feed the frozen lower layers: without this cut autograd keeps the activations
                    # of all 28 layers (measured: > 5 GB at batch 4) just to reach text_projection / codec embeddings
                    emb = emb.detach()
                if upstream_objective:      # the published script, verbatim: library losses (which shift the labels again)
                    o = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
                                     labels=labels[:, 1:], output_hidden_states=True)
                    hs = o.hidden_states[0][-1][codec_mask[:, :-1]]
                    _, sub_loss = model.talker.forward_sub_talker_finetune(codec_ids[codec_mask], hs)
                    talker_loss = o.loss
                else:                       # the objective the released weights were trained with (measured: docs/tts_qwen.md)
                    o = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
                                     output_hidden_states=True)
                    lg = o.logits.float()
                    talker_loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
                    cids = codec_ids[codec_mask]
                    hs = o.hidden_states[0][-1][codec_mask[:, 1:]]        # the state that PREDICTED each frame
                    sl, _ = model.talker.forward_sub_talker_finetune(cids, hs)
                    sub_loss = F.cross_entropy(sl.float().reshape(-1, sl.shape[-1]), cids[:, 1:].reshape(-1))
                loss = talker_loss + 0.3 * sub_loss
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
            ep_loss.append(float(loss.item()))
            ep_talker.append(float(talker_loss.item()))
            ep_sub.append(float(sub_loss.item()))
            if step % 20 == 0:
                log(f"epoch {epoch} step {step} loss {loss.item():.4f} (talker {talker_loss.item():.4f}, sub {sub_loss.item():.4f}) "
                    f"elapsed {(time.time() - t_start) / 60:.1f} min")
            step += 1
            if (time.time() - t_start) / 60 > max_minutes:
                stopped = f"time budget {max_minutes} min reached in epoch {epoch}"
                break
        hist.append({"epoch": epoch, "mean_loss": round(float(np.mean(ep_loss)), 4), "talker_ce": round(float(np.mean(ep_talker)), 4),
                     "sub_ce": round(float(np.mean(ep_sub)), 4), "steps": len(ep_loss)})
        log(f"epoch {epoch} done: mean loss {hist[-1]['mean_loss']}")
        if stopped:
            break
    peak = {"allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    # ---- one checkpoint: config (custom_voice + spk_id), small files copied, tokenizer symlinked, weights saved
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for f in init_path.iterdir():
        if f.name in ("model.safetensors", "config.json", "README.md", ".cache"):
            continue
        if f.is_dir():
            os.symlink(f.resolve(), out_dir / f.name)
        else:
            shutil.copy2(f, out_dir / f.name)
    cd = json.loads((init_path / "config.json").read_text())
    name = speaker.lower()
    cd["tts_model_type"] = "custom_voice"
    cd.setdefault("talker_config", {})
    cd["talker_config"]["spk_id"] = {name: 3000}
    cd["talker_config"]["spk_is_dialect"] = {name: False}
    (out_dir / "config.json").write_text(json.dumps(cd, indent=2, ensure_ascii=False))
    sd = {k: v.detach().to("cpu") for k, v in acc.unwrap_model(model).state_dict().items() if not k.startswith("speaker_encoder")}
    w = sd["talker.model.codec_embedding.weight"]
    w[3000] = xvec.to(w.dtype)
    save_file(sd, str(out_dir / "model.safetensors"))
    res = {"speaker": speaker, "speaker_id": name, "init": init, "checkpoint": str(out_dir), "epochs_run": len(hist), "history": hist,
           "optimizer_steps": step // int(grad_accum), "micro_steps": step, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
           "clips": len(data), "minutes_audio": info.get("minutes"), "train_minutes": round((time.time() - t_start) / 60, 2),
           "stopped_early": stopped, "vram_peak": peak, "precision": "bf16 weights + bf16 autocast (upstream recipe)",
           "trainable_params_m": round(n_train / 1e6, 1), "frozen_params_m": round(frozen / 1e6, 1),
           "freeze_text_embedding": bool(freeze_text_embedding), "freeze_layers": int(freeze_layers),
           "objective": "upstream script verbatim (raw text embedding, labels shifted twice, same-position hidden for the sub-talker)"
                        if upstream_objective else "corrected (projected text embedding, next-token CE, predicting hidden for the sub-talker)"}
    write_json(out_dir / "sft.json", res)
    log(f"fine-tune done: {res}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["data", "codes", "train"])
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--max-min", type=float, default=12.0, help="data: minutes of audio to collect")
    ap.add_argument("--device", default=None)
    ap.add_argument("--init", default="base", choices=["base", "custom_voice"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--freeze-text-embedding", action="store_true")
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze the lowest N talker layers (of 28)")
    ap.add_argument("--vram-cap-gb", type=float, default=None)
    ap.add_argument("--upstream-objective", action="store_true",
                    help="reproduce the loss construction of the published sft_12hz.py verbatim (for comparison only)")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if a.cmd == "data":
        if (a.device or "cpu") == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        build_data(a.speaker, a.max_min, device=a.device or "cpu")
    elif a.cmd == "codes":
        encode_codes(a.speaker, a.device or "cuda:0")
    else:
        train(a.speaker, a.init, a.epochs, a.lr, a.batch_size, a.grad_accum, a.max_minutes, a.tag,
              freeze_text_embedding=a.freeze_text_embedding, freeze_layers=a.freeze_layers, vram_cap_gb=a.vram_cap_gb,
              upstream_objective=a.upstream_objective)


if __name__ == "__main__":
    main()
