"""Measured probe: does an instruction (or an inline tag, or the choice of reference clip) change the delivery of a
cloned English voice, and what does it cost in speaker similarity and intelligibility? (docs/tts_qwen.md section 4)

Design: the same test sentences x {neutral, excited, sad, angry, whispering, nervous} per speaker and per METHOD; every
delivery number is compared with the SAME method's neutral take of the SAME sentence (paired), and a second neutral
take with another seed gives the noise floor. A condition "works" only if the expected delivery features move
(consistently across sentences and by more than a fixed threshold) while the similarity to the speaker stays within
0.05 of the neutral clone and the words stay intelligible.

Methods (how the instruction reaches the model):
  icl            Base, reference clip + transcript in context, NO instruction            (the neutral clone baseline)
  icl_inst       Base, in-context clone + the instruction passed as `instruct_ids`        (experiment a)
  xvec_inst      Base, speaker-embedding clone + instruction                              (experiment a, no prosody anchor)
  cv_xvec        CustomVoice weights + Base speaker embedding of the refs + instruction   (zero-training hybrid)
  cv_icl         CustomVoice weights + reference in context + instruction
  ft             fine-tuned single-speaker checkpoint (generate_custom_voice + instruct)  (experiment c)
  ref            Base in-context clone, NO instruction, reference picked by arousal       (experiment d)
  preset         CustomVoice preset speaker + instruction        (positive control: the path the model was trained for)
  tags_<method>  inline tag at the start of the text, e.g. "[excited] ..." , no instruction (experiment b)

    . scripts/env.sh ; W=$KOMI_WORK ; D=$W/runs/en_voice/probe/main
    $W/venvs/tts/bin/python -m src.eval.emotion_probe items --speakers Tadano,Najimi --methods icl,icl_inst,cv_xvec --out $D
    scripts/gpu_run.sh 8 $W/venvs/qwentts/bin/python -m src.generate.tts_qwen --in $D/items.jsonl --out-dir $D --batch-size 8
    scripts/gpu_run.sh 6 $W/venvs/tts/bin/python -m src.eval.emotion_probe measure --dir $D --device cuda
    $W/venvs/tts/bin/python -m src.eval.emotion_probe report --dir $D
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

from src.common.config import CFG
from src.common.io import log, read_json, read_jsonl, write_json, write_jsonl

Q = CFG.tts_qwen
SENTENCES = [
    "I didn't expect to see you here today.",
    "We should go to the library after class.",
    "Komi wrote something in her notebook again.",
    "The festival starts tomorrow morning at the school gate.",
    "Please tell me what you really think about it.",
    "I left my bag in the classroom on the second floor.",
    "Everyone was already waiting when we arrived.",
    "It's getting late, so let's walk home together.",
]
INSTRUCT = {
    "neutral": None,
    "excited": "Very excited and energetic, speaking fast with a bright, high-pitched voice.",
    "sad": "Sad and dejected, speaking slowly in a low, quiet voice.",
    "angry": "Angry and shouting, loud and forceful.",
    "whispering": "Whispering very quietly, almost inaudible.",
    "nervous": "Nervous and hesitant, stammering slightly with a shaky voice.",
}
TAGS = {"neutral": "", "excited": "[excited]", "sad": "[sad]", "angry": "[angry]", "whispering": "[whispers]",
        "nervous": "[nervous]", "sighing": "[sighing]", "laughing": "[laughing]"}
TAG_WORDS = {"excited", "sad", "angry", "whispers", "whisper", "whispering", "nervous", "sighing", "sigh", "laughing", "laugh"}
STYLE_OF = {"neutral": None, "excited": "high", "angry": "high", "sad": "low", "whispering": "low", "nervous": "low"}
# Expected direction of the paired difference to neutral (+1 up, -1 down) and the size that counts as a move.
# History (kept honest): the first version of this table, written before any audio existed, also expected the LEVEL
# to rise (excited, angry) or fall (sad, whispering) and used smaller thresholds (F0 1 st, range 1.5 st, voiced 0.10).
# The positive control (CustomVoice preset speakers + instruction, the path the model was trained for) showed that the
# output level is not controlled by instructions at all (sad came out LOUDER for one preset, quieter for the other;
# a mere seed change moves the mean level by 0.6 dB and the mean F0 by 1.1 st with 12 of 16 sentences agreeing), so the
# level was dropped from the criteria and the thresholds were raised above that seed noise. The level is still reported.
EXPECT = {
    "excited": {"f0_st": +1, "wps": +1},
    "sad": {"f0_st": -1, "f0_range_st": -1, "tilt_db": -1, "wps": -1},
    "angry": {"f0_st": +1, "tilt_db": +1, "wps": +1},
    "whispering": {"voiced_ratio": -1, "tilt_db": +1, "f0_range_st": -1},     # breathy, flat, less voicing
    "nervous": {"f0_st": +1, "wps": -1},               # weakly defined: tense (higher) and hesitant (slower)
}
MOVE = {"f0_st": 2.0, "f0_range_st": 3.0, "energy_db": 2.0, "wps": 0.3, "voiced_ratio": 0.06, "tilt_db": 1.5}
AGREE = 0.75           # share of sentence pairs that must move in the expected direction
FEATS = ["f0_mean", "f0_st", "f0_range_st", "energy_db", "wps", "voiced_ratio", "tilt_db"]
SIM_TOL, WER_MAX = 0.05, 0.20


def build_items(speakers: list[str], methods: list[str], out: Path, n_sent: int = 8, tag_sent: int = 4,
                preset: str = "ryan", neutral_repeat: bool = True) -> list[dict]:
    from src.generate.tts_qwen import voice_refs_en, voices_en

    bank = voices_en().get("speakers", {})
    items = []
    for spk in speakers:
        for m in methods:
            tags = m.startswith("tags_")
            base = m[5:] if tags else m
            conds = list(TAGS) if tags else list(INSTRUCT)
            if base in ("icl",) and not tags:
                conds = ["neutral"]
            if base == "ref":
                conds = ["neutral", "excited", "sad"]            # = default / high / low reference sets
            for cond in conds + (["neutral2"] if (neutral_repeat and not tags) else []):
                c = "neutral" if cond == "neutral2" else cond
                for k, sent in enumerate(SENTENCES[: (tag_sent if tags else n_sent)]):
                    it = {"id": f"{m}__{spk.replace(' ', '_')}__{cond}__s{k}", "speaker": spk, "lang": "EN",
                          "seed": 1000 + k + (500 if cond == "neutral2" else 0),
                          "meta": {"method": m, "cond": cond, "sent": k, "probe_speaker": spk}}
                    if tags:
                        t = TAGS[c]
                        it.update(text=(f"{t} {sent}" if c != "laughing" else f"{sent} {t}") if t else sent,
                                  strip_tags=False, instruct=None)
                        it["meta"]["plain_text"] = sent
                    else:
                        it.update(text=sent, instruct=None if base in ("icl", "ref") else INSTRUCT[c])
                    if base == "preset":
                        it.update(mode="custom", preset=preset, speaker=preset.capitalize())
                    else:
                        style = STYLE_OF.get(c) if base == "ref" else None
                        refs, texts, who = voice_refs_en(spk, style=style)
                        if base == "ref" and style and not (bank.get(spk, {}).get("style_refs") or {}).get(style):
                            continue
                        it.update(ref_wavs=refs[:3], ref_texts=texts[:3])
                        it["mode"] = {"icl": "icl", "icl_inst": "icl", "xvec_inst": "xvec", "cv_xvec": "cv_xvec",
                                      "cv_icl": "cv_icl", "ft": "custom", "ref": "icl"}[base]
                        if base in ("icl_inst", "xvec_inst"):
                            it["instruct_on_base"] = True
                    items.append(it)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "items.jsonl", items)
    log(f"{len(items)} probe items -> {out / 'items.jsonl'}")
    return items


def _prosody(args):
    path, text = args
    import soundfile as sf

    from src.eval.voice_metrics import prosody_features
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return path, prosody_features(x.mean(axis=1), sr, text=text)


def measure(d: Path, device: str = "cpu", workers: int = 8) -> list[dict]:
    """Per utterance: ECAPA similarity to the speaker (bank centroid + default refs), WER, delivery features."""
    import math
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    from src.eval.voice_metrics import SpeakerEmbedder, Transcriber, cos, mean_embedding, wer
    from src.generate.tts_qwen import voices_en

    rows = [r for r in read_jsonl(d / "tts_results.jsonl") if (r.get("meta") or {}).get("method")]
    bank = voices_en().get("speakers", {})
    emb = SpeakerEmbedder("cuda" if device.startswith("cuda") else "cpu")
    asr = Transcriber(device)
    yard: dict[str, dict] = {}
    for spk in sorted({r["meta"]["probe_speaker"] for r in rows}):
        b = bank.get(spk)
        if b:
            yard[spk] = {"centroid": np.load(b["centroid"]),
                         "default": mean_embedding([emb.embed_file(p) for p in b["default_refs"]])}
    ok = [r for r in rows if r.get("path") and Path(r["path"]).is_file()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pros = dict(ex.map(_prosody, [(r["path"], (r["meta"].get("plain_text") or r["text"])) for r in ok], chunksize=4))
    out = []
    for r in rows:
        m = r["meta"]
        it = {"id": r["id"], "method": m["method"], "cond": m["cond"], "sent": m["sent"], "speaker": m["probe_speaker"],
              "ok": bool(r.get("ok")), "error": r.get("error"), "dur": r.get("dur"), "takes": r.get("takes"),
              "synth_sec": r.get("synth_sec"), "energy_db": r.get("raw_rms_dbfs")}
        out.append(it)
        if r not in ok:
            continue
        plain = m.get("plain_text") or r["text"]
        g = emb.embed_file(r["path"])
        y = yard.get(m["probe_speaker"])
        if y is not None:
            it["sim_bank"] = round(cos(g, y["centroid"]), 4)
            it["sim_default"] = round(cos(g, y["default"]), 4)
        hyp = asr.transcribe(r["path"], language="en")
        w, we, wn = wer(plain, hyp)
        it.update(asr=hyp, wer=None if w is None else round(w, 4), wer_edits=we, wer_ref_len=wn)
        if m["method"].startswith("tags_"):
            said = set(re.findall(r"[a-z]+", hyp.lower())) - set(re.findall(r"[a-z]+", plain.lower()))
            it["tag_read_aloud"] = bool(said & TAG_WORDS)
        p = pros[r["path"]]
        it.update({k: p.get(k) for k in ("f0_mean", "f0_med", "f0_range_st", "voiced_ratio", "tilt_db", "wps", "speech_sec")})
        it["f0_st"] = round(12 * math.log2(p["f0_mean"] / 100.0), 3) if p.get("f0_mean") else None
    write_json(d / "probe_metrics.json", out)
    log(f"probe metrics for {len(out)} utterances -> {d / 'probe_metrics.json'}")
    return out


def _mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def summarise(items: list[dict]) -> dict:
    """(method, cond) -> means, paired deltas to the method's neutral, sign consistency, verdict."""
    by = defaultdict(dict)
    for it in items:
        by[(it["method"], it["cond"])][(it["speaker"], it["sent"])] = it
    res = {}
    for (method, cond), cell in sorted(by.items()):
        neutral = by.get((method, "neutral"), {})
        usable = [v for v in cell.values() if v.get("ok")]
        r = {"n": len(cell), "n_ok": sum(1 for v in cell.values() if v.get("ok")),
             "wer": _mean([v.get("wer") for v in cell.values() if v.get("ok")]),
             "sim_bank": _mean([v.get("sim_bank") for v in usable]), "sim_default": _mean([v.get("sim_default") for v in usable]),
             "takes": _mean([v.get("takes") for v in cell.values()]),
             **{f: _mean([v.get(f) for v in usable]) for f in FEATS}}
        if "tag_read_aloud" in next(iter(cell.values()), {}) or any("tag_read_aloud" in v for v in cell.values()):
            r["tag_read_aloud"] = _mean([1.0 if v.get("tag_read_aloud") else 0.0 for v in cell.values() if v.get("ok")])
        if cond != "neutral" and neutral:
            d, cons = {}, {}
            for f in FEATS + ["sim_bank", "sim_default", "wer"]:
                diffs = [cell[k][f] - neutral[k][f] for k in cell if k in neutral and cell[k].get("ok") and neutral[k].get("ok")
                         and cell[k].get(f) is not None and neutral[k].get(f) is not None]
                if diffs:
                    d[f] = statistics.fmean(diffs)
                    cons[f] = (sum(1 for x in diffs if x > 0), sum(1 for x in diffs if x < 0), len(diffs))
            r["delta"], r["sign"] = d, cons
            base = "neutral" if cond == "neutral2" else cond
            exp = EXPECT.get(base)
            if exp and cond != "neutral2":
                moved, against = [], []
                for f, sgn in exp.items():
                    if f not in d:
                        continue
                    up, down, n = cons[f]
                    agree = (up if sgn > 0 else down) / max(1, n)
                    if d[f] * sgn >= MOVE[f] and agree >= AGREE:
                        moved.append(f)
                    elif d[f] * sgn <= -MOVE[f] and agree <= 1 - AGREE:
                        against.append(f)
                level = "clear" if (len(moved) >= 2 and not against) else "partial" if (len(moved) == 1 and not against) else \
                    "wrong" if against else "none"
                sim_drop = -(d.get("sim_bank", 0.0) if "sim_bank" in d else 0.0)
                intelligible = (r["wer"] is not None and r["wer"] <= WER_MAX)
                keeps_voice = sim_drop <= SIM_TOL
                if method.split("_")[-1] == "preset" or method == "preset":
                    # positive control: the probe speaker's bank is not this voice, so only the delivery is judged
                    r.update(moved=moved, against=against, delivery=level, sim_drop=None, intelligible=intelligible, works=None,
                             verdict={"clear": "control: clear", "partial": "control: partly (one feature)",
                                      "wrong": "control: wrong direction", "none": "control: no clear change"}[level])
                    res[f"{method}|{cond}"] = r
                    continue
                r.update(moved=moved, against=against, delivery=level, sim_drop=sim_drop, intelligible=intelligible,
                         works=bool(level == "clear" and keeps_voice and intelligible),
                         verdict=("words break" if (level in ("clear", "partial") and not intelligible) else
                                  "works" if (level == "clear" and keeps_voice) else
                                  "partly (one feature)" if (level == "partial" and keeps_voice) else
                                  "delivery moves, voice drifts" if level in ("clear", "partial") else
                                  "wrong direction" if level == "wrong" else "no clear change"))
        res[f"{method}|{cond}"] = r
    return res


def _f(v, nd=2, plus=False):
    if v is None:
        return "–"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def report(d: Path) -> str:
    items = read_json(d / "probe_metrics.json")
    S = summarise(items)
    write_json(d / "probe_summary.json", S)
    methods = []
    for k in S:
        m = k.split("|")[0]
        if m not in methods:
            methods.append(m)
    L = []
    L.append("| method | condition | ok | WER | sim to speaker | Δsim | F0 mean Hz | ΔF0 st | ΔF0 range st | Δlevel dB | Δwords/s | Δvoiced | Δtilt dB | moved (expected) | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        for cond in list(INSTRUCT) + ["neutral2"] + [c for c in TAGS if c not in INSTRUCT]:
            r = S.get(f"{m}|{cond}")
            if not r:
                continue
            dl, sg = r.get("delta", {}), r.get("sign", {})
            def dd(f, nd=2):
                if f not in dl:
                    return "–"
                up, down, n = sg[f]
                return f"{dl[f]:+.{nd}f} ({up}↑{down}↓)"
            extra = ""
            if "tag_read_aloud" in r:
                extra = f" tag read aloud {r['tag_read_aloud']:.0%}"
            L.append(f"| {m} | {cond} | {r['n_ok']}/{r['n']} | {_f(r['wer'])} | {_f(r['sim_bank'], 3)} | {_f(dl.get('sim_bank'), 3, True)} | "
                     f"{_f(r['f0_mean'], 0)} | {dd('f0_st')} | {dd('f0_range_st')} | {dd('energy_db', 1)} | {dd('wps')} | {dd('voiced_ratio')} | "
                     f"{dd('tilt_db', 1)} | {', '.join(r.get('moved', [])) or '–'}{(' / against: ' + ', '.join(r['against'])) if r.get('against') else ''} | "
                     f"{r.get('verdict', 'baseline' if cond == 'neutral' else 'noise floor' if cond == 'neutral2' else '–')}{extra} |")
    md = "\n".join(L)
    (d / "probe_report.md").write_text(md + "\n")
    return md


def pooled(dirs: list[str], out: Path, rename: dict[str, str] | None = None, speakers: list[str] | None = None,
           methods: list[str] | None = None) -> dict:
    """Merge the per-utterance metrics of several probe directories and summarise them together.
    `dirs` entries may be "path" or "path=prefix" (prefix + "_" is put in front of the method names of that directory).
    Cells are keyed by (speaker, sentence): two directories that used different probe speakers pool into more pairs."""
    items = []
    for spec in dirs:
        path, _, prefix = spec.partition("=")
        for it in read_json(Path(path) / "probe_metrics.json"):
            it = dict(it)
            if prefix:
                it["method"] = f"{prefix}_{it['method']}" if not it["method"].startswith("tags_") else f"tags_{prefix}_{it['method'][5:]}"
            if rename and it["method"] in rename:
                it["method"] = rename[it["method"]]
            if speakers and it["speaker"] not in speakers:
                continue
            if methods and it["method"] not in methods:
                continue
            items.append(it)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "probe_metrics.json", items)
    md = report(out)
    return {"items": len(items), "md": md}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["items", "measure", "report", "pool"])
    ap.add_argument("--dirs", default=None, help="pool: comma-separated probe directories (path or path=prefix)")
    ap.add_argument("--only-speakers", default=None)
    ap.add_argument("--only-methods", default=None)
    ap.add_argument("--speakers", default="Tadano,Najimi")
    ap.add_argument("--methods", default="icl,icl_inst,cv_xvec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-sent", type=int, default=8)
    ap.add_argument("--tag-sent", type=int, default=4)
    ap.add_argument("--preset", default="ryan")
    ap.add_argument("--no-repeat", action="store_true")
    a = ap.parse_args()
    if a.cmd == "items":
        build_items(a.speakers.split(","), a.methods.split(","), Path(a.out), a.n_sent, a.tag_sent, a.preset, not a.no_repeat)
    elif a.cmd == "measure":
        if a.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        measure(Path(a.dir), a.device)
        print(report(Path(a.dir)))
    elif a.cmd == "pool":
        print(pooled(a.dirs.split(","), Path(a.out), speakers=a.only_speakers.split(",") if a.only_speakers else None,
                     methods=a.only_methods.split(",") if a.only_methods else None)["md"])
    else:
        print(report(Path(a.dir)))


if __name__ == "__main__":
    main()
