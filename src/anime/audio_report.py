"""Collect the measured numbers of the audio stack and (re)write the AUTO section of docs/audio.md.

    $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_report              # update docs/audio.md + print the section
    $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_report --summary    # update docs/audio.md + print a short summary
    $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_report --no-write   # print only

Everything in the AUTO section comes from files written by the stages themselves (nothing is typed by hand):
runs/audio_sep_compare/*/results.json, runs/audio_sep_compare/gpu_probe.json, data/interim/anime/epNN/separation.json,
data/processed/anime/asr/summary.json, data/processed/anime/spk/{clusters.json,embed_stats.json},
data/processed/bgm/stats.json, logs/audio_batch.state.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import ALL_EPS, ep_dir, ep_name
from src.common.config import CFG
from src.common.io import read_json

BEGIN, END = "<!-- AUTO:BEGIN (written by src/anime/audio_report.py - do not edit by hand) -->", "<!-- AUTO:END -->"
DOC = Path(CFG.paths.repo) / "docs" / "audio.md"
EP_SECONDS = 1488.0


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def collect() -> dict:
    """Machine-readable summary of what has been measured so far."""
    W, PR = Path(CFG.paths.work), Path(CFG.paths.processed)
    out: dict = {}
    cmp_ = {}
    for res in sorted((Path(CFG.paths.runs) / "audio_sep_compare").glob("*/results.json")):
        cmp_[res.parent.name] = read_json(res)
    out["comparisons"] = cmp_
    out["probe"] = read_json(Path(CFG.paths.runs) / "audio_sep_compare" / "gpu_probe.json", default=None)
    out["choice"] = read_json(Path(CFG.paths.interim) / "anime" / "separation_choice.json", default=None)
    out["separation"] = [read_json(ep_dir(ep) / "separation.json") for ep in ALL_EPS if (ep_dir(ep) / "separation.json").exists()]
    out["asr"] = read_json(PR / "anime" / "asr" / "summary.json", default=None)
    out["embed"] = read_json(PR / "anime" / "spk" / "embed_stats.json", default=None)
    cj = PR / "anime" / "spk" / "clusters.json"
    out["clusters"] = json.loads(cj.read_text()) if cj.exists() else None
    out["bgm"] = read_json(PR / "bgm" / "stats.json", default=None)
    st = W / "logs" / "audio_batch.state"
    out["state"] = st.read_text().splitlines()[-12:] if st.exists() else []
    return out


def summary_lines(c: dict) -> list[str]:
    """Per-stage runtime and peak VRAM, as far as measured."""
    L = []
    ch = c.get("choice")
    if ch:
        L.append(f"separation model: {ch.get('key')} ({ch.get('model')}), chosen under a {ch.get('vram_budget_gb')} GB budget, "
                 f"segment {ch.get('segment')}, peak VRAM {ch.get('peak_vram_mb')} MiB in the comparison")
    else:
        L.append("separation model: not chosen yet (comparison has not run) -> config default would be used")
    sep = c.get("separation") or []
    if sep:
        rt = [r["runtime_seconds"] for r in sep]
        pk = [r.get("peak_vram_mb") for r in sep if r.get("peak_vram_mb")]
        L.append(f"separation: {len(sep)}/24 episodes, {np.mean(rt):.0f} s per episode (min {np.min(rt):.0f}, max {np.max(rt):.0f}), "
                 f"peak VRAM {max(pk) if pk else '—'} MiB, segment {sorted({str(r.get('segment')) for r in sep})}, model {sorted({r['model'] for r in sep})}")
    else:
        full = (c.get("comparisons") or {}).get("full")
        if full and full.get("choice"):
            rtf = full["summary"][full["choice"]].get("rtf")
            if rtf:
                L.append(f"separation: no episode done yet; from the comparison: {rtf:.1f}x realtime -> ~{EP_SECONDS / rtf:.0f} s per episode (estimate)")
    emb = (c.get("embed") or {})
    eps = {k: v for k, v in emb.items() if k.startswith("ep")}
    if eps:
        L.append(f"speaker embeddings: {len(eps)}/24 episodes, {np.mean([v['seconds'] for v in eps.values()]):.0f} s per episode, "
                 f"peak VRAM {emb.get('_last_run', {}).get('peak_vram_mb', '—')} MiB ({emb.get('_last_run', {}).get('backend')})")
    a = c.get("asr") or {}
    eps = {k: v for k, v in a.items() if k.startswith("ep")}
    if eps:
        pk = [v.get("peak_vram_mb") for v in eps.values() if v.get("peak_vram_mb")]
        L.append(f"ASR: {len(eps)}/24 episodes, {np.mean([v['runtime_seconds'] for v in eps.values()]):.0f} s per episode, peak VRAM {max(pk) if pk else '—'} MiB, "
                 f"compute type {sorted({str(v.get('compute_type')) for v in eps.values()})}, line recall {np.mean([v['line_recall'] for v in eps.values()]):.1%}")
    b = c.get("bgm")
    if b:
        L.append(f"BGM: {b['cues']} cues from {len(b['episodes'])} episodes, CLAP indexing {b['index_seconds']} s on {b['clap_device']}, peak VRAM {b.get('peak_vram_mb', '—')} MiB")
    cl = c.get("clusters")
    if cl:
        L.append(f"voice clusters: {len(cl['clusters'])} from train episodes {cl['params']['train_episodes']} (threshold {cl['params']['distance_threshold']}), "
                 f"must-link agreement {cl['sanity']['must_link_same_cluster']}")
    p = c.get("probe")
    if p:
        for st in ("spk", "asr", "clap"):
            r = p.get(st) or {}
            if "error" in r:
                L.append(f"probe {st}: FAILED {r['error'][-160:]!r}")
            elif r:
                L.append(f"probe {st}: peak VRAM {r.get('peak_vram_mb')} MiB, " + ", ".join(f"{k}={r[k]}" for k in ("backend", "compute_type", "lines_per_s", "rtf", "cues_per_s",
                         "est_seconds_per_episode", "est_seconds_per_1000_cues", "min_cos_vs_onnxruntime_cpu", "line_recall") if k in r))
    return L


def section(c: dict) -> str:
    L = [f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Missing parts mean that stage has not run yet._", "",
         "### Per-stage runtime and peak VRAM (measured)", ""] + [f"- {x}" for x in summary_lines(c)] + [""]
    if c["state"]:
        L += ["### Batch state (last lines of logs/audio_batch.state)", "", "```"] + c["state"] + ["```", ""]

    # ---- separation comparison
    for name, d in c["comparisons"].items():
        cols = ["syn_sdr_voc", "syn_sdr_inst", "bleed_db", "gap_supp_db", "asr_logprob", "asr_wordprob", "cer_others", "line_cov",
                "inst_speech", "residual_db", "rtf"]
        L += [f"### Separation comparison `{name}` ({d['created']}, device {d['device']}, overlap {d['overlap']}, VRAM budget {d.get('vram_budget_gb')} GB)", ""]
        ex = [f"{Path(e['path']).name} ({e['kind']}, {e.get('dur')} s" + (f", dialogue coverage {e.get('dialogue_cov')}" if e.get("dialogue_cov") is not None else "") + ")"
              for e in d["excerpts"]]
        L += ["Excerpts: " + "; ".join(ex) + ".", ""]
        if d.get("synthetic"):
            s = d["synthetic"]
            L += [f"Synthetic remix: {s['speech_seconds']} s of speech from {len(s['speech_lines'])} dry dialogue lines + {len(s['music_spans'])} music-only spans "
                  f"(episodes {s['episodes']}), speech-to-music ratio {s['smr_db']} dB.", ""]
        L += ["| model | " + " | ".join(cols) + " | segment | peak VRAM MiB | mean rank |", "|---|" + "---|" * (len(cols) + 3)]
        for k, s in d["summary"].items():
            r = d["results"][k]
            L.append(f"| {k} (`{r['model']}`) | " + " | ".join(_fmt(s.get(col)) for col in cols)
                     + f" | {r.get('segment')} | {_fmt(r.get('peak_vram_mb'))} | {_fmt(d['mean_rank'].get(k))} |")
        for k, r in d["results"].items():
            if "error" in r:
                L.append(f"| {k} | failed: {r['error'][:100]} |")
        L += ["", f"Ranked on {d['rank_keys']}. **Choice: {d['choice']} (`{d['choice_model']}`)**. Stems for listening: `{Path(CFG.paths.runs) / 'audio_sep_compare' / name}`.", ""]
        oc = d.get("overlap_check")
        if oc:
            L += [f"Overlap check ({oc['model']}, synthetic remix): overlap {oc['overlap']} -> syn_sdr_voc {oc['syn_sdr_voc']} / syn_sdr_inst {oc['syn_sdr_inst']} dB "
                  f"at {oc['rtf']}x realtime; overlap {oc['vs_overlap']} (used for the batch) -> {oc['syn_sdr_voc_at_used']} / {oc['syn_sdr_inst_at_used']} dB "
                  f"at {oc['rtf_at_used']}x realtime.", ""]
        sc = d.get("segment_check")
        if sc:
            L += ["Segment size vs. quality and VRAM (winner, synthetic remix):", "",
                  "| setting | overlap | segment (dim_t) | syn_sdr_voc | syn_sdr_inst | rtf | peak VRAM MiB |", "|---|---|---|---|---|---|---|"]
            L += [f"| {r['setting']} | {r.get('overlap')} | {r.get('segment')} | {r.get('syn_sdr_voc')} | {r.get('syn_sdr_inst')} | {r.get('rtf')} | {r.get('peak_vram_mb')} |" for r in sc]
            L.append("")
        if d.get("asr_stats"):
            L += [f"ASR proxies ran with {d['asr_stats']}.", ""]

    # ---- GPU probe
    p = c.get("probe")
    if p:
        L += [f"### GPU probe of the other stages ({p.get('created')}, raw mix of episode {p.get('episode')} as input)", "", "```json",
              json.dumps({k: p.get(k) for k in ("spk", "asr", "clap")}, indent=1), "```", ""]

    # ---- batch separation
    rows = c["separation"]
    if rows:
        L += ["### Batch separation", "", "| ep | model | segment | overlap | runtime s | x realtime | peak VRAM MiB | residual dB | OOM fallbacks |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['episode']} | {r['model']} | {r.get('segment')} | {r.get('overlap')} | {r['runtime_seconds']} | {r['realtime_factor']} | {r.get('peak_vram_mb')} | {r['residual_db']} | {r.get('oom_fallbacks') or ''} |"
              for r in rows]
        miss = [ep_name(e) for e in ALL_EPS if not (ep_dir(e) / "vocals.wav").exists()]
        L += ["", f"Not separated yet: {', '.join(miss) if miss else 'none'}.", ""]

    # ---- ASR
    a = c["asr"]
    if a:
        eps = {k: v for k, v in a.items() if k.startswith("ep")}
        if eps:
            L += ["### ASR (faster-whisper large-v3, vocal stems)", "",
                  f"{len(eps)}/24 episodes. Mean line recall (subtitle dialogue lines with >= 1 ASR word inside) "
                  f"**{np.mean([v['line_recall'] for v in eps.values()]):.1%}**, ASR words inside subtitle dialogue/song/OP-ED spans "
                  f"{np.mean([v['word_in_dlg'] for v in eps.values() if v.get('word_in_dlg') is not None]):.1%}, "
                  f"mean word probability {np.mean([v['mean_word_prob'] for v in eps.values() if v.get('mean_word_prob')]):.3f}.", "",
                  "| ep | segments | words | line_recall | word_in_dlg | word_prob | song segs | runtime s | peak VRAM MiB | compute |", "|---|---|---|---|---|---|---|---|---|---|"]
            L += [f"| {k} | {v['segments']} | {v['words']} | {_fmt(v['line_recall'], 3)} | {_fmt(v.get('word_in_dlg'), 3)} | {_fmt(v.get('mean_word_prob'), 3)} | "
                  f"{v.get('song_segments', '')} | {v['runtime_seconds']} | {v.get('peak_vram_mb')} | {v.get('compute_type')} |" for k, v in sorted(eps.items())]
            L.append("")

    # ---- speakers
    cl = c["clusters"]
    if cl:
        p_, s = cl["params"], cl["sanity"]
        sizes = [x["size"] for x in cl["clusters"]]
        L += ["### Voice clusters", "",
              f"{len(cl['clusters'])} clusters from {s['train_lines']} train lines ({p_['core_lines']} core lines; train episodes with embeddings: {p_['train_episodes']}); "
              f"cosine-distance threshold {p_['distance_threshold']} ({p_['threshold_selection']}). Built {p_.get('built')}, "
              f"{p_.get('ids_inherited')} ids inherited from the previous run.",
              f"Sizes: {sizes[:25]}{' …' if len(sizes) > 25 else ''}. Train lines assigned: {s['train_assigned']:.1%}. "
              f"Held-out lines: {s['heldout_lines']} (assigned {_fmt(s['heldout_assigned'], 3)}, mean cosine {_fmt(s['heldout_mean_cos'], 3)}).", "",
              "Label-free purity checks:", "",
              f"- must-link pairs (subtitle continuation lines, {s['must_link_pairs']} pairs) in the same cluster: **{_fmt(s['must_link_same_cluster'], 3)}** "
              f"(random pairs: {_fmt(s['random_pair_same_cluster'], 3)})",
              f"- narrator intro lines: {json.dumps(s.get('narrator_intro'), ensure_ascii=False)}",
              f"- mean cosine to own centroid: core {_fmt(s['mean_cos_core'], 3)}, non-core {_fmt(s['mean_cos_noncore'], 3)}, "
              f"two-speaker/overlapping lines {_fmt(s['mean_cos_two_speaker_lines'], 3)}", "",
              "| id | size | core | episodes | seconds | mean cos | example lines |", "|---|---|---|---|---|---|---|"]
        for x in cl["clusters"][:40]:
            ex = " / ".join(e["text"][:40].replace("|", "/") for e in x["exemplars"][:3])
            L.append(f"| {x['id']} | {x['size']} | {x['core_size']} | {x['n_episodes']} | {x['total_dur']} | {x['mean_cos']} | {ex} |")
        L += ["", "Threshold sweep (t = distance threshold, J = must_link_same - random_same):", "",
              "| t | clusters>=min | must_link_same | random_same | J | coverage | largest share |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r['t']} | {r['clusters']} | {r['must_link_same']} | {r['random_same']} | {r['J']} | {r['coverage']} | {r.get('largest_share')} |" for r in p_["sweep"]]
        L.append("")

    # ---- BGM
    b = c["bgm"]
    if b:
        L += ["### BGM cue library", "",
              f"{b['cues']} cues ({b['total_seconds']:.0f} s, median {b['median_dur']} s) from episodes {b['episodes']}; "
              f"{b['low_dialogue_cues']} cues with <= 20 % dialogue coverage; {b['rejected_non_music']} candidates rejected as non-music by CLAP; "
              f"CLAP `{b['clap_model']}` on {b['clap_device']}, embedding + indexing {b['index_seconds']} s, peak VRAM {b.get('peak_vram_mb')} MiB. Built {b['built']}.", ""]
        try:
            from src.generate.bgm import retrieve

            L += ["Retrieval examples (top 3):", ""]
            for q in ("gentle_comedy", "tense, awkward silence", "heartwarming", "energetic"):
                r = retrieve(q, min_dur=10.0, k=3)
                L.append(f"- `{q}` -> " + ", ".join(f"{x['id']} ({x['dur']:.0f} s, sim {x['sim']:.2f}, tags {x['tags'][:2]})" for x in r))
            L.append("")
        except Exception as e:  # the text tower may be unavailable in this venv
            L += [f"(retrieval examples skipped: {e!r})", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--summary", action="store_true", help="print only the short per-stage summary")
    args = ap.parse_args()
    c = collect()
    body = section(c)
    print("\n".join(["--- audio stack: measured so far ---"] + summary_lines(c)) if args.summary else body)
    if args.no_write:
        return
    text = DOC.read_text() if DOC.exists() else "# Audio stack\n"
    block = f"{BEGIN}\n{body}\n{END}"
    if BEGIN in text and END in text:
        text = text[: text.index(BEGIN)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n## Measured results (auto-generated)\n\n" + block + "\n"
    tmp = DOC.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(DOC)


if __name__ == "__main__":
    main()
