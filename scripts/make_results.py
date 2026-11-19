"""Markdown block with every stage metric of a finished run (for RESULTS.md).

  python scripts/make_results.py --run ep22 [--episode 22] > runs/ep22/eval/results_block.md
Reads report.json, eval/*.json, tts/voice_metrics.json, runs/judge_calibration*/report.json, runs/api_cost.jsonl,
runs/gpu_cost.jsonl. Missing pieces are printed as "not run" instead of failing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--episode", type=int, default=22)
a = ap.parse_args()
rd = P("runs") / a.run
ed = rd / "eval"
out = []
w = out.append


def f(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def ab_rows(summ: dict, title: str, x_name: str, y_name: str) -> None:
    if not summ:
        w(f"*{title}: not run.*\n")
        return
    w(f"**{title}** — {summ.get('n_pairs_rated', '?')} pairs; raters: " + ", ".join(f"{k} ({v})" for k, v in (summ.get("raters") or {}).items()))
    w("")
    w(f"| rater | criterion | {x_name} wins | {y_name} wins | ties | {x_name} win rate (no ties) | 95 % CI | sign test p |")
    w("|---|---|---|---|---|---|---|---|")
    for rater, crits in (summ.get("by_rater") or {}).items():
        for crit, comps in crits.items():
            for c in comps:
                ci = c.get("ci95") or [None, None]
                w(f"| {rater} | {crit} | {c['wins_x']} | {c['wins_y']} | {c['ties']} | {f(c.get('win_rate_x'))} | {f(ci[0])}–{f(ci[1])} | {f(c.get('p_sign'), 4)} |")
    w("")


rep = read_json(rd / "report.json") or {}
ev = read_json(ed / "episode_eval.json") or {}
w(f"### Run `{a.run}` vs real episode {a.episode}\n")
# ---------------------------------------------------------------- director
d = ev.get("director") or (read_json(ed / "director_eval.json") or {}).get("summary") or {}
if d and "error" not in d:
    g, r = d.get("duration_generated", {}), d.get("duration_real", {})
    w("**Director** (shot list vs the real adaptation of the same chapters)\n")
    w("| | generated | real |")
    w("|---|---|---|")
    w(f"| shots | {d['shot_count']['generated']} | {d['shot_count']['real']} |")
    w(f"| total duration | {f(g.get('total'), 0)} s | {f(r.get('total'), 0)} s |")
    w(f"| shot duration mean / median | {f(g.get('mean'))} / {f(g.get('median'))} s | {f(r.get('mean'))} / {f(r.get('median'))} s |")
    w(f"| p10–p90 | {f(g.get('p10'))}–{f(g.get('p90'))} s | {f(r.get('p10'))}–{f(r.get('p90'))} s |")
    w(f"| panel coverage | {f(d['panel_coverage'].get('generated'), 3)} | {f(d['panel_coverage'].get('real'), 3)} |")
    w("")
    w(f"Duration distribution distance: KS = {f(d.get('duration_ks'))}, Wasserstein = {f(d.get('duration_wasserstein_sec'))} s. "
      f"Shot-size mix total-variation distance: {f(d.get('shot_size_tv_distance'))}.")
    mg, mr = d.get("shot_size_mix_generated") or {}, d.get("shot_size_mix_real") or {}
    if mg:
        keys = sorted(set(mg) | set(mr), key=lambda k: -(mg.get(k, 0)))
        w("")
        w("| shot size | " + " | ".join(keys) + " |")
        w("|---|" + "---|" * len(keys))
        w("| generated | " + " | ".join(f"{100 * mg.get(k, 0):.0f} %" for k in keys) + " |")
        w("| real | " + (" | ".join(f"{100 * mr.get(k, 0):.0f} %" for k in keys) if mr else "not captioned") + " |")
    w("")
else:
    w("*Director evaluation: not run.*\n")
# ---------------------------------------------------------------- keyframes
k = ev.get("keyframes") or {}
w("**Keyframes**\n")
if k and "error" not in k:
    ig, ir = k.get("identity_generated") or {}, k.get("identity_real") or {}
    w(f"- First-pass keyframe QA: {f(k.get('keyframe_first_pass_rate'), 3)}.")
    w(f"- Identity (CCIP vs character bank): generated {ig} · real frames {ir}.")
else:
    w(f"- First-pass keyframe QA: {f(rep.get('keyframe_first_pass_rate'), 3)} (identity comparison not run).")
w("")
ab_rows(read_json(ed / "ab_summary_keyframes.json") or {}, "Blind A/B, generated keyframe vs real frame of the matched shot", "generated", "real")
# ---------------------------------------------------------------- video
v = ev.get("video") or {}
w("**Video**\n")
if v:
    w("| set | n | FVD vs real ep. | KVD | DOVER overall | warp error | held-frame ratio | mean flow |")
    w("|---|---|---|---|---|---|---|---|")
    for name, key, skey in (("generated (final cut)", "fvd_generated_vs_real", "stats_generated"), ("animatic baseline", "fvd_animatic_vs_real", "stats_animatic"),
                            ("real training episode 20 (floor)", "fvd_real_ep20_vs_real", None), (f"real episode {a.episode}", None, "stats_real")):
        fv, st = (v.get(key) or {}) if key else {}, (v.get(skey) or {}) if skey else {}
        w(f"| {name} | {st.get('n', '–')} | {f(fv.get('fvd'), 1)} | {f(fv.get('kvd'), 2)} | {f(st.get('dover_overall'), 3)} | {f(st.get('warp_error'), 3)} | "
          f"{f(st.get('held_frame_ratio'), 2)} | {f(st.get('mean_flow'), 2)} |")
    w("")
ab_rows(read_json(ed / "ab_summary_animatic.json") or {}, "Blind A/B, Wan cut vs animatic baseline (shot level)", "wan", "animatic")
ab_rows(read_json(ed / "ab_summary_real.json") or {}, "Blind A/B, generated shot vs the real shot that adapted the same panel", "wan", "real")
for sf in sorted(ed.glob("ab_summary_*.json")):         # comparisons against another run of the same shot list (e.g. v2 vs v1)
    other = sf.stem.replace("ab_summary_", "")
    if other in ("animatic", "real", "keyframes"):
        continue
    if other.startswith("keyframes_"):
        ab_rows(read_json(sf) or {}, f"Blind A/B, keyframes of this run vs run `{other[10:]}` (same shots)", "this run", other[10:])
    else:
        ab_rows(read_json(sf) or {}, f"Blind A/B, clips of this run vs run `{other}` (same shots)", "this run", other)
# ---------------------------------------------------------------- voice
vm = (read_json(rd / "tts" / "voice_metrics.json") or {})
ag, cal = vm.get("aggregate") or {}, vm.get("calibration") or {}
if ag:
    w("**Voice** (IndexTTS-2.5 zero-shot, references from training episodes)\n")
    w(f"- Lines: {ag.get('n')} scripted, {ag.get('n_ok')} synthesised, {ag.get('n_failed')} silent by design (no speakable text).")
    w(f"- Speaker similarity to the character's voice bank: mean {f((ag.get('sim_bank') or {}).get('mean'), 3)}, median {f((ag.get('sim_bank') or {}).get('median'), 3)} "
      f"(real same-speaker clips: {f((cal.get('real_same_speaker_ref_vs_ref') or {}).get('mean'), 3)}; different speakers: {f((cal.get('real_cross_speaker_ref_vs_ref') or {}).get('mean'), 3)}).")
    w(f"- Character error rate (ASR of the generated line vs the Japanese script): micro {f(ag.get('cer_micro'), 3)}, mean {f((ag.get('cer') or {}).get('mean'), 3)}, "
      f"median {f((ag.get('cer') or {}).get('median'), 3)}; flagged lines: {ag.get('flagged')}.")
    w("")
# ---------------------------------------------------------------- QA / cost
if rep:
    w("**QA loop and cost**\n")
    w(f"- Shots: {rep.get('n_shots')} ({rep.get('duration_sec')} s). Final clips: {rep.get('final_wan')} Wan "
      f"({rep.get('final_wan_passed', '–')} passed, {rep.get('final_wan_kept_despite_content_failure', '–')} kept despite a content failure), "
      f"{rep.get('final_fallback')} pan/zoom fallbacks.")
    w(f"- Video QA: first-pass rate {f(rep.get('video_first_pass_rate'), 3)}, final pass rate {f(rep.get('video_final_pass_rate'), 3)}, "
      f"retry rate {f(rep.get('qa_retry_rate'), 3)} re-renders per shot ({rep.get('video_attempts_total')} renders in total). "
      f"Mean judge scores: {rep.get('mean_judge_scores')}.")
    t = rep.get("timing_sec") or {}
    w(f"- Local GPU time: {f(rep.get('gpu_hours'))} h (" + ", ".join(f"{k} {v / 60:.0f} min" for k, v in t.items()) + ").")
api = read_jsonl(P("runs") / "api_cost.jsonl")
gpu = read_jsonl(P("runs") / "gpu_cost.jsonl")
w(f"- DeepSeek API so far (whole project): ${sum(r.get('cost_usd', 0) for r in api):.2f} in {len(api)} calls. "
  f"HF Jobs logged so far: ${sum(r.get('usd', 0) for r in gpu):.2f} (finished jobs only).")
w("")
for name in ("judge_calibration", "judge_calibration_val"):
    c = read_json(P("runs") / name / "report.json")
    if c:
        w(f"- Judge calibration `{name}`: agreement {f(c.get('agreement'), 3)} on {c.get('n')} clips (real passed {f(c.get('real_pass_rate'), 3)}, "
          f"broken failed {f(c.get('broken_fail_rate'), 3)}); by defect: {c.get('by_kind')}.")
print("\n".join(out))
