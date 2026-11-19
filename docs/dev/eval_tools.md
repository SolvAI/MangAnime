# Evaluation tools: metrics, FVD, broken clips, blind A/B, review and labeling pages

Everything here runs locally and offline (weights are fetched once). Code: `src/eval/{video_metrics,fvd,broken}.py`,
`tools/{ab_page,review,align_label}/`, tests in `tests/test_eval_*.py`.

| Need | Command |
|---|---|
| Evidence metrics for clips | `python -m src.eval.video_metrics <clip-or-dir>... --out metrics.json` |
| FVD between two clip sets | `python -m src.eval.fvd --real <dir> --gen <dir> --out fvd.json` |
| Broken clips for judge calibration | `python -m src.eval.broken --src <dir of real shots> --out <dir> --n 50` |
| Blind A/B rating page | `python -m tools.ab_page.app --manifest m.json --db /home/user/komi-poc-work/runs/ab/results.duckdb --port 8300` |
| A/B numbers without the page | `python -m tools.ab_page.stats --db <db> --experiment <name>` |
| Review / rename clusters | `python -m tools.review.app --port 8301` |
| Gold alignment labeling | `python -m tools.align_label.app --port 8302` |
| Tests | `/home/user/komi-poc-work/venvs/main/bin/python -m pytest tests -q` |

Run from the repo root (`/home/user/komi-poc`) after `. scripts/env.sh`. All of it works in `venvs/main`; no extra
package is needed there.

## 1. Setup

**Weights** (`/home/user/komi-poc-work/models/eval/`, sha256 in `SHA256SUMS`, re-downloaded automatically if missing):

| File | Source | Used by |
|---|---|---|
| `raft_small_C_T_V2-01064c6d.pth`, `raft_large_C_T_SKHT_V2-ff5fadd5.pth` | download.pytorch.org (torchvision weights) | warp error, motion |
| `DOVER.pth` | github.com/VQAssessment/DOVER release v0.1.0 | `dover_score` |
| `i3d_torchscript.pt` | the StyleGAN-V Dropbox link (same sha256 as the copies on the HF Hub) | FVD |

**DOVER code**: `/home/user/komi-poc-work/third_party/DOVER`, commit `f1ddc96` (S-Lab licence, non-commercial).
Recreate with `git clone https://github.com/VQAssessment/DOVER /home/user/komi-poc-work/third_party/DOVER`.
Its `decord` and `scikit-video` imports are stubbed at import time (frames are decoded with PyAV), so it runs in
`venvs/main` as is.

**Device.** Every model function takes `device=None | "auto" | "cuda" | "cpu"`. `None`/`"auto"` means: `cuda` only if
`nvidia-smi` reports at least 4 GB free right now (6 GB for RAFT-large), otherwise `cpu`. The check does not create a
CUDA context. Overrides: `KOMI_EVAL_DEVICE=cpu`, or `CUDA_VISIBLE_DEVICES=""`. For planned GPU runs keep using the
admission script: `scripts/gpu_run.sh 4 python -m src.eval.video_metrics ... --device cuda`.
CPU threads: `KOMI_EVAL_THREADS` (default 8).

**Other venvs.** If a caller cannot import torch, set `KOMI_EVAL_PYTHON=/home/user/komi-poc-work/venvs/main/bin/python`
(or pass `python=`) and `dover_score` runs through a subprocess; `video_metrics.run_in_subprocess(paths, metrics)`
does the same for any metric and returns the CLI's JSON.

## 2. `src/eval/video_metrics.py`

```python
load_frames(path, fps=None, size=None, max_frames=None, start=0.0, return_fps=False) -> np.ndarray[T, H, W, 3] uint8 RGB
flicker_score(frames, fps=None, hold_mad=0.12, hold_block=1.5) -> dict
detect_cuts(frames, return_scores=False) -> list[int]            # index of the first frame of each new shot
flow_stats(frames, method="raft_small", device=None, height=360, iters=12, batch=None, max_pairs=None, skip_held=True) -> dict
warp_error(frames, method="raft_small", device=None, height=360, max_pairs=None, flow=None, return_details=False) -> float | dict
motion_stats(frames, fps=None, method="raft_small", device=None, height=360, max_pairs=None, flow=None) -> dict
dover_score(path, device=None, seed=0, python=None) -> dict
technical_check(path, expected_dur=None, need_audio=False, *, dur_tol=None, expected_fps=None, expected_size=None,
                allow_static=False, frozen_frac=0.98, frozen_min_dur=1.0, black_frac=0.95, silence_db=-60.0) -> dict
analyze_clip(path, metrics=("technical", "flicker", "motion", "warp", "dover"), *, expected_dur=None, need_audio=False,
             device=None, flow="raft_small", flow_height=360, fps=None, height=None, max_frames=None, max_pairs=None,
             allow_static=False) -> dict                         # one decode, one flow pass, never raises per metric
```

`size`: `None` = native, int = target height, `(W, H)` = exact. `fps`: frames are dropped or repeated, never blended.
`method`: `raft_small` | `raft_large` | `farneback` (OpenCV, CPU only, no weights). Pass the result of one
`flow_stats` call as `flow=` to both `warp_error` and `motion_stats` to compute flow once.

CLI inputs can be clips, directories, `.txt` lists, or `.jsonl` files with a `path` per row; rows may carry
`expected_dur`, `need_audio`, `allow_static` (the `labels.jsonl` of `broken.py` does), which override the flags.

```bash
python -m src.eval.video_metrics runs/x/shots --metrics technical,flicker,motion,warp,dover --out runs/x/metrics.json
python -m src.eval.video_metrics calib/labels.jsonl --metrics technical,flicker --out calib/tech.json     # CPU, fast
python -m src.eval.video_metrics clips/ --metrics motion,warp --flow farneback --flow-height 240 --max-pairs 16 --device cpu
```

Output: `{"clips": {path: {"technical", "flicker", "motion", "warp", "dover", "n_frames", "fps", "runtime_s"}}, "config", "runtime_s"}`.

Alias keys, kept because `src/eval/episode_eval.py` and `src/eval/ab_judge.py` read them: `flicker.held_frame_ratio`
(= `repeat_ratio`), `flicker.flicker_index` (= `luma_flicker`), `motion.mean_flow` (= `mean_flow_mag`),
`motion.static_ratio` (= `static_frame_ratio`), `warp.mean` (= `warp_error`).

### What each metric returns

**`flicker_score`** (pixel values 0..255, computed on 180 px luma)

| Key | Meaning |
|---|---|
| `mean_abs_diff`, `mean_abs_diff_moving` | mean frame difference, all pairs / pairs that are not held frames. `vbench_temporal_flicker` = 1 - mean/255 |
| `hf_energy`, `hf_energy_unique`, `hf_ratio_unique` | temporal high-pass energy on all frames / after removing held frames / relative to the mean change |
| `luma_flicker`, `luma_flicker_p95`, `chroma_flicker` | second difference of the per-frame mean luma / chroma: global brightness or colour pumping |
| `repeat_ratio`, `unique_frames`, `unique_ratio`, `effective_fps` | share of frame pairs that are exact repeats (held drawings) and what that leaves |
| `mean_hold`, `hold_hist` | frames per drawing, and the share of drawings held 1 / 2 / 3 / 4+ frames ("on ones / twos / threes") |

A pair counts as a repeat when the mean luma difference is below 0.12 and no 8x8 block changes by more than 1.5
(constants `HOLD_MAD`, `HOLD_BLOCK`): this keeps lip flaps and slow pans "moving" and tolerates x264 noise on true holds.

**`warp_error`** returns the mean masked L1 error (0..255) between frame t and frame t+1 warped back with the flow,
over non-occluded pixels (forward-backward consistency), cut pairs excluded. `return_details=True` adds
`warp_mse_x1000` (the E_warp of Lai et al.), `warp_error_moving` (held pairs left out), `warp_error_p95`,
`warp_error_max`, `occlusion_ratio`, `residual_ratio` (share of the raw frame difference that the flow does not
explain; `None` when the frames barely change), `n_pairs`, `n_cut_pairs`.

**`motion_stats`**: `mean_flow_mag` (px per frame at `flow_height`), `mean_flow_mag_rel` (% of frame height),
`mean_flow_mag_moving`, `flow_mag_p95`, `static_frame_ratio`, `moving_area_ratio`, `global_motion_mag` (median flow
vector: pan / shake), `local_motion_mag`, `hard_cuts` (frame indices), `n_hard_cuts`.

**`dover_score`**: `technical`, `aesthetic`, `overall` in 0..1 (higher = better), plus `technical_z`, `aesthetic_z`,
raw branch outputs. Deterministic (seeded sampling). Two things differ from running the DOVER repo scripts today,
both checked against the repo's own reference numbers:

1. The aesthetic view is resized **without antialiasing**. Current torchvision antialiases tensors by default; with
   that the aesthetic output of the repo demo `17734.mp4` is 0.040, without it 0.148, and 0.148 is the value the
   README's five dataset comparisons imply.
2. `fuse_results()` in the repo pairs the normalisation constants with the wrong branch; run as is it gave
   "technical" 0.04 to 0.07 and "aesthetic" 0.82 to 0.99 for all four videos I tried. The repo's prediction files show the technical
   branch outputs -0.083 +- 0.038 and the aesthetic branch 0.111 +- 0.074 on LIVE-VQC, which are exactly the two
   constant pairs. Here: `technical_z = (raw + 0.08285) / 0.03774`, `aesthetic_z = (raw - 0.1107) / 0.07355`,
   `overall = sigmoid(0.3896 technical_z + 0.6104 aesthetic_z)`.

**`technical_check`** returns `{"ok", "reasons", "warnings", "info"}`. Each reason is `"code: detail"`:

| Code | Trigger |
|---|---|
| `missing_file`, `empty_file`, `unreadable`, `no_video_stream`, `no_frames` | file cannot be used at all |
| `decode_errors` | ffmpeg reports errors during a full decode of the video stream |
| `truncated_stream` | container duration and decodable frames disagree by more than max(0.5 s, 10 %) |
| `wrong_duration` | decoded duration vs `expected_dur`, tolerance `dur_tol` (default max(0.25 s, 5 %)) |
| `wrong_fps`, `wrong_size` | only when `expected_fps` / `expected_size` are given |
| `black_video`, `blank_video` | at least 95 % of frames black / one flat colour |
| `frozen_video` | at least 98 % of frame pairs identical, no cut, clip of 1 s or more. A warning instead when `allow_static=True` |
| `no_audio_stream`, `silent_audio`, `audio_decode_errors`, `audio_duration_mismatch` | failures only when `need_audio=True`, otherwise warnings |

`info` has `duration`, `frames_decoded`, `fps`, size, `static_pair_frac`, `longest_static_run_s`, `black_frame_frac`,
`audio_rms_db`, `audio_peak_db`, `audio_silence_frac`. Warnings: `mostly_black`, `long_static_run`.

## 3. `src/eval/fvd.py`

```python
clip_windows(path, fps=16.0, n_frames=16, size=224, max_windows=4, resize="center_crop") -> uint8 [K, 16, 224, 224, 3]
clip_features(path, ..., device=None, batch=8, cache_dir=CACHE_DIR) -> np.ndarray [K, 400]      # cached
frechet_distance(feats_a, feats_b) -> float
kernel_distance(feats_a, feats_b) -> float                                                      # KVD (unbiased MMD)
fvd(real_paths, gen_paths, fps=16.0, n_frames=16, size=224, max_windows=4, resize="center_crop", device=None,
    batch=8, cache_dir=CACHE_DIR, baseline=False, seed=0) -> dict
```

```bash
python -m src.eval.fvd --real data/eval/ep23_real_shots --gen runs/ep23/shots --out runs/ep23/fvd.json --baseline
```

Protocol: I3D (Kinetics-400 torchscript of StyleGAN-V, 400-d features), 16 frames, 224 px, values in [-1, 1].
Both sets are resampled to 16 fps (Wan's rate), frames are resized to 224 px height and centre-cropped, and up to 4
non-overlapping windows per clip are used. Features are cached in `/home/user/komi-poc-work/cache/eval/fvd/` under
the sha256 of the file content plus the protocol. Output: `fvd`, `kvd`, clip and window counts, `protocol`,
`warnings`, `errors`, and with `--baseline` the FVD between random halves of the real set.

## 4. `src/eval/broken.py`

```python
KINDS, KIND_INFO                                   # 15 kinds with description, needs, hard_fail, evidence
make_broken(src_clip, kind, out_path, seed=0, **kw) -> dict
make_real(src_clip, out_path) -> dict              # same decode -> x264/AAC path, content untouched
build_set(src, out_dir, n_broken=50, n_real=None, kinds=KINDS, seed=0) -> list[dict]
effect_ok(row) -> bool
```

```bash
python -m src.eval.broken --src data/eval/heldout_shots --out data/eval/judge_calib --n 50 --n-real 50 --seed 0
python -m src.eval.broken --list-kinds
```

Writes `<out>/clips/c_0000.mp4 ...` (opaque shuffled names, so a path never reveals a label) and `<out>/labels.jsonl`:

```json
{"path": ".../clips/c_0007.mp4", "label": "broken", "kind": "freeze", "src": ".../shot_0123.mp4", "params": {"start_frac": 0.0},
 "expected_dur": 3.42, "need_audio": true, "allow_static": false, "hard_fail": true,
 "technical_ok": false, "technical_reasons": ["frozen_video"], "effect_mean": 16.7, "effect_local": 91.2}
```

`expected_dur`, `need_audio` and `allow_static` describe the **source** shot and stand in for the shot spec: pass
them to `technical_check` for real and broken clips alike. `hard_fail` says whether `technical_check` must reject
the clip; `build_set` verifies that and retries otherwise. `effect_*` measure how much the picture changed; a
source that cannot show a failure (freezing a still, shuffling a still, silencing a silent shot) is skipped for
that kind.

| Kind | Arguments | Hard fail | Evidence that should reveal it |
|---|---|---|---|
| `flicker` | `strength=0.35, rate=0.7, color=True` | no | `flicker.luma_flicker`, `chroma_flicker`, warp error |
| `frame_shuffle` | `window=8` (0 = whole clip) | no | warp error, `luma_flicker_p95`, captions |
| `freeze` | `start_frac=0.0` | yes when `start_frac` = 0 | `technical.frozen_video`, `repeat_ratio`, `longest_static_run_s` |
| `melt` | `strength=1.0, region="auto"\|"face"\|"full"\|(cx, cy), radius=0.3` | no | captions / artifact checklist, identity, frames |
| `ghost` | `shift=5, alpha=0.45, offset=(10, 4)` | no | captions, frames |
| `noise_blur` | `blur=2.2, noise=14` | no | DOVER technical, warp `residual_ratio`, `repeat_ratio` = 0 |
| `color_shift` | `hue=None, sat=1.3, mode="hue"\|"swap"` | no | captions (hair / uniform colours), identity, frames |
| `wrong_audio` | `other=<clip>` | no | transcript / WER vs script, speaker similarity |
| `silent_audio` | `drop=False` | yes | `technical.silent_audio` / `no_audio_stream` |
| `truncated` | `keep=0.5` | yes | `technical.wrong_duration` |
| `corrupt_file` | `mode="garble"\|"cut"\|"header"\|"empty"` | yes | `technical.decode_errors` / `unreadable` / `empty_file` |
| `garbled_text_overlay` | `style="both"\|"subtitle"\|"sign", morph_every=3` | no | captions / artifact checklist, OCR |
| `jitter` | `amp=0.012, rot=0.4` | no | `motion.global_motion_mag`, `static_frame_ratio` = 0 |
| `hard_cut` | `other=<clip>, at=0.5` | no | `motion.n_hard_cuts` |
| `black` | none | yes | `technical.black_video` |

`region="face"` uses the anime face detector of `dghs-imgutils` when it is importable and falls back to the most
active region otherwise.

Judge calibration loop (sketch):

```python
import json
from src.eval import video_metrics as vm
for row in map(json.loads, open("data/eval/judge_calib/labels.jsonl")):
    tc = vm.technical_check(row["path"], expected_dur=row["expected_dur"], need_audio=row["need_audio"],
                            allow_static=row["allow_static"])
    verdict = "fail" if not tc["ok"] else judge(evidence_for(row["path"]))      # hard failures skip the judge
    agree += (verdict == "pass") == (row["label"] == "real")
```

## 5. `tools/ab_page/` (blind A/B)

```bash
python -m tools.ab_page.app --manifest runs/ab/ep23_wan_vs_animatic.json \
    --db /home/user/komi-poc-work/runs/ab/results.duckdb --port 8300 [--host 127.0.0.1] [--transcode auto|always|never] [--check]
```

Manifest: `{"experiment", "criteria": [...], "pairs": [{"pair_id", "a", "b", "label_a", "label_b", "caption"?, "meta"?}]}`.
`--check` validates the manifest and the files and exits. Open `http://127.0.0.1:8300/`, enter a name, rate.

- The browser only receives two random one-time media tokens, a progress counter and the criteria. Labels, file
  names, pair ids and `meta` stay on the server. `caption` (optional) is shown to the rater.
- Left/right is drawn per rating, stays fixed if the page is reloaded, and is stored with the rating.
- Resume: a rater who comes back under the same name continues with the pairs not yet rated; the order is a fixed
  shuffle per rater. "Skip" moves a pair to the end; "Undo" removes the rater's last rating.
- Clips a browser cannot play (10-bit H.264, HEVC, MKV, PCM audio) are re-encoded once to H.264 yuv420p + AAC into
  `<db dir>/media_cache/` (`--transcode auto`, the default).
- No CDN, no build step: one small script and one stylesheet, plain forms (HTMX turned out unnecessary).

Keys: `1` left, `2` right, `=` (or `0`) tie, `Enter` submit, `Space` play/pause, `R` restart, `L` loop, `,` `.` frame
step, `A` / `S` / `D` audio left / right / off, arrow up/down change criterion, `U` undo.

Pages: `/rate`, `/results` (un-blinds: open after rating), `/api/results` (JSON), `/api/progress`, `/healthz`.

**Storage and judges** (`tools/ab_page/stats.py`). Table `ab_ratings`, one row per rating and criterion:
`rating_id, experiment, pair_id, rater, criterion, choice ('a'|'b'|'tie'), winner_label, label_a, label_b, left_side,
ts, duration_ms, note, meta`.

```python
from tools.ab_page.stats import record_rating, summarize
record_rating(db, "ep23_wan_vs_animatic", "s012", "judge:deepseek",
              {"overall": "a", "motion": "tie", "character consistency": "wan"},      # 'a' | 'b' | 'tie' or a label
              label_a="wan", label_b="animatic", left_side=None, meta={"cache_key": "..."})
s = summarize(db, "ep23_wan_vs_animatic")
s["groups"]["human"]["criteria"]["overall"][0]
# {"label_x": "wan", "label_y": "animatic", "wins_x", "wins_y", "ties", "n", "win_rate_x" (ties excluded),
#  "ci95" (exact binomial), "ci95_wilson", "p_sign" (two-sided exact sign test), "tie_rate", "winner", "significant"}
```

`summarize` returns groups `human`, `judge` (automatic raters: names starting with `judge:` or `metric:`) and `all`, each also as one vote per pair
(majority over raters), plus `by_rater`, `position_bias` (how often the left clip won) and `judge_agreement`
(each judge vs the human majority). Only the latest rating per rater, pair and criterion counts.
DuckDB allows one writer process per file; every call opens the file for a few milliseconds and retries while
another process holds it, so the page and judge scripts can write to the same file concurrently.

## 6. `tools/review/` (cluster names)

```bash
python -m tools.review.app --port 8301 [--processed /home/user/komi-poc/data/processed] [--source NAME=/abs/clusters.json] [--low-conf 0.6]
```

Reads `<processed>/characters/clusters.json` and `<processed>/anime/spk/clusters.json` when they exist; the index
page says which are missing. Clusters are sorted by confidence, unknown and low first; filters: all / not reviewed /
low confidence. Type a name and press Enter (focus jumps to the next cluster), or use Confirm (keeps the automatic
name), Junk (writes `_junk`), Clear. Every change is written at once to the sibling
`names_override.json` = `{"<cluster_id>": "<name>"}`.

The reader accepts `{"clusters": [...]}`, a bare list or an id-keyed dict; exemplars as `{"image"}`, `{"audio", "text"}`
or plain paths (relative paths resolve against the JSON's folder); missing `id`, `confidence`, `size` are fine.

## 7. `tools/align_label/` (gold alignment)

```bash
python -m tools.align_label.app --port 8302 [--processed /home/user/komi-poc/data/processed]
```

Reads `manga/panels.jsonl`, `anime/shots.jsonl`, `align/alignment.jsonl`, `align/gold/queue.json` under
`<processed>` (schemas of the brief; any of them may be missing and the page says so). Queue formats:
`[1, 2]`, `{"episodes": [1, 2]}`, or `[{"episode": 1, "chapters": [1, 2, 3]}]`. Without a queue every episode in
`shots.jsonl` is listed.

Page layout: panels in reading order on the left, the selected panel large next to the hovered shot, the shot strip
(or grid, `G`) below. Automatic matches are pre-filled in purple and are not gold until confirmed.
Click a shot to assign it to the active panel (shift-click for a range), `Enter` confirm and next, `X` no match and
next, `U` reset the panel, `J`/`K` or arrows to move, `N` next unreviewed, `[` `]` scroll the shots.
The chapters of an episode come from `?chapters=12-15`, the queue entry, the gold file, or the automatic alignment,
in that order; the box in the header changes them.

Every change is saved at once to `align/gold/epNN.jsonl`, one row per reviewed panel:

```json
{"chapter": 12, "panel_order": 17, "shots": [{"episode": 3, "shot": 112}], "match": "manual", "confidence": 1.0,
 "episode": 3, "status": "confirmed", "auto_shots": [112, 113], "ts": "2026-11-11T09:12:44"}
```

`status` is `confirmed` or `no_match` (panel checked, no shot in this episode, `shots` empty). Panels never
reviewed are not written. `panel.image` may be a panel crop or a whole page; a page is cropped with `bbox`.

Try both pages without real data:

```bash
python -m tools.review.fixture --out /home/user/komi-poc-work/tmp/fx && python -m tools.align_label.fixture --out /home/user/komi-poc-work/tmp/fx
python -m tools.review.app --processed /home/user/komi-poc-work/tmp/fx --port 8301
python -m tools.align_label.app --processed /home/user/komi-poc-work/tmp/fx --port 8302
```

## 8. Tests

`/home/user/komi-poc-work/venvs/main/bin/python -m pytest tests -q` runs 47 tests in about one minute on CPU, with
synthetic clips only: statistics against known values, DuckDB round trip, blindness and resume of the A/B app, every
broken kind (hard-fail kinds must fail `technical_check`, the others must pass it), held-frame and cut detection,
Farneback flow metrics, Fréchet distance on Gaussians, review and labeling pages with and without data files.
RAFT, DOVER and I3D need weights and are not part of the test suite; section 10 has their sanity numbers.

## 9. Measured runtimes

All numbers are **CPU** (i7-14700K, capped at 8 threads) and were taken on 2026-11-10 while the machine was
oversubscribed by other jobs (load average 30 to 50 on 28 threads); the GPU was in use by another job the whole
time, so **nothing here was run on the GPU**. Clips: 854x480, 23.976 fps, 2.4 to 4 s (57 to 96 frames).

| Step | Setting | Time per clip (median, max) |
|---|---|---|
| `technical_check` | full decode at 320x176 + audio levels | 0.15 s, 0.47 s |
| `load_frames` | native 480p | 0.13 s, 0.35 s |
| `flicker_score` | | 0.18 s, 0.28 s |
| `detect_cuts` | | 0.11 s |
| flow, `farneback` | 240 px, 16 pairs at most | 1.05 s, 2.4 s |
| flow, `raft_small` | 240 px, 6 pairs at most, held pairs skipped | 1.8 s, 10.5 s (2.1 s median, 117 s max over all 61 clips) |
| flow at the **default** settings | 360 px, every pair | Farneback 0.16 to 0.19 s per computed pair (4 to 18 s per clip); RAFT-small 0.9 to 1.2 s per computed pair: 22 s for a 68-frame limited-animation shot, 109 s for a 95-frame shot that moves on every frame |
| `dover_score` | native size, 96 + 32 sampled frames | 20 s, 30 s (42 s median in a second run under higher load) |
| I3D features (FVD) | 16 frames, 224 px | 0.3 to 2.7 s per window, depending on the machine load (0.45 s typical) |
| `make_broken` / `make_real` | decode, effect, x264 + AAC encode, spec checks | about 1 s |

On CPU the default RAFT settings are too slow for a per-shot QA loop: pass `max_pairs=8, flow_height=240` (about
2 s) or `flow="farneback"` whenever the GPU is not available.

`python -m pytest tests -q`: 47 tests, 55 to 125 s depending on the load.
RAFT-large was only smoke-tested on CPU (two frame pairs). A whole 24-minute episode decodes in `technical_check`
in about one minute.

To repeat the model runs on the GPU once it is free (expected: same numbers up to float noise):

```bash
T=/home/user/komi-poc-work/data/interim/eval_test; O=/home/user/komi-poc-work/runs/eval_dev
scripts/gpu_run.sh 4 python -m src.eval.video_metrics $T/real_a --metrics motion,warp,dover --device cuda --out $O/gpu_metrics.json
scripts/gpu_run.sh 6 python -m src.eval.video_metrics $T/real_a --metrics motion,warp --flow raft_large --device cuda --out $O/gpu_raft_large.json
scripts/gpu_run.sh 4 python -m src.eval.fvd --real $T/fvd_big/A --gen $T/fvd_big/A2 --device cuda --no-cache --out $O/gpu_fvd.json
```

## 10. Sanity numbers (CPU, 2026-11-10 and 11)

Test material: single-shot clips cut from the ep01 and ep02 proxies (train episodes) with a rough ffmpeg scene-cut
list, re-encoded to x264 crf 18 + AAC: `real_a` (12 clips, ep01), `real_b` (12, ep02), `real_a2` / `real_b2`
(11 each, other shots). Everything is under `/home/user/komi-poc-work/data/interim/eval_test/`; the scripts that
produced the numbers are in `/home/user/komi-poc-work/runs/eval_dev/`. I did not look at any frame of the real
material (it would have left the machine); the effects were checked by eye on a synthetic scene only.

### 10.1 Real anime shots (46 clips)

| | median | p10 | p90 | max |
|---|---|---|---|---|
| `repeat_ratio` | 0.34 | 0.00 | 0.88 | 1.00 |
| `effective_fps` | 15.9 | 3.2 | 24.0 | 24.0 |
| `mean_abs_diff` | 0.86 | 0.06 | 7.66 | 22.6 |
| `luma_flicker` | 0.22 | 0.03 | 1.10 | 2.33 |
| `luma_flicker_p95` | 0.58 | 0.07 | 3.06 | 11.7 |
| `chroma_flicker` | 0.06 | 0.01 | 0.20 | 0.41 |

- 16 of 46 shots change on every frame (pans, zooms, fast action), 28 are limited animation, 2 are complete stills.
- In the 28 limited-animation shots the median `repeat_ratio` is 0.58 and the median `effective_fps` 10.2; drawings
  are held for 1 / 2 / 3 / 4+ frames in 43 % / 15 % / 36 % / 6 % of cases. A raw diffusion clip has
  `repeat_ratio` 0, so this is the number to compare the "on twos" retimer against.
- `technical_check(need_audio=True)` passes 44 of 46; the two failures are the two stills (`frozen_video`).
  On the full ep01 and ep02 proxies, 0 of 307 and 4 of 261 shots of 1 s or more are frozen by the rule, and about
  5 % are at least 90 % static.
- Cut detector: 3 of the 46 "single-shot" clips contain a detected cut; ffmpeg's scene score at those frames is 9.0
  to 11.4, just under the threshold of 12 I had used to cut the clips, so they are most likely real cuts. On all of
  ep01 (34,332 frames) `detect_cuts` finds 467 cuts; it recovers 413 of the 442 cuts that ffmpeg `scdet` reports at
  threshold 12 (93 %), and 437 of its 467 cuts have an ffmpeg cut at threshold 10 within one frame. This compares
  two heuristics; TransNetV2 shots are the better reference once they exist.

### 10.2 Hard technical failures (calibration set: 30 broken + 15 real, `--seed 0`)

`technical_check` with the label row's `expected_dur`, `need_audio`, `allow_static`:

| Group | Result |
|---|---|
| real (15; 3 of them (almost) stills, 2 without sound) | 15 of 15 pass |
| hard-fail kinds (9 clips: black, full freeze, silent_audio, truncated, corrupt_file) | 9 of 9 fail, with the expected code |
| other broken kinds (21 clips, including one partial freeze) | 21 of 21 pass, as intended: these are for the judge |

Reason codes seen: `black_video`, `frozen_video`, `no_audio_stream`, `wrong_duration`, `unreadable`,
`decode_errors` (+ `audio_decode_errors`, `truncated_stream` on the garbled file).

### 10.3 DOVER reproduction

Repo demo `17734.mp4`: technical branch -0.082 here vs -0.079 implied by the README's five dataset comparisons;
aesthetic branch 0.148 vs 0.148 (0.040 with today's default antialiased resize). The 15 real anime clips score
technical 0.50 (0.36 to 0.73), aesthetic 0.52 (0.28 to 0.74), overall 0.54 (0.33 to 0.69): mid-scale, as expected
for clean 480p material judged by a model trained on user video.

### 10.4 Each broken clip against its own source (source → broken)

30 broken clips of the calibration set, two per kind; the "source" is the same shot re-encoded untouched through
the same writer. Flow at 240 px (RAFT-small: 6 pairs at most, Farneback: 16); DOVER was run for eight of the kinds.

| Kind | repeat ratio | luma flicker | mean abs diff | warp error (RAFT-small) | warp error (Farneback) | global motion | hard cuts | DOVER technical | DOVER aesthetic |
|---|---|---|---|---|---|---|---|---|---|
| `flicker` | 0.57 → 0.00 | 0.24 → 58.91 | 0.24 → 34.03 | 0.12 → 37.52 | 0.13 → 32.86 | 0.01 → 0.12 | 0 → 10 | 0.48 → 0.18 | 0.70 → 0.64 |
| `flicker` | 0.00 → 0.00 | 0.50 → 42.44 | 6.93 → 27.54 | 6.01 → 27.92 | 5.08 → 22.91 | 0.14 → 0.11 | 0 → 15 | 0.65 → 0.50 | 0.60 → 0.58 |
| `noise_blur` | 0.00 → 0.00 | 0.25 → 0.25 | 0.36 → 4.57 | 0.54 → 7.46 | 0.43 → 7.44 | 0.04 → 0.10 | 0 → 0 | 0.50 → 0.05 | 0.45 → 0.03 |
| `noise_blur` | 0.71 → 0.00 | 0.19 → 0.26 | 1.29 → 3.19 | 0.54 → 3.53 | 0.64 → 3.56 | 0.01 → 0.05 | 0 → 0 | 0.69 → 0.02 | 0.68 → 0.02 |
| `jitter` | 0.00 → 0.00 | 0.36 → 0.44 | 16.19 → 32.68 | 6.55 → 7.61 | 5.46 → 6.14 | 0.24 → 6.69 | 0 → 0 | 0.46 → 0.42 | 0.41 → 0.43 |
| `jitter` | 0.00 → 0.00 | 2.30 → 3.61 | 18.09 → 23.23 | 8.42 → 8.12 | 8.29 → 8.32 | 7.26 → 9.24 | 0 → 0 | 0.41 → 0.36 | 0.37 → 0.34 |
| `hard_cut` | 0.87 → 0.96 | 0.01 → 2.34 | 0.05 → 1.39 | 0.02 → 0.04 | 0.04 → 0.03 | 0.00 → 0.00 | 0 → 1 | n/a | n/a |
| `hard_cut` | 0.00 → 0.00 | 0.16 → 0.43 | 6.14 → 12.99 | 2.99 → 5.03 | 2.27 → 3.87 | 1.26 → 1.11 | 0 → 1 | n/a | n/a |
| `frame_shuffle` | 0.73 → 0.38 | 1.79 → 7.45 | 2.48 → 7.31 | 0.31 → 0.96 | 0.48 → 3.00 | 0.05 → 0.32 | 0 → 0 | 0.59 → 0.51 | 0.62 → 0.64 |
| `frame_shuffle` | 0.10 → 0.00 | 0.29 → 0.79 | 0.96 → 2.69 | 0.63 → 1.17 | 0.62 → 1.35 | 0.07 → 0.12 | 0 → 0 | 0.50 → 0.48 | 0.48 → 0.48 |
| `melt` | 0.73 → 0.00 | 1.79 → 1.90 | 2.48 → 4.44 | 0.31 → 1.87 | 0.48 → 1.50 | 0.05 → 0.12 | 0 → 0 | 0.59 → 0.72 | 0.62 → 0.61 |
| `melt` | 0.57 → 0.00 | 0.24 → 0.35 | 0.24 → 0.77 | 0.12 → 0.67 | 0.13 → 0.43 | 0.01 → 0.06 | 0 → 0 | 0.48 → 0.57 | 0.70 → 0.66 |
| `freeze (partial)` | 0.00 → 0.57 | 0.50 → 0.24 | 6.93 → 3.10 | 6.01 → 3.14 | 5.08 → 2.24 | 0.14 → 0.08 | 0 → 0 | 0.65 → n/a | 0.60 → n/a |
| `freeze (full)` | 0.73 → 1.00 | 1.79 → 0.00 | 2.48 → 0.00 | 0.31 → 0.00 | 0.48 → 0.00 | 0.05 → 0.00 | 0 → 0 | 0.59 → n/a | 0.62 → n/a |
| `garbled_text_overlay` | 0.64 → 0.32 | 0.68 → 0.67 | 0.92 → 1.26 | 0.53 → 0.85 | 0.27 → 0.69 | 0.03 → 0.03 | 0 → 0 | 0.55 → 0.56 | 0.74 → 0.72 |
| `garbled_text_overlay` | 0.63 → 0.41 | 0.03 → 0.12 | 0.50 → 0.99 | 0.80 → 1.21 | 0.55 → 1.02 | 1.06 → 1.06 | 0 → 0 | 0.69 → 0.64 | 0.62 → 0.63 |
| `ghost` | 0.71 → 0.68 | 0.19 → 0.14 | 1.29 → 1.00 | 0.54 → 0.61 | 0.64 → 0.76 | 0.01 → 0.01 | 0 → 0 | 0.69 → 0.56 | 0.68 → 0.54 |
| `ghost` | 0.57 → 0.80 | 0.24 → 0.21 | 0.24 → 0.24 | 0.12 → 0.05 | 0.13 → 0.12 | 0.01 → 0.00 | 0 → 0 | 0.48 → 0.38 | 0.70 → 0.69 |
| `color_shift` | 0.00 → 0.00 | 0.36 → 0.37 | 16.19 → 15.74 | 6.55 → 6.32 | 5.46 → 5.44 | 0.24 → 0.24 | 0 → 0 | 0.46 → 0.51 | 0.41 → 0.37 |
| `color_shift` | 0.81 → 0.72 | 0.16 → 0.17 | 0.23 → 0.26 | 0.19 → 0.25 | 0.14 → 0.18 | 0.08 → 0.09 | 0 → 0 | 0.28 → 0.34 | 0.47 → 0.40 |

Range over the 31 untouched clips (15 real copies + 16 sources): `luma_flicker` 0.01 to 2.30, warp error
(RAFT-small) 0.01 to 8.42, `global_motion_mag` 0.00 to 7.26, hard cuts 0 in all; DOVER technical 0.28 to 0.73.
RAFT-small and Farneback warp errors agree (correlation of the log values 0.99 over 43 clips).

What this means for the judge's evidence bundle:

| Failure | Seen by the local video metrics? |
|---|---|
| black, full freeze, missing / silent audio, truncated, corrupt file | yes, `technical_check` fails them (9 of 9) |
| flicker | yes: `luma_flicker` 0.2 to 0.5 becomes 42 to 59, warp error 0.1 to 6 becomes 28 to 38 |
| noise + blur | yes: DOVER technical drops to 0.02 to 0.05 (untouched clips: 0.28 to 0.73), held frames disappear, warp error rises 6 to 14 times |
| jitter | yes when the shot had no pan: `global_motion_mag` 0.24 becomes 6.7 (a shot that already pans: 7.3 becomes 9.2) |
| cut inside a shot | yes: `n_hard_cuts` 0 becomes 1 |
| partial freeze | only as `longest_static_run_s` / `repeat_ratio` (technical check passes it by design) |
| shuffled frames, melting | only relative to the source. Shuffle: warp error and `luma_flicker` rise 2 to 6 times. Melt: warp error rises about 6 times and every held frame disappears (`repeat_ratio` 0.6 to 0.7 becomes 0). All of it stays inside the range of real shots |
| ghosting, wrong colours, garbled text | no: all values stay in the real range (ghosting lowers DOVER technical by 0.10 to 0.13 against its source, garbled text halves `repeat_ratio`). These need the VLM captions / frames and identity scores |
| wrong audio | no (video metrics are identical): needs transcript / speaker evidence |

### 10.5 FVD

**Checks that passed.** The I3D file has the sha256 published on the HF Hub copies. A set against its own
re-encode gives FVD 5.2. Features are cached and reproducible (second run: 0.1 s, same value).

**Small sets (12 clips, 31 to 38 windows per set, default protocol).** The requested property, real-vs-real much
lower than real-vs-corrupted, does **not** hold at this size when the two sets show different shots:

| Comparison | FVD | KVD |
|---|---|---|
| real vs real, different shots (4 pairs of sets) | 1145, 1253, 1279, 1363 | 16.0 to 29.4 |
| real ep02 vs corrupted ep01 clips: color_shift / frame_shuffle / noise_blur | 1288 / 1279 / 1315 | 31.3 / 21.4 / 49.2 |
| real ep02 vs corrupted ep01 clips: jitter / melt / flicker | 1355 / 1537 / 1925 | 36.9 / 47.3 / 175.5 |
| ep01 clips vs the same clips corrupted: frame_shuffle / jitter / noise_blur | 500 / 497 / 457 | 2.3 / 11.3 / 20.8 |
| ep01 clips vs the same clips corrupted: color_shift / flicker / melt | 646 / 957 / 1267 | 10.5 / 115.6 / 53.4 |
| ep01 clips vs their own re-encode | 5.2 | -14.2 |

With about 36 windows in a 400-dimensional feature space the distance between any two sets of different shots sits
at 1150 to 1360; only strong flicker rises clearly above that band, whole-frame melting a little. The corruption is
visible when the content is the same (5 against 457 to 1267). KVD orders the cross-content cases a little better but
frame_shuffle (21.4) is inside the real-vs-real range. The negative KVD of the last row is the known behaviour of
the unbiased estimator when both sets contain the same samples.

**Larger sets (100 independent windows per set).** Design fixed before the results were seen: 1.25 s clips cut at
398x224, one window per clip; `A` = ep01 at t = 120 + 10k s, `A2` = ep01 at t = 125 + 10k s (other footage of the
same scenes), `B` = ep02 at t = 120 + 10k s, `X` = `A2` with one corruption applied to every clip
(`runs/eval_dev/fvd_big.py`). One run, CPU.

| Comparison | FVD | KVD |
|---|---|---|
| real vs real: `A` vs `A2` (same episode) | 499 | -0.9 |
| real vs real: `A` vs `B`, `A2` vs `B` (other episode) | 707, 627 | 10.7, 10.4 |
| `A` vs `X` noise_blur | 1427 | 150.8 |
| `A` vs `X` melt (whole frame) | 1397 | 63.9 |
| `A` vs `X` flicker | 957 | 65.6 |
| `A` vs `X` color_shift (hue 120°) | 730 | 19.3 |
| `A` vs `X` jitter | 669 | 15.9 |
| `A` vs `X` frame_shuffle (whole clip) | 581 | 5.3 |

Against its own footage (`A2` vs `X`): noise_blur 1257, melt 1276, flicker 705, color_shift 518, jitter 379,
frame_shuffle 327.

**Result: the sanity property holds for two of six corruptions, not in general.**

- Noise + blur and whole-frame melting sit at 2.8 times the same-episode baseline and about twice the
  other-episode baselines.
- Flicker is 1.9 times the same-episode baseline but only 1.35 to 1.5 times the other-episode ones. KVD separates
  it better (65.6 against 10.7 at most).
- The other three are **not** separated from real-vs-real of another episode (627 and 707): a 120° hue shift (730)
  is level with the higher of the two, jitter (669) lies between them, a full frame shuffle (581) is below both.
  The shuffled copy of `A2` is closer to `A2` (327) than any real set is to another real set.

So at these sizes FVD reacts to damage of the picture itself and hardly to temporal order, shake or a palette
change, and real footage from another episode already scores as high as three of the corrupted sets. Whether
those three separate with thousands of windows was not tested.

Sample size changes the value strongly: on random subsets of the same features, FVD(`A`, `A2`) is 1129, 759
and 499 for 36, 60 and 100 windows, and FVD(`A`, `X` noise_blur) 2194, 1850 and 1427. The values are still falling
at 100, so an FVD is only comparable with another one computed with the same protocol and the same number of
windows. The corruption strengths of this run are relative to 224 px clips and differ from the 480p run above, so
the two tables are not a clean comparison of sample sizes; the subset numbers are.

## 11. Limits and what is not verified

- **GPU**: no model here has run on the GPU yet (the card was busy). The CUDA paths are plain `.to(device)` code,
  but they are untested; section 9 has the three commands to check them.
- **Real frames were never inspected by eye.** Broken clips were checked numerically (effect size, metrics) and
  visually on a synthetic scene only. Look at a handful of `melt`, `ghost` and `garbled_text_overlay` clips made
  from real shots before using them for calibration.
- **Held-frame detection depends on encode quality.** The thresholds fit x264 crf 20 or better. The same on-threes
  shot gives `repeat_ratio` 0.64 at crf 12 and crf 18, 0.66 upscaled to 720p, but 0.47 at crf 28 and 0.53 at
  360p crf 23, because compression noise makes held frames differ. Compare clips encoded at similar quality, or
  pass looser `hold_mad` / `hold_block`.
- **Thresholds come from 46 clips of ep01 and ep02** (holds, cuts, frozen, silence). They were not re-fitted on
  other episodes or on generated video.
- **`frozen_video` fails real stills** unless `allow_static=True` is passed for shots whose spec is a still (0 to
  4 % of real shots are complete stills, about 5 % nearly so). Two stills joined by a cut are not reported as frozen.
- **`warp_error` mixes held and moving pairs.** A shot with many held frames scores lower. Use `warp_error_moving`
  to compare shots with different hold ratios, and read both against the motion level: real shots range from
  0.01 to 8.4, so a single threshold cannot separate good from bad.
- **`detect_cuts` reports isolated large changes.** Strong flicker is reported as many cuts (10 to 15 in the
  flicker clips); dissolves and fades are not detected.
- **DOVER** was trained on natural user video and looks at 32 to 96 sampled frames. It sees blur and noise, and
  nothing else in this test (melt, ghost, wrong colours, garbled text and shuffled frames all score like real
  shots). It needs clips of at least a few frames; short clips are sampled with wrap-around.
- **FVD was only tested with 36 and 100 windows per set**, far below the 400 feature dimensions and the usual
  2,048 clips. There it detects noise, blur and melting, and misses shuffled frames, jitter and a hue shift (10.5).
  Always pass `--baseline`, read KVD next to FVD, and compare only runs with the same protocol and the same number of
  windows. Do not read it as a motion-quality score. Only the first 64 s of a long video are used.
- **Blind A/B**: clips that differ in length, resolution or frame rate can still give the method away; cut both
  sides of a pair to the same length and size. `--transcode always` equalises codec and bitrate only. Raters are
  identified by a typed name (no login). The page was driven in headless Chromium (WebM, and H.264 in the Ubuntu
  Chromium build); Firefox and a human rating session are untested.
- **Review and labeling pages** were built against the schemas in the brief and tested on synthetic fixtures. The
  labeling page was also pointed at the real `panels.jsonl` (115 rows) and `alignment.jsonl` (106 rows) that exist
  now: both load, `bbox` is `[x, y, w, h]` and `image` is already the panel crop, as assumed. `shots.jsonl`, the
  queue and both `clusters.json` files did not exist yet, so no real episode has been labeled or reviewed.
- **`broken.py` and `load_frames` hold a whole clip in memory**; they are meant for shots of a few seconds
  (a 5 s 720p clip is about 330 MB).
- `tools` has no `__init__.py`; `python -m tools.ab_page.app` works because nothing else on the path is called
  `tools`. An empty `tools/__init__.py` would make that robust.
