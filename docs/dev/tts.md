# TTS: Japanese character voices (zero-shot cloning, duration fitting, voice metrics)

Workstream files: `src/generate/tts.py`, `src/train/voice_bank.py`, `src/eval/voice_metrics.py`, `configs/tts.yaml`
(read as `CFG.tts.*`). Everything runs locally; no audio leaves the machine.

| What | Choice |
|---|---|
| Model | **IndexTTS-2.5** (`IndexTeam/IndexTTS-2.5`, zh/en/**ja**/es/ar, zero-shot cloning from one clip, 8-dim emotion vector, `duration_factor` speed control), weights in `models/tts/IndexTTS-2.5` |
| Code | `third_party/index-tts` pinned at `d9e41aac89fd00b3d71497fddb287b7f24613712` (2026-09-30), inference only (the repo ships no training code) |
| Env | `venvs/tts` (Python 3.11.13, torch 2.8.0+cu128): `UV_PROJECT_ENVIRONMENT=$KOMI_WORK/venvs/tts uv sync --frozen` in `third_party/index-tts` (exact repo lock), then `uv pip install speechbrain faster-whisper pysubs2 pykakasi "protobuf==3.19.6"` (the protobuf pin keeps the locked env untouched) |
| Auxiliary models | w2v-bert-2.0, MaskGCT semantic codec, CAMPPlus, BigVGAN v2 22 kHz in `models/tts/IndexTTS-2.5/hf_cache/` (downloaded once, then `HF_HUB_OFFLINE=1`); ECAPA speaker model in `models/tts/spkrec-ecapa-voxceleb` |
| Fallbacks (not needed so far) | GPT-SoVITS, CosyVoice 3 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`), F5-TTS |

## 1. Usage

```bash
. /home/user/komi-poc/scripts/env.sh
PY=$KOMI_WORK/venvs/tts/bin/python

# (once) auxiliary model download
$PY -m src.generate.tts --prepare

# voice bank: per-character reference clips (CPU works: ~3-5 s per candidate clip on the loaded machine;
# for the whole cast use `scripts/gpu_run.sh 2 $PY -m src.train.voice_bank --device cuda`)
$PY -m src.train.voice_bank                       # reads data/processed/anime/lines_named.jsonl
#   -> data/processed/voices/<Speaker>/ref_00.wav.. + meta.json + centroid.npy, data/processed/voices/voices.json

# synthesis (GPU jobs always through gpu_run.sh; ~8 GB admission)
scripts/gpu_run.sh 8 $PY -m src.generate.tts --in lines.jsonl --out-dir runs/<run>/tts
#   -> runs/<run>/tts/<id>.wav, tts_results.jsonl, tts_run.json

# metrics (ASR on GPU, or --device cpu)
scripts/gpu_run.sh 6 $PY -m src.eval.voice_metrics --results runs/<run>/tts/tts_results.jsonl \
    --voices data/processed/voices/voices.json --device cuda
#   -> runs/<run>/tts/voice_metrics.json
```

From pipeline code in any venv (spawns the TTS venv through `gpu_run.sh`, returns the result rows in input order):

```python
from src.generate.tts import synthesize
from src.train.voice_bank import refs_for          # default ordered refs of a character ([] if unknown)
rows = synthesize([{"id": "c140_s023_l0", "speaker": "Tadano", "text": "古見さん、おはようございます。", "lang": "JA",
                    "ref_wavs": refs_for("Tadano"), "emotion": "shy", "target_dur": 2.8, "max_dur": 3.5}],
                  out_dir="runs/x/tts")
```

Useful flags of `src.generate.tts`: `--device cpu --threads 8` (no GPU; slow), `--no-cache`, `--fresh` (overwrite
`tts_results.jsonl` instead of merging by id), `--set KEY=VALUE` (config override, e.g. `--set tts.ref.mode=first`,
`--set tts.emotion.strength=0.8`), `--show-text "…"` (prints the normalised text and mora count).

### Input line

```json
{"id": "c140_s023_l0", "speaker": "Tadano", "text": "日本語のセリフ", "lang": "JA",
 "ref_wavs": ["/abs/ref1.wav", "..."], "emotion": "shy", "target_dur": 2.8, "max_dur": 3.5}
```

* `emotion`: a label (table in `configs/tts.yaml`: shy, happy, angry, sad, scared, surprised, calm, excited, embarrassed,
  deadpan, nervous, panicked, tsukkomi, gentle, ... plus aliases; combinations `"shy, happy"`, intensity words
  `"slightly nervous"`, `"very angry"`), an explicit 8-float vector in IndexTTS order
  `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, a dict `{"happy": 0.5}`, or `null` (prosody comes
  from the reference clip). Unknown labels fall back to neutral and are reported in `warnings`.
* Optional: `"seed"` (re-roll a take: the repair ladder can ask for another delivery), `"emo_ref"` (an emotional clip of
  the character used as emotion reference when no label/vector is given; wired to IndexTTS `emo_audio_prompt`, not
  exercised in any test yet), `"real_wav"` (ground truth, metrics only).

### Result row (`tts_results.jsonl`)

Required keys `id, path, dur, raw_dur, fit, ok, error`, plus `fit_ok, dur_err, over_max, duration_factor, atempo,
min_feasible_dur/max_feasible_dur` (when the fit failed), `emotion_resolved, emo_vector, seed, n_mora, sec_per_mora,
synth_sec, gpt_sec, takes, rtf, warnings, cached, key, prompt_wav, prompt_sec, text_norm` and the input fields.

* `ok=false` means no usable audio: bad input, no speakable text (`"……"` lines: skip them downstream), model error, or
  the take failed the sanity check after all re-rolls (`sanity_failed: too_long/too_short/runaway/silent`; the best take
  is still written so it can be auditioned).
* `fit`: `none` (no target/max given) | `native` (natural take already fits, or it was left untouched) | `speed`
  (IndexTTS `duration_factor`) | `stretch` (additional ffmpeg `atempo`). `fit_ok=false` = audio exists but misses
  `target_dur` by more than the tolerance or exceeds `max_dur`; `min_feasible_dur` tells the director how short this
  text can get, so the repair is "shorten the line", not "retry".
* Output wavs: mono, 44.1 kHz, 16-bit, silence trimmed (20 ms / 60 ms pads), 4/12 ms fades, peak -3 dBFS. The model's
  native rate is 22.05 kHz (upsampled 2x; there is no content above 11 kHz).

## 2. How it works

**Text front-end (Japanese).** NFKC; ruby `漢字(かんじ)` -> reading; brackets/decorations removed; name lexicon
(`tts.lexicon_ja`: 古見 -> こみ, 只野 -> ただの, 長名 -> おさな, ...; MeCab/unidic reads 古見 as "フルミ", so names must be
forced); digits -> kanji numerals; `え〜` -> `えー`; dashes/ellipses unified to `…`; spaces between Japanese characters
become `、`; terminal `。` added. IndexTTS has no Japanese number/symbol normaliser of its own.

**Speaker prompt.** IndexTTS takes one clip (truncated at 15 s). `ref_wavs` are trimmed, level-matched and concatenated
in order up to 12 s (`tts.ref.mode=concat`), cached by content hash in `cache/tts/refs/`.

**Emotion.** Label -> 8-dim vector, scaled by `tts.emotion.strength` (0.65) and IndexTTS's own per-dimension bias,
sum capped at 0.8 (model rule). Inside the model the vector replaces part of the prosody embedding taken from the
reference (`emovec = sum(w_i * e_i) + (1 - sum(w)) * emovec_ref`), so more strength means less of the reference's own
delivery; what that costs in speaker similarity is measured by the "no emotion vectors" ablation of the GPU test. The
bundled QwenEmotion text-to-emotion model (1.2 GB of weights) is not
loaded: the director already gives a label and a table is deterministic.

**Duration fitting.**
1. Native take at speed 1.0 -> `raw_dur` (trimmed).
2. If `target_dur` is missed by more than 60 % of the tolerance (tolerance = max(5 %, 80 ms)) or `max_dur` is exceeded:
   re-decode with `duration_factor = target/raw`, clamped to 0.75-1.30. The GPT stage (text -> semantic codes) is
   memoised per (prompt, emotion, seed, text), so this only re-runs the flow-matching decoder + vocoder on identical
   content (same words, same delivery, different speed); one refinement pass if needed. `tts_run.json` reports the
   GPT share of the compute.
3. Residual error -> pitch-preserving `atempo` stretch, clamped to 0.80-1.25 of the duration.
4. Limits: total range ~0.60x-1.63x of the natural duration. A line that would have to be slowed by more than 1.63x is
   left at natural speed (`fit_ok=false`, warning "pad with silence"): a short line in a long shot is natural, a
   slow-motion line is not. A line that cannot be compressed enough is compressed to the limit and flagged.

**Sanity check / short interjections.** Expected duration bounds come from the mora count (MeCab reading):
`0.045 s/mora <= raw_dur <= 0.35 s/mora + 1.5 s (+0.5 s per "…")`; the GPT token budget is capped accordingly (for
"えっ": 161 tokens = 3.2 s of audio instead of the model default of 1500 tokens = 30 s), so a runaway generation is
cut early and detected. A take outside the bounds, silent, or cut by the cap is
re-rolled with seed+1 (up to 3 times); the best take wins.

**Determinism and cache.** Seed = `tts.seed` + CRC of (speaker, normalised text) unless the line carries `seed`; the
decoder noise is re-seeded independently of the GPT stage. Finished lines are cached in `cache/tts/<key>.wav` keyed by
text, prompt content, emotion vector, target/max, seed, sampling/fit/trim config, model files and precision; a fully
cached batch never loads the model.

**Voice bank** (`src/train/voice_bank.py`). Candidates are single lines or runs of adjacent lines of one speaker
(gap <= 0.8 s), 3-12 s, from TRAIN episodes only (asserted), not within 0.15 s of any other dialogue line (named lines
and every dialogue subtitle). Each candidate is cut from `vocals.wav` when present (else `audio_jpn.wav`), trimmed and
scored: speaker consistency (ECAPA cosine to a robust per-character centroid, which also rejects wrong labels), an SNR
proxy (95th - 10th percentile frame level; low = music/SFX bleed), duration (5-10 s ideal), speech ratio, label
confidence. The best time-diverse clips become `ref_00..07.wav`; the top 3 are `default_refs`. `--transcribe` adds a
Japanese transcript per ref (needed by GPT-SoVITS / CosyVoice / F5 if we ever switch engine; IndexTTS needs none).

**Metrics** (`src/eval/voice_metrics.py`).
* `sim_ref`: cosine between ECAPA-TDNN embeddings (SpeechBrain `spkrec-ecapa-voxceleb`, non-gated) of the generated
  wav and the mean embedding of the prompt refs; `sim_bank` vs the voice-bank centroid; `sim_real`/`real_vs_ref` when a
  ground-truth clip is given. The report includes anchors measured on real clips (same character / different
  characters) because absolute cosine values of a VoxCeleb model on anime voices are not self-explanatory.
* `cer`: faster-whisper large-v3 (`language=ja`, beam 5) transcript vs the script. **CER, not WER**: Japanese has no
  word boundaries and WER would depend on the tokenizer. Both sides go through the TTS text front-end (names, numbers)
  and are compared in three forms with punctuation removed: written (`cer_surface`), kana pronunciation via
  MeCab/unidic-lite (`cer_mecab`) and kana reading via pykakasi (`cer_kakasi`). `cer` is the lowest of the three: a
  transcript that matches the script in any form was read correctly. One form alone is not reliable: Whisper wrote
  "コミュ障" for the script's "コミュ症" (written-form CER 0.09) and unidic-lite reads 障 as "さわり" (MeCab CER 0.23),
  while the audio was right (pykakasi CER 0.00).
* Duration: `dur_err`, `dur_rel_err`, `fit_ok`, `over_max`.
* `flags` (`low_sim`, `high_cer`, `dur_miss`, `over_max`) are evidence for the judge, thresholds in
  `tts.voice_metrics`. Sub-second clips are marked `sim_unreliable` / `cer_unreliable`.

## 3. Measured numbers

### 3.1 CPU validation (2026-11-10, GPU was occupied by another job)

The checks below ran on CPU (8 threads, fp32) while the machine was heavily loaded (load average ~50-65 on 28
cores), so the timings are upper bounds and say nothing about GPU speed. Only two utterances went through the real
model; similarity/CER numbers here are single examples, not statistics.

<!-- CPU_RESULTS_START -->
| Check | Result |
|---|---|
| Model load, CPU fp32, offline (`HF_HUB_OFFLINE=1`) | 79.5 s |
| `n01_narration` "古見さんは、コミュ症です。" (Narrator, 6.0 s ref, `narration`, target 2.6 s) | raw 2.426 s -> 2.601 s with model speed 1.0715 (2 takes, error +1 ms). ASR: "コミさんはコミュ障です", CER 0.00. `sim_ref` 0.752, `sim_bank` 0.727 |
| `t04_short_surprised` "えっ" (Tadano, 11.8 s concatenated refs, `surprised`, max 1.0 s) | 0.917 s, first take passed the sanity check, no fitting needed. ASR: "ええっ!?" (CER 0.5 = 1 edit on 2 characters, marked `cer_unreliable`). `sim_ref` 0.296 (marked `sim_unreliable`, < 1 s) |
| Same narrator line forced to 1.6 s (0.66x, `n01_narration_fast`) | model speed 0.75 + `atempo` 1.139 -> 1.613 s (error +13 ms, `fit=stretch`). ASR unchanged ("コミさんはコミュ障です。", CER 0.00); `sim_ref` drops from 0.752 to 0.665. GPT stage reused from the memo (0 s) |
| Determinism: `n01_narration` synthesised again in two new processes (`--no-cache`) | same length; 2 of 114 688 samples differ by 1 LSB of 16 bit (max abs diff 3.1e-5): identical delivery, not bit-identical on multi-threaded CPU |
| Similarity anchors (real clips, full mix) | same character, ref vs ref (Tadano): 0.519; different characters (Tadano vs narrator): 0.145 |
| Compute on CPU | first run (load average ~60): 706 s for 3.52 s of speech; second run (load ~30): 341 s for 4.21 s (81 s per second of speech, GPT stage 2.8 % of it: on CPU the flow-matching decoder dominates). Not representative of the GPU |
| Cache rerun of the same JSONL | 2/2 from cache, 0.07 s, model not loaded, byte-identical wavs |
| Bad lines through the CLI ("……", missing ref, no refs, negative target, invalid JSON) | 5/5 error rows, batch not interrupted, model not loaded |
| `python -m src.generate.tts_selftest` | 40 checks pass (text front-end, emotion mapping, every fitting branch, re-rolls, cache, merging) |
| Voice bank from 30 hand-labelled episode-1 lines (full mix) | Tadano: 10 candidates, 9 clean, 6 consistent, 2 refs written; Narrator: 2 candidates, 1 ref. 63 s on CPU (use `--device cuda` through `gpu_run.sh 2` for the full cast) |
| faster-whisper large-v3, CPU int8, 8 threads | 140 s of episode 1 in 368 s (+17 s load); metrics CLI on the 2 lines: 47 s |
<!-- CPU_RESULTS_END -->

### 3.2 GPU end-to-end test (queued; fills in automatically)

`runs/tts_test/run_gpu_test.sh` (tmux session `tts_gpu_test`) waits for the card through
`GPU_WAIT_MIN=900 scripts/gpu_run.sh 8 ...` and then runs: 16 test lines (12 hand-written lines for Tadano and the
narrator with different emotions/targets, 4 "reconstruction" lines = real held-out lines re-synthesised with the real
duration as target), a determinism rerun, a cache rerun, a robustness batch, two ablations (single reference clip; no
emotion vectors) and the metrics. References: `runs/tts_test/voices/` (built by `voice_bank.py` from a hand-labelled
JSONL of episode-1 subtitle timings, **full mix**, because `vocals.wav` did not exist yet). Results:
`runs/tts_test/SUMMARY.md|json`, samples `runs/tts_test/gpu/*.wav`, log `logs/tts_gpu_test.log`, status
`runs/tts_test/GPU_TEST_STATUS`. The block below is rewritten by `runs/tts_test/summarize.py` when the test finishes.

<!-- GPU_RESULTS_START -->
_Measured on the local RTX 4090 by the queued test, 2026-11-11 03:20:24 (`/home/user/komi-poc-work/runs/tts_test/SUMMARY.json`, samples in `/home/user/komi-poc-work/runs/tts_test/gpu/*.wav`; device cuda:0, bf16)._

| Metric | Value |
|---|---|
| Lines synthesised / ok | 16 / 16 of 16 |
| Compute per second of speech (all takes incl. fitting re-decodes) | 1.316 s/s |
| Takes (native + re-rolls + speed re-decodes) / share of compute in the GPT stage | 30 / 0.596 |
| Model load | 19.950 s |
| Peak VRAM (torch allocated / reserved; charged to the process by nvidia-smi at the end of the batch) | 5.700 / 6.020 GB; 6328 MiB |
| Speaker similarity to the prompt refs `sim_ref` (mean / median / min) | 0.614 / 0.651 / 0.137 |
| ... lines >= 1 s only | 0.646 |
| Anchor: real clip vs refs, same character (reconstruction lines) | 0.697 |
| Anchor: real refs, different characters | 0.145 |
| Generated vs the real recording of the same line `sim_real` | 0.632 |
| CER (best of written / MeCab / pykakasi form; mean per line / micro over characters) | 0.181 / 0.038 |
| ... lines with >= 4 reference characters only (mean) | 0.028 |
| CER, written form only (mean; penalises kanji/kana spelling) | 0.212 |
| Duration error, lines with target (mean abs s / mean rel) | 0.008 s / 0.003 |
| Lines within tolerance / over max_dur | 1.000 / 0 |
| Fit modes | {'speed': 13, 'native': 1, 'none': 1, 'stretch': 1} |
| Determinism (same input, second process, fresh synthesis) | 16/16 bit-identical, 16/16 within 2 LSB of 16 bit, 16/16 same length, max abs sample diff 0.00000 |
| Cache rerun | 16/16 from cache, model load 0.000 s, wall 0.020 s |

Ablations (same 16 lines):

| Variant | sim_ref mean | sim_real mean | CER mean | dur fit rate |
|---|---|---|---|---|
| default (2 refs concatenated, emotion labels) | 0.614 | 0.632 | 0.181 | 1.000 |
| single reference clip | 0.600 | 0.671 | 0.098 | 1.000 |
| no emotion vectors | 0.667 | 0.632 | 0.280 | 1.000 |

Per line (default run):

| id | emotion | target/max | raw -> dur | fit (speed, atempo) | sim_ref | CER | ASR | flags |
|---|---|---|---|---|---|---|---|---|
| t01_calm_intro | calm | 4.6/5.2 | 5.828 -> 4.598 | speed (0.7893, 1.0) | 0.5975 | 0.0312 | 僕の名前はただの人々ごく普通の高校1年生です |  |
| t02_gentle | gentle | 3.2/None | 4.272 -> 3.204 | speed (0.75, 1.0) | 0.5873 | 0.0 | コミさんは人と話すのが苦手なんですね |  |
| t03_panicked_fast | panicked | 2.6/3.0 | 2.914 -> 2.601 | speed (0.8922, 1.0) | 0.5853 | 0.0 | ちょ、ちょっと待ってくださいそれは誤解です |  |
| t04_short_surprised | surprised | None/1.0 | 0.809 -> 0.809 | native (1.0, 1.0) | 0.1366 | 0.5 | え? |  |
| t05_short_plain | None | None/None | 1.115 -> 1.115 | none (1.0, 1.0) | 0.2723 | 2.0 | はぁ!? | low_sim,high_cer |
| t06_happy_slow | happy | 3.4/None | 3.274 -> 3.39 | speed (1.0385, 1.0) | 0.6211 | 0.0 | よかった本当によかったです |  |
| t07_sad_max | sad | None/4.0 | 4.423 -> 3.878 | speed (0.8772, 1.0) | 0.6596 | 0.0 | すみません全部僕のせいです |  |
| t08_tsukkomi | tsukkomi | 1.8/None | 2.276 -> 1.8 | speed (0.791, 1.0) | 0.6422 | 0.0 | なんで、そうなるんですか!? |  |
| t09_vector_afraid | [0, 0, 0, 0.6, 0, 0, 0.2, 0] | 3.0/None | 2.38 -> 2.953 | speed (1.2606, 1.0) | 0.5963 | 0.0625 | ホミさんもしかして怒ってますか? |  |
| t10_shy_embarrassed | shy, embarrassed | 3.6/4.2 | 3.39 -> 3.599 | speed (1.0619, 1.0) | 0.6981 | 0.0 | あのよかったら一緒に帰りませんか |  |
| n01_narration | narration | 2.6/None | 2.345 -> 2.612 | speed (1.1086, 1.0) | 0.7145 | 0.0 | コミさんはコミュ障です |  |
| n02_narration_long | narration | 6.5/7.5 | 6.304 -> 6.502 | speed (1.0311, 1.0) | 0.7362 | 0.0303 | コミュ障とは人付き合いを苦手とする症状またはその症状を持つ人を指す |  |
| r01_recon_tadano | None | 5.12/None | 4.911 -> 5.12 | speed (1.0426, 1.0) | 0.7542 | 0.0 | 僕は知っている周りの空気を読んで目立たず生きていく |  |
| r02_recon_tadano | None | 2.58/None | 3.831 -> 2.592 | stretch (0.75, 1.1115) | 0.7226 | 0.0 | それが幸せへの近道なのだ |  |
| r03_recon_tadano | None | 5.29/None | 4.667 -> 5.294 | speed (1.1334, 1.0) | 0.7675 | 0.0 | まあ僕みたいな凡人がこれ以上関わることもないだろう |  |
| r04_recon_narrator | None | 2.5/None | 3.065 -> 2.508 | speed (0.8157, 1.0) | 0.7374 | 0.2667 | ただの君は子ごこん者です | high_cer |

Prosody proxy (does the emotion vector change the delivery? F0 median Hz / spread in semitones):

| id | emotion | with emotion | without |
|---|---|---|---|
| t01_calm_intro | calm | 191.4 / 5.98 | 178.5 / 6.89 |
| t02_gentle | gentle | 219.1 / 6.61 | 167.6 / 7.81 |
| t03_panicked_fast | panicked | 302.5 / 7.14 | 200.5 / 8.93 |
| t04_short_surprised | surprised | 184.6 / 7.19 | 153.3 / 7.11 |
| t06_happy_slow | happy | 257.4 / 7.48 | 177.4 / 7.3 |
| t07_sad_max | sad | 207.2 / 10.41 | 164.8 / 9.39 |
| t08_tsukkomi | tsukkomi | 205.7 / 5.31 | 176.4 / 6.93 |
| t09_vector_afraid | [0, 0, 0, 0.6, 0, 0, 0.2, 0] | 260.9 / 5.96 | 260.9 / 5.96 |
| t10_shy_embarrassed | shy, embarrassed | 249.8 / 7.79 | 160.8 / 7.64 |
| n01_narration | narration | 229.6 / 5.77 | 196.3 / 6.08 |
| n02_narration_long | narration | 228.8 / 5.94 | 210.0 / 6.01 |

Robustness batch: `bad_silent_line` -> error: ValueError: no_speakable_text (silent line: nothing to synth; `bad_missing_ref` -> error: FileNotFoundError: reference wav missing: /nonexistent/ref.w; `bad_no_refs` -> error: ValueError: no ref_wavs given; `bad_target` -> error: ValueError: target_dur must be a positive number, got -1; `ok_unknown_emotion` -> ok; `line00005` -> error: ValueError: line 5: invalid JSON (Expecting property name en
<!-- GPU_RESULTS_END -->

### 3.3 Full-mix references vs separated references (queued; fills in automatically)

`runs/tts_test/run_vocals_compare.sh` (tmux session `tts_vocals_wait`) waits until the GPU test above is done and the
audio workstream has written `data/interim/anime/ep01/vocals.wav`, rebuilds the same hand-labelled voice bank from the
separated stem (`runs/tts_test/voices_vocals/`), synthesises the same 16 lines (`runs/tts_test/gpu_vocals/`) and
scores both runs against one common yardstick: the mean ECAPA embedding of real held-out clips of the character cut
from `vocals.wav` (clips neither run used as a prompt). Results: `runs/tts_test/SUMMARY_vocals.md|json`, log
`logs/tts_vocals_compare.log`, status `runs/tts_test/VOCALS_TEST_STATUS` (gives up after 20 h).

<!-- VOCALS_RESULTS_START -->
(not run yet: waiting for vocals.wav and the GPU)
<!-- VOCALS_RESULTS_END -->

## 4. Limits and caveats

* **Nobody has listened to the samples yet.** Similarity, CER and duration are measured; naturalness, pitch accent,
  whether "shy" sounds shy, and whether the clone is convincing need ears (`runs/tts_test/gpu/*.wav` next to
  `runs/tts_test/real/*.wav` and the refs).
* **Music-contaminated references.** The test refs come from the full mix (BGM under the monologue; SNR proxy of the
  chosen clips 12-20 dB). A prompt-conditioned model reproduces what is in the prompt, so music/reverb in the reference
  is expected to come back as noise or colouring and to cost timbre fidelity; this has not been measured against clean
  references yet. Rebuild the bank from `vocals.wav` once separation has run (`voice_bank.py` prefers it automatically)
  and compare `sim_ref`/`sim_real` on the same test lines. Separation residue would be cloned as well.
* **Very short lines** ("えっ", "あ…", "はい"): the known weak spot of autoregressive TTS (swallowed, repeated or run-on
  output). Mitigations: token cap, mora-based sanity check, re-rolls. The one case run so far ("えっ", CPU) passed on
  the first take; the GPU test adds "あ…". Speaker similarity and CER are not meaningful
  below ~1 s (one wrong kana = 50 % CER). Lines with nothing speakable ("……", "！？") return `ok=false`
  (`no_speakable_text`); Komi's silent bubbles must be handled as silence/SFX by the director.
* **Duration control is a decoder-side time warp**, not a re-planned delivery. The limits (speed 0.75-1.30, stretch
  0.80-1.25) are conservative defaults set without listening; tighten or widen them in `tts.fit` after an audition.
  IndexTTS-2's token-count duration mode is not enabled in the released code (repo README).
* **Emotion vectors versus timbre.** A strong vector moves the delivery towards the model's emotion prototypes and
  away from the reference; `strength` 0.65 is an untuned default (see the ablation). For recurring strong emotions
  (Tadano's tsukkomi shouts, Najimi's hyper delivery) an emotional real clip as `emo_ref`, or as the reference itself,
  is the option to try next; neither has been tested.
* **Readings.** Kanji readings are the model's own guess; names are forced through `tts.lexicon_ja`. The upstream
  translator can force any reading by writing `漢字(かんじ)`. Numbers become kanji numerals; counters with irregular
  readings rely on the model.
* **Emotion-label coverage** is a hand-written table; the judge has no direct check of emotion (only ASR, similarity,
  duration).
* **Speaker-similarity model domain.** ECAPA is trained on VoxCeleb (real speech); use the anchors in the report, not
  absolute thresholds. The audio workstream's WeSpeaker ResNet293 embedder (`src/anime/audio_embed.py`) could be used
  for a second opinion.
* **ffmpeg and terminals.** The `atempo` step calls ffmpeg with `-nostdin` and `stdin=DEVNULL`. Without that, an ffmpeg
  started from a background process group of a terminal (seen with `timeout <cmd>` inside tmux) is stopped by
  SIGTTIN/SIGTTOU and the batch hangs silently. `src/common/io.py::ffmpeg()` has neither flag.
* **Licence.** IndexTTS-2.5 weights are under the bilibili Model Use License (research use here; voice clones stay
  private per the brief).

## 5. Is per-character fine-tuning worth it? (brief: "TTS trained")

Zero-shot is the working path now. Recommendation, in order:

1. **First rebuild the voice bank on separated vocals and listen** (30 min once `lines_named.jsonl` and `vocals.wav`
   exist: `voice_bank.py` ~1 min per character on CPU + one `tts_test` run). Most of the achievable gain over the
   current test comes from clean, well-chosen, emotion-matched references, not from training.
2. **Decide with the anchors.** If `sim_ref` on clean refs stays more than ~0.15 below the real-vs-real anchor of the
   same character, or the voice is recognisably "someone else" by ear for the main cast (Tadano, Najimi, Yamai, Agari,
   narrator), fine-tune those voices.
3. **What to train:** IndexTTS-2.5 has no released training code, so "TTS trained" means a second engine behind the
   same JSONL interface: **GPT-SoVITS** per-character fine-tunes (the brief's first fallback; strong on Japanese anime
   voices, needs ~3-30 min of clean speech + transcripts per character, which the separated stems + the ASR pass
   provide). Estimate (from general experience with that tool, nothing measured here) on the local 4090: data export
   from the voice bank ~1 h of work, training ~15-25 min per character
   (SoVITS + GPT stages) => ~2-3 h GPU for 6 voices, engine adapter + duration fitting via speed/atempo ~2-3 h of work;
   about half a day in total. It would lose IndexTTS's emotion vectors (emotion comes from the choice of reference
   clip) and has only speed control, so it should be an option per character, not a replacement.
4. Not recommended for the POC: fine-tuning CosyVoice 3 or F5-TTS (hours per voice, more data needed) or training the
   IndexTTS GPT stage from third-party scripts (unverified, 24 GB is tight).

No long fine-tune has been started.

## 6. Verified vs not verified

Verified with the real model on CPU (2 lines, see 3.1): environment, offline model load, Japanese synthesis that
Whisper transcribes back correctly, an emotion vector going through the model (its audible effect is not verified),
`duration_factor` speed fitting to 1 ms of the target, speed + `atempo` stretch to 13 ms of a 0.66x target, a short
interjection passing the sanity check, run-to-run determinism within 1 LSB, cache rerun, clean error rows, the
cross-venv `synthesize()` entry point (from `venvs/main`, cached lines), voice bank on a hand-labelled JSONL, ECAPA
similarity, the CER pipeline.
Verified only with the fake engine of `src/generate/tts_selftest.py` (no model): the unreachable-target branches,
sanity re-rolls, `max_dur` handling, result merging.
Verified on GPU: only what block 3.2 shows once the queued test has run (speed, VRAM, bf16 precision, determinism
across processes, faster-whisper on CUDA; the script falls back to fp32 synthesis / CPU ASR if those fail).
Not verified: listening quality and emotion appropriateness, references from separated vocals, `lines_named.jsonl`
from the real pipeline, long batches (hundreds of lines), the GPT-SoVITS fallback.
