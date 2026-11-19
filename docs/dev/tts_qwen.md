# English voices: dub voice bank + Qwen3-TTS 1.7B (cloning, and what "emotion control" really gives)

Workstream files: `src/anime/audio_en.py`, `src/anime/speakers_en.py`, `src/train/voice_bank.py` (`--lang en`),
`src/generate/tts_qwen.py`, `src/train/qwen_sft.py`, `src/eval/emotion_probe.py`, `src/eval/voice_metrics.py` (WER),
`configs/tts_qwen.yaml` (`CFG.tts_qwen`), `configs/audio.yaml` (`dub_en:` block). Everything runs on this machine; no
audio left it. The only outside calls were model downloads and 67 text-only DeepSeek calls ($0.009) that were shown
dub transcript lines to check speaker names.

Measured 2026-11-12 on the shared RTX 4090. Nobody has listened to anything: every claim below is a number.

## 0. Short answers

| Question | Answer (measured) |
|---|---|
| Is a subtitle track the dub's closed captions? | **No.** Every text track is a translation of the Japanese (word match with the dub 0.22-0.44 in episodes 1, 4, 6, 9, 13, 17, 18). Dub text = faster-whisper large-v3. |
| Is the dub in sync with the picture? | Season 1 files carry the dub with a 1 s stream delay that ffmpeg drops on extraction (measured lag -0.94 s); corrected, so all English times are on the video timeline. Season 2: 0. |
| English voice bank | 17 voices in `data/processed/voices_en/`, **14 of the 17 Japanese-bank speakers** (missing: Otori, Komi's father, Shousuke) + Naruse, Shinobino, Tadano's sister. Komi has a bank (6 clips, 27.9 s, all from her phone call in episode 19). |
| Does the clone call accept an emotion instruction? | Not in the API. It can be forced in (`instruct_ids` reaches `model.generate`), and then **changes nothing consistent** when the reference audio is in context. |
| Inline tags `[excited]` on the open 1.7B models? | **Read aloud** by the cloning model (75-100 % of takes). Stripped by default. |
| Fine-tune + `instruct`? | The published fine-tuning script builds its loss wrongly (starts at chance level and damages the voice). With the loss fixed, a 3-epoch fine-tune of Tadano keeps the voice at x-vector level (0.64 vs 0.67 for the plain clone) and `instruct` moves pitch and pace: **sad works; excited raises the pitch by 7.6 semitones but the similarity falls by 0.17**; angry / whispering / nervous do not. |
| Emotion-matched reference clip? | **The only lever that works on the plain clone**, and only downwards so far: the low-arousal set lowers F0 by 3.9 st and darkens the voice at -0.04 similarity. The high-arousal set raises F0 by 2.6 st but not consistently (10 of 16 sentences). |
| Recommended way to pass emotion | Section 5. In one line: clone with `icl`, let `voice_refs_en(speaker, instruct)` pick the reference set, write the emotion into the words and punctuation, and use `"mode": "custom"` (fine-tuned voice + instruction) only for the few lines of a main character that must sound clearly sad or keyed up. |

## 1. What exists

| Thing | Path |
|---|---|
| Dub speech stems, train episodes 1-21 | `data/interim/anime/epNN/vocals_en.wav` (44.1 kHz **mono** PCM16, on the video timeline), `separation_en.json`; `data/interim/anime/sync_en.json` |
| Dub ASR | `data/processed/anime/asr_en/epNN.json`, `summary.json` |
| Dub lines (before naming) | `data/processed/anime/dub_lines/epNN.json` (`idx` = position) |
| Speaker embeddings of dub lines | `data/processed/anime/spk_en/epNN.npz`, `name_centroids.npy`, `names_en.json` (all statistics) |
| **Named dub lines** | `data/processed/anime/lines_en.jsonl` = `{"ep","idx","start","end","text","speaker","conf","source"}` |
| **English voice bank** | `data/processed/voices_en/voices.json`, `<Speaker>/ref_NN.wav`, `style_high_NN.wav`, `style_low_NN.wav`, `meta.json`, `centroid.npy` |
| TTS weights | `models/tts/qwen3/Qwen3-TTS-12Hz-1.7B-Base` (4.3 GB), `...-CustomVoice` (4.3 GB), `ft/Tadano_base_e3` (3.6 GB) |
| TTS venv | `venvs/qwentts`: Python 3.12.2, torch 2.8.0+cu128, transformers 4.57.3, accelerate 1.12.0, `qwen-tts` 0.1.1 (identical to `qwen_tts/` of QwenLM/Qwen3-TTS at commit 022e286, kept in `third_party/Qwen3-TTS`) |
| Probe data (all takes + metrics) | `runs/en_voice/probe/{main,tags,control,ft_*}/`, pooled tables `runs/en_voice/probe/pooled_{all,ft,tags}/probe_report.md` |
| Fine-tuning set | `runs/qwen_sft/Tadano/{clips/,train_raw.jsonl,train_codes.jsonl,data.json}` |

`source` in `lines_en.jsonl`: `transfer+spk` (name from the Japanese line at the same time, confirmed by the English
voice), `spk` (named by voice only), `transfer` (group names Classmates / Other, or no embedding), `spk+llm` (voice
found by the discovery step), `context` (fragment between two lines of a discovered voice), `none` / `song` (speaker
`Unknown`). `idx` indexes `dub_lines/epNN.json`. Held-out episodes 22-24 are never read (asserted in the code).

## 2. How to run

```bash
. /home/user/komi-poc/scripts/env.sh ; W=$KOMI_WORK ; A=$W/venvs/audio/bin/python
# --- dub audio -> named lines (order matters; every step is idempotent)
$A -m src.anime.audio_en extract  --episodes train
AUDIO_VRAM_GB=5 scripts/gpu_run.sh 5 $A -m src.anime.audio_en separate --episodes train   # same model/settings as Japanese
AUDIO_ASR_VRAM_GB=8 scripts/gpu_run.sh 8 $A -m src.anime.audio_en asr --episodes train --batch 8
$A -m src.anime.audio_en lines --episodes train
scripts/gpu_run.sh 2 $A -m src.anime.audio_en embed --episodes train
$W/venvs/main/bin/python -m src.anime.speakers_en --llm-check --discover     # CPU; -> lines_en.jsonl, spk_en/names_en.json
# --- voice bank (ECAPA + Whisper transcripts of the clips)
scripts/gpu_run.sh 6 $W/venvs/tts/bin/python -m src.train.voice_bank --lang en --device cuda --transcribe
#   one speaker later, keeping the rest:  ... --lang en --speakers Komi --merge --transcribe --force
# --- synthesis
scripts/gpu_run.sh 8 $W/venvs/qwentts/bin/python -m src.generate.tts_qwen --in lines.jsonl --out-dir runs/<run>/tts
# --- metrics (WER for "lang": "EN" rows)
scripts/gpu_run.sh 6 $W/venvs/tts/bin/python -m src.eval.voice_metrics --results runs/<run>/tts/tts_results.jsonl \
    --voices $W/data/processed/voices_en/voices.json --device cuda
```

Queue scripts used for this run: `runs/queue/{en_sep.sh,en_asr_chain.sh,en_names.sh,en_probe.sh,en_ft_variant.sh}`.

From pipeline code in any venv:

```python
from src.generate.tts_qwen import synthesize, voice_refs_en
refs, texts, who = voice_refs_en("Komi", instruct="sad and quiet")   # picks the low-arousal clips when the bank has them
rows = synthesize([{"id": "c140_s023_l0", "speaker": who, "text": "I... I want to make a hundred friends.", "lang": "EN",
                    "ref_wavs": refs[:3], "ref_texts": texts[:3], "instruct": "sad and quiet",
                    "target_dur": None, "max_dur": 7.0}], out_dir="runs/x/tts")
```

`synthesize` writes the batch to `<out_dir>/tts_in_<hash>.jsonl`, runs it in `venvs/qwentts` through
`scripts/gpu_run.sh <tts_qwen.vram_gb>` and returns the rows in input order. When every line is already cached (or
invalid) it answers in the calling process: no venv switch, no GPU queue (measured 0.08 s for 12 lines).

### Input line

```json
{"id": "...", "speaker": "Tadano", "text": "Good morning, Komi.", "lang": "EN",
 "ref_wavs": ["/abs/ref_00.wav", "..."], "ref_texts": ["transcript of ref_00", "..."],
 "instruct": "sad and quiet" , "target_dur": 2.8, "max_dur": 3.5, "seed": 7}
```

Optional per line: `"mode"` (`icl` | `xvec` | `cv_xvec` | `cv_icl` | `custom`), `"preset": "ryan"` (mode `custom` with a
preset voice of the CustomVoice model), `"instruct_on_base": true`, `"strip_tags": false`, `"real_wav"`, `"meta"`.

| mode | model | what conditions the voice | instruction used? |
|---|---|---|---|
| `icl` (default) | Base | first ref clip + its transcript in context (cut to 12 s) + mean speaker embedding of up to 3 refs | no (`instruct_on_base: false`); it would have no consistent effect |
| `xvec` | Base | speaker embedding only | only with `instruct_on_base` |
| `cv_xvec` | CustomVoice | Base speaker embedding injected where a preset speaker's embedding goes | yes |
| `cv_icl` | CustomVoice | ref clip in context | yes (no effect measured) |
| `custom` | `finetuned.<speaker>` checkpoint, or a preset | the checkpoint's own speaker row | yes |

The speaker embedding comes from the Base model's speaker encoder, loaded **alone** from the checkpoint (12 M
parameters, not the 1.9 B model) and cached per reference file in `cache/tts_qwen/xvec/`.

### Result row (`tts_results.jsonl`)

`id, path, dur, raw_dur, fit, ok, error` plus `mode, model, text_norm, seed, instruct_applied, n_words, sec_per_word,
raw_rms_dbfs, raw_peak_dbfs` (level before normalisation), `fit_ok, dur_err, over_max, atempo, min_feasible_dur,
max_feasible_dur, tokens, takes, synth_sec, rtf, warnings, cached, key` and the input fields.

* Output: mono 44.1 kHz 16-bit, silence trimmed (20 / 60 ms pads), 4 / 12 ms fades, peak -3 dBFS. Model rate 24 kHz.
* `ok=false`: bad input, nothing speakable (`"..."`), model error, or the best take still failed the sanity check
  (`sanity_failed: too_long / too_short / runaway / silent`; the take is still written).
* Sanity bounds per take: `0.11 s/word <= raw_dur <= 0.85 s/word + 1.2 s (+0.4 s per "..." or dash)`; the token budget
  is capped at the upper bound (12.5 codec frames per second), a take that hits the cap is a runaway. Failing takes
  are re-rolled with seed+1 (3 times). In the 696-line probe 3 takes needed a re-roll (all in `cv_icl`).
* Duration fitting: Qwen3-TTS has no speed control, so `fit` is `none` | `native` | `stretch` (ffmpeg `atempo`, 0.80-1.25
  of the duration). A line that would need more is left alone with `fit_ok=false` and `min/max_feasible_dur`.
* Cache: `cache/tts_qwen/<key>.wav`, keyed by text, mode, checkpoint file, reference contents + transcript,
  instruction, seed, target/max and every sampling / fit / trim / sanity setting.
* `batch_size` 1 (default) makes every line reproducible on its own; larger batches are faster but a take then
  depends on its batch mates.

### Speed and memory (measured)

| setting | seconds per line | compute s per audio s | peak VRAM of the process |
|---|---|---|---|
| batch 1 (16 lines, 2.3 s each, voice-bank job on the GPU at the same time) | 1.45 | 0.62 | 5.1 GB (Base), 4.8 GB (CustomVoice) |
| batch 8 (544 probe lines, 2.8 s each) | 0.30 | 0.106 | 11.4 GB (Base), 14.2 GB (CustomVoice) - reserved by the allocator; allocated 5.8 / 6.9 GB |
| fine-tuned checkpoint, batch 8 | 0.25 | 0.10 | 6.3 GB |

Model load 1-5 s with warm disk cache. `tts_qwen.vram_gb: 8` is right for batch 1; request 15 for batch 8.
Under heavy GPU contention (two separation jobs + ASR) a single line took 9-13 s.

## 3. English dub lines and voice bank

**Audio.** `eng` track -> 44.1 kHz stereo -> the same separation as the Japanese stems (`mel_band_roformer_kim_ft2_
bleedless_unwa.ckpt`, overlap 4, chunk 801, peak VRAM 3236 MiB, reconstruction residual -89.2 dB). 174 s per episode
with two workers sharing the card (97 s alone). The vocal stem is stored mono (its side channel is 22.9 dB below the
mid) and the instrumental is not kept (same music as the Japanese stems); `audio_eng.wav` and the 16 kHz copies are
caches that were deleted (re-extract in 10 s per episode).

**Sync.** ffmpeg drops an audio stream's container start offset. Cross-correlating the dub mix with the Japanese mix
(shared music and effects, four 180 s windows per episode, all windows agree in all 21 episodes) gives a lag of -0.94 s
in episodes 1-12 (ep10: -0.91) and 0.00 in 13-21 (ep15: -0.02); `vocals_en.wav` is shifted accordingly. Check: the
dub's "I'm Hitohito Tadano" starts at 300.39 s, the Japanese subtitle at 300.38 s.

**Text.** faster-whisper large-v3, `language="en"`, float16, batched (8), VAD and decoding settings of the Japanese
pass, character names as `initial_prompt`: 41 983 words, mean word probability 0.953, 25 s per episode, peak 6.7 GB.
Lines are cut from the word timings at pauses >= 0.45 s and at sentence ends: **8 014 lines, 13 257 s**, 216 of them
inside OP / ED spans (`song`).

**Names.** (1) Transfer by time overlap from `lines_named.jsonl` (episodes 1-14): 4 193 lines get a name. (2) Per name,
a robust centroid of the WeSpeaker embeddings of its transferred lines (only those: a centroid cannot drift). (3) A
transferred name is kept while its own centroid is (nearly) the nearest; other lines, and all of episodes 15-21, take
the nearest centroid if it is clear (cosine >= 0.45, margin >= 0.06), else `Unknown`. (4) Names whose seeds sit on
someone else's voice are dropped. (5) Discovery: the remaining unnamed core lines are clustered by voice and each
cluster is identified from its text with dialogue context by a text-only LLM and by the Japanese voice cluster of
that name overlapping it.

| speaker | lines | seconds | episodes | ep 1-14 | ep 15-21 | transfer + voice | voice only | other |
|---|---|---|---|---|---|---|---|---|
| Unknown | 3353 | 4061 | 21 | 1913 | 1440 | 0 | 0 | none 3137, song 216 |
| Tadano | 1392 | 2579 | 21 | 1012 | 380 | 859 | 530 | transfer 3 |
| Najimi | 833 | 1585 | 20 | 705 | 128 | 423 | 409 | transfer 1 |
| Narrator | 790 | 1645 | 21 | 507 | 283 | 305 | 485 | |
| Yamai | 260 | 536 | 19 | 213 | 47 | 100 | 160 | |
| Other | 164 | 290 | 10 | 164 | 0 | 0 | 0 | transfer 164 |
| Komi's mother | 145 | 327 | 13 | 63 | 82 | 32 | 113 | |
| Nakanaka | 145 | 352 | 14 | 107 | 38 | 56 | 89 | |
| Onemine | 138 | 265 | 11 | 111 | 27 | 45 | 93 | |
| Agari | 119 | 231 | 12 | 79 | 40 | 41 | 78 | |
| Katai | 113 | 267 | 6 | 57 | 56 | 27 | 86 | |
| Yadano | 104 | 212 | 16 | 84 | 20 | 53 | 51 | |
| Inaka | 101 | 219 | 11 | 70 | 31 | 22 | 78 | transfer 1 |
| Teacher | 67 | 124 | 12 | 50 | 17 | 30 | 37 | |
| Classmates | 66 | 76 | 10 | 66 | 0 | 0 | 0 | transfer 66 |
| Shinobino | 62 | 190 | 15 | 36 | 26 | 8 | 54 | |
| Tadano's sister | 59 | 93 | 11 | 27 | 32 | 12 | 47 | |
| Naruse | 57 | 120 | 3 | 0 | 57 | 0 | 0 | discovered 50, context 7 |
| Komi | 46 | 85 | 7 | 6 | 40 | 0 | 0 | discovered 40, context 6 |

4 661 of 8 014 lines carry a name (58 %); of the 4 302 non-song lines of at least 1.2 s, 3 221 carry a character name.

**How pure are the labels? (no ground truth exists; these are the checks)**

| Check | Result |
|---|---|
| Raw time-overlap transfer, cross-episode consistency (centroids from odd episodes classify even ones and back) | **0.57** over 2 232 lines: the transfer alone is not trustworthy (the Japanese names are mostly text-only LLM guesses) |
| What the voice step did with the 4 193 transferred names | kept 54 %, renamed by voice 18 %, dropped to Unknown 29 % |
| Label-free voice clusters (average linkage), share of the majority FINAL name | 0.990 at cut 0.6 (42 clusters, 3 396 lines), 0.914 at cut 0.76 (29 clusters, 4 097 lines); same for the raw transfer: 0.70 / 0.63. Partly circular: final names come from the same embeddings |
| Margin of named core lines | 94.6 % are at least 0.10 cosine closer to their own centroid than to any other |
| Narrator's recurring introduction (text repeated in >= 5 episodes) | 43 of 43 lines in 18 episodes are `Narrator` |
| Blind text check (LLM sees the 28 most central lines of a voice without the name) | 14 of 14 seeded names confirmed (weakest: Shinobino 0.55, Agari 0.72; the rest 0.90-0.97) |
| Independent embedding model (ECAPA, voice-bank candidates of 3-12 s): share consistent with the speaker's own centroid | 685 of 754 = 0.91 over the named voices (lowest: Shinobino 29/42, Katai 45/60, Teacher 27/33, Inaka 49/60) |

Honest summary: the big voices (Tadano, Najimi, Narrator, Yamai) are solid; for lines of 1.2 s or more with source
`transfer+spk` or `spk` I estimate 0.9 or better voice consistency; short lines and the catch-all-prone adult voices
(Komi's mother, Teacher, Shinobino) are weaker, and 42 % of all lines stay `Unknown` on purpose.

**Names that were refused, and why**

* `Komi` by transfer: the 320 dub lines under Japanese "Komi" lines are Najimi's voice (centroid cosine 0.93 with
  Najimi; others speak while Komi writes). Her real voice was found by the discovery step: one voice cluster of 26
  core lines, 22 of them in episode 19 at 1058-1295 s (she phones Tadano about her junior-high field trip), 85 % of
  them overlapping the Japanese voice cluster that is named Komi. The text LLM alone gave "Komi" only 0.55.
* `Otori`: 11 seed lines; the voice that grew from them was a catch-all (5 % seeds: salon trainee, Komi's
  grandmother, tour guide). One candidate voice (22 lines, 57 s, episodes 11-12, mostly transferred as "Onemine")
  could be hers, but the text check said Kishi (0.55) and, with context, Yamai (0.95, although the voice is not
  Yamai's), and its speaking rate is ordinary (3.0 words/s). Not named.
* `Katou`, `Sasaki`: no Japanese seeds at all and no cluster the text check could pin down (a shogi line in episode
  21 sits in a mixed 11-line cluster). Not named.
* `Kometani`: three different voices fit "dry commentary" by text, so none was accepted.
* `Komi's father`, `Shousuke`: 4 and 3 transferred lines. Nothing to build on.

**Voice bank** (`voice_bank.py --lang en`; the Japanese rules: one speaker, 3-12 s, 0.15 s away from every other dub
line, SNR proxy >= 12 dB, ECAPA consistency; plus a Whisper transcript of every written clip = `ref_texts`, a
preference for clips that end on a finished sentence, and per speaker the 3 most and 3 least aroused clean clips =
`style_refs.high / low`, ranked by z(log F0) + 0.6 z(level) + 0.4 z(F0 range) within the speaker).

| speaker | refs | ref s | clean candidate s | lines | line s | ECAPA median | ECAPA of refs | consistent / clean | median F0 Hz | style sets |
|---|---|---|---|---|---|---|---|---|---|---|
| Tadano | 8 | 57.9 | 370.6 | 1392 | 2579 | 0.786 | 0.826 | 57/59 | 189 | high + low |
| Narrator | 8 | 52.2 | 391.2 | 790 | 1645 | 0.853 | 0.894 | 58/60 | 170 | high + low |
| Najimi | 8 | 53.3 | 404.2 | 833 | 1585 | 0.853 | 0.854 | 59/60 | 280 | high + low |
| Yamai | 8 | 56.5 | 410.3 | 260 | 536 | 0.849 | 0.831 | 59/60 | 387 | high + low |
| Nakanaka | 8 | 51.3 | 352.7 | 145 | 352 | 0.690 | 0.759 | 58/60 | 342 | high + low |
| Komi's mother | 8 | 50.1 | 335.4 | 145 | 327 | 0.780 | 0.803 | 55/60 | 384 | high + low |
| Katai | 8 | 56.5 | 314.1 | 113 | 267 | 0.824 | 0.823 | 45/60 | 137 | high + low |
| Onemine | 8 | 50.2 | 265.9 | 138 | 265 | 0.790 | 0.858 | 56/60 | 245 | high + low |
| Agari | 8 | 48.3 | 292.9 | 119 | 231 | 0.740 | 0.792 | 55/59 | 403 | high + low |
| Inaka | 8 | 46.2 | 302.5 | 101 | 219 | 0.799 | 0.792 | 49/60 | 394 | high + low |
| Yadano | 8 | 54.7 | 347.9 | 104 | 212 | 0.804 | 0.834 | 57/60 | 331 | high + low |
| Shinobino | 8 | 46.4 | 178.7 | 62 | 190 | 0.784 | 0.780 | 29/42 | 316 | high + low |
| Teacher | 8 | 40.1 | 115.8 | 67 | 124 | 0.799 | 0.822 | 27/33 | 272 | high + low |
| Naruse | 7 | 45.6 | 233.6 | 57 | 120 | 0.750 | 0.765 | 41/50 | 147 | high + low |
| Tadano's sister | 7 | 37.5 | 100.9 | 59 | 93 | 0.764 | 0.779 | 21/21 | 376 | high + low |
| Komi | 6 | 27.9 | 249.0 | 46 | 85 | 0.835 | 0.820 | 47/54 | 331 | high + low |
| Classmates | 4 | 20.6 | 27.8 | 67 | 78 | 0.709 | 0.667 | 6/13 | 379 | - |

"clean candidate s" counts overlapping windows more than once. Komi's clips are built with her own settings
(`voice_bank.speaker_overrides.Komi`): fragments joined across pauses up to 2.5 s, pauses then shortened to 0.35 s.
"Shinobino" is a high voice (316 Hz) with the weakest support of all names: treat it as "a high minor voice".
`Classmates` is several voices under one name, as in the Japanese bank.

**Speakers without a bank** (`voice_refs_en` -> `fallback_voice`): same gender as `data/processed/cards/cards.json`
says, a minor character's voice (Komi / Tadano / Najimi only as a last resort), never the Narrator or a group entry;
how a bank voice sounds is decided by its median F0 (< 150 Hz male, > 190 Hz female, the card in between); children of
10 or younger get one of the three highest minor voices (the bank's male voices are adult-sounding, so a little boy
gets a high voice of either gender). Deterministic. Today: Otori -> Shinobino's voice, Katou -> Tadano's sister,
Sasaki -> Onemine, Chiarai -> Naruse, Sonoda -> Katai, Tadano Jeanne -> Onemine, Onemine Kazuya -> Yamai,
Onemine Nono -> Yamai, Onemine Nana -> Inaka, Komi's father -> Katai, Shousuke -> Naruse.

## 4. The emotion question, measured

**Probe** (`src/eval/emotion_probe.py`). 8 neutral-content sentences x {neutral, excited, sad, angry, whispering,
nervous} for Tadano and Najimi, per method; each take is compared with the same method's neutral take of the same
sentence (16 pairs per cell for the pooled table, 8 for single-speaker tables); a second neutral take with another
seed gives the noise floor. Per take: ECAPA similarity to the speaker's bank centroid, WER (faster-whisper `en`),
mean F0 and its 5-95 % range (pYIN), level before normalisation, words per second, voiced ratio, spectral tilt
(level above minus below 2 kHz). 696 takes in the main and tag probes, 56 for the second control voice, 280 for the
fine-tunes; none failed the sanity check after re-rolls.

Instructions: "Very excited and energetic, speaking fast with a bright, high-pitched voice." / "Sad and dejected,
speaking slowly in a low, quiet voice." / "Angry and shouting, loud and forceful." / "Whispering very quietly, almost
inaudible." / "Nervous and hesitant, stammering slightly with a shaky voice."

**Verdict rule.** A feature "moves" when the mean paired difference exceeds a threshold (F0 2 st, F0 range 3 st,
0.3 words/s, voiced ratio 0.06, tilt 1.5 dB) and at least 75 % of the sentences agree. Expected: excited = F0 up,
faster; sad = F0 down, flatter, darker, slower; angry = F0 up, brighter, faster; whispering = less voicing, breathier,
flatter; nervous = F0 up, slower. `works` = two expected features move, none goes the wrong way, similarity within
0.05 of the method's neutral take, WER <= 0.20. `partly` = one feature.
*This rule was changed once, after the control was measured:* the first version (written before any audio existed)
also expected the level to rise or fall and used lower thresholds. The control showed that the model does not control
loudness (for "sad" one preset voice got 2.2 dB louder, the other 2.4 dB quieter) and that a mere change of seed moves
the mean F0 by 1.1 st with 12 of 16 sentences agreeing. The level is still reported in the full tables.

**Anchors for the similarity column.** Real clips against the speaker's own centroid: default refs 0.82-0.90; the
real high-arousal clips 0.73-0.86; the real low-arousal clips 0.65-0.85. A different speaker (preset voice against
Tadano's / Najimi's centroid): 0.12. So real emotional speech of the same actor already sits 0.05-0.17 below the
neutral clips: the 0.05 tolerance is strict.

### A. Instruction text (Tadano + Najimi pooled, 16 pairs per cell)

Cells: verdict; ΔF0 in semitones (sentences up / down), ΔF0 range, Δwords/s, Δtilt, Δvoiced ratio; Δsimilarity.

| method | neutral: sim / WER | excited | sad | angry | whispering | nervous |
|---|---|---|---|---|---|---|
| `preset` (control: CustomVoice preset voices Ryan + Aiden, the path the model was trained for) | 0.121 / 0.009 | **partly**: F0 +2.5 (10↑6↓), +0.37 w/s | **clear**: F0 -3.7 (4↑12↓), range -5.6, tilt -1.6 | **partly**: F0 +3.9 (14↑2↓), tilt +2.5 | **clear**: range -5.3, tilt +3.1, voiced -0.08 | none: F0 +1.3 (9↑7↓) |
| `icl_inst` (a: clone call + instruction) | 0.698 / 0.016 | **none**: F0 +1.9 (11↑5↓), +0.16 w/s; sim -0.08 | **wrong direction**: F0 +1.3 (12↑4↓), range +3.5, tilt +1.5, -0.35 w/s | **none**: F0 +0.9 | **none**: F0 +0.2, voiced 0.00 | **none** |
| `xvec_inst` (a without the reference audio in context) | 0.634 / 0.001 | none: F0 +3.8 (11↑5↓); sim -0.12 | **moves, voice drifts**: F0 -2.0 (4↑12↓), range -3.9, tilt -2.5, -0.32 w/s; sim -0.07 | none; sim -0.08 | wrong direction (tilt -2.5) | none; sim -0.10 |
| `cv_xvec` (CustomVoice weights + clone embedding) | 0.579 / 0.002 | **partly**: F0 +2.8 (13↑3↓), +0.27 w/s; sim 0.00 | moves, voice drifts: tilt -2.4; sim -0.10 | none | moves, voice drifts: range -3.6, voiced -0.08; sim -0.07 | none |
| `cv_icl` (CustomVoice weights + reference in context) | 0.697 / 0.025 | none | none | none | none | none |

Noise floor (second neutral seed): F0 -1.8 to +0.1 st, range -3.5 to +2.4 st, level -0.3 to +0.6 dB, -0.15 to +0.27
words/s, similarity -0.04 to +0.03.

Reading:

* **(a) The installed `generate_voice_clone` has no `instruct` argument.** Its `**kwargs` reach
  `Qwen3TTSForConditionalGeneration.generate`, which does accept `instruct_ids` for every model type, so the engine can
  pass one (`"instruct_on_base": true`). With the reference audio in context it does nothing reliable: no cell moves,
  "sad" even comes out higher and livelier. **Does not work.**
* Without the reference audio (`xvec_inst`) the Base model reacts to "sad" in the right direction, at the price of
  the voice (0.63 neutral, -0.07 with the instruction). The reference in context is what overrides the instruction.
* Even the control is modest: instructions reliably change pitch height and liveliness (sad, angry), make the voice
  breathier for "whispering" (the voiced ratio only falls by 0.08: it is not a whisper), speed up "excited", and do
  nothing measurable for "nervous". Do not expect acted emotions from instructions on this model.
* `cv_xvec` (no training) keeps the voice only at 0.58 (the plain clone: 0.70) and follows instructions weakly. For
  Tadano alone "excited" works (F0 +6.2 st in 8 of 8 sentences, +0.48 words/s, similarity -0.03), for Najimi nothing
  moves (her neutral take is already at 351 Hz).

### B. Inline tags (Tadano, 4 sentences per tag; tag at the start of the text, `[laughing]` at the end)

| method | [excited] | [sad] | [angry] | [whispers] | [nervous] | [sighing] | [laughing] |
|---|---|---|---|---|---|---|---|
| `icl` clone: tag spoken as a word (WER) | 100 % (0.13) | 100 % (0.16) | 100 % (0.16) | 75 % (0.09) | 100 % (0.13) | 100 % (0.13) | 100 % (0.16) |
| `cv_xvec`: tag spoken (WER) | 0 % (0.00) | 75 % (0.13) | 25 % (0.16) | 100 % (0.13) | 25 % (0.07) | 25 % (0.06) | 0 % (0.00) |
| CustomVoice preset Ryan: tag spoken (WER) | 25 % (0.10) | 50 % (0.16) | 50 % (0.07) | 75 % (0.09) | 0 % (0.04) | 0 % (0.00) | 0 % (0.06) |

**(b) Tags are read aloud by the cloning model** and often by the CustomVoice model too; no delivery change beyond
noise where they are not spoken. With the preset voice `[sighing]` and `[laughing]` are not transcribed as words and
the clip gets longer (-1.3 / -1.8 words/s), which fits a non-verbal sound or a pause; not verified by ear. The engine
strips `[...]` and `(...)` tags by default (`strip_tags: true`).

### C. Fine-tune + instruction (Tadano, 8 pairs per cell)

Data: 102 clean clips, 12.05 min, episodes 1-20, mean ECAPA consistency 0.72 (voice-bank rules; the bank's reference
clips are excluded). `src/train/qwen_sft.py`, adapted from the repository's `finetuning/` scripts.

**The published script does not continue the model's training objective.** Measured on the released Base weights with
these clips (qwen-tts 0.1.1, transformers 4.57.3, repository commit 022e286):

| Loss construction | talker loss at step 0 | sub-talker loss at step 0 |
|---|---|---|
| as published (`sft_12hz.py`) | 10.59 (chance level is 8.0) | 9.22 |
| text embeddings passed through `talker.text_projection`, as at inference | 8.54 | 10.17 |
| + plain next-token cross-entropy (the script shifts the labels by hand and the library loss shifts them again: its value equals the cross-entropy against the token two steps ahead to three decimals) | **1.06** | 7.18 |
| + sub-talker fed the hidden state that predicted the frame, not the frame's own position | 1.06 | **5.53** (per codebook 3.8 ... 6.3) |

The CustomVoice weights behave the same (1.11 with the corrected construction). `qwen_sft.py` uses the corrected
construction; `--upstream-objective` reproduces the published one.

Memory: a full fine-tune holds 1.93 B parameters x (2 weight + 2 gradient + 4 AdamW) bytes = 15.4 GB before
activations, and only 14 GB were free next to the pipeline's ComfyUI server. The runs therefore froze the text
embedding and the lowest 16 of 28 talker layers (800 M trainable parameters), cut the autograd graph below them, used
a per-tensor AdamW step, batch 2 x accumulation 4, lr 2e-5: peak 10.7 GB allocated / 12.1 GB reserved, **3 s per
epoch** (51 steps), 0.39 min wall for loading plus 3 epochs. Three earlier attempts ran out of memory inside the
13 GB cap (all 28 layers in the autograd graph, the default multi-tensor AdamW step).

| checkpoint | training loss (talker CE, first -> last epoch) | neutral: sim / WER | excited | sad | angry | whispering | nervous |
|---|---|---|---|---|---|---|---|
| zero-shot `icl` (no instruction) | - | **0.672** / 0.000 | - | - | - | - | - |
| zero-shot `xvec_inst` | - | 0.612 / 0.000 | none: F0 +6.0 (5↑3↓); sim -0.23 | moves, drifts: range -4.4, -0.43 w/s; sim -0.08 | none | moves, drifts | none |
| **corrected, 3 epochs (kept: `ft/Tadano_base_e3`)** | 1.07 -> 0.51 | 0.638 / 0.005 | **moves, voice drifts**: F0 +7.6 (8↑0↓), tilt +1.9; sim -0.17 | **works**: F0 -3.9 (0↑8↓), -0.32 w/s; sim -0.03 | none | none (tilt wrong way) | wrong: F0 -3.5, -0.46 w/s |
| corrected, 10 epochs | 1.07 -> 0.03 (memorised) | 0.587 / 0.012 | moves, drifts: F0 +7.1 (8↑0↓); sim -0.08 | **works**: F0 -3.9 (1↑7↓), tilt -2.7, -0.38 w/s; sim 0.00 | none | partly: voiced -0.06 | partly: -1.06 w/s |
| corrected, CustomVoice weights as start, 3 epochs | 1.09 -> 0.50 | 0.586 / 0.042 | moves, drifts: F0 +10.0 (7↑1↓); sim -0.17 | moves, drifts: -0.58 w/s; sim -0.14 | partly: F0 +3.9 (7↑1↓) | none; WER 0.24 | moves, drifts |
| **published objective, 3 epochs** | 6.73 -> 5.13 | **0.421** / 0.008 | none | moves, drifts | wrong | none | moves, drifts |

**(c)** The voice of the fine-tuned checkpoint matches at the level of the speaker-embedding clone and slightly below
the in-context clone (0.64 vs 0.67; two neutral seeds: 0.62 / 0.66 against 0.65 / 0.69): with 12 minutes of data the
fine-tune does **not** beat zero-shot cloning on similarity. `instruct` does work on it in the sense that pitch and
pace follow: "sad" passes the full test, "excited" raises the pitch in every sentence but the similarity falls by
0.08-0.17 (real excited clips of the actor sit 0.00-0.10 below his neutral ones), angry / whispering / nervous do
not pass. More epochs memorise the 102 clips and lose similarity. The published objective loses the voice in
3 epochs (0.42).

### D. Emotion-matched reference clip (no instruction; Tadano + Najimi, 16 pairs)

| reference set | sim / WER | ΔF0 st | ΔF0 range | Δw/s | Δtilt dB | Δsim | verdict |
|---|---|---|---|---|---|---|---|
| default refs | 0.697 / 0.000 | - | - | - | - | - | baseline |
| low-arousal set (for sad / quiet / nervous / whisper) | 0.654 / 0.009 | **-3.9 (2↑14↓)** | -1.6 | -0.10 | **-2.5 (3↑13↓)** | -0.04 | **works** |
| high-arousal set (for excited / angry / panicked) | 0.624 / 0.009 | +2.6 (10↑6↓) | -1.3 | +0.03 | +1.8 (12↑4↓) | -0.07 | no clear change (right direction, not consistent) |

**(d)** The reference clip is what steers the in-context clone. Lower / darker deliveries follow the low set reliably
at almost no cost. The high set pushes pitch and brightness up, but only in 10 of 16 sentences; per speaker: Tadano
+2.6 st with similarity -0.01, Najimi +2.5 st with similarity -0.13 (her high clips are shouted at 386-471 Hz and
their clones sit far from her centroid, as the real clips do). Speaking rate does not follow the reference.

## 5. Recommendation: how the pipeline should pass emotion

1. **Clone with `icl` (the default), 3 default refs + transcripts from `voices_en/voices.json`.** Best voice match
   (0.70 against 0.82-0.90 for real clips and 0.12 for another speaker), WER 0.01.
2. **Let the producer's emotion pick the reference set, not an instruction.** Call
   `voice_refs_en(speaker, instruct)`: an instruction containing a low-arousal word (sad, quiet, whisper, nervous,
   shy, tired, calm, ...) returns the speaker's low set, a high-arousal word (excited, angry, shouting, panicked,
   surprised, ...) the high set (`tts_qwen.ref_style` holds the word lists). Still put `instruct` into the line: it is
   recorded, part of the cache key, and usable by the fine-tuned mode. Expect a real effect for the low set and a
   modest, unreliable one for the high set.
3. **Write the emotion into the line itself.** The model reads punctuation and wording ("Wait, what?! No no no!",
   "I... I'm sorry."); that is free and keeps the voice. Never put `[tags]` in the text (they are stripped).
4. **For a main character's lines that must be clearly sad or keyed up, use the fine-tuned voice:**
   `"mode": "custom"` with a full-sentence instruction. Only Tadano has one (`tts_qwen.finetuned`). Najimi and
   the Narrator have the most labelled speech after him (1585 s and 1645 s of lines; how many clean minutes that
   yields was not measured) and a fine-tune costs under a minute of GPU
   (`qwen_sft data` -> `codes` -> `train --epochs 3 --freeze-text-embedding --freeze-layers 16`). Accept a voice match
   about 0.03 below the plain clone, and more drift for "excited". Keep one checkpoint per voice (3.6 GB each).
5. **Do not rely on**: instructions on the clone call, inline tags, `cv_xvec` (voice match 0.58), whispering by any
   route (even the preset voices only get breathier), "nervous" by any route, loudness (not controlled by this model;
   set levels in the mix).

## 6. English voice metrics

`src/eval/voice_metrics.py` evaluates `"lang": "EN"` rows with `language=en` ASR and **WER** (`wer`, `wer_edits`,
`wer_ref_len`, flag `high_wer` above `tts.voice_metrics.flag_wer_above` = 0.25 with at least 2 word errors; aggregate
`wer`, `wer_micro`, `wer_reliable`). Word normalisation: case, punctuation, bracketed tags, hesitation fillers and a few
spellings (okay / ok, gonna / going to, ...). The reference is the row's `text_norm` (after tag stripping). Japanese
rows are untouched (checked: `tts_selftest` passes all checks, the Japanese bank reports "up to date", the Japanese
CER example still gives 0.00, the aggregate of Japanese rows has no `wer` key). `prosody_features()` in the same module
is the delivery measurement of the probe and of the bank's style sets. Acceptance batch (12 lines through
`synthesize()` from `venvs/main`: clones of Tadano, Komi, Najimi, two fallback speakers, one fine-tuned line, one
duration fit, one tagged line, one bad line): 11 ok, WER micro 0.013, Komi's clone 0.75 / 0.80 against her own refs.

## 7. Limits and open points

* **No listening.** Similarity, WER and delivery numbers only. Whether the clones sound like the dub actors, whether
  "sad" sounds sad, and whether Komi's six clips give a pleasant voice need ears: `runs/en_voice/acceptance/*.wav`,
  `runs/en_voice/probe/main/*.wav`, refs in `data/processed/voices_en/`.
* **Probe size.** 2 speakers x 8 sentences per cell (the fine-tune table: 1 speaker, 8 sentences). The seed noise is
  large (F0 up to 1.8 st between two neutral batches). Verdicts near the thresholds can flip; the big effects
  (fine-tune "excited" +7.6 st in 8 of 8, low reference set -3.9 st in 14 of 16) will not.
* **The fine-tune is a partial one** (lowest 16 talker layers and the text embedding frozen) because of the shared
  card; a full fine-tune was not possible and is not measured. Its peak would be about 17 GB.
* **The finding about the published fine-tuning script** is a measurement on the pinned package versions listed
  above, not a reading of upstream intent; a newer repository version may differ.
* **In-context cloning copies the reference's delivery.** A reference with background residue or an odd delivery is
  reproduced. Komi's clone will always sound like her halting phone call.
* **Speaker names** are only as good as section 3 says; the bank can contain a wrong voice under a minor name
  (Shinobino is the likeliest). `Unknown` lines (42 %) hold more voices: one-off adults, an announcer, a tour guide,
  the cat, and possibly Otori.
* **Transcripts of reference clips** come from Whisper on the clip; a clip whose last word is cut is ranked last
  (`truncated`), not removed.
* `sox` is not installed: `qwen_tts` prints a warning at import (only its 25 Hz tokenizer needs it; unused).
* `flash-attn` is not installed; the library default attention is used.
* License: Qwen3-TTS weights and code are Apache-2.0; the voice clones stay private per the brief.
