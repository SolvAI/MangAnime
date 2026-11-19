# Anime audio stack

Source separation, Japanese ASR, speaker embeddings / voice clusters and the BGM cue library for the 24 episodes
(brief §3b "Anime parsing" / "Generation", §5.2, §6). Config: `configs/audio.yaml` (`CFG.audio`). Venv:
`/home/user/komi-poc-work/venvs/audio` (Python 3.11, torch 2.8.0+cu128, audio-separator 0.48.0, faster-whisper 1.2.1 /
ctranslate2 4.8.5, onnxruntime-gpu 1.31.1, transformers 4.57.6).

Everything runs locally. Weights were downloaded once (list at the end); the batch runs with `HF_HUB_OFFLINE=1`.
No audio leaves the machine.

**Measured numbers are in the last section, which is written by `src/anime/audio_report.py` from the files the stages
produce.** Anything not listed there has not been measured yet.

## Modules

| Stage | Module | Output |
|---|---|---|
| Separation | `src/anime/audio_separate.py` | `data/interim/anime/epNN/{vocals,instrumental}.wav` (44.1 kHz stereo PCM16), `separation.json`, `data/interim/anime/separation_settings.json` |
| Model comparison | `src/anime/audio_sepcompare.py` | `runs/audio_sep_compare/<name>/{results.json,results.md,transcripts.json,<model>/*.wav}`, `data/interim/anime/separation_choice.json` |
| ASR | `src/anime/audio_asr.py` | `data/processed/anime/asr/epNN.json`, `asr/summary.json` |
| Speaker embeddings | `src/anime/audio_embed.py` | `data/processed/anime/spk/epNN.npz` |
| Voice clusters | `src/anime/speakers.py` | `data/processed/anime/spk/{clusters.json,centroids.npy,centroids/cNNN.npy}` |
| BGM library | `src/anime/audio_bgm.py` | `data/processed/bgm/{cues/*.wav,cues.jsonl,clap.npy,rejected.jsonl,stats.json,episodes/epNN.json}` |
| BGM retrieval | `src/generate/bgm.py` | `retrieve(text_query, min_dur, k=5)` |
| GPU probe | `src/anime/audio_gpuprobe.py` | `runs/audio_sep_compare/gpu_probe.json` (peak VRAM + throughput of spk / ASR / CLAP) |
| Report | `src/anime/audio_report.py` | AUTO section of this file |
| Shared helpers | `src/anime/audio_common.py` | — |

**Batch entry point (launched by the lead, never auto-started):**
`/home/user/komi-poc-work/runs/queue/audio_batch.sh "<episodes>" [stages]` — stages `sep,spk,asr,bgm` (default, in that
order) plus `cmp` (model comparison), `probe` (1-minute VRAM/throughput probe), `report`. Sequential, one GPU process at
a time, every GPU step through `GPU_WAIT_MIN=900 scripts/gpu_run.sh <need_gb>`, idempotent/resumable, a second call
waits for the first (flock). Log `logs/audio_batch.log`, state `logs/audio_batch.state`.

```bash
Q=/home/user/komi-poc-work/runs/queue/audio_batch.sh
$Q "1,2,22,23,24"                    # everything for the early episodes (runs the comparison first if no model is chosen)
$Q "3-21"                            # the rest; `spk` re-clusters over all train episodes embedded so far
AUDIO_VRAM_GB=4 $Q "3-21" sep        # only separation with a 4 GB budget
$Q "5,16" cmp,probe                  # comparison + probe only (this is what is queued in tmux `audio_cmp`)
tail -f /home/user/komi-poc-work/logs/audio_batch.log ; cat /home/user/komi-poc-work/logs/audio_batch.state
```

Single stages by hand:

```bash
. /home/user/komi-poc/scripts/env.sh && cd /home/user/komi-poc
PY=$KOMI_WORK/venvs/audio/bin/python
scripts/gpu_run.sh 5 $PY -m src.anime.audio_sepcompare --episodes 5,16 --dur 180 --synthetic-episodes 2-11 --synthetic-dur 90 --name full --write-choice
AUDIO_VRAM_GB=5 scripts/gpu_run.sh 5 $PY -m src.anime.audio_separate --episodes 1,2,22-24     # --model <file> / --force
scripts/gpu_run.sh 2 $PY -m src.anime.audio_embed --episodes 1,2,22-24
$PY -m src.anime.speakers cluster                                    # CPU; --threshold 0.62 to override, --fresh for new ids
scripts/gpu_run.sh 4 $PY -m src.anime.audio_asr --episodes 1,2,22-24
scripts/gpu_run.sh 3 $PY -m src.anime.audio_bgm --episodes train     # AUDIO_BGM_DEVICE=cpu -> no GPU
scripts/gpu_run.sh 4 $PY -m src.anime.audio_gpuprobe
$PY -m src.anime.audio_report
python -m src.generate.bgm "tense, awkward silence" --min-dur 10 -k 5
```

### VRAM per stage

| Stage | `gpu_run.sh` request | How the budget is enforced | Expected peak (NOT measured yet) |
|---|---|---|---|
| `sep` / `cmp` | `AUDIO_VRAM_GB` (default 5) | torch allocator hard-capped at budget − 0.7 GB; the model's own chunk (dim_t 801 = 8 s) first, then segments 512 → 384 → 256 after a CUDA OOM (remembered in `data/interim/anime/separation_settings.json`); overlap-add buffers are on the CPU, so VRAM does not depend on episode length | 3–4.5 GB at dim_t 801 with autocast (estimate from the model size) |
| `spk` embeddings | `AUDIO_SPK_VRAM_GB` (2) | torch cap at 1.5 GB | ≈ 1 GB (ResNet293, fp32, one line at a time) |
| `asr` | `AUDIO_ASR_VRAM_GB` (4) | compute type: `int8_float16` below 6 GB, `float16` from 6 GB | ≈ 3.1 GB (`int8_float16`) / ≈ 4.8 GB (`float16`) — faster-whisper's published large-v2 numbers |
| `bgm` (CLAP) | `AUDIO_BGM_VRAM_GB` (3) | torch cap at 2.5 GB; `AUDIO_BGM_DEVICE=cpu` runs it without the GPU | ≈ 1.5 GB |
| clustering, report | none (CPU) | — | — |

Every stage records the real peak of its process as nvidia-smi sees it (`peak_vram_mb`): separation in
`epNN/separation.json` and in the comparison results (per model and per segment size), ASR in `asr/summary.json`,
embeddings in `spk/embed_stats.json`, CLAP in `bgm/stats.json`, the probe in `runs/audio_sep_compare/gpu_probe.json`.
They are collected in the last section of this file.

## 1. Separation

**Library.** The UVR5_UI Space (`TheStinger/UVR5_UI`; code and `assets/models.json` inspected in
`komi-poc-work/third_party/UVR5_UI_space/`, never executed, never called) is a Gradio front end for
`audio_separator.separator.Separator` (`audio-separator[gpu]==0.35.2` in its `requirements.txt`). Its model list has 145
entries: Roformer-class models (BS-Roformer and MelBand-Roformer checkpoints by viperx, Kimberley Jensen, unwa, becruily,
Gabox, aufr33, …), MDX23C, MDX-NET ONNX models and VR-arch models (`default_settings.json` also has a Demucs tab).
Its Roformer defaults are `segment_size=256` (only
used when `override_model_segment_size` is on, which it is not), `overlap=8`, `batch_size=1`, autocast on CUDA,
`normalization_threshold=0.9`.

We run the same library locally (0.48.0, newer than the Space's pin) with the model's own chunk size, with two
deliberate differences: (1) `normalization_threshold=1.0` / `amplification_threshold=0`: the stems keep the gain of the
mix, so `vocals + instrumental == mix` up to 16-bit rounding and the per-episode levels stay comparable;
(2) `overlap=4` instead of 8 (step = chunk / overlap; 4 is the BS-Roformer checkpoint's own default) to halve the
runtime of 10 h of audio on a shared GPU — the comparison re-runs the winner with overlap 8 on the synthetic remix and
reports the SDR difference ("overlap check"). For the single-target
Roformer vocal models the library computes `instrumental = mix − vocals`.

**Candidates** (`separation.candidates` in `configs/audio.yaml`):

| key | file | why |
|---|---|---|
| `melband_kim` | `vocals_mel_band_roformer.ckpt` (Kimberley Jensen) | the original MelBand-Roformer vocals checkpoint (11 s chunks) |
| `melband_kim_ft2_bleedless` | `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt` (unwa FT2 "bleedless") | fine-tune of the Kim model named for low instrumental bleed in the vocal stem — the property TTS references and speaker embeddings need most (already on disk) |
| `bs_roformer_viperx_1297` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (viperx) | viperx's BS-Roformer vocals checkpoint (SDR 12.98 dB on music per its file name), the Space's first entry |
| `htdemucs_ft` | `htdemucs_ft.yaml` (Demucs v4) | the brief's fallback, as a baseline |

**Proxies (no ground truth).** `audio_sepcompare.py` separates every candidate on
(a) real excerpts picked automatically from train episodes — the window with 35–75 % subtitle-dialogue coverage whose
voice-free gaps (no subtitle line within 0.5 s and no speech according to Silero VAD on the mix) are the loudest, i.e.
music keeps playing between the lines, outside OP/ED — and
(b) one synthetic remix with pseudo ground truth: "dry" dialogue lines (context before/after below −50 dBFS and at
least 28 dB under the line, pauses inside the line below −48 dBFS, so no music under them; 37 such lines = 84 s exist in
episodes 2–11) concatenated on one track, music-only spans (≥ 6 s, ≥ 1 s away from any subtitle line, outside song /
OP / ED spans, and with < 5 % voice activity according to Silero VAD — 35 of 47 candidate spans pass; the rejected ones
contain unsubtitled voices or singing, which the 11 s smoke test exposed as "music" landing in the vocal stem) on the
other, mixed at a speech-to-music ratio of +5 dB.
On (b) a real SDR can be computed.

| metric | definition | better |
|---|---|---|
| `syn_sdr_voc`, `syn_sdr_inst` | SDR of the vocal / instrumental stem against the dry-speech / music reference of the synthetic remix (dB) | higher |
| `bleed_db` | vocal-stem energy in voice-free frames (≥ 0.5 s from any subtitle line, no VAD speech in the mix) relative to its energy inside dialogue (dB) | lower |
| `gap_supp_db` | vocal-stem energy in those voice-free frames relative to the mix energy there (dB) | lower |
| `asr_logprob`, `asr_wordprob` | faster-whisper large-v3 confidence on the vocal stem | higher |
| `cer_others` | CER of this stem's transcript against the other candidates' transcripts (transcript stability) | lower |
| `line_cov` | share of subtitle lines with an ASR word inside | higher |
| `inst_speech` | characters Whisper still transcribes confidently from the *instrumental* stem / characters in the vocal stem | lower |
| `residual_db` | ‖mix − (vocals + instrumental)‖² / ‖mix‖² (sanity check only: ≈ quantisation noise by construction) | — |
| `rtf` | realtime factor | higher |

**Selection rule (fixed before any number was seen):** mean rank over
`[syn_sdr_voc, syn_sdr_inst, bleed_db, asr_logprob, cer_others, inst_speech]`, ties broken by `syn_sdr_voc`. The winner
is written to `data/interim/anime/separation_choice.json`, which `audio_separate.py` reads
(`--model` / `AUDIO_SEP_MODEL` > choice file > `separation.model` in the config, default
`mel_band_roformer_kim_ft2_bleedless_unwa.ckpt`; `audio_batch.sh` runs the comparison before the first separation if no
choice exists).

Caveats of the proxies: about 4 % of an episode is voiced but not subtitled (measured with VAD on episodes 2 and 5);
those frames are excluded from the bleed proxy, but vocalisations the VAD misses (gasps, crowd walla) still count as
"bleed" for every model;
"dry" lines can still carry room tone and "music-only" spans can carry SFX, so the synthetic SDR is a relative measure;
the ASR proxies saturate when all stems are clean. **Listening is not done** (an agent cannot listen): the stems of
every candidate are kept under `runs/audio_sep_compare/full/<model>/` for a human A/B.

**The comparison runs under the same VRAM budget as the batch** (`AUDIO_VRAM_GB`, default 5 GB): every candidate gets
the largest segment that fits, so the ranking reflects how the batch will actually run. For the winner the script also
reports, on the synthetic remix, overlap 8 vs. 4 and segment sizes 512 and 256 (SDR, speed and peak VRAM per setting),
i.e. what a smaller budget would cost.

**Deviation from the brief:** the local GPU was fully occupied by another user's job while this was written, and
Roformer inference on the shared CPU runs at 0.03–0.1× realtime (measured), so the comparison could not be run on CPU
even with a 60–90 s excerpt (estimated ≈ 3 h of 8 threads including the ASR proxies). The full comparison (2 × 180 s real excerpts
from episodes 5 and 16 + a synthetic remix of up to 90 s) is queued for the GPU (`tmux audio_cmp`) and picks the model
with the rule above; the code path was validated on CPU with an 11 s excerpt (see "Verification status").

## 2. ASR

faster-whisper large-v3 (local `models/asr/faster-whisper-large-v3`; `int8_float16` on GPU by default to stay near
3 GB, `float16` with `AUDIO_ASR_VRAM_GB=6`), Japanese, beam 5, word
timestamps, on `vocals.wav` (decoded to 16 kHz mono by faster-whisper). Silero VAD (shipped with faster-whisper) removes
music-only stretches before decoding (`min_silence 400 ms`, `speech_pad 250 ms`); `condition_on_previous_text=False`
avoids repetition loops; the main character names are passed as `hotwords`. Segments that consist of a known Whisper
hallucination ("ご視聴ありがとうございました", …) are dropped. WhisperX was not used: faster-whisper's own cross-attention
word timing is enough for alignment at subtitle granularity and avoids a second (wav2vec2) model.

`epNN.json` = `[{"start","end","text","words":[{"w","start","end","prob"}],"avg_logprob","no_speech_prob"}]`; segments
inside OP/ED/insert-song spans (song subtitles or `oped.json`) additionally carry `"song": true` (transcribed lyrics).
Sanity numbers (`asr/summary.json`): `line_recall` = share of subtitle dialogue lines (≥ 0.4 s) with at least one ASR
word inside (±0.3 s); `word_in_dlg` = share of ASR words inside a subtitle dialogue/song span (±0.5 s) or an OP/ED span.

## 3. Speaker embeddings and voice clusters

**Embedding model:** WeSpeaker ResNet293-LM (VoxCeleb2-dev, 256-d, 0.53 % EER on VoxCeleb1-O with plain cosine scoring per its model card;
CC-BY-4.0, not gated) from the official ONNX export `Wespeaker/wespeaker-voxceleb-resnet293-LM`. pyannote was not used
(gated pipeline, and diarisation is not needed: subtitle events already give the segments). The ONNX graph is converted
with onnx2torch and run by torch on CUDA, because the onnxruntime-gpu 1.31 wheel needs CUDA 13 while the torch wheel
ships CUDA 12 (outputs identical to onnxruntime to 3e-7); on CPU onnxruntime is used. Front end as in WeSpeaker: 80-dim
Kaldi fbank, energy-based frame selection (frames within 35 dB of the loud part), CMN.

**Per line:** every `kind == "dialogue"` event ≥ 0.4 s is cut from the vocal stem with 0.1 s padding per side, short
lines are padded symmetrically to 1.0 s, long ones centre-cropped to 12 s. Lines with < 0.25 s of energy in the vocal
stem (Komi's silent lines, title cards in the SRT track) are not embedded.
`spk/epNN.npz`: `idx` (index into `subs/epNN.json`), `emb` [N,256] L2-normalised, `dur`, `rms`, plus `start`, `end`,
`speech_dur`, `overlap` (line overlaps another line by > 0.2 s), `multi` (two-speaker subtitle "- A - B").

**Clustering (TRAIN episodes 1–21 only):** agglomerative, average linkage, cosine distance, on "core" lines
(≥ 1.2 s, ≥ 0.8 s of speech, not overlapping, not two-speaker). The distance threshold is chosen without labels:
*must-link pairs* are consecutive subtitle lines of an unfinished sentence (first line ends with a comma / dash /
ellipsis / no terminal punctuation, next line starts lower-case within 0.3 s) — almost always one speaker; for each
threshold `J = P(must-link pair in one cluster) − P(random pair in one cluster)` is computed and the smallest threshold
within 0.01 of the best `J` is used (conservative: over-splitting a character into two clusters is harmless for
naming-by-vote, merging two characters is not); cuts where one cluster holds > 40 % of the core lines are excluded. Clusters with < 8 core lines are dissolved. Centroid = normalised mean
of the core members. Every other train line (short, overlapping, dissolved) goes to the nearest centroid if cosine ≥ 0.35,
else it stays unassigned. Held-out lines (22–24) are only assigned to the nearest centroid (never used for centroids).

**Re-clustering as episodes arrive:** `python -m src.anime.speakers cluster` (the `spk` stage runs it every time) uses
every train `epNN.npz` present. Cluster ids are inherited from the previous `clusters.json` (greedy one-to-one match of
the new centroids to the old ones, cosine ≥ 0.75; unmatched clusters get new ids), so names attached to ids stay
valid; `id` is therefore not the row in `centroids.npy` (that is `row`). In a CPU test (4 → 8 train episodes, raw-mix
embeddings) 10 of 15 ids were inherited with centroid cosines 0.95–0.99 and 82 % of the lines kept their id.
`--fresh` renumbers from 0.

`clusters.json`:
```
{"params": {..., "sweep": [...]},
 "clusters": [{"id","row","id_from_previous_run","size","core_size","centroid_file","total_dur","mean_cos","n_episodes",
               "episodes":{"ep01":n,...},
               "exemplars":[{"ep","idx","start","end","text","dur","cos","core"}]}],   # 16 exemplars, spread over episodes
 "assignments": {"epNN:idx": cluster_id},     # every line that has a cluster: TRAIN lines + HELD-OUT lines (nearest centroid)
 "heldout_assignments": {"epNN:idx": {"cluster_id","nearest","cosine","heldout": true}},   # the held-out subset, flagged
 "unassigned": ["epNN:idx", ...],             # embedded lines below the cosine floor (not in "assignments")
 "sanity": {...}}
```
`ep` in exemplars is the episode number (int); `idx` indexes `subs/epNN.json`. This matches what
`src/anime/name_speakers.py` reads (`assignments["epNN:idx"]`, `clusters[].id`): held-out lines are in `assignments` so
they can *receive* a name, but sizes, centroids, exemplars and `cluster_lines()` are TRAIN-only, and anything that
collects voice references must filter by episode (or use `cluster_lines`, which drops episodes 22–24 by default).
Keys other stages write into a cluster (`name`, `confidence`, `votes`) are carried over when the cluster keeps its id.

```python
from src.anime.speakers import load_centroids, assign, get_assignment, cluster_lines
ids, C = load_centroids()                 # ids [K], C [K,256]
cid, cos = assign(emb)                    # nearest centroid; assign(emb, min_cos=0.35) returns -1 below the floor
get_assignment(22, 17)                    # {"cluster_id":..,"nearest":..,"cosine":..,"heldout":True}; cluster_id -1 = none
cluster_lines(3)                          # [(episode, subtitle idx)] train only; include_heldout=True adds ep22-24
from src.anime.audio_embed import SpeakerEmbedder   # same embedding for QA of generated speech
e = SpeakerEmbedder().embed_file("line.wav")
```

**Label-free purity checks** (`sanity` in `clusters.json`): must-link pairs in the same cluster vs. random pairs;
the narrator's fixed introduction (subtitle texts ≥ 20 characters that recur verbatim in ≥ 5 train episodes, e.g.
"they struggle to communicate with others") falling into one cluster;
mean cosine of two-speaker/overlapping lines vs. core lines.

## 4. BGM cue library (TRAIN only)

From `instrumental.wav`: frames above −45 dBFS, gaps ≤ 0.6 s closed, regions ≥ 8 s; regions longer than 40 s are cut at
the quietest point between 24 and 40 s. Skipped: `kind == "song"` subtitle spans (±4 s), spans in
`data/processed/anime/oped.json` if present, and "sung blocks" — ≥ 40 s where the *vocal* stem is active without
dialogue subtitles for > 35 % of a 20 s window (±8 s). The last rule matters because **season 2 (ep13–24) has no song
subtitles**, so its OP/ED can only be found through the singing in the vocal stem (or through `oped.json`). Cues with
> 85 % dialogue coverage or > 35 % unsubtitled vocal activity are dropped. Each cue is loudness-normalised to −20 LUFS
(peak ≤ −1 dBFS, gain ≤ +24 dB, 30 ms fades) and saved as `cues/<epNN>_<start>.wav`.

CLAP: **`laion/clap-htsat-unfused`** (local copy, transformers `ClapModel`), 48 kHz mono, mean of normalised 10 s
windows (5 s hop; windowing by hand because the checkpoint's default `rand_trunc` is random). The brief's suggestion
`laion/larger_clap_music` was tried first and is **not usable**: in its Hub checkpoint the projection layers are
degenerate (all text embeddings have cosine 0.999 to each other, `logit_scale` ≈ 1, audio–text cosines ≈ 0.01 for every
prompt; measured on CPU, no missing keys when loading). On four real excerpts and on 9 music-only spans vs. 30 dry
dialogue lines `clap-htsat-unfused` separated music from speech with a wider margin than `laion/larger_clap_general`
(median music score 0.28 vs. 0.08 for speech, AUC 0.87–0.89 for both; the general model's cosines stay below 0.2), so
it is the default; `larger_clap_general` is kept in `models/clap/` as an alternative (`bgm.clap_model`; embeddings are
cached per model). Newer music–text models (MuQ-MuLan, CLaMP 3) were not evaluated.
A candidate is rejected (→ `rejected.jsonl`) only when it is clearly not music: mean cosine to the music prompts
(`music_score`) < 0.12 **and** below the best non-music prompt (SFX / ambience / speech / silence; `music_margin` < 0).
Tags = top-3 zero-shot moods; `bpm` from librosa.

`cues.jsonl` row: `{"id","ep","start","end","path","dur","rms","emb_file","emb_idx","rms_db","lufs_in","gain_db",
"dlg_ratio","voc_ratio","sung_ratio","music_score","music_margin","tags","bpm"}`; row *i* ↔ `clap.npy[i]`; paths relative to
the repo root.

`retrieve(text_query, min_dur, k=5, max_dlg_ratio=1.0, dialogue_penalty=0.05, diverse=True, exclude_ids=None)`:
the query ("gentle_comedy" → alias text, or free text; three prompt templates averaged) is embedded with the CLAP text
tower on CPU (disk-cached in `cache/bgm_text/`, so repeated moods need no model), score = cosine − 0.05 · `dlg_ratio`
(prefers spans that had little dialogue over them), near-duplicate cues (audio cosine > 0.92, the same track in another
episode) are skipped, and if no cue is ≥ `min_dur` the longest ones come back with `"too_short": true`.

## Known failure cases / limits

- **Vocal models are trained on singing.** Shouts, whispers and breathy lines can be partly left in the instrumental;
  loud SFX with harmonic content (bells, chimes, phone jingles) can leak into the vocal stem. Crowd walla is split
  between the stems.
- **OP/ED and insert songs** end up in the vocal stem (singing) — fine for BGM (they are skipped) but sung lines that are
  subtitled as dialogue would be embedded as "speakers"; such clusters show lyrics as exemplars.
- **Season 2 OP/ED detection** relies on the sung-block heuristic until `oped.json` exists; instrumental intros/outros
  of the songs shorter than the ±8 s padding can slip into the cue library.
- **ASR**: character names come out in varying kanji (只野 / 忠野); very short interjections ("えっ", "あ") get low
  word probabilities; VAD can drop quiet mumbling, which lowers `line_recall` for Komi-style lines.
- **Speaker embeddings** are trained on real (mostly English) speech. Anime acting moves one voice actor across
  several clusters (shouting, whispering, chibi/comedic voice) and can merge similar young female voices. Lines shorter
  than ~1 s are unreliable (they are assigned, not used to build clusters). Subtitle timing, not ASR timing, defines the
  segments, so a line can contain a second voice (flags `overlap` / `multi`).
- **Cluster names** are not assigned here (done later from manga-alignment votes).
- **BGM cues** still contain SFX and ambience that were under the music, and separation residue where dialogue was
  (`dlg_ratio` tells how much). CLAP text→music retrieval is coarse: it separates fast/slow, bright/dark well, subtle
  moods ("awkward") less so. Tags are zero-shot guesses.
- The comparison's proxies are not a listening test (see §1).
- **VRAM fallback is sticky.** If separation hits a CUDA OOM (its own cap, or the card being full because a neighbour
  grew), it continues with a smaller segment and remembers it in `data/interim/anime/separation_settings.json` for the
  following episodes; `separation.json` lists `segment` and `oom_fallbacks` per episode. Delete the settings file to try
  the full 8 s chunk again; `--force` re-separates an episode. Shorter segments give the model less context (cost
  measured by the comparison's segment check).
- **Free-VRAM arithmetic on this card:** 24 564 MiB total, ≈ 0.7 GB used by the desktop. Next to an 18 GB server
  ≈ 5.4 GB are free (`AUDIO_VRAM_GB=5` works); next to a 19 GB server only ≈ 4.4 GB (`AUDIO_VRAM_GB=4`, ASR at its
  default 4 GB request still fits).

## Weights (downloaded once)

| What | Source | Local path |
|---|---|---|
| MelBand-Roformer Kim / unwa FT2 bleedless / becruily karaoke | already in `~/.cache/huggingface/hub` (symlinked) | `models/separation/*.ckpt` |
| BS-Roformer viperx 1297, htdemucs_ft, Roformer yaml configs | audio-separator model repository (GitHub releases / fbaipublicfiles) | `models/separation/` |
| faster-whisper large-v3 | `Systran/faster-whisper-large-v3` | `models/asr/faster-whisper-large-v3` |
| WeSpeaker ResNet293-LM (+ ResNet34-LM, unused) | `Wespeaker/wespeaker-voxceleb-resnet293-LM` | `models/spk/` |
| CLAP | `laion/clap-htsat-unfused` (default), `laion/larger_clap_general` (alternative) | `models/clap/` |

## Verification status (2026-11-10, before anything ran on the GPU)

The local GPU was fully occupied by another user's job, so everything below was checked on CPU only
(`CUDA_VISIBLE_DEVICES=""`, ≤ 8 threads, machine load average ≈ 50).

Verified on CPU:

- venv, imports, config (`CFG.audio`) in `venvs/audio` and `venvs/main`.
- Separation code path through audio-separator for the three Roformer candidates (unwa FT2 bleedless, Kim, BS-Roformer
  viperx 1297) on 11–12 s excerpts of episode 2 and a short synthetic remix: stems are 44.1 kHz stereo PCM16 with the
  length of the input, `vocals + instrumental` reproduces the mix to ≤ 1 LSB. CPU speed: MelBand ≈ 0.07–0.1× realtime
  at overlap 1, BS-Roformer ≈ 0.03× — the reason the real comparison runs on the GPU. htdemucs_ft loads and runs in
  the library but needs ≈ 35 min per 11 s clip on this CPU, so only its stem post-processing (4 stems → vocals +
  `mix − vocals`) was unit-tested.
- The whole comparison script end to end with two candidates (metrics, ASR proxies, ranking, overlap and segment
  checks, results files). On a 12 s clip at overlap 1 the proxies already behave sensibly (FT2 bleedless: synthetic
  SDR 9.1 dB vocals / 4.7 dB instrumental from an input of +4.4 / −4.4 dB, bleed −7.9 dB) — a smoke test, **not** the
  comparison result.
- Excerpt picker and synthetic remix builder on the real episodes (37 dry lines / 84 s of dry speech in episodes 2–11;
  47 candidate music-only spans, 35 after the VAD filter; the remix is 89 s with 78 s of speech).
- ASR module with faster-whisper large-v3 `int8` on 40 s of the raw mix of episode 2: 11/11 subtitle lines covered,
  99 % of the words inside subtitle spans, mean word probability 0.89, transcript reads correctly.
- Speaker embedding module on the **raw mix** of episodes 1–8 and 22 (onnxruntime CPU, ≈ 0.3 s per line) and the whole
  clustering / assignment code on those embeddings: threshold picked automatically (0.72), 18 clusters, 8 of them
  present in ≥ 7 of the 8 episodes, held-out episode 22 assigned without touching the centroids; the narrator's
  recurring intro lines (24 lines in 8 episodes) all fall into one cluster; the narrator's intro of the held-out episode
  goes to that same cluster with cosine 0.73–0.84. These numbers are for raw-mix embeddings and only validate the code;
  the real clusters are built from the vocal stems by the batch.
- BGM module (cue cutting, loudness normalisation to −20.0 LUFS, CLAP audio/text embeddings, zero-shot filter, index,
  `retrieve`) on fake stems (first 7 minutes of the raw mix of episode 2 as "instrumental") in a temp directory:
  8 cues found, music scores 0.19–0.45, tags such as "funny and mischievous / playful and silly / gentle comedy",
  `retrieve("gentle_comedy")` ranks them with similarities 0.47 / 0.31 / 0.27, the text tower alone reproduces the full
  model's text embedding (cosine 1.0). This is also where the degenerate `larger_clap_music` checkpoint was caught.
- onnx2torch conversion of the WeSpeaker model equals onnxruntime (max abs diff 3e-7); CUDA/cuDNN library preloading
  for ctranslate2 resolves all libraries (without touching the GPU).

- `audio_batch.sh` control flow with a mocked GPU wrapper and mocked Python (stage order, automatic comparison before
  the first separation, retries, lock, state file); the VRAM fallback ladder with a simulated out-of-memory error
  (801 → 512 → 384, remembered for the next episode); stable cluster ids across re-clustering.

Not verified yet (needs the GPU): every GPU code path (autocast Roformer inference under the VRAM cap, the
torch-on-CUDA speaker embedder, faster-whisper `int8_float16`/`float16` on CUDA, CLAP on CUDA), **the real peak VRAM of
every stage** (the table above holds estimates), GPU runtimes per episode, the comparison numbers and the model choice,
all batch outputs, clustering quality on vocal stems, the BGM library on real stems, and any listening. The queued job
`tmux audio_cmp` (`audio_batch.sh "5,16" cmp,probe`, ≤ 5 GB, about 10–12 minutes once 5 GB are free) produces the
comparison, the choice, the separation peak VRAM per model/segment and the peak VRAM + speed of the other three stages,
and writes them into the section below.

## Measured results (auto-generated)

<!-- AUTO:BEGIN (written by src/anime/audio_report.py - do not edit by hand) -->
_Generated 2026-11-11 05:14:09. Missing parts mean that stage has not run yet._

### Per-stage runtime and peak VRAM (measured)

- separation model: melband_kim_ft2_bleedless (mel_band_roformer_kim_ft2_bleedless_unwa.ckpt), chosen under a 4.0 GB budget, segment [801], peak VRAM 3236 MiB in the comparison
- separation: 24/24 episodes, 247 s per episode (min 116, max 2549), peak VRAM 3236 MiB, segment ['801'], model ['mel_band_roformer_kim_ft2_bleedless_unwa.ckpt']
- speaker embeddings: 24/24 episodes, 5 s per episode, peak VRAM 1308 MiB (torch)
- BGM: 462 cues from 21 episodes, CLAP indexing 89.5 s on cuda, peak VRAM 1482 MiB
- voice clusters: 37 from train episodes [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] (threshold 0.76), must-link agreement 0.8324

### Batch state (last lines of logs/audio_batch.state)

```
2026-11-11 05:02:46 spk_cluster RUNNING eps=9-21 try=1
2026-11-11 05:02:48 spk_cluster OK eps=9-21
2026-11-11 05:02:48 BATCH_DONE eps=9-21 stages=sep,spk
2026-11-11 05:02:48 bgm RUNNING eps=1-21 try=1
2026-11-11 05:06:26 bgm OK eps=1-21
2026-11-11 05:06:30 BATCH_DONE eps=1-21 stages=bgm
2026-11-11 05:06:30 sep RUNNING eps=22-24 try=1
2026-11-11 05:13:42 sep OK eps=22-24
2026-11-11 05:13:42 spk_embed RUNNING eps=22-24 try=1
2026-11-11 05:14:07 spk_embed OK eps=22-24
2026-11-11 05:14:07 spk_cluster RUNNING eps=22-24 try=1
2026-11-11 05:14:09 spk_cluster OK eps=22-24
```

### Separation comparison `full` (2026-11-11 03:22:28, device cuda, overlap 4, VRAM budget 4.0 GB)

Excerpts: ep05_0885_180s.wav (real, 180.0 s, dialogue coverage 0.63); ep16_0695_180s.wav (real, 180.0 s, dialogue coverage 0.727); synthetic_mix.wav (synthetic, 88.98 s).

Synthetic remix: 78.1 s of speech from 30 dry dialogue lines + 9 music-only spans (episodes [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), speech-to-music ratio 5.0 dB.

| model | syn_sdr_voc | syn_sdr_inst | bleed_db | gap_supp_db | asr_logprob | asr_wordprob | cer_others | line_cov | inst_speech | residual_db | rtf | segment | peak VRAM MiB | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| melband_kim (`vocals_mel_band_roformer.ckpt`) | 8.18 | 3.74 | -7.07 | -6.12 | -0.31 | 0.87 | 0.18 | 0.83 | 0.00 | -89.71 | 6.28 | [1101] | 3562 | 2.50 |
| melband_kim_ft2_bleedless (`mel_band_roformer_kim_ft2_bleedless_unwa.ckpt`) | 8.17 | 3.74 | -7.65 | -6.71 | -0.22 | 0.89 | 0.17 | 0.85 | 0.00 | -89.71 | 6.47 | [801] | 3236 | 2.00 |
| bs_roformer_viperx_1297 (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) | 8.12 | 3.68 | -6.78 | -5.79 | -0.26 | 0.89 | 0.17 | 0.83 | 0.00 | -60.02 | 3.46 | [801] | 2818 | 3.17 |
| htdemucs_ft (`htdemucs_ft.yaml`) | 8.40 | 3.96 | -7.82 | -6.91 | -0.27 | 0.87 | 0.23 | 0.77 | 0.00 | -89.71 | 4.33 | None | 2060 | 2.33 |

Ranked on ['syn_sdr_voc', 'syn_sdr_inst', 'bleed_db', 'asr_logprob', 'cer_others', 'inst_speech']. **Choice: melband_kim_ft2_bleedless (`mel_band_roformer_kim_ft2_bleedless_unwa.ckpt`)**. Stems for listening: `/home/user/komi-poc-work/runs/audio_sep_compare/full`.

Overlap check (melband_kim_ft2_bleedless, synthetic remix): overlap 8 -> syn_sdr_voc 8.18 / syn_sdr_inst 3.74 dB at 4.24x realtime; overlap 4 (used for the batch) -> 8.17 / 3.74 dB at 7.01x realtime.

Segment size vs. quality and VRAM (winner, synthetic remix):

| setting | overlap | segment (dim_t) | syn_sdr_voc | syn_sdr_inst | rtf | peak VRAM MiB |
|---|---|---|---|---|---|---|
| used in the comparison | 4 | 801 | 8.17 | 3.74 | 7.01 | 2824 |
| segment 512 | 4 | 512 | 8.2 | 3.77 | 8.89 | 2950 |
| segment 256 | 4 | 256 | 8.32 | 3.89 | 9.76 | 2640 |

ASR proxies ran with {'peak_vram_mb': 3608, 'compute_type': 'int8_float16', 'seconds': 300.9, 'audio_seconds': 3240.0, 'rtf': 10.8}.

### Batch separation

| ep | model | segment | overlap | runtime s | x realtime | peak VRAM MiB | residual dB | OOM fallbacks |
|---|---|---|---|---|---|---|---|---|
| 1 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 185.8 | 7.71 | 3236 | -90.99 |  |
| 2 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 189.69 | 7.55 | 2642 | -91.42 |  |
| 3 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 185.86 | 7.7 | 2642 | -89.66 |  |
| 4 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 2548.96 | 0.56 | 2642 | -90.94 |  |
| 5 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 143.71 | 9.97 | 3236 | -89.85 |  |
| 6 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 140.29 | 10.21 | 2642 | -89.31 |  |
| 7 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 137.22 | 10.44 | 2642 | -90.69 |  |
| 8 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 137.39 | 10.42 | 2642 | -87.37 |  |
| 9 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 142.16 | 10.07 | 3236 | -90.04 |  |
| 10 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 144.95 | 9.88 | 2642 | -91.4 |  |
| 11 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 141.68 | 10.11 | 2642 | -90.74 |  |
| 12 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 115.63 | 12.34 | 2642 | -92.39 |  |
| 13 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 132.27 | 10.83 | 2642 | -89.63 |  |
| 14 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 143.31 | 9.99 | 2642 | -90.75 |  |
| 15 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 143.77 | 9.96 | 2642 | -89.39 |  |
| 16 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 145.39 | 9.85 | 2642 | -89.15 |  |
| 17 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 143.53 | 9.98 | 2642 | -90.52 |  |
| 18 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 162.78 | 8.8 | 2642 | -89.13 |  |
| 19 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 141.45 | 10.12 | 2642 | -90.22 |  |
| 20 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 145.1 | 9.87 | 2642 | -89.99 |  |
| 21 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 145.59 | 9.84 | 2642 | -86.69 |  |
| 22 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 139.55 | 10.26 | 3236 | -89.18 |  |
| 23 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 138.49 | 10.34 | 2642 | -89.75 |  |
| 24 | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | 801 | 4 | 140.19 | 10.22 | 2642 | -88.7 |  |

Not separated yet: none.

### Voice clusters

37 clusters from 7292 train lines (5235 core lines; train episodes with embeddings: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]); cosine-distance threshold 0.76 (auto: smallest distance threshold whose J = must_link_same - random_same is within 0.01 of the maximum (cuts where one cluster holds > 40 % of the core lines are excluded)). Built 2026-11-11 05:14:09, 37 ids inherited from the previous run.
Sizes: [1313, 1027, 1177, 617, 525, 341, 312, 300, 119, 115, 61, 65, 111, 78, 42, 34, 36, 22, 57, 53, 50, 34, 60, 44, 20] …. Train lines assigned: 96.1%. Held-out lines: 1030 (assigned 0.955, mean cosine 0.582).

Label-free purity checks:

- must-link pairs (subtitle continuation lines, 185 pairs) in the same cluster: **0.832** (random pairs: 0.101)
- narrator intro lines: {"lines": 63, "episodes": 16, "modal_cluster": 4, "modal_share": 0.9206, "unassigned": 0, "example_texts": ["A communication disorder", "It doesn't mean they don't want to.", "Or references a person having such symptoms.", "The rules are simple!", "They only struggle to form connections,", "They only struggle to form connections."]}
- mean cosine to own centroid: core 0.605, non-core 0.544, two-speaker/overlapping lines 0.563

| id | size | core | episodes | seconds | mean cos | example lines |
|---|---|---|---|---|---|---|
| 0 | 1313 | 1030 | 21 | 3239.8 | 0.61 | Maybe if you tried writing what you want / So after all of that, you weren't able t / It's so dark out, I'll walk you home. |
| 1 | 1027 | 832 | 21 | 2474.4 | 0.5932 | Komi-san's budget is 10 000 Yen, it shou / Is there anything you'd like to eat? / The coworkers who usually work with me c |
| 3 | 1177 | 692 | 21 | 2417.6 | 0.5844 | Owwwwww! / Pocket tissues. / Ow! |
| 2 | 617 | 497 | 21 | 1641.4 | 0.5749 | but for some reasons, it doesn't make yo / Ten dregs are worth one point. / If I do something wrong, don't hesitate  |
| 4 | 525 | 433 | 21 | 1417.6 | 0.6854 | So Tadano stayed home even though he did / who fell in love with his reflection in  / Neither of them could focus on the test. |
| 7 | 341 | 280 | 20 | 885.6 | 0.5725 | Because Komi-san is a beauty that's hard / First, who decided that it was better to / I realized at which degree my heart was  |
| 23 | 312 | 259 | 19 | 810.9 | 0.5548 | I'll elegantly escort Tadano! / This is the room where Tadano and I and  / Everyone usually looks away when I meet  |
| 18 | 300 | 253 | 19 | 802.5 | 0.5661 | Wherever you sit on the veranda, one of  / I'll buy you a souvenir at the hot sprin / You're looking ill again! |
| 24 | 119 | 71 | 17 | 295.3 | 0.5789 | It's a class tug of war tournament! / Coming next is the brilliantly silent pe / Why did I dress up like a girl? |
| 25 | 115 | 61 | 19 | 261.2 | 0.5292 | I have perfected the theme song "Endless / No! Absolumentely not! / Say to me, "I lost." |
| 26 | 61 | 55 | 10 | 166.4 | 0.6887 | I-I didn't have anyone close enough to c / Why can't I talk in front of them? / Tangerines (mikan) on an aluminum can (m |
| 27 | 65 | 49 | 13 | 144.0 | 0.5867 | This statement isn't meant to be dirty. / No, not long. I just got here. / Inaka is from the countryside and everyo |
| 17 | 111 | 47 | 18 | 215.8 | 0.4904 | S-She was a customer after all! / Wait, Komi? Hanging out with us? / Komi-san! |
| 28 | 78 | 38 | 21 | 179.9 | 0.517 | You want me to persevere. / As expected, grand gestures that stands  / She's really raring to go! |
| 16 | 42 | 38 | 3 | 101.2 | 0.682 | One point is a hundred yen. Winnings go  / Any anxiety or difficult situations? / It'll be a good experience. Go on. |
| 12 | 34 | 31 | 2 | 85.9 | 0.6106 | I am one of the few to have passed the a / In my fifteen years as a Standbakes bari / As a coffee master and a Dark Apron Bari |
| 29 | 36 | 28 | 12 | 80.9 | 0.5816 | Who would I thought I'd be battling you  / Ren Yamai! / Oh, wow! |
| 30 | 22 | 21 | 2 | 64.1 | 0.6469 | Once again today, the cafe is full of id / You thought I wouldn't act on that infor / Humans don't think about cats' feelings  |
| 31 | 57 | 20 | 17 | 111.4 | 0.5131 | Being alone with Komi? / There's not really a point to reading ou / My mentor is so formidable! |
| 32 | 53 | 19 | 18 | 127.1 | 0.5277 | I've joined in! Can I make an order? / "I'll do what Yamai does, / I'm so glad we came to Movie Land. |
| 33 | 50 | 19 | 18 | 100.6 | 0.5564 | Please advertise the maid cafe! / have to win! / Please come and visit us please. |
| 34 | 34 | 19 | 15 | 74.4 | 0.5296 | Eh? A school swimsuit? / W- W- What's wrong‪ Komi-san! / Let's go! Three, two, one! |
| 35 | 60 | 18 | 20 | 123.3 | 0.503 | Hey, Tadano! What is she doing?! / Is it such a sin?! / Me too! I don't want it to end either! |
| 36 | 44 | 17 | 19 | 105.1 | 0.4925 | First make a cute face and then smile! / M-My Chrysanthemum Blade of Darkness! / W-Who's that girl? |
| 37 | 20 | 16 | 6 | 55.2 | 0.5941 | Akira is around there somewhere, so coul / What? Is that Shoko? / We're leaving Kyoto already?! |
| 38 | 59 | 16 | 17 | 122.3 | 0.5263 | It's my chance! Hey, Komi… / Huh? Wow, that's awesome. / How will she manage to lose today? |
| 39 | 30 | 14 | 5 | 68.9 | 0.5643 | Although you're disguised as a maiko, I  / L-Like this? Is that alright? / Commu is watching me. |
| 40 | 48 | 13 | 16 | 80.9 | 0.5357 | I HOPE WE CAN GO TO THE ARCADE AGAIN. / WITH OLD BUILDINGS! / Oh no! |
| 41 | 31 | 13 | 17 | 85.1 | 0.5204 | My goal for today is to touch Komi's boo / Komi-san's swimsuit! / And she's pretty cute! |
| 42 | 28 | 11 | 9 | 52.3 | 0.4971 | Can… we join in? / Please, come in! / She's… |
| 14 | 55 | 11 | 17 | 106.2 | 0.5196 | Yahoo! / Najimi stop! / So? A single cup!? No? No? |
| 43 | 17 | 10 | 8 | 33.1 | 0.5977 | Noko, here's our temporary replacement. / They're hot, so be careful! / These pancakes are delicious! |
| 44 | 32 | 10 | 17 | 50.8 | 0.496 | Oh! You! / Umeboshi? / The granny's aura just changed! |
| 45 | 12 | 9 | 6 | 23.2 | 0.6172 | Who can answer this question? / Looking for... four! / Oh, thank you, Tadano. |
| 46 | 40 | 9 | 17 | 57.5 | 0.5038 | Boom! / Uh, Komi? / Moun- Mountains...? |
| 47 | 26 | 8 | 14 | 45.0 | 0.5281 | Yes, we were, Tadano! / It has come to this. Time for the big gu / Shouko! Shouko you're- |
| 48 | 15 | 8 | 8 | 28.4 | 0.5675 | It's a business after all... / You're right, sweetie! / Ah, me too! Me too! |

Threshold sweep (t = distance threshold, J = must_link_same - random_same):

| t | clusters>=min | must_link_same | random_same | J | coverage | largest share |
|---|---|---|---|---|---|---|
| 0.3 | 22 | 0.1243 | 0.0005 | 0.1238 | 0.0734 | 0.0122 |
| 0.32 | 32 | 0.1568 | 0.0007 | 0.1561 | 0.1074 | 0.0139 |
| 0.34 | 44 | 0.2 | 0.0008 | 0.1992 | 0.1467 | 0.0155 |
| 0.36 | 46 | 0.2486 | 0.001 | 0.2476 | 0.183 | 0.0189 |
| 0.38 | 48 | 0.2541 | 0.0014 | 0.2527 | 0.2159 | 0.0201 |
| 0.4 | 54 | 0.2919 | 0.0022 | 0.2897 | 0.264 | 0.0248 |
| 0.42 | 61 | 0.3135 | 0.0037 | 0.3099 | 0.3169 | 0.0319 |
| 0.44 | 70 | 0.3568 | 0.005 | 0.3518 | 0.3664 | 0.0357 |
| 0.46 | 72 | 0.4054 | 0.0087 | 0.3968 | 0.4159 | 0.0518 |
| 0.48 | 79 | 0.4324 | 0.0105 | 0.422 | 0.4669 | 0.0531 |
| 0.5 | 77 | 0.4811 | 0.0127 | 0.4684 | 0.5123 | 0.0562 |
| 0.52 | 80 | 0.4973 | 0.0159 | 0.4814 | 0.567 | 0.06 |
| 0.54 | 80 | 0.5459 | 0.0256 | 0.5204 | 0.6071 | 0.114 |
| 0.56 | 88 | 0.573 | 0.0271 | 0.5459 | 0.6609 | 0.114 |
| 0.58 | 78 | 0.6162 | 0.0309 | 0.5854 | 0.7009 | 0.1175 |
| 0.6 | 84 | 0.6378 | 0.0335 | 0.6044 | 0.7488 | 0.1226 |
| 0.62 | 73 | 0.6757 | 0.0385 | 0.6372 | 0.7832 | 0.1301 |
| 0.64 | 71 | 0.6865 | 0.0468 | 0.6397 | 0.8197 | 0.1509 |
| 0.66 | 66 | 0.7081 | 0.0518 | 0.6563 | 0.8426 | 0.162 |
| 0.68 | 62 | 0.7135 | 0.0571 | 0.6564 | 0.867 | 0.1668 |
| 0.7 | 53 | 0.7459 | 0.0718 | 0.6742 | 0.8926 | 0.1685 |
| 0.72 | 49 | 0.7892 | 0.0899 | 0.6993 | 0.9146 | 0.1882 |
| 0.74 | 46 | 0.7892 | 0.0936 | 0.6956 | 0.9362 | 0.1914 |
| 0.76 | 37 | 0.8378 | 0.105 | 0.7328 | 0.9503 | 0.1968 |
| 0.78 | 33 | 0.8432 | 0.1247 | 0.7185 | 0.9597 | 0.2124 |
| 0.8 | 22 | 0.8541 | 0.1685 | 0.6856 | 0.9679 | 0.2596 |
| 0.82 | 18 | 0.8595 | 0.2188 | 0.6406 | 0.9815 | 0.3257 |
| 0.84 | 15 | 0.8811 | 0.336 | 0.5451 | 0.9899 | 0.4783 |
| 0.86 | 12 | 0.8919 | 0.3615 | 0.5304 | 0.9952 | 0.4909 |

### BGM cue library

462 cues (11214 s, median 25.0 s) from episodes [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]; 52 cues with <= 20 % dialogue coverage; 27 candidates rejected as non-music by CLAP; CLAP `laion/clap-htsat-unfused` on cuda, embedding + indexing 89.5 s, peak VRAM 1482 MiB. Built 2026-11-11 05:06:25.

Retrieval examples (top 3):

- `gentle_comedy` -> ep17_0081.06 (24 s, sim 0.62, tags ['funny and mischievous', 'playful and silly']), ep13_1040.70 (14 s, sim 0.61, tags ['funny and mischievous', 'playful and silly']), ep12_0583.84 (25 s, sim 0.60, tags ['funny and mischievous', 'playful and silly'])
- `tense, awkward silence` -> ep01_0938.04 (29 s, sim 0.46, tags ['emotional piano', 'dramatic and epic']), ep12_0583.84 (25 s, sim 0.42, tags ['funny and mischievous', 'playful and silly']), ep17_0081.06 (24 s, sim 0.40, tags ['funny and mischievous', 'playful and silly'])
- `heartwarming` -> ep01_0938.04 (29 s, sim 0.62, tags ['emotional piano', 'dramatic and epic']), ep12_0583.84 (25 s, sim 0.56, tags ['funny and mischievous', 'playful and silly']), ep21_0924.88 (16 s, sim 0.57, tags ['emotional piano', 'funny and mischievous'])
- `energetic` -> ep18_1310.88 (28 s, sim 0.63, tags ['heartwarming', 'energetic and upbeat']), ep08_0299.30 (13 s, sim 0.57, tags ['funny and mischievous', 'playful and silly']), ep19_0689.00 (26 s, sim 0.54, tags ['funny and mischievous', 'playful and silly'])

<!-- AUTO:END -->
