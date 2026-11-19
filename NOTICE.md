# Notice: third-party works, media in this repository, removal requests

**This is an independent, fan-made research project. It is not affiliated with, endorsed by or sponsored by the
creators, publishers, producers or distributors of any work it was tested on.**

## The work the pipeline was tested on

*Komi Can't Communicate* (古見さんは、コミュ症です。) is a manga by Tomohito Oda, published by Shogakukan; its TV anime
adaptation was produced by OLM and is distributed internationally by Netflix. All characters, artwork, episodes, music
and performances belong to their respective owners.

## What this repository contains, and what it does not

| In the repository | Not in the repository |
|---|---|
| Source code of the pipeline (MIT) | Any manga volume or page set |
| Short, silent, downscaled excerpts used to show evaluation results: generated clips, and a few seconds each of twelve shots of the real episode as the reference they are measured against | Any episode, in full or in substantial part |
| A few single manga panels shown as the input of those twelve samples | Model weights or LoRAs trained on the show |
| | Any audio: no voices, no music |

**Voices.** The research build synthesised English voices conditioned on the official English dub, to study how far
voice and pacing can be matched. A voice belongs to the person who has it. No cloned audio, voice embedding or reference
clip is published here, and every video in this repository is silent. If you run the voice stage, use voices you have the
right and the consent to use.

**Running the code.** You need your own lawfully obtained copies of whatever manga and anime you run it on, and you are
responsible for what you do with the output. The code is published for research and education in automatic adaptation,
shot planning and evaluation.

## Removal requests

If you hold rights in material shown here and want it removed, open an issue at
https://github.com/SolvAI/MangAnime/issues (or contact the repository owner through GitHub) and it will be taken down.
