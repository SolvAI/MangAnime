# Support MangAnime

MangAnime is built and run by one person on one shared consumer GPU. If you want it to go further, there are three ways
to help.

## 1. Star the repository

It costs nothing and it is how other people find the project: **[star MangAnime on GitHub](https://github.com/SolvAI/MangAnime/stargazers)**.

## 2. Fund the next GPU

**Goal: US$7,000 for one NVIDIA RTX PRO 5000 (48 GB) workstation GPU.**

> The fundraiser page is being set up. Its link will appear here and on the website; until then, watch or star the
> repository to be told when it opens.

What the card changes, concretely:

| Today, on one shared 24 GB card | With 48 GB |
|---|---|
| One large model in memory at a time: the image model, the video model and the voice model take turns, and every swap costs a reload | The three stay loaded together; a retake of a single shot no longer waits for a model swap |
| Clips rendered at 832×480 and scaled up to 720p | Room to render at a higher native resolution |
| LoRA experiments rented by the hour in the cloud | Fine-tuning and ablations run locally |
| One episode render (about 6 hours) blocks everything else, including evaluation | Rendering and evaluation can run side by side |

What the work would go to next is in the [roadmap](README.md#roadmap): sprint 6, lighter line work, a human blind study
and a second series.

## 3. Sponsor or collaborate

Studios, labs, tool makers and individuals who want to sponsor the work, lend compute or collaborate on evaluation:
**[open a discussion](https://github.com/SolvAI/MangAnime/discussions)** or reach **[@SolvAI](https://github.com/SolvAI)** on GitHub.

---

MangAnime is independent fan-made research and is not affiliated with the rights holders of any work it was tested on
([NOTICE.md](NOTICE.md)). Contributions fund hardware and running costs of the open-source pipeline; they do not buy
episodes, and no episodes are distributed.
