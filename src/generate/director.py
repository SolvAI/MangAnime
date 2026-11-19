"""Director / screenwriter (brief §7.1): parsed chapter (text only) -> shotlist.json via DeepSeek V4.1 Flash.

Only text leaves the machine: panel JSON (characters, dialogue, captions). Panel images are never sent.
Few-shot examples are the most similar *training* chapters with their real, aligned shot lists; the prompt is
constrained by adaptation_stats.json. Calls are cached by content hash (offline replay works).

  python -m src.generate.director --chapter 1 --out runs/m1/shotlist.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json
from src.llm.client import LLM, BudgetExceeded, remote_or_local

EMOTIONS = ["neutral", "calm", "happy", "excited", "shy", "embarrassed", "nervous", "scared", "surprised", "angry", "sad", "deadpan"]
SHOT_SIZES = ["extreme close-up", "close-up", "medium close-up", "medium", "medium wide", "wide", "extreme wide", "insert"]

SHOT_SCHEMA = {
    "type": "object",
    "required": ["shots"],
    "properties": {
        "shots": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["source_panels", "duration", "shot_size", "camera", "characters", "action", "dialogue"],
                "properties": {
                    "source_panels": {"type": "array"},
                    "duration": {"type": ["number", "string"]},
                    "shot_size": {"type": "string"},
                    "camera": {"type": "string"},
                    "characters": {"type": "array"},
                    "action": {"type": "string"},
                    "dialogue": {"type": "array", "items": {"type": "object", "required": ["speaker", "text"]}},
                },
            },
        }
    },
}

DEFAULT_STATS = {
    "source": "defaults (alignment stats not built yet)",
    "shots_per_panel": 1.3, "panel_coverage": 0.85, "median_shot_sec": 3.0, "mean_shot_sec": 3.6,
    "p10_shot_sec": 1.2, "p90_shot_sec": 6.5, "sec_per_dialogue_line": 2.4, "sec_per_manga_page": 22.0,
    "shot_size_mix": {"close-up": 0.3, "medium close-up": 0.2, "medium": 0.25, "medium wide": 0.1, "wide": 0.1, "insert": 0.05},
}


def load_stats() -> dict:
    return read_json(P("processed") / "align" / "adaptation_stats.json") or DEFAULT_STATS


def roster_text(names: set[str] | None = None) -> str:
    rows = []
    for c in CFG.characters.roster:
        if names is not None and c["name"] not in names and c["role"] not in ("main", "voice_only"):
            continue
        rows.append(f"- {c['name']} ({c['full']}): {c['look']}. Voice: {c['voice']}")
    return "\n".join(rows)


def render_chapter(panels: list[dict]) -> str:
    """Compact text rendering of the parsed chapter (what the director reads)."""
    lines = []
    for p in panels:
        w, h = p["bbox"][2], p["bbox"][3]
        shape = "wide" if w > 1.6 * h else ("tall" if h > 1.4 * w else "squarish")
        head = f"P{p['order']} (page {p['page']}, {shape} panel)"
        if p.get("characters"):
            head += f" chars: {', '.join(p['characters'])}"
        if p.get("shot_size"):
            head += f" | framing: {p['shot_size']}"
        if p.get("location"):
            head += f" | location: {p['location']}"
        lines.append(head)
        if p.get("caption"):
            lines.append(f"    picture: {p['caption']}")
        for d in p["dialogue"]:
            spk = d.get("speaker") or "?"
            lines.append(f"    - [{d['kind']}] {spk}: \"{d['text']}\"")
    return "\n".join(lines)


def render_real_shots(example: dict) -> str:
    """Few-shot: a training chapter and how the real anime shot it."""
    out = [f"### Example: manga chapter {example['chapter']} as adapted in the real anime "
           f"({len(example['shots'])} shots, {example['duration']:.0f} s for {example['n_panels']} panels)"]
    out.append("Manga:\n" + example["manga_text"])
    out.append("Real shot list (JSON lines):")
    for s in example["shots"]:
        out.append(json.dumps(s, ensure_ascii=False))
    return "\n".join(out)


def fewshot_for(chapter_panels: list[dict], k: int = 3, max_shots: int = 45) -> list[dict]:
    """Retrieve the k most similar TRAIN chapters that have an aligned real shot list (built by src.align.fewshot)."""
    bank = read_json(P("processed") / "align" / "fewshot_bank.json")
    if not bank:
        return []
    from src.align.text_align import embed

    q = " ".join(d["text"] for p in chapter_panels for d in p["dialogue"] if d["kind"] != "sfx")[:4000]
    qe = embed([q])[0]
    train_eps = set(CFG.split.train_episodes)
    cands = [b for b in bank if b["episode"] in train_eps and b["chapter"] != chapter_panels[0]["chapter"]]
    if not cands:
        return []
    embs = embed([b["summary_text"][:4000] for b in cands])
    order = np.argsort(-(embs @ qe))[:k]
    ex = []
    for i in order:
        b = dict(cands[int(i)])
        b["shots"] = b["shots"][:max_shots]
        ex.append(b)
    return ex


SYSTEM = """You are the series director and screenwriter of the TV anime "{title}".
You adapt one manga chapter into an anime shot list that looks and paces exactly like the existing anime.

Show bible
- Setting: {setting}
- Look: {style}
- Conventions:
{conventions}

Characters (use exactly these names)
{roster}

How the real anime adapts the manga (statistics measured on the training episodes)
{stats}

Rules
1. Follow the chapter in order and keep its story, jokes and dialogue. Cover (almost) every panel; a panel may become
   several shots (set-up, reaction, insert), and several small panels may merge into one shot. `source_panels` lists
   the P-numbers a shot is based on (first one = the panel whose composition the first frame should follow).
2. Anime pacing: most shots last 1.5–5 s. Do NOT cut on every line: the real anime holds a shot while a character says
   two or three consecutive lines, and answers often stay in the same two-shot. Give a silent beat or reaction its own
   shot only when it carries the joke. Respect the shot limit given with the chapter. A shot must be long enough for its dialogue (Japanese speech ≈ 7–8 morae per
   second, plus ~0.3 s of air). Never exceed 8 s; split long speeches across shots (speaker, listener reaction, insert).
3. Every dialogue line keeps the manga wording in `text` (natural English, fix ALL-CAPS and OCR glitches) and gets a
   natural spoken Japanese version in `text_ja` in that character's voice (this is what the voice actor says).
   `kind` is one of: speech, thought, narration, notebook (Komi's handwritten notebook: shown on screen, not spoken;
   still give `text_ja`), sfx_text (on-screen text only). `emotion` is one of: {emotions}.
   Unknown speakers ("?") must be resolved from context; unnamed students are "Classmates".
   The speaker labels in the parsed manga come from an automatic vision model and are NOISY: it over-assigns lines to
   Komi and confuses side characters. Correct them with the dialogue itself and your knowledge of the series:
   Komi almost never speaks aloud — her spoken lines are stammers and one or two words ("A-", "G-good...", "...yes");
   fluent sentences labelled Komi belong to someone else (the friend she is with, a classmate, the Narrator) unless they
   are her inner thoughts (kind "thought") or notebook writing (kind "notebook"). Generic explanations ("People with a
   communication disorder...") are the Narrator. A character cannot answer themselves: alternate speakers sensibly.
   `text_ja` is ordinary written Japanese (kanji + kana, no spaces between words). The ONLY exception: character NAMES are
   written in hiragana (こみさん, ただのくん, なじみ, やまいさん …), never in kanji, so the speech synthesiser reads them correctly.
   Keep lines short and speakable.
4. `shot_size` is one of: {sizes}. `camera` is a short phrase such as "static", "slow push-in", "pan right",
   "tilt up", "handheld shake", "quick zoom-in".
5. HOW FRAMES ARE MADE: the first frame of a shot is a colourised 16:9 crop of its FIRST source panel (the picture is
   not redrawn). So choose the panel that already shows what the shot needs, and write `first_frame` as one sentence
   describing what THAT panel shows (who, pose, expression, framing, background) — never invent a different
   composition, location or cast for it. A second shot on the same panel becomes a punch-in on its characters.
   Use `source_panels: []` only when no panel fits (the frame is then generated from the previous shot; keep it rare,
   at most a handful per chapter). `shot_size` should match the panel's framing. `action`: what moves/changes during the shot in simple visual terms (limited TV animation: mouth flaps,
   blinks, hair sway, a head turn, a hand gesture, a simple walk). Avoid complex choreography.
6. `location` and `time_of_day` must stay consistent between consecutive shots unless the scene changes.
7. `bgm_cue`: a short mood tag for music retrieval (e.g. gentle_comedy, awkward_silence, heartwarming, energetic_comedy,
   tense_comedic, melancholic, triumphant, none). Change the cue only when the mood changes.
8. `sfx`: optional short sound-effect tags (e.g. "pencil scratching", "door slide", "heartbeat", "crowd murmur").
9. The first shot of the chapter is the title card: shot_size "insert", characters [], action "title card",
   notes "TITLE_CARD: <chapter title>", duration 3.0, no dialogue.
Return ONLY a JSON object: {{"shots": [ ... ]}} with the fields
source_panels, duration, shot_size, camera, characters, location, time_of_day, first_frame, action, dialogue, bgm_cue, sfx, notes."""


def build_messages(chapter: int, panels: list[dict], title: str, examples: list[dict], stats: dict, max_shots: int | None = None) -> list[dict]:
    names = {n for p in panels for n in p.get("characters", [])} | {d.get("speaker") for p in panels for d in p["dialogue"] if d.get("speaker")}
    show = CFG.characters.show
    stats_txt = json.dumps({k: v for k, v in stats.items() if k != "per_episode"}, ensure_ascii=False)
    system = SYSTEM.format(
        title=show.title, setting=show.setting, style=show.style,
        conventions="\n".join(f"  - {c}" for c in show.conventions),
        roster=roster_text(names if any(names) else None), stats=stats_txt,
        emotions=", ".join(EMOTIONS), sizes=", ".join(SHOT_SIZES),
    )
    user = []
    for ex in examples:
        user.append(render_real_shots(ex))
    n_pages = len({p["page"] for p in panels})
    target = stats.get("sec_per_manga_page", 22.0) * n_pages
    n_shots = max_shots or max(8, round(target / stats.get("mean_shot_sec", 3.6)))
    user.append(f"### Now adapt manga chapter {chapter}" + (f" — \"{title}\"" if title else "") +
                f"\n{len(panels)} panels on {n_pages} pages. Target running time ≈ {target:.0f} s in about {n_shots} shots. "
                f"HARD LIMIT: at most {round(n_shots * 1.08)} shots including the title card "
                f"(average shot ≈ {target / n_shots:.1f} s).\n\n" + render_chapter(panels))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(user)}]


_JA_MORA = re.compile(r"[ぁ-んァ-ヶー一-龯]")


def est_speech_sec(text_ja: str) -> float:
    n = len(_JA_MORA.findall(text_ja or ""))
    return 0.25 + n / 7.5 if n else 0.0


def postprocess(chapter: int, shots: list[dict], panels: list[dict]) -> list[dict]:
    valid = {p["order"] for p in panels}
    out = []
    for i, s in enumerate(shots):
        s = dict(s)
        s["shot_id"] = f"c{chapter:03d}_s{i:03d}"
        s["chapter"] = chapter
        sp = []
        for x in s.get("source_panels") or []:
            m = re.search(r"\d+", str(x))
            if m and int(m.group()) in valid:
                sp.append(int(m.group()))
        s["source_panels"] = sp
        try:
            s["duration"] = float(re.search(r"[\d.]+", str(s.get("duration", 3))).group())
        except Exception:
            s["duration"] = 3.0
        s["characters"] = [str(c) for c in (s.get("characters") or []) if c]
        for k in ("shot_size", "camera", "action", "location", "time_of_day", "first_frame", "bgm_cue", "notes"):
            if s.get(k) is None or not isinstance(s.get(k), str):
                s[k] = "" if s.get(k) is None else str(s[k])
        if not isinstance(s.get("sfx"), list):
            s["sfx"] = []
        s["dialogue"] = [d for d in (s.get("dialogue") or []) if isinstance(d, dict) and (d.get("text") or d.get("text_ja"))]
        for d in s["dialogue"]:
            d["speaker"] = str(d.get("speaker") or "Classmates")
            d["text"] = str(d.get("text") or "")
            d["kind"] = str(d.get("kind") or "speech").lower()
            if d["kind"] not in ("speech", "thought", "narration", "notebook", "sfx_text"):
                d["kind"] = "speech"
            d.setdefault("emotion", "neutral")
            if d["emotion"] not in EMOTIONS:
                d["emotion"] = "neutral"
            d.setdefault("text_ja", "")
        spoken = sum(est_speech_sec(d["text_ja"]) for d in s["dialogue"] if d["kind"] in ("speech", "thought", "narration"))
        s["duration"] = float(min(8.0, max(float(s["duration"]), spoken + 0.3 if spoken else 0.8)))
        for k, v in (("location", ""), ("time_of_day", ""), ("first_frame", ""), ("bgm_cue", "none"), ("sfx", []), ("notes", "")):
            s.setdefault(k, v)
        out.append(s)
    return out


def direct_chapter(chapter: int, *, use_fewshot: bool = True, thinking: bool = True, max_shots: int | None = None) -> dict:
    panels = [p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] == chapter]
    if not panels:
        raise ValueError(f"chapter {chapter} is not parsed")
    panels.sort(key=lambda p: p["order"])
    chapters = {c["chapter"]: c for c in read_json(P("processed") / "manga" / "chapters.json", [])}
    title = (chapters.get(chapter) or {}).get("title", "")
    stats = load_stats()
    examples = fewshot_for(panels) if use_fewshot else []
    msgs = build_messages(chapter, panels, title, examples, stats, max_shots)
    llm = remote_or_local("director")
    try:
        res = llm.chat(msgs, schema=SHOT_SCHEMA, max_tokens=60000, thinking=thinking, effort="low", tag=f"director:c{chapter}")
    except BudgetExceeded:
        log("director: API budget reached -> local Qwen fallback")
        llm = LLM("local", role="director")
        res = llm.chat(msgs, schema=SHOT_SCHEMA, max_tokens=12000, tag=f"director:c{chapter}")
    shots = postprocess(chapter, res["shots"], panels)
    return {"chapter": chapter, "title": title, "endpoint": llm.endpoint, "n_panels": len(panels),
            "fewshot_chapters": [e["chapter"] for e in examples], "shots": shots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fewshot", action="store_true")
    a = ap.parse_args()
    res = direct_chapter(a.chapter, use_fewshot=not a.no_fewshot)
    write_json(a.out, res)
    d = sum(s["duration"] for s in res["shots"])
    log(f"chapter {a.chapter}: {len(res['shots'])} shots, {d:.0f}s, endpoint={res['endpoint']} -> {a.out}")


if __name__ == "__main__":
    main()
