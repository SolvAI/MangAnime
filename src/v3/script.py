"""v3 steps 2-3 — WRITING (D-043): the chapter readings become a movie script (DeepSeek, text only).

  bible   one call for the whole episode: running order, continuity facts, cast notes        -> script/bible.json
  draft   one call per chapter: scenes with geography, filmable action, English dialogue     -> script/c<ch>_draft.json
  revise  one call per chapter: spatial sense, pacing, cuts / expansions, continuity         -> script/c<ch>.json

  python -m src.v3.script --run ep22_v3 --chapters 114,116,117,118 [--steps bible,draft,revise]
The manga is the source of story, jokes and lines; the script is free in time: a panel may last half a second or carry a
minute of screen time, several panels may fuse into one continuous action, and connective action may be added.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import CFG, P
from src.common.io import log, read_json, write_json
from src.llm.client import writer

READING = P("processed") / "v3" / "reading"

SHOW = """SHOW: "Komi Can't Communicate" — TV anime, school comedy, all ages. Itan Private High School, class 1-1.
- Komi Shouko is admired by everyone and almost unable to speak. Aloud she only stammers or says one or two words; what she
  wants to say she writes in a notebook that is shown on screen (kind "notebook": seen, not spoken). Her inner voice may be
  heard as a thought.
- Tadano Hitohito is her first friend, an ordinary boy who reads people well; much of the comedy is his inner commentary.
- A calm, slightly amused female Narrator explains situations ("People with a communication disorder ...").
- Each manga chapter is one segment ("Communication N") of an episode and opens on a short title card.
- Comedy beats use simplified chibi faces, bold reactions and quick cuts; warm beats are slow and quiet.
- School uniform, all year, never anything else at school: girls wear a dark navy blazer over a white shirt with a red
  striped bow and a red plaid pleated skirt (Komi with black tights); boys wear the same navy blazer with a red striped
  necktie and grey trousers. It is NOT a sailor uniform and NOT a gakuran. In winter people add coats and scarves outdoors.
- Use each person's card id as their name in every field ("Komi", "Tadano", "Onemine" ...), exactly as given."""

BIBLE_SYS = """You are the showrunner of the TV anime "Komi Can't Communicate". You plan ONE episode made of several manga
chapters. You get, per chapter, a script supervisor's reading (summary, scenes, who is present) and the character cards.

{show}

Plan the episode so the segments feel like one episode, not four unrelated shorts. Return ONE JSON object:
{{"episode_title": "...", "logline": "one sentence",
  "running_order": [{{"chapter": <n>, "segment_title": "...", "target_seconds": <int>,
                      "what_it_is_about": "one sentence", "tone": "few words", "opens_on": "how the segment starts on screen",
                      "hands_over_with": "how it ends / leads into the next segment"}}],
  "continuity": {{"season_and_weather": "...", "time_span": "e.g. one school day, the week before ...",
                  "school_outfit": "exact uniform everybody wears at school in this episode",
                  "facts_that_carry_over": ["things established in one segment that later segments must respect"]}},
  "locations": [{{"id": "short_snake_case", "name": "...", "look": "what the place looks like, layout, light — 1-2 sentences"}}],
  "cast_notes": [{{"id": "...", "role_in_episode": "one sentence", "voice_direction": "how they should sound, 3-8 words"}}],
  "running_gags": ["..."]}}
Rules: target_seconds — a TV episode is about 22 minutes without opening and ending; give each chapter the time its story
needs (a dense chapter may get 7 minutes, a small gag 2). Reuse location ids across chapters when it is the same place.
Do not invent plot. Skip covers, tables of contents, previews and other front matter."""

DRAFT_SYS = """You are the head writer adapting one manga chapter into a segment of the TV anime "Komi Can't Communicate".
You write a MOVIE SCRIPT in English: what the camera sees and what is heard, in the order the audience gets it.

{show}

You get: the episode plan (running order, continuity, locations), the character cards of everyone in the chapter, and the
script supervisor's reading of the chapter (scenes, then every panel: who is visible, what happens, every printed text).

HOW TO ADAPT
1. The manga gives the story, the jokes and the lines. Keep them. But an episode is not a slideshow of panels:
   - a panel can be half a second (a flash reaction) or the seed of a whole minute (an awkward silence that plays out);
   - several panels usually fuse into ONE continuous action in one place;
   - the manga skips connective tissue the screen needs: someone walking over, sitting down, a look held one beat too long,
     the room reacting. Write that in. Cut what only works on paper (arrows, repeated captions, labels explaining the joke).
2. Write filmable action in the present tense: who is where, who moves, what changes. No camera jargon and no shot list yet.
3. GEOGRAPHY: every scene states the place, where each person is at the start (seat, doorway, by the window ...) and keeps
   that consistent. If someone needs to be elsewhere, move them on screen.
4. DIALOGUE in natural spoken English for a dub. Keep the meaning and the punchlines of the printed lines; tidy clumsy
   translation; trim repetition. kind is one of: speech, thought (voice-over), narration (the Narrator), notebook (written
   and shown, not spoken), sign (on-screen text only). Use the speakers from the reading; where the reading is unsure or
   implausible (e.g. Komi speaking a fluent sentence aloud), fix it with the story and the character cards.
   Each line gets "delivery": a short direction for the voice actor (emotion, volume, pace).
5. Sound effects lettering becomes sound ("sfx") or action, never spoken words.
6. Every beat lists the reading's panel numbers it draws on in "from_panels" ([] for added connective action).
7. Give each beat "seconds": the screen time it needs. Speech runs at about 2.5 words per second plus pauses. Let quiet
   moments breathe and gags snap.
8. Stay inside what the chapter shows. Do not add plot, characters or jokes of your own; added material is connective only.
9. Start the segment with a title-card beat (type "title_card", text = the segment title, 3 seconds).

RETURN ONE JSON OBJECT
{{"chapter": <n>, "segment_title": "...",
  "scenes": [{{"scene_id": "<chapter>.<k>", "location_id": "<id from the plan or a new snake_case id>",
              "slug": "INT./EXT. PLACE — TIME", "set": "what the place looks like and where everyone is when the scene starts",
              "cast": [{{"id": "...", "wears": "outfit in this scene", "state": "mood / what they want at the start"}}],
              "beats": [{{"beat": <k>, "type": "title_card|action|dialogue|narration|insert|transition",
                         "from_panels": [<n>, ...],
                         "action": "what we see",
                         "lines": [{{"speaker": "...", "kind": "...", "text": "...", "delivery": "..."}}],
                         "sfx": ["..."], "seconds": <number>}}]}}],
  "adaptation_notes": ["what you merged, expanded, cut or moved, and why — one line each"]}}"""

REVISE_SYS = """You are the series director of the TV anime "Komi Can't Communicate" doing the script pass before
storyboarding. You get a segment's draft script, the script supervisor's reading of the manga chapter, and the episode plan.
Return the REVISED script in the same JSON format, plus a change log.

{show}

WHAT TO FIX
1. SPACE. Read each scene as if you had to stage it: is it clear where everyone stands or sits, who faces whom, how they
   got there? Fix jumps (someone answers from a place they never went to; a prop appears from nowhere). Add the missing
   movement, or re-block the scene. Put the staging in "set" and in the beat actions.
2. SCREEN LOGIC. A manga cuts between tiny moments; a film plays actions through. Merge beats that are one continuous action
   in one place; split a beat that hides two different actions. Each beat should be one playable piece of action.
3. TIME. Check every "seconds". Long speeches need reaction beats; a punchline needs a beat of silence before or after;
   nothing but talk for 20 seconds is too long for this show. Trim lines that repeat information, expand moments the manga
   compresses but that carry feeling. The segment should land near the target length given in the plan (±25 %).
4. WHO SPEAKS. Every spoken line must come from someone present in the scene (or be a thought / narration). Komi does not
   speak fluent sentences aloud. A character cannot answer themselves. Fix the delivery notes so each line is playable.
5. CONTINUITY with the plan: outfits, season, time of day, facts that carry over, location ids.
6. Keep the story, jokes and lines of the manga. Do not add plot or new jokes.
Mark each beat with "continues": true when it carries on in the same place and moment as the previous beat without a
visible jump (the picture could run on), false when there is a cut to something else.

RETURN ONE JSON OBJECT: the full revised script (same keys as the draft, every beat with "continues"), plus
"changes": ["one line per change: what and why"], "runtime_seconds": <sum of beat seconds>."""

TIGHTEN_SYS = """You are the script editor of the TV anime "Komi Can't Communicate" doing the last pass on a segment before it is
storyboarded and recorded. The draft was written from the manga chapter and has grown: it rephrases lines the manga already
had right, adds explanations, and runs long. The real series, measured on its earlier episodes, uses about {ratio:.0%} of the
words printed in the manga chapter and tells a chapter in {sec_lo}-{sec_hi} seconds for a chapter of this size.

{show}

You get the script (JSON) and the manga chapter's printed texts per panel (the source). Return the script in the SAME JSON
format, edited as follows.

1. WORD BUDGET: all spoken lines together (kinds speech, thought, narration) come to between {words_lo} and {words} words. The
   script has {have} now. Count before you answer; do not cut below the lower figure.
2. KEEP THE MANGA'S LINES, SAY THEM SHORTER. The real series keeps about 85 % of the manga's printed lines and gets shorter by
   trimming inside the lines, not by dropping them. So: every printed text of the source (speech, thought, narration box)
   keeps a line in the script, said by the same person, unless it only repeats the text before it - at most one source
   text in seven may go. Start from the source's wording and tighten it the way a dub script does: drop filler, stammer
   repeats, doubled adjectives and trailing explanations; join two texts that form one sentence. Do not swap the line for
   a paraphrase in other words and do not add to it.
   The draft has also LOST printed lines. Go through the source panel by panel: a printed speech, thought or narration text
   that has no line in the script is put back, said by its speaker, in the beat that draws on that panel ("from_panels"),
   unless it only repeats its neighbour or is a label / sound effect.
3. CUT WHOLE LINES only among these, in this order: lines of the script that have no printed source (added narration,
   scene-setting, extra reactions); a thought that repeats the line before; a second or third reaction to the same thing.
   Keep every joke. Never give a line to someone else and never invent a new one.
4. A beat that loses all its lines stays if its action is needed (a look, a silence, a move) and is dropped if it was only
   there to carry the cut lines. Renumber "beat" inside each scene from 1. Keep "from_panels", "continues", "type", the
   scene ids, places, "set" and "cast" as they are.
5. TIME: set "seconds" of every beat again. Speech runs at 2.6 words per second; add at most half a second of air per line;
   a silent action beat lasts 1.5-3 seconds, a held emotional moment up to 5. The segment should come to {sec_lo}-{sec_hi}
   seconds.
6. Delivery notes stay with their lines; shorten them to a few words.

RETURN ONE JSON OBJECT: the edited script (same keys), plus "cut": ["one line per thing removed or restored to the manga's
wording"], "dialogue_words": <your count>, "runtime_seconds": <sum of beat seconds>."""

BIBLE_SCHEMA = {"type": "object", "required": ["running_order", "continuity", "locations"]}
SCRIPT_SCHEMA = {"type": "object", "required": ["scenes"], "properties": {"scenes": {"type": "array", "minItems": 1, "items": {
    "type": "object", "required": ["scene_id", "beats"], "properties": {"beats": {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": ["action", "seconds"]}}}}}}}


def cards_text(ids: set[str]) -> str:
    cards = {c["id"]: c for c in (read_json(P("processed") / "cards" / "cards.json") or [])}
    rows = []
    for i in sorted(ids):
        c = cards.get(i)
        if c:
            rows.append(f"- {c['id']} ({c.get('full_name', '')}): LOOKS {c.get('look_anime') or c.get('look_manga') or 'n/a'} "
                        f"PERSONALITY {c.get('personality') or 'n/a'} SPEECH {c.get('speech_style') or 'n/a'}")
    if not rows:                       # cards not built yet: the hand-written roster
        for c in CFG.characters.roster:
            if c["name"] in ids:
                rows.append(f"- {c['name']} ({c['full']}): LOOKS {c['look']} VOICE {c['voice']}")
    return "\n".join(rows)


def reading(chapter: int) -> dict:
    r = read_json(READING / f"c{chapter:03d}.json")
    if not r:
        raise FileNotFoundError(f"chapter {chapter} has not been read yet (python -m src.v3.read --chapters {chapter})")
    return r


def story_scenes(r: dict) -> list[dict]:
    """Scenes that are story (covers, contents pages and previews are dropped by their kind or by having no dialogue panels)."""
    return [s for s in r.get("scenes", []) if str(s.get("kind", "story")).lower() in ("story", "")]


def compact_reading(r: dict) -> dict:
    keep = {s["scene"] for s in story_scenes(r)}
    return {"title": r.get("title"), "summary": r.get("summary"),
            "characters": [{k: c.get(k) for k in ("id", "known", "seen_as")} for c in r.get("characters", [])],
            "scenes": [{k: s.get(k) for k in ("scene", "location", "time_of_day", "present", "what_happens", "mood", "first_panel", "last_panel")}
                       for s in r.get("scenes", []) if s["scene"] in keep],
            "panels": [{"n": p["n"], "scene": p.get("scene"), "framing": p.get("framing"),
                        "visible": [f"{v.get('id')} ({v.get('where', '')}): {v.get('doing', '')}" for v in p.get("visible", [])],
                        "action": p.get("action"),
                        "lines": [{k: l.get(k) for k in ("speaker", "kind", "text", "to", "delivery")} for l in p.get("lines", [])]}
                       for p in r.get("panels", []) if p.get("scene") in keep]}


def chapter_ids(r: dict) -> set[str]:
    return {c["id"] for c in r.get("characters", []) if c.get("id")} | {"Narrator"}


class Script:
    def __init__(self, run: str):
        self.dir = P("runs") / run / "script"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ episode plan
    def bible(self, chapters: list[int], redo: bool = False) -> dict:
        out = self.dir / "bible.json"
        if out.exists() and not redo:
            return read_json(out)
        rs = [reading(c) for c in chapters]
        ids = set().union(*[chapter_ids(r) for r in rs])
        user = ["CHARACTER CARDS\n" + cards_text(ids)]
        for c, r in zip(chapters, rs):
            n_pages = len({(p["volume"], p["page"]) for p in r["panels"]})
            user.append(f"CHAPTER {c} — \"{r.get('title', '')}\" ({n_pages} pages, {len(r['panels'])} panels, "
                        f"{sum(len(p['lines']) for p in r['panels'])} texts)\n"
                        + json.dumps({"summary": r.get("summary"), "scenes": compact_reading(r)["scenes"],
                                      "characters": [c2.get("id") for c2 in r.get("characters", [])]}, ensure_ascii=False))
        user.append(f"Plan the episode from chapters {', '.join(map(str, chapters))} in this order.")
        res = writer("script").chat([{"role": "system", "content": BIBLE_SYS.format(show=SHOW)}, {"role": "user", "content": "\n\n".join(user)}],
                                    schema=BIBLE_SCHEMA, max_tokens=12000, thinking=True, effort="medium", tag="script:bible")
        res["chapters"] = chapters
        write_json(out, res)
        log("episode plan: " + ", ".join(f"ch {x.get('chapter')} {x.get('target_seconds')} s" for x in res.get("running_order", [])))
        return res

    # ------------------------------------------------------------------ draft
    def draft(self, chapter: int, redo: bool = False) -> dict:
        out = self.dir / f"c{chapter:03d}_draft.json"
        if out.exists() and not redo:
            return read_json(out)
        r, bible = reading(chapter), read_json(self.dir / "bible.json") or {}
        plan = next((x for x in bible.get("running_order", []) if x.get("chapter") == chapter), {})
        user = ("EPISODE PLAN\n" + json.dumps({k: bible.get(k) for k in ("episode_title", "running_order", "continuity", "locations", "cast_notes", "running_gags")}, ensure_ascii=False)
                + f"\n\nTHIS SEGMENT: chapter {chapter}, target about {plan.get('target_seconds', '?')} seconds."
                + "\n\nCHARACTER CARDS\n" + cards_text(chapter_ids(r))
                + "\n\nREADING OF THE CHAPTER\n" + json.dumps(compact_reading(r), ensure_ascii=False))
        res = writer("script").chat([{"role": "system", "content": DRAFT_SYS.format(show=SHOW)}, {"role": "user", "content": user}],
                                    schema=SCRIPT_SCHEMA, max_tokens=60000, thinking=True, effort="medium", tag=f"script:draft:c{chapter}")
        res["chapter"] = chapter
        write_json(out, res)
        log(f"draft chapter {chapter}: {len(res['scenes'])} scenes, {sum(len(s['beats']) for s in res['scenes'])} beats, {runtime(res):.0f} s")
        return res

    # ------------------------------------------------------------------ revision
    def revise(self, chapter: int, redo: bool = False) -> dict:
        out = self.dir / f"c{chapter:03d}.json"
        if out.exists() and not redo:
            return read_json(out)
        r, bible, draft = reading(chapter), read_json(self.dir / "bible.json") or {}, self.draft(chapter)
        plan = next((x for x in bible.get("running_order", []) if x.get("chapter") == chapter), {})
        user = ("EPISODE PLAN\n" + json.dumps({k: bible.get(k) for k in ("running_order", "continuity", "locations", "cast_notes")}, ensure_ascii=False)
                + f"\n\nTHIS SEGMENT: chapter {chapter}, target about {plan.get('target_seconds', '?')} seconds; the draft runs {runtime(draft):.0f} seconds."
                + "\n\nCHARACTER CARDS\n" + cards_text(chapter_ids(r))
                + "\n\nREADING OF THE CHAPTER (the source)\n" + json.dumps(compact_reading(r), ensure_ascii=False)
                + "\n\nDRAFT SCRIPT\n" + json.dumps({k: draft.get(k) for k in ("segment_title", "scenes", "adaptation_notes")}, ensure_ascii=False))
        res = writer("script").chat([{"role": "system", "content": REVISE_SYS.format(show=SHOW)}, {"role": "user", "content": user}],
                                    schema=SCRIPT_SCHEMA, max_tokens=60000, thinking=True, effort="medium", tag=f"script:revise:c{chapter}")
        res["chapter"] = chapter
        res["runtime_seconds"] = round(runtime(res), 1)
        write_json(out, res)
        log(f"revised chapter {chapter}: {len(res['scenes'])} scenes, {sum(len(s['beats']) for s in res['scenes'])} beats, {runtime(res):.0f} s, "
            f"{len(res.get('changes', []))} changes")
        return res


    # ------------------------------------------------------------------ last pass: the anime's economy (D-056)
    def tighten(self, chapter: int, redo: bool = False) -> dict:
        """Edits the revised script down to the word and time budget the real series shows on its training episodes
        (src/v3/pace.py) and back to the manga's wording. The revised script is kept as c<ch>_revised.json."""
        from src.v3.pace import adaptation_stats, budget

        out, keep = self.dir / f"c{chapter:03d}.json", self.dir / f"c{chapter:03d}_revised.json"
        if keep.exists() and out.exists() and not redo:
            return read_json(out)
        rev = read_json(keep) if keep.exists() else self.revise(chapter)
        if not keep.exists():
            write_json(keep, rev)
        st, b, r = adaptation_stats(), budget(chapter), reading(chapter)
        hi = int(round(b["manga_words"] * st["seconds_per_manga_word"]["p75"]))
        source = [{"panel": p["n"], "texts": [{"speaker": l.get("speaker"), "kind": l.get("kind"), "text": l.get("text")} for l in p.get("lines", [])]}
                  for p in r["panels"] if p.get("lines")]
        user = ("SOURCE: printed texts of the manga chapter, per panel\n" + json.dumps(source, ensure_ascii=False)
                + "\n\nSCRIPT TO EDIT\n" + json.dumps({k: rev.get(k) for k in ("segment_title", "scenes")}, ensure_ascii=False))
        res = writer("script").chat([{"role": "system", "content": TIGHTEN_SYS.format(show=SHOW, ratio=st["anime_words_per_manga_word"]["median"], words=b["dialogue_words"],
                                                                                  words_lo=int(0.9 * b["dialogue_words"]),
                                                                                  have=dialogue_words(rev), sec_lo=b["seconds"], sec_hi=hi)},
                                     {"role": "user", "content": user}],
                                    schema=SCRIPT_SCHEMA, max_tokens=60000, thinking=True, effort="medium", tag=f"script:tighten:c{chapter}")
        res["chapter"] = chapter
        res["segment_title"] = res.get("segment_title") or rev.get("segment_title")
        res["runtime_seconds"] = round(runtime(res), 1)
        res["dialogue_words"] = dialogue_words(res)
        res["budget"] = {**b, "seconds_hi": hi}
        write_json(out, res)
        log(f"tightened chapter {chapter}: words {dialogue_words(rev)} -> {res['dialogue_words']} (budget {b['dialogue_words']}), "
            f"{runtime(rev):.0f} s -> {runtime(res):.0f} s (budget {b['seconds']}-{hi}), beats {sum(len(s['beats']) for s in rev['scenes'])} -> {sum(len(s['beats']) for s in res['scenes'])}")
        return res


def dialogue_words(script: dict) -> int:
    return sum(len(str(l.get("text") or "").split()) for s in script.get("scenes", []) for b in s.get("beats", []) for l in (b.get("lines") or [])
               if l.get("kind") in ("speech", "thought", "narration"))


def runtime(script: dict) -> float:
    t = 0.0
    for s in script.get("scenes", []):
        for b in s.get("beats", []):
            try:
                t += float(b.get("seconds") or 0)
            except Exception:
                pass
    return t


def to_text(script: dict) -> str:
    """Screenplay-style text for reading and for the comparison sheets."""
    out = [f"# {script.get('segment_title', '')} (chapter {script.get('chapter')}, {runtime(script):.0f} s)"]
    for s in script.get("scenes", []):
        out.append(f"\n## {s.get('scene_id')}  {s.get('slug', '')}")
        if s.get("set"):
            out.append(f"_{s['set']}_")
        for b in s.get("beats", []):
            src = ",".join(map(str, b.get("from_panels") or [])) or "added"
            out.append(f"\n[{b.get('beat')}] ({b.get('seconds')} s, panels {src}{', continues' if b.get('continues') else ''}) {b.get('action', '')}")
            for l in b.get("lines") or []:
                out.append(f"    {str(l.get('speaker', '')).upper()} [{l.get('kind')}; {l.get('delivery', '')}]: {l.get('text', '')}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--steps", default="bible,draft,revise")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    chs = [int(c) for c in a.chapters.split(",") if c]
    sc = Script(a.run)
    steps = a.steps.split(",")
    if "bible" in steps:
        sc.bible(chs, a.redo)
    for step in ("draft", "revise"):
        if step in steps:
            with ThreadPoolExecutor(4) as ex:
                res = list(ex.map(lambda c: getattr(sc, step)(c, a.redo), chs))
            if step == "revise":
                for c, r in zip(chs, res):
                    (sc.dir / f"c{c:03d}.md").write_text(to_text(r))


if __name__ == "__main__":
    main()
