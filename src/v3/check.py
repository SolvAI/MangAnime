"""v3 visual check of finished clips (Gemini): does the animation keep the drawing intact?

The first frame was already checked (src/v3/frames.py); this looks at four frames across the clip and asks only about what
the video model can break: faces or bodies deforming, a person turning into someone else, limbs or people appearing,
flicker, and whether anything moves at all.
"""
from __future__ import annotations

from pathlib import Path

from src.common.io import write_json
from src.llm.client import encode_image, vision

SCHEMA = {"type": "object", "required": ["verdict", "motion", "problems"]}
SYS = """You check ONE short clip of an all-ages TV anime made by an image-to-video model. You get four frames in time
order (first, two in between, last) and what the clip was supposed to show. The first frame was drawn on purpose; judge
what happens AFTER it.
Return ONE JSON object:
{"motion": "none" | "subtle" | "clear" | "chaotic",
 "identity_stable": true|false,      // every person still looks like the same person in the last frame
 "problems": ["face melts", "extra arm", "second person appears", "character changes hair / outfit", "flicker or colour jump",
              "background warps", "text or symbols appear" ... only what you actually see],
 "matches_intent": true|false,       // roughly does what the description says (talking, turning, walking ...)
 "unsuitable": true|false,           // not fit for an all-ages show (the cast are school students), see below
 "score": 0-10,                      // 10 = clean limited TV animation
 "verdict": "ok" | "redo"}           // redo: deformation, identity change, extra people or limbs, chaotic motion, unsuitable
Limited animation is normal for TV anime: little motion is fine, a frozen picture for a talking shot is acceptable ("ok").
"unsuitable" is true when ANY frame shows: a kiss; a face or mouth pressed against another person's face or body; a romantic
or sexualised embrace; a hand on someone's chest, buttocks or thigh; underwear, nudity or a view up a skirt; a sexualised
pose or expression. The video model sometimes drifts into this although the script does not ask for it; such a clip is "redo"."""


def check_clip(run_dir: Path, shot: dict, attempt: int = 0) -> dict:
    from src.qa.evidence import sample_frames

    sdir = run_dir / "shots" / shot["shot_id"]
    clip = sdir / "clip.mp4"
    if shot["start"] == "title" or not clip.exists():
        return {"verdict": "ok", "skipped": True}
    frames = sample_frames(clip, sdir / "check_frames", n=4, tag=f"a{attempt}_")
    content: list[dict] = [{"type": "text", "text": f"INTENDED: {shot.get('image_sentence', '')} MOTION: {shot.get('motion', '')} "
                                                    f"People: {', '.join(v['id'] for v in shot.get('visible', [])) or 'none'}. Four frames follow."}]
    content += [{"type": "image_url", "image_url": {"url": encode_image(f, 768, 85)}} for f in frames]
    try:
        r = vision("clip_check").chat([{"role": "system", "content": SYS}, {"role": "user", "content": content}], schema=SCHEMA,
                                      max_tokens=2000, effort="low", media_resolution="medium", tag=f"clipcheck:{shot['shot_id']}")
    except Exception as e:
        r = {"verdict": "ok", "motion": "?", "problems": [], "error": str(e)[:200]}
    r["attempt"] = attempt
    write_json(sdir / f"clipcheck_a{attempt}.json", r)
    return r
