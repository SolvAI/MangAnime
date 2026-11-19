"""VLM captioning with the local Qwen 3.8 27B (shots, clips, generated frames). One prompt family, JSON output.

Used for: shots.jsonl captions (§5.2), Wan LoRA training captions (§6), QA visual evidence (§8).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import CFG
from src.llm.client import LLM

SHOT_SIZES = ["extreme close-up", "close-up", "medium close-up", "medium", "medium wide", "wide", "extreme wide", "insert"]
CAMERAS = ["static", "pan left", "pan right", "tilt up", "tilt down", "zoom in", "zoom out", "tracking", "handheld", "other"]

CAPTION_SCHEMA = {
    "type": "object",
    "required": ["characters", "shot_size", "camera", "action", "caption"],
    "properties": {
        "characters": {"type": "array", "items": {"type": "string"}},
        "shot_size": {"type": "string"},
        "camera": {"type": "string"},
        "setting": {"type": "string"},
        "action": {"type": "string"},
        "caption": {"type": "string"},
        "on_screen_text": {"type": "boolean"},
        "is_title_card": {"type": "boolean"},
        "is_chibi": {"type": "boolean"},
    },
}


def roster_block(max_roles: tuple[str, ...] = ("main", "recurring", "family", "minor")) -> str:
    return "\n".join(f"- {c['name']}: {c['look']}" for c in CFG.characters.roster if c["role"] in max_roles)


SYSTEM = (
    "You annotate shots from the TV anime \"Komi Can't Communicate\" for a video dataset. "
    "Be literal and concise. Name characters ONLY from this roster when their look matches; otherwise use "
    "\"unknown girl\", \"unknown boy\", \"crowd\". Never guess a name from dialogue alone.\n"
    "Roster:\n{roster}\n"
    "Return ONE JSON object with keys: characters (list of names), shot_size (one of: {sizes}), "
    "camera (one of: {cameras}), setting (few words), action (one sentence: what moves or changes between the frames; "
    "say 'almost static, only mouth and eyes move' when that is the case), caption (1-2 sentences usable as a "
    "video-generation prompt: who is visible, what they look like, what they do, how the camera moves; "
    "no subtitles, no episode trivia), on_screen_text (true if large typeset text/signs dominate the image), "
    "is_title_card (true for title cards, credits, eyecatches, text-only screens), "
    "is_chibi (true if characters are drawn in the simplified comedic style)."
)


def caption_shot(llm: LLM, frames: list[str | Path], subtitle: str = "", extra: str = "", max_side: int = 640) -> dict:
    sys = SYSTEM.format(roster=roster_block(), sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERAS))
    n = len(frames)
    user = (f"The {n} images are frames of ONE shot in time order (first to last)." if n > 1 else "The image is one frame of a shot.")
    if subtitle:
        user += f"\nDialogue heard during the shot (for context only): \"{subtitle[:300]}\""
    if extra:
        user += "\n" + extra
    res = llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}], schema=CAPTION_SCHEMA,
                   images=list(frames), max_tokens=400, temperature=0.1, image_max_side=max_side, tag="caption_shot")
    if res.get("shot_size") not in SHOT_SIZES:
        res["shot_size"] = "medium"
    if res.get("camera") not in CAMERAS:
        res["camera"] = "other"
    return res


def caption_many(items: list[dict], workers: int | None = None) -> list[dict | None]:
    """items: [{"frames": [...], "subtitle": "...", "extra": "..."}] -> captions (None on failure). Parallel over server slots."""
    llm = LLM("local", role="caption")
    workers = workers or int(CFG.llm.local.parallel)

    def one(it):
        try:
            return caption_shot(llm, it["frames"], it.get("subtitle", ""), it.get("extra", ""))
        except Exception as e:  # keep the batch alive
            return {"error": str(e)[:200]}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, items))
