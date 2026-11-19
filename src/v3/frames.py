"""v3 step 5a — FIRST FRAMES (D-043): Illustrious-XL draws the first frame of every shot that needs one; Gemini checks it.

  start = "panel"  panel crop -> layout (softened init + line-art ControlNet) + tag prompt
          "new"    tag prompt only (txt2img)
          "chain"  no frame here: the clip starts on the last frame of the previous clip (src/v3/render.py)
          "title"  typographic title card
The prompt is assembled in code from the shot plan and the character cards: who is in the frame (card tags), what they wear
(the school uniform at school, the scene's outfit elsewhere), the shot's own tags, the scene's background tags.
"""
from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, sha, write_json
from src.generate.prompts import ILL_FRAMING, ILL_NEG, ILL_QUALITY, ILL_SERIES, ILL_UNIFORM_F, ILL_UNIFORM_M, _count_tags, is_school
from src.generate.tags import _tags
from src.llm.client import encode_image, vision
from src.v3.cast import card, gender, is_known

_UNIFORM_WORD = re.compile(r"uniform|blazer|sailor|gakuran|neckerchief|serafuku|necktie|bowtie|bow\b|skirt|trousers|pants|shirt|jacket|tights|school", re.I)
_ADULT = re.compile(r"mother|father|teacher|adult|woman|man\b", re.I)
DENOISE = (0.85, 0.88, 0.92)


def person_tags(v: dict, shot: dict) -> str:
    cid = v["id"]
    c = card(cid)
    seg: list[str] = []
    if c:
        seg.append(str(c.get("full_name") or cid).lower())
        seg += _tags(c.get("danbooru_tags"))[:7]
    else:
        seg += _tags(v.get("look") or "")[:6]
    outfit = _tags((shot.get("outfits") or {}).get(cid))
    if shot.get("wardrobe") and outfit:                 # the costume pass fixed this outfit for the whole scene (src/v3/wardrobe.py): use it whole
        seg += outfit[:9]
        outfit = []
    student = not _ADULT.search(f"{cid} {(c or {}).get('class_or_occupation') or ''}") or "student" in str((c or {}).get("class_or_occupation") or "").lower()
    if is_school(shot.get("location", "")) and student:
        seg += [t for t in outfit if not _UNIFORM_WORD.search(t)][:3]          # accessories only; the uniform is added once for the frame
    else:
        seg += outfit[:6]
    out, seen = [], set()
    for t in seg:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


def frame_prompt(shot: dict, hint: str = "") -> str:
    vis = [v for v in shot.get("visible", []) if v["id"] != "Classmates"][:4]
    crowd = any(v["id"] == "Classmates" for v in shot.get("visible", []))
    parts = [ILL_QUALITY, ILL_SERIES]
    if not vis and not crowd:
        parts.append("no humans")
    else:
        cnt = _count_tags([gender(v["id"], str(v.get("pose", ""))) for v in vis])
        if cnt:
            parts.append(cnt)
        parts += [person_tags(v, shot) for v in vis]
        if is_school(shot.get("location", "")) and vis and not shot.get("wardrobe"):
            parts.append(ILL_UNIFORM_M if {gender(v["id"]) for v in vis} == {"m"} else ILL_UNIFORM_F)
        if crowd:
            parts.append("people in background, crowd" if vis else "multiple girls, multiple boys, crowd")
    fr = ILL_FRAMING.get(shot.get("framing", ""))
    if fr:
        parts.append(fr)
    parts.append(", ".join(_tags(shot.get("image_tags"))[:16]))
    parts.append(", ".join(_tags(shot.get("background_tags"))[:8]))
    parts.append("general")
    sent = str(shot.get("image_sentence") or "").strip().rstrip(".")
    if sent:
        parts.append(sent[:220])
    if hint:
        parts.append(hint.strip().rstrip(".")[:200])
    return ", ".join(p for p in parts if p)


_POS = {"left": 0, "center": 1, "centre": 1, "middle": 1, "right": 2}


def regions_for(shot: dict, sdir: Path) -> tuple[str, list[tuple[Path, str]]] | None:
    """(base prompt, [(mask file, person prompt)]) for frames with 2-3 named people placed left / center / right.

    The base prompt keeps what the frame shares (quality, head count, uniform, framing, action, background); each person's
    name, look and pose go into a vertical band of the frame, so features do not leak from one character to the next."""
    import numpy as np
    from PIL import Image, ImageFilter

    vis = [v for v in shot.get("visible", []) if v["id"] != "Classmates" and str(v.get("position", "")).lower() in _POS][:3]
    named = [v for v in shot.get("visible", []) if v["id"] != "Classmates"]
    if len(vis) != 2 or len(named) != 2:            # measured: bands help two-person frames a little, three-person frames not at all
        return None
    vis = sorted(vis, key=lambda v: _POS[str(v["position"]).lower()])          # stable: listing order breaks ties
    n = len(vis)
    school = is_school(shot.get("location", "")) and not shot.get("wardrobe")        # with a wardrobe every person carries their own outfit
    parts = [ILL_QUALITY, ILL_SERIES, _count_tags([gender(v["id"], str(v.get("pose", ""))) for v in vis])]
    if school:
        parts.append(ILL_UNIFORM_M if {gender(v["id"]) for v in vis} == {"m"} else ILL_UNIFORM_F)
    if len(named) != len(shot.get("visible", [])):
        parts.append("people in background")
    parts += [ILL_FRAMING.get(shot.get("framing", ""), ""), ", ".join(_tags(shot.get("image_tags"))[:16]),
              ", ".join(_tags(shot.get("background_tags"))[:8]), "general", str(shot.get("image_sentence") or "").strip().rstrip(".")[:220]]
    base = ", ".join(p for p in parts if p)
    W, H = 1344, 768
    out = []
    for i, v in enumerate(vis):
        x0, x1 = max(0.0, i / n - 0.06), min(1.0, (i + 1) / n + 0.06)
        m = np.zeros((H, W), np.uint8)
        m[:, int(x0 * W): int(x1 * W)] = 255
        mp = sdir / f"region_{i}.png"
        Image.fromarray(m).filter(ImageFilter.GaussianBlur(24)).convert("RGB").save(mp)
        g = gender(v["id"], str(v.get("pose", "")))
        who = person_tags(v, shot)
        uni = (ILL_UNIFORM_M if g == "m" else ILL_UNIFORM_F) if school else ""
        act = ", ".join(x for x in (str(v.get("pose") or "").strip(), str(v.get("expression") or "").strip()) if x)[:160]
        txt = ", ".join(p for p in ("anime screencap, anime coloring", {"f": "1girl", "m": "1boy"}.get(g, "1other"), who, uni, act) if p)
        out.append((mp, txt))
    return base, out


def _panels(chapter: int) -> dict[int, dict]:
    return {p["order"]: p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] == chapter}


def make_frame(comfy, run_dir: Path, shot: dict, panel: dict | None, *, attempt: int = 0, hint: str = "") -> Path | None:
    from src.generate.comfy import wf_illustrious
    from src.generate.keyframes import (KF_H, KF_W, compose_cover, lineart_control, mean_saturation, prep_panel, remove_balloons,
                                        soften_layout, title_card)

    sdir = run_dir / "shots" / shot["shot_id"]
    sdir.mkdir(parents=True, exist_ok=True)
    out = sdir / f"kf_a{attempt}.png"
    if shot["start"] == "chain":
        return None
    if shot["start"] == "title":
        title_card(shot["notes"].split(":", 1)[-1].strip() or "…", sdir / "kf.png")
        return sdir / "kf.png"
    prompt = frame_prompt(shot, hint)
    reg = regions_for(shot, sdir)
    regions = None
    if reg:
        prompt = reg[0] + (", " + hint.strip().rstrip(".")[:200] if hint else "")
        regions = [(comfy.put_input(mp, f"{run_dir.name}_{shot['shot_id']}_{mp.name}"), txt) for mp, txt in reg[1]]
    image = control = None
    cn_s, cn_e, denoise, layout = 0.0, 0.0, 1.0, "text"
    if shot["start"] == "panel" and panel is not None:
        prep_panel(panel["image"], sdir / "panel.png")
        src = compose_cover(panel, sdir / "panel_canvas.png", zoom=0.92 if attempt % 2 == 1 else 1.0)
        src, _ = remove_balloons(src, sdir / "panel_clean.png")
        cn_s, cn_e = max(0.25, 0.5 - 0.1 * attempt), 0.6
        control = comfy.put_input(lineart_control(src, sdir / "panel_lines.png"), f"{run_dir.name}_{shot['shot_id']}_lines.png")
        image = comfy.put_input(soften_layout(src, sdir / "panel_layout.png"), f"{run_dir.name}_{shot['shot_id']}_panel.png")
        denoise, layout = DENOISE[min(attempt, len(DENOISE) - 1)], "panel_i2i"
    seed = int(sha(shot["shot_id"] + "v3" + str(attempt), 8), 16) % (2**31)
    retries, sat = 0, 0.0
    while True:
        files = comfy.run(wf_illustrious(image, prompt, ILL_NEG, seed=seed + retries, denoise=denoise, out_size=(KF_W, KF_H),
                                         control_image=control, control_strength=cn_s, control_end=cn_e, regions=regions,
                                         prefix=f"{run_dir.name}/{shot['shot_id']}_kf_a{attempt}"), timeout=900)
        if not files:
            raise RuntimeError("frame workflow produced no image")
        sat = mean_saturation(files[0])
        if sat >= 28 or retries >= 2 or image is None:
            break
        retries += 1                                   # a busy black-and-white panel pulled the result back to monochrome
        denoise, cn_s = min(0.96, denoise + 0.06), cn_s * 0.7
        if "colorful" not in prompt:
            prompt = prompt.replace(", general", ", colorful, vibrant colors, general", 1)
    shutil.copyfile(files[0], out)
    shutil.copyfile(out, sdir / "kf.png")
    write_json(sdir / f"kf_a{attempt}.json", {"prompt": prompt, "negative": ILL_NEG, "seed": seed + retries, "attempt": attempt, "start": shot["start"],
                                               "panel": shot.get("panel"), "layout": layout, "denoise": denoise, "saturation": round(sat, 1),
                                               "regions": [t for _, t in (reg[1] if reg else [])],
                                               "controlnet": {"strength": cn_s, "end": cn_e} if control else None})
    return out


# ----------------------------------------------------------------------------- visual check (Gemini)
CHECK_SCHEMA = {"type": "object", "required": ["verdict", "people_drawn", "cast", "problems"]}
CHECK_SYS = """You check ONE frame drawn by an image model for an all-ages TV anime ("Komi Can't Communicate") against
what the frame was supposed to show. You get the intended content, reference pictures of the intended characters (when
they exist) and the frame (the LAST image).
Return ONE JSON object:
{"people_drawn": <how many people are clearly drawn>,
 "cast": [{"id": "<intended id>", "present": true|false, "looks_right": true|false, "note": "what differs (hair, glasses, outfit ...)"}],
 "extra_people": "<who is drawn that was not intended, or empty>",
 "problems": ["anatomy / duplicated person / leftover speech balloon or lettering / black-and-white or sketchy / wrong place ..."],
 "unsuitable": true|false,        // anything not fit for an all-ages show
 "score": 0-10,                    // 10 = a clean frame that shows exactly the intended people and action
 "verdict": "ok" | "redo",         // redo: a person is missing or clearly someone else, wrong head count, broken anatomy, leftovers, unsuitable
 "hint": "if redo: up to 8 Danbooru-style tags to add so the next attempt fixes it, else empty"}
Be strict about WHO is in the frame and how many; be lenient about background and small pose differences."""


def check_frame(run_dir: Path, shot: dict, attempt: int = 0) -> dict:
    sdir = run_dir / "shots" / shot["shot_id"]
    kf = sdir / f"kf_a{attempt}.png"
    if shot["start"] in ("chain", "title") or not kf.exists():
        return {"verdict": "ok", "score": 10, "skipped": True}
    vis = [v for v in shot.get("visible", []) if v["id"] != "Classmates"][:4]
    content: list[dict] = [{"type": "text", "text": "INTENDED FRAME: " + (shot.get("image_sentence") or "") +
                            f"\nFraming: {shot.get('framing')}. People intended: {len(vis)}" + (" plus a background crowd." if len(vis) != len(shot.get('visible', [])) else ".")}]
    from src.v3.staging import camera_line

    if camera_line(shot):
        content[0]["text"] += ("\nCamera, as intended: " + camera_line(shot) + " A person meant to be seen from behind or in the foreground of an "
                               "over-the-shoulder shot counts as present and right when hair and clothes match; do not ask for their face.")
    root = Path(CFG.paths.repo) / "data" / "processed" / "cards"
    for v in vis:
        c = card(v["id"])
        desc = f"{v['id']} — {(c or {}).get('look_anime') or 'no card: judge by the description'}; position {v.get('position', '?')}; {v.get('pose', '')}; {v.get('expression', '')}"
        content.append({"type": "text", "text": desc})
        if c and c.get("ref_sheet") and (root / c["ref_sheet"]).exists():
            content.append({"type": "image_url", "image_url": {"url": encode_image(root / c["ref_sheet"], 640, 85)}})
    content.append({"type": "text", "text": "THE FRAME TO CHECK:"})
    content.append({"type": "image_url", "image_url": {"url": encode_image(kf, 1024, 90)}})
    try:
        r = vision("frame_check").chat([{"role": "system", "content": CHECK_SYS}, {"role": "user", "content": content}],
                                       schema=CHECK_SCHEMA, max_tokens=2500, effort="low", media_resolution="medium", tag=f"framecheck:{shot['shot_id']}")
    except Exception as e:
        r = {"verdict": "ok", "score": 5, "people_drawn": None, "cast": [], "problems": [], "error": str(e)[:200]}
    r["attempt"] = attempt
    write_json(sdir / f"check_a{attempt}.json", r)
    return r


def fix_people(comfy, run_dir: Path, shot: dict, base_attempt: int, new_attempt: int, check: dict) -> Path | None:
    """Second pass for frames with 2-3 named people: every person the check found missing or off-model is redrawn alone,
    inside their side of the frame, from a single-character prompt (single-character frames pass the check ~9 times in 10)."""
    import numpy as np
    from PIL import Image, ImageFilter

    from src.generate.comfy import wf_illustrious_inpaint
    from src.generate.keyframes import KF_H, KF_W

    sdir = run_dir / "shots" / shot["shot_id"]
    vis = [v for v in shot.get("visible", []) if v["id"] != "Classmates"][:3]
    if len(vis) < 2 or not (sdir / f"kf_a{base_attempt}.png").exists():
        return None
    verdict = {str(c.get("id")): c for c in (check.get("cast") or []) if isinstance(c, dict)}
    order = sorted(range(len(vis)), key=lambda i: (_POS.get(str(vis[i].get("position", "")).lower(), 1), i))
    slot = {i: k for k, i in enumerate(order)}                 # left-to-right slot of each person
    n = len(vis)
    cur = sdir / f"kf_a{base_attempt}.png"
    school = is_school(shot.get("location", "")) and not shot.get("wardrobe")
    control = comfy.put_input(sdir / "panel_lines.png", f"{run_dir.name}_{shot['shot_id']}_lines.png") if (sdir / "panel_lines.png").exists() and shot["start"] == "panel" else None
    done = 0
    for i, v in enumerate(vis):
        c = verdict.get(v["id"]) or {}
        if c.get("present", True) and c.get("looks_right", True):
            continue
        k = slot[i]
        x0, x1 = max(0.0, k / n - 0.04), min(1.0, (k + 1) / n + 0.04)
        m = np.zeros((768, 1344), np.uint8)
        m[:, int(x0 * 1344): int(x1 * 1344)] = 255
        mp = sdir / f"fix_mask_{i}.png"
        Image.fromarray(m).filter(ImageFilter.GaussianBlur(18)).convert("RGB").save(mp)
        g = gender(v["id"], str(v.get("pose", "")))
        act = ", ".join(x for x in (str(v.get("pose") or "").strip(), str(v.get("expression") or "").strip()) if x)[:160]
        prompt = ", ".join(p for p in (ILL_QUALITY, ILL_SERIES, {"f": "1girl, solo", "m": "1boy, solo, male focus"}.get(g, "1other, solo"), person_tags(v, shot),
                                       (ILL_UNIFORM_M if g == "m" else ILL_UNIFORM_F) if school else "", ILL_FRAMING.get(shot.get("framing", ""), ""), act,
                                       ", ".join(_tags(shot.get("background_tags"))[:6]), "general") if p)
        seed = int(sha(shot["shot_id"] + f"fix{new_attempt}_{i}", 8), 16) % (2**31)
        files = comfy.run(wf_illustrious_inpaint(comfy.put_input(cur, f"{run_dir.name}_{shot['shot_id']}_fixsrc{i}.png"),
                                                 comfy.put_input(mp, f"{run_dir.name}_{shot['shot_id']}_fixmask{i}.png"), prompt, ILL_NEG, seed=seed,
                                                 denoise=0.86 if not c.get("present", True) else 0.78, out_size=(KF_W, KF_H),
                                                 control_image=control, control_strength=0.4, control_end=0.5,
                                                 prefix=f"{run_dir.name}/{shot['shot_id']}_fix{new_attempt}_{i}"), timeout=900)
        if not files:
            continue
        cur = sdir / f"fix_a{new_attempt}_{i}.png"
        shutil.copyfile(files[0], cur)
        done += 1
    if not done:
        return None
    out = sdir / f"kf_a{new_attempt}.png"
    shutil.copyfile(cur, out)
    write_json(sdir / f"kf_a{new_attempt}.json", {"method": "per-person redraw", "base_attempt": base_attempt, "people_redrawn": done})
    return out


def run_fix_people(run_dir: Path, shots: list[dict], only: set[str] | None = None) -> dict:
    """Per-person redraw for every multi-person frame whose best attempt failed the check; keeps the better of the two."""
    from src.common import servers
    from src.generate.comfy import Comfy

    state = read_json(run_dir / "frames_state.json") or {}
    todo = []
    for s in shots:
        st = state.get(s["shot_id"]) or {}
        if s["start"] in ("chain", "title") or st.get("final_verdict") != "redo" or (only is not None and s["shot_id"] not in only):
            continue
        if len([v for v in s.get("visible", []) if v["id"] != "Classmates"]) >= 2 and not st.get("people_fixed"):
            todo.append(s)
    if not todo:
        return state
    servers.start("comfy")
    comfy = Comfy()
    made = []
    for i, s in enumerate(todo):
        st = state[s["shot_id"]]
        base = st.get("final_attempt", 0)
        chk = read_json(run_dir / "shots" / s["shot_id"] / f"check_a{base}.json") or {}
        new = max(c["attempt"] for c in st["checks"]) + 1
        try:
            if fix_people(comfy, run_dir, s, base, new, chk):
                made.append((s, new))
        except Exception as e:
            log(f"per-person redraw {s['shot_id']} failed: {str(e)[:200]}")
        st["people_fixed"] = True
        if (i + 1) % 25 == 0:
            log(f"per-person redraw: {i + 1}/{len(todo)}")
    with ThreadPoolExecutor(8) as ex:
        checks = list(ex.map(lambda sn: check_frame(run_dir, sn[0], sn[1]), made))
    better = 0
    for (s, new), c in zip(made, checks):
        st = state[s["shot_id"]]
        st["checks"].append({k: c.get(k) for k in ("attempt", "verdict", "score", "people_drawn", "problems", "unsuitable", "extra_people")})
        old_score = st.get("final_score") or 0
        if not c.get("unsuitable") and (c.get("verdict") == "ok" or (c.get("score") or 0) > old_score):
            st["final_attempt"], st["final_verdict"], st["final_score"] = new, c.get("verdict"), c.get("score")
            shutil.copyfile(run_dir / "shots" / s["shot_id"] / f"kf_a{new}.png", run_dir / "shots" / s["shot_id"] / "kf.png")
            better += 1
    write_json(run_dir / "frames_state.json", state)
    log(f"per-person redraw: {len(made)} frames redrawn, {better} kept as better, "
        f"{sum(1 for (s, _), c in zip(made, checks) if c.get('verdict') == 'ok')} now accepted")
    return state


def run_frames(run_dir: Path, shots: list[dict], *, rounds: int = 1, only: set[str] | None = None) -> dict:
    """Draw every first frame, check each with Gemini, redraw the failed ones `rounds` times and keep the best attempt."""
    from src.common import servers
    from src.generate.comfy import Comfy

    todo = [s for s in shots if s["start"] != "chain" and (only is None or s["shot_id"] in only)]
    state = read_json(run_dir / "frames_state.json") or {}
    panels = {ch: _panels(ch) for ch in {s["chapter"] for s in todo}}
    pending = [s for s in todo if s["shot_id"] not in state or not (run_dir / "shots" / s["shot_id"] / "kf.png").exists()]
    for rnd in range(rounds + 1):
        if not pending:
            break
        servers.start("comfy")
        comfy = Comfy()
        for i, s in enumerate(pending):
            st = state.setdefault(s["shot_id"], {"attempt": -1, "checks": []})
            st["attempt"] += 1
            try:
                make_frame(comfy, run_dir, s, panels[s["chapter"]].get(s.get("panel")), attempt=st["attempt"], hint=st.get("hint", ""))
            except Exception as e:
                log(f"frame {s['shot_id']} attempt {st['attempt']} failed: {str(e)[:200]}")
            if (i + 1) % 50 == 0:
                log(f"frames round {rnd}: {i + 1}/{len(pending)}")
        with ThreadPoolExecutor(8) as ex:
            checks = list(ex.map(lambda s: check_frame(run_dir, s, state[s["shot_id"]]["attempt"]), pending))
        nxt = []
        for s, c in zip(pending, checks):
            st = state[s["shot_id"]]
            st["checks"].append({k: c.get(k) for k in ("attempt", "verdict", "score", "people_drawn", "problems", "unsuitable", "extra_people")})
            st["hint"] = str(c.get("hint") or "")[:200]
            if c.get("verdict") == "redo" and rnd < rounds and s["start"] != "title":
                nxt.append(s)
        write_json(run_dir / "frames_state.json", state)
        log(f"frames round {rnd}: {len(pending) - len(nxt)}/{len(pending)} accepted")
        pending = nxt
    # keep the best-scoring attempt of every shot; a frame flagged unsuitable in its best attempt is listed for review
    for s in todo:
        st = state.get(s["shot_id"]) or {}
        cs = [c for c in st.get("checks", []) if not c.get("unsuitable")] or st.get("checks", [])
        if not cs or s["start"] == "title":
            continue
        best = max(cs, key=lambda c: (c.get("verdict") == "ok", c.get("score") or 0))
        st["final_attempt"], st["final_verdict"], st["final_score"] = best["attempt"], best.get("verdict"), best.get("score")
        src = run_dir / "shots" / s["shot_id"] / f"kf_a{best['attempt']}.png"
        if src.exists():
            shutil.copyfile(src, run_dir / "shots" / s["shot_id"] / "kf.png")
    write_json(run_dir / "frames_state.json", state)
    return state
