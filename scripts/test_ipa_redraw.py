"""Test: does a reference picture of the character help the per-person redraw of rejected multi-person frames? (D-051)

  python scripts/test_ipa_redraw.py --run ep22_v3 --n 24 build     # jobs for two arms: with the card picture / without (control)
  scripts/gpu_run.sh 12 venvs/sdxl/bin/python src/generate/ipa_redraw.py --jobs <jobs file>      (printed by build)
  python scripts/test_ipa_redraw.py --run ep22_v3 check            # Gemini frame check of both arms + a comparison sheet
Nothing of the run is changed: results go to shots/<id>/kf_a20.png (reference) and kf_a21.png (control) and eval/ipa_test.*.
The frames are the ones whose best attempt failed the check with 2-3 named people; only the people the check flagged are
redrawn, each inside their side of the frame, from a single-character prompt - exactly what frames.fix_people does.
"""
import argparse
import random
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json, sha, write_json, write_jsonl  # noqa: E402
from src.generate.prompts import ILL_FRAMING, ILL_NEG, ILL_QUALITY, ILL_SERIES, ILL_UNIFORM_F, ILL_UNIFORM_M, is_school  # noqa: E402
from src.generate.tags import _tags  # noqa: E402
from src.v3.cast import card, gender  # noqa: E402
from src.v3.frames import _POS, check_frame, person_tags  # noqa: E402

REF, CTRL = 20, 21
CARDS = P("processed") / "cards"
ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["build", "check"])
ap.add_argument("--run", default="ep22_v3")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--ip-scale", type=float, default=0.6)
a = ap.parse_args()
rd = P("runs") / a.run
shots = {s["shot_id"]: s for s in read_json(rd / "shotlist.json")["shots"]}
state = read_json(rd / "frames_state.json") or {}
ev = rd / "eval"


def picture(cid: str) -> str | None:
    c = card(cid) or {}
    if c.get("img_anime") and not c.get("img_anime_heldout") and (CARDS / c["img_anime"]).exists():
        return str(CARDS / c["img_anime"])
    return None


if a.step == "build":
    cands = []
    for sid, st in state.items():
        s = shots.get(sid)
        if not s or s["start"] in ("chain", "title") or st.get("final_verdict") != "redo":
            continue
        vis = [v for v in s.get("visible", []) if v["id"] != "Classmates"][:3]
        base = st.get("final_attempt", 0)
        chk = read_json(rd / "shots" / sid / f"check_a{base}.json") or {}
        verdict = {str(c.get("id")): c for c in (chk.get("cast") or []) if isinstance(c, dict)}
        flagged = [i for i, v in enumerate(vis) if not ((verdict.get(v["id"]) or {}).get("present", True) and (verdict.get(v["id"]) or {}).get("looks_right", True))]
        if len(vis) >= 2 and flagged and any(picture(vis[i]["id"]) for i in flagged) and (rd / "shots" / sid / f"kf_a{base}.png").exists():
            cands.append((sid, base, vis, verdict, flagged))
    random.Random(0).shuffle(cands)
    jobs, meta = [], []
    for sid, base, vis, verdict, flagged in cands[: a.n]:
        s, sdir = shots[sid], rd / "shots" / sid
        order = sorted(range(len(vis)), key=lambda i: (_POS.get(str(vis[i].get("position", "")).lower(), 1), i))
        slot = {i: k for k, i in enumerate(order)}
        n = len(vis)
        cur = {REF: str(sdir / f"kf_a{base}.png"), CTRL: str(sdir / f"kf_a{base}.png")}
        for i in flagged:
            v, c = vis[i], verdict.get(vis[i]["id"]) or {}
            k = slot[i]
            x0, x1 = max(0.0, k / n - 0.04), min(1.0, (k + 1) / n + 0.04)
            m = np.zeros((768, 1344), np.uint8)
            m[:, int(x0 * 1344) : int(x1 * 1344)] = 255
            mp = sdir / f"ipa_mask_{i}.png"
            Image.fromarray(m).filter(ImageFilter.GaussianBlur(18)).convert("RGB").save(mp)
            g = gender(v["id"], str(v.get("pose", "")))
            act = ", ".join(x for x in (str(v.get("pose") or "").strip(), str(v.get("expression") or "").strip()) if x)[:160]
            prompt = ", ".join(p for p in (ILL_QUALITY, ILL_SERIES, {"f": "1girl, solo", "m": "1boy, solo, male focus"}.get(g, "1other, solo"), person_tags(v, s),
                                           (ILL_UNIFORM_M if g == "m" else ILL_UNIFORM_F) if is_school(s.get("location", "")) else "",
                                           ILL_FRAMING.get(s.get("framing", ""), ""), act, ", ".join(_tags(s.get("background_tags"))[:6]), "general") if p)
            seed = int(sha(sid + f"ipa_{i}", 8), 16) % (2**31)
            pic = picture(v["id"])
            for arm in (REF, CTRL):
                last = i == flagged[-1]
                out = str(sdir / (f"kf_a{arm}.png" if last else f"ipa_a{arm}_{i}.png"))
                jobs.append({"id": f"{sid}_a{arm}_{i}", "image": cur[arm], "mask": str(mp), "prompt": prompt, "negative": ILL_NEG, "seed": seed,
                             "refs": [pic] if (arm == REF and pic) else [], "ip_scale": a.ip_scale, "ip_end": 0.85,
                             "denoise": 0.86 if not c.get("present", True) else 0.78, "out": out})
                cur[arm] = out
        meta.append({"shot_id": sid, "base_attempt": base, "redrawn": [vis[i]["id"] for i in flagged], "with_picture": [vis[i]["id"] for i in flagged if picture(vis[i]["id"])],
                     "base_score": state[sid].get("final_score")})
    ev.mkdir(parents=True, exist_ok=True)
    write_jsonl(ev / "ipa_test_jobs.jsonl", jobs)
    write_json(ev / "ipa_test.json", {"frames": meta, "ip_scale": a.ip_scale})
    print(f"{len(cands)} candidate frames, {len(meta)} taken, {len(jobs)} redraw jobs\n"
          f"next: scripts/gpu_run.sh 12 {P('work') if False else '/home/user/komi-poc-work'}/venvs/sdxl/bin/python src/generate/ipa_redraw.py --jobs {ev / 'ipa_test_jobs.jsonl'}")
else:
    t = read_json(ev / "ipa_test.json")
    todo = [(m, arm) for m in t["frames"] for arm in (REF, CTRL) if (rd / "shots" / m["shot_id"] / f"kf_a{arm}.png").exists()]
    with ThreadPoolExecutor(8) as ex:
        checks = list(ex.map(lambda ma: check_frame(rd, shots[ma[0]["shot_id"]], ma[1]), todo))
    for (m, arm), c in zip(todo, checks):
        m["ref" if arm == REF else "control"] = {k: c.get(k) for k in ("verdict", "score", "problems")} | {
            "cast": [{k: x.get(k) for k in ("id", "present", "looks_right", "note")} for x in (c.get("cast") or []) if isinstance(x, dict)]}
    done = [m for m in t["frames"] if "ref" in m and "control" in m]

    def right(m, arm):            # redrawn people the check now finds present and on-model
        v = {x["id"]: x for x in m[arm]["cast"]}
        return sum(1 for p in m["redrawn"] if (v.get(p) or {}).get("present") and (v.get(p) or {}).get("looks_right"))

    summ = {"frames": len(done), "people_redrawn": sum(len(m["redrawn"]) for m in done),
            "accepted_with_picture": sum(m["ref"]["verdict"] == "ok" for m in done), "accepted_control": sum(m["control"]["verdict"] == "ok" for m in done),
            "people_right_with_picture": sum(right(m, "ref") for m in done), "people_right_control": sum(right(m, "control") for m in done),
            "mean_score_base": round(float(np.mean([m["base_score"] or 0 for m in done])), 2),
            "mean_score_with_picture": round(float(np.mean([m["ref"]["score"] or 0 for m in done])), 2),
            "mean_score_control": round(float(np.mean([m["control"]["score"] or 0 for m in done])), 2)}
    t["summary"] = summ
    write_json(ev / "ipa_test.json", t)
    print(summ)
    f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    TW, TH, RW = 448, 256, 150
    for si in range(0, len(done), 8):
        part = done[si : si + 8]
        sheet = Image.new("RGB", (TW * 3 + RW + 25, (TH + 40) * len(part) + 30), "white")
        d = ImageDraw.Draw(sheet)
        for x, tx in ((5, "card picture(s)"), (RW + 10, "frame that failed the check"), (RW + TW + 15, "redrawn, no picture (control)"), (RW + TW * 2 + 20, "redrawn with the card picture")):
            d.text((x, 6), tx, fill="black", font=fb)
        for k, m in enumerate(part):
            y = 28 + k * (TH + 40)
            sd = rd / "shots" / m["shot_id"]
            for j, p in enumerate(m["with_picture"][:2]):
                im = Image.open(picture(p)).convert("RGB")
                im.thumbnail((RW, TH // 2 - 2))
                sheet.paste(im, (5, y + j * (TH // 2)))
            for x, f in ((RW + 10, sd / f"kf_a{m['base_attempt']}.png"), (RW + TW + 15, sd / f"kf_a{CTRL}.png"), (RW + TW * 2 + 20, sd / f"kf_a{REF}.png")):
                im = Image.open(f).convert("RGB")
                im.thumbnail((TW, TH))
                sheet.paste(im, (x, y))
            d.text((RW + 10, y + TH + 2), f"{m['shot_id']} redraw: {', '.join(m['redrawn'])} (score {m['base_score']})"[:64], fill=(90, 90, 90), font=f1)
            for x, arm in ((RW + TW + 15, "control"), (RW + TW * 2 + 20, "ref")):
                c = m[arm]
                for li, ln in enumerate(textwrap.wrap(f"{c['verdict']} {c['score']} " + "; ".join(c.get("problems") or []), 66)[:2]):
                    d.text((x, y + TH + 2 + li * 15), ln, fill=(0, 110, 0) if c["verdict"] == "ok" else (170, 0, 0), font=f1)
        sheet.save(ev / f"ipa_test_{si // 8 + 1}.jpg", quality=88)
    print(f"sheets: {ev}/ipa_test_*.jpg")
