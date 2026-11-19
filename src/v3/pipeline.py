"""v3 pipeline (D-043): manga chapters -> reading -> movie script -> shot plan -> voices, first frames, clips -> episode.

  python -m src.v3.pipeline --run ep22_v3 --chapters 114,116,117,118 [--stages read,script,plan,tts,frames,video,mix,assemble,report]

Models: Gemini 3.8 Flash reads and checks pictures, DeepSeek V4.1 Flash writes (script, shots, prompts), Illustrious-XL
draws first frames, Wan 2.2 I2V animates, Qwen3-TTS speaks (English). State lives in runs/<run>/; every stage resumes.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from src.common.config import CFG, P
from src.common.io import log, read_json, write_json

# "tighten", "wardrobe" and "continuity" were added on 2026-11-12 (D-052, D-055, D-056); "fix" belongs to the Illustrious frame
# backend, "continuity" to the Gemini one - each is skipped under the other backend.
# "edit" (D-059) and "reframe" (D-060) were added on 2026-11-13: the first cuts long multi-line shots on the line once the
# voices have set the durations, the second crops the finished first frames to the shot sizes of the show.
# "stage" (D-063, camera pass) follows the editor pass so that the new half-shots are staged too.
ALL = ["read", "script", "tighten", "plan", "wardrobe", "tts", "edit", "stage", "frames", "fix", "continuity", "reframe", "video", "mix", "assemble", "report"]
MAX_SHOT_SEC = 14.0


class RunV3:
    def __init__(self, name: str, mode: str = "threes"):
        self.name, self.dir, self.mode = name, P("runs") / name, mode
        self.check_clips = True
        self.frame_backend = "illustrious"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timing: dict = read_json(self.dir / "timing.json") or {}

    def shots(self) -> list[dict]:
        return (read_json(self.dir / "shotlist.json") or {}).get("shots", [])

    def save_shots(self, shots: list[dict]) -> None:
        cur = read_json(self.dir / "shotlist.json") or {}
        cur["shots"] = shots
        write_json(self.dir / "shotlist.json", cur)

    def tick(self, key: str, t0: float) -> None:
        self.timing[key] = round(self.timing.get(key, 0.0) + time.time() - t0, 1)
        write_json(self.dir / "timing.json", self.timing)

    # ------------------------------------------------------------------ text stages
    def read(self, chapters: list[int]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from src.v3.read import read_chapter

        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(read_chapter, chapters))
        self.tick("read", t0)

    def script(self, chapters: list[int]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from src.v3.script import Script, to_text

        t0 = time.time()
        sc = Script(self.name)
        sc.bible(chapters)
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(sc.draft, chapters))
            for c, r in zip(chapters, ex.map(sc.revise, chapters)):
                (sc.dir / f"c{c:03d}.md").write_text(to_text(r))
        self.tick("script", t0)

    def tighten(self, chapters: list[int]) -> None:
        """Last script pass: the word and time budget the real series shows on its training episodes (src/v3/pace.py)."""
        from concurrent.futures import ThreadPoolExecutor

        from src.v3.pace import adaptation_stats
        from src.v3.script import Script, to_text

        t0 = time.time()
        adaptation_stats()                                  # computed once, before the threads
        sc = Script(self.name)
        with ThreadPoolExecutor(4) as ex:
            for c, r in zip(chapters, ex.map(sc.tighten, chapters)):
                (sc.dir / f"c{c:03d}.md").write_text(to_text(r))
        self.tick("tighten", t0)

    def wardrobe(self) -> None:
        """One coloured outfit per character and scene, carried word for word by every shot (src/v3/wardrobe.py)."""
        from src.v3.wardrobe import apply, chapter_wardrobe

        cur = read_json(self.dir / "shotlist.json") or {}
        if cur.get("wardrobe") or not cur.get("shots"):
            return
        t0 = time.time()
        w: dict = {}
        for ch in cur.get("chapters") or sorted({s["chapter"] for s in cur["shots"]}):
            w.update(chapter_wardrobe(ch, cur["shots"]))
        shots, changed = apply(cur["shots"], w)
        write_json(self.dir / "wardrobe.json", {"outfits": w, "shots_changed": changed})
        write_json(self.dir / "shotlist.json", {**cur, "shots": shots, "wardrobe": True})
        log(f"wardrobe: {len(w)} scenes, {sum(len(v) for v in w.values())} outfits")
        self.tick("wardrobe", t0)

    def continuity(self) -> None:
        """Script-supervisor pass on the first frames: flagged frames are redrawn against the frame they contradict."""
        if self.frame_backend != "gemini":
            return
        from src.v3.continuity_fix import repair

        t0 = time.time()
        repair(self.name, rounds=2)
        self.tick("continuity", t0)

    def plan(self, chapters: list[int]) -> None:
        if self.shots():
            return
        from concurrent.futures import ThreadPoolExecutor

        from src.v3.shots import plan_chapter

        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            per = list(ex.map(lambda c: plan_chapter(self.name, c), chapters))
        write_json(self.dir / "shotlist.json", {"chapters": chapters, "pipeline": "v3", "shots": [s for part in per for s in part]})
        self.tick("plan", t0)

    # ------------------------------------------------------------------ voices
    def tts(self) -> None:
        if (self.dir / "tts" / ".done").exists():
            return
        from src.common import servers
        from src.generate.audio import GAP, LEAD, SPOKEN, TAIL
        from src.v3.voice import speakable, synth_lines

        t0 = time.time()
        if servers.free_vram_gb() < 9:             # the voice model (8 GB) fits beside the keyframe model, not beside Wan
            servers.stop("comfy")
        shots = self.shots()
        for s in shots:                            # "A..." is read as the letter's name: interjections in a speakable spelling (D-065)
            for d in s["dialogue"]:
                if d["kind"] in SPOKEN:
                    d["text"] = speakable(d["text"])
        # main audio: CustomVoice + cloned embedding + instruction (D-047); second audio: plain cloning.
        # Shots are timed to hold the longer of the two takes, so both mixes fit the same picture.
        res = synth_lines(self.dir, shots, mode="cv_xvec", sub="tts")
        alt = synth_lines(self.dir, shots, mode="icl", sub="tts_clone")
        for s in shots:
            ids = [(k, f"{s['shot_id']}_l{k}") for k, d in enumerate(s["dialogue"]) if d["kind"] in SPOKEN]
            s["planned_duration"] = s.get("planned_duration", s["duration"])
            need = 0.0
            for r in (res, alt):
                durs = [r[i]["dur"] for _, i in ids if i in r]
                if durs:
                    need = max(need, LEAD + sum(durs) + GAP * (len(durs) - 1) + TAIL)
            if need:
                s["duration"] = round(float(min(MAX_SHOT_SEC, max(need, min(s["planned_duration"], need + 1.5)))), 3)
            s["tts"] = {f"l{k}": res[i]["dur"] for k, i in ids if i in res}
            s["tts_clone"] = {f"l{k}": alt[i]["dur"] for k, i in ids if i in alt}
        self.save_shots(shots)
        (self.dir / "tts" / ".done").write_text(time.strftime("%F %T"))
        self.tick("tts", t0)

    def edit(self) -> None:
        """Editor pass (D-059): a long shot that holds several spoken lines is cut in two on a line. In place, once, before the frames."""
        if (read_json(self.dir / "shotlist.json") or {}).get("edited_from") or not (self.dir / "tts" / ".done").exists():
            return
        from src.v3.edit import apply

        t0 = time.time()
        apply(self.name, self.name)
        self.tick("edit", t0)

    def stage(self) -> None:
        """Camera pass (D-063): an angle, a view and sometimes a foreground for every drawn shot (used by the Gemini frame backend)."""
        if self.frame_backend != "gemini":
            return
        from src.v3.staging import direct

        t0 = time.time()
        direct(self.name)
        self.tick("stage", t0)

    # ------------------------------------------------------------------ pictures
    def reframe(self) -> None:
        """Reframing pass (D-060): crop each finished first frame to the shot size its plan asked for (measured on Gemini frames only).
        A frame whose clip already exists is left alone - `python -m src.v3.reframe --deferred` is the way to redo those."""
        if self.frame_backend != "gemini":
            return
        from src.v3.reframe import run as reframe_run

        t0 = time.time()
        have = {s["shot_id"] for s in self.shots() if (self.dir / "shots" / s["shot_id"] / "clip.mp4").exists()
                and not (self.dir / "shots" / s["shot_id"] / "kf_wide.png").exists()}
        reframe_run(self.name, apply=True, lock=have)
        self.tick("reframe", t0)

    def frames(self, rounds: int = 1) -> None:
        """First frames: backend "illustrious" (ComfyUI, D-038) or "gemini" (image model with reference pictures, D-053)."""
        t0 = time.time()
        if self.frame_backend == "gemini":
            from src.generate.keyframes import title_card
            from src.v3.frames_gemini import run_frames_gemini

            for s in self.shots():
                if s["start"] == "title" and not (self.dir / "shots" / s["shot_id"] / "kf.png").exists():
                    (self.dir / "shots" / s["shot_id"]).mkdir(parents=True, exist_ok=True)
                    title_card(s["notes"].split(":", 1)[-1].strip() or "…", self.dir / "shots" / s["shot_id"] / "kf.png")
            st = run_frames_gemini(self.dir, self.shots(), rounds=rounds)
        else:
            from src.v3.frames import run_frames

            st = run_frames(self.dir, self.shots(), rounds=rounds)
        fin = [x for x in st.values() if x.get("final_verdict")]
        log(f"frames: {sum(1 for x in fin if x['final_verdict'] == 'ok')}/{len(fin)} accepted by the visual check")
        self.tick("frames", t0)

    def fix(self) -> None:
        """Multi-person frames that failed the check: redraw each wrong person alone in their part of the frame."""
        if self.frame_backend != "illustrious":
            return
        from src.v3.frames import run_fix_people

        t0 = time.time()
        st = run_fix_people(self.dir, self.shots())
        fin = [x for x in st.values() if x.get("final_verdict")]
        log(f"frames after the per-person redraw: {sum(1 for x in fin if x['final_verdict'] == 'ok')}/{len(fin)} accepted")
        self.tick("fix", t0)

    def video(self) -> None:
        """Clips in dependency order: shots with their own first frame, then shots that start on the previous clip's last frame."""
        from src.common import servers
        from src.generate.comfy import Comfy
        from concurrent.futures import ThreadPoolExecutor

        from src.generate.video import last_frame, make_videos_batched, still_clip
        from src.v3.check import check_clip

        t0 = time.time()
        shots = self.shots()
        clip = lambda s: self.dir / "shots" / s["shot_id"] / "clip.mp4"            # noqa: E731
        for s in shots:
            if s["start"] == "title" and not clip(s).exists():
                still_clip(self.dir / "shots" / s["shot_id"] / "kf.png", clip(s), s["duration"], "static")
        if all(clip(s).exists() for s in shots):
            return
        servers.only("comfy")
        comfy = Comfy()
        checks: dict = read_json(self.dir / "clips_state.json") or {}
        for wave in range(4):
            jobs = []
            for i, s in enumerate(shots):
                if clip(s).exists() or s["start"] == "title":
                    continue
                sdir = self.dir / "shots" / s["shot_id"]
                if s["start"] == "chain":
                    prev = shots[i - 1] if i > 0 else None
                    if prev is None or not clip(prev).exists():
                        continue                                   # its predecessor is rendered in an earlier wave
                    sdir.mkdir(parents=True, exist_ok=True)
                    last_frame(clip(prev), sdir / "kf.png")
                if (sdir / "kf.png").exists():
                    jobs.append({"shot": s, "attempt": 0, "hint": ""})
            if not jobs:
                break
            log(f"video wave {wave}: {len(jobs)} shots")
            for attempt in (0, 1):
                for i in range(0, len(jobs), 48):
                    part = [{**j, "attempt": attempt} for j in jobs[i : i + 48]]
                    res = make_videos_batched(comfy, self.dir, part, mode=self.mode, trigger=False, log_fn=log)
                    for j in part:
                        s = j["shot"]
                        if isinstance(res.get(s["shot_id"]), Exception) or res.get(s["shot_id"]) is None:
                            log(f"video {s['shot_id']} failed: {str(res.get(s['shot_id']))[:160]} -> still")
                            still_clip(self.dir / "shots" / s["shot_id"] / "kf.png", clip(s), s["duration"], s.get("camera", "static"))
                            checks[s["shot_id"]] = {"final": "still", "why": "render error"}
                    log(f"video wave {wave} attempt {attempt}: {min(i + 48, len(jobs))}/{len(jobs)} ({(time.time() - t0) / 60:.0f} min)")
                if not self.check_clips:
                    break
                # Gemini looks at four frames per clip; a deformed clip is rendered once more, then replaced by the still
                todo = [j for j in jobs if checks.get(j["shot"]["shot_id"], {}).get("final") != "still"]
                with ThreadPoolExecutor(8) as ex:
                    verdicts = list(ex.map(lambda j: check_clip(self.dir, j["shot"], attempt), todo))
                redo = []
                for j, v in zip(todo, verdicts):
                    sid = j["shot"]["shot_id"]
                    c = checks.setdefault(sid, {"checks": []})
                    c.setdefault("checks", []).append({k: v.get(k) for k in ("attempt", "verdict", "score", "motion", "problems", "identity_stable")})
                    if v.get("verdict") == "redo":
                        redo.append(j)
                    else:
                        c["final"] = "wan"
                log(f"video wave {wave} attempt {attempt}: clip check accepted {len(todo) - len(redo)}/{len(todo)}")
                if attempt == 1:
                    for j in redo:
                        s = j["shot"]
                        still_clip(self.dir / "shots" / s["shot_id"] / "kf.png", clip(s), s["duration"], s.get("camera", "static"))
                        checks[s["shot_id"]]["final"] = "still"
                jobs = redo
                write_json(self.dir / "clips_state.json", checks)
                if not jobs:
                    break
        write_json(self.dir / "clips_state.json", checks)
        self.tick("video", t0)

    # ------------------------------------------------------------------ sound and cut
    def mix(self) -> None:
        from src.generate import audio as A

        t0 = time.time()
        shots = self.shots()
        A.mix_stage(self.dir, shots)
        if any(s.get("tts_clone") for s in shots):         # second audio: the plain-clone voices on the same timeline
            A.mix_stage(self.dir, [{**s, "tts": s.get("tts_clone") or {}} for s in shots], tts_dir="tts_clone", out_dir="audio_clone")
        self.tick("mix", t0)

    def assemble(self) -> None:
        from src.common import servers
        from src.generate.assemble import assemble

        t0 = time.time()
        servers.stop("comfy")
        shots = self.shots()
        for s in shots:                                # a chained shot without a clip still needs a picture for the fallback
            kf = self.dir / "shots" / s["shot_id"] / "kf.png"
            if not kf.exists():
                prev = next((p for p in reversed(shots[: shots.index(s)]) if (self.dir / "shots" / p["shot_id"] / "kf.png").exists()), None)
                if prev:
                    kf.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.dir / "shots" / prev["shot_id"] / "kf.png", kf)
        assemble(self.dir, shots, variant="wan")
        if (self.dir / "audio_clone" / "mix.wav").exists():
            from src.generate.assemble import mux, write_subs

            subs = write_subs([{**s, "tts": s.get("tts_clone") or {}} for s in shots], self.dir / "episode_clonevoice.ass")
            mux(self.dir / "video_only.mp4", self.dir / "audio_clone" / "mix.wav", subs, self.dir / "episode_clonevoice.mp4", burn=True)
        self.tick("assemble", t0)

    def report(self) -> dict:
        shots = self.shots()
        fs = read_json(self.dir / "frames_state.json") or {}
        fin = [x for x in fs.values() if x.get("final_verdict")]
        from src.llm.client import api_spend

        rep = {"run": self.name, "pipeline": "v3", "n_shots": len(shots), "duration_sec": round(sum(s["duration"] for s in shots), 1),
               "start": {k: sum(1 for s in shots if s["start"] == k) for k in ("panel", "new", "chain", "title")},
               "frames_checked": len(fin), "frames_ok": sum(1 for x in fin if x["final_verdict"] == "ok"),
               "frames_first_attempt_ok": sum(1 for x in fs.values() if x.get("checks") and x["checks"][0].get("verdict") == "ok"),
               "clips": sum(1 for s in shots if (self.dir / "shots" / s["shot_id"] / "clip.mp4").exists()),
               "clips_final": {k: sum(1 for c in (read_json(self.dir / "clips_state.json") or {}).values() if c.get("final") == k) for k in ("wan", "still")},
               "spoken_lines": sum(len(s.get("tts") or {}) for s in shots), "timing_sec": self.timing, "api_spend_total_usd": round(api_spend(), 2)}
        write_json(self.dir / "report.json", rep)
        log("report: " + str({k: rep[k] for k in ("n_shots", "duration_sec", "start", "frames_ok", "frames_checked", "clips", "spoken_lines")}))
        return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--chapters", default="")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--mode", default="threes", choices=["twos", "threes", "smooth"])
    ap.add_argument("--frame-rounds", type=int, default=1)
    ap.add_argument("--frames", default="illustrious", choices=["illustrious", "gemini"], help="who draws the first frames")
    a = ap.parse_args()
    run = RunV3(a.run, a.mode)
    run.frame_backend = a.frames
    chapters = [int(c) for c in a.chapters.split(",") if c] or (read_json(run.dir / "shotlist.json") or {}).get("chapters", [])
    for stg in (ALL if a.stages == "all" else a.stages.split(",")):
        log(f"=== v3 stage: {stg}")
        if stg in ("read", "script", "tighten", "plan"):
            getattr(run, stg)(chapters)
        elif stg == "frames":
            run.frames(a.frame_rounds)
        else:
            getattr(run, stg)()
    log("v3 pipeline done")


if __name__ == "__main__":
    main()
