"""Generation pipeline orchestrator (brief §7 + §8): chapters -> runs/<run>/episode.mp4 (+ animatic baseline).

  python -m src.generate.pipeline --run m1 --chapters 1 [--max-shots 12] [--stages director,tts,keyframes,...]

Stages (each idempotent; state in runs/<run>/state.json):
  director -> tts -> keyframes -> kfqa -> continuity -> video -> qa (repair ladder rounds) -> mix -> assemble -> report
GPU discipline: one large model at a time (ComfyUI for Qwen-Edit/Wan, llama.cpp Qwen for evidence/rewrites, TTS alone).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common import servers
from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json
from src.generate import audio as A
from src.generate.assemble import assemble
from src.generate.comfy import Comfy
from src.generate.keyframes import make_keyframe, run_keyframes
from src.generate.prompts import video_prompt
from src.generate.video import make_video, make_videos_batched, still_clip
from src.qa import repair as R

ALL_STAGES = ["director", "tts", "keyframes", "kfqa", "continuity", "video", "qa", "mix", "assemble", "report"]


class Run:
    def __init__(self, name: str, loras: dict | None = None, mode: str = "twos"):
        self.dir = P("runs") / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state: dict = read_json(self.dir / "state.json") or {}
        self.loras = loras or {}
        self.mode = mode
        self.wan_trigger: str = ""            # trigger words of a third-party Wan LoRA ("" = this project's trigger)
        self.repair_scope: str = "all"        # "motion": content-only failures keep their clip without a repair attempt (D-036)
        self.timing: dict = read_json(self.dir / "timing.json") or {}

    # ------------------------------------------------------------------ persistence
    def shots(self) -> list[dict]:
        return (read_json(self.dir / "shotlist.json") or {}).get("shots", [])

    def save_shots(self, shots: list[dict], meta: dict | None = None) -> None:
        cur = read_json(self.dir / "shotlist.json") or {}
        cur.update(meta or {})
        cur["shots"] = shots
        write_json(self.dir / "shotlist.json", cur)

    def st(self, sid: str) -> dict:
        return self.state.setdefault(sid, {"kf_attempt": 0, "vid_attempt": 0, "rung": "initial", "attempts": 0, "verdict": None,
                                           "final": None, "hints": {}, "history": []})

    def save_state(self) -> None:
        write_json(self.dir / "state.json", self.state)

    def tick(self, key: str, t0: float) -> None:
        self.timing[key] = round(self.timing.get(key, 0.0) + time.time() - t0, 1)
        write_json(self.dir / "timing.json", self.timing)

    def kf_loras(self):
        return self.loras.get("kf") or None

    def _prev_kf(self, shots: list[dict], sid: str) -> Path | None:
        ids = [s["shot_id"] for s in shots]
        i = ids.index(sid) if sid in ids else 0
        p = self.dir / "shots" / ids[i - 1] / "kf.png" if i > 0 else None
        return p if p is not None and p.exists() and "TITLE_CARD" not in (shots[i - 1].get("notes") or "") else None

    # ------------------------------------------------------------------ stages
    def director(self, chapters: list[int], max_shots: int | None = None, shot_budget: int | None = None) -> None:
        """`shot_budget`: total shots for the whole run, split over the chapters by page count (GPU time is ~linear in shots)."""
        if self.shots():
            return
        from src.generate.director import direct_chapter

        t0 = time.time()
        shots, metas = [], []
        pages = {c["chapter"]: c.get("n_pages", 10) for c in read_json(P("processed") / "manga" / "chapters.json", [])}
        total_pages = sum(pages.get(c, 10) for c in chapters) or 1
        def one(ch):
            cap = max(8, round(shot_budget * pages.get(ch, 10) / total_pages)) if shot_budget else None
            return direct_chapter(ch, max_shots=cap)

        with ThreadPoolExecutor(4) as ex:          # chapters are independent API calls
            results = list(ex.map(one, chapters))
        for ch, res in zip(chapters, results):
            write_json(self.dir / "director" / f"c{ch:03d}.json", res)
            shots += res["shots"]
            metas.append({k: res[k] for k in ("chapter", "title", "endpoint", "n_panels", "fewshot_chapters")})
            log(f"director: chapter {ch} -> {len(res['shots'])} shots")
        if max_shots:
            shots = shots[:max_shots]
        self.save_shots(shots, {"chapters": chapters, "director": metas})
        write_json(self.dir / "shotlist_director.json", {"shots": shots})       # untouched copy for the director evaluation
        self.tick("director", t0)

    def tts(self) -> None:
        if (self.dir / "tts" / ".done").exists():
            return
        t0 = time.time()
        servers.stop("qwen")
        servers.stop("comfy")
        shots = A.tts_stage(self.dir, self.shots())
        self.save_shots(shots)
        # voice metrics (CER vs script, speaker similarity) = audio evidence for the judge
        try:
            py = P("work") / "venvs" / "tts" / "bin" / "python"
            subprocess.run([f"{CFG.paths.repo}/scripts/gpu_run.sh", "6", str(py), "-m", "src.eval.voice_metrics",
                            "--results", str(self.dir / "tts" / "tts_results.jsonl"),
                            "--voices", str(P("processed") / "voices" / "voices.json"),
                            "--out", str(self.dir / "tts" / "voice_metrics.json"), "--device", "cuda"],
                           cwd=CFG.paths.repo, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
        except Exception as e:
            log(f"voice metrics skipped: {str(e)[:120]}")
        (self.dir / "tts" / ".done").write_text(time.strftime("%F %T"))
        self.tick("tts", t0)

    def keyframes(self) -> None:
        shots = self.shots()
        if all((self.dir / "shots" / s["shot_id"] / "kf.png").exists() for s in shots):
            return
        t0 = time.time()
        from src.generate.keyframes import backend

        if backend() == "illustrious":
            from src.generate.tags import tag_shots

            have = read_json(self.dir / "kf_tags.json") or {}
            if any(s["shot_id"] not in have for s in shots if "TITLE_CARD" not in (s.get("notes") or "")):
                servers.only("qwen")
                tag_shots(self.dir, shots)
        self._free_gpu()
        servers.only("comfy")
        run_keyframes(self.dir, shots, loras=self.kf_loras())
        self.tick("keyframes", t0)

    def _metrics(self, shots: list[dict], with_clip: bool) -> dict[str, dict]:
        from src.qa.evidence import metrics_evidence

        out = {}
        for s in shots:
            sdir = self.dir / "shots" / s["shot_id"]
            clip = sdir / "clip.mp4" if with_clip else None
            try:
                out[s["shot_id"]] = metrics_evidence(self.dir, s, clip=clip)
            except Exception as e:
                out[s["shot_id"]] = {"shot_id": s["shot_id"], "error": str(e)[:200], "frames": []}
        self._free_gpu()
        return out

    @staticmethod
    def _free_gpu() -> None:
        """Drop every model this process put on the GPU (RAFT / DOVER / DINOv2), otherwise the VRAM admission of the next
        large server (ComfyUI needs 22 GB free) waits forever on our own 2 GB."""
        import gc

        try:
            from src.eval import video_metrics as vm

            vm.unload_models()
        except Exception:
            pass
        try:
            from src.align import visual as av

            av._state.clear()
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _visual_and_judge(self, shots: list[dict], evs: dict[str, dict], stage: str) -> dict[str, dict]:
        from src.llm.client import LLM
        from src.qa.evidence import audio_evidence, hard_fail, visual_evidence
        from src.qa.judge import judge_shot

        servers.only("qwen")
        qwen = LLM("local", role="qa_visual")

        def vis(s):
            ev = evs[s["shot_id"]]
            if ev.get("frames") and "TITLE_CARD" not in (s.get("notes") or ""):
                ev["visual"] = visual_evidence(qwen, ev["frames"], s)
            if stage == "video":
                ev["audio"] = audio_evidence(self.dir, s)
            return s["shot_id"]

        with ThreadPoolExecutor(int(CFG.llm.local.parallel)) as ex:
            list(ex.map(vis, shots))

        def jdg(s):
            ev = evs[s["shot_id"]]
            if "TITLE_CARD" in (s.get("notes") or ""):
                return s["shot_id"], {"verdict": "pass", "scores": {"identity": 10, "spec_match": 10, "motion": 10, "audio": 10},
                                      "failure_reasons": [], "evidence_cited": [], "repair_hint": "", "stage": stage, "judge_endpoint": "rule"}
            hf = hard_fail(ev)
            if hf:
                return s["shot_id"], {"verdict": "fail", "scores": {"identity": 0, "spec_match": 0, "motion": 0, "audio": 0},
                                      "failure_reasons": hf, "evidence_cited": hf, "repair_hint": "regenerate", "suggested_repair": "reseed",
                                      "stage": stage, "judge_endpoint": "hard_fail"}
            try:
                return s["shot_id"], judge_shot(s, ev, stage=stage)
            except Exception as e:
                log(f"judge error {s['shot_id']}: {str(e)[:160]}")
                return s["shot_id"], {"verdict": "pass", "scores": {}, "failure_reasons": [], "evidence_cited": [], "repair_hint": "",
                                      "stage": stage, "judge_endpoint": "error", "error": str(e)[:200]}

        with ThreadPoolExecutor(6) as ex:
            verdicts = dict(ex.map(jdg, shots))
        for s in shots:
            sid = s["shot_id"]
            st = self.st(sid)
            n = st["kf_attempt"] if stage == "keyframe" else st["vid_attempt"]
            write_json(self.dir / "shots" / sid / f"qa_{stage}_a{n}.json", {"evidence": evs[sid], "verdict": verdicts[sid]})
        return verdicts

    def kfqa(self, rounds: int = 2) -> None:
        """Judge first frames before spending video time; regenerate failing keyframes (ladder rung 3)."""
        shots = self.shots()
        pending = [s for s in shots if not self.st(s["shot_id"]).get("kf_ok")]
        t0 = time.time()
        for rnd in range(rounds + 1):
            if not pending:
                break
            servers.stop("comfy")
            evs = self._metrics(pending, with_clip=False)
            verdicts = self._visual_and_judge(pending, evs, "keyframe")
            failed = []
            for s in pending:
                st, v = self.st(s["shot_id"]), verdicts[s["shot_id"]]
                st["history"].append({"stage": "keyframe", "attempt": st["kf_attempt"], "verdict": v["verdict"], "reasons": v.get("failure_reasons", [])})
                unsuitable = v["verdict"] != "pass" and any("unsuitable" in str(r).lower() or "all-ages" in str(r).lower() or "kiss" in str(r).lower()
                                                           for r in v.get("failure_reasons", []))
                if unsuitable and rnd == rounds:
                    # hard rule: a frame flagged as unsuitable is never shown. Use the colourised panel of the reference run.
                    self._kf_fallback(s, st)
                    st["kf_ok"], st["kf_verdict"] = True, "fallback_unsuitable"
                elif v["verdict"] == "pass" or rnd == rounds:
                    st["kf_ok"] = True
                    st["kf_verdict"] = v["verdict"]
                else:
                    R.log_failure(self.dir, s["shot_id"], st["kf_attempt"], "regen_keyframe", v)
                    st["hints"]["kf"] = str(v.get("repair_hint", ""))[:400]
                    failed.append(s)
            self.save_state()
            log(f"kfqa round {rnd}: {len(pending) - len(failed)}/{len(pending)} keyframes passed")
            if not failed:
                break
            self._free_gpu()
            servers.only("comfy")
            comfy = Comfy()
            panels_all = read_jsonl(P("processed") / "manga" / "panels.jsonl")
            for s in failed:
                st = self.st(s["shot_id"])
                st["kf_attempt"] += 1
                panels = {p["order"]: p for p in panels_all if p["chapter"] == s["chapter"]}
                try:
                    make_keyframe(comfy, self.dir, s, panels, attempt=st["kf_attempt"], hint=st["hints"].get("kf", ""), loras=self.kf_loras(),
                                  prev_kf=self._prev_kf(shots, s["shot_id"]))
                except Exception as e:
                    log(f"keyframe regen {s['shot_id']} failed: {str(e)[:200]}")
            self.save_state()
            pending = failed
        self.tick("kfqa", t0)

    def _kf_fallback(self, s: dict, st: dict) -> None:
        """Replace the keyframe by the reference run's one (KOMI_KF_FALLBACK_RUN, default: the run name without its suffix)."""
        import os
        import shutil

        ref = os.environ.get("KOMI_KF_FALLBACK_RUN") or self.dir.name.split("_")[0]
        src = P("runs") / ref / "shots" / s["shot_id"] / "kf.png"
        dst = self.dir / "shots" / s["shot_id"] / "kf.png"
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
            st["kf_fallback"] = ref
            log(f"keyframe {s['shot_id']}: flagged unsuitable -> replaced by the keyframe of run {ref}")
        else:
            from src.generate.keyframes import title_card

            title_card("…", dst)                 # last resort: a neutral card rather than a flagged frame
            st["kf_fallback"] = "card"

    def continuity(self) -> None:
        if (self.dir / "continuity.json").exists():
            return
        from src.qa.continuity import check_continuity

        t0 = time.time()
        shots = self.shots()
        flags = check_continuity(self.dir, shots)
        write_json(self.dir / "continuity.json", flags)
        redo = [f for f in flags.get("issues", []) if f.get("fix_keyframe") and f.get("shot_id") in {s["shot_id"] for s in shots}]
        if redo:
            self._free_gpu()
            servers.only("comfy")
            comfy = Comfy()
            panels_all = read_jsonl(P("processed") / "manga" / "panels.jsonl")
            by = {s["shot_id"]: s for s in shots}
            for f in redo[: max(3, len(shots) // 5)]:
                s, st = by[f["shot_id"]], self.st(f["shot_id"])
                st["kf_attempt"] += 1
                st["hints"]["kf"] = (st["hints"].get("kf", "") + " " + f.get("hint", "")).strip()[:500]
                panels = {p["order"]: p for p in panels_all if p["chapter"] == s["chapter"]}
                try:
                    make_keyframe(comfy, self.dir, s, panels, attempt=st["kf_attempt"], hint=st["hints"]["kf"], loras=self.kf_loras())
                    st["history"].append({"stage": "continuity", "issue": f.get("issue")})
                except Exception as e:
                    log(f"continuity regen {s['shot_id']} failed: {str(e)[:200]}")
            self.save_state()
        log(f"continuity: {len(flags.get('issues', []))} issues, {len(redo)} keyframes regenerated")
        self.tick("continuity", t0)

    def _render(self, comfy: Comfy, s: dict, st: dict, spec: dict | None = None) -> bool:
        sdir = self.dir / "shots" / s["shot_id"]
        spec = spec or s
        if "TITLE_CARD" in (s.get("notes") or ""):
            still_clip(sdir / "kf.png", sdir / "clip.mp4", s["duration"], "static")
            st["final"] = "title"
            return True
        for oom_retry in range(2):
            try:
                make_video(comfy, self.dir, spec, attempt=st["vid_attempt"], hint=st["hints"].get("video", ""),
                           loras_high=self.loras.get("wan_high"), loras_low=self.loras.get("wan_low"), mode=self.mode)
                st["final"] = "wan"
                return True
            except Exception as e:
                msg = str(e)
                log(f"video {s['shot_id']} attempt {st['vid_attempt']} error: {msg[:240]}")
                if "out of memory" in msg.lower() or "OOM" in msg:   # pipeline self-healing: free and retry once
                    comfy.free()
                    time.sleep(5)
                    continue
                break
        return False

    def _render_batch(self, comfy: Comfy, items: list[tuple[dict, dict]]) -> None:
        """items: [(spec, state)] -> Wan clips with one expert swap per chunk; failures fall back to the pan/zoom clip."""
        jobs = [{"shot": spec, "attempt": st["vid_attempt"], "hint": st["hints"].get("video", "")} for spec, st in items]
        res = make_videos_batched(comfy, self.dir, jobs, mode=self.mode, loras_high=self.loras.get("wan_high"),
                                  loras_low=self.loras.get("wan_low"), trigger=self.wan_trigger or True, log_fn=log)
        for spec, st in items:
            sid = spec["shot_id"]
            r = res.get(sid)
            sdir = self.dir / "shots" / sid
            if isinstance(r, Exception) or r is None:
                log(f"video {sid} attempt {st['vid_attempt']} failed: {str(r)[:200]}")
                try:
                    still_clip(sdir / "kf.png", sdir / "clip.mp4", spec["duration"], spec.get("camera", "static"))
                except Exception as e:
                    log(f"fallback clip {sid} failed: {str(e)[:120]}")
                st["final"], st["rung"] = "fallback", "fallback"
            else:
                st["final"] = "wan"

    def video(self) -> None:
        shots = self.shots()
        todo = [s for s in shots if not (self.dir / "shots" / s["shot_id"] / "clip.mp4").exists()]
        if not todo:
            return
        t0 = time.time()
        servers.only("comfy")
        comfy = Comfy()
        items = []
        for s in todo:
            st = self.st(s["shot_id"])
            sdir = self.dir / "shots" / s["shot_id"]
            if "TITLE_CARD" in (s.get("notes") or ""):
                still_clip(sdir / "kf.png", sdir / "clip.mp4", s["duration"], "static")
                st["final"] = "title"
            else:
                items.append((s, st))
            st["attempts"] = max(1, st.get("attempts", 0))
        # in slices, so progress is saved regularly
        for i in range(0, len(items), 48):
            self._render_batch(comfy, items[i : i + 48])
            self.save_state()
            log(f"video {min(i + 48, len(items))}/{len(items)} ({(time.time() - t0) / max(1, min(i + 48, len(items))):.0f}s/shot)")
        self.save_state()
        self.tick("video", t0)

    def qa(self, max_rounds: int = 3) -> None:
        """Judge every generated shot; climb the repair ladder for failures (batched per round to limit model swaps)."""
        from src.llm.client import LLM

        shots = self.shots()
        by = {s["shot_id"]: s for s in shots}
        pending = [s for s in shots if self.st(s["shot_id"]).get("verdict") is None and self.st(s["shot_id"]).get("final") in ("wan", "fallback")]
        t0 = time.time()
        for rnd in range(max_rounds + 1):
            if not pending:
                break
            servers.stop("comfy")
            evs = self._metrics(pending, with_clip=True)
            verdicts = self._visual_and_judge(pending, evs, "video")
            failed = []
            qwen = LLM("local", role="repair")
            for s in pending:
                sid, v = s["shot_id"], verdicts[s["shot_id"]]
                st = self.st(sid)
                st["history"].append({"stage": "video", "attempt": st["vid_attempt"], "rung": st["rung"], "verdict": v["verdict"],
                                      "reasons": v.get("failure_reasons", []), "scores": v.get("scores", {})})
                if v["verdict"] == "pass" or st.get("final") == "fallback":
                    st["verdict"] = v["verdict"]
                    continue
                R.log_failure(self.dir, sid, st["vid_attempt"], st["rung"], v)
                if self.repair_scope == "motion" and not R.motion_failure(v):
                    # content-only failure (identity / spec / leftovers): neither a re-render nor the still fixes it
                    st["rung"], st["verdict"] = "keep", "fail_kept"
                    continue
                rung = R.next_rung(st["rung"], v.get("suggested_repair"))
                if st["attempts"] >= R.MAX_ATTEMPTS or rnd == max_rounds:
                    # last rung. The pan/zoom still only helps when the MOTION is the problem; a content failure
                    # (wrong character, spec mismatch, leftovers) is in the keyframe and would survive it.
                    rung = "fallback" if R.motion_failure(v) else "keep"
                st["rung"] = rung
                if rung in ("rewrite_prompt", "regen_keyframe", "simplify"):
                    prev = (read_json(self.dir / "shots" / sid / f"video_a{st['vid_attempt']}.json") or {}).get("prompt") or video_prompt(s)
                    h = R.rewrite_hints(qwen, s, prev, v)
                    st["hints"]["video"] = h.get("video_hint", "")[:400]
                    st["hints"]["kf"] = h.get("keyframe_hint", "")[:400]
                failed.append(s)
            self.save_state()
            log(f"qa round {rnd}: {len(pending) - len(failed)}/{len(pending)} passed")
            if not failed:
                break
            # ---- apply repairs (ComfyUI phase)
            self._free_gpu()
            servers.only("comfy")
            comfy = Comfy()
            panels_all = read_jsonl(P("processed") / "manga" / "panels.jsonl")
            nxt, batch = [], []
            for s in failed:
                sid = s["shot_id"]
                st = self.st(sid)
                sdir = self.dir / "shots" / sid
                rung = st["rung"]
                if rung == "fallback":
                    still_clip(sdir / "kf.png", sdir / "clip.mp4", s["duration"], s.get("camera", "static"))
                    st["final"], st["verdict"] = "fallback", "fallback"
                    continue
                if rung == "keep":
                    st["verdict"] = "fail_kept"          # animated clip stays; listed in review.json
                    continue
                spec = R.simplify(s) if rung == "simplify" else s
                if rung in ("regen_keyframe", "simplify"):
                    st["kf_attempt"] += 1
                    panels = {p["order"]: p for p in panels_all if p["chapter"] == s["chapter"]}
                    try:
                        make_keyframe(comfy, self.dir, spec, panels, attempt=st["kf_attempt"], hint=st["hints"].get("kf", ""),
                                      loras=self.kf_loras(), max_chars=1 if rung == "simplify" else None, prev_kf=self._prev_kf(shots, sid))
                    except Exception as e:
                        log(f"repair keyframe {sid} failed: {str(e)[:200]}")
                st["vid_attempt"] += 1
                st["attempts"] += 1
                batch.append((spec, st))
                nxt.append(s)
            self._render_batch(comfy, batch)
            for s in list(nxt):          # shots whose re-render failed technically ended on the fallback clip
                if self.st(s["shot_id"]).get("final") == "fallback":
                    self.st(s["shot_id"])["verdict"] = "fallback"
                    nxt.remove(s)
            self.save_state()
            pending = nxt
        R.review_list(self.dir, self.state)
        self.tick("qa", t0)

    def mix(self) -> None:
        t0 = time.time()
        A.mix_stage(self.dir, self.shots())
        self.tick("mix", t0)

    def assemble(self) -> None:
        t0 = time.time()
        servers.stop("comfy")
        shots = self.shots()
        assemble(self.dir, shots, variant="wan")
        assemble(self.dir, shots, variant="animatic")
        self.tick("assemble", t0)

    def report(self) -> dict:
        shots = self.shots()
        st = [self.st(s["shot_id"]) for s in shots]
        gen = [x for x in st if x.get("final") in ("wan", "fallback")]
        first_pass = sum(1 for x in gen if any(h["stage"] == "video" and h["attempt"] == 0 and h["verdict"] == "pass" for h in x["history"]))
        rungs: dict = {}
        for x in gen:
            rungs[x.get("rung", "initial")] = rungs.get(x.get("rung", "initial"), 0) + 1
        scores: dict = {}
        for x in gen:
            vids = [h for h in x["history"] if h["stage"] == "video" and h.get("scores")]
            if vids:
                for k, v in vids[-1]["scores"].items():
                    if isinstance(v, (int, float)):
                        scores.setdefault(k, []).append(v)
        kf_hist = [h for x in st for h in x["history"] if h["stage"] == "keyframe"]
        rep = {
            "run": self.dir.name, "n_shots": len(shots), "duration_sec": round(sum(s["duration"] for s in shots), 1),
            "generated_shots": len(gen), "final_wan": sum(1 for x in gen if x.get("final") == "wan"),
            "final_fallback": sum(1 for x in gen if x.get("final") == "fallback"),
            "final_wan_passed": sum(1 for x in gen if x.get("final") == "wan" and x.get("verdict") == "pass"),
            "final_wan_kept_despite_content_failure": sum(1 for x in gen if x.get("verdict") == "fail_kept"),
            "video_first_pass_rate": round(first_pass / max(1, len(gen)), 3),
            "video_final_pass_rate": round(sum(1 for x in gen if x.get("verdict") == "pass") / max(1, len(gen)), 3),
            "keyframe_first_pass_rate": round(sum(1 for h in kf_hist if h["attempt"] == 0 and h["verdict"] == "pass") / max(1, sum(1 for h in kf_hist if h["attempt"] == 0)), 3) if kf_hist else None,
            "video_attempts_total": sum(x.get("vid_attempt", 0) + 1 for x in gen),
            "qa_retry_rate": round(sum(x.get("vid_attempt", 0) for x in gen) / max(1, len(gen)), 3),
            "final_rung_counts": rungs,
            "mean_judge_scores": {k: round(sum(v) / len(v), 2) for k, v in scores.items()},
            "timing_sec": self.timing, "gpu_hours": round(sum(v for k, v in self.timing.items() if k in ("tts", "keyframes", "kfqa", "continuity", "video", "qa")) / 3600, 2),
            "loras": self.loras,
        }
        write_json(self.dir / "report.json", rep)
        log("report: " + str({k: rep[k] for k in ("n_shots", "duration_sec", "final_wan", "final_fallback", "video_first_pass_rate", "video_final_pass_rate")}))
        return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--chapters", default="")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--max-shots", type=int, default=None, help="truncate the run to the first N shots (vertical slice)")
    ap.add_argument("--shot-budget", type=int, default=None, help="total shot budget for the run, split over chapters by page count")
    ap.add_argument("--mode", default="twos", choices=["twos", "threes", "smooth"])
    ap.add_argument("--wan-lora-high", default=None, help="file name in models/comfy/loras[:strength]")
    ap.add_argument("--wan-lora-low", default=None)
    ap.add_argument("--kf-lora", default=None)
    ap.add_argument("--qa-rounds", type=int, default=3)
    ap.add_argument("--kfqa-rounds", type=int, default=2)
    ap.add_argument("--kf-backend", default=None, choices=["qwen_edit", "illustrious"], help="keyframe model (default: env KOMI_KF_BACKEND or qwen_edit)")
    ap.add_argument("--wan-trigger", default="", help="trigger words of a third-party Wan LoRA, e.g. 'An1meStyl3, AnimeStyle'")
    ap.add_argument("--repair-scope", default="all", choices=["all", "motion"])
    a = ap.parse_args()

    def lora(arg):
        if not arg:
            return None
        name, _, strength = arg.partition(":")
        return [(name, float(strength) if strength else 1.0)]

    if a.kf_backend:
        import os

        os.environ["KOMI_KF_BACKEND"] = a.kf_backend
    run = Run(a.run, {"wan_high": lora(a.wan_lora_high), "wan_low": lora(a.wan_lora_low), "kf": lora(a.kf_lora)}, mode=a.mode)
    run.wan_trigger, run.repair_scope = a.wan_trigger, a.repair_scope
    stages = ALL_STAGES if a.stages == "all" else a.stages.split(",")
    chapters = [int(c) for c in a.chapters.split(",") if c]
    for stg in stages:
        log(f"=== stage: {stg}")
        if stg == "director":
            run.director(chapters, a.max_shots, a.shot_budget)
        elif stg == "qa":
            run.qa(a.qa_rounds)
        elif stg == "kfqa":
            run.kfqa(a.kfqa_rounds)
        else:
            getattr(run, stg)()
    log("pipeline done")


if __name__ == "__main__":
    main()
