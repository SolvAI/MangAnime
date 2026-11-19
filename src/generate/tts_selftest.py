"""Model-free self-test of the TTS batch logic (text front-end, emotion mapping, duration fitting, sanity re-rolls,
cache, error handling). Uses a fake engine that returns a tone whose length follows the mora count and the requested
duration factor, so every fitting branch is exercised in a few seconds without GPU or weights.

  $KOMI_WORK/venvs/tts/bin/python -m src.generate.tts_selftest          (exit code 0 = all checks passed)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.common.config import CFG
from src.generate import tts

SEC_PER_MORA = 0.13


class FakeEngine:
    """Stands in for IndexTTSEngine: deterministic tone, duration = mora * 0.13 s * duration_factor (+ silence pads)."""
    device, precision, load_sec = "mock", "mock", 0.0

    def __init__(self):
        self.calls = 0

    def synth(self, text, prompt_wav, prompt_key, lang="JA", emo_vector=None, seed=0, duration_factor=1.0,
              max_mel_tokens=1500, emo_ref=None):
        self.calls += 1
        dur = tts.count_mora(text) * SEC_PER_MORA
        hit_cap = False
        if "らんなうぇい" in text and seed % 2 == 0:       # runaway on even seeds, fine on the re-roll
            dur, hit_cap = max_mel_tokens / float(tts.T.mel_tokens_per_sec), True
        if "むおん" in text:                               # always silent
            return np.zeros(int(0.5 * tts.MODEL_SR), dtype="float32"), {"synth_sec": 0.01, "gpt_sec": 0.0,
                                                                        "gpt_cached": False, "mel_tokens": 10,
                                                                        "hit_cap": False}
        if "ながすぎ" in text:                             # always far too long
            dur = 30.0
        n = int(dur * duration_factor * tts.MODEL_SR)
        t = np.arange(n) / tts.MODEL_SR
        tone = (0.3 * np.sin(2 * np.pi * (180 + seed % 40) * t)).astype("float32")
        pad = np.zeros(int(0.2 * tts.MODEL_SR), dtype="float32")
        return np.concatenate([pad, tone, pad]), {"synth_sec": 0.01, "gpt_sec": 0.005, "gpt_cached": False,
                                                  "mel_tokens": int(dur * 50), "hit_cap": hit_cap}

    def vram_peak_gb(self):
        return {"allocated": 0.0, "reserved": 0.0}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tts_selftest_", dir=str(CFG.paths.tmp)))
    OmegaConf.update(CFG, "tts.cache_dir", str(tmp / "cache"))
    OmegaConf.update(CFG, "tts.ref_cache_dir", str(tmp / "cache" / "refs"))
    fails: list[str] = []

    def check(name: str, cond: bool, info: object = "") -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {info if not cond else ''}")
        if not cond:
            fails.append(name)

    try:
        # ---------------------------------------------------------------- text + emotion
        n = tts.normalize_ja
        check("names forced", n("古見さんと只野くん") == "こみさんとただのくん。", n("古見さんと只野くん"))
        check("numbers", n("友達100人") == "友達百人。" and tts.int_to_kanji(10000000) == "一千万")
        check("ruby", n("漢字(かんじ)です") == "かんじです。")
        check("silent line", n("……") == "" and n("！？") == "")
        check("interjection", n("えっ") == "えっ。" and n("あ…") == "あ…" and n("え〜っ！？") == "えーっ!?")
        check("space is a pause", n("ちょっと待って　それは違う") == "ちょっと待って、それは違う。")
        check("mora", tts.count_mora("えっ。") == 2 and 9 <= tts.count_mora("こみさんは、こみゅしょうです。") <= 12)
        v, lab, w = tts.resolve_emotion("shy")
        check("label -> vector", v is not None and len(v) == 8 and sum(v) <= 0.8 + 1e-6 and v[3] == max(v))
        check("null emotion", tts.resolve_emotion(None)[0] is None and tts.resolve_emotion("neutral")[0] is None)
        check("explicit vector capped", abs(sum(tts.resolve_emotion([1] * 8)[0]) - 0.8) < 1e-3)
        check("unknown label -> neutral + warning", tts.resolve_emotion("wistfully-quantum")[0] is None
              and tts.resolve_emotion("wistfully-quantum")[2] is not None)
        check("combo + intensity", tts.resolve_emotion("very angry")[0][1] > tts.resolve_emotion("slightly angry")[0][1]
              and tts.resolve_emotion("shy, happy")[1] == "shy+happy")
        check("alias", tts.resolve_emotion("fearful")[1] == "scared")

        # ---------------------------------------------------------------- batch logic with the fake engine
        ref = tmp / "ref.wav"
        t = np.arange(int(4.0 * 44100)) / 44100
        tts.write_wav(ref, (0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32"), 44100)
        text = "こんにちは、きょうはいいてんきですね。"          # 17 mora -> ~2.2 s natural
        base = {"speaker": "S", "text": text, "lang": "JA", "ref_wavs": [str(ref)]}
        eng = FakeEngine()
        raw = tts.synthesize_batch([dict(base, id="probe")], tmp / "probe", engine=eng)[0]["raw_dur"]   # natural length
        items = [
            dict(base, id="no_target"),
            dict(base, id="native", target_dur=round(raw * 1.01, 3)),
            dict(base, id="speed_fast", target_dur=round(raw * 0.85, 3)),
            dict(base, id="stretch_fast", target_dur=round(raw * 0.66, 3)),
            dict(base, id="too_fast", target_dur=round(raw * 0.45, 3)),
            dict(base, id="speed_slow", target_dur=round(raw * 1.2, 3)),
            dict(base, id="stretch_slow", target_dur=round(raw * 1.5, 3)),
            dict(base, id="too_slow", target_dur=round(raw * 2.5, 3)),
            dict(base, id="max_ok", max_dur=round(raw * 1.3, 3)),
            dict(base, id="max_over", max_dur=round(raw * 0.8, 3)),
            dict(base, id="target_capped_by_max", target_dur=round(raw * 1.0, 3), max_dur=round(raw * 0.9, 3)),
            dict(base, id="reroll", text="らんなうぇいです。", seed=10),
            dict(base, id="always_long", text="ながすぎ。"),
            dict(base, id="silent_audio", text="むおん。"),
            dict(base, id="silent_line", text="……"),
            dict(base, id="missing_ref", ref_wavs=[str(tmp / "nope.wav")]),
            dict(base, id="no_refs", ref_wavs=[]),
            dict(base, id="bad_target", target_dur=-1),
            dict(base, id="emotion_vec", emotion=[0, 0, 0.8, 0, 0, 0, 0, 0], target_dur=round(raw, 3)),
            {"_parse_error": "broken json"},
        ]
        out = tmp / "out"
        rows = {r["id"]: r for r in tts.synthesize_batch([dict(i) for i in items], out, engine=eng)}
        tol = lambda r: max(float(tts.T.fit.tolerance_rel) * r["target_dur"], float(tts.T.fit.tolerance_abs))  # noqa: E731

        def fits(i):
            r = rows[i]
            return r["ok"] and r["fit_ok"] and abs(r["dur"] - r["target_dur"]) <= tol(r)

        check("no target -> fit none", rows["no_target"]["ok"] and rows["no_target"]["fit"] == "none")
        check("within tolerance -> native", fits("native") and rows["native"]["fit"] == "native", rows["native"])
        check("0.85x -> model speed", fits("speed_fast") and rows["speed_fast"]["fit"] == "speed"
              and 0.8 < rows["speed_fast"]["duration_factor"] < 0.9, rows["speed_fast"])
        check("0.66x -> speed + stretch", fits("stretch_fast") and rows["stretch_fast"]["fit"] == "stretch"
              and rows["stretch_fast"]["duration_factor"] == 0.75, rows["stretch_fast"])
        check("0.45x -> flagged, feasible range reported", rows["too_fast"]["ok"] and not rows["too_fast"]["fit_ok"]
              and rows["too_fast"]["min_feasible_dur"] > rows["too_fast"]["target_dur"], rows["too_fast"])
        check("1.2x -> model speed", fits("speed_slow") and rows["speed_slow"]["fit"] == "speed", rows["speed_slow"])
        check("1.5x -> speed + stretch", fits("stretch_slow") and rows["stretch_slow"]["fit"] == "stretch",
              rows["stretch_slow"])
        check("2.5x -> left natural", rows["too_slow"]["ok"] and not rows["too_slow"]["fit_ok"]
              and rows["too_slow"]["fit"] == "native" and abs(rows["too_slow"]["dur"] - rows["too_slow"]["raw_dur"]) < 0.05,
              rows["too_slow"])
        check("max_dur respected (already short)", rows["max_ok"]["fit"] == "native" and rows["max_ok"]["fit_ok"])
        check("max_dur respected (compressed)", rows["max_over"]["fit_ok"]
              and rows["max_over"]["dur"] <= rows["max_over"]["max_dur"] + 0.02, rows["max_over"])
        check("max_dur wins over target", rows["target_capped_by_max"]["dur"] <= rows["target_capped_by_max"]["max_dur"] + 0.02,
              rows["target_capped_by_max"])
        check("sanity re-roll", rows["reroll"]["ok"] and rows["reroll"]["takes"] >= 2 and rows["reroll"]["seed"] == 11
              and any("runaway" in w for w in rows["reroll"]["warnings"]), rows["reroll"])
        check("always too long -> ok=false but wav kept", not rows["always_long"]["ok"]
              and "sanity_failed" in rows["always_long"]["error"] and Path(rows["always_long"]["path"]).is_file(),
              rows["always_long"])
        check("silent audio -> ok=false", not rows["silent_audio"]["ok"] and "silent" in rows["silent_audio"]["error"])
        check("silent line -> no_speakable_text", not rows["silent_line"]["ok"]
              and "no_speakable_text" in rows["silent_line"]["error"])
        check("missing ref -> error row", not rows["missing_ref"]["ok"] and "missing" in rows["missing_ref"]["error"])
        check("no refs -> error row", not rows["no_refs"]["ok"])
        check("bad target -> error row", not rows["bad_target"]["ok"] and "target_dur" in rows["bad_target"]["error"])
        check("invalid JSON line -> error row", any(not r["ok"] and "invalid JSON" in str(r["error"]) for r in rows.values()))
        check("explicit vector passed through", rows["emotion_vec"]["emo_vector"][2] == 0.8)
        import soundfile as sf
        info = sf.info(rows["speed_fast"]["path"])
        check("output format", info.samplerate == 44100 and info.channels == 1 and info.subtype == "PCM_16")
        x, _ = sf.read(rows["speed_fast"]["path"], dtype="float32")
        check("peak normalised", abs(20 * np.log10(np.abs(x).max()) - float(tts.T.peak_dbfs)) < 0.2)
        res = [json.loads(l) for l in (out / tts.RESULTS_NAME).read_text().splitlines()]
        check("results file: one row per input line", len(res) == len(items))

        # ---------------------------------------------------------------- cache + merge
        calls = eng.calls
        rows2 = {r["id"]: r for r in tts.synthesize_batch([dict(i) for i in items[:11]], tmp / "out2", engine=eng)}
        check("cache: no synthesis on the rerun", eng.calls == calls and all(r.get("cached") for r in rows2.values()),
              eng.calls - calls)
        check("cache: identical audio", (out / "speed_fast.wav").read_bytes() == (tmp / "out2" / "speed_fast.wav").read_bytes())
        tts.synthesize_batch([dict(base, id="extra_line")], out, engine=eng)
        res = [json.loads(l) for l in (out / tts.RESULTS_NAME).read_text().splitlines()]
        check("results merge by id", len(res) == len(items) + 1 and res[-1]["id"] == "extra_line")
        rows3 = tts.synthesize_batch([dict(base, id="no_target", seed=5)], tmp / "out3", engine=eng)
        check("seed re-roll is a new cache entry", not rows3[0].get("cached") and rows3[0]["seed"] == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
