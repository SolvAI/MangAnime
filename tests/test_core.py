"""Unit tests for the pure-python parts of the pipeline (no GPU, no network)."""
from __future__ import annotations

import numpy as np
import pytest


def test_extract_json_variants():
    from src.llm.client import extract_json

    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('<think>hmm {"x": 0}</think> result: {"b": "c}"} trailing') == {"b": "c}"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_peak_hours():
    from datetime import datetime, timezone

    from src.llm.client import _is_peak

    assert _is_peak(datetime(2026, 11, 11, 2, 0, tzinfo=timezone.utc))        # Wednesday 02:00 UTC
    assert not _is_peak(datetime(2026, 11, 11, 5, 0, tzinfo=timezone.utc))
    assert not _is_peak(datetime(2026, 11, 14, 2, 0, tzinfo=timezone.utc))    # Saturday


def test_dp_align_monotonic_with_skips_and_merges():
    from src.align.text_align import dp_align

    S = np.full((4, 6), 0.1, np.float32)
    S[0, 1] = 0.9          # bubble 0 -> line 1
    S[1, 2] = 0.85         # bubble 1 -> lines 2 and 3 (long bubble split over two subtitles)
    S[1, 3] = 0.8
    S[3, 5] = 0.95         # bubble 2 has no match (cut by the anime)
    score, pairs = dp_align(S, 0.55)
    assert (0, 1) in pairs and (1, 2) in pairs and (1, 3) in pairs and (3, 5) in pairs
    assert all(i != 2 for i, _ in pairs)
    assert pairs == sorted(pairs)
    assert score > 1.0


def test_gap_fill_respects_order():
    from src.align.build import gap_fill

    sim = np.array([[0.2, 0.8, 0.3, 0.1], [0.1, 0.2, 0.2, 0.9], [0.9, 0.1, 0.1, 0.1]], np.float32)
    out = gap_fill(sim, 0.42)
    assert out[0] == 1 and out[1] == 3
    assert out[2] is None            # its best shot (0) would break the order


def test_ocr_text_cleanup():
    from src.manga.ocr import clean_text, collapse_repeats

    assert collapse_repeats("DOOM DOOM DOOM DOOM DOOM DOOM") == "DOOM DOOM DOOM"
    assert clean_text("EMBARRAS-\nING STORY") == "EMBARRASING STORY"
    assert clean_text("・・・・・・・・・・・・") == "・・・"


def test_chapter_detection_from_banners():
    from src.manga.build import detect_chapters

    def page(n, banners=(), texts=("some long dialogue line here", "another long dialogue line")):
        return {"volume": 1, "page": n, "ocr": list(texts), "essential": [True] * len(texts), "banners": [{"text": b} for b in banners]}

    pages = [page(1, texts=("Volume",)), page(2), page(3, ["Comm. 1 A Normal Person"]), page(4), page(5, ["Comm. 1 END"]),
             page(6, ["Comm. 2 Peaceful"]), page(7), page(8, ["Comm. 2 END"])]
    ch = detect_chapters(pages)
    assert [c["chapter"] for c in ch] == [1, 2]
    assert (ch[0]["start_page"], ch[0]["end_page"]) == (2, 5)       # prologue page 2 belongs to chapter 1
    assert (ch[1]["start_page"], ch[1]["end_page"]) == (6, 8)
    assert ch[1]["title"] == "Peaceful"


def test_wan_length_and_timeline():
    from src.generate.audio import frames_of, timeline
    from src.generate.video import wan_length

    for mode in ("twos", "threes", "smooth"):
        for sec in (0.5, 1.0, 2.3, 3.5, 5.0, 9.0):
            n = wan_length(sec, mode)
            assert (n - 1) % 4 == 0 and 17 <= n <= 81
    assert wan_length(4.0, "twos") < wan_length(4.0, "smooth")
    shots = [{"duration": 2.0}, {"duration": 3.5}, {"duration": 0.8}]
    tl = timeline(shots)
    assert tl[0][0] == 0.0 and abs(tl[1][0] - tl[0][1]) < 1e-9
    assert abs(tl[-1][1] - sum(frames_of(s["duration"]) for s in shots) / (24000 / 1001)) < 1e-9


def test_repair_ladder_never_goes_down():
    from src.qa.repair import RUNGS, next_rung, simplify

    assert next_rung("initial", None) == "reseed"
    assert next_rung("reseed", "reseed") == "rewrite_prompt"
    assert next_rung("initial", "regen_keyframe") == "regen_keyframe"
    assert next_rung("simplify", "reseed") == "fallback"
    assert next_rung("initial", None, stage="keyframe") == "regen_keyframe"
    assert RUNGS.index(next_rung("fallback", None)) == len(RUNGS) - 1
    s = simplify({"characters": ["Komi", "Tadano"], "dialogue": [{"speaker": "Tadano", "kind": "speech"}], "first_frame": "x", "action": "run"})
    assert s["characters"] == ["Tadano"] and s["camera"] == "static"


def test_director_postprocess_coerces_and_fits_dialogue():
    from src.generate.director import est_speech_sec, postprocess

    panels = [{"order": 0}, {"order": 1}]
    shots = [{"source_panels": ["P1", 7], "duration": "1.0", "shot_size": "close-up", "camera": None, "characters": ["Komi", None],
              "action": "blink", "dialogue": [{"speaker": "Tadano", "text": "Hello", "text_ja": "こみさん、おはようございます。きょうもいいてんきですね。", "kind": "SPEECH", "emotion": "weird"}]}]
    out = postprocess(5, shots, panels)
    s = out[0]
    assert s["shot_id"] == "c005_s000" and s["source_panels"] == [1] and s["characters"] == ["Komi"]
    assert s["dialogue"][0]["kind"] == "speech" and s["dialogue"][0]["emotion"] == "neutral"
    assert s["duration"] >= est_speech_sec(s["dialogue"][0]["text_ja"])       # long enough for the line


def test_prompts_mention_trigger_and_cleanup():
    from src.generate.prompts import TRIGGER, keyframe_prompt, video_prompt

    shot = {"characters": ["Komi"], "shot_size": "close-up", "camera": "static", "action": "she blinks", "first_frame": "Komi looks down",
            "dialogue": [{"speaker": "Komi", "kind": "notebook", "text": "hi"}], "location": "classroom", "time_of_day": "day"}
    kp = keyframe_prompt(shot, True, ["Komi"])
    assert "Picture 1" in kp and "Picture 2" in kp and "speech bubble" in kp
    vp = video_prompt(shot)
    assert vp.startswith(TRIGGER) and "Mouths stay closed" in vp          # notebook lines are not spoken


def test_oped_merge_and_content():
    from src.anime.oped import is_content, merge

    assert merge([[0, 10], [11, 20], [40, 50]], gap=3) == [[0, 20], [40, 50]]
    assert is_content(1, 100, 105, [(0, 90)]) and not is_content(1, 80, 90, [(0, 90)])
