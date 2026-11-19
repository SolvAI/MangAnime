"""broken.py: every failure kind on a synthetic clip; labels must be right by construction."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
from test_eval_helpers import FPS, have_ffmpeg, moving_frames, write_clip  # noqa: E402

from src.eval import broken  # noqa: E402
from src.eval import video_metrics as vm  # noqa: E402

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not found")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("src")
    a = write_clip(d / "a.mp4", moving_frames(48, hold=2, seed=0), freq=440)
    b = write_clip(d / "b.mp4", moving_frames(48, hold=1, seed=5, speed=7.0), freq=880)
    c = write_clip(d / "c.mp4", moving_frames(36, hold=3, seed=9), freq=660)
    return d, a, b, c


def _tech(row):
    return vm.technical_check(row["path"], expected_dur=row["expected_dur"], need_audio=row["need_audio"],
                              allow_static=row["allow_static"])


def test_kind_table_is_complete():
    required = {"flicker", "frame_shuffle", "freeze", "melt", "ghost", "noise_blur", "color_shift", "wrong_audio",
                "silent_audio", "truncated", "corrupt_file", "garbled_text_overlay"}
    assert required <= set(broken.KINDS)
    for k in broken.KINDS:
        assert {"desc", "needs", "hard_fail", "evidence"} <= set(broken.KIND_INFO[k])


def test_make_real_passes_technical(src, tmp_path):
    _, a, _, _ = src
    row = broken.make_real(a, tmp_path / "real.mp4")
    assert row["label"] == "real" and row["kind"] == "none" and row["need_audio"] and not row["allow_static"]
    assert abs(row["expected_dur"] - 2.0) < 1e-6
    assert _tech(row)["ok"]


@pytest.mark.parametrize("kind", broken.KINDS)
def test_every_kind(kind, src, tmp_path):
    _, a, b, _ = src
    kw = {"other": b} if "other" in broken.KIND_INFO[kind]["needs"] else {}
    row = broken.make_broken(a, kind, tmp_path / f"{kind}.mp4", seed=3, **kw)
    assert row["label"] == "broken" and row["kind"] == kind and row["src"] == str(a)
    assert (tmp_path / f"{kind}.mp4").exists()
    assert broken.effect_ok(row), row
    tc = _tech(row)
    if broken.KIND_INFO[kind]["hard_fail"]:
        assert row["hard_fail"] and not tc["ok"], (kind, tc)
    else:
        assert not row["hard_fail"] and tc["ok"], (kind, tc)
        out = vm.load_frames(row["path"])
        assert out.shape == vm.load_frames(a).shape           # same length and size as the source
    if kind in broken._PIXEL_KINDS:
        assert row["effect_mean"] > 0


def test_expected_failure_codes(src, tmp_path):
    _, a, b, _ = src

    def codes(kind, **kw):
        row = broken.make_broken(a, kind, tmp_path / f"x_{kind}_{len(kw)}.mp4", **kw)
        return [r.split(":")[0] for r in _tech(row)["reasons"]]

    assert codes("freeze") == ["frozen_video"]
    assert codes("black") == ["black_video"]
    assert codes("silent_audio") == ["silent_audio"]
    assert codes("silent_audio", drop=True) == ["no_audio_stream"]
    assert codes("truncated", keep=0.4) == ["wrong_duration"]
    assert "unreadable" in codes("corrupt_file", mode="cut")
    assert "unreadable" in codes("corrupt_file", mode="header")
    assert codes("corrupt_file", mode="empty") == ["empty_file"]
    assert "decode_errors" in codes("corrupt_file", mode="garble") or "unreadable" in codes("corrupt_file", mode="garble")


def test_partial_freeze_is_not_a_hard_fail(src, tmp_path):
    _, a, _, _ = src
    row = broken.make_broken(a, "freeze", tmp_path / "pf.mp4", start_frac=0.4)
    assert not row["hard_fail"]
    tc = _tech(row)
    assert tc["ok"] and tc["info"]["longest_static_run_s"] > 1.0


def test_metrics_see_the_damage(src, tmp_path):
    _, a, b, _ = src
    base = vm.load_frames(a)
    f0 = vm.flicker_score(base, fps=FPS)

    def frames(kind, **kw):
        return vm.load_frames(broken.make_broken(a, kind, tmp_path / f"m_{kind}.mp4", seed=1, **kw)["path"])

    assert vm.flicker_score(frames("flicker"))["luma_flicker"] > 5 * max(f0["luma_flicker"], 0.05)
    assert vm.flicker_score(frames("freeze"))["repeat_ratio"] > 0.97
    assert vm.flicker_score(frames("jitter"))["repeat_ratio"] < 0.1 < f0["repeat_ratio"]
    assert vm.flicker_score(frames("noise_blur"))["repeat_ratio"] < 0.1
    assert len(vm.detect_cuts(frames("hard_cut", other=b))) == 1
    assert vm.detect_cuts(base) == []


def test_wrong_audio_uses_the_other_track(src, tmp_path):
    _, a, b, _ = src
    row = broken.make_broken(a, "wrong_audio", tmp_path / "wa.mp4", other=b)
    assert _tech(row)["ok"]                                   # technically fine: only the content is wrong

    def dominant(path):
        import subprocess

        raw = subprocess.run([vm.ffmpeg_bin(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                             capture_output=True).stdout
        x = np.frombuffer(raw, np.int16).astype(np.float32)
        return float(np.fft.rfftfreq(len(x), 1 / 8000)[np.abs(np.fft.rfft(x)).argmax()])

    assert abs(dominant(a) - 440) < 15 and abs(dominant(row["path"]) - 880) < 15
    with pytest.raises(ValueError):
        broken.make_broken(a, "wrong_audio", tmp_path / "wa2.mp4")


def test_static_source_cannot_show_a_freeze(src, tmp_path):
    from test_eval_helpers import background

    d, _, _, _ = src
    still = write_clip(tmp_path / "still.mp4", np.repeat(background(2)[None], 36, 0))
    row = broken.make_broken(still, "freeze", tmp_path / "still_freeze.mp4")
    assert row["allow_static"] and not broken.effect_ok(row)   # nothing changed -> not a valid broken example


def test_unknown_kind_and_bad_args(src, tmp_path):
    _, a, _, _ = src
    with pytest.raises(ValueError):
        broken.make_broken(a, "nope", tmp_path / "x.mp4")
    with pytest.raises(TypeError):
        broken.make_broken(a, "truncated", tmp_path / "x.mp4", bogus=1)


def test_build_set_labels(src, tmp_path):
    d, a, b, c = src
    out = tmp_path / "set"
    kinds = ["flicker", "freeze", "truncated", "silent_audio", "wrong_audio", "corrupt_file"]
    rows = broken.build_set(d, out, n_broken=6, n_real=3, kinds=kinds, seed=0, verbose=False)
    lab = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines()]
    assert len(lab) == len(rows) == 9
    assert sum(r["label"] == "real" for r in lab) == 3
    assert sorted(r["kind"] for r in lab if r["label"] == "broken") == sorted(kinds)
    for r in lab:
        assert {"path", "label", "kind", "src", "expected_dur", "need_audio", "allow_static", "hard_fail",
                "technical_ok"} <= set(r)
        assert "flicker" not in r["path"] and "real" not in r["path"].split("/")[-1]       # opaque file names
        if r["label"] == "real":
            assert r["technical_ok"]
        if r["hard_fail"]:
            assert not r["technical_ok"]
    # the CLI entry point writes the same structure
    assert broken.main(["--src", str(d), "--out", str(tmp_path / "set2"), "--n", "2", "--n-real", "1",
                        "--kinds", "black,ghost", "--seed", "1"]) == 0
    assert len((tmp_path / "set2" / "labels.jsonl").read_text().splitlines()) == 3
    assert broken.main(["--list-kinds"]) == 0
