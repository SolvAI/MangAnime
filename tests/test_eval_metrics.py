"""video_metrics / fvd: pure-python + CPU-only parts (no model weights needed)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
from test_eval_helpers import FPS, H, W, background, have_ffmpeg, moving_frames, write_clip  # noqa: E402

from src.eval import video_metrics as vm  # noqa: E402

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not found")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    out = {
        "ones": write_clip(d / "ones.mp4", moving_frames(48, hold=1)),
        "twos": write_clip(d / "twos.mp4", moving_frames(48, hold=2)),
        "static": write_clip(d / "static.mp4", np.repeat(background(3)[None], 48, 0)),
        "noaudio": write_clip(d / "noaudio.mp4", moving_frames(48), audio=False),
        "black": write_clip(d / "black.mp4", np.zeros((48, H, W, 3), np.uint8)),
    }
    cut = np.concatenate([moving_frames(24, seed=0), moving_frames(24, seed=7)[:, ::-1, ::-1]])
    out["cut"] = write_clip(d / "cut.mp4", cut)
    return out


def test_load_frames_shapes(clips):
    fr, fps = vm.load_frames(clips["ones"], return_fps=True)
    assert fr.shape == (48, H, W, 3) and fr.dtype == np.uint8 and abs(fps - FPS) < 1e-3
    assert vm.load_frames(clips["ones"], max_frames=5).shape[0] == 5
    assert vm.load_frames(clips["ones"], size=88).shape[1:3] == (88, 160)
    assert vm.load_frames(clips["ones"], size=(64, 48)).shape[1:3] == (48, 64)
    half = vm.load_frames(clips["ones"], fps=12)
    assert half.shape[0] == 24
    double = vm.load_frames(clips["ones"], fps=48)
    assert double.shape[0] == 96 and np.array_equal(double[0], double[1])
    with pytest.raises(Exception):
        vm.load_frames(clips["ones"].with_suffix(".nope"))


def test_video_info_and_listing(clips):
    info = vm.video_info(clips["ones"])
    assert info["has_video"] and info["has_audio"] and info["width"] == W and abs(info["fps"] - FPS) < 1e-3
    assert not vm.video_info(clips["noaudio"])["has_audio"]
    assert len(vm.list_clips([clips["ones"].parent])) == len(clips)


def test_flicker_held_frames(clips):
    ones = vm.flicker_score(vm.load_frames(clips["ones"]), fps=FPS)
    twos = vm.flicker_score(vm.load_frames(clips["twos"]), fps=FPS)
    static = vm.flicker_score(vm.load_frames(clips["static"]), fps=FPS)
    assert ones["repeat_ratio"] < 0.05 and ones["effective_fps"] > 22
    assert 0.4 < twos["repeat_ratio"] < 0.6 and 11 < twos["effective_fps"] < 13.5
    assert twos["hold_hist"]["2"] > 0.8 and 1.8 < twos["mean_hold"] < 2.2
    assert static["repeat_ratio"] > 0.97 and static["unique_frames"] <= 2
    # the on-twos clip must not look like flicker once held frames are removed
    assert twos["luma_flicker"] < 1.0 and twos["hf_energy_unique"] <= ones["hf_energy_unique"] * 2.5 + 0.5


def test_flicker_detects_brightness_pumping(clips):
    fr = vm.load_frames(clips["ones"]).astype(np.float32)
    gains = np.where(np.arange(len(fr)) % 2 == 0, 1.0, 0.6)[:, None, None, None]
    bad = np.clip(fr * gains, 0, 255).astype(np.uint8)
    a, b = vm.flicker_score(vm.load_frames(clips["ones"])), vm.flicker_score(bad)
    assert b["luma_flicker"] > 10 * max(a["luma_flicker"], 0.05)


def test_detect_cuts(clips):
    assert vm.detect_cuts(vm.load_frames(clips["ones"])) == []
    assert vm.detect_cuts(vm.load_frames(clips["twos"])) == []
    assert vm.detect_cuts(vm.load_frames(clips["cut"])) == [24]


def test_flow_metrics_farneback(clips):
    mov = vm.load_frames(clips["ones"])
    sta = vm.load_frames(clips["static"])
    fs = vm.flow_stats(mov, method="farneback", height=176)
    assert len(fs["warp_l1"]) == len(mov) - 1
    m_mov = vm.motion_stats(mov, fps=FPS, flow=fs)
    m_sta = vm.motion_stats(sta, method="farneback", height=176)
    assert m_mov["mean_flow_mag"] > 0.05 and m_mov["static_frame_ratio"] < 0.2 and m_mov["n_hard_cuts"] == 0
    assert m_sta["mean_flow_mag"] < 0.01 and m_sta["static_frame_ratio"] > 0.95
    w_mov = vm.warp_error(mov, flow=fs, return_details=True)
    assert isinstance(vm.warp_error(mov, flow=fs), float)
    assert 0 <= w_mov["warp_error"] < 20 and w_mov["n_pairs"] == len(mov) - 1
    # motion compensation must explain most of the change of a rigidly moving square
    assert w_mov["residual_ratio"] is not None and w_mov["residual_ratio"] < 0.9
    assert vm.warp_error(sta, method="farneback", height=176, return_details=True)["residual_ratio"] is None
    # shuffled frames are harder to explain by flow than the ordered clip
    rng = np.random.default_rng(0)
    shuf = mov[rng.permutation(len(mov))]
    assert vm.warp_error(shuf, method="farneback", height=176) > w_mov["warp_error"]
    cut = vm.motion_stats(vm.load_frames(clips["cut"]), method="farneback", height=176)
    assert cut["hard_cuts"] == [24]


def test_technical_check_ok_and_failures(clips, tmp_path):
    ok = vm.technical_check(clips["ones"], expected_dur=2.0, need_audio=True)
    assert ok["ok"] and ok["reasons"] == [] and ok["info"]["frames_decoded"] == 48

    def codes(r):
        return [x.split(":")[0] for x in r["reasons"]]

    assert codes(vm.technical_check(clips["ones"], expected_dur=5.0)) == ["wrong_duration"]
    assert vm.technical_check(clips["ones"], expected_dur=2.2, dur_tol=0.5)["ok"]
    assert codes(vm.technical_check(clips["noaudio"], need_audio=True)) == ["no_audio_stream"]
    assert vm.technical_check(clips["noaudio"])["ok"]
    assert codes(vm.technical_check(clips["black"])) == ["black_video"]
    assert codes(vm.technical_check(clips["static"])) == ["frozen_video"]
    st = vm.technical_check(clips["static"], allow_static=True)
    assert st["ok"] and any(w.startswith("frozen_video") for w in st["warnings"])
    assert vm.technical_check(clips["twos"], need_audio=True)["ok"]          # held frames are not a freeze
    assert codes(vm.technical_check(tmp_path / "missing.mp4")) == ["missing_file"]
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert codes(vm.technical_check(empty)) == ["empty_file"]
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    assert codes(vm.technical_check(junk)) == ["unreadable"]
    assert codes(vm.technical_check(clips["ones"], expected_fps=16)) == ["wrong_fps"]
    assert codes(vm.technical_check(clips["ones"], expected_size=(640, 360))) == ["wrong_size"]


def test_technical_check_silent_audio(clips, tmp_path):
    from src.eval import broken

    fr = vm.load_frames(clips["ones"])
    silent = broken.write_video(fr, tmp_path / "silent.mp4", FPS, silent=True)
    r = vm.technical_check(silent, need_audio=True)
    assert [x.split(":")[0] for x in r["reasons"]] == ["silent_audio"]
    r2 = vm.technical_check(silent)
    assert r2["ok"] and any(w.startswith("silent_audio") for w in r2["warnings"])


def test_analyze_clip_and_cli(clips, tmp_path):
    r = vm.analyze_clip(clips["twos"], metrics=("technical", "flicker", "motion", "warp"), flow="farneback",
                        flow_height=176, need_audio=True)
    assert r["technical"]["ok"] and "repeat_ratio" in r["flicker"] and "mean_flow_mag" in r["motion"]
    assert "warp_error" in r["warp"] and r["n_frames"] == 48
    # alias keys that episode_eval / ab_judge read
    assert r["flicker"]["held_frame_ratio"] == r["flicker"]["repeat_ratio"] and "flicker_index" in r["flicker"]
    assert r["motion"]["mean_flow"] == r["motion"]["mean_flow_mag"] and r["motion"]["static_ratio"] == r["motion"]["static_frame_ratio"]
    assert r["warp"]["mean"] == r["warp"]["warp_error"]
    out = tmp_path / "m.json"
    assert vm.main([str(clips["ones"]), str(clips["black"]), "--metrics", "technical,flicker", "--out", str(out), "--quiet"]) == 0
    import json

    d = json.loads(out.read_text())["clips"]
    assert d[str(clips["ones"])]["technical"]["ok"] and not d[str(clips["black"])]["technical"]["ok"]
    bad = vm.analyze_clip(tmp_path / "nope.mp4", metrics=("technical", "flicker"))
    assert not bad["technical"]["ok"] and "flicker" not in bad


def test_pick_device_respects_env(monkeypatch):
    monkeypatch.setenv("KOMI_EVAL_DEVICE", "cpu")
    assert vm.pick_device(None) == "cpu"
    assert vm.pick_device("cuda") == "cuda"          # explicit choice is never overridden
    monkeypatch.delenv("KOMI_EVAL_DEVICE")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert vm.free_vram_gb() == 0.0 and vm.pick_device("auto") == "cpu"


def test_frechet_and_kernel_distance():
    from src.eval import fvd

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (600, 16))
    b = rng.normal(0, 1, (600, 16))
    c = rng.normal(0.5, 1, (600, 16))
    same, shifted = fvd.frechet_distance(a, b), fvd.frechet_distance(a, c)
    assert same < 0.5 and 3.0 < shifted < 5.5            # expected ||mu||^2 = 16 * 0.25 = 4
    assert abs(fvd.frechet_distance(a, a)) < 1e-6
    # rank-deficient covariance (fewer samples than dimensions) must still give a finite real number
    small = fvd.frechet_distance(rng.normal(0, 1, (8, 32)), rng.normal(0, 1, (8, 32)))
    assert np.isfinite(small) and small > 0
    assert fvd.kernel_distance(a, c) > 10 * abs(fvd.kernel_distance(a, b))
    with pytest.raises(ValueError):
        fvd.frechet_distance(a[:1], b)


def test_fvd_clip_windows(clips):
    from src.eval import fvd

    w = fvd.clip_windows(clips["ones"], fps=16, n_frames=16, size=64, max_windows=4)
    assert w.shape == (2, 16, 64, 64, 3)                  # 2 s at 16 fps = 32 frames = 2 windows
    assert fvd.clip_windows(clips["ones"], fps=16, n_frames=16, size=64, max_windows=1).shape[0] == 1
    assert fvd.clip_windows(clips["ones"], fps=16, n_frames=16, size=64, resize="squash").shape[2:4] == (64, 64)
    short = fvd.clip_windows(clips["ones"], fps=4, n_frames=16, size=64)      # 8 frames -> padded to one window
    assert short.shape[0] == 1
