"""A/B page: statistics, DuckDB round trip, blindness of the served page, resume."""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("duckdb")
from tools.ab_page import stats as S  # noqa: E402


def test_binomial_helpers_match_known_values():
    assert S.sign_test_p(0, 10) == pytest.approx(2 / 1024)
    assert S.sign_test_p(3, 10) == pytest.approx(0.34375)
    assert S.sign_test_p(5, 10) == 1.0 and S.sign_test_p(0, 0) == 1.0
    assert S.sign_test_p(70, 100) == pytest.approx(7.85e-5, rel=0.02)
    lo, hi = S.binom_ci(14, 20)
    assert lo == pytest.approx(0.45721, abs=1e-4) and hi == pytest.approx(0.88107, abs=1e-4)
    assert S.binom_ci(0, 10)[0] == 0.0 and S.binom_ci(0, 10)[1] == pytest.approx(0.30850, abs=1e-4)
    assert S.binom_ci(10, 10) == (pytest.approx(0.69150, abs=1e-4), 1.0)
    assert S.binom_ci(0, 0) == (0.0, 1.0)
    wl, wh = S.wilson_ci(31, 50)
    assert wl == pytest.approx(0.4815, abs=1e-3) and wh == pytest.approx(0.7414, abs=1e-3)


def test_record_and_summarize_round_trip(tmp_path):
    db = tmp_path / "sub" / "r.duckdb"
    exp = "ep23_wan_vs_animatic"
    # 10 pairs, two humans + one judge; wan wins "overall" 8:1 (+1 tie) for alice
    for i in range(10):
        ch = "a" if i < 8 else ("tie" if i == 8 else "b")
        S.record_rating(db, exp, f"s{i:03d}", "alice", {"overall": ch, "motion": "b"}, "wan", "animatic",
                        left_side="a" if i % 2 else "b", duration_ms=1000 + i)
        S.record_rating(db, exp, f"s{i:03d}", "bob", {"overall": "wan", "motion": "tie"}, "wan", "animatic", left_side="a")
        S.record_rating(db, exp, f"s{i:03d}", "judge:deepseek", {"overall": "a" if i < 5 else "b"}, "wan", "animatic")
    S.record_rating(db, "other_experiment", "s000", "alice", {"overall": "b"}, "wan", "animatic")
    s = S.summarize(db, exp)
    assert s["n_ratings"] == 30 and s["n_pairs_rated"] == 10
    assert s["raters"] == {"alice": 10, "bob": 10, "judge:deepseek": 10}
    assert S.is_automatic("metric:dover") and S.is_automatic("judge:x") and not S.is_automatic("alice")
    hum = s["groups"]["human"]["criteria"]["overall"][0]
    assert (hum["label_x"], hum["label_y"]) == ("wan", "animatic")
    assert (hum["wins_x"], hum["wins_y"], hum["ties"], hum["n"]) == (18, 1, 1, 20)
    assert hum["win_rate_x"] == pytest.approx(18 / 19) and hum["winner"] == "wan" and hum["significant"]
    assert hum["p_sign"] == pytest.approx(S.sign_test_p(18, 19))
    assert hum["ci95"][0] == pytest.approx(S.binom_ci(18, 19)[0]) and hum["ci95"][0] > 0.5
    mot = s["groups"]["human"]["criteria"]["motion"][0]
    assert (mot["wins_x"], mot["wins_y"], mot["ties"]) == (0, 10, 10) and mot["winner"] == "animatic"
    jud = s["groups"]["judge"]["criteria"]["overall"][0]
    assert (jud["wins_x"], jud["wins_y"]) == (5, 5) and not jud["significant"] and jud["winner"] is None
    assert s["groups"]["all"]["criteria"]["overall"][0]["n"] == 30
    # metric-based raters are automatic too: they must never be counted as humans
    S.record_rating(db, exp, "s000", "metric:dover", {"overall": "b"}, "wan", "animatic")
    s2 = S.summarize(db, exp)
    assert s2["groups"]["human"]["criteria"]["overall"][0]["n"] == 20 and s2["groups"]["judge"]["n_raters"] == 2
    S.delete_rating(db, [r for r in S.fetch_ratings(db, exp) if r["rater"] == "metric:dover"][0]["rating_id"])
    maj = s["groups"]["human"]["criteria_by_pair_majority"]["overall"][0]
    # per pair: 0..7 both say wan; 8 = tie + wan -> wan; 9 = animatic + wan -> no majority
    assert (maj["n"], maj["wins_x"], maj["wins_y"], maj["ties"]) == (10, 9, 0, 1)
    assert s["by_rater"]["bob"]["overall"][0]["wins_x"] == 10
    # the judge agrees with the human majority on pairs 0..4 and disagrees on 5..8; pair 9 has no majority
    ag = s["judge_agreement"]["judge:deepseek"]
    assert (ag["n"], ag["agree"]) == (9, 5) and ag["rate"] == pytest.approx(5 / 9)
    pb = s["position_bias"]
    assert pb["left"] + pb["right"] + pb["ties"] == 40
    json.dumps(s, default=str)                                                   # serialisable


def test_latest_rating_wins_and_validation(tmp_path):
    db = tmp_path / "r.duckdb"
    S.record_rating(db, "e", "p1", "ann", {"overall": "a"}, "x", "y")
    rid = S.record_rating(db, "e", "p1", "ann", {"overall": "b"}, "x", "y")
    c = S.summarize(db, "e")["groups"]["all"]["criteria"]["overall"][0]
    assert (c["wins_x"], c["wins_y"], c["n"]) == (0, 1, 1)
    S.delete_rating(db, rid)
    c = S.summarize(db, "e")["groups"]["all"]["criteria"]["overall"][0]
    assert (c["wins_x"], c["wins_y"]) == (1, 0)
    assert S.rated_pairs(db, "e", "ann") == {"p1"} and S.rated_pairs(db, "e", "zoe") == set()
    for bad in ({"overall": "left"}, {}):
        with pytest.raises(ValueError):
            S.record_rating(db, "e", "p2", "ann", bad, "x", "y")
    with pytest.raises(ValueError):
        S.record_rating(db, "e", "p2", "ann", {"overall": "a"}, "x", "x")
    with pytest.raises(ValueError):
        S.record_rating(db, "e", "p2", "", {"overall": "a"}, "x", "y")
    assert S.summarize(tmp_path / "missing.duckdb", "e")["n_ratings"] == 0


def test_three_way_experiment_keeps_comparisons_apart(tmp_path):
    db = tmp_path / "r.duckdb"
    for i in range(6):
        S.record_rating(db, "e", f"a{i}", "ann", {"overall": "a"}, "wan", "animatic")
        S.record_rating(db, "e", f"b{i}", "ann", {"overall": "b"}, "wan", "real")
    comps = S.summarize(db, "e")["groups"]["human"]["criteria"]["overall"]
    got = {(c["label_x"], c["label_y"]): (c["wins_x"], c["wins_y"]) for c in comps}
    assert got == {("wan", "animatic"): (6, 0), ("wan", "real"): (0, 6)}


# ---------------------------------------------------------------------------------------------------------------
# web app
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tools.ab_page import app as A

    vids = tmp_path / "vids"
    vids.mkdir()
    pairs = []
    for i in range(4):
        a, b = vids / f"wan_secret_{i}.mp4", vids / f"animatic_secret_{i}.mp4"
        a.write_bytes(b"A" * (1000 + i))
        b.write_bytes(b"B" * (2000 + i))
        pairs.append({"pair_id": f"shot{i:03d}", "a": str(a), "b": str(b), "label_a": "wan", "label_b": "animatic",
                      "meta": {"prompt": "hidden meta"}})
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"experiment": "t_exp", "criteria": ["overall", "motion"], "pairs": pairs}))
    db = tmp_path / "ab.duckdb"
    app = A.create_app(A.load_manifest(man), db, transcode="never")
    return TestClient(app), app, db


def _page(c):
    r = c.get("/rate")
    assert r.status_code == 200
    pres = re.search(r'name="pres" value="([^"]+)"', r.text)
    return r.text, (pres.group(1) if pres else None), re.findall(r"/media/([A-Za-z0-9_\-]+)", r.text)


def test_app_is_blind_and_records_sides(client):
    c, app, db = client
    assert c.get("/", follow_redirects=False).status_code == 200
    assert c.get("/rate", follow_redirects=False).status_code == 303           # no name yet
    assert c.post("/login", data={"rater": "judge:fake"}, follow_redirects=False).status_code == 200   # reserved prefix
    assert c.post("/login", data={"rater": "Metric:dover"}, follow_redirects=False).status_code == 200
    assert c.post("/login", data={"rater": "  Lin  "}, follow_redirects=False).status_code == 303
    seen_left = set()
    for n in range(4):
        html, pres, toks = _page(c)
        assert len(toks) == 2 and pres
        for leak in ("wan", "animatic", "secret", ".mp4", "shot0", "hidden meta", "label_a"):
            assert leak not in html, leak
        st = app.state.open_presentations[pres]
        seen_left.add(st["left_side"])
        left = c.get(f"/media/{toks[0]}")
        right = c.get(f"/media/{toks[1]}")
        assert left.content[:1] == (b"A" if st["left_side"] == "a" else b"B")
        assert right.content[:1] == (b"B" if st["left_side"] == "a" else b"A")
        assert "content-disposition" not in left.headers and "etag" not in left.headers
        part = c.get(f"/media/{toks[0]}", headers={"Range": "bytes=10-19"})
        assert part.status_code == 206 and len(part.content) == 10 and part.headers["content-range"].startswith("bytes 10-19/")
        assert c.get(f"/media/{toks[0]}", headers={"Range": "bytes=999999-"}).status_code == 416
        # reloading the page must not reshuffle the sides
        html2, pres2, toks2 = _page(c)
        assert pres2 == pres and toks2 == toks
        # the rater always prefers the WAN clip on "overall", wherever it is shown
        wan_side = "left" if st["left_side"] == "a" else "right"
        r = c.post("/rate", data={"pres": pres, "c0": wan_side, "c1": "tie", "duration_ms": "1500"}, follow_redirects=False)
        assert r.status_code == 303
        assert c.get(f"/media/{toks[0]}").status_code == 404                     # tokens die with the rating
    assert "All done" in c.get("/rate").text
    s = S.summarize(db, "t_exp")
    ov = s["groups"]["human"]["criteria"]["overall"][0]
    assert (ov["label_x"], ov["wins_x"], ov["wins_y"], ov["ties"]) == ("wan", 4, 0, 0)
    assert s["groups"]["human"]["criteria"]["motion"][0]["ties"] == 4
    assert s["raters"] == {"Lin": 4}
    rows = S.fetch_ratings(db, "t_exp")
    assert {r["left_side"] for r in rows} == seen_left and all(r["duration_ms"] == 1500 for r in rows)
    res = c.get("/results")
    assert res.status_code == 200 and "wan" in res.text and "animatic" in res.text
    assert c.get("/api/results").json()["n_ratings"] == 4


def test_app_resume_skip_undo_and_incomplete(client):
    c, app, db = client
    c.post("/login", data={"rater": "ann"})
    html, pres, toks = _page(c)
    first = app.state.open_presentations[pres]["pair_id"]
    # incomplete form: nothing stored, same pair again
    assert c.post("/rate", data={"pres": pres, "c0": "left"}, follow_redirects=False).status_code == 303
    assert S.fetch_ratings(db, "t_exp") == []
    # skip pushes the pair to the end
    c.post("/skip", data={"pres": pres})
    html, pres, toks = _page(c)
    second = app.state.open_presentations[pres]["pair_id"]
    assert second != first
    c.post("/rate", data={"pres": pres, "c0": "left", "c1": "right"})
    assert c.get("/api/progress").json() == {"rater": "ann", "done": 1, "total": 4}
    # a new browser session with the same name resumes where it stopped
    from fastapi.testclient import TestClient

    c2 = TestClient(app)
    c2.post("/login", data={"rater": "ann"})
    html, pres, toks = _page(c2)
    assert app.state.open_presentations[pres]["pair_id"] not in (second,)
    assert c2.get("/api/progress").json()["done"] == 1
    # stale presentation tokens are ignored
    assert c2.post("/rate", data={"pres": "bogus", "c0": "left", "c1": "left"}, follow_redirects=False).status_code == 303
    assert len({r["rating_id"] for r in S.fetch_ratings(db, "t_exp")}) == 1
    # undo removes the last rating of that rater
    c.post("/undo")
    assert S.fetch_ratings(db, "t_exp") == []


def test_manifest_validation(tmp_path):
    pytest.importorskip("fastapi")
    from tools.ab_page import app as A

    def load(obj):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(obj))
        return A.load_manifest(p)

    ok = {"experiment": "e", "pairs": [{"pair_id": 1, "a": "/x", "b": "/y", "label_a": "p", "label_b": "q"}]}
    m = load(ok)
    assert m["criteria"] == ["overall"] and m["pairs"][0]["pair_id"] == "1"
    for bad in ({"pairs": ok["pairs"]}, {"experiment": "e", "pairs": []},
                {"experiment": "e", "pairs": [dict(ok["pairs"][0], label_b="p")]},
                {"experiment": "e", "pairs": ok["pairs"] * 2},
                {"experiment": "e", "criteria": ["a", "a"], "pairs": ok["pairs"]},
                {"experiment": "e", "pairs": [{"pair_id": 1, "a": "/x"}]}):
        with pytest.raises(ValueError):
            load(bad)
