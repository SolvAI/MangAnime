"""Review page and alignment labeling page on synthetic fixtures (and with the data files missing)."""
from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient  # noqa: E402


def test_review_page_round_trip(tmp_path):
    from tools.review import app as R
    from tools.review import fixture

    fx = fixture.make_fixture(tmp_path / "fx")
    c = TestClient(R.create_app(R.default_sources(fx)))
    assert "characters" in c.get("/").text
    page = c.get("/s/characters").text
    ids = [x.split('"')[0] for x in page.split('data-id="')[1:]]
    assert ids == ["4", "2", "3", "1", "0"]                     # unknown confidence first, then ascending
    assert c.get("/s/speakers").text.count("<audio") == 9
    th = c.get("/media/characters/0/0?w=64")
    assert th.status_code == 200 and th.headers["content-type"] == "image/jpeg"
    assert c.get("/media/speakers/0/0").headers["content-type"] == "audio/wav"
    assert c.get("/media/characters/99/0").status_code == 404
    assert c.post("/api/characters/name", json={"id": "2", "name": "  Yamai   Ren "}).json()["override"] == "Yamai Ren"
    assert c.post("/api/characters/name", json={"id": "0", "name": "Komi"}).json()["n_overrides"] == 2
    assert c.post("/api/characters/name", json={"id": "77", "name": "x"}).status_code == 400
    ov = json.loads((fx / "characters" / "names_override.json").read_text())
    assert ov == {"0": "Komi", "2": "Yamai Ren"}
    assert c.post("/api/characters/name", json={"id": "0", "name": ""}).json()["override"] is None
    assert json.loads((fx / "characters" / "names_override.json").read_text()) == {"2": "Yamai Ren"}
    assert not (fx / "anime" / "spk" / "names_override.json").exists()
    assert c.get("/s/characters?show=todo").text.count('data-id="') == 4
    assert c.get("/s/nope").status_code == 404


def test_review_tolerates_odd_schemas_and_missing_files(tmp_path):
    from tools.review import app as R

    p = tmp_path / "clusters.json"
    p.write_text(json.dumps([{"name": "A", "exemplars": ["rel/a.png", {"wav": "b.wav", "transcript": "hi"}]},
                             {"id": "x7", "conf": "0.2", "images": ["c.jpg"]}, "garbage", {"id": 5}]))
    cl = R.load_clusters(p)
    assert [c["id"] for c in cl] == ["0", "x7", "5"]
    assert cl[0]["exemplars"][0]["kind"] == "image" and cl[0]["exemplars"][0]["path"] == tmp_path / "rel/a.png"
    assert cl[0]["exemplars"][1]["kind"] == "audio" and cl[0]["exemplars"][1]["text"] == "hi"
    assert cl[1]["confidence"] == 0.2 and cl[2]["exemplars"] == []
    c = TestClient(R.create_app({"odd": p, "characters": tmp_path / "missing" / "clusters.json"}))
    assert c.get("/s/odd").status_code == 200 and "missing file" in c.get("/s/odd").text
    assert "not created yet" in c.get("/").text
    assert "File not found" in c.get("/s/characters").text
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    c = TestClient(R.create_app({"bad": bad}))
    assert c.get("/").status_code == 200 and "Could not read" in c.get("/s/bad").text


def test_align_label_round_trip(tmp_path):
    from PIL import Image

    from tools.align_label import app as L
    from tools.align_label import fixture

    fx = fixture.make_fixture(tmp_path / "fx")
    c = TestClient(L.create_app(fx))
    assert "ep01" in c.get("/").text
    d = c.get("/api/ep/1").json()
    assert d["chapters"] == [1, 2] and len(d["panels"]) == 16 and len(d["shots"]) == 24 and d["gold"] == {}
    assert [p["key"] for p in d["panels"][:3]] == ["1:1", "1:2", "1:3"]         # reading order
    assert d["auto"]["1:1"]["shots"] == [1, 2] and "1:3" not in d["auto"]
    im = Image.open(io.BytesIO(c.get(f"/img/panel/{d['panels'][0]['idx']}?h=150").content))
    assert im.size[1] == 150 and im.size[0] < 150                               # the panel was cropped out of the page
    assert c.get("/img/shot/1/3/1").status_code == 200 and c.get("/img/shot/1/3/9").status_code == 404
    rows = [{"key": "1:1", "shots": [2, 1], "status": "confirmed"}, {"key": "1:2", "shots": [4], "status": "no_match"},
            {"key": "1:3", "shots": [], "status": "confirmed"}, {"key": "1:4", "shots": [5], "status": "auto"},
            {"key": "bad", "shots": [1], "status": "confirmed"}]
    assert c.post("/api/ep/1/save", json={"rows": rows}).json()["n"] == 2
    gold = [json.loads(l) for l in (fx / "align" / "gold" / "ep01.jsonl").read_text().splitlines()]
    assert gold[0] == {**gold[0], "chapter": 1, "panel_order": 1, "match": "manual", "confidence": 1.0,
                       "shots": [{"episode": 1, "shot": 1}, {"episode": 1, "shot": 2}], "status": "confirmed"}
    assert gold[1]["shots"] == [] and gold[1]["status"] == "no_match" and gold[1]["auto_shots"] == [2]
    assert c.get("/api/ep/1").json()["gold"] == {"1:1": {"shots": [1, 2], "status": "confirmed"},
                                                 "1:2": {"shots": [], "status": "no_match"}}
    assert c.get("/api/ep/1?chapters=2").json()["chapters"] == [2]
    assert c.post("/api/ep/1/save", json={"nope": 1}).status_code == 400
    assert c.get("/ep/1").status_code == 200


def test_align_label_without_data(tmp_path):
    from tools.align_label import app as L

    c = TestClient(L.create_app(tmp_path / "nothing"))
    assert c.get("/").status_code == 200 and "missing" in c.get("/").text
    d = c.get("/api/ep/3").json()
    assert d["panels"] == [] and d["shots"] == [] and d["chapters"] == []
    assert c.get("/ep/3").status_code == 200 and c.get("/img/panel/0").status_code == 404
    assert L.parse_chapters("12-15, 17,x,3-1") == [1, 2, 3, 12, 13, 14, 15, 17]
    # queue formats
    g = tmp_path / "q" / "align" / "gold"
    g.mkdir(parents=True)
    for q, want in (([1, 2], [1, 2]), ({"episodes": ["ep03", 4]}, [3, 4]),
                    ([{"episode": 5, "chapters": [60, 61]}], [5])):
        (g / "queue.json").write_text(json.dumps(q))
        data = L.Data(tmp_path / "q")
        assert [e["episode"] for e in data.queue()] == want
    assert L.Data(tmp_path / "q").chapters_for(5) == ([60, 61], "queue")
