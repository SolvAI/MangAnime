"""Gold-alignment labeling page: manga panels (reading order) <-> anime shots of one episode (brief §5.3 item 4).

    python -m tools.align_label.app --port 8302 [--processed /home/user/komi-poc/data/processed]

Reads (all optional; the page explains what is missing)
    <processed>/manga/panels.jsonl       chapter, page, panel, order, bbox [x,y,w,h], image, dialogue [{speaker,text,kind}]
    <processed>/anime/shots.jsonl        episode, shot, start, end, keyframes [paths]  (+ dialogue / caption if present)
    <processed>/align/alignment.jsonl    chapter, panel_order, shots [{episode, shot}], match, confidence   (pre-fill)
    <processed>/align/gold/queue.json    episodes to label: [1, 2] | {"episodes": [...]} | [{"episode": 1, "chapters": [1, 2]}]
Writes
    <processed>/align/gold/epNN.jsonl    one row per *reviewed* panel, same schema as alignment.jsonl:
        {"chapter", "panel_order", "shots": [{"episode", "shot"}], "match": "manual", "confidence": 1.0,
         "episode": NN, "status": "confirmed" | "no_match", "auto_shots": [...], "ts": "..."}
    "no_match" = the labeler checked the panel and it has no shot in this episode (shots = []).
    Panels that were never reviewed are not written, so the file only contains gold.

Which chapters are shown for an episode: ?chapters=12-15,17 in the URL > queue.json entry > chapters already in the
gold file > chapters whose automatic alignment points into the episode.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

HERE = Path(__file__).resolve().parent
DEFAULT_PROCESSED = Path(os.environ.get("KOMI_PROCESSED", "/home/user/komi-poc/data/processed"))


def read_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader: missing file -> [], broken lines are skipped."""
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict):
                rows.append(r)
    return rows


def parse_chapters(spec: str | None) -> list[int]:
    """'12-15,17' -> [12, 13, 14, 15, 17] (anything unparsable is ignored)."""
    import re

    out: list[int] = []
    for part in (spec or "").replace(" ", "").split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        out += list(range(min(lo, hi), max(lo, hi) + 1))
    return sorted(set(out))


def _int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


class Data:
    """Lazy, mtime-checked view of the processed directory."""

    def __init__(self, processed: Path):
        self.root = Path(processed)
        self.panels_path = self.root / "manga" / "panels.jsonl"
        self.shots_path = self.root / "anime" / "shots.jsonl"
        self.align_path = self.root / "align" / "alignment.jsonl"
        self.gold_dir = self.root / "align" / "gold"
        self.queue_path = self.gold_dir / "queue.json"
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _load(self, path: Path) -> list[dict]:
        if not path.is_file():
            return []
        mt = path.stat().st_mtime
        with self._lock:
            c = self._cache.get(str(path))
            if c is None or c[0] != mt:
                c = (mt, read_jsonl(path))
                self._cache[str(path)] = c
            return c[1]

    def panels(self) -> list[dict]:
        rows = self._load(self.panels_path)
        for i, r in enumerate(rows):
            r.setdefault("_idx", i)
        return rows

    def shots(self) -> list[dict]:
        return self._load(self.shots_path)

    def alignment(self) -> list[dict]:
        return self._load(self.align_path)

    def status(self) -> dict:
        return {k: {"path": str(p), "exists": p.is_file(), "rows": len(self._load(p)) if p.suffix == ".jsonl" else None}
                for k, p in (("panels", self.panels_path), ("shots", self.shots_path),
                             ("alignment", self.align_path), ("queue", self.queue_path))}

    def queue(self) -> list[dict]:
        """[{"episode": int, "chapters": [int] | None}] from queue.json (or every episode in shots.jsonl)."""
        items: list[dict] = []
        if self.queue_path.is_file():
            try:
                q = json.loads(self.queue_path.read_text())
            except json.JSONDecodeError:
                q = []
            if isinstance(q, dict):
                q = q.get("episodes", q.get("queue", []))
            for e in q if isinstance(q, list) else []:
                if isinstance(e, dict) and "episode" in e:
                    ch = e.get("chapters")
                    items.append({"episode": _int(e["episode"]), "chapters": [_int(c) for c in ch] if ch else None})
                elif isinstance(e, (int, str)):
                    ep = _int(str(e).lower().replace("ep", ""), -1)
                    if ep >= 0:
                        items.append({"episode": ep, "chapters": None})
        if not items:
            items = [{"episode": ep, "chapters": None} for ep in sorted({_int(s.get("episode"), -1) for s in self.shots()}) if ep >= 0]
        return items

    def gold_path(self, ep: int) -> Path:
        return self.gold_dir / f"ep{ep:02d}.jsonl"

    def gold(self, ep: int) -> list[dict]:
        return read_jsonl(self.gold_path(ep))

    def chapters_for(self, ep: int, spec: str | None = None) -> tuple[list[int], str]:
        if spec:
            return parse_chapters(spec), "url"
        for q in self.queue():
            if q["episode"] == ep and q["chapters"]:
                return sorted(set(q["chapters"])), "queue"
        g = sorted({_int(r.get("chapter"), -1) for r in self.gold(ep)} - {-1})
        counts: dict[int, int] = {}
        for r in self.alignment():
            if any(_int(s.get("episode"), -1) == ep for s in r.get("shots") or [] if isinstance(s, dict)):
                counts[_int(r.get("chapter"), -1)] = counts.get(_int(r.get("chapter"), -1), 0) + 1
        auto = sorted(c for c, n in counts.items() if c >= 0 and n >= 2) or sorted(c for c in counts if c >= 0)
        if g or auto:
            return sorted(set(g) | set(auto)), "gold+alignment" if g else "alignment"
        return [], "none"

    def resolve(self, p: str | None, base: Path) -> Path | None:
        if not p:
            return None
        q = Path(p)
        if q.is_absolute():
            return q
        for b in (base, self.root, self.root.parent, self.root.parent.parent):
            if (b / q).is_file():
                return b / q
        return base / q

    def episode_payload(self, ep: int, spec: str | None = None) -> dict:
        chapters, origin = self.chapters_for(ep, spec)
        chs = set(chapters)
        panels = [p for p in self.panels() if _int(p.get("chapter"), -1) in chs]
        panels.sort(key=lambda p: (_int(p.get("chapter")), _int(p.get("order"), _int(p.get("page")) * 100 + _int(p.get("panel")))))
        shots = [s for s in self.shots() if _int(s.get("episode"), -1) == ep]
        shots.sort(key=lambda s: (float(s.get("start") or 0), _int(s.get("shot"))))
        shot_ids = {_int(s.get("shot")) for s in shots}
        auto: dict[str, dict] = {}
        for r in self.alignment():
            if _int(r.get("chapter"), -1) not in chs:
                continue
            ss = sorted({_int(s.get("shot")) for s in r.get("shots") or [] if isinstance(s, dict)
                         and _int(s.get("episode"), -1) == ep and _int(s.get("shot")) in shot_ids})
            if ss:
                auto[f"{_int(r.get('chapter'))}:{_int(r.get('panel_order'))}"] = {
                    "shots": ss, "match": r.get("match"), "confidence": r.get("confidence")}
        gold = {}
        for r in self.gold(ep):
            gold[f"{_int(r.get('chapter'))}:{_int(r.get('panel_order'))}"] = {
                "shots": sorted({_int(s.get("shot")) for s in r.get("shots") or [] if isinstance(s, dict)}),
                "status": r.get("status") or ("confirmed" if r.get("shots") else "no_match")}

        def dlg(rows) -> str:
            out = []
            for d in rows or []:
                if isinstance(d, dict) and d.get("text"):
                    sp = d.get("speaker")
                    kind = d.get("kind")
                    out.append((f"{sp}: " if sp else "") + str(d["text"]) + (f" [{kind}]" if kind and kind != "speech" else ""))
                elif isinstance(d, str):
                    out.append(d)
            return " / ".join(out)

        return {
            "episode": ep, "chapters": chapters, "chapters_origin": origin,
            "panels": [{"key": f"{_int(p.get('chapter'))}:{_int(p.get('order'))}", "idx": p["_idx"],
                        "chapter": _int(p.get("chapter")), "page": _int(p.get("page")), "panel": _int(p.get("panel")),
                        "order": _int(p.get("order")), "text": dlg(p.get("dialogue")), "caption": p.get("caption") or ""}
                       for p in panels],
            "shots": [{"shot": _int(s.get("shot")), "start": float(s.get("start") or 0), "end": float(s.get("end") or 0),
                       "nkf": len(s.get("keyframes") or []), "text": dlg(s.get("dialogue"))} for s in shots],
            "auto": auto, "gold": gold, "gold_path": str(self.gold_path(ep)),
        }

    def save_gold(self, ep: int, rows: list[dict]) -> int:
        """rows: [{"key": "chapter:order", "shots": [int], "status": "confirmed"|"no_match"}] -> gold/epNN.jsonl."""
        auto = {}
        for r in self.alignment():
            ss = sorted({_int(s.get("shot")) for s in r.get("shots") or [] if isinstance(s, dict) and _int(s.get("episode"), -1) == ep})
            if ss:
                auto[(_int(r.get("chapter")), _int(r.get("panel_order")))] = ss
        now = _dt.datetime.now().isoformat(timespec="seconds")
        old = {(_int(r.get("chapter")), _int(r.get("panel_order"))): r for r in self.gold(ep)}
        out = []
        for r in rows:
            try:
                ch, order = (int(x) for x in str(r["key"]).split(":"))
            except (KeyError, ValueError):
                continue
            status = r.get("status")
            if status not in ("confirmed", "no_match"):
                continue
            shots = [] if status == "no_match" else sorted({int(s) for s in r.get("shots") or []})
            if status == "confirmed" and not shots:
                continue
            prev = old.get((ch, order))
            same = prev is not None and prev.get("status") == status and \
                sorted(_int(s.get("shot")) for s in prev.get("shots") or []) == shots
            out.append({"chapter": ch, "panel_order": order, "shots": [{"episode": ep, "shot": s} for s in shots],
                        "match": "manual", "confidence": 1.0, "episode": ep, "status": status,
                        "auto_shots": auto.get((ch, order), []), "ts": prev.get("ts", now) if same else now})
        out.sort(key=lambda r: (r["chapter"], r["panel_order"]))
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        path = self.gold_path(ep)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return len(out)


def create_app(processed: str | Path = DEFAULT_PROCESSED) -> FastAPI:
    data = Data(Path(processed))
    env = Environment(loader=FileSystemLoader(str(HERE / "templates")), autoescape=select_autoescape(["html"]))
    thumbs: OrderedDict[tuple, bytes] = OrderedDict()
    tlock = threading.Lock()
    app = FastAPI(title="Alignment labeling", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.data = data

    def page(name: str, **ctx) -> HTMLResponse:
        return HTMLResponse(env.get_template(name).render(**ctx), headers={"Cache-Control": "no-store"})

    def thumb(path: Path | None, h: int, bbox=None) -> Response:
        if path is None or not path.is_file():
            return Response(status_code=404)
        h = max(32, min(int(h or 160), 1200))
        key = (str(path), path.stat().st_mtime_ns, h, tuple(bbox) if bbox else None)
        with tlock:
            b = thumbs.get(key)
        if b is None:
            from PIL import Image

            with Image.open(path) as im:
                im = im.convert("RGB")
                if bbox and len(bbox) == 4:
                    x, y, w, hh = (float(v) for v in bbox)
                    # the image is a whole page if it is clearly larger than the panel box -> crop the panel
                    if w > 4 and hh > 4 and (im.width > 1.15 * w or im.height > 1.15 * hh) \
                            and x + w <= im.width * 1.02 and y + hh <= im.height * 1.02:
                        im = im.crop((int(x), int(y), int(min(im.width, x + w)), int(min(im.height, y + hh))))
                im.thumbnail((h * 4, h))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=86)
            b = buf.getvalue()
            with tlock:
                thumbs[key] = b
                while len(thumbs) > 6000:
                    thumbs.popitem(last=False)
        return Response(b, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/", response_class=HTMLResponse)
    def index():
        st = data.status()
        eps = []
        for q in data.queue():
            ep = q["episode"]
            g = data.gold(ep)
            chapters, origin = data.chapters_for(ep)
            chs = set(chapters)
            n_pan = sum(1 for p in data.panels() if _int(p.get("chapter"), -1) in chs)
            eps.append({"episode": ep, "chapters": chapters, "origin": origin, "n_panels": n_pan,
                        "n_shots": sum(1 for s in data.shots() if _int(s.get("episode"), -1) == ep),
                        "n_gold": len(g), "gold_path": str(data.gold_path(ep))})
        return page("index.html", status=st, episodes=eps, root=str(data.root))

    @app.get("/ep/{ep}", response_class=HTMLResponse)
    def episode(ep: int, chapters: str | None = None):
        return page("label.html", ep=ep, chapters=chapters or "", status=data.status())

    @app.get("/api/ep/{ep}")
    def api_episode(ep: int, chapters: str | None = None):
        return JSONResponse(data.episode_payload(ep, chapters))

    @app.post("/api/ep/{ep}/save")
    async def api_save(ep: int, request: Request):
        body = await request.json()
        rows = body.get("rows") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return JSONResponse({"ok": False, "error": "rows missing"}, status_code=400)
        n = data.save_gold(ep, rows)
        return {"ok": True, "n": n, "path": str(data.gold_path(ep))}

    @app.get("/img/panel/{idx}")
    def img_panel(idx: int, h: int = 200):
        ps = data.panels()
        if not (0 <= idx < len(ps)):
            return Response(status_code=404)
        p = ps[idx]
        return thumb(data.resolve(p.get("image"), data.panels_path.parent), h, p.get("bbox"))

    @app.get("/img/shot/{ep}/{shot}/{k}")
    def img_shot(ep: int, shot: int, k: int, h: int = 120):
        for s in data.shots():
            if _int(s.get("episode"), -1) == ep and _int(s.get("shot"), -1) == shot:
                kf = s.get("keyframes") or []
                if isinstance(kf, str):
                    kf = [kf]
                if 0 <= k < len(kf):
                    return thumb(data.resolve(kf[k], data.shots_path.parent), h)
                break
        return Response(status_code=404)

    return app


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Gold alignment labeling page (manga panels <-> anime shots).")
    ap.add_argument("--processed", default=str(DEFAULT_PROCESSED))
    ap.add_argument("--port", type=int, default=8302)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    import uvicorn

    app = create_app(a.processed)
    for k, v in app.state.data.status().items():
        print(f"{k}: {v['path']} ({'found' if v['exists'] else 'missing'})")
    print(f"label at http://{a.host}:{a.port}/")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
