"""Review / rename automatically named clusters (character bank and voice clusters; brief §5.2).

    python -m tools.review.app --port 8301 [--processed /home/user/komi-poc/data/processed]
                               [--source name=/abs/clusters.json ...]

Default sources (shown when the file exists):
    characters   <processed>/characters/clusters.json
    speakers     <processed>/anime/spk/clusters.json

Expected (but tolerated loosely) schema::

    {"clusters": [{"id": 3, "name": "Komi" | null, "confidence": 0.41, "size": 812,
                   "exemplars": [{"image": "/abs/a.jpg"}, {"audio": "/abs/b.wav", "text": "..."}]}]}

Corrections are written next to each clusters.json as ``names_override.json`` = ``{"<cluster_id>": "<name>"}``
(atomic write on every change). Conventions: confirming the automatic name writes it as an override; the name
``_junk`` marks a cluster that is not a character / not a voice; clearing removes the override.
Clusters are listed low-confidence first.
"""
from __future__ import annotations

import io
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

HERE = Path(__file__).resolve().parent
DEFAULT_PROCESSED = Path(os.environ.get("KOMI_PROCESSED", "/home/user/komi-poc/data/processed"))
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
AUD_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
AUD_MIME = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
            ".m4a": "audio/mp4", ".opus": "audio/ogg"}
JUNK = "_junk"


def _exemplar(x: Any, base: Path) -> dict | None:
    """Normalise one exemplar entry to {"kind": "image"|"audio", "path": Path, "text": str}."""
    text = ""
    path = None
    kind = None
    if isinstance(x, str):
        path = x
    elif isinstance(x, dict):
        text = str(x.get("text") or x.get("caption") or x.get("transcript") or "")
        for k, kd in (("image", "image"), ("img", "image"), ("frame", "image"), ("crop", "image"),
                      ("audio", "audio"), ("wav", "audio"), ("clip", "audio"), ("path", None)):
            if x.get(k):
                path, kind = str(x[k]), kd
                break
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    if kind is None:
        kind = "audio" if p.suffix.lower() in AUD_EXT else "image"
    return {"kind": kind, "path": p, "text": text}


def load_clusters(path: Path) -> list[dict]:
    """Tolerant reader: returns [{"id", "name", "confidence", "size", "exemplars": [...], "extra": {...}}]."""
    raw = json.loads(Path(path).read_text())
    items = raw.get("clusters", raw) if isinstance(raw, dict) else raw
    if isinstance(items, dict):                       # {"3": {...}, "4": {...}}
        items = [dict(v, id=v.get("id", k)) if isinstance(v, dict) else {"id": k, "exemplars": v} for k, v in items.items()]
    out = []
    for i, c in enumerate(items or []):
        if not isinstance(c, dict):
            continue
        ex_raw = c.get("exemplars") or c.get("examples") or c.get("samples") or []
        if not ex_raw:
            ex_raw = list(c.get("images") or []) + list(c.get("audio") or [])
        ex = [e for e in (_exemplar(x, Path(path).parent) for x in ex_raw) if e]
        conf = c.get("confidence", c.get("conf", c.get("score")))
        try:
            conf = None if conf is None else float(conf)
        except (TypeError, ValueError):
            conf = None
        name = c.get("name")
        extra = {k: v for k, v in c.items() if k in ("votes", "candidates", "note", "notes", "episodes", "source")}
        out.append({"id": str(c.get("id", i)), "name": None if name in (None, "") else str(name), "confidence": conf,
                    "size": c.get("size", len(ex)), "exemplars": ex, "extra": extra})
    return out


class Source:
    def __init__(self, name: str, path: Path):
        self.name, self.path = name, Path(path)
        self.override_path = self.path.parent / "names_override.json"
        self._lock = threading.Lock()
        self._cache: tuple[float, list[dict]] | None = None

    def exists(self) -> bool:
        return self.path.is_file()

    def clusters(self) -> list[dict]:
        mt = self.path.stat().st_mtime
        if self._cache is None or self._cache[0] != mt:
            self._cache = (mt, load_clusters(self.path))
        return self._cache[1]

    def overrides(self) -> dict[str, str]:
        try:
            d = json.loads(self.override_path.read_text())
            return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def set_name(self, cluster_id: str, name: str | None) -> dict[str, str]:
        with self._lock:
            ov = self.overrides()
            if name is None or not name.strip():
                ov.pop(str(cluster_id), None)
            else:
                ov[str(cluster_id)] = " ".join(name.split())
            tmp = self.override_path.with_suffix(".json.tmp")
            keyf = lambda kv: (0, int(kv[0])) if kv[0].lstrip("-").isdigit() else (1, kv[0])  # noqa: E731
            tmp.write_text(json.dumps(dict(sorted(ov.items(), key=keyf)), ensure_ascii=False, indent=1))
            os.replace(tmp, self.override_path)
            return ov


def create_app(sources: dict[str, Path], low_conf: float = 0.6) -> FastAPI:
    env = Environment(loader=FileSystemLoader(str(HERE / "templates")), autoescape=select_autoescape(["html"]))
    srcs = {k: Source(k, v) for k, v in sources.items()}
    thumbs: OrderedDict[tuple, bytes] = OrderedDict()
    tlock = threading.Lock()
    app = FastAPI(title="Cluster review", docs_url=None, redoc_url=None, openapi_url=None)

    def page(name: str, **ctx) -> HTMLResponse:
        return HTMLResponse(env.get_template(name).render(**ctx), headers={"Cache-Control": "no-store"})

    def view(src: Source) -> list[dict]:
        ov = src.overrides()
        rows = []
        for idx, c in enumerate(src.clusters()):
            o = ov.get(c["id"])
            rows.append({
                "idx": idx, "id": c["id"], "auto_name": c["name"], "override": o, "name": o if o is not None else c["name"],
                "confidence": c["confidence"], "size": c["size"], "extra": c["extra"],
                "reviewed": o is not None, "junk": o == JUNK,
                "low": c["confidence"] is None or c["confidence"] < low_conf or not c["name"],
                "exemplars": [{"i": i, "kind": e["kind"], "text": e["text"], "ok": e["path"].is_file()}
                              for i, e in enumerate(c["exemplars"])],
            })
        rows.sort(key=lambda r: (-1.0 if r["confidence"] is None else r["confidence"], r["idx"]))
        return rows

    @app.get("/", response_class=HTMLResponse)
    def index():
        info = []
        for s in srcs.values():
            d: dict[str, Any] = {"name": s.name, "path": str(s.path), "exists": s.exists(), "error": None}
            if s.exists():
                try:
                    rows = view(s)
                    d.update(n=len(rows), reviewed=sum(r["reviewed"] for r in rows), low=sum(r["low"] for r in rows),
                             low_open=sum(r["low"] and not r["reviewed"] for r in rows))
                except Exception as e:
                    d["error"] = f"{type(e).__name__}: {e}"
            info.append(d)
        return page("index.html", sources=info)

    @app.get("/s/{name}", response_class=HTMLResponse)
    def source_page(name: str, show: str = "all"):
        s = srcs.get(name)
        if s is None:
            return HTMLResponse("unknown source", status_code=404)
        if not s.exists():
            return page("source.html", src=name, path=str(s.path), rows=[], names=[], missing=True, error=None,
                        show=show, low_conf=low_conf, override_path=str(s.override_path), counts={}, junk=JUNK)
        try:
            rows = view(s)
        except Exception as e:
            return page("source.html", src=name, path=str(s.path), rows=[], names=[], missing=False,
                        error=f"{type(e).__name__}: {e}", show=show, low_conf=low_conf,
                        override_path=str(s.override_path), counts={}, junk=JUNK)
        names = sorted({r["name"] for r in rows if r["name"] and r["name"] != JUNK}
                       | {r["auto_name"] for r in rows if r["auto_name"]})
        counts = {"all": len(rows), "todo": sum(not r["reviewed"] for r in rows),
                  "low": sum(r["low"] and not r["reviewed"] for r in rows)}
        if show == "todo":
            rows = [r for r in rows if not r["reviewed"]]
        elif show == "low":
            rows = [r for r in rows if r["low"] and not r["reviewed"]]
        return page("source.html", src=name, path=str(s.path), rows=rows, names=names, missing=False, error=None,
                    show=show, low_conf=low_conf, override_path=str(s.override_path), counts=counts, junk=JUNK)

    @app.post("/api/{name}/name")
    async def set_name(name: str, request: Request):
        s = srcs.get(name)
        if s is None or not s.exists():
            return JSONResponse({"ok": False, "error": "unknown source"}, status_code=404)
        body = await request.json()
        cid = str(body.get("id", ""))
        if cid not in {c["id"] for c in s.clusters()}:
            return JSONResponse({"ok": False, "error": f"unknown cluster {cid}"}, status_code=400)
        new = body.get("name")
        ov = s.set_name(cid, None if new is None else str(new)[:120])
        return {"ok": True, "id": cid, "override": ov.get(cid), "n_overrides": len(ov)}

    @app.get("/api/{name}/overrides")
    def get_overrides(name: str):
        s = srcs.get(name)
        if s is None:
            return JSONResponse({}, status_code=404)
        return s.overrides()

    @app.get("/media/{name}/{idx}/{i}")
    def media(name: str, idx: int, i: int, w: int = 0):
        s = srcs.get(name)
        try:
            e = s.clusters()[idx]["exemplars"][i]
        except Exception:
            return Response(status_code=404)
        p: Path = e["path"]
        if not p.is_file():
            return Response(status_code=404)
        if e["kind"] == "audio":
            return FileResponse(p, media_type=AUD_MIME.get(p.suffix.lower(), "application/octet-stream"))
        if w and 16 <= w <= 1024:
            key = (str(p), p.stat().st_mtime_ns, w)
            with tlock:
                data = thumbs.get(key)
            if data is None:
                from PIL import Image

                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((w, w))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=85)
                data = buf.getvalue()
                with tlock:
                    thumbs[key] = data
                    while len(thumbs) > 4000:
                        thumbs.popitem(last=False)
            return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
        return FileResponse(p)

    return app


def default_sources(processed: Path) -> dict[str, Path]:
    return {"characters": processed / "characters" / "clusters.json",
            "speakers": processed / "anime" / "spk" / "clusters.json"}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Review / rename automatically named clusters.")
    ap.add_argument("--processed", default=str(DEFAULT_PROCESSED))
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                    help="extra (or replacement) clusters.json, e.g. --source characters=/abs/clusters.json")
    ap.add_argument("--low-conf", type=float, default=0.6, help="confidence below which a cluster is flagged")
    ap.add_argument("--port", type=int, default=8301)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    sources = default_sources(Path(a.processed))
    for s in a.source:
        k, _, v = s.partition("=")
        if not v:
            ap.error("--source expects NAME=PATH")
        sources[k] = Path(v)
    import uvicorn

    for k, v in sources.items():
        print(f"{k}: {v} ({'found' if v.is_file() else 'missing'})")
    print(f"review at http://{a.host}:{a.port}/")
    uvicorn.run(create_app(sources, a.low_conf), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
