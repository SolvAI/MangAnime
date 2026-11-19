"""Blind A/B rating page (brief §3b "Blind A/B", §10). Local, offline, no build step, no CDN.

    python -m tools.ab_page.app --manifest <manifest.json> --db /home/user/komi-poc-work/runs/ab/results.duckdb --port 8300

Manifest::

    {"experiment": "ep23_wan_vs_animatic",
     "criteria": ["overall", "looks like real anime", "motion", "character consistency"],
     "pairs": [{"pair_id": "s012", "a": "/abs/x.mp4", "b": "/abs/y.mp4", "label_a": "wan", "label_b": "animatic",
                "caption": "optional neutral text shown to the rater", "meta": {...}}]}

What the browser gets: two videos behind random one-time tokens (``/media/<token>``), a progress counter and the
criteria. Labels, file names, pair ids and ``meta`` never leave the server. Left/right is randomised per rating and
stored with it. Ratings go to the DuckDB table described in :mod:`tools.ab_page.stats` (``/results`` shows win
rates, ties, exact 95 % CI and sign-test p-values; it un-blinds, so open it after rating).

Pages: ``/`` (name), ``/rate``, ``/results``, ``/api/results`` (JSON), ``/api/progress``.
Keys on the rating page: 1 = left, 2 = right, = (or 0) = tie, Enter = submit, Space = play/pause, R = restart,
L = loop, , / . = frame step, A / S / D = audio left / right / off, arrow up/down = change criterion, U = undo.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import stats

HERE = Path(__file__).resolve().parent
BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_PIXFMT = {"yuv420p", "yuvj420p"}
BROWSER_AUDIO = {"aac", "mp3", "opus", "vorbis", "flac"}
MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/mp4", ".mkv": "video/x-matroska"}
COOKIE = "ab_rater"


# ----------------------------------------------------------------------------------------------------------------
# manifest + media
# ----------------------------------------------------------------------------------------------------------------
def load_manifest(path: str | Path) -> dict:
    """Read and validate a manifest; returns it with normalised pairs. Raises ValueError with a clear message."""
    m = json.loads(Path(path).read_text())
    if not isinstance(m, dict) or not m.get("experiment"):
        raise ValueError("manifest needs an 'experiment' name")
    crit = m.get("criteria") or ["overall"]
    if not isinstance(crit, list) or not all(isinstance(c, str) and c.strip() for c in crit):
        raise ValueError("'criteria' must be a list of non-empty strings")
    if len(set(crit)) != len(crit):
        raise ValueError("'criteria' contains duplicates")
    pairs = m.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest has no 'pairs'")
    seen = set()
    out = []
    for i, p in enumerate(pairs):
        for k in ("pair_id", "a", "b", "label_a", "label_b"):
            if not p.get(k):
                raise ValueError(f"pair #{i}: missing '{k}'")
        pid = str(p["pair_id"])
        if pid in seen:
            raise ValueError(f"duplicate pair_id {pid!r}")
        seen.add(pid)
        if p["label_a"] == p["label_b"]:
            raise ValueError(f"pair {pid}: label_a and label_b are identical")
        out.append({"pair_id": pid, "a": str(p["a"]), "b": str(p["b"]), "label_a": str(p["label_a"]),
                    "label_b": str(p["label_b"]), "caption": p.get("caption"), "meta": p.get("meta") or {}})
    return {"experiment": str(m["experiment"]), "criteria": [c.strip() for c in crit], "pairs": out}


def _ffbin(name: str) -> str | None:
    import shutil

    for c in (os.environ.get(f"KOMI_{name.upper()}"), shutil.which(name),
              str(Path(os.environ.get("KOMI_WORK", "/home/user/komi-poc-work")) / "tools" / "ffenv" / "bin" / name)):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


class Media:
    """Maps manifest files to files a browser can play (transcoding into a cache when needed) and to opaque tokens."""

    def __init__(self, cache_dir: Path, transcode: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.transcode = transcode
        self._lock = threading.Lock()
        self._served: dict[str, Path] = {}       # source path -> playable path
        self._tokens: dict[str, Path] = {}       # token -> playable path

    def _playable(self, src: Path) -> bool:
        ffprobe = _ffbin("ffprobe")
        if ffprobe is None:
            return True                           # cannot check: serve as is
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt",
                            "-of", "json", str(src)], capture_output=True, text=True)
        try:
            streams = json.loads(r.stdout).get("streams", [])
        except json.JSONDecodeError:
            return True                           # unreadable file: let the browser show the error
        for s in streams:
            if s.get("codec_type") == "video" and (s.get("codec_name") not in BROWSER_VIDEO
                                                   or s.get("pix_fmt") not in BROWSER_PIXFMT):
                return False
            if s.get("codec_type") == "audio" and s.get("codec_name") not in BROWSER_AUDIO:
                return False
        return src.suffix.lower() in (".mp4", ".m4v", ".webm")

    def resolve(self, src: str | Path) -> Path:
        """Playable version of ``src`` (possibly a cached H.264/AAC transcode)."""
        src = Path(src)
        key = str(src)
        with self._lock:
            if key in self._served:
                return self._served[key]
            out = src
            need = self.transcode == "always" or (self.transcode == "auto" and src.is_file() and not self._playable(src))
            ffmpeg = _ffbin("ffmpeg")
            if need and ffmpeg and src.is_file():
                st = src.stat()
                h = hashlib.sha1(f"{src}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:20]
                dst = self.cache_dir / f"{h}.mp4"
                if not dst.exists():
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_suffix(".tmp.mp4")
                    r = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:a", "aac", "-b:a", "160k",
                                        "-threads", "4", "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
                    if r.returncode == 0:
                        os.replace(tmp, dst)
                    else:
                        tmp.unlink(missing_ok=True)
                if dst.exists():
                    out = dst
            self._served[key] = out
            return out

    def token(self, src: str | Path) -> str:
        path = self.resolve(src)
        tok = secrets.token_urlsafe(18)
        with self._lock:
            self._tokens[tok] = path
        return tok

    def path(self, token: str) -> Path | None:
        return self._tokens.get(token)

    def drop(self, *tokens: str) -> None:
        with self._lock:
            for t in tokens:
                self._tokens.pop(t, None)


# ----------------------------------------------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------------------------------------------
def create_app(manifest: dict, db_path: str | Path, transcode: str = "auto", cache_dir: str | Path | None = None,
               seed: int | None = None):
    """FastAPI app for one manifest. ``seed`` fixes the left/right randomisation (tests only)."""
    db_path = Path(db_path)
    experiment = manifest["experiment"]
    criteria: list[str] = manifest["criteria"]
    pairs: list[dict] = manifest["pairs"]
    by_id = {p["pair_id"]: p for p in pairs}
    media = Media(Path(cache_dir) if cache_dir else db_path.parent / "media_cache", transcode)
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    env = Environment(loader=FileSystemLoader(str(HERE / "templates")), autoescape=select_autoescape(["html"]))
    lock = threading.Lock()
    open_pres: dict[str, dict] = {}           # presentation token -> state
    current: dict[str, str] = {}              # rater -> presentation token
    skipped: dict[str, list[str]] = {}        # rater -> pair_ids pushed to the end of the queue
    last_rating: dict[str, str] = {}          # rater -> rating_id of the last submission (for undo)

    app = FastAPI(title=f"A/B: {experiment}", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    app.state.media = media
    app.state.open_presentations = open_pres

    def render(name: str, **ctx) -> HTMLResponse:
        return HTMLResponse(env.get_template(name).render(experiment=experiment, **ctx),
                            headers={"Cache-Control": "no-store"})

    def rater_of(request: Request) -> str | None:
        v = request.cookies.get(COOKIE)
        v = unquote(v).strip() if v else None
        return v or None

    def order_for(rater: str) -> list[str]:
        ids = [p["pair_id"] for p in pairs]
        random.Random(hashlib.sha256(f"{experiment}|{rater}".encode()).hexdigest()).shuffle(ids)
        return ids

    def next_pair(rater: str) -> tuple[str | None, int]:
        done = stats.rated_pairs(db_path, experiment, rater)
        todo = [i for i in order_for(rater) if i not in done]
        sk = [i for i in skipped.get(rater, []) if i in todo]
        todo = [i for i in todo if i not in sk] + sk
        return (todo[0] if todo else None), len([i for i in by_id if i in done])

    def close_pres(tok: str | None) -> None:
        st = open_pres.pop(tok, None) if tok else None
        if st:
            media.drop(st["tok_left"], st["tok_right"])

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "experiment": experiment, "pairs": len(pairs)}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if rater_of(request):
            return RedirectResponse("/rate", status_code=303)
        return render("login.html", n_pairs=len(pairs), criteria=criteria, error=None)

    @app.post("/login")
    def login(rater: str = Form(...)):
        name = " ".join(rater.split())[:64]
        if not name or stats.is_automatic(name):
            return render("login.html", n_pairs=len(pairs), criteria=criteria,
                          error="Please enter a name (names starting with 'judge:' or 'metric:' are reserved).")
        resp = RedirectResponse("/rate", status_code=303)
        resp.set_cookie(COOKIE, quote(name), max_age=180 * 24 * 3600, httponly=True, samesite="lax")
        return resp

    @app.get("/logout")
    def logout():
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    @app.get("/rate", response_class=HTMLResponse)
    def rate(request: Request):
        rater = rater_of(request)
        if not rater:
            return RedirectResponse("/", status_code=303)
        with lock:
            pid, n_done = next_pair(rater)
            if pid is None:
                close_pres(current.pop(rater, None))
                return render("done.html", rater=rater, n_done=n_done, n_pairs=len(pairs),
                              can_undo=rater in last_rating)
            tok = current.get(rater)
            st = open_pres.get(tok) if tok else None
            if st is None or st["pair_id"] != pid:
                close_pres(tok)
                p = by_id[pid]
                left = rng.choice(["a", "b"])
                right = "b" if left == "a" else "a"
                tok = secrets.token_urlsafe(18)
                st = {"pair_id": pid, "rater": rater, "left_side": left, "t0": time.time(),
                      "tok_left": media.token(p[left]), "tok_right": media.token(p[right])}
                open_pres[tok] = st
                current[rater] = tok
        return render("rate.html", rater=rater, pres=tok, tok_left=st["tok_left"], tok_right=st["tok_right"],
                      criteria=criteria, n_done=n_done, n_pairs=len(pairs), caption=by_id[pid].get("caption"),
                      can_undo=rater in last_rating)

    @app.post("/rate")
    async def submit(request: Request):
        rater = rater_of(request)
        if not rater:
            return RedirectResponse("/", status_code=303)
        form = await request.form()
        tok = str(form.get("pres", ""))
        with lock:
            st = open_pres.get(tok)
            if st is None or st["rater"] != rater:
                return RedirectResponse("/rate", status_code=303)        # stale page (server restarted, undo, ...)
            p = by_id[st["pair_id"]]
            left = st["left_side"]
            right = "b" if left == "a" else "a"
            choices = {}
            for i, c in enumerate(criteria):
                v = str(form.get(f"c{i}", ""))
                if v not in ("left", "right", "tie"):
                    return RedirectResponse("/rate?incomplete=1", status_code=303)
                choices[c] = left if v == "left" else right if v == "right" else "tie"
            try:
                dur = int(float(form.get("duration_ms", "") or 0)) or int((time.time() - st["t0"]) * 1000)
            except ValueError:
                dur = int((time.time() - st["t0"]) * 1000)
            note = (str(form.get("note", "")).strip() or None)
            rid = stats.record_rating(db_path, experiment, p["pair_id"], rater, choices, p["label_a"], p["label_b"],
                                      left_side=left, duration_ms=dur, note=note[:1000] if note else None,
                                      meta={"source": "ab_page"})
            last_rating[rater] = rid
            close_pres(tok)
            current.pop(rater, None)
        return RedirectResponse("/rate", status_code=303)

    @app.post("/skip")
    async def skip(request: Request):
        rater = rater_of(request)
        if not rater:
            return RedirectResponse("/", status_code=303)
        form = await request.form()
        with lock:
            st = open_pres.get(str(form.get("pres", "")))
            if st and st["rater"] == rater:
                lst = skipped.setdefault(rater, [])
                if st["pair_id"] in lst:
                    lst.remove(st["pair_id"])
                lst.append(st["pair_id"])
                close_pres(current.pop(rater, None))
        return RedirectResponse("/rate", status_code=303)

    @app.post("/undo")
    def undo(request: Request):
        rater = rater_of(request)
        if not rater:
            return RedirectResponse("/", status_code=303)
        with lock:
            rid = last_rating.pop(rater, None)
            if rid:
                stats.delete_rating(db_path, rid)
                close_pres(current.pop(rater, None))
        return RedirectResponse("/rate", status_code=303)

    @app.get("/media/{token}")
    def media_file(token: str, request: Request):
        path = media.path(token)
        if path is None or not path.is_file():
            return Response(status_code=404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng_h = request.headers.get("range", "")
        if rng_h.startswith("bytes="):
            s, _, e = rng_h[6:].split(",")[0].strip().partition("-")
            try:
                if s == "":
                    start = max(0, size - int(e))
                else:
                    start = int(s)
                    end = min(int(e), size - 1) if e else size - 1
            except ValueError:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            if start > end or start >= size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            status = 206

        def body(start=start, end=end):
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    left -= len(chunk)
                    yield chunk

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1), "Cache-Control": "no-store"}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(body(), status_code=status, headers=headers,
                                 media_type=MIME.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/api/progress")
    def progress(request: Request):
        rater = rater_of(request)
        done = len(stats.rated_pairs(db_path, experiment, rater)) if rater else 0
        return {"rater": rater, "done": done, "total": len(pairs)}

    @app.get("/api/results")
    def api_results():
        return JSONResponse(json.loads(json.dumps(stats.summarize(db_path, experiment), default=str)))

    @app.get("/results", response_class=HTMLResponse)
    def results():
        s = stats.summarize(db_path, experiment)
        return render("results.html", s=s, criteria=criteria, n_pairs=len(pairs))

    return app


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Blind A/B rating page (local).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", default="/home/user/komi-poc-work/runs/ab/results.duckdb")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: this machine only)")
    ap.add_argument("--transcode", default="auto", choices=["auto", "always", "never"],
                    help="auto: re-encode to H.264/yuv420p + AAC only clips a browser cannot play (cached next to the db)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--check", action="store_true", help="validate the manifest and the files, then exit")
    a = ap.parse_args(argv)

    m = load_manifest(a.manifest)
    missing = [f for p in m["pairs"] for f in (p["a"], p["b"]) if not Path(f).is_file()]
    print(f"experiment={m['experiment']} pairs={len(m['pairs'])} criteria={m['criteria']} missing_files={len(missing)}")
    for f in missing[:10]:
        print(f"  missing: {f}")
    if a.check:
        return 1 if missing else 0
    import uvicorn

    app = create_app(m, a.db, transcode=a.transcode, cache_dir=a.cache_dir)
    print(f"rate at http://{a.host}:{a.port}/   results at http://{a.host}:{a.port}/results   db={a.db}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
