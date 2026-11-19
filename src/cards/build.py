"""Character cards from the Komi-san fandom wiki: the cast catalogue read by the vision and writing models.

  python -m src.cards.build fetch      # Category:Characters -> wikitext, infobox, anime/manga pictures (all cached)
  python -m src.cards.build condense   # DeepSeek writes one compact card per early character -> cards.json
  python -m src.cards.build sheet      # ref/<id>.jpg (manga | anime side by side) + sheets/cast_NN.jpg
  python -m src.cards.build all

Every step is idempotent: API responses and pictures are cached under data/interim/cards/raw, LLM replies in the
LLM cache. Outputs live in data/processed/cards (cards.json, img/, ref/, sheets/, README.md).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.common.config import CFG, P
from src.common.io import log, read_json, sha, write_json

API = "https://komisan.fandom.com/api.php"
WIKI = "https://komisan.fandom.com/wiki/"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-script"
INTERIM = P("interim") / "cards"
RAW = INTERIM / "raw"
OUT = P("processed") / "cards"
PAGES, CONDENSED = INTERIM / "pages.json", INTERIM / "condensed.json"

MAX_CHAPTER = 140                 # condense characters that debut up to this chapter (plus roster characters)
MAX_IMG_BYTES = 8 * 1024 * 1024
BATCH = 6                         # characters per DeepSeek call
LIMITS = {"look_anime": 45, "look_manga": 35, "personality": 40, "speech_style": 30, "relationships": 30}
NOT_DESCRIBED = "not described on the wiki"
TIERS = {"Major Characters": "main", "Supporting Characters": "recurring", "Minor Characters": "minor"}
ROLE_ORDER = ["main", "recurring", "family", "minor", "other"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ------------------------------------------------------------------ wiki API (cached, throttled)
_last_request = [0.0]


def _throttle(gap: float = 0.3) -> None:
    wait = _last_request[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.time()


def api(**params) -> dict:
    """One GET on the MediaWiki API; the response is cached on disk by its parameters."""
    params = {**params, "format": "json", "formatversion": 2}
    f = RAW / "api" / f"{sha(params)}.json"
    if f.exists():
        return read_json(f)
    _throttle()
    r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if "error" in d:
        raise RuntimeError(f"wiki API error: {d['error']}")
    write_json(f, d)
    return d


def api_all(**params) -> list[dict]:
    """Same, following `continue` until the listing is complete."""
    out, cont = [], {}
    while True:
        d = api(**params, **cont)
        out.append(d)
        if "continue" not in d:
            return out
        cont = d["continue"]


def category(name: str) -> list[str]:
    return [m["title"] for d in api_all(action="query", list="categorymembers", cmtitle=f"Category:{name}", cmlimit=500)
            for m in d["query"]["categorymembers"] if m["ns"] == 0]


def wikitexts(titles: list[str]) -> dict[str, str]:
    out = {}
    for i in range(0, len(titles), 50):
        for d in api_all(action="query", prop="revisions", rvprop="content", rvslots="main", titles="|".join(titles[i:i + 50])):
            for p in d["query"].get("pages", []):
                if p.get("revisions"):
                    out[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
    return out


def file_infos(names: list[str]) -> dict[str, dict]:
    """wiki file name -> {url, size, mime} (missing files are left out)."""
    out = {}
    for i in range(0, len(names), 50):
        chunk = names[i:i + 50]
        for d in api_all(action="query", prop="imageinfo", iiprop="url|size|mime", titles="|".join("File:" + n for n in chunk)):
            norm = {n["from"]: n["to"] for n in d["query"].get("normalized", [])}
            by_title = {p["title"]: p for p in d["query"].get("pages", [])}
            for n in chunk:
                ii = (by_title.get(norm.get("File:" + n, "File:" + n)) or {}).get("imageinfo")
                if ii:
                    out[n] = {"url": ii[0]["url"], "size": ii[0].get("size", 0), "mime": ii[0].get("mime", "")}
    return out


# ------------------------------------------------------------------ wikitext parsing (regex only)
def _template(text: str, name_re: str) -> tuple[str | None, int]:
    """Body of the first {{name ...}} (balanced braces) and the offset just after it."""
    m = re.search(r"\{\{\s*" + name_re, text, re.I)
    if not m:
        return None, 0
    depth, i, low = 0, m.start(), text.lower()
    while i < len(text):
        if low.startswith("<tabber", i) and "</tabber>" in low[i:]:   # stray braces inside tab blocks do not count
            i = low.index("</tabber>", i) + len("</tabber>")
        elif text.startswith("{{", i):
            depth, i = depth + 1, i + 2
        elif text.startswith("}}", i):
            depth, i = depth - 1, i + 2
            if depth == 0:
                return text[m.end():i - 2], i
        else:
            i += 1
    return text[m.end():], len(text)


def _split_params(body: str) -> dict[str, str]:
    """Top-level |key=value pairs of a template body ('|' inside {{}}, [[]] and <tabber> does not split)."""
    parts, buf, depth, i, low = [], [], 0, 0, body.lower()
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth = max(0, depth - 1)
            buf.append(two)
            i += 2
        elif low.startswith("<tabber", i):
            j = low.find("</tabber>", i)
            j = len(body) if j < 0 else j + len("</tabber>")
            buf.append(body[i:j])
            i = j
        else:
            if body[i] == "|" and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(body[i])
            i += 1
    parts.append("".join(buf))
    out: dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = re.sub(r"\s*\d+$", "", " ".join(k.split()))          # "Family 2" joins "Family"
        if v.strip():
            out[k] = (out[k] + "<br>" + v.strip()) if k in out else v.strip()
    return out


def _tpl(m: re.Match) -> str:
    parts = m.group(1).split("|")
    name, args = parts[0].strip().lower(), [a.strip() for a in parts[1:] if "=" not in a]
    if name == "nihongo" and args:
        return f"{args[0]} ({args[1]})" if len(args) > 1 and args[1] else args[0]
    if name == "ruby" and args:
        return args[0]
    if name == "cm2ftin" and args:
        return args[-1] if "cm" in args[-1] else args[-1] + " cm"
    return ""


def plain(t: str, title: str = "", br: str = " ") -> str:
    """Wikitext -> plain text (links, templates, refs and html removed)."""
    t = re.sub(r"<!--.*?-->|<ref[^>]*/>|<ref[^>]*>.*?</ref>|<gallery.*?</gallery>", "", t, flags=re.S | re.I)
    t = t.replace("{{PAGENAME}}", title).replace("{{KCC}}", "Komi Can't Communicate").replace("{{!}}", "|")
    for _ in range(8):                                            # innermost templates first
        t, n = re.subn(r"\{\{([^{}]*)\}\}", _tpl, t)
        if not n:
            break
    t = re.sub(r"\[\[\s*(?:File|Image|Category)\s*:(?:[^\[\]]|\[\[[^\[\]]*\]\])*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"^={3,}\s*(.*?)\s*={3,}\s*$", r"[\1]", t, flags=re.M)   # sub-headings -> [name]
    t = re.sub(r"<br\s*/?>", br, t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(re.sub(r"'{2,}", "", t)).replace("\xa0", " ")
    t = re.sub(r"^[*#:]+\s*", "- ", t, flags=re.M)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +([,.;)])", r"\1", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def _field(v: str | None, title: str = "") -> str | None:
    """Infobox value -> one clean line ('; ' between <br> / list items)."""
    if not v:
        return None
    v = re.sub(r"^\s*\*+\s*", "<br>", v, flags=re.M)
    t = re.sub(r"\s*;[\s;]*", "; ", plain(v, title, br="; ").replace("\n", "; ")).strip(" ;")
    return t or None


def _first_number(v: str | None, word: str) -> int | None:
    """Smallest 'Chapter N' / 'Episode N' of an infobox value, ignoring the one-shot pilot and mere mentions."""
    nums = []
    for piece in re.split(r"<br\s*/?>|\n", v or "", flags=re.I):
        if re.search(r"one[- ]?shot|pilot|mentioned", piece, re.I):
            continue
        nums += [int(n) for n in re.findall(word + r"s?\s*#?(\d+)", piece, re.I)]
    return min(nums) if nums else None


def _medium(label: str) -> str | None:
    lab = label.lower()
    if "anime" in lab:
        return "anime"
    if "manga" in lab:
        return "manga"
    if re.search(r"live|drama|novel|one[- ]?shot|pilot|game|stage", lab):
        return "other"
    return None                                                   # an age / outfit sub-tab: medium unchanged


def infobox_images(v: str) -> list[dict]:
    """Pictures of the infobox Image field in page order, each with the tab it sits under."""
    v = (v or "").replace("{{!}}", "|")
    toks = [(m.start(), "label", m.group(1).strip()) for m in re.finditer(r"(?:tabber\s*\||\|-\||<tabber>)\s*([^=\[\]{}|<>\n]{1,40})=", v, re.I)]
    toks += [(m.start(), "file", m.group(1).strip()) for m in re.finditer(r"\[\[\s*(?:File|Image)\s*:\s*([^|\]]+)", v, re.I)]
    out, medium, sub = [], None, None
    for _, kind, val in sorted(toks):
        if kind == "label":
            med = _medium(val)
            medium, sub = (med, None) if med else (medium, val)
        else:
            by_name = _medium(val) if _medium(val) in ("anime", "manga") else None
            out.append({"medium": medium or by_name, "tab": sub, "file": val.replace("_", " ")})
    return out


def _shorten(t: str, n: int) -> str:
    if len(t) <= n:
        return t
    cut = t[:n]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return (cut[:end + 1] if end > n * 0.5 else cut.rsplit(" ", 1)[0] + " ...").strip()


def _short_relationships(t: str) -> str:
    """Relationships text: at most the first ~230 characters of each [person] block, 1500 in total."""
    blocks = [b.strip().replace("\n", " ") for b in re.split(r"\n(?=\[[^\]]+\]\n)", "\n" + t + "\n")]
    blocks = [b for b in blocks if b and not re.fullmatch(r"\[[^\]]+\]", b)]      # stub pages: heading without text
    return _shorten("\n".join(_shorten(b, 230) for b in blocks), 1500)


def parse_page(title: str, text: str, rel_text: str | None) -> dict | None:
    body, end = _template(text, r"Infobox[ _]*Character")
    if body is None:
        return None                                               # list / quote pages: not a character
    raw = _split_params(body)
    heads = list(re.finditer(r"^==([^=\n][^\n]*?)==[ \t]*$", text, re.M))
    lead_end = next((h.start() for h in heads if h.start() >= end), len(text))
    sections = {}
    for h, nxt in zip(heads, heads[1:] + [None]):
        name = plain(h.group(1)).lower()
        sections.setdefault(name, plain(text[h.end():nxt.start() if nxt else len(text)], title))
    rel = sections.get("relationships") or (plain(re.sub(r"^==([^=\n][^\n]*?)==[ \t]*$", "", rel_text, flags=re.M), title) if rel_text else "")
    background = next((sections[k] for k in ("background", "history", "history and role", "story", "synopsis") if sections.get(k)), "")
    info = {k: _field(v, title) for k, v in raw.items() if k != "Image"}
    return {
        "title": title, "url": WIKI + title.replace(" ", "_"),
        "info": {k: v for k, v in info.items() if v},
        "first_chapter": _first_number(raw.get("Manga"), "Chapter"),
        "first_episode": _first_number(raw.get("Anime"), "Episode"),
        "categories": re.findall(r"\[\[\s*Category\s*:\s*([^\]|]+)", text, re.I),
        "images": infobox_images(raw.get("Image", "")),
        "lead": plain(text[end:lead_end], title),
        "appearance": sections.get("appearance", ""), "personality": sections.get("personality", ""),
        "relationships": _short_relationships(rel) if rel else "", "background": _shorten(background, 900),
        "trivia": _shorten(sections.get("trivia", ""), 900),
    }


# ------------------------------------------------------------------ ids, roles, order
def slug(cid: str) -> str:
    return cid.replace(" ", "_")


def _tokens(s: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", s.lower()))


def assign_ids(pages: list[dict]) -> None:
    """Roster short name where the page is in configs/characters.yaml, else the family name (full name on a clash)."""
    roster = list(CFG.characters.roster)
    by_tokens = {_tokens(p["title"]): p for p in pages}
    for i, r in enumerate(roster):
        p = by_tokens.get(_tokens(r["full"]))
        if p is None:                                             # spelling differs: unique family-name match
            fam = [q for q in pages if q["title"].split()[0] == r["name"] and "id" not in q]
            p = fam[0] if len(fam) == 1 and len(r["name"].split()) == 1 else None
        if p is not None and "id" not in p:
            p.update(id=r["name"], roster_index=i, roster_role=r["role"], roster_full=r["full"], roster_aliases=list(r.get("aliases", [])))
    family = lambda p: re.sub(r"\(.*?\)", "", p["title"]).split()[0]  # noqa: E731
    taken = [family(p) for p in pages]
    used = {p["id"] for p in pages if "id" in p}
    for p in pages:
        if "id" not in p:
            p["id"] = family(p) if taken.count(family(p)) == 1 and family(p) not in used else p["title"]
            used.add(p["id"])


def assign_roles(pages: list[dict]) -> None:
    """Roster role if known, else family (relative of Komi / Tadano), else the wiki's Major / Supporting / Minor category."""
    relatives = " ".join(p["info"].get("Family", "") for p in pages if p["title"] in ("Komi Shouko", "Tadano Hitohito"))
    for p in pages:
        tier = next((TIERS[t] for t in TIERS if t in p["categories"]), "other")
        p["role"] = p.get("roster_role") or ("family" if p["title"] in relatives or "Komi Family" in p["categories"] else tier)


def importance(p: dict) -> tuple:
    return (p.get("roster_index", 999), ROLE_ORDER.index(p["role"]), p["first_chapter"] if p["first_chapter"] is not None else 9999, p["title"])


# ------------------------------------------------------------------ fetch
def _is_colour(path: Path) -> bool:
    im = Image.open(path)
    im = im.convert("RGBA").resize((96, 96))
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])
    hist = bg.convert("HSV").split()[1].histogram()
    return sum(hist[26:]) / sum(hist) > 0.05              # unlabelled wiki pictures are either pure greyscale or clearly coloured


def _download(name: str, info: dict) -> Path | None:
    """Picture into the raw cache (original size, keyed by wiki file name)."""
    dst = RAW / "img" / name.replace("/", "_")
    if dst.exists():
        return dst
    if info["size"] > MAX_IMG_BYTES:
        log(f"skip {name}: {info['size'] / 1e6:.1f} MB")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("format=original", ""):                        # a few originals are corrupt: fall back to the wiki's re-encoded rendition
        _throttle()
        r = requests.get(info["url"] + ("&" if "?" in info["url"] else "?") + suffix, headers={"User-Agent": UA}, timeout=120)
        if r.status_code != 200 or not r.content:
            continue
        dst.write_bytes(r.content)
        try:
            Image.open(dst).verify()
            return dst
        except Exception:
            dst.unlink()
    log(f"download failed: {name}")
    return None


def fetch() -> None:
    titles = category("Characters")
    tier = {t: c for c in TIERS for t in category(c)}
    texts = wikitexts(titles)
    rels = wikitexts([t + "/Relationships" for t in titles])
    pages, skipped = [], []
    (RAW / "wikitext").mkdir(parents=True, exist_ok=True)
    for t in titles:
        (RAW / "wikitext" / (t.replace("/", "_") + ".wiki")).write_text(texts.get(t, ""))
        p = parse_page(t, texts.get(t, ""), rels.get(t + "/Relationships"))
        if p is None:
            skipped.append(t)
            continue
        if t in tier and tier[t] not in p["categories"]:
            p["categories"].append(tier[t])
        pages.append(p)
    assign_ids(pages)
    assign_roles(pages)

    infos = file_infos(sorted({im["file"] for p in pages for im in p["images"]}))
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    for p in pages:
        in_anime = p["first_episode"] is not None or bool(p["info"].get("Japanese"))
        chosen: dict[str, Path] = {}
        for im in p["images"]:
            if im["medium"] == "other" or im["file"] not in infos or re.search(r"no image", im["file"], re.I):
                continue
            if im["medium"] in chosen or (im["medium"] is None and len(chosen) == 2):
                continue
            src = _download(im["file"], infos[im["file"]])
            if src is None:
                continue
            med = im["medium"] or ("anime" if in_anime and _is_colour(src) else "manga")
            im["guessed_medium"] = None if im["medium"] else med
            chosen.setdefault(med, src)
        for med in ("anime", "manga"):
            p["img_" + med] = None
            for old in (OUT / "img").glob(f"{slug(p['id'])}_{med}.*"):
                old.unlink()
            if med in chosen:
                ext = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}.get(Image.open(chosen[med]).format, "png")
                dst = OUT / "img" / f"{slug(p['id'])}_{med}.{ext}"
                shutil.copyfile(chosen[med], dst)
                p["img_" + med] = f"img/{dst.name}"
    pages.sort(key=importance)
    write_json(PAGES, {"pages": pages, "skipped": skipped})
    both = sum(1 for p in pages if p["img_anime"] and p["img_manga"])
    log(f"fetch: {len(titles)} category pages, {len(pages)} characters ({len(skipped)} non-character pages skipped: {skipped}); "
        f"{both} with both pictures, {sum(1 for p in pages if p['first_chapter'] is not None)} with first_chapter")
    write_cards()


# ------------------------------------------------------------------ condense (DeepSeek)
SYSTEM = f"""You write compact character cards for a production pipeline that adapts the manga "Komi Can't Communicate" into anime shots. The period covered is the first high-school year (manga chapters 1-{MAX_CHAPTER}, anime seasons 1-2).
For every character in the input you get wiki text. Use ONLY that text: add nothing from memory and never guess from a name, job or role. Where the text does not describe something, write exactly "{NOT_DESCRIBED}" (empty string for danbooru_tags, null for age / class_or_occupation).
Return one JSON object {{"cards": [...]}} with one card per input character, same order, same "id". Fields (hard word limits):
- look_anime (max {LIMITS['look_anime']} words): hair (colour, length, style), eyes, build, usual outfit, distinguishing marks, in colour. The high-school / first-appearance look, not childhood or later looks. Only from the appearance text, the lead and the infobox (Hair / Eye / Height); every detail must be stated there: never infer clothing from a job, or build from height, age or gender.
- look_manga (max {LIMITS['look_manga']} words): the same facts restated as cues for a BLACK-AND-WHITE manga panel. Start with the hair: its tone when certain, then length / shape / style. Tone rule: hair the text calls black (also mixed descriptions such as "dark purple/black") -> "dark hair"; blonde, platinum, white, silver or pastel (pink, lavender) hair -> "light hair"; any other colour (brown, red, orange, green, blue, purple, grey) -> no tone word. Then accessories and marks, silhouette / build, and a habitual expression only if the text states one (otherwise leave it out without comment). No colour words. Nothing that is not in the text.
- personality (max {LIMITS['personality']} words): only what the text says about character and behaviour.
- speech_style (max {LIMITS['speech_style']} words): how they talk, for writing English dialogue and directing a voice actor. (a) If the text says how the character speaks (silent, stutter, drawl, dialect, catchphrase, tone, loudness, writes instead of speaking), state that. (b) Otherwise, if the text describes the personality, derive the delivery from it and start the field with "Inferred: ". (c) If the text has neither, write exactly "{NOT_DESCRIBED}".
- relationships (max {LIMITS['relationships']} words): people only, the most important first, as "Name (relation)" separated by "; ". State of the first high-school year. Couples among the students form after that period: mark a boyfriend / girlfriend as "(later ...)".
- age: at first appearance / in the first high-school year when the text gives it (e.g. "15"), else the wiki's value with its qualifier (e.g. "17-18 (currently)").
- class_or_occupation: the first-year state, short, with Itan Private High School written "Itan High", e.g. "Itan High, 1st Year Class 1 student" (add a stated function such as class representative). If no first-year class is given, the earliest one given, e.g. "Itan High student (2nd Year Class 3 later)". For others their school or job.
- danbooru_tags: lowercase, comma-separated, words separated by spaces, real Danbooru tag names where they exist, taken from look_anime. ONLY hair length, hair colour, hair style, eye colour, skin tone, body build and permanent accessories or marks (glasses, eyepatch, hair ribbon, hairpin, ahoge, mole, freckles, fang, facial hair, bandaid on nose). NO clothing, no behaviour, no judgement words such as beautiful. 2 to 10 tags.
Plain factual phrases, no markdown. Format example with an invented character (copy the style, never the content):
{{"id": "Example", "look_anime": "Short spiky red hair, green eyes, stocky build; boys' uniform with the blazer open; scar on the left cheek.", "look_manga": "Short spiky hair, stocky silhouette, scar on the left cheek, open blazer; usually grinning.", "personality": "Loud and impulsive but loyal; picks fights for his friends, cries at sad films.", "speech_style": "Shouts most lines, casual and slangy, ends sentences with 'ya know'.", "relationships": "Komi Shouko (classmate); Tadano Hitohito (friend); Example Hana (older sister)", "age": "15", "class_or_occupation": "Itan High, 1st Year Class 1 student", "danbooru_tags": "short hair, spiky hair, red hair, green eyes, stocky, scar on cheek"}}"""

CLOTHING = re.compile(r"skirt|tights|pantyhose|uniform|cardigan|sweater|jacket|blazer|socks|kneehighs|stockings|thighhighs|shirt|dress|necktie|\btie\b|"
                      r"vest|pants|jeans|shorts|shoes|boots|sandals|gloves|armor|kimono|hoodie|parka|coat|apron|pajamas|tokkofuku|backpack|handbag|suit\b")


def _tags(t: str) -> str:
    """Tag list as the image model wants it: lowercase, spaces, no clothing, no duplicates, at most 10."""
    tags = dict.fromkeys(x.strip().lower().replace("_", " ") for x in t.split(","))
    return ", ".join([x for x in tags if x and x != NOT_DESCRIBED and not CLOTHING.search(x)][:10])


def _opt(v) -> str | None:
    v = (v or "").strip()
    return None if v.lower() in ("", "null", "none", "unknown", NOT_DESCRIBED) else v


CARD_SCHEMA = {
    "type": "object", "required": ["cards"],
    "properties": {"cards": {"type": "array", "items": {
        "type": "object",
        "required": ["id", "look_anime", "look_manga", "personality", "speech_style", "relationships", "danbooru_tags"],
        "properties": {**{k: {"type": "string"} for k in ("id", *LIMITS, "danbooru_tags")},
                       "age": {"type": ["string", "null"]}, "class_or_occupation": {"type": ["string", "null"]}}}}},
}
INFO_KEYS = ("Full Name", "Aliases", "Age", "Gender", "Height", "Hair", "Eye", "Occupation", "School", "Year/Course", "Affiliation", "Family")


def _llm_input(p: dict) -> dict:
    d = {"id": p["id"], "name": p["title"], "infobox": {k: _shorten(p["info"][k], 400) for k in INFO_KEYS if p["info"].get(k)},
         "lead": _shorten(p["lead"], 900), "appearance": _shorten(p["appearance"], 2500), "personality": _shorten(p["personality"], 2500),
         "relationships": p["relationships"], "background": p["background"], "trivia": p["trivia"]}
    return {k: v for k, v in d.items() if v}


def _trim(s: str, n: int) -> str:
    words = s.split()
    if len(words) <= n:
        return s
    cut = " ".join(words[:n])
    end = max(cut.rfind(";"), cut.rfind(","), cut.rfind("."))
    return (cut[:end] if end > len(cut) * 0.6 else cut).rstrip(" ,;.")


COLOURS = re.compile(r"\b(black|white|gr[ae]y|silver|blond\w*|platinum|brown|auburn|red|crimson|pink|orange|yellow|green|olive|blue|purple|lavender)\b", re.I)


META = re.compile(r"not described|not stated|no stated|not mentioned|not specified", re.I)


def _problems(card: dict) -> dict[str, str]:
    """Fields that break a hard rule: too long, colour words in the black-and-white look, remarks about what the wiki lacks."""
    out = {}
    for k, n in LIMITS.items():
        why = [f"over {n} words"] if len(card[k].split()) > n else []
        why += ["has colour words"] if k == "look_manga" and COLOURS.search(card[k]) else []
        why += ["remarks on missing information"] if card[k] != NOT_DESCRIBED and META.search(card[k]) else []
        if why:
            out[k] = "; ".join(why)
    return out


def wants_card(p: dict) -> bool:
    """Early characters get a DeepSeek card: debut by chapter 140, any roster character, or chapter unknown but seen in the anime."""
    if "roster_index" in p or (p["first_chapter"] is not None and p["first_chapter"] <= MAX_CHAPTER):
        return True
    return p["first_chapter"] is None and p["first_episode"] is not None


def _ask(llm, user: str, batch: list[dict]) -> dict[str, dict]:
    """One DeepSeek call. Cards are matched to the batch by position (the model sometimes shortens an id) and the ids are checked."""
    cards = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                     schema=CARD_SCHEMA, max_tokens=4000, temperature=0.0, tag="cards")["cards"]
    if len(cards) != len(batch) or any(not _tokens(c["id"]) <= _tokens(p["id"] + " " + p["title"]) for c, p in zip(cards, batch)):
        raise RuntimeError(f"cards do not match the request: {[c['id'] for c in cards]} vs {[p['id'] for p in batch]}")
    return {p["id"]: {**c, "id": p["id"]} for c, p in zip(cards, batch)}


def condense() -> None:
    from src.llm.client import LLM

    pages = read_json(PAGES)["pages"]
    todo = [p for p in pages if wants_card(p)]
    done: dict[str, dict] = {}
    empty = [p for p in todo if not (p["lead"] or p["appearance"] or p["personality"])]
    for p in empty:                                               # nothing to condense: no LLM call
        done[p["id"]] = {**{k: NOT_DESCRIBED for k in LIMITS}, "danbooru_tags": "", "age": None, "class_or_occupation": None}
    todo = [p for p in todo if p not in empty]
    llm = LLM("deepseek", role="cards")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        got = _ask(llm, "Characters:\n" + json.dumps([_llm_input(p) for p in batch], ensure_ascii=False, indent=1), batch)
        repaired = {}
        for _ in range(2):                                        # up to two repair calls for the cards that break a hard rule
            over = {cid: _problems(c) for cid, c in got.items() if _problems(c)}
            if not over:
                break
            repaired.update(over)
            note = ("These cards break a rule in the listed fields. Rewrite ONLY those fields and copy every other field unchanged, then return "
                    "{\"cards\": [...]} again. 'over N words': shorten, keeping what matters most for recognising and voicing the character. "
                    "'has colour words' (look_manga): black becomes dark, white / blonde / platinum becomes light, every other colour word is dropped (keep the phrase readable). "
                    "'remarks on missing information': delete the remark and keep the rest.\n"
                    f"Problems: {json.dumps(over)}\nCards:\n{json.dumps([got[cid] for cid in over], ensure_ascii=False, indent=1)}")
            got.update(_ask(llm, note, [p for p in batch if p["id"] in over]))
        for cid, c in got.items():
            done[cid] = {**{k: _trim(c[k].strip(), n) for k, n in LIMITS.items()}, "danbooru_tags": _tags(c["danbooru_tags"]),
                         "age": _opt(c.get("age")), "class_or_occupation": _opt(c.get("class_or_occupation"))}
        log(f"condense: batch {i // BATCH + 1}/{-(-len(todo) // BATCH)} ({', '.join(p['id'] for p in batch)})" + (f" - repaired {repaired}" if repaired else ""))
    write_json(CONDENSED, done)
    log(f"condense: {len(done)} cards ({len(empty)} without any wiki text)")
    write_cards()


# ------------------------------------------------------------------ cards.json + README
def _aliases(p: dict, full: str, other_ids: set[str]) -> list[str]:
    """Roster aliases, given name, the infobox spelling of the name and the wiki's nicknames (never another character's id)."""
    wiki = [re.sub(r"\s*\([^)]*\)", "", a).strip(" \"'") for a in re.split(r"[;/]", p["info"].get("Aliases") or "")]
    given = p["title"].split()[1:2] if len(p["title"].split()) == 2 else []
    out, seen = [], {full.lower(), p["id"].lower()} | other_ids
    for a in [*p.get("roster_aliases", []), *given, *(p["info"].get("Full Name") or "").split("; "), *wiki]:
        if a and len(a) <= 40 and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out[:10]


def _class_or_occupation(info: dict) -> str | None:
    return "; ".join(v for v in (info.get("Occupation"), info.get("School"), info.get("Year/Course")) if v) or None


def write_cards() -> list[dict]:
    """Assemble cards.json from the parsed pages, the condensed fields and the pictures that exist on disk."""
    data = read_json(PAGES)
    cond = read_json(CONDENSED, {})
    cards, ids = [], {p["id"].lower() for p in data["pages"]}
    for p in data["pages"]:
        info, c = p["info"], cond.get(p["id"])
        full = p["roster_full"] if _tokens(p.get("roster_full", "")) == _tokens(p["title"]) else (
            info.get("Full Name") if "(" in p["title"] and info.get("Full Name") else p["title"])
        ref = OUT / "ref" / f"{slug(p['id'])}.jpg"
        card = {
            "id": p["id"], "full_name": full, "kanji": info.get("Kanji"), "aliases": _aliases(p, full, ids), "role": p["role"],
            "gender": info.get("Gender"),
            "age": (c or {}).get("age") or info.get("Age"),
            "class_or_occupation": (c or {}).get("class_or_occupation") or _class_or_occupation(info),
            "first_chapter": p["first_chapter"], "first_episode": p["first_episode"],
            "voice_en": info.get("English"), "voice_jp": info.get("Japanese"),
            **{k: (c or {}).get(k) for k in (*LIMITS, "danbooru_tags")},
            "img_anime": p["img_anime"], "img_manga": p["img_manga"], "ref_sheet": f"ref/{ref.name}" if ref.exists() else None,
            "img_anime_heldout": bool(p["img_anime"]) and _heldout_debut(p["first_episode"]),
            "wiki_title": p["title"], "wiki_url": p["url"], "condensed": c is not None,
        }
        if c is None:
            card["infobox"] = info
        cards.append(card)
    write_json(OUT / "cards.json", cards)
    _readme(cards, data["skipped"])
    return cards


def _readme(cards: list[dict], skipped: list[str]) -> None:
    n = len(cards)
    cnt = lambda f: sum(1 for c in cards if f(c))  # noqa: E731
    roster = {r["name"]: r for r in CFG.characters.roster}
    gaps = [f"- `{name}`: no wiki page (roster-only entry, not in cards.json)" for name in roster if name not in {c["id"] for c in cards}]
    for c in cards:
        if c["id"] not in roster:
            continue
        miss = [k for k in ("img_anime", "img_manga", "first_chapter", "first_episode", "voice_en", "voice_jp") if c[k] is None]
        miss += [k for k in LIMITS if c[k] in (None, NOT_DESCRIBED)]
        if miss:
            gaps.append(f"- `{c['id']}` ({c['wiki_title']}): missing {', '.join(miss)}")
        if _tokens(roster[c["id"]]["full"]) != _tokens(c["wiki_title"]):
            gaps.append(f"- `{c['id']}`: roster full name \"{roster[c['id']]['full']}\" differs from the wiki page \"{c['wiki_title']}\" (matched by family name)")
    (OUT / "README.md").write_text(f"""# Character cards (cast catalogue)

Built by `python -m src.cards.build fetch|condense|sheet` (or `all`) from the fandom wiki
<https://komisan.fandom.com/wiki/Category:Characters>. Wiki text is CC-BY-SA; pictures are wiki uploads kept for private research use only.

## Contents
- `cards.json` - list of {n} cards in order of importance (roster order, then main / recurring / family / minor / other, then first chapter).
  `id` is the short name of `configs/characters.yaml` where the character is in that roster, else the family name (full name on a clash).
  Paths (`img_anime`, `img_manga`, `ref_sheet`) are relative to this folder.
  - `condensed: true` cards (first chapter <= {MAX_CHAPTER}, every roster character, and characters with no chapter given but seen in the anime)
    carry the DeepSeek-written fields `look_anime`, `look_manga`, `personality`, `speech_style`, `relationships`, `danbooru_tags` and a compact
    `age` / `class_or_occupation`. They are written only from the wiki text: "{NOT_DESCRIBED}" means the page has no such text, and a
    `speech_style` starting with "Inferred:" is derived from the personality text because the page says nothing about speech.
  - `condensed: false` cards (all other characters) carry the parsed `infobox` instead; their text fields are null.
- `img/<id>_anime.<ext>`, `img/<id>_manga.<ext>` - infobox pictures at original size (spaces in the id become `_`).
- `ref/<id>.jpg` - reference sheet: manga picture (left) and anime picture (right), each fitted into 384x384 on white, name underneath
  (768x420; 384x420 when only one picture exists).
- `sheets/cast_NN.jpg` - contact sheets of the reference sheets, 20 characters each, same order as `cards.json`.

## Counts
- characters: {n} (non-character pages of the category skipped: {', '.join(skipped) or 'none'})
- with both pictures: {cnt(lambda c: c['img_anime'] and c['img_manga'])}; anime only: {cnt(lambda c: c['img_anime'] and not c['img_manga'])}; manga only: {cnt(lambda c: c['img_manga'] and not c['img_anime'])}; none: {cnt(lambda c: not c['img_anime'] and not c['img_manga'])}
- with `first_chapter`: {cnt(lambda c: c['first_chapter'] is not None)}; with `first_episode`: {cnt(lambda c: c['first_episode'] is not None)}
- with a condensed card: {cnt(lambda c: c['condensed'])}; infobox only: {cnt(lambda c: not c['condensed'])}
- with a reference sheet: {cnt(lambda c: c['ref_sheet'])}
- in the roster: {cnt(lambda c: c['id'] in roster)} of {len(roster)} roster entries

## Rebuild
`. scripts/env.sh && python -m src.cards.build all` - wiki responses and pictures are cached in `data/interim/cards/raw/`, DeepSeek replies in
the LLM cache, so a rerun costs nothing. Delete `data/interim/cards/raw/api/` to refetch the wiki.

## Known gaps
- Looks come from the wiki's Appearance text only; it often omits eye colour and never describes the manga drawing separately
  (`look_manga` is the same description turned into black-and-white cues; hair tone is given only for black or blonde / white / pastel hair).
- The wiki describes the whole series: ages, classes and relationships may mention events after chapter {MAX_CHAPTER} (marked "later" where DeepSeek saw it).
- A picture under no Anime / Manga tab is filed by file name, else as anime when it is in colour and the character is in the anime, else as manga.
- Wiki pictures vary in quality: some anime pictures are blurry screenshots or partial views, some manga pictures are colour illustrations,
  and a few originals are corrupt JPEGs (the wiki's re-encoded webp rendition is stored instead). Check `sheets/` before relying on one.
- Roster characters with something missing or mismatched:
{chr(10).join(gaps) or '- none'}
""")


# ------------------------------------------------------------------ reference sheets
def _picture(path: Path, box: int = 384) -> Image.Image:
    im = Image.open(path)
    im.seek(0)
    im = im.convert("RGBA")
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])
    return ImageOps.contain(bg, (box, box), Image.LANCZOS)


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _heldout_debut(first_episode) -> bool:
    """True when the character's first anime episode is a held-out one: any anime picture of them comes from held-out footage."""
    try:
        return int(first_episode) in set(CFG.split.heldout_episodes)
    except (TypeError, ValueError):
        return False


def ref_sheet(card: dict) -> Image.Image | None:
    """Manga | anime picture side by side. The anime picture is left out for characters who debut in a held-out episode
    (strict split, D-050): the pipeline must not see what the real episode made them look like."""
    pics = [(m, OUT / card["img_" + m]) for m in ("manga", "anime") if card["img_" + m] and not (m == "anime" and card.get("img_anime_heldout"))]
    if not pics:
        return None
    sheet = Image.new("RGB", (384 * len(pics), 420), "white")
    d = ImageDraw.Draw(sheet)
    for k, (med, path) in enumerate(pics):
        im = _picture(path)
        sheet.paste(im, (384 * k + (384 - im.width) // 2, (384 - im.height) // 2))
        d.text((6, 412) if k == 0 else (sheet.width - 6, 412), med, fill=(140, 140, 140), font=_font(13), anchor="ls" if k == 0 else "rs")
    size = 24
    while size > 12 and d.textlength(card["id"], font=_font(size)) > sheet.width - 120:
        size -= 2
    d.text((sheet.width // 2, 401), card["id"], fill="black", font=_font(size), anchor="mm")
    return sheet


def sheet() -> None:
    cards = write_cards()
    for sub in ("ref", "sheets"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(parents=True)
    refs = []
    for c in cards:
        im = ref_sheet(c)
        if im is not None:
            im.save(OUT / "ref" / f"{slug(c['id'])}.jpg", quality=92)
            refs.append(im)
    for n in range(0, len(refs), 20):
        page = Image.new("RGB", (4 * 384, 5 * 210), "white")
        for k, im in enumerate(refs[n:n + 20]):
            cell = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
            page.paste(cell, ((k % 4) * 384 + (384 - cell.width) // 2, (k // 4) * 210))
        page.save(OUT / "sheets" / f"cast_{n // 20 + 1:02d}.jpg", quality=90)
    write_cards()
    log(f"sheet: {len(refs)} reference sheets, {-(-len(refs) // 20)} contact sheets -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["fetch", "condense", "sheet", "all"])
    step = ap.parse_args().step
    for name, fn in (("fetch", fetch), ("condense", condense), ("sheet", sheet)):
        if step in (name, "all"):
            fn()
