"""Cast catalogue for v3: the character cards (data/processed/cards/cards.json, built from the wiki) with canonical ids.

Every stage names people by card id ("Komi", "Tadano", "Onemine" ...). People without a card keep the id the reader gave
them ("girl_A"); voices of nobody in particular are "Narrator" and "Classmates".
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.common.config import P
from src.common.io import read_json

GENERIC = {"crowd": "Classmates", "classmates": "Classmates", "classmate": "Classmates", "students": "Classmates", "student": "Classmates",
           "everyone": "Classmates", "narrator": "Narrator", "narration": "Narrator"}
_HON = re.compile(r"[-\s](san|kun|chan|sama|senpai|sensei)$", re.I)


@lru_cache(maxsize=1)
def cards() -> dict[str, dict]:
    return {c["id"]: c for c in (read_json(P("processed") / "cards" / "cards.json") or [])}


@lru_cache(maxsize=1)
def _alias() -> dict[str, str]:
    m: dict[str, str] = {}
    cs = cards()
    for cid, c in cs.items():                       # aliases first, ids last so an id always wins a clash
        for a in [c.get("full_name"), *(c.get("aliases") or [])]:
            if a:
                m.setdefault(str(a).lower().strip(), cid)
                m.setdefault(_HON.sub("", str(a).lower().strip()), cid)
    for cid in cs:
        m[cid.lower()] = cid
    return m


def canon(name: str | None) -> str:
    n = str(name or "").strip()
    if not n:
        return "Classmates"
    low = n.lower()
    if low in GENERIC:
        return GENERIC[low]
    a = _alias()
    return a.get(low) or a.get(_HON.sub("", low)) or n


def card(cid: str) -> dict | None:
    return cards().get(canon(cid))


def gender(cid: str, hint: str = "") -> str:
    """'f' | 'm' | 'o' for the image model's count tags."""
    c = card(cid)
    g = str((c or {}).get("gender") or "").lower()
    txt = f"{cid} {hint}".lower()
    if g.startswith("f") or (not c and re.search(r"girl|woman|mother|sister|lady|female", txt)):
        return "f"
    if g.startswith("m") or (not c and re.search(r"boy|man\b|father|brother|male|guy", txt)):
        return "m"
    return "o"


def is_known(cid: str) -> bool:
    return canon(cid) in cards()
