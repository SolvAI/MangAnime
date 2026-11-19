"""Config loading (OmegaConf). `CFG` merges every yaml in configs/ (base.yaml first)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(os.environ.get("KOMI_REPO", Path(__file__).resolve().parents[2]))


@lru_cache(maxsize=1)
def load_config():
    cfg_dir = REPO / "configs"
    cfg = OmegaConf.load(cfg_dir / "base.yaml")
    # every other yaml is namespaced under its file stem (configs/tts.yaml -> CFG.tts) so stages cannot clobber base keys
    for f in sorted(cfg_dir.glob("*.yaml")):
        if f.name in ("base.yaml", "comfy_extra_model_paths.yaml"):
            continue
        try:
            sub = OmegaConf.load(f)
            if f.stem in cfg and OmegaConf.is_dict(cfg[f.stem]):
                cfg[f.stem] = OmegaConf.merge(cfg[f.stem], sub)
            else:
                cfg[f.stem] = sub
        except Exception:
            pass
    try:
        OmegaConf.resolve(cfg)
    except Exception:
        pass
    return cfg


CFG = load_config()


def P(key: str) -> Path:
    """Path helper: P('processed') -> Path(cfg.paths.processed)."""
    return Path(CFG.paths[key])


def load_env() -> None:
    """Load .env into os.environ without ever printing it."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
