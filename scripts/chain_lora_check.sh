#!/usr/bin/env bash
# After the v2 evaluation: is the higher flicker of the v2 Wan clips caused by the Civitai anime-style LoRA?
# Re-render every 12th v2 shot from the SAME keyframes with base Wan (no LoRA) and compare.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 GPU_PRIO=1 GPU_WAIT_MIN=900 KOMI_KF_BACKEND=illustrious
R="$KOMI_WORK/runs"
until grep -q "CHAIN_FINAL_V2_DONE" "$KOMI_WORK/logs/chain_final_v2.log"; do sleep 20; done
echo "=== $(date +%T) re-render 24 v2 shots without the anime LoRA"
python scripts/make_variant_run.py --src ep22_v2 --dst ep22_v2_nolora --every 12 --offset 4 --only-wan
python -m src.generate.pipeline --run ep22_v2_nolora --stages video --mode threes 2>&1 | grep -v "Warn\|warn" | tail -2
python -m src.common.servers stop comfy
echo "=== $(date +%T) A/B: no-LoRA clip (label wan) vs the v2 clip with the LoRA (label base)"
python -m src.eval.ab_judge --run ep22_v2_nolora --versus base 2>&1 | grep -v "Warn\|warn" | grep "judged"
python -m src.eval.ab_judge --run ep22_v2_nolora --versus base --no-metrics --tag frames-only 2>&1 | grep -v "Warn\|warn" | grep "judged"
python - <<'PY'
import statistics as S
from src.common.io import read_json
from src.common.config import P
m = read_json(P("runs") / "ep22_v2_nolora" / "eval" / "ab_metrics_base.json") or {}
for k in ("flicker_index", "warp_error", "mean_flow", "overall", "held_frame_ratio"):
    a = [v["wan"][k] for v in m.values() if isinstance((v.get("wan") or {}).get(k), (int, float))]
    b = [v["base"][k] for v in m.values() if isinstance((v.get("base") or {}).get(k), (int, float))]
    if a and b:
        print(f"  {k:18s} no LoRA {S.mean(a):.3f} | with anime LoRA {S.mean(b):.3f}   (n={len(a)})")
PY
echo "=== $(date +%T) CHAIN_LORA_CHECK_DONE"
