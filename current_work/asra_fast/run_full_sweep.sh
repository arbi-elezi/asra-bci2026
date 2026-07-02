#!/usr/bin/env bash
# Full decode-controlled frontier sweep across base-policy regimes.
# Waits for trainings, preps Fisher, runs matrices (assertive first = richest frontier).
set -u
cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies
RES=current_work/results_v3
SEEDS=${SEEDS:-18}
NEP=${NEP:-60}
NPROC=${NPROC:-14}

echo "[sweep] waiting for assertive + moderate 200k snapshots..."
until [ -f $BP/assertive_seed0_step200000.pt ] && [ -f $BP/moderate_seed0_step200000.pt ]; do sleep 15; done
echo "[sweep] trainings done."

# regime: tag  base_snapshot
declare -A BASE=(
  [assertive]=$BP/assertive_seed0_step200000.pt
  [dense]=$BP/highway_seed0_step200000.pt
  [moderate]=$BP/moderate_seed0_step200000.pt
  [light]=$BP/light_seed0_step200000.pt
)

for tag in assertive dense moderate light; do
  base=${BASE[$tag]}
  fish=$BP/${tag}_final_fisher.pt
  echo "[sweep] === $tag ==="
  [ -f "$fish" ] || python -m current_work.asra_fast.prep_fisher --base "$base" --out "$fish" --n_dref 3000 2>&1 | grep -vi "warn\|deprecat" | tail -1
  python -m current_work.asra_fast.experiments --base "$base" --fisher "$fish" \
      --out $RES/${tag}_final.json --seeds $SEEDS --n_ep $NEP --n_proc $NPROC \
      --max_steps ${MAXSTEPS:-150} --scenario $tag 2>&1 | grep -vi "warn\|deprecat" | tail -3
  python -m current_work.asra_fast.analyze_results --path $RES/${tag}_final.json 2>&1 | grep -vi "warn\|deprecat" > $RES/${tag}_analysis.txt
  echo "[sweep] $tag analysis -> $RES/${tag}_analysis.txt"
done
echo "[sweep] ALL DONE"
