#!/usr/bin/env bash
set -u; cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
for scen in merge roundabout; do
  echo "[scenarios] === $scen ==="
  python -m current_work.asra_fast.multiscenario --scenario $scen --steps 100000 \
      --seeds 6 --n_ep 40 --n_proc 8 --max_steps 100 --vehicles 8 --density 1.0 \
      2>&1 | grep -vi "warn\|deprecat\|pkg_resources"
done
echo "[scenarios] ALL DONE"
