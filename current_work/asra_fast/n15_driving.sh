#!/usr/bin/env bash
# Fix P0-2/P0-3 (n=6 underpowered for the sign test + Holm): train 9 MORE independently-seeded
# driving policies (seed 7..15, same easy config as easy_seed1..6: vehicles=4 density=0.5 hsr=0.6),
# then re-run the principled consequence-shaped comparison at n=15 (deploy-hard 15/1.5). With enough
# independent policies the exact sign test can reach significance and survive Holm-Bonferroni.
set -u; cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies; RES=current_work/results_v3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[n15drv] training seed 7..15 (easy: veh4 den0.5 hsr0.6, 90k steps)"
for s in 7 8 9 10 11 12 13 14 15; do
  [ -f $BP/easy_seed${s}_final.pt ] && continue
  nohup python -c "import sys;sys.path.insert(0,'.');
from current_work.asra_fast.multiscenario import train_scn
import torch
a,n=train_scn('highway',steps=90000,seed=${s},hidden=64,vehicles=4,density=0.5,hsr=0.6),64
" > $BP/train_easy_${s}.log 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge 8 ]; do sleep 3; done
done
wait
echo "[n15drv] training done; collecting policies"
# train_scn saves scn_highway_seed{s}_final.pt; symlink/copy naming into the principled glob set
for s in 7 8 9 10 11 12 13 14 15; do
  [ -f $BP/scn_highway_seed${s}_final.pt ] && cp -f $BP/scn_highway_seed${s}_final.pt $BP/easy_seed${s}_final.pt
done
POLS=$(ls $BP/easy_seed*_final.pt | sort -t d -k3 -n | tr '\n' ' ')
echo "[n15drv] policies: $(echo $POLS | wc -w)"
python current_work/asra_fast/principled_asra.py --policies $POLS --seeds 6 --n_ep 30 --n_proc 12 \
  --out $RES/principled_asra_n15.json 2>&1 | grep -vi "warn\|deprecat" | tail -12
echo "[n15drv] ALL DONE"
