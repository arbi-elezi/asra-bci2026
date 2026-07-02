#!/usr/bin/env bash
# Harden the v03 cross-scenario claim from n=3 to n=6 policies per scenario (matching driving), to
# kill the "underpowered / 3 policies" reviewer caveat on "ASRA-greedy beats the myopic mask on merge".
# Trains seeds 3,4,5 with the SAME config as seeds 0,1,2 (steps=80000, vehicles=8, density=1.0,
# hsr=0.6 default), then re-runs the multi-policy matched-CR eval on all 6 -> *_n6.json (n=3 files kept).
set -u; cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies; RES=current_work/results_v3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1     # keep tiny-MLP trainings from oversubscribing the LLM's CPU-side env stepping

echo "[n6] training seeds 3,4,5 for merge + roundabout (identical config to seeds 0-2)"
for scen in merge roundabout; do for s in 3 4 5; do
  if [ -f $BP/scn_${scen}_seed${s}_final.pt ]; then echo "[n6] $scen seed$s exists, skip"; continue; fi
  nohup python -c "import sys;sys.path.insert(0,'.');from current_work.asra_fast.multiscenario import train_scn;train_scn('${scen}',steps=80000,seed=${s},vehicles=8,density=1.0)" > $BP/train_scn_${scen}_${s}.log 2>&1 &
done; done
wait
echo "[n6] all trainings done"

for scen in merge roundabout; do
  POLS=$(ls $BP/scn_${scen}_seed*_final.pt | sort | tr '\n' ' ')
  echo "[n6] $scen policies: $POLS"
  python -m current_work.asra_fast.multipolicy --policies $POLS --out $RES/multipolicy_${scen}_n6.json \
     --seeds 8 --n_ep 40 --n_proc 8 --max_steps 120 2>&1 | grep -vi "warn\|deprecat" | tail -10
done
echo "[n6] eval done — running decode-matched analysis"
python current_work/asra_fast/decode_matched.py $RES/multipolicy_merge_n6.json $RES/multipolicy_roundabout_n6.json > $RES/decode_matched_n6.txt 2>&1
cat $RES/decode_matched_n6.txt
echo "[n6] ALL DONE"
