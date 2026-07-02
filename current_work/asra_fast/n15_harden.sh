#!/usr/bin/env bash
# Scale the cross-scenario claim from n=6 to n=15 independently-trained policies per scenario, to
# answer the reviewer's "n=6 is few; ask for >=15" directly. Trains seeds 6..14 (9 more) with the
# SAME config as seeds 0..5, then re-runs the matched-CR multi-policy eval + decode-matched acid test
# on all 15 -> *_n15.json. Sign tests + policy-level bootstrap reported by decode_matched/analyze.
set -u; cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies; RES=current_work/results_v3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[n15] training seeds 6..14 for merge + roundabout"
for scen in merge roundabout; do for s in 6 7 8 9 10 11 12 13 14; do
  if [ -f $BP/scn_${scen}_seed${s}_final.pt ]; then continue; fi
  nohup python -c "import sys;sys.path.insert(0,'.');from current_work.asra_fast.multiscenario import train_scn;train_scn('${scen}',steps=80000,seed=${s},vehicles=8,density=1.0)" > $BP/train_scn_${scen}_${s}.log 2>&1 &
  # throttle: at most ~8 concurrent trainings
  while [ "$(jobs -r | wc -l)" -ge 8 ]; do sleep 3; done
done; done
wait
echo "[n15] all trainings done"

for scen in merge roundabout; do
  POLS=$(ls $BP/scn_${scen}_seed*_final.pt | sort -t d -k3 -n | tr '\n' ' ')
  echo "[n15] $scen: $(echo $POLS | wc -w) policies"
  python -m current_work.asra_fast.multipolicy --policies $POLS --out $RES/multipolicy_${scen}_n15.json \
     --seeds 8 --n_ep 40 --n_proc 10 --max_steps 120 2>&1 | grep -vi "warn\|deprecat" | tail -8
done
echo "[n15] eval done — decode-matched + Holm(real p) analysis"
python current_work/asra_fast/decode_matched.py $RES/multipolicy_merge_n15.json $RES/multipolicy_roundabout_n15.json > $RES/decode_matched_n15.txt 2>&1
cat $RES/decode_matched_n15.txt
for j in multipolicy_merge_n15 multipolicy_roundabout_n15; do
  python -c "import sys;sys.path.insert(0,'.');from current_work.asra_fast.multipolicy import analyze;print('=== $j ===');analyze('$RES/$j.json')" 2>&1 | grep -vi "warn\|deprecat"
done
echo "[n15] ALL DONE"
