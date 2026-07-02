#!/usr/bin/env bash
set -u; cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies; RES=current_work/results_v3
# train 2 more policies per scenario (seed0 already exists) -> 3 policies each, low CPU, parallel
for scen in merge roundabout; do for s in 1 2; do
  [ -f $BP/scn_${scen}_seed${s}_final.pt ] && continue
  nohup python -c "import sys;sys.path.insert(0,'.');from current_work.asra_fast.multiscenario import train_scn;train_scn('${scen}',steps=80000,seed=${s},vehicles=8,density=1.0)" > $BP/train_scn_${scen}_${s}.log 2>&1 &
done; done
wait
echo "[scn-mp] scenario policies trained"
# wait for the driving multipolicy run to free the big core pool
until grep -q "saved current_work.results_v3.multipolicy_driving.json" $RES/multipolicy_driving.log 2>/dev/null; do sleep 20; done
echo "[scn-mp] driving done; running scenario multipolicy eval"
for scen in merge roundabout; do
  POLS=$(ls $BP/scn_${scen}_seed*_final.pt | tr '\n' ' ')
  python -m current_work.asra_fast.multipolicy --policies $POLS --out $RES/multipolicy_${scen}.json \
     --seeds 8 --n_ep 40 --n_proc 12 --max_steps 120 2>&1 | grep -vi "warn\|deprecat" | tail -8
done
echo "[scn-mp] ALL DONE"
