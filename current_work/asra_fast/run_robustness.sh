#!/usr/bin/env bash
# Multi-policy robustness: train several independent policies (moderate config, which tends
# to produce risk-seeking greedy), classify each by greedy brake-fraction, run the decode-
# controlled frontier matrix on each, and confirm the H1/H3 signs across the risk-seeking set.
set -u
cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies; RES=current_work/results_v3
SEEDS_LIST="1 2 3 4 5 6"
TOTAL=${TOTAL:-120000}
VEH=6; DEN=1.0; HSR=0.4

echo "[robust] training policies (seeds $SEEDS_LIST)..."
for s in $SEEDS_LIST; do
  nohup python -m current_work.asra_fast.trainer --total $TOTAL --seed $s --tag rob \
      --vehicles $VEH --density $DEN --high_speed_reward $HSR \
      > $BP/train_rob_seed${s}.log 2>&1 &
done
# wait for all final snapshots
for s in $SEEDS_LIST; do
  until [ -f $BP/rob_seed${s}_final.pt ]; do sleep 10; done
done
echo "[robust] all trainings done."

for s in $SEEDS_LIST; do
  base=$BP/rob_seed${s}_final.pt; fish=$BP/rob_seed${s}_fisher.pt
  python -m current_work.asra_fast.prep_fisher --base "$base" --out "$fish" --n_dref 2500 2>&1 | grep -vi "warn\|deprecat" | tail -1
  echo "[robust] === seed $s diagnose ==="
  python -m current_work.asra_fast.diagnose_policy --base "$base" --vehicles $VEH --density $DEN --n_ep 40 2>&1 | grep -vi "warn\|deprecat" > $RES/rob_seed${s}_diag.txt
  cat $RES/rob_seed${s}_diag.txt
  python -m current_work.asra_fast.experiments --base "$base" --fisher "$fish" \
      --out $RES/rob_seed${s}.json --seeds 10 --n_ep 40 --n_proc 14 --max_steps 120 \
      --scenario rob$s 2>&1 | grep -vi "warn\|deprecat" | tail -2
  python -m current_work.asra_fast.analyze_results --path $RES/rob_seed${s}.json 2>&1 | grep -vi "warn\|deprecat" > $RES/rob_seed${s}_analysis.txt
  echo "[robust] seed $s: $(grep -E '^H1|^H3' $RES/rob_seed${s}_analysis.txt | tr '\n' ' ')"
done
echo "[robust] ALL DONE"
