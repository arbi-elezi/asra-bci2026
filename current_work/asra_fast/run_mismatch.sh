#!/usr/bin/env bash
# Risk-profile-mismatch protocol (the paper's actual use-case): train policies in an EASY
# context (low density -> risk-seeking greedy that accelerates), DEPLOY in a HARD context
# (high density). Confirm across seeds that ASRA reduces CR (H1) and dominates the swept
# brake rule (H3) at deployment -- a seed-robust positive.
set -u
cd /d/bci-2026 2>/dev/null || cd "D:/bci-2026"
BP=current_work/base_policies; RES=current_work/results_v3
SEEDS_LIST="1 2 3 4 5 6"
TOTAL=${TOTAL:-90000}
TRV=4; TRD=0.5; HSR=0.6          # easy training context
EVV=15; EVD=1.5                  # hard deployment context

echo "[mismatch] training easy policies (seeds $SEEDS_LIST)..."
for s in $SEEDS_LIST; do
  [ -f $BP/easy_seed${s}_final.pt ] && continue
  nohup python -m current_work.asra_fast.trainer --total $TOTAL --seed $s --tag easy \
      --vehicles $TRV --density $TRD --high_speed_reward $HSR \
      > $BP/train_easy_seed${s}.log 2>&1 &
done
for s in $SEEDS_LIST; do until [ -f $BP/easy_seed${s}_final.pt ]; do sleep 10; done; done
echo "[mismatch] trainings done."

for s in $SEEDS_LIST; do
  base=$BP/easy_seed${s}_final.pt; fish=$BP/easy_seed${s}_fisher.pt
  python -m current_work.asra_fast.prep_fisher --base "$base" --out "$fish" --n_dref 2500 2>&1 | grep -vi "warn\|deprecat" | tail -1
  echo "[mismatch] === seed $s greedy @deploy ($EVV veh, d$EVD) ==="
  python -m current_work.asra_fast.diagnose_policy --base "$base" --vehicles $EVV --density $EVD --n_ep 40 2>&1 | grep -vi "warn\|deprecat" | grep greedy > $RES/mm_seed${s}_diag.txt
  cat $RES/mm_seed${s}_diag.txt
  python -m current_work.asra_fast.experiments --base "$base" --fisher "$fish" \
      --out $RES/mm_seed${s}.json --seeds 10 --n_ep 40 --n_proc 14 --max_steps 120 \
      --eval_vehicles $EVV --eval_density $EVD --scenario mm$s 2>&1 | grep -vi "warn\|deprecat" | tail -2
  python -m current_work.asra_fast.analyze_results --path $RES/mm_seed${s}.json 2>&1 | grep -vi "warn\|deprecat" > $RES/mm_seed${s}_analysis.txt
  echo "[mismatch] seed $s: $(grep -E '^H1|^H3' $RES/mm_seed${s}_analysis.txt | tr '\n' ' ')"
done
echo "[mismatch] ALL DONE"
