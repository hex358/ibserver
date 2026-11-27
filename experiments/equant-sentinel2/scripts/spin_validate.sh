#!/usr/bin/env bash
# Проверка вращения на всех найденных объектах одинаково (results/pose_pix_off/*.json): настоящий снимок (эталон для
# работающих спутников — не вращается), подстановка 0 °/с (ложные срабатывания) и 60 °/с (точность скорости и оси).
#   scripts/spin_validate.sh <номер воркера> <число воркеров>
cd "$(dirname "$0")/.."
k=$1; K=$2; i=0
for f in results/pose_pix_off/*.json; do
  n=$(basename $f .json); i=$((i+1)); [ $(( (i-1) % K )) -eq $k ] || continue
  [ -f results/spin_real/${n}_inj60_nongrey.json ] || python3 scripts/spin_real.py $n --nongrey --inject 60 --threads 1 > results/spin_real/${n}_inj60.log 2>&1
  [ -f results/spin_real/${n}_nongrey.json ] || python3 scripts/spin_real.py $n --nongrey --threads 1 > results/spin_real/${n}_real.log 2>&1
  [ -f results/spin_real/${n}_inj0_nongrey.json ] || python3 scripts/spin_real.py $n --nongrey --inject 0 --threads 1 > results/spin_real/${n}_inj0.log 2>&1
  echo "VDONE $n" >> results/spin_real/progress.log
done
echo "VWORKER $k FINISHED" >> results/spin_real/progress.log
