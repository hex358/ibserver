#!/usr/bin/env bash
# Подстановка неподвижного объекта (0 °/с) на геометрии каждого пролёта: находит ли решатель «вращение» в шуме
# (ложное срабатывание на этой геометрии). Запускать после spin_validate.sh.
cd "$(dirname "$0")/.."
k=$1; K=$2; i=0
for f in results/pose_pix_off/*.json; do
  n=$(basename $f .json); i=$((i+1)); [ $(( (i-1) % K )) -eq $k ] || continue
  [ -f results/spin_real/${n}_inj0_nongrey.json ] || python3 scripts/spin_real.py $n --nongrey --inject 0 > results/spin_real/${n}_inj0.log 2>&1
  echo "V0DONE $n" >> results/spin_real/progress.log
done
echo "V0WORKER $k FINISHED" >> results/spin_real/progress.log
