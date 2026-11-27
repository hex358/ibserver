#!/usr/bin/env bash
# Проверка вращения на всех найденных объектах одинаково (results/pose_pix_off/*.json): настоящий снимок (эталон для
# работающих спутников — не вращается), подстановка 0 °/с (ложные срабатывания) и 60 °/с (точность скорости и оси).
#   scripts/spin_validate.sh <номер воркера> <число воркеров> [--shade]
cd "$(dirname "$0")/.."
k=$1; K=$2; SH=$3; SFX=_nongrey; [ "$SH" = "--shade" ] && SFX=_nongrey_shade; i=0
for f in results/pose_pix_off/*.json; do
  n=$(basename $f .json); i=$((i+1)); [ $(( (i-1) % K )) -eq $k ] || continue
  [ -f results/spin_real/${n}_inj60${SFX}.json ] || python3 scripts/spin_real.py $n --nongrey $SH --inject 60 --threads 1 > results/spin_real/${n}_inj60${SFX}.log 2>&1
  [ -f results/spin_real/${n}${SFX}.json ] || python3 scripts/spin_real.py $n --nongrey $SH --threads 1 > results/spin_real/${n}_real${SFX}.log 2>&1
  [ -f results/spin_real/${n}_inj0${SFX}.json ] || python3 scripts/spin_real.py $n --nongrey $SH --inject 0 --threads 1 > results/spin_real/${n}_inj0${SFX}.log 2>&1
  echo "VDONE $n $SFX" >> results/spin_real/progress.log
done
echo "VWORKER $k FINISHED $SFX" >> results/spin_real/progress.log
