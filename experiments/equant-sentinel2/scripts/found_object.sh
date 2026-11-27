#!/usr/bin/env bash
# конвейер для найденного объекта: окна → орбита → поза по пикселям → запись позы → лист → вращение (настоящий, 60, 0)
cd /home/user/ibserver/experiments/equant-sentinel2; ev=$1; n=$(basename $ev .json); L=results/pipeline_$n.log
PREP_UNTIL=20261127T1755 python3 scripts/shape_fit.py prep $ev >> $L 2>&1
[ -f results/orbits/$n.json ] || python3 scripts/orbit_check.py $n >> $L 2>&1
python3 scripts/pose_pix.py $n --band-off 25 --nongrey --threads 2 > results/pose_pix_$n.log 2>&1 || { echo POSEFAIL >> $L; exit 1; }
python3 scripts/pose_write.py $n >> $L 2>&1
python3 scripts/object_sheet.py $n >> $L 2>&1
for m in "" "--inject 60" "--inject 0"; do t=real; [ -n "$m" ] && t=inj${m#--inject }; python3 scripts/spin_real.py $n --nongrey $m --threads 2 > results/spin_real/${n}_${t}_nongrey.log 2>&1; done
echo PIPEDONE >> $L
