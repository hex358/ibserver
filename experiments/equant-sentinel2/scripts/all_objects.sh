#!/usr/bin/env bash
# Все найденные объекты: орбита (GPS + 3D) → поза по пикселям с опорами полос → лист → фотореалистичные кадры.
#   scripts/all_objects.sh <номер воркера> <число воркеров>
cd "$(dirname "$0")/.."
k=$1; K=$2; i=0
# сначала объекты без кадров, затем остальные
names=$( (for f in results/shape/*.pkl; do n=$(basename $f .pkl); [ -f results/realistic/${n}_wide.png ] || echo "0 $n"; done;
          for f in results/shape/*.pkl; do n=$(basename $f .pkl); [ -f results/realistic/${n}_wide.png ] && echo "1 $n"; done) | sort | awk '{print $2}')
for n in $names; do
  i=$((i+1)); [ $(( (i-1) % K )) -eq $k ] || continue
  grep -q "^DONE $n$" results/pipeline/progress.log && continue
  log=results/pipeline/$n.log; mkdir -p results/pipeline; : > $log
  python3 scripts/orbit_check.py $n >> $log 2>&1
  python3 scripts/pose_pix.py $n --n-rot 3000 --band-off 25 --threads 1 >> $log 2>&1 && python3 scripts/pose_write.py $n --src results/pose_pix_off >> $log 2>&1
  python3 scripts/object_sheet.py $n >> $log 2>&1 && echo "SHEET $n" >> results/pipeline/progress.log
  timeout 2400 python3 scripts/realistic_view.py $n --spp 64 --out results/realistic >> $log 2>&1
  python3 scripts/object_sheet.py $n >> $log 2>&1
  echo "DONE $n" >> results/pipeline/progress.log
done
echo "WORKER $k FINISHED" >> results/pipeline/progress.log
