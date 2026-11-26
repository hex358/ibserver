#!/usr/bin/env bash
# Этап 3 для одного объекта — всё по пикселям, без законов ориентации и без подбора на глаз:
#   1. орбита: положение по точной орбите Sentinel (GPS + 3D-углы) → опоры полос (scripts/orbit_check.py);
#   2. поза по пикселям: корпус по всей SO(3) + угол каждого крыла, градиент; σ, другие минимумы, перекрёстная
#      проверка и подстановка (scripts/pose_pix.py --checks --write);
#   3. фотометрия: интегральный контраст рендера против снимка в R, G, B → измеренная поправка яркости
#      (scripts/photometry_check.py);
#   4. лист: каналы снимка / модель / остаток (scripts/object_sheet.py);
#   5. фотореалистичные кадры с поправкой яркости, Солнце и свет Земли на момент съёмки (scripts/realistic_view.py).
#
#   scripts/object_pipeline.sh STARLINK-30245_20261120T110322 [ещё имена...]
set -u
cd "$(dirname "$0")/.."
for n in "$@"; do
  log=results/pipeline_$n.log; : > "$log"
  [ -f results/orbits/$n.json ] || python3 scripts/orbit_check.py "$n" >> "$log" 2>&1
  python3 scripts/pose_pix.py "$n" --checks --write >> "$log" 2>&1 || { echo "POSEFAIL $n" >> "$log"; continue; }
  python3 scripts/photometry_check.py "$n" >> "$log" 2>&1
  python3 scripts/object_sheet.py "$n" >> "$log" 2>&1
  timeout 3000 python3 scripts/realistic_view.py "$n" --spp 128 --out results/realistic >> "$log" 2>&1
  echo "PIPEDONE $n" >> "$log"
done
