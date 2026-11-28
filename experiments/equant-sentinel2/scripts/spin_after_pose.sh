#!/usr/bin/env bash
# После позы по пикселям (results/pose_pix_off/<имя>.json) — вращение: настоящий снимок, подстановки 60 °/с (без света
# и с освещённой истиной чужой моделью отражения) и 0 °/с (освещённая истина). Одинаково для всех объектов.
#   scripts/spin_after_pose.sh <имя> [потоков]
cd "$(dirname "$0")/.."; n=$1; th=${2:-1}
until [ -f results/pose_pix_off/$n.json ]; do pgrep -f "pose_pix.py $n" > /dev/null || { echo "нет позы $n"; exit 1; }; sleep 30; done
python3 scripts/spin_real.py $n --nongrey --threads $th > results/spin_real/${n}_real_nongrey.log 2>&1
python3 scripts/spin_real.py $n --nongrey --inject 60 --threads $th > results/spin_real/${n}_inj60_nongrey.log 2>&1
python3 scripts/spin_real.py $n --nongrey --truth mismatch --inject 60 --threads $th > results/spin_real/${n}_inj60_tmm.log 2>&1
python3 scripts/spin_real.py $n --nongrey --truth mismatch --inject 0 --threads $th > results/spin_real/${n}_inj0_tmm.log 2>&1
echo SPINDONE >> results/spin_real/${n}_inj0_tmm.log
