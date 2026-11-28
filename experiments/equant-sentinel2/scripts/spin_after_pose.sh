#!/usr/bin/env bash
# После позы по пикселям (results/pose_pix_off/<имя>.json) — вращение: настоящий снимок, подстановки 60 °/с (без света
# и с освещённой истиной чужой моделью отражения) и 0 °/с (освещённая истина). Одинаково для всех объектов.
#   scripts/spin_after_pose.sh <имя> [потоков]
cd "$(dirname "$0")/.."; n=$1; th=${2:-1}
until [ -f results/pose_pix_off/$n.json ]; do pgrep -f "pose_pix.py $n" > /dev/null || { echo "нет позы $n"; exit 1; }; sleep 30; done
# опоры снимков = геометрия + подогнанные позой сдвиги (решатель вращения сдвиги опор не подгоняет: без этого шага
# несовпадение опор между снимками объясняется «вращением» — 30245/32851 со всеми снимками, выигрыш 91 и 816)
python3 scripts/pose_write.py $n > results/pose_write_$n.log 2>&1 || { echo "pose_write не прошёл для $n"; exit 1; }
python3 scripts/spin_real.py $n --nongrey --threads $th > results/spin_real/${n}_real_nongrey.log 2>&1
python3 scripts/spin_real.py $n --nongrey --inject 60 --threads $th > results/spin_real/${n}_inj60_nongrey.log 2>&1
python3 scripts/spin_real.py $n --nongrey --truth mismatch --inject 60 --threads $th > results/spin_real/${n}_inj60_tmm.log 2>&1
python3 scripts/spin_real.py $n --nongrey --truth mismatch --inject 0 --threads $th > results/spin_real/${n}_inj0_tmm.log 2>&1
echo SPINDONE >> results/spin_real/${n}_inj0_tmm.log
