#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until [ $(grep -c "^VWORKER .* FINISHED" results/spin_real/progress.log) -ge 2 ]; do sleep 60; done
scripts/spin_validate0.sh 0 2 & scripts/spin_validate0.sh 1 2 & wait
python3 scripts/spin_summary.py > results/spin_real/summary.txt 2>&1
echo "ALLVALID DONE" >> results/spin_real/progress.log
