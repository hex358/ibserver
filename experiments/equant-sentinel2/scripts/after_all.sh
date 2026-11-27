#!/usr/bin/env bash
# После конвейера всех объектов: карточки (мусор и параметры) по позам по пикселям, коллаж, коммит.
cd "$(dirname "$0")/.."
until [ $(grep -c "WORKER .* FINISHED" results/pipeline/progress.log) -ge 2 ]; do sleep 30; done
python3 scripts/debris_card.py --spin none > results/pipeline/cards.log 2>&1
python3 scripts/collage.py > results/pipeline/collage.log 2>&1
echo "AFTERALL DONE" >> results/pipeline/progress.log
