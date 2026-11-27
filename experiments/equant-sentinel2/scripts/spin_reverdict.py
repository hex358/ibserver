"""Пересчёт вердиктов вращения по сохранённым результатам (results/spin_real/*.json) после изменения правила
решения — без пересчёта решателя: выигрыш над статикой, порог и выигрыши по отложенным моментам уже записаны.
Прежний вердикт сохраняется в поле verdict_prev. Центр масс для новых «вращается» здесь не считается (нужен решатель).
  python scripts/spin_reverdict.py
"""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from s2sat.diffpose import cv_decision

for f in sorted((ROOT / "results/spin_real").glob("*.json")):
    if f.name.startswith("baseline"):
        continue
    r = json.loads(f.read_text())
    if "cv" not in r or "per_cluster" not in r["cv"]:
        continue
    a = r["all"]; kap = a["thr"] / 3 / math.log(max(2, round(math.exp(1))) ) if False else None
    d = list(r["cv"]["per_cluster"].values())
    # κ: порог отложенных = 2κ·K записан → κ = threshold / (2K)
    K = max(1, len(d)); kap = r["cv"]["threshold"] / (2 * K) if d else 1.0
    old = r["verdict"]
    if not r.get("identifiable", True):
        continue
    if a["dL"] <= a["thr"] or not d:
        continue
    cons, tot, thr, nb, ninf = cv_decision(d, kap)
    spinning = cons
    verdict = ("ВРАЩАЕТСЯ (вращение предсказывает отложенные моменты съёмки)" if cons else
               "НЕ вращается: «вращение» подстраивается под полосы и не предсказывает отложенные моменты")
    if verdict != old:
        r.update(verdict_prev=old, verdict=verdict, spinning=spinning)
        r["cv"].update(confirmed=cons, n_better=nb, n_informative=ninf)
        f.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"{f.name}: {old[:40]} → {verdict[:40]} (лучше {nb} из {ninf} информативных, сумма {tot:+.0f}, порог {thr:.0f})")
print("готово")
