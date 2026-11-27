"""Сводка проверки вращения по всем объектам (results/spin_real), одинаково для всех:
  * настоящий снимок — вердикт решателя;
  * подстановка 0 °/с на той же геометрии — ложное срабатывание (находит ли «вращение» в шуме);
  * подстановка 60 °/с — точность скорости и оси на этой геометрии.
Итог по объекту: вердикт по настоящему снимку принимается, только если на этой геометрии решатель не путается
(0 °/с → «не вращается», 60 °/с → скорость ±15 %); иначе — «геометрия пролёта не определяет вращение».
"""
import json, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/spin_real"


import sys
SFX = "_nongrey" + ("_shade" if "--shade" in sys.argv else "")


def load(n, tag):
    f = D / f"{n}{tag}{SFX}.json"
    return json.loads(f.read_text()) if f.exists() else None


names = sorted({Path(f).name.split("_inj")[0].replace(f"{SFX}.json", "") for f in glob.glob(str(D / f"*{SFX}.json"))})
print(f"{'объект':32s} | {'настоящий снимок':42s} | {'подст. 0 °/с':14s} | {'подст. 60 °/с: найдено, ось (до двойника)':28s} | итог")
for n in names:
    r, r0, r60 = load(n, ""), load(n, "_inj0"), load(n, "_inj60")
    v = (r or {}).get("verdict", "—")
    fa = None if r0 is None else bool(r0.get("spinning"))
    s60 = None if r60 is None else (r60["all"]["rate"], r60.get("inject", {}).get("axis_err_mod_mirror_deg", r60.get("inject", {}).get("axis_err_deg")),
                                     bool(r60.get("spinning")))
    ok_geo = (fa is False) and s60 is not None and abs(s60[0] - 60) / 60 <= 0.15 and s60[2]
    if r is None:
        final = "—"
    elif fa is None or s60 is None:
        final = "проверка геометрии не закончена"
    elif r.get("identifiable") is False:
        final = "вращение по пролёту не определяется (< 3 моментов съёмки)"
    elif ok_geo:
        final = ("ВРАЩАЕТСЯ %.1f °/с" % r["all"]["rate"]) if r.get("spinning") else "не вращается"
        if r.get("spinning"):
            a_ = r["all"]; final += ", направление " + ("определено" if a_.get("direction_resolved") else "до зеркального двойника")
            if r.get("com"):
                final += ", центр масс ⊥ оси ±%.1f м" % max(r["com"]["sigma_perp_m"])
    else:
        final = "геометрия пролёта не определяет вращение" + (" (ложное срабатывание на 0 °/с)" if fa else "") + \
                (" (60 °/с → %.0f °/с)" % s60[0] if s60 and abs(s60[0] - 60) / 60 > 0.15 else "")
    t0 = "—" if fa is None else ("ЛОЖНОЕ" if fa else "чисто")
    t60 = "—" if s60 is None else f"{s60[0]:.1f} °/с, {s60[1]:.0f}°" if s60[1] is not None else f"{s60[0]:.1f} °/с"
    print(f"{n:32s} | {v[:42]:42s} | {t0:14s} | {t60:28s} | {final}")
