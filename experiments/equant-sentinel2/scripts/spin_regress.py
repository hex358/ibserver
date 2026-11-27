"""Регрессия решателя вращения: текущие результаты (results/spin_real/*.json) против эталона
(results/spin_real/baseline.json) — по каждому объекту. Падает (код 1), если хоть один объект стал хуже.

  python scripts/spin_regress.py            # сравнить
  python scripts/spin_regress.py --save     # записать текущие результаты эталоном (только если сравнение прошло)

Метрики по объекту (только пролёты, где вращение измеримо — подстановка 60 °/с подтверждена вращением):
  * ошибка скорости на подстановке 60 °/с, % (хуже — если выросла больше чем на 5 п. п.);
  * ошибка оси с учётом зеркального двойника, ° (хуже — если выросла больше чем на 10°);
  * ложное срабатывание на подстановке 0 °/с (хуже — если появилось);
  * вердикт по настоящему снимку работающего спутника (эталон — не вращается; хуже — если стал «вращается»).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/spin_real"


SFX = "_nongrey" + ("_shade" if "--shade" in sys.argv else "")


def current():
    out = {}
    for f in D.glob(f"*_inj60{SFX}.json"):
        n = f.name.replace(f"_inj60{SFX}.json", "")
        r60 = json.loads(f.read_text())
        f0, fr = D / f"{n}_inj0{SFX}.json", D / f"{n}{SFX}.json"
        r0 = json.loads(f0.read_text()) if f0.exists() else None
        rr = json.loads(fr.read_text()) if fr.exists() else None
        inj = r60.get("inject", {})
        out[n] = dict(rate_err_pct=abs(r60["all"]["rate"] - 60.0) / 60.0 * 100, measured=bool(r60.get("spinning")),
                      axis_err_deg=inj.get("axis_err_mod_mirror_deg", inj.get("axis_err_deg")),
                      axis_err_signed_deg=inj.get("axis_err_deg"),
                      false_pos=None if r0 is None else bool(r0.get("spinning")),
                      real_spin=None if rr is None else bool(rr.get("spinning")))
    return out


cur = current(); bf = D / ("baseline_shade.json" if "--shade" in sys.argv else "baseline.json")
base = json.loads(bf.read_text()) if bf.exists() else {}
worse = []
print(f"{'объект':32s} | {'60 °/с: ошибка скорости':24s} | {'ошибка оси':14s} | {'ложн. на 0':10s} | {'настоящий':10s} | сравнение с эталоном")
for n in sorted(cur):
    c, b = cur[n], base.get(n)
    notes = []
    if b:
        if c["rate_err_pct"] > b["rate_err_pct"] + 5:
            notes.append(f"скорость хуже ({b['rate_err_pct']:.0f}% → {c['rate_err_pct']:.0f}%)")
        if c["axis_err_deg"] is not None and b["axis_err_deg"] is not None and c["axis_err_deg"] > b["axis_err_deg"] + 10:
            notes.append(f"ось хуже ({b['axis_err_deg']:.0f}° → {c['axis_err_deg']:.0f}°)")
        if c["false_pos"] and not b["false_pos"]:
            notes.append("появилось ложное срабатывание")
        if c["real_spin"] and not b["real_spin"]:
            notes.append("работающий спутник стал «вращается»")
    if notes:
        worse.append(n)
    ax = "—" if c["axis_err_deg"] is None else f"{c['axis_err_deg']:.0f}°"
    print(f"{n:32s} | {c['rate_err_pct']:5.0f}% {'(подтв.)' if c['measured'] else '(не подтв.)':16s} | {ax:14s} | "
          f"{'—' if c['false_pos'] is None else ('ДА' if c['false_pos'] else 'нет'):10s} | "
          f"{'—' if c['real_spin'] is None else ('вращ.' if c['real_spin'] else 'нет'):10s} | "
          f"{'; '.join(notes) if notes else ('эталона нет' if not b else 'не хуже')}")
print(f"\nИТОГ: {'ХУЖЕ на ' + str(len(worse)) + ' объектах: ' + ', '.join(worse) if worse else 'ни один объект не стал хуже'}")
if "--save" in sys.argv and not worse:
    bf.write_text(json.dumps(cur, ensure_ascii=False, indent=1)); print(f"эталон записан: {bf}")
sys.exit(1 if worse else 0)
