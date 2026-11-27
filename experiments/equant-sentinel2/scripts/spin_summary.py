"""Сводка проверки вращения по всем объектам (results/spin_real): настоящий снимок и подстановка 60 °/с."""
import json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(glob.glob(str(ROOT / "results/spin_real/*_nongrey.json"))):
    if "_inj" in f:
        continue
    n = Path(f).name.replace("_nongrey.json", ""); r = json.loads(Path(f).read_text())
    fi = ROOT / "results/spin_real" / f"{n}_inj60_nongrey.json"; ri = json.loads(fi.read_text()) if fi.exists() else None
    rows.append((n, r.get("verdict", ""), ri["all"]["rate"] if ri else None, ri["inject"]["axis_err_deg"] if ri and "inject" in ri else None, ri.get("verdict", "") if ri else ""))
print(f"{'объект':34s} | настоящий снимок (эталон: не вращается) | подстановка 60 °/с: найдено, ошибка оси, вердикт")
for n, v, rr, ae, vi in rows:
    print(f"{n:34s} | {v[:40]:40s} | {('%.1f' % rr) if rr is not None else '—':>6s} °/с, {('%.0f' % ae) if ae is not None else '—':>4s}°, {vi[:30]}")
