"""Автоматическая реконструкция: модель из каталога + поза по законам ориентации, без ручных решений (THEORY §24).

  python scripts/auto_recon.py ISS_20261118T222808 STARLINK-30580_20261120T111642 ...
Результат: results/auto/<имя>.json — модель, все гипотезы с невязками на отложенных полосах, выбранная поза,
флаг нештатной ориентации.
"""
import json, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG


def run(name, maxfev=160):
    t0 = time.time()
    rc = pickle.loads((ROOT / "results/shape" / f"{name}.pkl").read_bytes()); MR.prepare(rc)
    spec = MC.spec_for(rc.get("cat", 0), name)
    print(f"{name}: модель — {spec['source']} ({spec['kind']}), {len(spec['F'])} граней, законов {len(spec['laws'])}", flush=True)
    C = SG.lvlh_to_fit(name, rc)                        # инерциальные орбитальные оси → оси реконструкции
    out = AP.solve(rc, spec, maxfev=maxfev, C=C)
    out.update(name=name, seconds=round(time.time() - t0))
    Path(ROOT / "results/auto").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/auto" / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"  → {out['chosen']}{'  [НЕШТАТНАЯ ОРИЕНТАЦИЯ]' if out['anomalous'] else ''}; запас над второй гипотезой {out['margin']}; {out['seconds']} с", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "results/shape").glob("*.pkl"))   # без аргументов — все
    for n in names:
        if (ROOT / "results/auto" / f"{n}.json").exists():
            continue
        try:
            run(n)
        except Exception as ex:                                   # noqa: BLE001 — пакет не должен останавливаться
            print(f"{n}: ОШИБКА {type(ex).__name__}: {ex}", flush=True)
