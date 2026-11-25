"""Детальные модели для реалистичных кадров (THEORY §23–24): одна функция для всех объектов.

Модель берётся из каталога (s2sat/modelcat: NASA 3D Resources, satvis-3d-models, KeepTrack, Sketchfab) в полной
детализации и ставится в ту же позу, что подобрана по снимку (autopose.pose_body: закон ориентации, приводы панелей,
найденные в модели автоматически). Материалы — из glTF (PBR, текстуры); у параметрических моделей — панели/корпус.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _mat_props(mat):
    """(имя, цвет rgb 0..1, текстура PIL|None, металл, шероховатость) из PBR-материала trimesh."""
    if mat is None:
        return ("default", (0.5, 0.5, 0.5), None, 0.2, 0.5)
    f = getattr(mat, "baseColorFactor", None)
    col = (1.0, 1.0, 1.0) if f is None else tuple((np.asarray(f[:3], float) / (255.0 if np.max(f) > 1.0 else 1.0)).tolist())
    tex = getattr(mat, "baseColorTexture", None)
    met = getattr(mat, "metallicFactor", None); rough = getattr(mat, "roughnessFactor", None)
    return (getattr(mat, "name", None) or "m", col, tex, 1.0 if met is None else float(met), 1.0 if rough is None else float(rough))   # умолчания glTF


EXTRA = {  # материалы, которых нет в glTF: (цвет, металл, шероховатость, текстура)
    "backsheet": ((0.06, 0.06, 0.07), 0.1, 0.6, None),
    "frame": ((0.40, 0.41, 0.43), 0.8, 0.35, None),
}


def mitsuba_shapes(parts, work, smooth=True):
    """OBJ по материалам (v, vt) + principled BSDF; текстуры glTF — PNG рядом."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from s2sat.modelcat import _is_solar
    groups = {}
    for p in parts:
        if "mname" in p and p.get("mat") is None:
            key = p["mname"]
        else:
            key = _mat_props(p["mat"])[0]
        if p.get("node") and _is_solar(dict(node=p["node"], mname=p.get("mname") or "")):
            key = key + "|cells"                           # панели — свой материал (стекло поверх элементов)
        groups.setdefault(key, []).append(p)
    shapes = {}
    for i, (key, ps) in enumerate(sorted(groups.items())):
        has_uv = all(p["UV"] is not None for p in ps)
        lines_v, lines_t, lines_f = [], [], []; off = 0
        for p in ps:
            u_idx, inv = np.unique(p["F"], return_inverse=True)
            V = p["V"][u_idx]
            lines_v.append("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V))
            if has_uv:
                lines_t.append("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in p["UV"][u_idx]))
            tri = inv.reshape(-1, 3) + off + 1
            lines_f.append("".join((f"f {a}/{a} {b}/{b} {c}/{c}\n" if has_uv else f"f {a} {b} {c}\n") for a, b, c in tri))
            off += len(V)
        fn = work / f"p{i}.obj"
        fn.write_text("".join(lines_v) + "".join(lines_t) + "".join(lines_f))
        base = key.split("|")[0]
        if base in EXTRA:
            col, met, rough, tex = EXTRA[base]
        else:
            _, col, texi, met, rough = _mat_props(ps[0]["mat"])
            tex = None
            if texi is not None and has_uv:
                tex = work / f"t{i}.png"
                im = np.asarray(texi.convert("RGB"), np.float32) / 255.0 * np.asarray(col, np.float32)
                Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(tex)
        bc = {"type": "bitmap", "filename": str(tex)} if tex is not None else {"type": "rgb", "value": list(map(float, col))}
        if key.endswith("|cells"):
            # солнечные элементы: диффузная подложка (цвет/текстура модели) под покровным стеклом n = 1.5 с почти
            # зеркальным отражением — на снимках панели отражают Землю и дают блики
            bsdf = {"type": "roughplastic", "distribution": "ggx", "alpha": 0.04, "int_ior": 1.5, "diffuse_reflectance": bc}
        else:
            bsdf = {"type": "principled", "base_color": bc, "metallic": float(met), "roughness": float(max(rough, 0.05))}
        shapes[f"hd{i}"] = {"type": "obj", "filename": str(fn), "face_normals": not smooth, "bsdf": {"type": "twosided", "bsdf": bsdf}}
    return shapes


def all_vertices(parts):
    return np.vstack([p["V"][np.unique(p["F"])] for p in parts])


def parts_for(name, sun_b, norad=None):
    """Детальная модель объекта в позе из автоматической реконструкции (results/auto/<имя>.json, THEORY §24) — одинаково
    для всех объектов: модель каталога в полной детализации, та же функция позы (autopose.pose_body), что при подгонке,
    те же шарниры панелей. Возвращает (части, подпись, гладкие ли нормали)."""
    import json
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from s2sat import autopose as AP, modelcat as MC
    f = ROOT / "results/auto" / f"{name}.json"
    au = json.loads(f.read_text()) if f.exists() else None
    sun_b = np.asarray(sun_b, float)
    spec = MC.spec_for(norad or 0, name)
    if au:
        law, x = au["law"], np.asarray(au["x"], float)
        note = f"поза автоматически: {au['chosen']}" + (" — НЕШТАТНАЯ" if au.get("anomalous") else "")
    else:
        law, x = spec["laws"][0], np.zeros(7)
        note = f"поза по первому закону класса ({law['name']}; автоматической реконструкции нет)"
    e = spec.get("catalog")
    if e is None:                                              # параметрическая модель GCAT: 0 — панели, 1 — корпус
        Vb, _, _ = AP.pose_body(spec, law, x, sun_b)
        F, G = spec["F"], spec["G"]
        parts = [dict(node=f"g{g}", V=Vb, F=F[G == g], UV=None, mat=None, mname="backsheet" if g == 0 else "frame") for g in np.unique(G)]
        return parts, note + f"; модель: {spec['source']}", False
    M = MC.load_catalog_model(e, spec.get("gcat"), fit=False)
    full = dict(V=M["V"], F=M["F"], G=M["G"], M2B=M["M2B"], arrays=M["arrays"])
    Vb, _, _ = AP.pose_body(full, law, x, sun_b)
    parts = []
    for k, p in enumerate(M["parts"]):
        u, off = M["gidx"][k]
        V = np.array(p["V"], float); V[u] = Vb[off:off + len(u)]
        parts.append(dict(p, V=V))
    smooth = not e["source"].startswith("satvis")                # процедурные модели — плоские грани
    return parts, note + f"; модель: {e['key']} ({M['tri']} тр.)", smooth
