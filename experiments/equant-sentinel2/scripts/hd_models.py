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
    mrt = getattr(mat, "metallicRoughnessTexture", None)
    if met is None and rough is None and mrt is None:
        # материал без PBR-описания (старые экспорты): по стандарту был бы шероховатый металл, на деле — краска/плёнка
        return (getattr(mat, "name", None) or "m", col, tex, 0.0, 0.6)
    return (getattr(mat, "name", None) or "m", col, tex, 1.0 if met is None else float(met), 1.0 if rough is None else float(rough))   # умолчания glTF


def _phys_maps(rgb):
    """Физический класс поверхности по цвету (линейный RGB 0..1, массив …×3) → (металличность, шероховатость).
    Карты металличности моделей NASA сделаны под студийный свет (белые модули — «металл», отражающий белые стены);
    в космосе отражать нечего, поэтому свойства берутся по классу покрытия космических аппаратов:
      белая краска / бета-ткань / белые радиаторы (светлое, нейтральное) — матовый диэлектрик, ρ ≈ 0.8, шерох. 0.85;
      золотая / алюминированная плёнка (жёлто-оранжевый оттенок) — металл, шерох. 0.3 (мятая фольга);
      алюминий конструкции (средне-серое, нейтральное) — металл 0.6, шерох. 0.45;
      чёрная краска, тёмная плёнка — диэлектрик, шерох. 0.7."""
    c = np.clip(rgb, 0, 1); mx = c.max(-1); mn = c.min(-1)
    lum = c @ np.array([0.2126, 0.7152, 0.0722])
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    gold = np.clip((sat - 0.25) / 0.2, 0, 1) * ((r >= g) & (g > b)) * np.clip((lum - 0.06) / 0.1, 0, 1)
    neutral = 1 - np.clip((sat - 0.12) / 0.15, 0, 1)
    white = neutral * np.clip((lum - 0.30) / 0.15, 0, 1)
    alu = neutral * (1 - white) * np.clip((lum - 0.06) / 0.08, 0, 1)
    met = gold * 1.0 + alu * 0.6
    rough = gold * 0.30 + white * 0.85 + alu * 0.45 + np.clip(1 - gold - white - alu, 0, 1) * 0.70
    return np.clip(met, 0, 1), np.clip(rough, 0.05, 1)


def _mr_maps(mat, met, rough, work, i, tex_rgb=None, col=(1, 1, 1)):
    """Металличность и шероховатость для Mitsuba: у текстурированных материалов — карты по физическому классу каждого
    пикселя текстуры цвета; у нетекстурированных — явные коэффициенты материала, если заданы, иначе — по цвету."""
    if tex_rgb is not None:
        lin = (np.asarray(tex_rgb, np.float32) / 255.0) ** 2.2 * np.asarray(col, np.float32)
        m, r = _phys_maps(lin)
        fm, fr = work / f"m{i}.png", work / f"r{i}.png"
        Image.fromarray((m * 255).astype(np.uint8)).save(fm); Image.fromarray((r * 255).astype(np.uint8)).save(fr)
        return ({"type": "bitmap", "filename": str(fm), "raw": True}, {"type": "bitmap", "filename": str(fr), "raw": True})
    explicit = mat is not None and getattr(mat, "metallicFactor", None) is not None
    if explicit:
        return float(met), float(max(rough, 0.05))
    m, r = _phys_maps(np.asarray(col, np.float32)[None, :])
    return float(m[0]), float(r[0])


EXTRA = {  # материалы, которых нет в glTF: (цвет, металл, шероховатость, текстура)
    "backsheet": ((0.06, 0.06, 0.07), 0.1, 0.6, None),
    "frame": ((0.40, 0.41, 0.43), 0.8, 0.35, None),
}


# ---- процедурные текстуры для моделей каталога без текстур (satvis, параметрические): общее правило по классу
# покрытия — панели солнечных батарей (ячейки под стеклом), светлое — краска/фольга со швами, серое — алюминий
# со швами и крепежом, тёмное — палуба. Позу и геометрию не меняют.
def _save_srgb(img, path):
    Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)).save(path)
    return path


def _box_uv(L, F, scale):
    """UV по граням коробки, в метрах / scale: проекция на две оси, перпендикулярные нормали грани."""
    n = np.cross(L[F[:, 1]] - L[F[:, 0]], L[F[:, 2]] - L[F[:, 0]])
    ax = np.argmax(np.abs(n), 1)
    UV = np.zeros((len(L), 2))
    for f, a_ in zip(F, ax):
        i, j = [(1, 2), (0, 2), (0, 1)][a_]
        UV[f] = np.c_[L[f, i], L[f, j]] / scale
    return UV


def cell_texture2(path, seed, tile=(0.76, 2.01), cell=(0.07, 0.04), px=0.0025):
    """Панель солнечной батареи вблизи: почти чёрные ячейки (своя небольшая разница тона у каждой), тёмные щели
    подложки 1.5 мм, тонкие серебристые шины, едва заметные пальцы токосъёма, алюминиевая рамка 5 мм; у каждой панели
    свой лёгкий сдвиг оттенка."""
    W, H = int(tile[0] / px), int(tile[1] / px)
    rng = np.random.default_rng(seed)
    nx, ny = int(round(tile[0] / cell[0])), int(round(tile[1] / cell[1]))
    base = np.array([0.010, 0.014, 0.034]) * (1 + 0.07 * rng.standard_normal(3) * np.array([1, 1, 1.6]))
    per = base[None, None, :] * (1 + 0.10 * rng.standard_normal((ny, nx, 1)))
    yy = (np.arange(H) * ny // H).clip(0, ny - 1); xx = (np.arange(W) * nx // W).clip(0, nx - 1)
    img = per[yy][:, xx].copy()
    cxp, cyp = np.arange(W) % (W / nx), np.arange(H) % (H / ny)
    img[:, cxp < 0.0015 / px] = [0.022, 0.022, 0.025]; img[cyp < 0.0015 / px, :] = [0.022, 0.022, 0.025]
    bb_ = (np.abs(cxp - 0.1 * W / nx) < 0.6) | (np.abs(cxp - 0.9 * W / nx) < 0.6)
    img[:, bb_] = img[:, bb_] * 0.3 + 0.16
    fy = (np.arange(H) % max(2, int(0.003 / px))) == 0
    img[fy, :] = img[fy, :] * 0.85 + 0.008
    b = max(1, int(0.005 / px)); img[:b] = img[-b:] = img[:, :b] = img[:, -b:] = [0.30, 0.31, 0.33]
    return _save_srgb(img, path)


def metal_texture(path, seed=3, size=1.0, px=0.0025, base=0.42, seam=0.5):
    """Алюминиевая обшивка: пятнистость, следы шлифовки, швы панелей каждые seam м, крепёж вдоль швов."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed); N = int(size / px)
    mott = gaussian_filter(rng.standard_normal((N, N)), 40, mode="wrap"); mott /= np.abs(mott).max()
    brush = gaussian_filter(rng.standard_normal((N, N)), (0.5, 30), mode="wrap"); brush /= np.abs(brush).max()
    g = base * (1 + 0.05 * mott + 0.03 * brush)
    k = int(seam / px)
    for o in range(0, N, k):
        g[:, o:o + 2] *= 0.35; g[o:o + 2, :] *= 0.35
        for t in range(k // 2 // 10, N, max(1, int(0.05 / px))):
            for (yy_, xx_) in ((t, o + 6), (o + 6, t)):
                g[max(0, yy_ - 2):yy_ + 2, max(0, xx_ - 2):xx_ + 2] *= 0.6
    img = np.stack([g, g * 1.005, g * 1.02], -1)
    return _save_srgb(img, path)


def deck_texture(path, seed=5, size=1.0, px=0.0025):
    """Тёмные панели палубы: почти чёрное покрытие с пятнистостью и светлыми швами каждые 0.33 м."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed); N = int(size / px)
    mott = gaussian_filter(rng.standard_normal((N, N)), 25, mode="wrap"); mott /= np.abs(mott).max()
    g = 0.035 * (1 + 0.15 * mott)
    k = int(0.333 / px)
    for o in range(0, N, k):
        g[:, o:o + 2] = 0.10; g[o:o + 2, :] = 0.10
    return _save_srgb(np.stack([g, g, g * 1.08], -1), path)



def _proc_texture(key, col, work, i):
    """Путь к процедурной текстуре 1 × 1 м для группы без текстуры (или None)."""
    lum = float(np.dot(np.asarray(col, float)[:3], [0.2126, 0.7152, 0.0722]))
    f = Path(work) / f"proc{i}.png"
    if f.exists():
        return f
    if key.endswith("|cells"):
        return cell_texture2(f, seed=11 + i, tile=(1.0, 1.0))
    if lum < 0.12:
        return deck_texture(f, seed=5 + i)
    return metal_texture(f, seed=3 + i, base=min(0.9, max(0.15, lum)), seam=0.5 if lum < 0.6 else 1.0)


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
        proc = False
        if not has_uv and key.split("|")[0] not in EXTRA:
            texi0 = _mat_props(ps[0]["mat"])[2] if ps[0].get("mat") is not None else None
            if texi0 is None:                                 # модель без текстур — процедурная текстура по UV коробки
                for p in ps:
                    p["UV"] = _box_uv(p["V"], p["F"], 1.0)
                has_uv = True; proc = True
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
        if proc:
            tex = _proc_texture(key, col, work, i)
        bc = {"type": "bitmap", "filename": str(tex)} if tex is not None else {"type": "rgb", "value": list(map(float, col))}
        mtl = ps[0].get("mat")
        if base in EXTRA:
            metv, roughv = float(met), float(max(rough, 0.05))
        else:
            trgb = np.asarray(texi.convert("RGB")) if (texi is not None and has_uv) else None
            metv, roughv = _mr_maps(mtl, met, rough, work, i, trgb, col)
        if key.endswith("|cells"):
            # солнечные элементы: диффузная подложка (цвет/текстура модели) под покровным стеклом n = 1.5 с почти
            # зеркальным отражением — на снимках панели отражают Землю и дают блики
            bsdf = {"type": "roughplastic", "distribution": "ggx", "alpha": 0.04, "int_ior": 1.5, "diffuse_reflectance": bc}
        else:
            bsdf = {"type": "principled", "base_color": bc, "metallic": metv, "roughness": roughv}
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
