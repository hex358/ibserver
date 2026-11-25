"""Детальные модели для реалистичных кадров (THEORY §23):
  * МКС — модель NASA 3D Resources «ISS (stationary)», 247 тыс. треугольников, 26 текстур (не охраняется авторским
    правом США; credit NASA). Источник: raw.githubusercontent.com/0crispy/iss-tracker/main/project/models/iss/
    ISS_stationary.glb → data/models/hd/ISS_stationary.glb.
  * Starlink V2 mini — генератор satvis-3d-models (MIT, © 2026 F. Mauracher): размеры шины 4.1 × 2.7 × 0.3 м и крыльев
    12.8 × 4.1 м из заявки SpaceX в FCC (4.10.2022), 2 × 16 панелей на крыло, штанги, лазерные терминалы, антенны,
    двигатель → data/models/hd/starlink_v2mini_satvis.glb (node --experimental-strip-types gen.ts). Поверх — процедурная
    текстура ячеек (8 × 4 см, токосъёмные шины), тыльная сторона панелей — тёмная подложка.
Оси glTF обеих моделей: +Z — вдоль скорости, +Y — зенит, +X — левый борт → тело (вдоль, поперёк, радиус) = (Z, X, Y).
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HD = ROOT / "data/models/hd"
ISS_GLB = HD / "ISS_stationary.glb"
SL_GLB = HD / "starlink_v2mini_satvis.glb"
OUTBOARD = ("23 S4", "24 S5", "32 S6", "20 P4", "21 P5", "08 P6")     # за SARJ: вращаются вместе с крыльями


def _flatten(path):
    import trimesh
    sc = trimesh.load(path, force="scene")
    parts = []
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]; m = sc.geometry[g]
        if len(m.faces) == 0:
            continue
        uv = getattr(m.visual, "uv", None)
        parts.append(dict(node=node, V=trimesh.transform_points(m.vertices, T), Vloc=np.asarray(m.vertices, float),
                          F=np.asarray(m.faces), UV=None if uv is None or len(uv) != len(m.vertices) else np.asarray(uv, float),
                          mat=getattr(m.visual, "material", None)))
    return parts


def _mat_props(mat):
    """(имя, цвет rgb 0..1, текстура PIL|None, металл, шероховатость) из PBR-материала trimesh."""
    if mat is None:
        return ("default", (0.5, 0.5, 0.5), None, 0.2, 0.5)
    f = getattr(mat, "baseColorFactor", None)
    col = (1.0, 1.0, 1.0) if f is None else tuple((np.asarray(f[:3], float) / (255.0 if np.max(f) > 1.0 else 1.0)).tolist())
    tex = getattr(mat, "baseColorTexture", None)
    met = getattr(mat, "metallicFactor", None); rough = getattr(mat, "roughnessFactor", None)
    return (getattr(mat, "name", None) or "m", col, tex, 1.0 if met is None else float(met), 1.0 if rough is None else float(rough))   # умолчания glTF


def _rot2(Y, Z, a, y0=0.0, z0=0.0):
    c, s = math.cos(a), math.sin(a)
    y, z = Y - y0, Z - z0
    return y0 + c * y - s * z, z0 + s * y + c * z


def iss_parts(sun_b):
    """МКС NASA в штатной ориентации +XVV, внешние секции (S4–S6, P4–P6) повёрнуты SARJ так, что нормаль крыльев
    смотрит на проекцию Солнца на плоскость (радиус, вдоль). Возвращает части в осях тела и угол SARJ по сторонам."""
    parts = _flatten(ISS_GLB)
    s2 = np.array([sun_b[2], sun_b[0]])                      # Солнце в плоскости (Y, Z) glTF = (радиус, вдоль)
    angs = {}
    for side, pref in (("S", ("23 S4", "24 S5", "32 S6")), ("P", ("20 P4", "21 P5", "08 P6"))):
        out = [p for p in parts if p["node"].startswith(pref)]
        hub = [p for p in out if "_01" not in p["node"] and "_02" not in p["node"]]
        Vh = np.vstack([p["V"] for p in hub]); y0, z0 = Vh[:, 1].mean(), Vh[:, 2].mean()
        # нормаль крыльев: средняя по площади нормаль граней одеяла панелей
        ns = []
        for p in out:
            if "_01" in p["node"] or "_02" in p["node"]:
                V, F = p["V"], p["F"]
                ns.append(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 1:3])
        n2 = np.vstack(ns); ref = n2[np.argmax(np.linalg.norm(n2, axis=1))]
        nsum = (n2 * np.sign(n2 @ ref + 1e-12)[:, None]).sum(0)
        th_n = math.atan2(nsum[1], nsum[0]); th_s = math.atan2(s2[1], s2[0])
        a = (th_s - th_n + math.pi / 2) % math.pi - math.pi / 2   # нормаль двусторонняя: ближайший из ±n
        angs[side] = math.degrees(a)
        for p in out:
            p["V"] = p["V"].copy(); p["V"][:, 1], p["V"][:, 2] = _rot2(p["V"][:, 1], p["V"][:, 2], a, y0, z0)
    for p in parts:
        p["V"] = p["V"][:, [2, 0, 1]]
    return parts, angs


def cell_texture(path, tile=(0.76, 2.01), cell=(0.08, 0.04), px=0.002, seed=0):
    """Ячейки 8 × 4 см (тройные GaAs-ячейки обычного космического размера; конкретный размер у Starlink не опубликован)
    с серебристыми межэлементными шинами, рамка панели."""
    W, H = int(tile[0] / px), int(tile[1] / px)
    rng = np.random.default_rng(seed)
    nx, ny = int(tile[0] / cell[0]), int(tile[1] / cell[1])
    img = np.zeros((H, W, 3), np.float32)
    base = np.array([0.020, 0.030, 0.075])
    tint = base[None, None, :] * (1 + 0.18 * rng.standard_normal((ny, nx, 1)))
    yy = (np.arange(H) * ny // H).clip(0, ny - 1); xx = (np.arange(W) * nx // W).clip(0, nx - 1)
    img[:] = tint[yy][:, xx]
    cx = (np.arange(W) % (W / nx)) < 1.5; cy = (np.arange(H) % (H / ny)) < 1.2
    img[:, cx] = [0.55, 0.56, 0.58]; img[cy, :] = [0.35, 0.36, 0.38]
    b = int(0.012 / px); img[:b] = img[-b:] = img[:, :b] = img[:, -b:] = [0.45, 0.46, 0.48]
    Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(path)
    return path


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


def array_texture(path, size=0.3, px=0.0025):
    """Фазированная решётка снизу: серые плитки с излучателями через 3 см."""
    N = int(size / px); g = np.full((N, N), 0.30)
    st = int(0.03 / px)
    for y0 in range(0, N, st):
        for x0 in range(0, N, st):
            g[y0 + 3:y0 + st - 3, x0 + 3:x0 + st - 3] = 0.22
    return _save_srgb(np.stack([g, g * 1.01, g * 1.04], -1), path)


def starlink_parts(psi, phi, seed=7):
    """Starlink V2 mini (satvis) в позе реконструкции: ось крыльев под углом ψ к направлению полёта, крылья повёрнуты
    на φ вокруг своей оси — те же определения, что в model_recon.body_vertices. Панели слегка неровные (±0.6°), у каждой
    свой вариант текстуры; обшивка и палуба — с текстурами по граням."""
    parts = _flatten(SL_GLB)
    rng = np.random.default_rng(seed)
    out = []
    for p in parts:
        name = _mat_props(p["mat"])[0]
        if name == "cells":                                  # UV по локальным осям панели; низ — подложка, торцы — рамка
            L = p["Vloc"]; ext = L.max(0) - L.min(0); c0 = L.min(0)
            uv = np.c_[(L[:, 0] - c0[0]) / ext[0], (L[:, 2] - c0[2]) / ext[2]]
            n = np.cross(L[p["F"][:, 1]] - L[p["F"][:, 0]], L[p["F"][:, 2]] - L[p["F"][:, 0]])
            top = n[:, 1] > 0.5 * np.linalg.norm(n, axis=1); bot = n[:, 1] < -0.5 * np.linalg.norm(n, axis=1)
            # микронаклон панели вокруг её центра (панели на петлях не идеально в одной плоскости)
            V = p["V"].copy(); cen = V.mean(0)
            ax_, az_ = np.radians(rng.normal(0, 0.6, 2))
            Rx = np.array([[1, 0, 0], [0, math.cos(ax_), -math.sin(ax_)], [0, math.sin(ax_), math.cos(ax_)]])
            Rz = np.array([[math.cos(az_), -math.sin(az_), 0], [math.sin(az_), math.cos(az_), 0], [0, 0, 1]])
            V = (V - cen) @ (Rz @ Rx).T + cen
            k = int(rng.integers(0, 6))
            for sel, mname in ((top, f"cells_tex{k}"), (bot, "backsheet"), (~(top | bot), "frame")):
                if sel.any():
                    out.append(dict(node=p["node"], V=V, F=p["F"][sel], UV=uv if mname.startswith("cells_tex") else None, mat=None, mname=mname))
        elif name in ("aluminium", "deck", "phasedArray"):
            scale = {"aluminium": 1.0, "deck": 1.0, "phasedArray": 0.3}[name]
            out.append(dict(p, UV=_box_uv(p["Vloc"], p["F"], scale), mat=None, mname=name + "_tex"))
        else:
            out.append(dict(p, mname=name))
    c, s = math.cos(phi), math.sin(phi)
    u = np.array([math.cos(psi), math.sin(psi), 0.0]); er = np.array([0.0, 0.0, 1.0]); w = np.cross(u, er)
    for p in out:
        X, Y, Z = p["V"][:, 0], p["V"][:, 1], p["V"][:, 2]
        if p["node"].startswith(("starboard-wing", "port-wing")):
            Y, Z = c * Y - s * Z, s * Y + c * Z
        p["V"] = X[:, None] * u + Y[:, None] * er + Z[:, None] * w
    return out


def _proc_bsdf(key, work):
    """BSDF процедурных материалов Starlink; None — не процедурный."""
    two = lambda b: {"type": "twosided", "bsdf": b}
    if key.startswith("cells_tex"):
        tex = cell_texture2(work / f"{key}.png", seed=100 + int(key[-1]))
        # ячейки под покровным стеклом: тёмная подложка + гладкое стекло n = 1.5 (отражает Землю и Солнце)
        return two({"type": "roughplastic", "distribution": "ggx", "alpha": 0.035, "int_ior": 1.5,
                    "diffuse_reflectance": {"type": "bitmap", "filename": str(tex)}})
    if key == "malacodart":
        return _mal_bsdf(work)
    if key == "malacodart_array":
        return _mal_array_bsdf(work)
    if key == "aluminium_tex":
        tex = metal_texture(work / "alu.png")
        return two({"type": "principled", "base_color": {"type": "bitmap", "filename": str(tex)}, "metallic": 0.85, "roughness": 0.32})
    if key == "deck_tex":
        tex = deck_texture(work / "deck.png")
        return two({"type": "principled", "base_color": {"type": "bitmap", "filename": str(tex)}, "metallic": 0.1, "roughness": 0.42})
    if key == "phasedArray_tex":
        tex = array_texture(work / "pa.png")
        return two({"type": "principled", "base_color": {"type": "bitmap", "filename": str(tex)}, "metallic": 0.4, "roughness": 0.35})
    return None


EXTRA = {  # материалы, которых нет в glTF: (цвет, металл, шероховатость, текстура)
    "backsheet": ((0.06, 0.06, 0.07), 0.1, 0.6, None),
    "frame": ((0.40, 0.41, 0.43), 0.8, 0.35, None),
    "busplate": ((0.022, 0.022, 0.025), 0.2, 0.55, None),
}


def mitsuba_shapes(parts, work, smooth=True):
    """OBJ по материалам (v, vt) + principled BSDF; текстуры glTF — PNG рядом."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in parts:
        if "mname" in p and p.get("mat") is None:
            key = p["mname"]
        else:
            key = _mat_props(p["mat"])[0]
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
        pb = _proc_bsdf(key, work) if has_uv else None
        if pb is not None:
            shapes[f"hd{i}"] = {"type": "obj", "filename": str(fn), "face_normals": not smooth, "bsdf": pb}
            continue
        if key in EXTRA:
            col, met, rough, tex = EXTRA[key]
        elif key == "cells_tex":
            col, met, rough, tex = (1, 1, 1), 0.0, 0.18, cell_texture(work / "cells.png")
        else:
            _, col, texi, met, rough = _mat_props(ps[0]["mat"])
            tex = None
            if texi is not None and has_uv:
                tex = work / f"t{i}.png"
                im = np.asarray(texi.convert("RGB"), np.float32) / 255.0 * np.asarray(col, np.float32)
                Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(tex)
        bc = {"type": "bitmap", "filename": str(tex)} if tex is not None else {"type": "rgb", "value": list(map(float, col))}
        shapes[f"hd{i}"] = {"type": "obj", "filename": str(fn), "face_normals": not smooth,
                            "bsdf": {"type": "twosided", "bsdf": {"type": "principled", "base_color": bc,
                                                                   "metallic": float(met), "roughness": float(max(rough, 0.05))}}}
    return shapes


def all_vertices(parts):
    return np.vstack([p["V"][np.unique(p["F"])] for p in parts])


def starlink_pose(name, psi_fit, phi_fit):
    """Единое правило позы крыльев: по умолчанию крылья на Солнце (так Starlink получает питание); подобранный по снимку
    угол — только если снимок заметно его предпочитает: Δχ²(Солнце − подгонка) > 25 (results/recon_out/wing_check.json).
    Возвращает (ψ°, φ°, подпись)."""
    import json
    f = ROOT / "results/recon_out/wing_check.json"
    w = json.loads(f.read_text()).get(name) if f.exists() else None
    if w and w["use"] == "sun":
        return w["psi_sun"], w["phi_sun"], f"крылья на Солнце (снимок их не отвергает: Δχ² {w['dchi2_sun_minus_fit']:+.0f})"
    d = f", снимок предпочитает его на Δχ² {w['dchi2_sun_minus_fit']:.0f}" if w else ""
    return psi_fit, phi_fit, f"поза из реконструкции (ψ {psi_fit:+.0f}°, φ {phi_fit:+.0f}°{d})"


MAL_DIR = HD / "starlink_malacodart"     # «Starlink Spacex Satellite», Malacodart, CC-BY-4.0 (Sketchfab), Starlink v1


def _mal_meshes():
    import trimesh
    sc = trimesh.load(MAL_DIR / "scene.gltf", force="scene")
    out = {}
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]; m = sc.geometry[g]
        out[node] = (trimesh.transform_points(m.vertices, T), np.asarray(m.faces), np.asarray(m.visual.uv, float))
    return out


def starlink_parts_mal(psi, phi):
    """Starlink V2 mini, собранный из деталей детальной модели Starlink v1 (Malacodart, CC-BY-4.0, текстуры 4096²: цвет,
    металл/шероховатость, нормали) в размерах V2 mini по заявке SpaceX в FCC: шина 4.1 × 2.7 × 0.3 м — две шины v1
    рядом (масштаб 1.46 по длине, 1.10 поперёк), крылья 12.8 × 4.1 м — по две панели v1 встык (×1.48 по ширине);
    коромысла — из модели satvis. Оси: X — ось крыльев (поперёк), Y — зенит, Z — вдоль скорости; поза ψ, φ — как в
    starlink_parts."""
    M = _mal_meshes()
    keys = sorted(M, key=lambda k: -np.ptp(M[k][0][:, 1]))           # самая высокая по Y — крыло
    arr_k, bus_ks = keys[0], keys[1:]
    out = []
    # шина: Malacodart X (длина 2.8) → Z, Z (ширина 1.22) → X, Y → Y
    Vb = np.vstack([M[k][0] for k in bus_ks]); c = (Vb.max(0) + Vb.min(0)) / 2
    half_w = np.ptp(Vb[:, 2]) * 1.10
    for side in (1, -1):
        for k in bus_ks:
            V, F, UV = M[k]
            x = side * (half_w / 2 + 1.10 * (V[:, 2] - c[2]))           # вторая шина — зеркальная
            W = np.c_[x, V[:, 1] - c[1], 1.46 * (V[:, 0] - c[0])]
            out.append(dict(node=f"bus-{side}-{k}", V=W, F=F if side > 0 else F[:, ::-1], UV=UV, mat=None, mname="malacodart"))
    # крыло: Malacodart X (ширина 2.77) → Z (×1.48), Y (длина) → ±X, Z (нормаль) → Y
    V, F, UV = M[arr_k]
    y0, xc = V[:, 1].min(), (V[:, 0].max() + V[:, 0].min()) / 2
    zc = float(np.median(V[:, 2])); L = np.ptp(V[:, 1])
    for side, nm in ((1, "port-wing"), (-1, "starboard-wing")):
        for cpy in (0, 1):
            x = side * (4.15 + cpy * L + (V[:, 1] - y0))
            W = np.c_[x, V[:, 2] - zc, 1.48 * (V[:, 0] - xc)]
            out.append(dict(node=f"{nm}-mal-{cpy}", V=W, F=F if side > 0 else F[:, ::-1], UV=UV, mat=None, mname="malacodart_array"))
    # сплошной корпус 4.1 × 2.7 м под двумя половинами (у V2 mini шина прямоугольная) — из satvis, чуть тоньше
    for p in _flatten(SL_GLB):
        if p["node"] in ("bus", "spine"):
            V = p["V"].copy(); V[:, 1] = V[:, 1] * 0.133 - 0.08      # тонкая подложка на уровне нижней плиты v1 (−0.10…−0.06)
            out.append(dict(p, V=V, UV=None if p["node"] == "bus" else _box_uv(p["Vloc"], p["F"], 1.0), mat=None,
                            mname="busplate" if p["node"] == "bus" else "aluminium_tex"))
    # коромысла и усилитель из satvis (алюминий)
    for p in _flatten(SL_GLB):
        if p["node"].startswith(("starboard-wing-yoke", "port-wing-yoke", "starboard-wing-stiffener", "port-wing-stiffener")):
            out.append(dict(p, UV=_box_uv(p["Vloc"], p["F"], 1.0), mat=None, mname="aluminium_tex"))
    cph, sph = math.cos(phi), math.sin(phi)
    u = np.array([math.cos(psi), math.sin(psi), 0.0]); er = np.array([0.0, 0.0, 1.0]); w = np.cross(u, er)
    for p in out:
        X, Y, Z = p["V"][:, 0], p["V"][:, 1], p["V"][:, 2]
        if p["node"].startswith(("starboard-wing", "port-wing")):
            Y, Z = cph * Y - sph * Z, sph * Y + cph * Z
        p["V"] = X[:, None] * u + Y[:, None] * er + Z[:, None] * w
    return out


def _mal_bsdf(work):
    mr = np.asarray(Image.open(MAL_DIR / "textures/Starlink_metallicRoughness.png").convert("RGB"))
    Image.fromarray(mr[..., 1]).save(work / "mal_rough.png"); Image.fromarray(mr[..., 2]).save(work / "mal_metal.png")
    b = {"type": "principled", "base_color": {"type": "bitmap", "filename": str(MAL_DIR / "textures/Starlink_baseColor.png")},
         "roughness": {"type": "bitmap", "filename": str(work / "mal_rough.png"), "raw": True},
         "metallic": {"type": "bitmap", "filename": str(work / "mal_metal.png"), "raw": True}}
    return {"type": "twosided", "bsdf": {"type": "normalmap", "normalmap": {"type": "bitmap", "raw": True,
            "filename": str(MAL_DIR / "textures/Starlink_normal.png")}, "bsdf": b}}


def _mal_array_bsdf(work):
    """Крылья: ячейки — диэлектрик под покровным стеклом (а не металл, как в исходных текстурах, — отсюда блёстки):
    цвет из текстуры модели, приглушённый до альбедо ячеек ~0.03–0.08, гладкое стекло n = 1.5, карта нормалей модели."""
    bc = np.asarray(Image.open(MAL_DIR / "textures/Starlink_baseColor.png").convert("RGB"), np.float32) / 255.0
    lin = bc ** 2.2 * 0.30
    Image.fromarray((np.clip(lin, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)).save(work / "mal_array_base.png")
    b = {"type": "roughplastic", "distribution": "ggx", "alpha": 0.06, "int_ior": 1.5,
         "diffuse_reflectance": {"type": "bitmap", "filename": str(work / "mal_array_base.png")}}
    return {"type": "twosided", "bsdf": {"type": "normalmap", "normalmap": {"type": "bitmap", "raw": True,
            "filename": str(MAL_DIR / "textures/Starlink_normal.png")}, "bsdf": b}}


_starlink_parts_satvis = starlink_parts


def starlink_parts(psi, phi, seed=7):
    """По умолчанию — V2 mini из деталей детальной модели Malacodart (если она скачана), иначе — схема satvis."""
    if (MAL_DIR / "scene.gltf").exists():
        return starlink_parts_mal(psi, phi)
    return _starlink_parts_satvis(psi, phi, seed)
