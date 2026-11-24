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


def starlink_parts(psi, phi):
    """Starlink V2 mini (satvis) в позе реконструкции: ось крыльев под углом ψ к направлению полёта, крылья повёрнуты
    на φ вокруг своей оси — те же определения, что в model_recon.body_vertices."""
    parts = _flatten(SL_GLB)
    out = []
    for p in parts:
        name = _mat_props(p["mat"])[0]
        if name == "cells":                                  # UV по локальным осям панели; низ — подложка, торцы — рамка
            L = p["Vloc"]; ext = L.max(0) - L.min(0); c0 = L.min(0)
            uv = np.c_[(L[:, 0] - c0[0]) / ext[0], (L[:, 2] - c0[2]) / ext[2]]
            n = np.cross(L[p["F"][:, 1]] - L[p["F"][:, 0]], L[p["F"][:, 2]] - L[p["F"][:, 0]])
            top = n[:, 1] > 0.5 * np.linalg.norm(n, axis=1); bot = n[:, 1] < -0.5 * np.linalg.norm(n, axis=1)
            for sel, mname in ((top, "cells_tex"), (bot, "backsheet"), (~(top | bot), "frame")):
                if sel.any():
                    out.append(dict(node=p["node"], V=p["V"], F=p["F"][sel], UV=uv if mname == "cells_tex" else None, mat=None, mname=mname))
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


EXTRA = {  # материалы, которых нет в glTF: (цвет, металл, шероховатость, текстура)
    "backsheet": ((0.06, 0.06, 0.07), 0.1, 0.6, None),
    "frame": ((0.55, 0.56, 0.58), 0.8, 0.35, None),
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
