"""Детальные модели для рендера (render_file в каталоге): собираются из открытых детальных моделей в осях и размерах
модели каталога, чтобы поза и шарниры подгонки переносились без изменений.

  python scripts/build_hd_assets.py
Сейчас: STARLINK-V2-MINI ← детали «Starlink Spacex Satellite» (Malacodart, Sketchfab, CC-BY-4.0; Starlink v1, текстуры
4096²: цвет, металл/шероховатость, нормали) в размерах V2 mini по заявке SpaceX в FCC: шина 4,1 × 2,7 м — две шины v1
рядом (×1,46 по длине, ×1,10 поперёк) на тонкой подложке из модели satvis; крылья 12,8 × 4,1 м — по две панели v1
встык (×1,48 по ширине); коромысла — из satvis. Оси satvis: X — ось крыльев, Y — зенит, Z — вдоль скорости.
Ячейки крыльев: цвет текстуры приглушён до альбедо ячеек (в исходной модели они металлические — отсюда блёстки).
Пишет data/models/catalog/hd/starlink-v2-mini-hd.glb.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import modelcat as MC

MAL = ROOT / "data/models/hd/starlink_malacodart"
OUT = ROOT / "data/models/catalog/hd"


def mal_meshes():
    sc = trimesh.load(MAL / "scene.gltf", force="scene")
    out = {}
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]; m = sc.geometry[g]
        out[node] = (trimesh.transform_points(m.vertices, T), np.asarray(m.faces), np.asarray(m.visual.uv, float))
    return out


def box_uv(V, F, scale=1.0):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); ax = np.argmax(np.abs(n), 1)
    UV = np.zeros((len(V), 2))
    for f, a_ in zip(F, ax):
        i, j = [(1, 2), (0, 2), (0, 1)][a_]
        UV[f] = np.c_[V[f, i], V[f, j]] / scale
    return UV


def pbr(name, base, mr=None, normal=None, metallic=0.0, rough=0.6):
    return trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=base, metallicRoughnessTexture=mr, normalTexture=normal,
                                               metallicFactor=metallic, roughnessFactor=rough, doubleSided=True)


def starlink_v2mini():
    M = mal_meshes()
    keys = sorted(M, key=lambda k: -np.ptp(M[k][0][:, 1]))                 # самая высокая по Y — крыло
    arr_k, bus_ks = keys[0], keys[1:]
    base = Image.open(MAL / "textures/Starlink_baseColor.png").convert("RGB")
    mrt = Image.open(MAL / "textures/Starlink_metallicRoughness.png").convert("RGB")
    nrm = Image.open(MAL / "textures/Starlink_normal.png").convert("RGB")
    lin = (np.asarray(base, np.float32) / 255.0) ** 2.2 * 0.30
    cells = Image.fromarray((np.clip(lin, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8))
    m_bus = pbr("malacodart-bus", base, mrt, nrm, metallic=1.0, rough=1.0)
    m_arr = pbr("malacodart-cells", cells, None, nrm, metallic=0.0, rough=0.1)
    sc = trimesh.Scene()

    def add(name, V, F, UV, mat):
        sc.add_geometry(trimesh.Trimesh(V, F, visual=trimesh.visual.TextureVisuals(uv=UV, material=mat), process=False),
                        node_name=name, geom_name=name)
    # шина: Malacodart X (длина 2,8) → Z, Z (ширина 1,22) → X, Y → Y; вторая — зеркально
    Vb = np.vstack([M[k][0] for k in bus_ks]); c = (Vb.max(0) + Vb.min(0)) / 2; half_w = np.ptp(Vb[:, 2]) * 1.10
    for side in (1, -1):
        for k in bus_ks:
            V, F, UV = M[k]
            W = np.c_[side * (half_w / 2 + 1.10 * (V[:, 2] - c[2])), V[:, 1] - c[1], 1.46 * (V[:, 0] - c[0])]
            add(f"bus-hd-{'p' if side > 0 else 's'}-{k}", W, F if side > 0 else F[:, ::-1], UV, m_bus)
    # крылья: Malacodart X (ширина 2,77) → Z (×1,48), Y (длина) → ±X, Z (нормаль) → Y; по две панели встык
    V, F, UV = M[arr_k]
    y0, xc = V[:, 1].min(), (V[:, 0].max() + V[:, 0].min()) / 2; zc = float(np.median(V[:, 2])); L = np.ptp(V[:, 1])
    for side, nm in ((1, "port-wing"), (-1, "starboard-wing")):
        for cpy in (0, 1):
            W = np.c_[side * (4.15 + cpy * L + (V[:, 1] - y0)), V[:, 2] - zc, 1.48 * (V[:, 0] - xc)]
            add(f"{nm}-hd-{cpy}", W, F if side > 0 else F[:, ::-1], UV, m_arr)
    # подложка шины и коромысла — из модели каталога satvis (алюминий)
    alu = Image.fromarray(np.full((8, 8, 3), 150, np.uint8))
    m_alu = pbr("aluminium", alu, None, None, metallic=0.6, rough=0.45)
    for p in MC._scene_parts(ROOT / "data/models/catalog/satvis/starlink-v2-mini.glb"):
        nd = p["node"]
        if nd in ("bus", "spine"):
            V = np.array(p["V"], float); V[:, 1] = V[:, 1] * 0.133 - 0.08
            add(f"{nd}-plate", V, p["F"], box_uv(V, p["F"]), m_alu)
        elif nd.startswith(("starboard-wing-yoke", "port-wing-yoke", "starboard-wing-stiffener", "port-wing-stiffener")):
            V = np.array(p["V"], float); add(nd, V, p["F"], box_uv(V, p["F"]), m_alu)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "starlink-v2-mini-hd.glb"; sc.export(out)
    ext = np.ptp(np.vstack([g.vertices for g in sc.dump()]), 0)
    print(out, "размеры (X — ось крыльев, Y — зенит, Z — вдоль), м:", np.round(ext, 2))
    return out


if __name__ == "__main__":
    starlink_v2mini()
