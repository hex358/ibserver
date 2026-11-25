"""Каталог 3D-моделей для автоматической реконструкции (THEORY §24).

По номеру NORAD объекта выбирается модель — без ручного вмешательства:
  1) кураторская модель, если есть (МКС — сетка KeepTrack; Starlink V2 mini — KeepTrack; детальные модели для
     рендера — hd_models); у неё заданы оси, шарниры солнечных батарей и законы ориентации класса;
  2) иначе — параметрическая модель из GCAT (J. McDowell, CC BY 4.0): форма («Cyl», «Box + 2 Pan», «Cyl + 2 Pan»,
     «Step Cyl + 2 Pan», …), длина, диаметр, размах; тип объекта (P — аппарат, R — ступень, D — обломок) задаёт
     законы ориентации: аппарат — ориентирован на Землю, панели на Солнце; ступень и обломок — произвольная ось.
Сетка возвращается в своих координатах (м) + матрица M2B в оси тела (вдоль, поперёк, радиус) при штатной позе.
"""
from __future__ import annotations

import csv, gzip, io, math, re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "data/models"
CAT = ROOT / "data/catalogs"


@lru_cache(maxsize=1)
def _gcat():
    f = CAT / "gcat_satcat.tsv.gz"
    if not f.exists():
        return {}
    rows = csv.reader(io.TextIOWrapper(gzip.open(f), encoding="utf-8", errors="replace"), delimiter="\t")
    hdr = [h.strip("#").strip() for h in next(rows)]
    out = {}
    for r in rows:
        if len(r) < len(hdr) or r[0].startswith("#"):
            continue
        d = {h: v.strip() for h, v in zip(hdr, r)}
        if d.get("Satcat", "").isdigit():
            out[int(d["Satcat"])] = d
    return out


def gcat(norad):
    return _gcat().get(int(norad))


def _num(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ сетки
def load_obj(path):
    V, F, MAT, OBJ = [], [], [], []
    o = mat = None
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("o ") or ln.startswith("g "):
            o = ln.split()[1] if len(ln.split()) > 1 else o
        elif ln.startswith("usemtl"):
            mat = ln.split()[1]
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]]); MAT.append(mat); OBJ.append(o)
    kd = {}
    mtl = Path(path).with_suffix(".mtl")
    if mtl.exists():
        cur = None
        for ln in open(mtl):
            if ln.startswith("newmtl"):
                cur = ln.split()[1]
            elif ln.strip().startswith("Kd") and cur:
                kd[cur] = np.array([float(v) for v in ln.split()[1:4]])
    return np.array(V, float), np.array(F, int), np.array(MAT), np.array(OBJ), kd


def _groups(MAT, kd, panel_mats):
    """Группы материалов для НМНК: 0 — солнечные панели, 1 — светлое, 2 — тёмное."""
    G = np.full(len(MAT), 2)
    for i, m in enumerate(MAT):
        if m in panel_mats:
            G[i] = 0
        elif m in kd and kd[m].mean() > 0.45:
            G[i] = 1
    return G


def _box(cx, cy, cz, sx, sy, sz):
    x0, x1 = cx - sx / 2, cx + sx / 2; y0, y1 = cy - sy / 2, cy + sy / 2; z0, z1 = cz - sz / 2, cz + sz / 2
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    return V, F


def _cyl(L, D, n=20, steps=None):
    """Цилиндр вдоль X (центр в 0); steps — [(доля длины, доля диаметра)] для ступенчатого."""
    prof = steps or [(1.0, 1.0)]
    V, F = [], []
    x = -L / 2
    for frac, dfr in prof:
        x1 = x + frac * L; r = D * dfr / 2; base = len(V)
        for xx in (x, x1):
            for k in range(n):
                a = 2 * math.pi * k / n; V.append([xx, r * math.cos(a), r * math.sin(a)])
        for k in range(n):
            a, b, c, d = base + k, base + (k + 1) % n, base + n + (k + 1) % n, base + n + k
            F += [[a, b, c], [a, c, d]]
        for xx, off, flip in ((x, base, True), (x1, base + n, False)):     # торцы
            ci = len(V); V.append([xx, 0, 0])
            for k in range(n):
                tri = [ci, off + k, off + (k + 1) % n]
                F.append(tri[::-1] if flip else tri)
        x = x1
    return np.array(V, float), np.array(F, int)


def parametric(g):
    """Модель по строке GCAT. Ось тела X — длина; Y — поперёк (вдоль панелей); Z — третья ось."""
    shape = (g.get("Shape") or "Cyl").lower()
    L = _num(g.get("Length"), 2.0) or 2.0; D = _num(g.get("Diameter"), 1.0) or 1.0; S = _num(g.get("Span"), L) or L
    parts = []
    if "box" in shape:
        V, F = _box(0, 0, 0, L, D, D); parts.append((V, F, 1))
    elif "step" in shape:
        V, F = _cyl(L, D, steps=[(0.45, 1.0), (0.55, 0.75)]); parts.append((V, F, 1))
    else:
        V, F = _cyl(L, D); parts.append((V, F, 1))
    if "ant" in shape and S > L * 1.2:                       # штанга / антенна (например, «Целина»: Cyl + Ant)
        Vb_, Fb_ = _box(L / 2 + (S - L) / 2, 0, 0, S - L, 0.08, 0.08); parts.append((Vb_, Fb_, 1))
    m = re.search(r"(\d+)\s*pan", shape)
    npan = int(m.group(1)) if m else (1 if "pan" in shape else 0)
    arrays = []
    if npan and S > max(L, D) * 1.05:
        w = min(0.8 * L, max(1.0, 0.6 * L)); plen = (S - D) / 2 if npan >= 2 else (S - D)
        for side in ((1, -1) if npan >= 2 else (1,)):
            Vp, Fp = _box(0, side * (D / 2 + plen / 2 + 0.2), 0, w, plen, 0.05)
            parts.append((Vp, Fp, 0)); arrays.append(dict(side=side))
    V, F, G, masks = [], [], [], []
    for Vp, Fp, gid in parts:
        off = len(V); V += list(Vp); F += list(Fp + off); G += [gid] * len(Fp)
    V = np.array(V, float); F = np.array(F, int); G = np.array(G, int)
    arr = []
    for k, a in enumerate(arrays):
        Vp, Fp, _ = parts[1 + k]
        idx = np.unique(F[G == 0]); mask = np.zeros(len(V), bool)
        sel = idx[np.sign(V[idx, 1]) == a["side"]]; mask[sel] = True
        arr.append(dict(mask=mask, point=np.zeros(3), axis=np.array([0.0, 1.0, 0.0]), normal0=np.array([0.0, 0.0, 1.0])))
    return V, F, G, arr, f"параметрическая GCAT: {g.get('Shape')} L {L} D {D} размах {S}"


# ------------------------------------------------------------------ каталог детальных моделей (scripts/fetch_models.py)
CATDIR = MD / "catalog"
MANIFEST = MD / "catalog.json"
SATVIS_M2B = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])     # satvis (X port, Y зенит, Z скорость) → (вдоль, поперёк, радиус)
RB_M2B = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])         # ступень KeepTrack: ось Y сетки → ось X тела
SOLAR_RE = re.compile(r"solar|cell|photovolt|array|wing", re.I)
NONCELL_RE = re.compile(r"phased|yoke|stiffen|strut|brace|rail|edge|hardware|back|frame|rim|boom|hinge|bracket|connect|nub|trim|"
                        r"base|arm|instrument|radiator|side ?panel|antenna|dish|mount|handrail|detail|misc|gold|silver|block", re.I)
FIT_FACES = 8000                                                    # сетка для подгонки: не больше граней (растр на 1 м)


@lru_cache(maxsize=1)
def _manifest():
    import json
    return json.loads(MANIFEST.read_text())["models"] if MANIFEST.exists() else []


def _axis_rot(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a); t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def frame_matrix(e, g=None, ext=None):
    """Матрица 3×3: координаты сцены источника → оси каталога (satvis), метры. Для scale='gcat' — по размерам GCAT."""
    AX = dict(velocity=[0, 0, 1.0], port=[1.0, 0, 0], zenith=[0, 1.0, 0])
    R = np.eye(3)
    for st in e.get("rotate") or []:
        R = _axis_rot(AX[st["axis"]], st["degrees"]) @ R                      # шаги по порядку (кватернион step·q)
    sc = e.get("scale", 1.0)
    if sc == "gcat":
        sc = 1.0
        if g and ext is not None:
            L = _num(g.get("Length")); D = _num(g.get("Diameter")); S = _num(g.get("Span"))
            big = max([v for v in (L, D, S) if v] or [0])
            if big > 0:
                sc = big / float(np.max(ext))
    S3 = np.full(3, float(sc))
    if e.get("mirror"):
        S3[np.argmax(AX[e["mirror"]])] *= -1
    return R @ np.diag(S3)


JOINT_RE = re.compile(r"(^|[_\W])(rot|joint|gimbal|hinge|sada|drive|bapta)([_\W]|$)", re.I)


def _scene_parts(path):
    """Части сцены glTF/OBJ: dict(node, V (мир сцены), F, UV, mat, mname, joints). joints — цепочка узлов-шарниров
    модели над частью (от корня к листу): [(имя, начало узла в мире сцены)] — кинематическое дерево, если оно есть."""
    import trimesh
    sc = trimesh.load(str(path), force="scene", process=False)
    par = getattr(sc.graph.transforms, "parents", {}) or {}
    org = {}
    parts = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]; m = sc.geometry[gname]
        if not hasattr(m, "faces") or len(m.faces) == 0:
            continue
        chain, n = [], par.get(node)
        while n is not None and n != sc.graph.base_frame:
            if JOINT_RE.search(str(n)):
                if n not in org:
                    org[n] = np.asarray(sc.graph.get(n)[0], float)[:3, 3]
                chain.append((str(n), org[n]))
            n = par.get(n)
        uv = getattr(m.visual, "uv", None); mat = getattr(m.visual, "material", None)
        F = np.asarray(m.faces, int)
        if np.linalg.det(np.asarray(T, float)[:3, :3]) < 0:          # зеркальное преобразование узла: обход граней обратный
            F = F[:, ::-1]
        parts.append(dict(node=str(node), V=trimesh.transform_points(np.asarray(m.vertices, float), T), F=F,
                          UV=None if uv is None or len(uv) != len(m.vertices) else np.asarray(uv, float), mat=mat,
                          mname=str(getattr(mat, "name", None) or ""), joints=chain[::-1]))
    return parts


def _lum(mat):
    f = getattr(mat, "baseColorFactor", None)
    if f is None:
        f = getattr(mat, "main_color", None)
    if f is None:
        return 0.5
    f = np.asarray(f[:3], float); f = f / 255.0 if f.max() > 1.0 else f
    return float(f @ np.array([0.2126, 0.7152, 0.0722]))


def _is_solar(p):
    return bool((SOLAR_RE.search(p["node"]) or SOLAR_RE.search(p["mname"])) and not NONCELL_RE.search(p["node"]))


def _plum(p):
    """Яркость части после упрощения: медиана яркостей граней по текстуре, иначе — цвет материала."""
    return float(np.median(p["flum"])) if p.get("flum") is not None and len(p["flum"]) else p.get("lum", 0.5)


def _cluster(p, h):
    """Упрощение части кластеризацией вершин на сетке шага h (м): вершины в одной ячейке сливаются, вырожденные грани
    удаляются. Детерминированно, для частей любого размера; принадлежность граней части сохраняется."""
    u = np.unique(p["F"]); V = p["V"][u]
    key = np.floor(V / h).astype(np.int64)
    _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    remap = np.full(int(u.max()) + 1, -1); remap[u] = inv.ravel()
    F = remap[p["F"]]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    if len(F) == 0:
        return None
    F = np.unique(np.sort(F, 1), axis=0) if False else F
    Vn = np.zeros((len(rep), 3)); np.add.at(Vn, inv.ravel(), V); Vn /= np.bincount(inv.ravel(), minlength=len(rep))[:, None]
    return dict(p, V=Vn, F=F, UV=None, lum=_plum(p))


def _decimate(parts, target):
    """Сетка для подгонки (растр 1 м, пиксель снимка 10 м): кластеризация вершин с шагом от 0,1 % размера модели, шаг
    удваивается, пока граней больше target (не крупнее 3 % размера). Части меньше шага исчезают — на снимке их не видно."""
    tot = sum(len(p["F"]) for p in parts)
    if tot <= target:
        return parts
    allV = np.vstack([p["V"][np.unique(p["F"])] for p in parts]); size = float(np.max(allV.max(0) - allV.min(0)))
    h = 0.001 * size; out = parts
    while h <= 0.03 * size:
        out = [q for q in (_cluster(p, h) for p in parts) if q is not None]
        if sum(len(p["F"]) for p in out) <= target:
            break
        h *= 2
    return out


def _surface_samples(parts, n, seed=0):
    """Равномерные по площади точки на поверхности частей."""
    rng = np.random.default_rng(seed)
    T = np.vstack([p["V"][p["F"]] for p in parts if len(p["F"])])
    A = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    if A.sum() <= 0:
        return T.reshape(-1, 3)
    k = rng.choice(len(T), size=n, p=A / A.sum()); r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    return (1 - r1)[:, None] * T[k, 0] + (r1 * (1 - r2))[:, None] * T[k, 1] + (r1 * r2)[:, None] * T[k, 2]


def _wings(parts, solar, ext):
    """Крылья солнечных батарей: кластеры солнечных частей; шарнир — ось от корпуса к кластеру (с привязкой к осям
    каталога), точка — корень крыла; нормаль — наименьшая главная ось пластины. Возвращает список крыльев и число
    неподвижных (на корпусе) панелей."""
    size = float(np.max(ext)) if ext is not None else 1.0
    sp = [i for i, s in enumerate(solar) if s]
    if not sp:
        return [], 0
    box = {i: (parts[i]["V"].min(0), parts[i]["V"].max(0)) for i in range(len(parts))}
    pad = 0.02 * size; lab = {i: i for i in sp}

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]; i = lab[i]
        return i
    for a_ in sp:
        for b_ in sp:
            if a_ < b_ and np.all(box[a_][0] - pad <= box[b_][1]) and np.all(box[b_][0] - pad <= box[a_][1]):
                lab[find(a_)] = find(b_)
    clusters = {}
    for i in sp:
        clusters.setdefault(find(i), []).append(i)
    nons = [i for i in range(len(parts)) if not solar[i]]
    used = {i: np.unique(parts[i]["F"]) for i in range(len(parts))}
    Vn = np.vstack([parts[i]["V"][used[i]] for i in nons]) if nons else np.vstack([parts[i]["V"][used[i]] for i in sp])
    c0 = np.median(Vn, 0); r0 = float(np.median(np.linalg.norm(Vn - c0, axis=1))) + 1e-6
    # кластер → (часть, индексы вершин); кластер, охватывающий корпус с двух сторон (оба крыла в одной части), — делится
    groups = []
    for members in clusters.values():
        sel = [(i, used[i]) for i in members]
        Vc = np.vstack([parts[i]["V"][ix] for i, ix in sel])
        p = np.linalg.svd(Vc - Vc.mean(0), full_matrices=False)[2][0]
        t = (Vc - c0) @ p
        if len(sel) == 1 and np.linalg.norm(Vc.mean(0) - c0) < 0.8 * r0 and t.min() < -0.8 * r0 and t.max() > 0.8 * r0 \
                and np.mean(np.abs(t) < 0.3 * r0) < 0.05:
            # одна часть сетки, оба крыла по разные стороны корпуса — делится по вершинам
            for sgn in (1, -1):
                groups.append([(i, ix[np.sign((parts[i]["V"][ix] - c0) @ p) == sgn]) for i, ix in sel])
            continue
        # несколько частей: делятся по частям, если вдоль длины между ними есть разрыв (крылья по обе стороны фермы)
        todo = [sel]
        while todo:
            cur = todo.pop()
            if len(cur) < 2:
                groups.append(cur); continue
            Vk = np.vstack([parts[i]["V"][ix] for i, ix in cur]); pk = np.linalg.svd(Vk - Vk.mean(0), full_matrices=False)[2][0]
            iv = sorted(((float(((parts[i]["V"][ix] - Vk.mean(0)) @ pk).min()), float(((parts[i]["V"][ix] - Vk.mean(0)) @ pk).max()), (i, ix))
                         for i, ix in cur), key=lambda z: z[0])
            span = iv[-1][1] - iv[0][0] + 1e-9; hi_ = iv[0][1]; cut = None
            for k in range(1, len(iv)):
                if iv[k][0] - hi_ > 0.01 * span:
                    cut = k; break
                hi_ = max(hi_, iv[k][1])
            if cut is None:
                groups.append(cur)
            else:
                todo += [[z[2] for z in iv[:cut]], [z[2] for z in iv[cut:]]]
    wings, nfixed = [], 0
    for sel in groups:
        sel = [(i, ix) for i, ix in sel if len(ix)]
        if not sel:
            continue
        Vc = np.vstack([parts[i]["V"][ix] for i, ix in sel]); d = Vc.mean(0) - c0
        sv, W = np.linalg.svd(Vc - Vc.mean(0), full_matrices=False)[1:]
        elong = sv[0] > 1.3 * sv[1]

        def snap(v):
            v = v / (np.linalg.norm(v) + 1e-12); k = int(np.argmax(np.abs(v)))
            return np.eye(3)[k] * np.sign(v[k]) if abs(v[k]) > math.cos(math.radians(30)) else v
        through = elong and np.linalg.norm(d) < 0.8 * r0                    # одна панель через корпус (Kuiper)
        dn = d / (np.linalg.norm(d) + 1e-12)
        if through or (elong and abs(W[0] @ dn) > 0.5):                    # шарнир — вдоль длины крыла
            u = snap(W[0]); u = u * (np.sign(u @ d) or 1.0)
        elif np.linalg.norm(d) > 1e-6:
            u = snap(d)
        else:
            nfixed += 1; continue
        # панели на корпусе неподвижны: раскрывающееся крыло выходит далеко за ядро аппарата (за 2 медианных радиуса
        # несолнечной части от центра), панель на корпусе — нет
        pw = max(pad, 0.08 * float(np.max(Vc.max(0) - Vc.min(0))))        # допуск для каркаса крыла (тыльные балки)
        lo_, hi_ = Vc.min(0) - pw, Vc.max(0) + pw
        dq = np.linalg.norm(Vc - c0, axis=1)
        if np.quantile(dq, 0.9 if through else 0.75) < 2.0 * r0:            # не выходит за ядро — на корпусе
            nfixed += 1; continue
        pr = (Vc - c0) @ u
        root = Vc.mean(0) if through else Vc[pr <= pr.min() + 0.1 * (pr.max() - pr.min())].mean(0)
        e1 = np.cross(u, [0.0, 0, 1] if abs(u[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        Q = Vc - Vc.mean(0); Q2 = np.c_[Q @ e1, Q @ e2]                     # сечение крыла поперёк шарнира
        w2 = np.linalg.svd(Q2, full_matrices=False)[2][-1]                    # наименьший разброс — нормаль пластины
        n0 = w2[0] * e1 + w2[1] * e2; n0 /= np.linalg.norm(n0) + 1e-12
        lo, hi = lo_, hi_
        att = []                                                                # стрингеры, тыльные листы, коромысла
        for i in nons:
            ix = used[i]; Vi = parts[i]["V"][ix]
            inb = np.all((Vi >= lo) & (Vi <= hi), axis=1)
            if inb.mean() > 0.9:
                att.append((i, ix))
        # второй привод: первая ось — от корпуса к корню крыла (через корень), вторая — вдоль длины крыла (через его
        # середину). Если оси совпадают (крыло вытянуто от корпуса) — привод один.
        w = dict(members=sel + att, point=root, axis=u, normal0=n0, lo=lo, hi=hi, axis2=None, point2=None)
        if not through:                                    # ферменная схема: корень далеко, крыло поперёк направления на корень
            near = np.linalg.norm(Vc - c0, axis=1); rt = Vc[near <= np.quantile(near, 0.1)].mean(0)
            a1 = snap(rt - c0); a2 = snap(W[0]); a2 = a2 * (np.sign(a2 @ (Vc.mean(0) - rt)) or 1.0)
            length = float(np.ptp((Vc - Vc.mean(0)) @ W[0]))
            if np.linalg.norm(rt - c0) > 0.5 * length and abs(a1 @ a2) < 0.3 and np.max(np.abs(a1)) > 0.999:
                n1 = np.linalg.svd(Vc - Vc.mean(0), full_matrices=False)[2][-1]       # нормаль пластины
                w.update(point=rt, axis=a1, normal0=n1, axis2=a2, point2=Vc.mean(0))
        wings.append(w)
    return wings, nfixed


def _snap(v):
    v = np.asarray(v, float); v = v / (np.linalg.norm(v) + 1e-12); k = int(np.argmax(np.abs(v)))
    return np.eye(3)[k] * np.sign(v[k]) if abs(v[k]) > math.cos(math.radians(30)) else v


def _joint_wings(parts, solar, c0):
    """Крылья по кинематическому дереву модели: лист — ближайший к панели узел-шарнир (привод крыла), его родитель-шарнир —
    общий привод (у МКС BETA_ROT → ALPHA_ROT). Ось привода крыла — длина крыла; ось общего привода — от центра модели
    к шарниру. None — шарниров над панелями нет."""
    leaf = {}
    for i, (p, s) in enumerate(zip(parts, solar)):
        if s and p.get("joints"):
            leaf.setdefault(p["joints"][-1][0], (p["joints"], []))[1].append(i)
    if not leaf:
        return None
    wings = []
    for jn, (chain, sidx) in leaf.items():
        mem = [(i, np.unique(p["F"])) for i, p in enumerate(parts) if any(n == jn for n, _ in p.get("joints") or [])]
        Vs = np.vstack([parts[i]["V"][np.unique(parts[i]["F"])] for i in sidx])
        Vm = np.vstack([parts[i]["V"][ix] for i, ix in mem])
        piv = chain[-1][1]
        W = np.linalg.svd(Vs - Vs.mean(0), full_matrices=False)[2]
        a2 = _snap(W[0]); a2 = a2 * (np.sign(a2 @ (Vs.mean(0) - piv)) or 1.0); n0 = W[-1]
        w = dict(members=mem, joint=jn, normal0=n0, lo=Vm.min(0), hi=Vm.max(0))
        if len(chain) >= 2:
            pp = chain[-2][1]
            w.update(axis=_snap(pp - c0), point=pp, axis2=a2, point2=piv, joint1=chain[-2][0])
        else:
            w.update(axis=a2, point=piv, axis2=None, point2=None)
        wings.append(w)
    return wings


def _cell_normal(parts, w):
    """Нормаль стороны крыла с солнечными элементами (оси каталога). Элементы темнее подложки: грани крыла взвешиваются
    (медианная яркость − яркость грани) — у двух совпадающих листов (элементы / подложка) с противоположным обходом
    вклады складываются в сторону элементов. None — сторону не определить (лист одинаков с обеих сторон)."""
    acc = np.zeros(3); tot = 0.0; recs = []
    for i, p in enumerate(parts):
        if w.get("joint"):
            if not any(n == w["joint"] for n, _ in p.get("joints") or []):
                continue
        else:
            ix = np.unique(p["F"]); inb = np.all((p["V"][ix] >= w["lo"]) & (p["V"][ix] <= w["hi"]), axis=1)
            if inb.mean() < 0.9:
                continue
        V, F = p["V"], p["F"]
        c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        lum = p["flum"] if p.get("flum") is not None and len(p["flum"]) == len(F) else np.full(len(F), p.get("lum", 0.5))
        recs.append((c, lum))
    if not recs:
        return None
    allc = np.vstack([c for c, _ in recs]); alll = np.concatenate([l for _, l in recs])
    area = np.linalg.norm(allc, axis=1); med = float(np.median(np.repeat(alll, 1)))
    acc = ((med - alll)[:, None] * allc).sum(0); tot = float((np.abs(med - alll) * area).sum())
    if tot <= 0 or np.linalg.norm(acc) < 0.2 * tot:
        return None
    return acc / np.linalg.norm(acc)


def load_catalog_model(e, g=None, measure_only=False, fit=True):
    """Модель каталога в осях каталога (satvis, метры). Возвращает dict: V, F, G, arrays (для подгонки — упрощённая сетка),
    parts (полная детализация для рендера), M2B, tri, dims_m (вдоль, поперёк, радиус), n_arrays, n_fixed."""
    path = (CATDIR / e["file"]).resolve()
    if path.suffix == ".obj":
        V, F, MAT, OBJ, kd = load_obj(path)
        parts = [dict(node=str(m), V=V, F=F[MAT == m], UV=None, mat=None, mname=str(m)) for m in np.unique(MAT)]
        for p in parts:
            p["lum"] = float(kd.get(p["mname"], np.array([0.5] * 3)).mean())
    else:
        parts = _scene_parts(path)
        for p in parts:
            p["lum"] = _lum(p["mat"])
            tex = getattr(p["mat"], "baseColorTexture", None)
            if tex is not None and p["UV"] is not None and len(p["F"]):          # яркость граней по текстуре
                T = np.asarray(tex.convert("RGB"), np.float32) / 255.0
                uv = p["UV"][p["F"]].mean(1); h, w = T.shape[:2]
                cx = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int); cy = np.clip((1 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
                p["flum"] = (T[cy, cx] @ np.array([0.2126, 0.7152, 0.0722])) * p["lum"]
    allV = np.vstack([p["V"][np.unique(p["F"])] for p in parts]); ext0 = allV.max(0) - allV.min(0)
    if e.get("frame") == "rb":                                    # ступень: ось — Y сетки; длина и диаметр — из GCAT
        L = _num((g or {}).get("Length")); D = _num((g or {}).get("Diameter"))
        sy = L / ext0[1] if L else 1.0; sxz = D / max(ext0[0], ext0[2]) if D else 1.0
        M = np.diag([sxz, sy, sxz]); M2B = RB_M2B
    else:
        M = frame_matrix(e, g, ext0); M2B = SATVIS_M2B
    for p in parts:
        p["V"] = p["V"] @ M.T
    if np.linalg.det(M) < 0:                                       # зеркало: вернуть обход граней
        for p in parts:
            p["F"] = p["F"][:, ::-1]
    allV = np.vstack([p["V"][np.unique(p["F"])] for p in parts]); c = 0.5 * (allV.min(0) + allV.max(0))
    for p in parts:
        p["V"] = p["V"] - c
        if p.get("joints"):
            p["joints"] = [(n, M @ o - c) for n, o in p["joints"]]
    ext = allV.max(0) - allV.min(0)
    tri = int(sum(len(p["F"]) for p in parts))
    dp = _decimate(parts, FIT_FACES)                               # шарниры ищутся на упрощённой сетке (одинаково для
    dsol = [_is_solar(p) for p in dp]                              # подгонки и рендера)
    nfixed = 0
    wings = None
    if e.get("kind") != "rocketbody":
        Vn = np.vstack([p["V"][np.unique(p["F"])] for p, s_ in zip(dp, dsol) if not s_] or [np.zeros((1, 3))])
        wings = _joint_wings(dp, dsol, np.median(Vn, 0))           # кинематическое дерево модели, если есть
        if wings is None:
            wings, nfixed = _wings(dp, dsol, ext)                  # иначе — по геометрии
    wings = wings or []
    for w in wings:                                                # сторона с элементами — по полной сетке и текстуре
        w["cell_normal"] = _cell_normal(parts, w)
    fp = dp if fit else parts
    solar = [_is_solar(p) for p in fp]
    if not fit:                                                    # перенести крылья на полную сетку
        for w in wings:
            mem = []
            for i, p in enumerate(fp):
                ix = np.unique(p["F"])
                if w.get("joint"):                                 # по шарниру — точно
                    if any(n == w["joint"] for n, _ in p.get("joints") or []):
                        mem.append((i, ix))
                    continue
                inb = np.all((p["V"][ix] >= w["lo"]) & (p["V"][ix] <= w["hi"]), axis=1)   # по габариту крыла
                if solar[i] and inb.any():
                    mem.append((i, ix[inb]))
                elif not solar[i] and inb.mean() > 0.9:
                    mem.append((i, ix))
            w["members"] = mem
    out = dict(tri=tri, dims_m=[round(float(ext[2]), 2), round(float(ext[0]), 2), round(float(ext[1]), 2)],
               n_arrays=len(wings), n_fixed=nfixed, solar_faces=int(sum(len(p["F"]) for p, s in zip(fp, solar) if s)), M2B=M2B)
    if measure_only:
        return out
    V, F, G, gidx = [], [], [], {}
    for k, (p, s) in enumerate(zip(fp, solar)):
        u, inv = np.unique(p["F"], return_inverse=True); off = len(V)
        V += list(p["V"][u]); F += list(inv.reshape(-1, 3) + off)
        if s:
            G += [0] * len(p["F"])
        elif "flum" in p and len(p["flum"]) == len(p["F"]):
            G += list(np.where(p["flum"] > 0.45, 1, 2))
        else:
            G += [1 if p["lum"] > 0.45 else 2] * len(p["F"])
        gidx[k] = (u, off)                                        # исходный индекс вершины части → общий индекс
    V = np.array(V, float); F = np.array(F, int); G = np.array(G, int)
    arrays = []
    for w in wings:
        mask = np.zeros(len(V), bool)
        for k, ix in w["members"]:
            u, off = gidx[k]; mask[off + np.searchsorted(u, ix)] = True
        sub = [] if w.get("axis2") is None else [dict(mask=mask, point=w["point2"], axis=w["axis2"], normal0=w["normal0"])]
        arrays.append(dict(mask=mask, point=w["point"], axis=w["axis"], normal0=w["normal0"], sub=sub, cell=w.get("cell_normal")))
    out.update(V=V, F=F, G=G, arrays=arrays, parts=fp, wings=wings, gidx=gidx)
    return out


def _rb_entry(g):
    """Ступень → сетка KeepTrack: силуэт по форме GCAT, семейство по имени, иначе — по длине."""
    import json
    man = {e["key"]: e for e in _manifest()}
    rules = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    shape = re.sub(r"\s+", "", (g.get("Shape") or "").lower())
    key = rules.get("rb_shapes", {}).get(shape)
    nm = (g.get("Name") or "").lower()
    if not key:
        for rx, k in rules.get("rb_families", []):
            if re.search(rx, nm):
                key = k; break
        if key == "soviet":
            L = _num(g.get("Length"), 0) or 0; D = _num(g.get("Diameter"), 1) or 1
            key = "rb-cyl-soviet-b" if L / D < 2.0 else "rb-cyl-soviet"
    if not key:
        L = _num(g.get("Length"), 0) or 0
        key = "rb-cyl-gray-s" if L < 5 else "rb-cyl-gray-m" if L < 10 else "rb-cyl-gray-l"
    return man.get(key)


def resolve(norad, name=""):
    """Запись каталога для объекта: номер NORAD → шина GCAT → имя → ступень по правилам; None — нет модели."""
    norad = int(norad or 0); g = gcat(norad) or {}
    man = _manifest()
    for e in man:
        if norad and norad in (e["match"].get("norad") or []):
            return e
    bus = (g.get("Bus") or "").strip().lower()
    if bus and bus != "-":
        for e in man:
            if bus in [b.lower() for b in e["match"].get("bus") or []]:
                return e
    nm = g.get("Name") or name
    typ = (g.get("Type") or "")[:1]
    for e in man:
        rx = e["match"].get("name")
        if rx and re.search(rx, nm, re.I) and (not e["match"].get("type") or e["match"]["type"] == typ):
            return e
    if typ == "R":
        return _rb_entry(g)
    return None


# ------------------------------------------------------------------ законы ориентации (одни на класс, без исключений)
ARRAY_MODES = (("sun", "панели на Солнце"), ("edge", "панели ребром к Солнцу"), ("flat", "панели плашмя"),
               ("vertical", "панели перпендикулярно корпусу"), ("fixed", "панели неподвижны"),
               ("angle", "панели под фиксированным углом (угол по снимку)"))
YAWS = (0, 90, 180, 270)


@lru_cache(maxsize=1)
def _ops():
    """Статус работы из CelesTrak SATCAT: '+' работает, 'P' частично, 'B' резерв, 'S' запасной, 'X' продлён, '-' не работает,
    '?' неизвестно."""
    f = CAT / "celestrak_satcat.csv.gz"
    if not f.exists():
        return {}
    return {int(r["NORAD_CAT_ID"]): r["OPS_STATUS_CODE"] for r in csv.DictReader(io.TextIOWrapper(gzip.open(f), encoding="utf-8"))}


def is_active(norad, year=2026):
    """Работает ли аппарат (управляемая ориентация): по SATCAT; если статуса нет — моложе 15 лет."""
    st = _ops().get(int(norad or 0), "")
    if st:
        return st in "+PBSX"
    return year >= 2011


def laws_for(kind, has_arrays, frame_ok, year, active=True):
    """Законы ориентации класса. Аппарат с выверенными осями модели: на Землю (оси тела = орбитальные), рысканье
    0/90/180/270°, режим панелей; неработающий (SATCAT) — ещё гравитационная стабилизация и кувыркание. Оси модели
    не выверены — только оценка поворота. Ступень, обломок — ось по сфере."""
    if kind in ("rocketbody", "debris"):
        return [dict(name="кувыркание: ось по сфере", tumble=True)]
    if not frame_ok:
        return [dict(name="поворот оценивается (оси модели не выверены)", free3=True)]
    if has_arrays:                                         # углы приводов панелей — по снимку (режим — вывод);
        laws = [dict(name=f"на Землю, рысканье {y}°, углы панелей по снимку", yaw=y, arrays="est") for y in YAWS]
        # соперник: слежение за Солнцем без подбора углов — если свободные углы не предсказывают отложенные полосы
        # лучше, отклонение от слежения снимком не подтверждено
        laws += [dict(name=f"на Землю, рысканье {y}°, панели следят за Солнцем", yaw=y, arrays="sun") for y in YAWS]
    else:
        laws = [dict(name=f"на Землю, рысканье {y}°", yaw=y, arrays="fixed") for y in YAWS]
    if not active:
        laws += [dict(name="гравитационная стабилизация: продольная ось к Земле", gg=True),
                 dict(name="кувыркание: ось по сфере", tumble=True)]
    return laws


def spec_for(norad, name=""):
    """Модель, оси и законы ориентации для любого объекта — один и тот же путь для всех: каталог (NORAD → шина GCAT →
    имя → правила ступеней), иначе параметрическая модель по размерам GCAT. Возвращает dict(kind, V, F, G, M2B, arrays,
    laws, source, catalog, gcat)."""
    norad = int(norad or 0)
    g = gcat(norad) or {}
    typ = (g.get("Type") or "P")[:1]
    year = int((g.get("LDate") or "2026")[:4]) if (g.get("LDate") or "")[:4].isdigit() else 2026
    e = resolve(norad, name)
    if e is not None and not e.get("error"):
        M = load_catalog_model(e, g)
        kind = "rocketbody" if e["kind"] == "rocketbody" or typ == "R" else "debris" if typ == "D" else "payload"
        return dict(kind=kind, V=M["V"], F=M["F"], G=M["G"], M2B=M["M2B"], arrays=M["arrays"],
                    laws=laws_for(kind, bool(M["arrays"]), e["frame"] == "satvis", year, is_active(norad, year)),
                    source=f"каталог: {e['key']} ({e['source']}; {M['tri']} тр., для подгонки {len(M['F'])})", catalog=e, gcat=g)
    V, F, G, arrays, src = parametric(g)
    kind = "payload" if typ == "P" else "rocketbody" if typ == "R" else "debris"
    return dict(kind=kind, V=V, F=F, G=G, M2B=np.eye(3), arrays=arrays, laws=laws_for(kind, bool(arrays), True, year, is_active(norad, year)),
                source=src, catalog=None, gcat=g)
