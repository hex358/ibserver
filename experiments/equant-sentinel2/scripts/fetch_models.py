"""Каталог детальных 3D-моделей (THEORY §24): скачать и собрать воспроизводимо, по закреплённым версиям источников.

  python scripts/fetch_models.py            # скачать/сгенерировать в data/models/catalog/, записать data/models/catalog.json
  python scripts/fetch_models.py --manifest # только пересобрать манифест из уже скачанного

Источники (лицензии — в манифесте у каждой модели):
  * NASA-3D-Resources (github.com/nasa/NASA-3D-Resources, коммит NASA_COMMIT) — модели аппаратов NASA. Для тех, что
    выверены satvis-3d-models (оси, масштаб в метрах, зеркальность), берётся его рецепт (build.yaml); остальные —
    с масштабом по размерам из GCAT и без выверенных осей (поза — только свободная).
  * satvis-3d-models (github.com/Flowm/satvis-3d-models, MIT, коммит SATVIS_COMMIT) — процедурные модели по
    опубликованным размерам: Тяньгун, Starlink v1/V2 mini/DTC/V3, OneWeb, Kuiper, Qianfan, Guowang, ГЛОНАСС-М, BeiDou,
    GPS, Galileo, Iridium NEXT, ICEYE, Globalstar, O3b, Planet, Kepler, кубсаты, геостационарные платформы.
  * KeepTrack (keeptrack.space) — ступени ракет (лежат в data/models/rb-*.obj, в метрах 1:1).
Система осей каталога — satvis/Cesium: +Z вдоль скорости, +Y в зенит, +X к левому борту (port), метры.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data/models/catalog"
MANIFEST = ROOT / "data/models/catalog.json"
NASA_REPO = "nasa/NASA-3D-Resources"
NASA_COMMIT = "11ebb4ee043715aefbba6aeec8a61746fad67fa7"
SATVIS_REPO = "https://github.com/Flowm/satvis-3d-models"
SATVIS_COMMIT = "31ca896dcd278f4778ff8e06f0948834bcac2c58"
NASA_LICENSE = "NASA media usage guidelines (public domain, без логотипов NASA как одобрения)"

# Модели NASA, которых нет в рецепте satvis: ключ → (путь в репозитории, сопоставление с GCAT, тип).
# match: norad — номера; bus — шина GCAT; name — регулярное выражение по имени GCAT.
EXTRA_NASA = {
    "TSELINA-2": ("Tselina-2/Tselina-2.glb", dict(bus=["Tselina-2"]), "payload"),
    "HST": ("Hubble Space Telescope (B)/Hubble Space Telescope (B).glb", dict(norad=[20580], bus=["HST"]), "payload"),
    "AQUA": ("Aqua (C)/Aqua (C).glb", dict(norad=[27424]), "payload"),
    "LANDSAT-7": ("Landsat 7/Landsat 7.glb", dict(norad=[25682]), "payload"),
    "LANDSAT-5": ("Landsat 4 and 5/Landsat 4 and 5.glb", dict(norad=[13367, 14780]), "payload"),
    "LANDSAT-1": ("Landsat 1, 2, and 3/Landsat 1, 2, and 3.glb", dict(norad=[6126, 7615, 10702]), "payload"),
    "CALIPSO": ("Cloud-Aerosol Lidar and Infrared Pathfinder Satellite (CALIPSO)/Cloud-Aerosol Lidar and Infrared Pathfinder Satellite (CALIPSO).glb",
                dict(norad=[29108]), "payload"),
    "NOAA-POES": ("Polar Operational Environmental Satellite (POES)/Polar Operational Environmental Satellite (POES).glb",
                  dict(bus=["ATN", "TIROS-N"]), "payload"),
    "QUIKSCAT": ("Quick Scatterometer (QuikSCAT)/Quick Scatterometer (QuikSCAT).glb", dict(norad=[25789]), "payload"),
    "TOPEX": ("TOPEX-Poseidon/TOPEX-Poseidon.glb", dict(norad=[22076]), "payload"),
    "JASON-1": ("Jason 1/Jason 1.glb", dict(norad=[26997]), "payload"),
    "JASON-2": ("Ocean Surface Topography Mission (OSTM Jason-2)/Ocean Surface Topography Mission (OSTM Jason-2).glb",
                dict(norad=[33105]), "payload"),
    "SORCE": ("Solar Radiation and Climate Experiment (SORCE)/Solar Radiation and Climate Experiment (SORCE).glb", dict(norad=[27651]), "payload"),
    "RADARSAT-1": ("Radar Satellite-1 (RADARSAT-1)/Radar Satellite-1 (RADARSAT-1).glb", dict(norad=[23710]), "payload"),
    "ORBVIEW-2": ("SeaStar/SeaStar.glb", dict(norad=[24883]), "payload"),
    "SAC-C": ("Satellite for Scientific Applications (SAC-C)/Satellite for Scientific Applications (SAC-C).glb", dict(norad=[26620]), "payload"),
    "EO-1": ("Earth Observing-1 (EO-1)/Earth Observing-1 (EO-1).glb", dict(norad=[26619]), "payload"),
    "ACRIMSAT": ("Active Cavity Irradiance Monitor Satellite (AcrimSAT) (A)/Active Cavity Irradiance Monitor Satellite (AcrimSAT) (A).glb",
                 dict(norad=[26033]), "payload"),
    "ICON": ("Ionospheric Connection Explorer (ICON)/Ionospheric Connection Explorer (ICON).glb", dict(norad=[44628]), "payload"),
    "SWAS": ("Submillimeter Wave Astronomy Satellite (SWAS)/Submillimeter Wave Astronomy Satellite (SWAS).glb", dict(norad=[25560]), "payload"),
    "AIM": ("Aeronomy of Ice in the Mesosphere/Aeronomy of Ice in the Mesosphere.glb", dict(norad=[31304]), "payload"),
    "AQUARIUS": ("Aquarius (A)/Aquarius (A).glb", dict(norad=[37673]), "payload"),
    "ICESAT": ("Ice, Clouds, and Land Elevation Satellite (ICESat) (A)/Ice, Clouds, and Land Elevation Satellite (ICESat) (A).glb",
               dict(norad=[27642]), "payload"),
    "GRACE": ("Gravity Recovery and Climate Experiment (GRACE) (A)/Gravity Recovery and Climate Experiment (GRACE) (A).glb",
              dict(norad=[27391, 27392]), "payload"),
    "TRMM": ("Tropical Rainfall Measuring Mission (TRMM)/Tropical Rainfall Measuring Mission (TRMM).glb", dict(norad=[25063]), "payload"),
    "CNOFS": ("Communication and Navigation Outage Forecast System (CNOFS)/Communication and Navigation Outage Forecast System (CNOFS).glb",
              dict(norad=[32765]), "payload"),
    "RHESSI": ("HESSI-RHESSI/HESSI-RHESSI.glb", dict(norad=[27370]), "payload"),
    "SUZAKU": ("Suzaku/Suzaku.glb", dict(norad=[28773]), "payload"),
    "CGRO": ("Gamma Ray Observatory/Gamma Ray Observatory.glb", dict(norad=[21225]), "payload"),
    "WIRE": ("Wide-Field Infrared Explorer (WIRE)/Wide-Field Infrared Explorer (WIRE).glb", dict(norad=[25646]), "payload"),
    "IBEX": ("Interstellar Boundary Explorer (IBEX)/Interstellar Boundary Explorer (IBEX).glb", dict(norad=[33401]), "payload"),
    "TOMS-EP": ("Total Ozone Mapping Spectrometer (TOMS)/Total Ozone Mapping Spectrometer (TOMS).glb", dict(name=r"^TOMS"), "payload"),
    "CUBESAT-ICECUBE": ("CubeSat - ICECube/CubeSat - ICECube.glb", dict(name=r"^IceCube$"), "payload"),
    "CUBESAT-MIRATA": ("CubeSat - MiRaTa/CubeSat - MiRaTa.glb", dict(name=r"^MiRaTA"), "payload"),
    "AGENA": ("Agena Target Vehicle/Agena Target Vehicle.glb", dict(name=r"Agena", type="R"), "rocketbody"),
    "MIR": ("Mir/Mir.glb", dict(norad=[16609]), "station"),
    "SKYLAB": ("Skylab/Skylab.glb", dict(norad=[6633]), "station"),
}

# Ступени KeepTrack (data/models/rb-*.obj): ось ступени — Y сетки, метры. Правила — как в resolver KeepTrack:
# силуэт по форме GCAT, затем семейство ракеты по имени, затем размер.
RB_SHAPES = {"cyl+cone": "rb-cylcone-soviet", "stepcyl": "rb-stepcyl-soviet", "cyl+cyl": "rb-stepcyl-soviet",
             "cyl+2nozzle": "rb-cyl2n-legacy", "sphere+cone": "rb-sphercone-kick", "trunccone": "rb-trunccone-gray",
             "truncatedcone": "rb-trunccone-gray"}
RB_FAMILIES = [(r"falcon", "rb-cyl-kerolox"), (r"electron", "rb-cyl-electron"), (r"ariane|vega", "rb-cyl-euro"),
               (r"chang zheng|long march|^cz[\s-]", "rb-cyl-china"), (r"atlas|centaur|titan", "rb-cyl-centaur"),
               (r"delta (?:iv|4)", None), (r"delta", "rb-cyl-delta"), (r"proton|briz|breeze|fregat", "rb-cyl-proton"),
               (r"kosmos|cosmos|tsiklon|tsyklon|cyclone|vostok|voskhod|molniya|soyuz|rokot|dnepr|zenit|shtil|angara|sputnik|strela",
                "soviet")]


def sh(*cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: {r.stderr[-400:]}")
    return r.stdout


def nasa_url(path):
    return f"https://raw.githubusercontent.com/{NASA_REPO}/{NASA_COMMIT}/" + "/".join(urllib.parse.quote(s) for s in path.split("/"))


def curl(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    sh("curl", "-sfL", "--retry", "3", "-o", str(out), url)


def satvis_recipe(work):
    src = work / "satvis-3d-models"
    if not src.exists():
        sh("git", "clone", "-q", SATVIS_REPO, str(src)); sh("git", "checkout", "-q", SATVIS_COMMIT, cwd=src)
    import yaml
    text = (src / "build.yaml").read_text()
    try:
        y = yaml.safe_load(text)
    except yaml.YAMLError:
        y = yaml.safe_load(text.replace("?", ""))
    return src, y["models"]


def generate_satvis(src, names, out):
    """Генераторы satvis: Node + @gltf-transform/core, сетки без сжатия и упрощения (полная детализация)."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (out / f"{n}.glb").exists()]
    if not todo:
        return
    gen = src / "gen_ocellus.ts"
    gen.write_text(GEN_TS)
    if not (src / "node_modules/@gltf-transform/core").exists():
        sh("npm", "install", "--no-save", "--silent", "@gltf-transform/core@4", cwd=src)
    print(sh("node", str(gen), str(out), cwd=src)[-2000:])


GEN_TS = r'''
import { NodeIO } from "@gltf-transform/core";
import { CUBESATS } from "./generators/cubesat.ts";
import * as GEO from "./generators/geostationary.ts";
import * as GN from "./generators/gnss.ts";
import { globalstar2 } from "./generators/globalstar.ts";
import { guowang } from "./generators/guowang.ts";
import { iceye } from "./generators/iceye.ts";
import { iridiumNext } from "./generators/iridium.ts";
import { keplerTranche1, keplerTranche1Safire } from "./generators/kepler.ts";
import { kuiper } from "./generators/kuiper.ts";
import { o3b, o3bMpower } from "./generators/o3b.ts";
import { oneweb } from "./generators/oneweb.ts";
import { pelican, skySat, superDove } from "./generators/planet.ts";
import { qianfan } from "./generators/qianfan.ts";
import { tiangong } from "./generators/tiangong.ts";
import * as SL from "./generators/starlink.ts";
const out = process.argv[2];
const all: Record<string, () => any> = {
  "css-(tianhe)": tiangong, "starlink-v1": SL.starlinkV1, "starlink-v2-mini": SL.starlinkV2Mini,
  "starlink-v2-mini-direct-to-cell": SL.starlinkV2MiniDirectToCell, "starlink-v3": SL.starlinkV3, oneweb, kuiper, qianfan, guowang,
  "glonass-m": GN.glonassM, "beidou-3-cast": GN.beidou3Cast, "beidou-3-secm": GN.beidou3Secm, "gps-iir": GN.gpsIIR, "gps-iif": GN.gpsIIF,
  "gps-iii": GN.gpsIII, "galileo-foc": GN.galileoFoc, "iridium-next": iridiumNext, iceye, "globalstar-2": globalstar2, o3b,
  "o3b-mpower": o3bMpower, superdove: superDove, skysat: skySat, pelican, "kepler-tranche-1": keplerTranche1,
  "kepler-tranche-1-safire": keplerTranche1Safire, "eurostar-3000": GEO.eurostar3000, "bss-702hp": GEO.bss702hp, "dfh-4": GEO.dfh4,
  a2100: GEO.a2100, "dfh-3": GEO.dfh3, "dfh-3-navigation": GEO.dfh3Navigation, "spacebus-4000": GEO.spacebus4000,
  "spacebus-neo": GEO.spacebusNeo, "eurostar-neo": GEO.eurostarNeo, "bss-702sp": GEO.bss702sp, "i-2k": GEO.i2k, "insat-3d": GEO.insat3d,
  "i-1k": GEO.i1k, "i-3k": GEO.i3k, "ekspress-1000": GEO.ekspress1000,
};
for (const [k, f] of Object.entries(CUBESATS)) all[k.toLowerCase()] = f;
const io = new NodeIO();
for (const [name, f] of Object.entries(all)) {
  try { await io.write(`${out}/${name}.glb`, f()); console.log("ok", name); } catch (e) { console.log("FAIL", name, String(e).slice(0, 200)); }
}
'''


def fetch():
    work = Path(tempfile.gettempdir()) / "ocellus_models"; work.mkdir(exist_ok=True)
    src, models = satvis_recipe(work)
    shutil.copy(src / "LICENSE", CAT / "satvis_LICENSE") if CAT.exists() else None
    for m in models:
        s = m.get("source", {})
        if s.get("id") == "nasa":
            curl(nasa_url(s["path"]), CAT / "nasa" / Path(m["file"]).name)
            print("NASA", m["file"])
    generate_satvis(src, [m["source"]["generator"] for m in models if m.get("source", {}).get("generator")], CAT / "satvis")
    for k, (p, _, _) in EXTRA_NASA.items():
        curl(nasa_url("3D Models/" + p), CAT / "nasa_extra" / f"{k}.glb"); print("NASA+", k)
    (CAT / "satvis_build.yaml").write_text((src / "build.yaml").read_text())
    shutil.copy(src / "LICENSE", CAT / "satvis_LICENSE")


def manifest():
    import yaml
    text = (CAT / "satvis_build.yaml").read_text()
    try:
        models = yaml.safe_load(text)["models"]
    except yaml.YAMLError:
        models = yaml.safe_load(text.replace("?", ""))["models"]
    ent = []
    for m in models:
        s = m.get("source", {})
        if s.get("id") == "nasa":
            f = CAT / "nasa" / Path(m["file"]).name; src = f"NASA-3D-Resources@{NASA_COMMIT[:7]}: {s['path']}"; lic = NASA_LICENSE
        elif s.get("generator"):
            f = CAT / "satvis" / f"{s['generator']}.glb"; src = f"satvis-3d-models@{SATVIS_COMMIT[:7]} генератор {s['generator']}"; lic = "MIT (satvis-3d-models)"
        else:
            continue
        if not f.exists():
            print("нет файла", f); continue
        key = Path(m["file"]).stem
        norad = [int(v["noradId"]) for v in m.get("satellites", []) if v.get("noradId")]
        kind = "station" if key in ("ISS-(ZARYA)", "CSS-(TIANHE)") else "payload"
        ent.append(dict(key=key, file=str(f.relative_to(CAT)), source=src, license=lic, frame="satvis",
                        rotate=m.get("rotate") or [], mirror=m.get("mirror"), scale=m.get("scale") or 1.0,
                        match=dict(norad=norad, bus=list(m.get("buses") or [])), kind=kind))
    for k, (p, match, kind) in EXTRA_NASA.items():
        f = CAT / "nasa_extra" / f"{k}.glb"
        if f.exists():
            ent.append(dict(key=k, file=str(f.relative_to(CAT)), source=f"NASA-3D-Resources@{NASA_COMMIT[:7]}: 3D Models/{p}",
                            license=NASA_LICENSE, frame="unverified", rotate=[], mirror=None, scale="gcat", match=match, kind=kind))
    for f in sorted((ROOT / "data/models").glob("rb-*.obj")):
        ent.append(dict(key=f.stem, file="../" + f.name, source="KeepTrack (keeptrack.space) scripts/mesh-gen", license="AGPL-3.0 (KeepTrack)",
                        frame="rb", rotate=[], mirror=None, scale="gcat", match=dict(rb=True), kind="rocketbody"))
    sys.path.insert(0, str(ROOT))
    from s2sat import modelcat as MC
    G = MC._gcat()
    for e in ent:                                    # измерить: треугольники, габариты в осях каталога, панели
        mt = e["match"]; g = None                    # строка GCAT для масштаба «gcat»: первый подходящий объект
        for n in mt.get("norad") or []:
            g = G.get(int(n)) or g
            if g:
                break
        if g is None and (mt.get("bus") or mt.get("name")):
            bl = [b.lower() for b in mt.get("bus") or []]
            g = next((r for r in G.values() if (r.get("Bus") or "").lower() in bl
                      or (mt.get("name") and re.search(mt["name"], r.get("Name") or "", re.I))), None)
        e["matched_gcat"] = sum(1 for r in G.values() if (r.get("Bus") or "").lower() in [b.lower() for b in mt.get("bus") or []]) \
            + len(mt.get("norad") or [])
        try:
            M = MC.load_catalog_model(e, g=g, measure_only=True)
            e.update(tri=M["tri"], dims_m=M["dims_m"], arrays=M["n_arrays"], fixed_panels=M["n_fixed"], solar_faces=M["solar_faces"])
        except Exception as ex:                      # noqa: BLE001 — отметить и продолжить
            e["error"] = str(ex)[:200]
        print(f"{e['key']:28s} {e.get('tri', 0):9d} тр.  {e.get('dims_m')}  крыльев {e.get('arrays')}  {e.get('error', '')}")
    out = dict(frame="satvis/Cesium: +Z вдоль скорости, +Y в зенит, +X к левому борту; метры",
               sources=dict(nasa=f"{NASA_REPO}@{NASA_COMMIT}", satvis=f"{SATVIS_REPO}@{SATVIS_COMMIT}"),
               rb_shapes=RB_SHAPES, rb_families=RB_FAMILIES, models=ent)
    MANIFEST.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("манифест:", MANIFEST, len(ent), "моделей")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", action="store_true"); a = ap.parse_args()
    CAT.mkdir(parents=True, exist_ok=True)
    if not a.manifest:
        fetch()
    manifest()
