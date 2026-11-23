"""История орбит из веб-архива Internet Archive (без регистрации): архивные копии каталогов CelesTrak.

Запускается в GitHub Action (из контейнера web.archive.org закрыт): читает data/tle_history_request.txt
  строка 1: from YYYYMMDD, строка 2: to YYYYMMDD, строка 3: NORAD ID через запятую (+ всегда SENTINEL-2*)
и сохраняет из каждой архивной копии (gp.php?GROUP=starlink/active, sup-gp.php?FILE=starlink, starlink.txt …)
только нужные объекты: data/tle_history/<timestamp>_<вид>.tle. Повторные копии (тот же digest) пропускаются.
"""
import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "data/tle_history_request.txt"
OUT = ROOT / "data/tle_history"
lines = [l.strip() for l in REQ.read_text().splitlines() if l.strip() and not l.startswith("#")]
d0, d1 = lines[0], lines[1]
cats = set(x.strip() for x in lines[2].split(",") if x.strip())
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "s2sat-research"}


def get(url, tries=4, timeout=120):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as ex:
            print("  retry", i, url[:120], repr(ex)[:120], flush=True)
            time.sleep(5 * (i + 1))
    return None


def kind_of(orig):
    o = orig.lower()
    sup = "supplemental" in o or "sup-gp" in o
    grp = re.search(r"(?:group|file)=([a-z0-9\-]+)", o)
    name = grp.group(1) if grp else Path(urllib.parse.urlparse(o).path).stem
    return ("supgp_" if sup else "gp_") + name


caps = []
for host in ("celestrak.org", "celestrak.com", "www.celestrak.com", "www.celestrak.org"):
    q = (f"https://web.archive.org/cdx/search/cdx?url={host}/NORAD/elements/&matchType=prefix&from={d0}&to={d1}"
         f"&filter=statuscode:200&output=json&fl=timestamp,original,digest,mimetype,length&collapse=digest")
    raw = get(q)
    if not raw:
        continue
    try:
        rows = json.loads(raw)[1:]
    except Exception:
        continue
    for ts, orig, dig, mt, ln in rows:
        o = orig.lower()
        if any(k in o for k in ("starlink", "active", "sentinel", "resource", "visual")) and "format=json" not in o and "format=csv" not in o \
                and "format=xml" not in o and "format=kvn" not in o:
            caps.append((ts, orig, dig))
print(f"архивных копий: {len(caps)}", flush=True)
seen = set(); n_ok = 0
for ts, orig, dig in sorted(caps):
    if dig in seen:
        continue
    seen.add(dig)
    raw = get(f"https://web.archive.org/web/{ts}id_/{orig}")
    if not raw:
        continue
    txt = raw.decode("utf-8", "replace").splitlines()
    keep = []
    for i in range(len(txt) - 2):
        l0, l1, l2 = txt[i].rstrip(), txt[i + 1].rstrip(), txt[i + 2].rstrip()
        if l1.startswith("1 ") and l2.startswith("2 ") and not l0.startswith(("1 ", "2 ")):
            cat = l1[2:7].strip().lstrip("0")
            if cat in cats or l0.strip().startswith("SENTINEL-2"):
                keep += [l0, l1, l2]
    if keep:
        (OUT / f"{ts}_{kind_of(orig)}.tle").write_text("\n".join(keep) + "\n")
        n_ok += 1
        print(f"{ts} {kind_of(orig)}: {len(keep) // 3} объектов", flush=True)
print(f"сохранено файлов: {n_ok}", flush=True)
