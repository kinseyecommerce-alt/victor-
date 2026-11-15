"""Download NSE Indices press releases (ind_prsDDMMYYYY*.pdf, 2011 -> today) listed in the Internet Archive CDX
index, from niftyindices.com (fallback: the archived copy), convert to text with pdftotext, keep the text
(gz) for all and the PDF only when the text concerns index replacements.  1 request / 2 s.  Resume-safe.
Usage: python3 -I fetch_press.py CDX_TXT [CDX_TXT ...] OUTDIR"""
import gzip, os, re, subprocess, sys, time, urllib.request, urllib.error
OUT = sys.argv[-1]
UA = "Mozilla/5.0"
os.makedirs(os.path.join(OUT, "txt"), exist_ok=True)
os.makedirs(os.path.join(OUT, "pdf"), exist_ok=True)
KEEP = re.compile(r"replacement|being excluded|being included|will be excluded|will be included|reconstitution|"
                  r"changes? (in|to) (the )?(indices|index)", re.I)

items = {}
for cdx in sys.argv[1:-1]:
    for line in open(cdx):
        p = line.split()
        if len(p) < 3 or p[2] != "200":
            continue
        m = re.search(r"/(ind_prs(\d{2})(\d{2})(\d{4})[^/?]*?\.pdf)", p[1], re.I)
        if not m or int(m.group(4)) < 2011:
            continue
        name = m.group(1).lower()
        items.setdefault(name, (p[0], p[1], m.group(1)))


def get(url):
    for attempt in range(4):
        time.sleep(2.0)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
                return None
            print("HTTP", e.code, url[-60:], flush=True); time.sleep(15 * 2 ** attempt)
        except Exception as e:
            print("ERR", repr(e)[:80], flush=True); time.sleep(15 * 2 ** attempt)
    return None


print("press releases to consider:", len(items), flush=True)
for name in sorted(items, key=lambda n: (n[11:15], n[9:11], n[7:9], n)):
    ts, orig, real = items[name]
    tfn = os.path.join(OUT, "txt", name[:-4] + ".txt.gz")
    if os.path.exists(tfn):
        continue
    src = "niftyindices"
    data = get("https://www.niftyindices.com/Press_Release/" + real)
    if (not data or not data.startswith(b"%PDF")) and os.environ.get("NO_WAYBACK"):
        print(name, "not on niftyindices (wayback pass later)", flush=True); continue
    if not data or not data.startswith(b"%PDF"):
        src = "wayback"
        time.sleep(8)
        data = get(f"https://web.archive.org/web/{ts}id_/{orig.split('?')[0]}")
    if not data or not data.startswith(b"%PDF"):
        print(name, "FAILED", flush=True); continue
    pfn = os.path.join(OUT, "pdf", name)
    open(pfn, "wb").write(data)
    try:
        txt = subprocess.run(["pdftotext", "-layout", pfn, "-"], capture_output=True, timeout=120).stdout
    except Exception as e:
        txt = b""
    with gzip.open(tfn + ".tmp", "wb") as fh:
        fh.write(txt)
    os.replace(tfn + ".tmp", tfn)
    keep = bool(KEEP.search(txt.decode("utf8", "replace")))
    if not keep:
        os.remove(pfn)
    print(name, src, len(data), len(txt), "keep" if keep else "", flush=True)
print("DONE", flush=True)
