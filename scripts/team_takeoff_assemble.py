"""Assemble a TEAM-FORMAT takeoff: join schedule props (by tag) with the plan
neck-size/CFM stream, group by (product, tag, neck size), write the team's columns.
Runs in ppenv (PaddleOCR + fitz + openpyxl).
"""
import os, json, glob, re, sys
os.environ["FLAGS_use_mkldnn"] = "0"
import fitz, numpy as np
from collections import defaultdict
from paddleocr import PaddleOCR
import openpyxl

PROJ_DIR = sys.argv[1]
OUT_XLSX = sys.argv[2]

dj = glob.glob(os.path.join(PROJ_DIR, "*_detections.json"))[0]
vj = glob.glob(os.path.join(PROJ_DIR, "*_variables.json"))[0]
d = json.load(open(dj, encoding="utf-8"))
variables = json.load(open(vj, encoding="utf-8"))
dpi = d["dpi"]; scale = dpi / 72.0

# tag -> schedule props
by_tag = {}
for v in variables:
    if v.get("tag"):
        by_tag[v["tag"].upper()] = v.get("properties", {}) or {}

def prop(props, *keys):
    for k in keys:
        for pk, pv in props.items():
            if k in pk.upper() and pv and str(pv).strip() not in (".", "-"):
                return str(pv).strip()
    return ""

BRANDS = {"TITUS","PRICE","KRUEGER","NAILOR","METALAIRE","METAL-AIRE","CARNES","TUTTLE",
          "TRANE","CARRIER","GREENHECK","BROAN","RUSKIN","POTTORFF","COOK","DAIKIN",
          "MITSUBISHI","LG","LENNOX","YORK","AAON","REZNOR","TUTTLE & BAILEY","SEIHO"}

def split_brand(raw):
    if not raw: return "", ""
    toks = raw.split()
    if toks and toks[0].upper().strip(".,") in BRANDS:
        return toks[0], " ".join(toks[1:])
    return "", raw

def _ok(n): return 3 <= n <= 24   # plausible neck dim; ducts (>24) excluded
def norm_size(s):
    s = s.replace("×","x").replace("X","x").replace(" ","").replace('"','').replace("'","").replace("”","")
    m = re.match(r'^(\d{1,2})x(\d{1,2})$', s)
    if m and _ok(int(m.group(1))) and _ok(int(m.group(2))): return f'{int(m.group(1))}X{int(m.group(2))}'
    m = re.match(r'^(\d{1,2})[0OoøØ]$', s)
    if m and _ok(int(m.group(1))): return f'{int(m.group(1))}"'
    m = re.match(r'^(\d{1,2})$', s)
    if m and _ok(int(m.group(1))): return f'{int(m.group(1))}"'
    return None
CFM_RE = re.compile(r'(\d{2,4})\s*CFM', re.I)
TYP_RE = re.compile(r'TYP\.?\s*(\d+)', re.I)

print("loading PaddleOCR...", flush=True)
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, enable_mkldnn=False)
doc = fitz.open(d["pdf"])

rows = []
for pno, dets in d["pages"].items():
    page = doc[int(pno)]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    H, W = img.shape[:2]
    for det in dets:
        x1,y1,x2,y2 = det["x1"],det["y1"],det["x2"],det["y2"]
        cx,cy=(x1+x2)/2,(y1+y2)/2; bw,bh=x2-x1,y2-y1; max_dist=max(bw,bh)*1.2; pad=80
        crop = img[int(max(0,y1-pad)):int(min(H,y2+pad)), int(max(0,x1-pad)):int(min(W,x2+pad))]
        ox,oy = max(0,x1-pad), max(0,y1-pad)
        sizes,cfms,typ=[],[],1
        for r in ocr.predict(crop):
            for t,poly in zip(r.get("rec_texts",[]), r.get("rec_polys",r.get("dt_polys",[]))):
                p=np.array(poly); tx,ty=p[:,0].mean()+ox,p[:,1].mean()+oy
                dist=((tx-cx)**2+(ty-cy)**2)**0.5
                if dist>max_dist: continue
                ns=norm_size(t.strip())
                if ns: sizes.append((dist,ns))
                mc=CFM_RE.search(t);
                if mc: cfms.append((dist,mc.group(1)))
                mt=TYP_RE.search(t)
                if mt: typ=max(typ,int(mt.group(1)))
        sizes.sort(); cfms.sort()
        rows.append({"cls":det["cls"],"tag":(det.get("tag") or ""),
                     "neck":sizes[0][1] if sizes else "","cfm":cfms[0][1] if cfms else "","typ":typ})

# group by (product, tag, neck) -> qty + carry schedule props by tag
g = defaultdict(lambda: {"qty":0,"cfm":""})
for r in rows:
    k=(r["cls"], r["tag"], r["neck"])
    g[k]["qty"] += r["typ"]
    if r["cfm"]: g[k]["cfm"]=r["cfm"]

wb=openpyxl.Workbook(); ws=wb.active; ws.title="Triune Takeoff"
HEADERS=["PRODUCT","BRAND","MODEL","QTY","TAG","NECK SIZE","MODULE SIZE","DUCT SIZE","CFM","TYPE","MOUNTING","ACCESSORIES","REMARK"]
ws.append(HEADERS)
print("\n"+" | ".join(HEADERS))
for (cls,tag,neck),info in sorted(g.items()):
    props=by_tag.get(tag.upper(),{})
    brand=prop(props,"MANUFACTURER","BRAND","BASES OF DESIGN")
    raw_model=prop(props,"MFR. & MODEL","MAKE / MODEL","MFR & MODEL","MAKE/MODEL","MODEL")
    if not brand:
        brand, raw_model = split_brand(raw_model)
    model=raw_model
    mod_size=prop(props,"MODULE","FACE SIZE","NOMINAL SIZE","PANEL SIZE")
    typ_=prop(props,"TYPE","SERVICE","STYLE","DESCRIPTION")
    mount=prop(props,"MOUNTING","MOUNT","INSTALLATION")
    acc=prop(props,"ACCESSOR","DAMPER","OBD","OPTION","MATERIAL","REMARK")
    row=[cls,brand,model,info["qty"],tag,neck,mod_size,"",info["cfm"],typ_,mount,acc,"SEE SCHEDULE" if props else ""]
    ws.append(row)
    print(" | ".join(str(c) for c in row))
wb.save(OUT_XLSX)
print("\nsaved", OUT_XLSX)
