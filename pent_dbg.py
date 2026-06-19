"""Dump per-detection debug: local window w/ pentagon outline + the isolated letter crop + OCR."""
import sys, json, os, math
import cv2, numpy as np
import pent_tags2 as P

pdf, page_idx, dpi, det_json = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
allow = sys.argv[5] if len(sys.argv) > 5 else 'ABCD'
N = int(sys.argv[6]) if len(sys.argv) > 6 else 16
outdir = os.path.join(os.path.dirname(__file__), 'pent_dbg_out')
os.makedirs(outdir, exist_ok=True)
for fn in os.listdir(outdir):
    os.remove(os.path.join(outdir, fn))

import easyocr
img = P._render(pdf, page_idx, dpi)
H, W = img.shape[:2]
gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
d = json.load(open(det_json, encoding='utf-8'))
s = dpi / d.get('dpi', 200)
centers = []
for pg, lst in d['pages'].items():
    for x in lst:
        if x['cls'].startswith('AD-'):
            centers.append(((x['x1']+x['x2'])/2*s, (x['y1']+x['y2'])/2*s))
reader = easyocr.Reader(['en'], gpu=False)

sc = dpi/300.0
win = int(120*sc); area_lo = int(700*sc*sc); area_hi = int(6000*sc*sc)
for di, (dcx, dcy) in enumerate(centers[:N]):
    x0, y0 = max(0,int(dcx-win)), max(0,int(dcy-win))
    x1, y1 = min(W,int(dcx+win)), min(H,int(dcy+win))
    gwin = gray_full[y0:y1, x0:x1]
    vis = cv2.cvtColor(gwin, cv2.COLOR_GRAY2BGR)
    pents = P._find_pentagons_local(gwin, area_lo, area_hi)
    lcx, lcy = dcx-x0, dcy-y0
    cv2.drawMarker(vis, (int(lcx),int(lcy)), (0,0,255), cv2.MARKER_CROSS, 14, 2)
    pents.sort(key=lambda p: math.hypot(p[1][0]-lcx, p[1][1]-lcy))
    read = '_'
    for k,(approx,(pcx,pcy),area) in enumerate(pents[:3]):
        col = (0,200,0) if k==0 else (200,150,0)
        cv2.drawContours(vis, [approx], -1, col, 2)
        if k==0:
            crop = P._letter_crop_from_poly(gwin, approx)
            r = P._ocr_letter(reader, crop, allow)
            if r: read = f"{r[0]}{r[1]:.2f}"
            if crop is not None and crop.size:
                up = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(outdir, f"det{di:02d}_letter_{read}.png"), up)
    cv2.imwrite(os.path.join(outdir, f"det{di:02d}_win_read={read}_npent={len(pents)}.png"), vis)
    print(f"det{di:02d}: pents={len(pents)} read={read}")
print("saved to", outdir)
