"""Diagnostic: render Cityvet page 3, inspect grille neighborhoods + pentagon contours."""
import sys, json, os
import cv2, numpy as np, fitz

pdf = sys.argv[1]; page_idx = int(sys.argv[2]); dpi = int(sys.argv[3]); det_json = sys.argv[4]
outdir = os.path.join(os.path.dirname(__file__), 'pent_diag_out')
os.makedirs(outdir, exist_ok=True)

doc = fitz.open(pdf); page = doc[page_idx]
pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
H, W = img.shape[:2]
print(f"page {page_idx}  img {W}x{H}  dpi={dpi}")

d = json.load(open(det_json, encoding='utf-8'))
s = dpi / d.get('dpi', 200)
dets = []
for pg, lst in d['pages'].items():
    for x in lst:
        if x['cls'].startswith('AD-'):
            dets.append((x['cls'], (x['x1']+x['x2'])/2*s, (x['y1']+x['y2'])/2*s,
                         (x['x2']-x['x1'])*s, (x['y2']-x['y1'])*s))
print(f"AD detections: {len(dets)}")

# pentagon contours over full page, report area histogram
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
areas = []
for c in cnts:
    a = cv2.contourArea(c)
    if a < 50:
        continue
    approx = cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True), True)
    if len(approx) == 5 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        if h and 0.5 <= w/float(h) <= 2.0:
            areas.append(a)
areas.sort()
print(f"5-gon convex contours (area>=50, ar 0.5-2.0): {len(areas)}")
if areas:
    import statistics
    print(f"  area min={areas[0]:.0f} p25={areas[len(areas)//4]:.0f} med={statistics.median(areas):.0f} p75={areas[3*len(areas)//4]:.0f} max={areas[-1]:.0f}")

# save neighborhood crops for first 8 detections (±70px window)
R = int(70 * dpi/200)
for i, (cls, cx, cy, bw_, bh) in enumerate(dets[:8]):
    x0, y0 = max(0,int(cx-R)), max(0,int(cy-R))
    x1, y1 = min(W,int(cx+R)), min(H,int(cy+R))
    crop = img[y0:y1, x0:x1].copy()
    # mark detection center
    cv2.circle(crop, (int(cx-x0), int(cy-y0)), 4, (0,0,255), -1)
    cv2.imwrite(os.path.join(outdir, f"det{i:02d}_{cls.replace(' ','_')}.png"), crop)
print(f"saved {min(8,len(dets))} crops to {outdir}")
