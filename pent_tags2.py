"""Pentagon tag reader v2 — local per-detection search + border-free letter crop.

Marker geometry (validated on Cityvet): a left-pointing pentagon containing one
letter (A/B/C/D/E), with neck-size + CFM printed to its right. The detection
center lands near the neck text, the pentagon sits lower-left of it.

Strategy per detection:
  1. crop a local window around the detection center (cheap, low-noise search)
  2. find the pentagon contour (5-gon convex, size-gated) nearest the center
  3. mask to the polygon, ERODE to drop the border strokes, isolate the dark
     interior strokes = the letter, tight-crop to them
  4. upscale + multi-threshold OCR with an allowlist of ONLY the valid tags
"""
from __future__ import annotations
import math, sys, json, os
import cv2, numpy as np


def _render(pdf, page_idx, dpi):
    import fitz
    doc = fitz.open(pdf); page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def _find_pentagons_local(gray, area_lo, area_hi, ar_lo=0.55, ar_hi=1.9):
    """Return [(approx_poly, (cx,cy), area)] for pentagon contours in a local crop."""
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)  # close small gaps in border
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_lo or a > area_hi:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 5 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if h and ar_lo <= w / float(h) <= ar_hi:
                    out.append((approx, (x + w / 2.0, y + h / 2.0), a))
                break
    return out


def _letter_crop_from_poly(gray, approx):
    """Mask to the pentagon, erode off the border, isolate the dark letter strokes,
    return a tight grayscale crop of just the glyph (or None)."""
    x, y, w, h = cv2.boundingRect(approx)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, approx.reshape(-1, 2), 255)
    er = max(2, int(min(w, h) * 0.18))               # pull inside the border
    mask = cv2.erode(mask, np.ones((er, er), np.uint8), iterations=1)
    dark = (gray < 160).astype(np.uint8) * 255
    letter = cv2.bitwise_and(dark, mask)
    ys, xs = np.where(letter > 0)
    if len(xs) < 8:
        return None
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    pad = 2
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(gray.shape[1] - 1, x1 + pad), min(gray.shape[0] - 1, y1 + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return gray[y0:y1 + 1, x0:x1 + 1]


def _ocr_letter(reader, crop_gray, allow):
    if crop_gray is None or crop_gray.size == 0:
        return None
    up = cv2.resize(crop_gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pad white border so glyph isn't flush to edge (helps the detector)
    otsu = cv2.copyMakeBorder(otsu, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    up_p = cv2.copyMakeBorder(up, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    best = None
    for im in (otsu, 255 - otsu, up_p):
        if im.shape[0] < 10 or im.shape[1] < 10:
            continue
        try:
            res = reader.readtext(im, detail=1, allowlist=allow,
                                  text_threshold=0.15, low_text=0.1, mag_ratio=2)
        except Exception:
            continue
        for _, txt, conf in res:
            t = txt.strip().upper()
            if len(t) == 1 and t in allow and (best is None or conf > best[1]):
                best = (t, conf)
    return best


def read_tags(img_bgr, det_centers, reader, allow='ABCDEF', dpi=300,
              win=None, area_lo=None, area_hi=None, min_conf=0.30):
    from collections import Counter
    H, W = img_bgr.shape[:2]
    sc = dpi / 300.0
    win = win or int(120 * sc)
    area_lo = area_lo or int(700 * sc * sc)
    area_hi = area_hi or int(6000 * sc * sc)
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    assignments, tagset = {}, Counter()
    dbg = []
    for di, (dcx, dcy) in enumerate(det_centers):
        x0, y0 = max(0, int(dcx - win)), max(0, int(dcy - win))
        x1, y1 = min(W, int(dcx + win)), min(H, int(dcy + win))
        gwin = gray_full[y0:y1, x0:x1]
        pents = _find_pentagons_local(gwin, area_lo, area_hi)
        # nearest pentagon to the (local) detection center
        lcx, lcy = dcx - x0, dcy - y0
        pents.sort(key=lambda p: math.hypot(p[1][0] - lcx, p[1][1] - lcy))
        got = None
        for approx, (pcx, pcy), area in pents[:3]:
            d = math.hypot(pcx - lcx, pcy - lcy)
            if d > win:
                continue
            crop = _letter_crop_from_poly(gwin, approx)
            r = _ocr_letter(reader, crop, allow)
            if r and r[1] >= min_conf:
                got = (r[0], r[1], d)
                break
        if got:
            assignments[di] = {'letter': got[0], 'conf': float(got[1]), 'dist': float(got[2])}
            tagset[got[0]] += 1
        dbg.append((di, got))
    return assignments, tagset, dbg


if __name__ == '__main__':
    import easyocr
    pdf, page_idx = sys.argv[1], int(sys.argv[2])
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    det_json = sys.argv[4]
    allow = sys.argv[5] if len(sys.argv) > 5 else 'ABCDEF'
    img = _render(pdf, page_idx, dpi)
    d = json.load(open(det_json, encoding='utf-8'))
    s = dpi / d.get('dpi', 200)
    centers = []
    for pg, lst in d['pages'].items():
        for x in lst:
            if x['cls'].startswith('AD-'):
                centers.append(((x['x1'] + x['x2']) / 2 * s, (x['y1'] + x['y2']) / 2 * s))
    reader = easyocr.Reader(['en'], gpu=False)
    asg, tagset, dbg = read_tags(img, centers, reader, allow=allow, dpi=dpi)
    print(f"grille detections: {len(centers)}  allow={allow}  dpi={dpi}")
    print(f"tagged {len(asg)}/{len(centers)}")
    print("letter distribution:", dict(sorted(tagset.items())))
