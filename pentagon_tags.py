"""Pentagon tag-marker reader for air-distribution devices.

On many mechanical plans the air-device tag is a single letter drawn inside a
small PENTAGON (house-shaped) marker, with neck size + CFM beside it. The trained
tag-bubble detector (circles/ovals) misses these, and a wide-window OCR can't read
the glyph because it's tiny inside the larger neighbourhood. But a *tight* crop of
just the pentagon interior, upscaled + Otsu-binarized, OCRs the letter reliably
(validated: 'B' at conf 1.0 on Cityvet).

Design (important): this runs ONLY on the mechanical plan pages, and only OCRs
pentagons that sit near a YOLO grille detection. So it never scans non-mech pages,
makes ~one OCR call per grille (not per page-contour), and can't pick up stray
pentagons elsewhere on the sheet.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


# ---- letter OCR on a tight pentagon-interior crop -------------------------

def _ocr_letter(reader, crop_bgr, allow='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 3 or crop_bgr.shape[1] < 3:
        return None
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(g, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = None
    for im in (otsu, up):
        if im.shape[0] < 8 or im.shape[1] < 8:
            continue
        try:
            res = reader.readtext(im, detail=1, allowlist=allow,
                                  text_threshold=0.2, low_text=0.15, mag_ratio=2)
        except Exception:
            continue
        for _, txt, conf in res:
            t = txt.strip().upper()
            if len(t) == 1 and t.isalpha() and (best is None or conf > best[1]):
                best = (t, conf)
    return best  # (letter, conf) or None


# ---- pentagon localisation (no OCR — fast) --------------------------------

def find_pentagon_contours(img_bgr, min_area=120, max_area=6000, ar_lo=0.6, ar_hi=1.7):
    """Return [{cx, cy, x, y, w, h}] for every pentagon-shaped contour. No OCR."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) != 5 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0 or not (ar_lo <= w / float(h) <= ar_hi):
            continue
        out.append({'cx': x + w / 2.0, 'cy': y + h / 2.0, 'x': x, 'y': y, 'w': w, 'h': h})
    return out


# ---- per-detection tag reading (the integration entry point) --------------

def read_tags_for_detections(img_bgr, det_centers, reader,
                             max_dist=70, pad=2, min_conf=0.30,
                             min_area=120, max_area=6000):
    """For each detection center (in img pixel coords), find the nearest pentagon
    marker within ``max_dist`` and OCR its letter.

    Returns (assignments, discovered_tagset):
      assignments      = {det_index: {'letter','conf','dist'}}
      discovered_tagset = Counter of confidently-read letters
    Only pentagons near a detection get OCR'd, so cost ~= number of detections.
    """
    from collections import Counter
    pentagons = find_pentagon_contours(img_bgr, min_area=min_area, max_area=max_area)
    H, W = img_bgr.shape[:2]
    assignments = {}
    tagset = Counter()
    ocr_cache = {}  # contour idx -> (letter, conf) | None
    for di, (dcx, dcy) in enumerate(det_centers):
        # candidate pentagons within max_dist, nearest first
        cands = []
        for pi, p in enumerate(pentagons):
            d = math.hypot(p['cx'] - dcx, p['cy'] - dcy)
            if d <= max_dist:
                cands.append((d, pi))
        cands.sort()
        for d, pi in cands:
            if pi not in ocr_cache:
                p = pentagons[pi]
                x0, y0 = max(0, p['x'] + pad), max(0, p['y'] + pad)
                x1, y1 = min(W, p['x'] + p['w'] - pad), min(H, p['y'] + p['h'] - pad)
                ocr_cache[pi] = _ocr_letter(reader, img_bgr[y0:y1, x0:x1]) if (x1 - x0 >= 4 and y1 - y0 >= 4) else None
            got = ocr_cache[pi]
            if got and got[1] >= min_conf:
                assignments[di] = {'letter': got[0], 'conf': float(got[1]), 'dist': float(d)}
                tagset[got[0]] += 1
                break
    return assignments, tagset


if __name__ == '__main__':
    import sys, json
    from collections import Counter
    import easyocr, fitz
    pdf, page_idx = sys.argv[1], int(sys.argv[2])
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    det_json = sys.argv[4] if len(sys.argv) > 4 else None
    doc = fitz.open(pdf); page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    reader = easyocr.Reader(['en'], gpu=False)
    centers = []
    if det_json:
        d = json.load(open(det_json, encoding='utf-8'))
        s = dpi / d.get('dpi', 200)
        for pg, dets in d['pages'].items():
            for x in dets:
                if x['cls'].startswith('AD-'):
                    centers.append(((x['x1'] + x['x2']) / 2 * s, (x['y1'] + x['y2']) / 2 * s))
    print(f'grille detections: {len(centers)}  (dpi={dpi})')
    asg, tagset = read_tags_for_detections(img, centers, reader, max_dist=int(70 * dpi / 200))
    print(f'tagged {len(asg)}/{len(centers)} grilles')
    print('letter distribution:', dict(sorted(tagset.items())))
