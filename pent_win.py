"""Window-OCR tag reader: no pentagon detection. The neck-size/CFM are digits,
the tag is a letter — so OCR the local window with a LETTERS-ONLY allowlist and
take the single-letter box nearest the detection center."""
import sys, json, math
import cv2, numpy as np
import pent_tags2 as P


def read_window(gray_full, dcx, dcy, reader, allow='ABCDEF', dpi=300, min_conf=0.40):
    H, W = gray_full.shape[:2]
    sc = dpi / 300.0
    win = int(95 * sc)
    x0, y0 = max(0, int(dcx - win)), max(0, int(dcy - win))
    x1, y1 = min(W, int(dcx + win)), min(H, int(dcy + win))
    g = gray_full[y0:y1, x0:x1]
    if g.size == 0:
        return None
    up = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    cands = []
    for im in (up,):
        try:
            res = reader.readtext(im, detail=1, allowlist=allow,
                                  text_threshold=0.25, low_text=0.2, mag_ratio=2)
        except Exception:
            continue
        for box, txt, conf in res:
            t = txt.strip().upper()
            if len(t) == 1 and t in allow and conf >= min_conf:
                bx = np.array(box); bcx = bx[:, 0].mean() / 4 + x0; bcy = bx[:, 1].mean() / 4 + y0
                d = math.hypot(bcx - dcx, bcy - dcy)
                cands.append((d, t, conf))
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2], cands[0][0]


if __name__ == '__main__':
    import easyocr
    from collections import Counter
    pdf, page_idx, dpi, det_json = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    allow = sys.argv[5] if len(sys.argv) > 5 else 'ABCDEF'
    img = P._render(pdf, page_idx, dpi)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = json.load(open(det_json, encoding='utf-8'))
    s = dpi / d.get('dpi', 200)
    centers = []
    for pg, lst in d['pages'].items():
        for x in lst:
            if x['cls'].startswith('AD-'):
                centers.append(((x['x1']+x['x2'])/2*s, (x['y1']+x['y2'])/2*s))
    reader = easyocr.Reader(['en'], gpu=False)
    tagset = Counter(); tagged = 0
    for (dcx, dcy) in centers:
        r = read_window(gray, dcx, dcy, reader, allow=allow, dpi=dpi)
        if r:
            tagged += 1; tagset[r[0]] += 1
    print(f"grille detections: {len(centers)}  allow={allow}  dpi={dpi}")
    print(f"tagged {tagged}/{len(centers)}")
    print("letter distribution:", dict(sorted(tagset.items())))
