"""legend_match.py — template-match legend symbol graphics across plan pages.

EXPERIMENTAL / precision-biased. Template matching from legend crops is known
to be noisy (ductwork + ceiling-grid false positives — see CLAUDE.md §10
"What Failed #1"). So this is deliberately conservative:

  • crop ONLY the graphic to the left of an equipment-symbol label
  • drop templates that are tiny, near-blank, or line-like (linetypes →
    false-positive machines)
  • match multi-scale across PLAN pages only, high correlation threshold
  • per-page NMS + per-template caps
  • every hit is a CANDIDATE for human confirmation, never an auto-detection

Returns candidates the UI can overlay (flagged from_legend) so the estimator
confirms them — which then feeds the same correction/training loop.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import fitz  # PyMuPDF

from legend_reader import find_legend_page, normalize_label_to_class


def _render_gray(doc, pidx: int, dpi: int) -> np.ndarray:
    pix = doc[pidx].get_pixmap(dpi=dpi, annots=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(img[:, :, 0])


def _trim_to_ink(gray: np.ndarray, pad: int = 3):
    """Crop to the bounding box of dark ink. Returns None if effectively blank."""
    ink = gray < 200
    if ink.sum() < 20:
        return None
    ys, xs = np.where(ink)
    y1, y2 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad + 1)
    return gray[y1:y2, x1:x2]


def get_symbol_templates(doc, legend_pidx: int, dpi: int) -> list[dict]:
    """Crop the graphic to the LEFT of each equipment-symbol label on the
    legend page. One template per class (the first usable one)."""
    page = doc[legend_pidx]
    words = page.get_text('words')  # (x0,y0,x1,y1, text, block, line, wordno) in points
    scale = dpi / 72.0
    gray = _render_gray(doc, legend_pidx, dpi)
    H, W = gray.shape

    lines = defaultdict(list)
    for w in words:
        lines[(w[5], w[6])].append(w)

    out: list[dict] = []
    seen_class: set[str] = set()
    for ws in lines.values():
        ws.sort(key=lambda w: w[0])
        text = ' '.join(w[4] for w in ws).strip()
        cls = normalize_label_to_class(text)
        if not cls or cls in seen_class:
            continue
        # Symbol region: a band the height of the row, ending just left of the label.
        lx = ws[0][0]
        y0 = min(w[1] for w in ws)
        y1 = max(w[3] for w in ws)
        row_h = (y1 - y0)
        sym_w = max(row_h * 3.5, 24 / scale)  # ~3.5× row height of graphic
        cx1 = int((lx - sym_w) * scale)
        cx2 = int((lx - 2) * scale)
        cy1 = int((y0 - row_h * 0.2) * scale)
        cy2 = int((y1 + row_h * 0.2) * scale)
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(W, cx2), min(H, cy2)
        if cx2 - cx1 < 16 or cy2 - cy1 < 12:
            continue
        crop = _trim_to_ink(gray[cy1:cy2, cx1:cx2])
        if crop is None:
            continue
        h, w = crop.shape
        # Guards against false-positive-prone templates:
        if h < 14 or w < 14:          # too small
            continue
        if crop.std() < 14:           # near-uniform (blank)
            continue
        if h < 10 or w / max(1, h) > 6 or h / max(1, w) > 6:  # line-like
            continue
        out.append({'class': cls, 'label': text[:50], 'img': crop})
        seen_class.add(cls)
    return out


def _nms(boxes: list[dict], iou_thr: float = 0.3) -> list[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -b['score'])
    keep = []
    for b in boxes:
        ok = True
        for k in keep:
            ix1, iy1 = max(b['x1'], k['x1']), max(b['y1'], k['y1'])
            ix2, iy2 = min(b['x2'], k['x2']), min(b['y2'], k['y2'])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            ua = ((b['x2'] - b['x1']) * (b['y2'] - b['y1'])
                  + (k['x2'] - k['x1']) * (k['y2'] - k['y1']) - inter)
            if ua > 0 and inter / ua > iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def match_legend_symbols(pdf_path, classifications=None, conf: float = 0.66,
                         match_dpi: int = 120, out_dpi: int = 200,
                         per_template_cap: int = 60) -> dict:
    """Crop legend symbols and template-match them across plan pages.

    Returns {legend_page, templates:[{class,label,w,h}], dpi:out_dpi,
             pages:{<0-based page>: [{x1,y1,x2,y2,class,label,score}]},
             total, note}.  Boxes are in out_dpi pixel space (matches detections).
    """
    legend_pidx = find_legend_page(pdf_path, classifications=classifications)
    if legend_pidx is None:
        return {'legend_page': None, 'templates': [], 'pages': {}, 'total': 0,
                'note': 'no legend page found'}

    # Plan pages only (1-based 'page' in classifications → 0-based).
    plan_pages = []
    for c in (classifications or []):
        if (c.get('is_plan') if isinstance(c, dict) else getattr(c, 'is_plan', False)):
            p = c.get('page') if isinstance(c, dict) else getattr(c, 'page', None)
            if p:
                plan_pages.append(p - 1)

    doc = fitz.open(str(pdf_path))
    try:
        if not plan_pages:
            plan_pages = [p for p in range(doc.page_count) if p != legend_pidx]

        templates = get_symbol_templates(doc, legend_pidx, match_dpi)
        scale_out = out_dpi / match_dpi
        pages_out: dict[str, list[dict]] = {}
        for pidx in plan_pages:
            page_gray = _render_gray(doc, pidx, match_dpi)
            PH, PW = page_gray.shape
            hits: list[dict] = []
            for t in templates:
                tmpl0 = t['img']
                per_t: list[dict] = []
                for s in (0.8, 1.0, 1.25, 1.5):
                    th = int(tmpl0.shape[0] * s)
                    tw = int(tmpl0.shape[1] * s)
                    if th < 12 or tw < 12 or th >= PH or tw >= PW:
                        continue
                    tmpl = cv2.resize(tmpl0, (tw, th), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(page_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    ys, xs = np.where(res >= conf)
                    for (y, x) in zip(ys.tolist(), xs.tolist()):
                        per_t.append({
                            'x1': x, 'y1': y, 'x2': x + tw, 'y2': y + th,
                            'score': float(res[y, x]),
                            'class': t['class'], 'label': t['label'],
                        })
                per_t = _nms(per_t, 0.3)[:per_template_cap]
                hits.extend(per_t)
            hits = _nms(hits, 0.4)
            if hits:
                pages_out[str(pidx)] = [{
                    'x1': round(h['x1'] * scale_out), 'y1': round(h['y1'] * scale_out),
                    'x2': round(h['x2'] * scale_out), 'y2': round(h['y2'] * scale_out),
                    'class': h['class'], 'label': h['label'], 'score': round(h['score'], 3),
                } for h in hits]
    finally:
        doc.close()

    total = sum(len(v) for v in pages_out.values())
    return {
        'legend_page': legend_pidx + 1,
        'dpi': out_dpi,
        'templates': [{'class': t['class'], 'label': t['label'],
                       'h': t['img'].shape[0], 'w': t['img'].shape[1]} for t in templates],
        'pages': pages_out,
        'total': total,
        'note': 'EXPERIMENTAL — candidates for human confirmation, not trusted detections',
    }


def save_symbol_crops(pdf_path, classifications=None, out_dir=None, dpi: int = 200) -> dict:
    """Crop each equipment-symbol graphic from the legend and save it as a PNG
    (for DISPLAY next to its label in the UI — not matching). Returns
    {class: crop_filename}. This is the honest, false-positive-free use of the
    legend symbols: show the estimator what each symbol looks like on THIS set.
    """
    legend_pidx = find_legend_page(pdf_path, classifications=classifications)
    if legend_pidx is None or out_dir is None:
        return {}
    doc = fitz.open(str(pdf_path))
    try:
        templates = get_symbol_templates(doc, legend_pidx, dpi)
    finally:
        doc.close()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops: dict[str, str] = {}
    for t in templates:
        slug = re.sub(r'[^A-Za-z0-9]+', '_', t['class']).strip('_')[:40]
        fn = f'sym_{slug}.png'
        if cv2.imwrite(str(out / fn), t['img']):
            crops[t['class']] = fn
    return crops


if __name__ == '__main__':  # pragma: no cover
    import sys, json
    cls = None
    if len(sys.argv) > 2:
        cls = json.loads(open(sys.argv[2], encoding='utf-8').read())
    r = match_legend_symbols(sys.argv[1], classifications=cls)
    print('legend page:', r['legend_page'])
    print('templates:', [(t['class'], f"{t['w']}x{t['h']}") for t in r['templates']])
    print('total candidate matches:', r['total'])
    for p, hits in r['pages'].items():
        from collections import Counter
        c = Counter(h['class'] for h in hits)
        print(f'  page {int(p)+1}: {len(hits)} — {dict(c)}')
