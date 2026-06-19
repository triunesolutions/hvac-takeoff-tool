"""
template_matcher.py — match legend-extracted symbols against floor plans.

Given a folder of symbol PNG templates (from legend_reader) and a plan
page rendered as an image, returns candidate locations where each
template appears. The intent is to add per-project adaptation: even
when YOLO doesn't recognize a symbol style, if it appears in the legend
we can find its occurrences via classical template matching.

Approach:
  1. Load each template image, preprocess (grayscale + binary).
  2. Multi-scale slide: templates in legends are usually drawn ~10-20×
     larger than instances on the plan, so we resize and try several scales.
  3. Use OpenCV's matchTemplate with TM_CCOEFF_NORMED → similarity map.
  4. Threshold to extract candidate centers; apply per-class non-max
     suppression to dedupe overlapping detections.
  5. Reconcile with YOLO detections: confirm where they agree, flag
     YOLO-missed-by-template and template-missed-by-YOLO separately.

This is intentionally classical / non-ML: template matching is a
solved problem in OpenCV, and we want predictable behavior, not
another model to train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _try_imports():
    try:
        import cv2
        import numpy as np
        return cv2, np
    except ImportError:
        return None, None


def preprocess_template(template_img):
    """Convert template to grayscale + binary for robust matching."""
    cv2, np = _try_imports()
    if cv2 is None or template_img is None or template_img.size == 0:
        return None
    if len(template_img.shape) == 3:
        gray = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = template_img
    # Otsu binarization keeps the symbol shape, strips JPEG/raster noise
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def crop_to_symbol(template_binary, padding: int = 3):
    """Crop the template tightly around the non-white pixels."""
    cv2, np = _try_imports()
    if template_binary is None or template_binary.size == 0:
        return template_binary

    # Find foreground pixels (black-on-white symbol)
    coords = cv2.findNonZero(cv2.bitwise_not(template_binary))
    if coords is None:
        return template_binary
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = template_binary.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return template_binary
    return template_binary[y1:y2, x1:x2]


def match_template_multi_scale(plan_img_gray, template_binary,
                              scales=(0.15, 0.20, 0.30, 0.50),
                              threshold: float = 0.65):
    """Run template matching at multiple scales. Returns list of
    {x, y, w, h, score, scale} candidates."""
    cv2, np = _try_imports()
    if cv2 is None or template_binary is None or plan_img_gray is None:
        return []

    h_p, w_p = plan_img_gray.shape[:2]
    h_t, w_t = template_binary.shape[:2]
    if h_t < 5 or w_t < 5:
        return []

    candidates = []
    for scale in scales:
        new_w = max(8, int(w_t * scale))
        new_h = max(8, int(h_t * scale))
        if new_w >= w_p or new_h >= h_p:
            continue
        try:
            resized = cv2.resize(template_binary, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        except Exception:
            continue

        # Match
        try:
            result = cv2.matchTemplate(plan_img_gray, resized,
                                      cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue

        # Find peaks above threshold
        loc_y, loc_x = np.where(result >= threshold)
        for y, x in zip(loc_y, loc_x):
            candidates.append({
                'x': int(x),
                'y': int(y),
                'w': new_w,
                'h': new_h,
                'cx': int(x + new_w / 2),
                'cy': int(y + new_h / 2),
                'score': float(result[y, x]),
                'scale': scale,
            })

    return candidates


def nms(candidates, iou_threshold: float = 0.4):
    """Greedy non-max suppression on candidates. Highest-score wins."""
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda c: -c['score'])
    kept = []
    while sorted_c:
        head = sorted_c.pop(0)
        kept.append(head)
        sorted_c = [c for c in sorted_c
                   if _iou(head, c) < iou_threshold]
    return kept


def _iou(a, b) -> float:
    ax1, ay1 = a['x'], a['y']
    ax2, ay2 = ax1 + a['w'], ay1 + a['h']
    bx1, by1 = b['x'], b['y']
    bx2, by2 = bx1 + b['w'], by1 + b['h']
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = a['w'] * a['h']
    b_area = b['w'] * b['h']
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0


def match_legend_on_plan(legend_dict_path: Path,
                         legend_dir: Path,
                         plan_image_rgb,
                         threshold: float = 0.65) -> dict:
    """For each templated legend entry, find candidate positions on the plan.

    Args:
        legend_dict_path: path to legend_reader output JSON
        legend_dir: directory containing the symbol PNG crops
        plan_image_rgb: numpy RGB image of the floor plan
        threshold: similarity floor (0..1)

    Returns:
        {entries: [{label, normalized_class, matches: [{cx, cy, score, ...}]}],
         total_matches: int}
    """
    cv2, np = _try_imports()
    if cv2 is None:
        return {'entries': [], 'reason': 'opencv unavailable'}

    legend = json.loads(Path(legend_dict_path).read_text(encoding='utf-8'))
    legend_dir = Path(legend_dir)

    # Plan to grayscale
    if len(plan_image_rgb.shape) == 3:
        plan_gray = cv2.cvtColor(plan_image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        plan_gray = plan_image_rgb
    _, plan_binary = cv2.threshold(plan_gray, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    out_entries = []
    total = 0

    for entry in legend.get('entries', []):
        img_name = entry.get('symbol_image')
        if not img_name:
            continue
        img_path = legend_dir / img_name
        if not img_path.exists():
            continue

        tmpl = cv2.imread(str(img_path))
        if tmpl is None:
            continue

        tmpl_bin = preprocess_template(tmpl)
        tmpl_crop = crop_to_symbol(tmpl_bin) if tmpl_bin is not None else None
        if tmpl_crop is None or tmpl_crop.size == 0:
            continue

        # Skip degenerate (mostly white) templates — they match everywhere
        if (tmpl_crop > 0).mean() > 0.95:
            continue

        cands = match_template_multi_scale(plan_binary, tmpl_crop,
                                          threshold=threshold)
        kept = nms(cands)

        if not kept:
            continue
        total += len(kept)
        out_entries.append({
            'label': entry.get('label'),
            'normalized_class': entry.get('normalized_class'),
            'symbol_image': img_name,
            'n_matches': len(kept),
            'matches': [
                {'cx': c['cx'], 'cy': c['cy'],
                 'w': c['w'], 'h': c['h'],
                 'score': round(c['score'], 3),
                 'scale': c['scale']}
                for c in kept[:200]  # cap per entry
            ],
        })

    return {
        'entries': out_entries,
        'total_matches': total,
        'plan_shape': list(plan_image_rgb.shape),
    }


def reconcile_with_yolo(template_matches: dict,
                        yolo_detections_on_page: list,
                        plan_image_shape_300dpi: tuple,
                        plan_image_shape_200dpi: tuple,
                        iou_threshold: float = 0.3) -> dict:
    """Compare template-matcher results against YOLO. Categorize:
       - both: agreed on (high confidence)
       - template_only: template found, YOLO missed (potential YOLO recall gain)
       - yolo_only: YOLO found, template didn't (existing YOLO result)

    Coordinates: template matches are at 300 DPI; YOLO detections at 200 DPI.
    Scale YOLO bboxes to 300 DPI for comparison.
    """
    yolo_scaled = []
    scale = 1.5  # 200 → 300
    for det in yolo_detections_on_page:
        yolo_scaled.append({
            'x': det['x1'] * scale,
            'y': det['y1'] * scale,
            'w': (det['x2'] - det['x1']) * scale,
            'h': (det['y2'] - det['y1']) * scale,
            'cls': det.get('cls'),
            'tag': det.get('tag'),
        })

    both = 0
    template_only = 0
    yolo_matched = set()

    # For each template match, check if any YOLO det covers it
    for entry in template_matches.get('entries', []):
        for m in entry['matches']:
            m_box = {'x': m['cx'] - m['w'] / 2,
                    'y': m['cy'] - m['h'] / 2,
                    'w': m['w'], 'h': m['h']}
            matched_yolo = None
            for yi, yd in enumerate(yolo_scaled):
                if _iou(m_box, yd) >= iou_threshold:
                    matched_yolo = yi
                    break
            if matched_yolo is not None:
                both += 1
                yolo_matched.add(matched_yolo)
            else:
                template_only += 1

    yolo_only = len(yolo_scaled) - len(yolo_matched)

    return {
        'both': both,
        'template_only': template_only,
        'yolo_only': yolo_only,
        'total_yolo': len(yolo_scaled),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    cv2, np = _try_imports()

    ap = argparse.ArgumentParser()
    ap.add_argument('--legend-dict', required=True,
                    help='Path to legend_reader output JSON')
    ap.add_argument('--legend-dir', required=True,
                    help='Folder containing the symbol PNG crops')
    ap.add_argument('--plan-image', required=True,
                    help='Path to a plan page image (PNG/JPG)')
    ap.add_argument('--threshold', type=float, default=0.65)
    ap.add_argument('--out')
    args = ap.parse_args()

    if cv2 is None:
        print('opencv-python required')
        raise SystemExit(1)

    plan = cv2.imread(args.plan_image)
    if plan is None:
        print(f'failed to load {args.plan_image}')
        raise SystemExit(1)
    plan_rgb = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)

    result = match_legend_on_plan(
        Path(args.legend_dict),
        Path(args.legend_dir),
        plan_rgb,
        threshold=args.threshold,
    )

    print(f'Plan shape: {plan_rgb.shape}')
    print(f'Total template matches: {result.get("total_matches", 0)}')
    print()
    for e in result.get('entries', [])[:20]:
        print(f"  {e.get('normalized_class') or e.get('label')[:30]} "
              f"({e['n_matches']} hits)")
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=2, default=str), encoding='utf-8'
        )
        print(f'Wrote {args.out}')
