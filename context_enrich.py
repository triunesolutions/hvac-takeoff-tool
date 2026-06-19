"""
context_enrich.py

Apply Deck-2-tagging rules to raw YOLO detections before they get stamped
onto the PDF as Bluebeam annotations. The rules come from the team's
"UNIQUE FILE TAGGING" deck.

Three rules implemented:

  Rule A — Fire Smoke Damper context (deck slides 18, 20)
      * FSD in duct        → tag 'FSD',     type 'INLINE DUCTED'
      * FSD attached to GRD → tag 'FSD-OP', type 'OUT OF PARTITION'
      Detection: FSD bbox close to or overlapping any AD-GRD bbox.

  Rule B — Ceiling Radiation Damper (deck slide 6)
      * If 'CRD' text appears near an AD-GRD detection,
        attach damper_type='CRD' so the estimator sees it on hover.

  Rule C — Continuous linear diffuser merging (deck slide 11)
      * Colinear AD-LINEAR PLENUM / SLOT DIFFUSER boxes that share a
        baseline and sit end-to-end count as ONE part with summed face length.

Coordinate convention:
  All detection bboxes are in pixel space at the rendered DPI from
  the detections.json (typically 200). PDF text words are in PDF
  points (72 DPI); we scale them up.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


# ---- geometry helpers ----------------------------------------------------

def _bbox_iou(a: dict, b: dict) -> float:
    ix1 = max(a['x1'], b['x1']); iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2']); iy2 = min(a['y2'], b['y2'])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a['x2']-a['x1']) * (a['y2']-a['y1'])
    ub = (b['x2']-b['x1']) * (b['y2']-b['y1'])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _bbox_center_distance(a: dict, b: dict) -> float:
    acx = (a['x1']+a['x2']) / 2; acy = (a['y1']+a['y2']) / 2
    bcx = (b['x1']+b['x2']) / 2; bcy = (b['y1']+b['y2']) / 2
    return ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5


GRD_CLASSES = {
    'AD-GRD', 'AD-T-BAR SUPPLY', 'AD-T-BAR RETURN',
    'AD-SURF SUPPLY', 'AD-SURF RETURN',
}


# ---- Rule A: FSD context -------------------------------------------------

def apply_fsd_context(dets_on_page: list[dict]) -> int:
    """In-place. Returns count of FSDs reclassified as FSD-OP."""
    grds = [d for d in dets_on_page if d.get('cls') in GRD_CLASSES]
    n_op = 0
    for det in dets_on_page:
        if det.get('cls') != 'FIRE SMOKE DAMPER':
            continue
        # 'With GRD' means overlapping OR very close (< 100 px ~ 0.5 inch at 200 DPI)
        is_with_grd = any(
            _bbox_iou(det, g) > 0.05 or _bbox_center_distance(det, g) < 100
            for g in grds
        )
        if is_with_grd:
            det['context_tag'] = 'FSD-OP'
            det['context_type'] = 'OUT OF PARTITION'
            n_op += 1
        else:
            det['context_tag'] = 'FSD'
            det['context_type'] = 'INLINE DUCTED'
    return n_op


# ---- Rule B: CRD detection from page text -------------------------------

def apply_crd_detection(pdf_path: Path, page_num_1based: int,
                       dets_on_page: list[dict], px_per_pt: float) -> int:
    """In-place. Look for 'CRD' word labels on the page, attach damper_type
    to nearby AD-GRD detections. Returns count of CRDs found.
    """
    doc = fitz.open(str(pdf_path))
    try:
        if page_num_1based - 1 >= doc.page_count:
            return 0
        page = doc[page_num_1based - 1]
        words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
    finally:
        doc.close()

    # Find CRD labels (also accept C.R.D., common variants)
    crd_centers_px = []
    for w in words:
        text = (w[4] or '').upper().replace('.', '').strip()
        if text == 'CRD':
            cx_pt = (w[0] + w[2]) / 2
            cy_pt = (w[1] + w[3]) / 2
            crd_centers_px.append((cx_pt * px_per_pt, cy_pt * px_per_pt))

    if not crd_centers_px:
        return 0

    # Tag each AD-GRD whose center is within 200 px of any CRD label.
    # 200 px @ 200 DPI ~ 1 inch ~ typical label-to-symbol distance.
    PROX_PX = 200
    for det in dets_on_page:
        if det.get('cls') not in GRD_CLASSES:
            continue
        dcx = (det['x1'] + det['x2']) / 2
        dcy = (det['y1'] + det['y2']) / 2
        for (cx, cy) in crd_centers_px:
            if abs(dcx - cx) < PROX_PX and abs(dcy - cy) < PROX_PX:
                det['damper_type'] = 'CRD'
                break

    return len(crd_centers_px)


# ---- Rule C: merge continuous linear diffusers --------------------------

LINEAR_CLASSES = {'AD-LINEAR PLENUM', 'AD-LINEAR SLOT DIFFUSER'}

# How far apart (in pixels at the rendered DPI) two linear boxes can be along
# their major axis and still count as colinear/continuous.
GAP_TOLERANCE_PX = 80

# How tight the perpendicular alignment must be. Two boxes are colinear iff
# their min/max along the minor axis match within this fraction of the box
# size along that axis.
ALIGN_TOLERANCE_FRAC = 0.3


def _is_horizontal_run(a: dict, b: dict) -> bool:
    """Same Y baseline + adjacent in X."""
    a_h = a['y2'] - a['y1']
    if abs(a['y1'] - b['y1']) > a_h * ALIGN_TOLERANCE_FRAC: return False
    if abs(a['y2'] - b['y2']) > a_h * ALIGN_TOLERANCE_FRAC: return False
    # Adjacency: minimum end-to-end gap
    gap = max(b['x1'] - a['x2'], a['x1'] - b['x2'])
    return -10 <= gap <= GAP_TOLERANCE_PX  # allow tiny overlap


def _is_vertical_run(a: dict, b: dict) -> bool:
    a_w = a['x2'] - a['x1']
    if abs(a['x1'] - b['x1']) > a_w * ALIGN_TOLERANCE_FRAC: return False
    if abs(a['x2'] - b['x2']) > a_w * ALIGN_TOLERANCE_FRAC: return False
    gap = max(b['y1'] - a['y2'], a['y1'] - b['y2'])
    return -10 <= gap <= GAP_TOLERANCE_PX


def merge_linear_diffusers(dets_on_page: list[dict]) -> list[dict]:
    """Return a NEW list where colinear runs are merged into single dets."""
    targets = [d for d in dets_on_page if d.get('cls') in LINEAR_CLASSES]
    others = [d for d in dets_on_page if d.get('cls') not in LINEAR_CLASSES]
    if len(targets) < 2:
        return list(dets_on_page)

    # Union-find: group indices that are adjacent
    parent = list(range(len(targets)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(targets):
        for j in range(i+1, len(targets)):
            b = targets[j]
            if a['cls'] != b['cls']:
                continue
            if _is_horizontal_run(a, b) or _is_vertical_run(a, b):
                union(i, j)

    # Bucket members by root
    groups: dict[int, list[int]] = {}
    for i in range(len(targets)):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged_list = []
    for root, idxs in groups.items():
        members = [targets[i] for i in idxs]
        if len(members) == 1:
            merged_list.append(members[0])
            continue
        x1 = min(m['x1'] for m in members)
        y1 = min(m['y1'] for m in members)
        x2 = max(m['x2'] for m in members)
        y2 = max(m['y2'] for m in members)
        # Compute face length (the longer dimension)
        face_px = max(x2 - x1, y2 - y1)
        merged_list.append({
            'cls': members[0]['cls'],
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'conf': sum(m.get('conf', 0) for m in members) / len(members),
            'tag': None,
            'tag_method': 'merged-linear',
            'merged_count': len(members),
            'face_length_px': face_px,
        })

    return others + merged_list


# ---- Rule D: TYP/TYPICAL marking detection (Deck 2 slide 4) -------------

def apply_typ_marking(pdf_path: Path, page_num_1based: int,
                     dets_on_page: list[dict], px_per_pt: float) -> int:
    """Per Deck 2 slide 4: when AD-GRD is marked "TYP" or "TYPICAL", that one
    symbol's properties propagate to all similar-placement detections on the
    same page. We flag the marked detections so the estimator can extend
    properties; we do NOT auto-replicate (that would create phantom counts).
    Returns count of TYP-marked detections flagged.
    """
    doc = fitz.open(str(pdf_path))
    try:
        if page_num_1based - 1 >= doc.page_count:
            return 0
        page = doc[page_num_1based - 1]
        words = page.get_text("words")
    finally:
        doc.close()

    typ_centers_px = []
    for w in words:
        text = (w[4] or '').upper().strip(' .,')
        if text in ('TYP', 'TYPICAL', 'TYP.'):
            cx_px = (w[0] + w[2]) / 2 * px_per_pt
            cy_px = (w[1] + w[3]) / 2 * px_per_pt
            typ_centers_px.append((cx_px, cy_px))
    if not typ_centers_px:
        return 0

    PROX_PX = 200
    n = 0
    for det in dets_on_page:
        dcx = (det['x1'] + det['x2']) / 2
        dcy = (det['y1'] + det['y2']) / 2
        for (cx, cy) in typ_centers_px:
            if abs(dcx - cx) < PROX_PX and abs(dcy - cy) < PROX_PX:
                det['typical_marker'] = True
                n += 1
                break
    return n


# ---- Top-level entry point ----------------------------------------------

def enrich(detections: dict, pdf_path: Path) -> dict:
    """Take a detections.json dict and return an enriched copy.

    The input structure is preserved:
        {pdf, dpi, pages: {page_num_str: [det, ...]}}
    """
    out = deepcopy(detections)
    dpi = out.get('dpi', 200)
    px_per_pt = dpi / 72.0

    stats = {'fsd_op': 0, 'crd_pages': 0, 'merged_groups': 0, 'merged_into': 0,
             'typ_flagged': 0, 'mitered_rectangles': 0, 'curved_segments': 0}

    # Lazy imports for the additional rules
    try:
        from mitered_corners import annotate_mitered_corners
    except Exception:
        annotate_mitered_corners = None
    try:
        from curved_diffuser import annotate_curved_segments
    except Exception:
        annotate_curved_segments = None

    new_pages = {}
    for pkey, dets in out['pages'].items():
        pno_1 = int(pkey)
        # In-place rules
        stats['fsd_op'] += apply_fsd_context(dets)
        n_crd = apply_crd_detection(pdf_path, pno_1, dets, px_per_pt)
        if n_crd:
            stats['crd_pages'] += 1
        stats['typ_flagged'] += apply_typ_marking(pdf_path, pno_1, dets, px_per_pt)
        # Mitered-corner rectangles (4 linears forming a closed quad)
        if annotate_mitered_corners:
            stats['mitered_rectangles'] += annotate_mitered_corners(dets)
        # Curved slot diffuser arc length
        if annotate_curved_segments:
            stats['curved_segments'] += annotate_curved_segments(dets)
        # Merge produces a new list
        before = len(dets)
        dets = merge_linear_diffusers(dets)
        if len(dets) < before:
            stats['merged_groups'] += 1
            stats['merged_into'] += (before - len(dets))
        new_pages[pkey] = dets

    out['pages'] = new_pages
    out['_enrichment_stats'] = stats
    return out


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--detections', required=True)
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--out', help='Where to write enriched JSON (default: stdout summary)')
    args = ap.parse_args()

    raw = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    enriched = enrich(raw, Path(args.pdf))
    if args.out:
        Path(args.out).write_text(json.dumps(enriched, indent=2), encoding='utf-8')
        print(f'Wrote {args.out}')
    print(f'Enrichment stats: {enriched["_enrichment_stats"]}')
