"""
typ_uno_nic.py — apply TYP / NIC plan-note semantics to detections.

This is the MVP slice of TYP/UNO/NIC handling. Two unambiguous, count-safe
rules are implemented here; the harder cases are deliberately left as flags
rather than silent count changes.

  Rule NIC — "Not In Contract" / "By Others"
      Tokens: NIC, N.I.C, N.I.C., or the phrases "NOT IN CONTRACT" /
      "BY OTHERS" found near a detection mark that detection with
      det['nic'] = True. NIC items are EXCLUDED from billable counts
      (see tag_report._effective_count) but still listed separately so the
      estimator can see what was carved out.

  Rule TYP-OF-N — explicit "(TYP OF 6)" multiplier
      Phrases: "TYP OF N", "TYP. OF N", "TYPICAL OF N" (parens optional)
      found near a detection set det['typ_of_count'] = N. The detection
      then counts as N units instead of 1.

NOT handled here (left for a later slice — see scope notes):
  • Bare "TYP" / "TYPICAL" with no explicit count. That is already FLAGGED
    elsewhere (context_enrich.apply_typ_marking sets det['typical_marker']);
    auto-multiplying it would create phantom counts, so we never do.
  • UNO ("unless noted otherwise") attribute defaults from general notes.

Coordinate convention matches context_enrich.py: detection bboxes are in
pixel space at the detections.json DPI; PDF words are in points (72 DPI)
and scaled up by px_per_pt.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


# Proximity from a note label to the symbol it annotates. 200 px @ 200 DPI
# ~ 1 inch — the same label-to-symbol distance the other rules assume.
PROX_PX = 200

# Single-token NIC variants (after stripping '.' and surrounding punctuation).
_NIC_TOKENS = {'NIC'}

# Multi-word NIC phrases, as tuples of normalized words.
_NIC_PHRASES = [
    ('NOT', 'IN', 'CONTRACT'),
    ('BY', 'OTHERS'),
]

# "TYP OF 6", "TYPICAL OF 6", "TYP. OF (6)" — paren/punct tolerant.
_TYP_OF_RE = re.compile(r'\bTYP(?:ICAL)?\.?\s*OF\s*\(?\s*(\d{1,3})\b', re.IGNORECASE)


def _norm(word: str) -> str:
    """Uppercase, strip surrounding punctuation and internal dots."""
    return (word or '').upper().strip(' .,():;').replace('.', '')


def _page_words(pdf_path: Path, page_num_1based: int):
    """Return PyMuPDF words list for a page, or [] if out of range."""
    doc = fitz.open(str(pdf_path))
    try:
        if page_num_1based - 1 >= doc.page_count or page_num_1based < 1:
            return []
        return doc[page_num_1based - 1].get_text("words")
    finally:
        doc.close()


def _nearest_det(cx: float, cy: float, dets: list[dict]):
    """Return the detection whose center is closest to (cx, cy) within
    PROX_PX, or None."""
    best = None
    best_d = PROX_PX
    for det in dets:
        dcx = (det['x1'] + det['x2']) / 2
        dcy = (det['y1'] + det['y2']) / 2
        d = ((dcx - cx) ** 2 + (dcy - cy) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = det
    return best


def _find_nic_labels(words, px_per_pt: float) -> list[tuple[float, float]]:
    """Return pixel-space centers of every NIC label/phrase on the page."""
    centers: list[tuple[float, float]] = []
    norm = [_norm(w[4]) for w in words]

    # Single-token NIC
    for i, t in enumerate(norm):
        if t in _NIC_TOKENS:
            w = words[i]
            cx = (w[0] + w[2]) / 2 * px_per_pt
            cy = (w[1] + w[3]) / 2 * px_per_pt
            centers.append((cx, cy))

    # Multi-word phrases
    for phrase in _NIC_PHRASES:
        L = len(phrase)
        for i in range(len(norm) - L + 1):
            if tuple(norm[i:i + L]) == phrase:
                xs = [words[j][0] for j in range(i, i + L)] + \
                     [words[j][2] for j in range(i, i + L)]
                ys = [words[j][1] for j in range(i, i + L)] + \
                     [words[j][3] for j in range(i, i + L)]
                cx = (min(xs) + max(xs)) / 2 * px_per_pt
                cy = (min(ys) + max(ys)) / 2 * px_per_pt
                centers.append((cx, cy))
    return centers


def _find_typ_of_labels(words, px_per_pt: float) -> list[tuple[float, float, int]]:
    """Return (cx_px, cy_px, n) for every '(TYP OF N)' label.

    PyMuPDF splits on whitespace, so "TYP OF 6" arrives as separate words.
    We reconstruct short text windows and regex them, then locate the window.
    """
    out: list[tuple[float, float, int]] = []
    n = len(words)
    for i in range(n):
        if _norm(words[i][4])[:3] != 'TYP':
            continue
        # Join this word with up to the next 3 to capture "TYP OF (6)"
        window_words = words[i:i + 4]
        text = ' '.join(w[4] for w in window_words)
        m = _TYP_OF_RE.search(text)
        if not m:
            continue
        count = int(m.group(1))
        if count < 2:
            continue
        xs = [w[0] for w in window_words] + [w[2] for w in window_words]
        ys = [w[1] for w in window_words] + [w[3] for w in window_words]
        cx = (min(xs) + max(xs)) / 2 * px_per_pt
        cy = (min(ys) + max(ys)) / 2 * px_per_pt
        out.append((cx, cy, count))
    return out


def apply_typ_uno_nic(pdf_path: Path, detections: dict,
                      plan_pages_1based: list[int]) -> dict:
    """Mutate `detections` in place: set det['nic'] / det['typ_of_count'].

    Returns a summary dict suitable for writing as {stem}_typ_uno_nic.json.
    """
    dpi = detections.get('dpi', 200)
    px_per_pt = dpi / 72.0
    pages = detections.get('pages', {})

    summary = {
        'dpi': dpi,
        'n_nic_labels': 0,
        'n_nic_detections': 0,
        'n_typ_of_labels': 0,
        'n_typ_of_detections': 0,
        'typ_of_extra_units': 0,   # sum of (N-1) added by the multiplier
        'nic_items': [],
        'typ_of_items': [],
        'by_page': {},
    }

    plan_set = set(plan_pages_1based) if plan_pages_1based else None

    for pkey, dets in pages.items():
        pno = int(pkey)
        if plan_set is not None and pno not in plan_set:
            continue
        if not dets:
            continue
        words = _page_words(Path(pdf_path), pno)
        if not words:
            continue

        nic_centers = _find_nic_labels(words, px_per_pt)
        typ_labels = _find_typ_of_labels(words, px_per_pt)
        if not nic_centers and not typ_labels:
            continue

        page_stats = {'nic_labels': len(nic_centers),
                      'typ_of_labels': len(typ_labels),
                      'nic_detections': 0, 'typ_of_detections': 0}

        for (cx, cy) in nic_centers:
            det = _nearest_det(cx, cy, dets)
            if det is not None and not det.get('nic'):
                det['nic'] = True
                page_stats['nic_detections'] += 1
                summary['n_nic_detections'] += 1
                summary['nic_items'].append({
                    'page': pno, 'cls': det.get('cls'),
                    'tag': det.get('tag'),
                    'bbox': [det['x1'], det['y1'], det['x2'], det['y2']],
                })

        for (cx, cy, count) in typ_labels:
            det = _nearest_det(cx, cy, dets)
            if det is not None and not det.get('typ_of_count'):
                det['typ_of_count'] = count
                det['typ_flag'] = 'typ_of'
                page_stats['typ_of_detections'] += 1
                summary['n_typ_of_detections'] += 1
                summary['typ_of_extra_units'] += (count - 1)
                summary['typ_of_items'].append({
                    'page': pno, 'cls': det.get('cls'),
                    'tag': det.get('tag'), 'count': count,
                    'bbox': [det['x1'], det['y1'], det['x2'], det['y2']],
                })

        summary['n_nic_labels'] += len(nic_centers)
        summary['n_typ_of_labels'] += len(typ_labels)
        summary['by_page'][str(pno)] = page_stats

    return summary


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--detections', required=True)
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--plan-pages', nargs='*', type=int, default=None)
    ap.add_argument('--out')
    args = ap.parse_args()

    dets = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    s = apply_typ_uno_nic(Path(args.pdf), dets, args.plan_pages)
    if args.out:
        Path(args.out).write_text(json.dumps(dets, indent=2), encoding='utf-8')
        print(f'Wrote enriched detections to {args.out}')
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ('nic_items', 'typ_of_items')}, indent=2))
