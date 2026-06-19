"""
legend_reader.py — extract symbol templates + labels from a project's legend page.

For each entry in the legend, we want:
  - A cropped image of the symbol (template, used downstream for template
    matching against the floor plan)
  - The OCR'd text label next to it ("SUPPLY DIFFUSER", "RETURN GRILLE", ...)
  - A canonical class name normalized against our YOLO class vocabulary

Why this exists: the YOLO model only recognizes shapes it was trained on.
A drawing's legend tells us what symbols THIS drawing uses. By extracting
templates here, we can match them on the plans even if YOLO has never
seen this specific drawing style — which closes the per-project adaptation
gap identified in PLAN.md §3.

Approach:
  1. Find the legend page via page_classifier (or fall back to OCR keyword
     search for "LEGEND" / "SYMBOL LEGEND" / "ABBREVIATIONS").
  2. OCR the page at 300 DPI to get every text item with bbox.
  3. Cluster OCR text items into horizontal rows (a legend entry per row).
  4. For each row, find a graphics region to the LEFT of the text — crop it.
  5. Save crops to disk + emit a manifest JSON of (symbol_image, label, normalized_class).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


# Lazy OCR
_easyocr_reader = None
def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


LEGEND_KEYWORDS = (
    'LEGEND', 'SYMBOL LEGEND', 'SYMBOLS', 'ABBREVIATIONS',
    'KEYED NOTES', 'MECHANICAL SYMBOLS', 'HVAC SYMBOLS',
    'MECHANICAL LEGEND', 'EQUIPMENT LEGEND',
)


# Map OCR'd legend labels to our YOLO class vocabulary
LABEL_TO_YOLO_CLASS = {
    # Diffusers / grilles
    r'SUPPLY\s+DIFFUSER': 'AD-T-BAR SUPPLY',
    r'SUPPLY\s+REGISTER': 'AD-T-BAR SUPPLY',
    r'CEILING\s+DIFFUSER': 'AD-T-BAR SUPPLY',
    r'T[-\s]?BAR\s+SUPPLY': 'AD-T-BAR SUPPLY',
    r'RETURN\s+GRILLE': 'AD-T-BAR RETURN',
    r'RETURN\s+AIR': 'AD-T-BAR RETURN',
    r'T[-\s]?BAR\s+RETURN': 'AD-T-BAR RETURN',
    r'EXHAUST\s+GRILLE': 'AD-T-BAR RETURN',
    r'TRANSFER\s+GRILLE': 'AD-T-BAR RETURN',
    r'LINEAR\s+(SLOT\s+)?DIFFUSER': 'AD-LINEAR SLOT DIFFUSER',
    r'LINEAR\s+PLENUM': 'AD-LINEAR PLENUM',

    # Fans
    r'EXHAUST\s+FAN': 'EXHAUST FAN',
    r'SUPPLY\s+FAN': 'FAN',
    r'INLINE\s+FAN': 'FAN',
    r'ROOF\s+FAN': 'EXHAUST FAN',
    r'CEILING\s+FAN': 'FAN',

    # Equipment
    r'CONDENSING\s+UNIT': 'CONDENSING UNIT',
    r'AIR\s+COOLED\s+CONDENSING': 'AIR COOLED CONDENSING UNIT',
    r'ROOFTOP\s+UNIT': 'PACKAGED ROOFTOP UNIT',
    r'PACKAGED\s+ROOFTOP': 'PACKAGED ROOFTOP UNIT',
    r'RTU\b': 'PACKAGED ROOFTOP UNIT',
    r'AHU\b': 'AIR HANDLING UNIT',
    r'AIR\s+HANDLING': 'AIR HANDLING UNIT',
    r'FAN\s+COIL': 'FAN COIL UNIT',
    r'ENERGY\s+RECOVERY|ERV\b': 'CONDENSING UNIT',
    r'SPLIT\s+SYSTEM': 'SPLIT SYSTEM OUTDOOR',

    # Dampers
    r'FIRE\s+SMOKE\s+DAMPER|FIRE/SMOKE\s+DAMPER': 'FIRE SMOKE DAMPER',
    r'FIRE\s+DAMPER': 'FIRE DAMPER',
    r'SMOKE\s+DAMPER': 'FIRE SMOKE DAMPER',
    r'MANUAL\s+VOLUME\s+DAMPER|MVD\b|VOLUME\s+DAMPER': 'MANUAL VOLUME DAMPER',
    r'MOTORIZED\s+DAMPER|MD\b': 'MOTORIZED DAMPER',
    r'BACKDRAFT\s+DAMPER|BDD\b': 'BACKDRAFT DAMPER',

    # Misc
    r'LOUVER|LVR\b': 'LOUVER',
    r'VENT\s+CAP': 'VENT CAP',
    r'RAIN\s+CAP|ROOF\s+CAP': 'RAIN CAP',
    r'RELIEF\s+HOOD|ROOF\s+HOOD': 'RELIEF HOOD',
    r'HEATER|EUH\b|UH\b': 'HEATER',
    r'GAS\s+UNIT\s+HEATER': 'GAS UNIT HEATER',
}


def normalize_label_to_class(label: str) -> str | None:
    """Map a legend label string to a canonical YOLO class. None if no match."""
    if not label:
        return None
    upper = label.upper().strip()
    for pat, cls in LABEL_TO_YOLO_CLASS.items():
        if re.search(pat, upper):
            return cls
    return None


def find_legend_page(pdf_path: Path,
                    classifications: list | None = None) -> int | None:
    """Return 0-indexed page of the legend, or None if not found.

    Scoring approach (vs first-match): a real symbol legend has BOTH
    legend keywords AND a high density of short HVAC class labels
    (SUPPLY DIFFUSER, RETURN GRILLE etc). Title sheets often just say
    "LEGEND OF DRAWINGS" in passing.
    """
    # Prefer page_classifier's answer if provided
    if classifications:
        legend_pages = []
        for c in classifications:
            ctype = c.get('type') if isinstance(c, dict) else getattr(c, 'type', None)
            page = c.get('page') if isinstance(c, dict) else getattr(c, 'page', None)
            if ctype == 'legend' and page is not None:
                legend_pages.append(page - 1)
        # If multiple, score and pick the best below; if one, use it
        if len(legend_pages) == 1:
            return legend_pages[0]

    # Score every page on (legend keywords + class-label density)
    doc = fitz.open(str(pdf_path))
    try:
        scores = []
        for pno in range(doc.page_count):
            text = (doc[pno].get_text('text') or '').upper()
            short = text[:5000]
            kw_hits = sum(1 for kw in LEGEND_KEYWORDS if kw in short)
            # Class-label density: how many distinct HVAC class patterns
            # appear in the page text? More = more likely a symbol legend.
            label_hits = 0
            for pat in LABEL_TO_YOLO_CLASS:
                if re.search(pat, text):
                    label_hits += 1
            score = kw_hits * 2 + label_hits
            scores.append((score, pno, kw_hits, label_hits))
        # Pick highest-score page if it has at least one legend keyword OR
        # at least 4 class-label matches
        scores.sort(key=lambda x: -x[0])
        for score, pno, kw, labels in scores:
            if kw >= 1 and labels >= 2:
                return pno
            if labels >= 4:
                return pno
        # Fall back to first kw hit
        for score, pno, kw, labels in scores:
            if kw >= 1:
                return pno
    finally:
        doc.close()
    return None


def render_page_image(pdf_path: Path, page_index_0based: int,
                     dpi: int = 300):
    """Render a page to RGB numpy array."""
    import numpy as np
    doc = fitz.open(str(pdf_path))
    try:
        if page_index_0based < 0 or page_index_0based >= doc.page_count:
            return None
        page = doc[page_index_0based]
        pix = page.get_pixmap(dpi=dpi, annots=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = img[:, :, :3]
        return img
    finally:
        doc.close()


def ocr_page(img) -> list[dict]:
    """OCR a rendered page image. Returns words with bbox+conf+text."""
    reader = _get_ocr_reader()
    raw = reader.readtext(img, detail=1, paragraph=False)
    out = []
    for box, text, conf in raw:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        out.append({
            'text': (text or '').strip(),
            'x1': float(min(xs)), 'y1': float(min(ys)),
            'x2': float(max(xs)), 'y2': float(max(ys)),
            'cx': (min(xs) + max(xs)) / 2,
            'cy': (min(ys) + max(ys)) / 2,
            'conf': float(conf or 0),
        })
    return out


def cluster_into_rows(items: list[dict], row_tol_px: float = 30) -> list[list[dict]]:
    """Group OCR items into horizontal rows."""
    if not items:
        return []
    sorted_items = sorted(items, key=lambda d: d['cy'])
    rows = [[sorted_items[0]]]
    last_y = sorted_items[0]['cy']
    for it in sorted_items[1:]:
        if abs(it['cy'] - last_y) <= row_tol_px:
            rows[-1].append(it)
        else:
            rows.append([it])
        last_y = it['cy']
    # Sort each row left-to-right
    for r in rows:
        r.sort(key=lambda d: d['cx'])
    return rows


# Abbreviation / linetype legend pattern (text-layer path). Only the reliable
# trailing form: "CONDENSATE DRAIN (CD)" → CD = Condensate Drain. The leading
# "ABBR  term" form is NOT used — CAD title blocks linearize the abbr column
# separately, so it mangles multi-word terms ("AIR CONDITIONING" → AIR=CONDITIONING).
_ABBR_TRAILING = re.compile(r'^(.{2,48}?)\s*\(([A-Z][A-Z0-9/\-]{0,9})\)\s*$')
_NOTE_WORDS = ('SHALL', 'PROVIDE', 'CONTRACTOR', 'PRIOR', 'REVIEW', 'INSTALL',
               'REQUIRED', 'SUBMIT', 'ACCORDANCE', 'PER ', 'NOT BE', 'SYSTEM(S)')


def extract_legend(pdf_path: Path, classifications: list | None = None,
                   page_override_1based: int | None = None) -> dict:
    """Read the legend from the PDF TEXT LAYER (fast + clean). Extracts an
    abbreviation/linetype dictionary plus any equipment-symbol labels that map
    to a YOLO class. Falls back to the OCR crop reader only for raster legends.

    Combined "General Notes AND Legend" sheets are common — note prose is
    filtered out so it doesn't pollute the dictionary.
    """
    if page_override_1based is not None:
        page_idx = page_override_1based - 1
    else:
        page_idx = find_legend_page(pdf_path, classifications=classifications)
    if page_idx is None:
        return {'page': None, 'source': None, 'abbreviations': [], 'symbols': [],
                'reason': 'no legend page identified'}

    doc = fitz.open(str(pdf_path))
    try:
        text = doc[page_idx].get_text('text') or ''
    finally:
        doc.close()

    abbr: dict[str, str] = {}
    symbols: list[dict] = []
    seen_sym = set()
    for raw in text.splitlines():
        line = ' '.join(raw.split())
        if not line:
            continue
        up = line.upper()

        m = _ABBR_TRAILING.match(line)
        if m and not any(w in up for w in _NOTE_WORDS):
            term, a = m.group(1).strip(' .,-'), m.group(2).strip()
            if term and re.search(r'[A-Za-z]{2,}', term) and a not in abbr:
                abbr[a] = term
                continue

        # Equipment-symbol label → YOLO class (short label-like lines only).
        if len(line) <= 38 and len(line.split()) <= 5 and not any(w in up for w in _NOTE_WORDS):
            cls = normalize_label_to_class(line)
            if cls and up not in seen_sym:
                seen_sym.add(up)
                symbols.append({'label': line, 'class': cls})

    abbreviations = [{'abbr': k, 'term': v} for k, v in abbr.items()]

    # If the text layer is too thin (raster legend), fall back to OCR crops.
    if len(abbreviations) + len(symbols) < 3:
        ocr = extract_legend_entries(pdf_path, classifications=classifications,
                                     page_override_1based=(page_idx + 1))
        ocr['source'] = 'ocr'
        ocr.setdefault('abbreviations', [])
        ocr['symbols'] = [{'label': e['label'], 'class': e['normalized_class']}
                          for e in ocr.get('entries', []) if e.get('normalized_class')]
        return ocr

    return {
        'page': page_idx + 1,
        'source': 'text',
        'abbreviations': abbreviations,
        'symbols': symbols,
        'n_abbr': len(abbreviations),
        'n_symbols': len(symbols),
    }


def extract_legend_entries(pdf_path: Path,
                          classifications: list | None = None,
                          output_dir: Path | None = None,
                          dpi: int = 300,
                          page_override_1based: int | None = None) -> dict:
    """Top-level: find legend page, extract per-row (symbol image, label).

    Returns:
      {
        'page': 0-indexed page where legend was found, or None,
        'entries': [
          {label, normalized_class, symbol_image_path, text_box},
          ...
        ],
        'n_total_rows': int,
        'n_matched_class': int,
      }
    """
    if page_override_1based is not None:
        page_idx = page_override_1based - 1
    else:
        page_idx = find_legend_page(pdf_path, classifications=classifications)
    if page_idx is None:
        return {
            'page': None,
            'entries': [],
            'reason': 'no legend page identified',
        }

    img = render_page_image(pdf_path, page_idx, dpi=dpi)
    if img is None:
        return {'page': page_idx + 1, 'entries': [],
                'reason': 'page render failed'}

    ocr_items = ocr_page(img)
    if not ocr_items:
        return {'page': page_idx + 1, 'entries': [],
                'reason': 'OCR returned no text'}

    rows = cluster_into_rows(ocr_items)

    # For each row: text region is the rightmost contiguous text cluster;
    # symbol region is the area to the LEFT of that text. Crop the leftmost
    # ~150px of the row's bounding box as the symbol candidate.
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        cv2 = None

    entries = []
    for ri, row in enumerate(rows):
        # Combine row text
        label = ' '.join(it['text'] for it in row if it['text']).strip()
        if not label or len(label) < 3:
            continue

        # Skip header rows
        if any(kw in label.upper() for kw in
               ('LEGEND', 'SYMBOLS', 'DESCRIPTION', 'DRAWING')):
            continue

        # Map to YOLO class
        normalized = normalize_label_to_class(label)

        # Compute row bbox; symbol crop is the area LEFT of the first text item
        row_y1 = min(it['y1'] for it in row)
        row_y2 = max(it['y2'] for it in row)
        text_start_x = row[0]['x1']
        # Symbol region: a 200px wide region ending at text_start_x
        sym_x1 = max(0, int(text_start_x - 200))
        sym_x2 = int(text_start_x - 5)
        sym_y1 = max(0, int(row_y1 - 15))
        sym_y2 = min(img.shape[0], int(row_y2 + 15))

        if sym_x2 <= sym_x1 + 10 or sym_y2 <= sym_y1 + 10:
            continue  # degenerate region

        symbol_image_path = None
        if output_dir and cv2:
            crop = img[sym_y1:sym_y2, sym_x1:sym_x2]
            if crop.size:
                # Save as PNG, converting RGB → BGR for cv2.imwrite
                if crop.shape[2] == 3:
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                else:
                    crop_bgr = crop
                stem_label = re.sub(r'[^A-Za-z0-9_-]', '_',
                                   label.upper())[:40]
                fname = f'legend_row{ri:02d}_{stem_label}.png'
                fpath = output_dir / fname
                cv2.imwrite(str(fpath), crop_bgr)
                symbol_image_path = str(fpath.name)

        entries.append({
            'row_index': ri,
            'label': label,
            'normalized_class': normalized,
            'symbol_image': symbol_image_path,
            'symbol_bbox_px': [sym_x1, sym_y1, sym_x2, sym_y2],
            'text_bbox_px': [int(row[0]['x1']), int(row_y1),
                            int(row[-1]['x2']), int(row_y2)],
            'ocr_confidence': round(sum(it['conf'] for it in row) / max(1, len(row)), 3),
        })

    return {
        'page': page_idx + 1,
        'page_index_0based': page_idx,
        'image_dpi': dpi,
        'entries': entries,
        'n_total_rows': len(rows),
        'n_matched_class': sum(1 for e in entries if e.get('normalized_class')),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--page-classifications', help='optional path to page_classifications.json')
    ap.add_argument('--page', type=int, default=None,
                    help='1-indexed page number override (skip auto-detection)')
    ap.add_argument('--out-dir', default='./legend_out',
                    help='where to save symbol crops')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    classifications = None
    if args.page_classifications:
        classifications = json.loads(Path(args.page_classifications).read_text(encoding='utf-8'))

    result = extract_legend_entries(
        Path(args.pdf),
        classifications=classifications,
        output_dir=Path(args.out_dir),
        dpi=args.dpi,
        page_override_1based=args.page,
    )

    print(f'Legend page: {result.get("page")}')
    print(f'Rows scanned: {result.get("n_total_rows", 0)}')
    print(f'Entries extracted: {len(result.get("entries", []))}')
    print(f'With class match: {result.get("n_matched_class", 0)}')
    print()
    for e in result.get('entries', [])[:20]:
        cls = e.get('normalized_class') or '(no match)'
        print(f"  [{e['row_index']:02d}] {e['label'][:50]:50s} -> {cls}")

    out_json = Path(args.out_dir) / 'legend_dictionary.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str),
                       encoding='utf-8')
    print(f'\nWrote {out_json}')
