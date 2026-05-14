"""
Text-layer-first tag extractor.

Many PDF plans have the equipment tag strings embedded directly in the text
layer of each page — even when pdfplumber's table-based schedule extraction
finds nothing. Example: Yucaipa A page.get_text() already contains every
`IDU-01A..IDU-30D`, `ODU-01..29`, `EF-01`, and `DBF-01` tag.

This module walks every page, regex-pulls tag-shaped tokens, applies the
same noise filters as the schedule parser (sheet refs, code refs,
refrigerants, room labels), and returns TagVariable dicts in the same shape
as `schedule_parser.parse_pdf_schedules`.

Free, instant, zero OCR cost. Use it as the primary tag-discovery path and
fall back to OCR only when the text layer is empty.

Usage:
    from text_layer_tag_extractor import extract_tags_from_text_layer
    extra_vars = extract_tags_from_text_layer(pdf_path)
"""
import re
import sys
from collections import OrderedDict
from pathlib import Path

import fitz

from tag_inference import _infer_class_from_tag
from schedule_parser import normalize_tag, REFRIGERANT_PATTERN, BANNED_TAG_PREFIXES


# Tag-like tokens in the text layer. Same shape as OCR_TAG_SHAPE but allows
# the digit-letter-digit suffix combinations we saw on Yucaipa (IDU-01A,
# IDU-07D) and ARE Campus Point (CU-1-3, AHU-1-3).
TEXT_TAG_PATTERN = re.compile(
    r'(?<![A-Z0-9-])'                       # not preceded by alphanum or hyphen
    r'([A-Z]{1,5}-\d{1,3}[A-Z]?(?:-\d{1,3})?)'   # core tag shape
    r'(?![A-Z0-9])',                        # not followed by alphanum
)

# Building / regulation / drawing prefixes that LOOK like tags but never are.
# CMC = California Mechanical Code, CEC = California Energy Code,
# CGBC = California Green, IBC, IECC, NFPA, ASME, ASHRAE, etc. T-24 = Title 24.
CODE_PREFIXES = {
    'CMC', 'CEC', 'CGBC', 'IBC', 'IRC', 'IFC', 'IECC', 'IMC', 'IPC',
    'NFPA', 'ASME', 'ASHRAE', 'ASTM', 'UL', 'CSA', 'AHRI',
    'T',          # Title 24 ("T-24")
    'CSI',        # spec division
    'TYPE',       # cross-ref ("TYPE-1")
    'DET',        # detail callout
    'SIM',        # similar-detail callout
    'REV',        # revision number
    'SEC',        # section
    'NOTE',
}

# Sheet number shape: single uppercase letter + 2–4 digits (M101, P201, E-301).
SHEET_REF_RE = re.compile(r'^[A-Z]-?\d{2,4}$')

# Room/unit label shape with single letter prefix + 3 digits (A101, B205).
# These DOMINATE residential floor plans (Yucaipa: A101..A310). They aren't
# equipment tags. Block them by default. Override with `keep_room_labels=True`
# if a project genuinely uses A100-series for equipment.
ROOM_LABEL_RE = re.compile(r'^[A-Z]\d{3}$')


def _is_noise_tag(tag):
    """Return True if the tag-shaped string is one of the known false
    positives we always want to reject from the text layer."""
    if not tag:
        return True
    if tag in BANNED_TAG_PREFIXES:
        return True
    if REFRIGERANT_PATTERN.match(tag):
        return True
    if SHEET_REF_RE.match(tag):
        return True
    if ROOM_LABEL_RE.match(tag):
        return True
    prefix_m = re.match(r'^([A-Z]+)', tag)
    if prefix_m and prefix_m.group(1) in CODE_PREFIXES:
        return True
    return False


def _extract_page_tags(text, min_occurrence=1):
    """Return Counter-like dict {tag: occurrence_count} for one page."""
    if not text:
        return {}
    counts = {}
    for m in TEXT_TAG_PATTERN.finditer(text):
        raw = m.group(1)
        normed = normalize_tag(raw)
        if not normed:
            continue
        if _is_noise_tag(normed):
            continue
        counts[normed] = counts.get(normed, 0) + 1
    return counts


def extract_tags_from_text_layer(pdf_path, verbose=False):
    """Walk every page, regex-pull tag-shaped tokens, return TagVariable list.

    Each unique (tag, first_page_seen) becomes one TagVariable. The
    `properties` dict carries `text_occurrences` (total count across the PDF)
    and `pages_seen` (comma-separated 1-indexed page list) so downstream
    callers can dedupe / compare against OCR or pdfplumber output.
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    per_tag_pages = OrderedDict()  # tag → {page_idx: count}
    for pi in range(page_count):
        try:
            text = doc[pi].get_text() or ''
        except Exception as e:
            if verbose:
                print(f"[text-layer] page {pi+1} read failed: {e}",
                      file=sys.stderr)
            continue
        page_counts = _extract_page_tags(text)
        if verbose:
            print(f"[text-layer] page {pi+1}: {len(text)} chars, "
                  f"{len(page_counts)} unique tags, "
                  f"{sum(page_counts.values())} occurrences")
        for tag, n in page_counts.items():
            slot = per_tag_pages.setdefault(tag, {})
            slot[pi + 1] = slot.get(pi + 1, 0) + n
    doc.close()

    variables = []
    for idx, (tag, pages) in enumerate(per_tag_pages.items()):
        first_page = min(pages.keys())
        total = sum(pages.values())
        page_list = ','.join(str(p) for p in sorted(pages.keys()))
        variables.append({
            'tag': tag,
            'schedule_name': f"TEXT LAYER",
            'page': first_page,
            'properties': {
                'text_occurrences': total,
                'pages_seen': page_list,
            },
            'inferred_yolo_class': _infer_class_from_tag(tag) or 'UNKNOWN',
            'source_row_index': idx,
            'extracted_via': 'text_layer',
        })
    return variables


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    print(f"PDF: {pdf.name}")
    vars_ = extract_tags_from_text_layer(str(pdf), verbose=args.verbose)
    print(f"\n=== {len(vars_)} unique tags from text layer ===\n")

    # Group by inferred class
    by_class = {}
    for v in vars_:
        by_class.setdefault(v['inferred_yolo_class'], []).append(v)
    for cls in sorted(by_class):
        rows = by_class[cls]
        total = sum(v['properties']['text_occurrences'] for v in rows)
        print(f"{cls:<25}  {len(rows):>3} tags, {total:>4} occurrences")
        for v in rows[:10]:
            occ = v['properties']['text_occurrences']
            pages = v['properties']['pages_seen']
            print(f"    {v['tag']:<12} occ={occ:<3} pages={pages}")
        if len(rows) > 10:
            print(f"    ... and {len(rows)-10} more")
