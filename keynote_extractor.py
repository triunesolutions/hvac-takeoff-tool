"""
keynote_extractor.py — Stage 5 of the upload pipeline.

Per Deck 1 slides 3 and 12, equipment details often live ONLY in keynotes,
not the schedule. This module:

  1. Finds the "KEYNOTES" / "GENERAL NOTES" / "KEYED NOTES" block on each page.
  2. Extracts every numbered note (1., 2., 3. or 1) 2) 3) etc.).
  3. Finds keynote callout markers on plan pages (numbers in circles, brackets,
     hexagons — drawn as text in the PDF).
  4. Cross-references: each detection on a plan can be linked to nearby
     keynote callouts.

This is text-layer based. Raster-only keynote blocks would need OCR fallback
analogous to schedule_ocr.py (not built yet — flag for future).
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


KEYNOTE_HEADER_PATTERNS = [
    re.compile(r'\bKEY(ED)?\s*NOTES?\b', re.I),
    re.compile(r'\bGENERAL\s+NOTES?\b', re.I),
    re.compile(r'\bMECHANICAL\s+NOTES?\b', re.I),
    re.compile(r'\bSHEET\s+NOTES?\b', re.I),
]

# Patterns for "N." or "N)" or "(N)" at start of a line, followed by text
NOTE_LINE_PATTERN = re.compile(r'^\s*[\(\[]?(\d{1,3})[\)\.\]:]\s+(.+)$')

# Callout markers on the plan — text like "<1>", "(1)", "[2]", or just a number
# in a circled symbol. We can only catch the text-layer ones reliably.
CALLOUT_PATTERN = re.compile(r'^[<(\[]?\s*(\d{1,3})\s*[>)\]]?$')


def extract_keynotes_from_page(pdf_path: Path, page_index: int) -> list[dict]:
    """Extract numbered notes from this page. Returns list of {number, text}."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        text = page.get_text('text') or ''
    finally:
        doc.close()

    # Find a keynote header position; collect notes that appear AFTER it
    header_match = None
    for pat in KEYNOTE_HEADER_PATTERNS:
        m = pat.search(text)
        if m:
            header_match = m
            break

    if header_match:
        # Everything after the header
        body = text[header_match.end():]
    else:
        # No header — try the whole text anyway (some sheets just have numbered list)
        body = text

    notes = []
    seen_numbers = set()
    for line in body.split('\n'):
        m = NOTE_LINE_PATTERN.match(line)
        if not m:
            continue
        num = int(m.group(1))
        body_text = m.group(2).strip()
        # Avoid duplicates and unreasonable note numbers
        if num in seen_numbers or num > 100 or len(body_text) < 3:
            continue
        seen_numbers.add(num)
        notes.append({
            'number': num,
            'text': body_text,
            'page': page_index + 1,
        })
    return notes


def find_callouts_on_page(pdf_path: Path, page_index: int) -> list[dict]:
    """Find numeric callout references on this page. Returns list with bbox."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        words = page.get_text('words')  # (x0, y0, x1, y1, text, ...)
    finally:
        doc.close()

    callouts = []
    for w in words:
        text = (w[4] or '').strip()
        m = CALLOUT_PATTERN.match(text)
        if not m:
            continue
        num = int(m.group(1))
        if num <= 0 or num > 100:
            continue
        callouts.append({
            'number': num,
            'page': page_index + 1,
            'bbox_pdf': (w[0], w[1], w[2], w[3]),
            'cx_pdf': (w[0] + w[2]) / 2,
            'cy_pdf': (w[1] + w[3]) / 2,
        })
    return callouts


def extract_all_keynotes(pdf_path: Path) -> dict:
    """Walk every page; build a global keynote map + per-page callouts list."""
    doc = fitz.open(str(pdf_path))
    n_pages = doc.page_count
    doc.close()

    notes_by_number: dict[int, dict] = {}
    callouts_by_page: dict[int, list[dict]] = {}
    for i in range(n_pages):
        notes = extract_keynotes_from_page(pdf_path, i)
        for n in notes:
            # Last note wins if same number appears on multiple pages
            notes_by_number[n['number']] = n
        callouts = find_callouts_on_page(pdf_path, i)
        if callouts:
            callouts_by_page[i + 1] = callouts

    # Compute discrepancies (Deck 1 slide 12: keynotes mentioned but not placed)
    referenced_numbers = set()
    for callouts in callouts_by_page.values():
        for c in callouts:
            referenced_numbers.add(c['number'])
    defined_numbers = set(notes_by_number.keys())

    unreferenced = sorted(defined_numbers - referenced_numbers)
    undefined_refs = sorted(referenced_numbers - defined_numbers)

    return {
        'notes': notes_by_number,
        'callouts_by_page': callouts_by_page,
        'unreferenced_notes': unreferenced,        # defined but no callout on plan
        'undefined_callouts': undefined_refs,      # callouts that point to non-existent notes
        'total_notes': len(notes_by_number),
        'total_callouts': sum(len(c) for c in callouts_by_page.values()),
    }


def link_detections_to_keynotes(detections: dict, keynotes: dict,
                                page_to_dpi: float = 72.0/200) -> dict:
    """For each detection, find the nearest keynote callout (if any) within ~200 px.
    detections: the detections.json shape {pages: {pno_str: [det, ...]}}
    keynotes:   output of extract_all_keynotes()
    page_to_dpi: conversion from PDF points -> detection-pixel coords.
                 default assumes detections rendered at 200 DPI (point/0.36 = px).
    Mutates detections dict in place.
    """
    PROX_PX = 200  # ~1 inch at 200 DPI

    for pkey, det_list in detections.get('pages', {}).items():
        pno = int(pkey)
        callouts = keynotes['callouts_by_page'].get(pno, [])
        if not callouts:
            continue
        # Convert callout positions to detection pixel space
        callouts_px = []
        for c in callouts:
            cx_px = c['cx_pdf'] / page_to_dpi
            cy_px = c['cy_pdf'] / page_to_dpi
            callouts_px.append((cx_px, cy_px, c['number']))

        for det in det_list:
            dcx = (det['x1'] + det['x2']) / 2
            dcy = (det['y1'] + det['y2']) / 2
            best_d, best_num = 1e9, None
            for (cx, cy, num) in callouts_px:
                d = ((dcx - cx) ** 2 + (dcy - cy) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_num = d, num
            if best_num is not None and best_d <= PROX_PX:
                det['keynote_number'] = best_num
                note = keynotes['notes'].get(best_num)
                if note:
                    det['keynote_text'] = note['text'][:200]
    return detections


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--out', help='Output JSON file')
    args = ap.parse_args()

    result = extract_all_keynotes(Path(args.pdf))
    print(f'Found {result["total_notes"]} notes and {result["total_callouts"]} callouts')
    print(f'Unreferenced notes (defined but no callout): {result["unreferenced_notes"]}')
    print(f'Undefined callouts (callout to non-existent note): {result["undefined_callouts"]}')
    print()
    if result['notes']:
        print('Sample notes:')
        for num, note in list(result['notes'].items())[:5]:
            print(f'  {num:3d}: {note["text"][:80]}')

    if args.out:
        # JSON can't have int keys for dicts; convert
        out = {
            'notes': {str(k): v for k, v in result['notes'].items()},
            'callouts_by_page': {str(k): v for k, v in result['callouts_by_page'].items()},
            'unreferenced_notes': result['unreferenced_notes'],
            'undefined_callouts': result['undefined_callouts'],
            'total_notes': result['total_notes'],
            'total_callouts': result['total_callouts'],
        }
        Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'Wrote {args.out}')
