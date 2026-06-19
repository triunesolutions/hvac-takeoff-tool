"""
room_counter.py — Stage 13.

Group AI detections by the room they sit in, producing per-room equipment
breakdowns ("Conference 302: 4 diffusers, 1 damper").

Approach:
  1. OCR every plan page once at moderate DPI to get room labels with bboxes.
     Room labels look like "302", "ROOM 302", "CONFERENCE 302", "OFFICE",
     "BREAK ROOM", etc.
  2. For each detection, find the nearest room label (by Euclidean distance
     between centers) and assign to that room.
  3. Aggregate by room → {class: count}.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict, Counter

import fitz  # PyMuPDF


# Pattern for "ROOM 302", "302", "RM 302", "302A". Bare numbers must be
# exactly 3 digits (100-999) to avoid catching CFM values and dimensions.
ROOM_PATTERNS = [
    re.compile(r'^\s*(?:ROOM|RM|R)\.?\s*(\d{3}[A-Z]?)\s*$', re.I),
    re.compile(r'^\s*(\d{3}[A-Z]?)\s*$'),  # bare 3-digit + optional letter
]

# Room-name words that, if present, mean this is a room label
ROOM_NAME_WORDS = {
    'OFFICE', 'CONFERENCE', 'LOBBY', 'KITCHEN', 'BREAK', 'STORAGE',
    'RESTROOM', 'TOILET', 'CORRIDOR', 'STAIR', 'ELEVATOR',
    'JANITOR', 'CLOSET', 'PATIENT',
    'EXAM', 'PROCEDURE', 'OPERATING', 'RECOVERY', 'WAITING',
    'NURSE', 'DOCTOR', 'LAB', 'X-RAY', 'PRE-OP', 'POST-OP',
    'SOILED', 'CLEAN', 'STERILE',
}

# Words that, if present, mean this is NOT a room label (notes, instructions, etc.)
NOT_ROOM_WORDS = {
    'SHALL', 'PROVIDE', 'INSTALL', 'CONTRACTOR', 'EQUIPMENT', 'NOTE',
    'TYPICAL', 'MAIN', 'CFM', 'DUCT', 'AIR', 'SYSTEM', 'CALIFORNIA',
    'CODE', 'PROVIDED', 'PER', 'WITH', 'AT', 'BE', 'OF', 'THE',
    'LOW-LOSS', 'TAG', 'TAGGED', 'MOUNTED', 'DAMPER', 'DAMPERS',
    'VENT', 'EXHAUST', 'RETURN', 'SUPPLY',  # these are equipment-flow words
}


_easyocr_reader = None
def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


def render_page_for_ocr(pdf_path: Path, page_index: int, dpi: int = 150):
    """Render a page to RGB numpy at moderate DPI for OCR."""
    import numpy as np
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi, annots=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        return img
    finally:
        doc.close()


def _is_room_label(text: str) -> tuple[bool, str]:
    """Decide if text looks like a room label. Returns (yes, normalized_label)."""
    s = text.strip().upper()
    if not s:
        return False, ''
    # Reject anything that contains note-like words
    words_in_s = set(re.findall(r'[A-Z][A-Z\-/]*', s))
    if words_in_s & NOT_ROOM_WORDS:
        return False, ''
    # Number patterns
    for pat in ROOM_PATTERNS:
        m = pat.match(s)
        if m:
            return True, m.group(1).upper()
    # Room-name words — must include at least one AND be short (1-3 words)
    if any(w in words_in_s for w in ROOM_NAME_WORDS) and len(s.split()) <= 3 and len(s) < 30:
        return True, s
    return False, ''


def find_rooms_on_page(pdf_path: Path, page_index: int,
                       ocr_dpi: int = 150) -> list[dict]:
    """Return list of {label, cx_px, cy_px, bbox_px} in the OCR-DPI pixel space.
    Caller scales to detection DPI if different."""
    img = render_page_for_ocr(pdf_path, page_index, ocr_dpi)
    reader = _get_reader()
    raw = reader.readtext(img, detail=1, paragraph=False)
    rooms = []
    for box, text, conf in raw:
        if conf < 0.3:
            continue
        is_room, label = _is_room_label(text)
        if not is_room:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        rooms.append({
            'label': label,
            'cx_px': (min(xs) + max(xs)) / 2,
            'cy_px': (min(ys) + max(ys)) / 2,
            'bbox_px': (min(xs), min(ys), max(xs), max(ys)),
            'conf': float(conf),
            'ocr_dpi': ocr_dpi,
        })
    return rooms


def assign_detections_to_rooms(detections: dict, rooms_by_page: dict[int, list[dict]],
                              detections_dpi: int = 200,
                              ocr_dpi: int = 150) -> dict:
    """For each detection, attach 'room' field with the nearest room label.
    Mutates detections in place. Returns aggregate breakdown.
    """
    scale = detections_dpi / ocr_dpi  # detection coords -> ocr-dpi coords if dpis differ
    breakdown: dict[str, Counter] = defaultdict(Counter)

    for pkey, det_list in detections.get('pages', {}).items():
        pno = int(pkey)
        rooms = rooms_by_page.get(pno, [])
        if not rooms:
            for det in det_list:
                det['room'] = None
                breakdown['(unassigned)'][det.get('cls', '?')] += 1
            continue
        for det in det_list:
            dcx = (det['x1'] + det['x2']) / 2 / scale
            dcy = (det['y1'] + det['y2']) / 2 / scale
            best_d, best = 1e9, None
            for r in rooms:
                d = ((dcx - r['cx_px']) ** 2 + (dcy - r['cy_px']) ** 2) ** 0.5
                if d < best_d:
                    best_d, best = d, r
            label = best['label'] if best else None
            det['room'] = label
            key = label or '(unassigned)'
            breakdown[key][det.get('cls', '?')] += 1

    # Convert Counter -> plain dict for JSON
    return {room: dict(counts) for room, counts in breakdown.items()}


def per_room_counts(pdf_path: Path, detections: dict,
                   plan_page_nums: list[int],
                   ocr_dpi: int = 150,
                   detections_dpi: int = 200) -> dict:
    """Top-level: OCR rooms on each plan page, group detections by room."""
    rooms_by_page: dict[int, list[dict]] = {}
    for p in plan_page_nums:
        try:
            rooms_by_page[p] = find_rooms_on_page(pdf_path, p - 1, ocr_dpi=ocr_dpi)
        except Exception as e:
            print(f'[room_counter] page {p}: {e}')
            rooms_by_page[p] = []
    breakdown = assign_detections_to_rooms(detections, rooms_by_page,
                                          detections_dpi=detections_dpi,
                                          ocr_dpi=ocr_dpi)
    return {
        'rooms_by_page': rooms_by_page,
        'breakdown': breakdown,
        'n_rooms_found': sum(len(v) for v in rooms_by_page.values()),
    }


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--detections', required=True)
    ap.add_argument('--plan-pages', nargs='+', type=int, required=True)
    args = ap.parse_args()

    dets = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    result = per_room_counts(Path(args.pdf), dets, args.plan_pages)
    print(f'Rooms found: {result["n_rooms_found"]}')
    print(f'Breakdown:')
    for room, counts in result['breakdown'].items():
        total = sum(counts.values())
        details = ', '.join(f'{n}×{c}' for c, n in counts.items())
        print(f'  {room:25s} {total:3d}  ({details})')
