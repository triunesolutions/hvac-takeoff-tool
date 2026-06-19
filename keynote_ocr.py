"""
keynote_ocr.py — OCR fallback for raster-only keynote blocks.

Mirrors schedule_ocr.py but for keynote text. Used when extract_keynotes_from_page()
finds no text-layer keynotes on a page that visually has them (CAD-export PDFs).
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz


# Re-use the rest of the readers' state
_reader = None
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


KEYNOTE_HEADER_KEYWORDS = ('KEYNOTES', 'KEYED NOTES', 'GENERAL NOTES',
                           'MECHANICAL NOTES', 'SHEET NOTES')

# A keynote line OCR'd usually looks like: "1.  THIS IS THE NOTE TEXT..."
# We use a forgiving regex tolerant of OCR noise.
NOTE_LINE_PATTERN = re.compile(r'^[\s\W]*(\d{1,3})\s*[\.\)\]:]\s+(.+)$')


def render_page(pdf_path: Path, page_index: int, dpi: int = 300):
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


def find_keynote_block_region(ocr_items: list[dict]) -> dict | None:
    """Find a region where a KEYNOTE-like header appears. Returns the header bbox
    in pixel space, or None if no obvious block."""
    for it in ocr_items:
        upper = it['text'].upper()
        if any(kw in upper for kw in KEYNOTE_HEADER_KEYWORDS):
            return it
    return None


def ocr_page(img) -> list[dict]:
    reader = _get_reader()
    raw = reader.readtext(img, detail=1, paragraph=True)  # paragraph=True groups multi-line
    out = []
    for box, text, conf in raw:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        out.append({
            'text': text.strip(),
            'bbox': (min(xs), min(ys), max(xs), max(ys)),
            'cx': (min(xs) + max(xs)) / 2,
            'cy': (min(ys) + max(ys)) / 2,
            'conf': float(conf) if conf is not None else 0.0,
        })
    return out


def extract_keynotes_ocr(pdf_path: Path, page_index: int, dpi: int = 300) -> list[dict]:
    """Run OCR on a page and try to recover numbered keynotes."""
    img = render_page(pdf_path, page_index, dpi=dpi)
    items = ocr_page(img)
    if not items:
        return []

    header = find_keynote_block_region(items)
    # Items BELOW the header are candidate notes (or all items if no header found)
    if header:
        body = [it for it in items if it['cy'] >= header['cy']]
    else:
        body = items

    notes = []
    seen = set()
    for it in sorted(body, key=lambda d: d['cy']):
        text = it['text']
        # The paragraph mode may concatenate "1. foo 2. bar" — split on number prefixes
        for chunk in re.split(r'(?=^\s*\d{1,3}[\.\)]\s)', text, flags=re.MULTILINE):
            chunk = chunk.strip()
            m = NOTE_LINE_PATTERN.match(chunk)
            if not m:
                continue
            num = int(m.group(1))
            body_text = m.group(2).strip()
            if num <= 0 or num > 100 or num in seen or len(body_text) < 5:
                continue
            seen.add(num)
            notes.append({
                'number': num,
                'text': body_text,
                'page': page_index + 1,
                '_source': 'ocr',
            })
    return notes


def extract_all_keynotes_ocr(pdf_path: Path, page_nums: list[int],
                            dpi: int = 300) -> list[dict]:
    out = []
    for p in page_nums:
        try:
            out.extend(extract_keynotes_ocr(pdf_path, p - 1, dpi=dpi))
        except Exception as e:
            print(f'[keynote_ocr] page {p}: {e}')
    return out


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--page', type=int, required=True)
    args = ap.parse_args()
    notes = extract_keynotes_ocr(Path(args.pdf), args.page - 1)
    print(f'Found {len(notes)} keynotes on page {args.page}:')
    for n in notes:
        print(f'  {n["number"]:>3d}: {n["text"][:80]}')
