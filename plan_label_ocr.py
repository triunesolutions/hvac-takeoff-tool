"""
plan_label_ocr.py — OCR-based per-instance plan-label extraction.

Mirrors diffuser_extractor.py (which uses the PDF text layer) but works on
raster-only plans like Art Vascular. Strategy:

  1. For each YOLO detection bbox on a plan page, render a tight crop at
     high DPI (300 or higher).
  2. EasyOCR the crop.
  3. Look for mark + neck + CFM patterns in the OCR'd text:
       * Mark      :  [A-Z]{1,2}\\d{1,3}[A-Z]?    (S1, A-12, RG-2, FCU-1)
       * Neck size :  \\d+(?:\\.\\d+)?\\"   or   \\d+ ?[/xX] ?\\d+
       * CFM       :  any 2-4 digit integer near the mark/size
  4. Return per-detection records compatible with diffuser_extractor output
     so the unified tag_report can consume them.

This is slower than text-layer extraction but works on any PDF.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


# Lazy EasyOCR
_reader = None
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


# Patterns — re-use the diffuser_extractor conventions
MARK_PATTERN = re.compile(r'^[A-Z]{1,3}-?\d{1,3}[A-Z]?$', re.I)
NECK_ROUND_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)["\'”]?$')          # 8", 10.5", 12"
NECK_RECT_PATTERN  = re.compile(r'^(\d+(?:\.\d+)?)\s*[xX/]\s*(\d+(?:\.\d+)?)$')  # 12x12, 24/18
CFM_PATTERN = re.compile(r'^(\d{2,4})$')                                     # 50, 100, 1200

# How much padding (in PDF points) to add around each detection's bbox
# when rendering the OCR crop. ~150 pts = ~2 inches in real-world.
CROP_PADDING_PT = 150


def _render_crop(pdf_path: Path, page_index: int, bbox_display_pt: tuple,
                 dpi: int = 300):
    """Render a rectangular crop of a page at high DPI. bbox in display points."""
    import numpy as np
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        x1, y1, x2, y2 = bbox_display_pt
        # Clip to page bounds
        rect = fitz.Rect(
            max(0, x1), max(0, y1),
            min(page.rect.width, x2), min(page.rect.height, y2),
        )
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, clip=rect, annots=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        return img
    finally:
        doc.close()


def _normalize_neck(text: str) -> str | None:
    """Return canonical neck size string ('round:8' / 'rect:12x12') or None."""
    s = text.strip().replace(' ', '')
    m = NECK_ROUND_PATTERN.match(s)
    if m:
        v = float(m.group(1))
        return f'round:{int(v) if v == int(v) else v}'
    m = NECK_RECT_PATTERN.match(s)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        def fmt(x): return int(x) if x == int(x) else x
        return f'rect:{fmt(a)}x{fmt(b)}'
    return None


def _parse_ocr_for_label(ocr_words: list[dict]) -> dict:
    """Given OCR words from a crop, look for mark + neck + CFM."""
    mark = neck_canon = None
    cfm = None

    for w in ocr_words:
        text = w['text'].strip()
        # Normalize quoted variants
        text_clean = text.replace('"', '"').replace('"', '"')
        if not mark and MARK_PATTERN.match(text):
            mark = text.upper()
            continue
        if not neck_canon:
            n = _normalize_neck(text_clean)
            if n:
                neck_canon = n
                continue
        if cfm is None:
            m = CFM_PATTERN.match(text)
            if m:
                val = int(m.group(1))
                # CFM range plausibility: 30 - 9999
                if 30 <= val <= 9999:
                    cfm = val
                    continue
    return {'mark': mark, 'neck_size_canon': neck_canon, 'cfm': cfm}


def _ocr_image(img) -> list[dict]:
    reader = _get_reader()
    raw = reader.readtext(img, detail=1, paragraph=False)
    out = []
    for box, text, conf in raw:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        out.append({
            'text': text.strip(),
            'bbox_px': (min(xs), min(ys), max(xs), max(ys)),
            'conf': float(conf or 0),
        })
    return out


def extract_plan_labels_via_ocr(
    pdf_path: Path,
    detections: dict,
    plan_pages_1based: list[int],
    dpi: int = 300,
    skip_if_low_conf: float = 0.3,
) -> list[dict]:
    """For each YOLO detection on a plan page, OCR a tight crop and try to
    extract the mark + neck size + CFM. Returns a list of records shaped like
    diffuser_extractor.extract_diffuser_instances output, so the tag_report
    can consume them transparently.
    """
    instances = []
    detections_dpi = detections.get('dpi', 200)
    px_to_pt = 72.0 / detections_dpi

    for pkey, det_list in detections.get('pages', {}).items():
        pno = int(pkey)
        if pno not in plan_pages_1based:
            continue
        for di, det in enumerate(det_list):
            # detection bbox is in pixel space at detections_dpi
            x1_pt = det['x1'] * px_to_pt - CROP_PADDING_PT
            y1_pt = det['y1'] * px_to_pt - CROP_PADDING_PT
            x2_pt = det['x2'] * px_to_pt + CROP_PADDING_PT
            y2_pt = det['y2'] * px_to_pt + CROP_PADDING_PT

            try:
                img = _render_crop(pdf_path, pno - 1, (x1_pt, y1_pt, x2_pt, y2_pt), dpi=dpi)
            except Exception as e:
                continue
            words = _ocr_image(img)
            # Drop low-confidence words
            words = [w for w in words if w['conf'] >= skip_if_low_conf]
            if not words:
                continue
            parsed = _parse_ocr_for_label(words)
            if parsed['mark'] or parsed['neck_size_canon'] or parsed['cfm']:
                instances.append({
                    'mark': parsed['mark'],
                    'neck_size_canon': parsed['neck_size_canon'],
                    'cfm': parsed['cfm'],
                    'page': pno,
                    'detection_class': det.get('cls'),
                    'detection_index': di,
                    '_source': 'ocr',
                    '_ocr_confidence': sum(w['conf'] for w in words) / max(1, len(words)),
                })
    return instances


def merge_with_diffuser_extractor(text_instances: list[dict],
                                  ocr_instances: list[dict]) -> list[dict]:
    """Prefer text-layer instances; supplement with OCR for detections without
    a text-layer match. Dedupe by approximate position."""
    out = list(text_instances)
    # OCR instances are per-detection; only keep those whose detection wasn't
    # already covered by a text-layer extract on the same page.
    text_pages = {i.get('page') for i in text_instances}
    for ocr in ocr_instances:
        # If the page has text-layer instances, skip OCR for the same page
        # unless the OCR found a mark that the text-layer didn't.
        if ocr.get('page') in text_pages:
            existing = [t for t in text_instances if t.get('page') == ocr['page']
                       and t.get('mark') == ocr.get('mark')]
            if existing:
                continue
        out.append(ocr)
    return out


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--detections', required=True)
    ap.add_argument('--plan-pages', nargs='+', type=int, required=True)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--out')
    args = ap.parse_args()

    dets = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    instances = extract_plan_labels_via_ocr(
        Path(args.pdf), dets, args.plan_pages, dpi=args.dpi,
    )
    print(f'Found {len(instances)} OCR-extracted plan label(s)')
    print()
    print(f'{"page":>4} {"det":>3} {"mark":<10} {"neck":<14} {"cfm":>5} {"class":<25}')
    for inst in instances[:30]:
        print(f'{inst["page"]:>4} {inst.get("detection_index", "-"):>3} '
              f'{(inst.get("mark") or "-"):<10} '
              f'{(inst.get("neck_size_canon") or "-"):<14} '
              f'{(inst.get("cfm") or "-"):>5} '
              f'{(inst.get("detection_class") or "-"):<25}')
    if args.out:
        Path(args.out).write_text(json.dumps(instances, indent=2), encoding='utf-8')
        print(f'\nWrote {args.out}')
