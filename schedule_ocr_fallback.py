"""
Schedule-page OCR fallback.

Some projects (Larchmont, Shamrock, Yucaipa A, Anaheim, Krispy Kreme, ...) have
schedule tables that are rasterized — pdfplumber's text/table extraction returns
nothing. This module renders those pages, OCRs them, and recovers a tag list so
downstream strict-mode tag counting still works.

Output schema matches `parse_pdf_schedules` variables: a list of TagVariable
dicts. Property cells are NOT recovered here — only the tag string + page +
inferred class. That's enough for strict-mode bubble matching; richer property
extraction can be layered on later.

Usage:
    from schedule_ocr_fallback import ocr_schedule_fallback
    extra_vars = ocr_schedule_fallback(pdf_path, dpi=200)
"""
import re
import sys
from collections import OrderedDict

import fitz
import numpy as np
import cv2

from tag_inference import _infer_class_from_tag, TAG_PREFIX_CLASS  # noqa: F401
from schedule_parser import normalize_tag

# Lazy-init RapidOCR (PaddleOCR PP-OCR models via ONNXRuntime — no
# paddlepaddle dependency, works on Python 3.14). A/B test on Yucaipa A
# page 4: RapidOCR returned cleaner reads than EasyOCR (DU-01A vs DU-01b,
# EF-01 vs EF-0). ~2× slower but worth it for schedule pages where every
# character matters.
_rapidocr_reader = None


def get_rapidocr():
    global _rapidocr_reader
    if _rapidocr_reader is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapidocr_reader = RapidOCR()
    return _rapidocr_reader

# Pages whose text layer contains any of these are good OCR-fallback candidates.
SCHEDULE_TEXT_HINTS = (
    'SCHEDULE', 'EQUIPMENT', 'DEVICE LIST', 'MARK', 'TAG',
)

# Skip pages dominated by these markers — they're plans, not schedule sheets.
PLAN_HINTS = (
    'MECHANICAL PLAN', 'FLOOR PLAN', 'CEILING PLAN', 'ROOF PLAN',
    'VENTILATION PLAN', 'PIPING PLAN',
)

# OCR-token shape that *might* be a tag. Looser than schedule_parser.TAG_REGEX
# because OCR introduces noise (lowercase, stray punctuation). normalize_tag()
# is the strict gate after this.
OCR_TAG_SHAPE = re.compile(r'^[A-Za-z]{1,5}-?\d{1,3}[A-Za-z]?$')


def _render_page_bgr(pdf_path, page_idx, dpi=200):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def _page_text_upper(pdf_path, page_idx):
    doc = fitz.open(pdf_path)
    try:
        t = doc[page_idx].get_text() or ''
    finally:
        doc.close()
    return t.upper()


def find_candidate_pages(pdf_path, max_pages=None):
    """Pages worth OCRing for schedules. Returns list of (page_idx, reason)."""
    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()
    if max_pages is not None:
        total = min(total, max_pages)

    candidates = []
    for pi in range(total):
        text_u = _page_text_upper(pdf_path, pi)
        # Empty text layer → raster page → always a candidate
        if len(text_u.strip()) < 50:
            candidates.append((pi, 'empty-text-layer'))
            continue
        # Skip plan sheets — they have schedule words too but aren't schedules
        if any(p in text_u for p in PLAN_HINTS):
            continue
        if any(h in text_u for h in SCHEDULE_TEXT_HINTS):
            candidates.append((pi, 'schedule-hint'))
    return candidates


def _ocr_full_page(img, conf_threshold=0.3):
    """Run RapidOCR (PaddleOCR PP-OCR via ONNX) on the full page.
    Returns list of {text, conf}."""
    reader = get_rapidocr()
    res, _elapsed = reader(img)
    out = []
    if not res:
        return out
    for entry in res:
        try:
            text, conf = str(entry[1]), float(entry[2])
        except Exception:
            continue
        if conf < conf_threshold:
            continue
        out.append({'text': text.strip(), 'conf': conf})
    return out


def _tokenize(text):
    """Split an OCR span into individual tokens that *might* be tags.

    OCR often returns a whole row glued together ('CU-1 CARRIER 40RUQA12 3650').
    Split on whitespace and common cell separators.
    """
    if not text:
        return []
    # Replace common separators with space, then split
    cleaned = re.sub(r'[|/,;:\\]', ' ', text)
    return [t.strip(' .()[]') for t in cleaned.split() if t.strip(' .()[]')]


def extract_tags_from_ocr(ocr_words, min_conf=0.3):
    """Filter OCR output to plausible tag strings → normalized tags.

    NOTE: We deliberately do NOT gate on TAG_PREFIX_CLASS here. Schedule pages
    define the project's tag universe (DU, IDU, KEF, DBF, etc. all appear in
    real projects but aren't in our static whitelist). Permissive at parse,
    strict at plan-match. False positives like sheet refs (M101) and room
    labels (X171) ARE accepted here, but they'll be filtered out downstream
    when bubble OCR on plan pages matches against this captured list — the
    sheet refs simply won't reappear in tag positions on the plan.
    """
    tags = []
    for w in ocr_words:
        if w['conf'] < min_conf:
            continue
        for tok in _tokenize(w['text']):
            if not OCR_TAG_SHAPE.match(tok):
                continue
            normed = normalize_tag(tok)
            if not normed:
                continue
            tags.append((normed, w['conf']))
    return tags


def ocr_schedule_fallback(pdf_path, dpi=200, max_pages=None, verbose=False):
    """Run OCR on candidate schedule pages and return TagVariable list.

    Returns: list of {tag, schedule_name, page, properties, inferred_yolo_class,
                       source_row_index}.
    """
    candidates = find_candidate_pages(pdf_path, max_pages=max_pages)
    if verbose:
        print(f"[ocr-fallback] {len(candidates)} candidate page(s): "
              f"{[(p+1, r) for p, r in candidates]}")

    by_tag_page = OrderedDict()  # (tag, page) → (best_conf, schedule_name)
    for page_idx, reason in candidates:
        try:
            img = _render_page_bgr(pdf_path, page_idx, dpi=dpi)
        except Exception as e:
            if verbose:
                print(f"[ocr-fallback] page {page_idx+1} render failed: {e}",
                      file=sys.stderr)
            continue

        try:
            words = _ocr_full_page(img)
        except Exception as e:
            if verbose:
                print(f"[ocr-fallback] page {page_idx+1} OCR failed: {e}",
                      file=sys.stderr)
            continue

        page_tags = extract_tags_from_ocr(words)
        if verbose:
            print(f"[ocr-fallback] page {page_idx+1} ({reason}): "
                  f"{len(words)} words, {len(page_tags)} tag matches")

        # Use a friendly schedule_name pulled from any OCR text containing
        # "SCHEDULE" — fall back to a generic label.
        schedule_name = f"OCR FALLBACK p{page_idx+1}"
        for w in words:
            txt = w['text'].upper()
            if 'SCHEDULE' in txt and len(txt) < 80:
                schedule_name = ' '.join(txt.split())
                break

        for tag, conf in page_tags:
            key = (tag, page_idx + 1)
            prev = by_tag_page.get(key)
            if prev is None or conf > prev[0]:
                by_tag_page[key] = (conf, schedule_name)

    variables = []
    for idx, ((tag, page), (conf, schedule_name)) in enumerate(by_tag_page.items()):
        variables.append({
            'tag': tag,
            'schedule_name': schedule_name,
            'page': page,
            'properties': {},
            'inferred_yolo_class': _infer_class_from_tag(tag) or 'UNKNOWN',
            'source_row_index': idx,
            'extracted_via': 'ocr_fallback',
            'ocr_confidence': round(conf, 3),
        })
    return variables


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--max-pages', type=int, default=None)
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    vars_ = ocr_schedule_fallback(args.pdf, dpi=args.dpi,
                                   max_pages=args.max_pages, verbose=args.verbose)
    print(f"\n=== {len(vars_)} variables recovered via OCR fallback ===")
    by_page = {}
    for v in vars_:
        by_page.setdefault(v['page'], []).append(v)
    for page in sorted(by_page):
        print(f"\nPage {page} — {by_page[page][0]['schedule_name']}:")
        for v in by_page[page]:
            print(f"  {v['tag']:<12} [{v['inferred_yolo_class']}] "
                  f"conf={v['ocr_confidence']}")
