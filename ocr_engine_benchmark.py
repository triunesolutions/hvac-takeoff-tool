"""
OCR engine A/B test for raster schedule pages.

Runs EasyOCR, PaddleOCR, and Surya on the same rendered page image and reports
how many plausible tag strings each one finds. Goal: pick the best engine
before re-tuning `schedule_ocr_fallback.py`.

Usage:
    python ocr_engine_benchmark.py <pdf> [--page N] [--dpi 300] [--engines easy,paddle,surya]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import fitz
import numpy as np
import cv2

from tag_inference import TAG_PREFIX_CLASS
from schedule_parser import normalize_tag
from schedule_ocr_fallback import OCR_TAG_SHAPE, _tokenize


def render_page(pdf_path, page_idx, dpi=300):
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


def filter_tags(words):
    """words = list of (text, conf). Returns set of normalized tags."""
    tags = set()
    raw_tokens = []
    for text, conf in words:
        for tok in _tokenize(text):
            raw_tokens.append((tok, conf))
            if not OCR_TAG_SHAPE.match(tok):
                continue
            normed = normalize_tag(tok)
            if not normed:
                continue
            prefix_m = re.match(r'^([A-Z]+)', normed)
            prefix = prefix_m.group(1) if prefix_m else ''
            if prefix and prefix in TAG_PREFIX_CLASS:
                tags.add(normed)
            elif len(normed) == 1 and normed in 'ABCDEF':
                tags.add(normed)
    return tags, raw_tokens


def run_easyocr(img):
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    out = reader.readtext(img)
    return [(str(t).strip(), float(c)) for _b, t, c in out if c >= 0.3]


def run_paddleocr(img):
    # paddlepaddle has no Python 3.14 wheel. Use RapidOCR which ships
    # PaddleOCR's PP-OCR models via ONNXRuntime — same recognition model,
    # different runtime.
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    res, _elapsed = ocr(img)
    words = []
    if not res:
        return words
    for entry in res:
        # entry = [box, text, conf]
        try:
            text, conf = str(entry[1]), float(entry[2])
        except Exception:
            continue
        if conf >= 0.3:
            words.append((text.strip(), conf))
    return words


def run_surya(img):
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()
    preds = rec_predictor([pil], [['en']], det_predictor)
    words = []
    for page in preds:
        for line in page.text_lines:
            conf = float(getattr(line, 'confidence', 1.0) or 1.0)
            if conf >= 0.3:
                words.append((str(line.text).strip(), conf))
    return words


ENGINES = {'easy': run_easyocr, 'paddle': run_paddleocr, 'surya': run_surya}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--page', type=int, default=1, help='1-indexed PDF page')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--engines', default='easy,paddle,surya',
                    help='comma-separated subset of easy,paddle,surya')
    ap.add_argument('--save-image', action='store_true',
                    help='write rendered image to disk for visual inspection')
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    print(f"PDF:  {pdf.name}")
    print(f"Page: {args.page}  |  DPI: {args.dpi}")

    img = render_page(str(pdf), args.page - 1, dpi=args.dpi)
    print(f"Img:  {img.shape[1]}x{img.shape[0]}")

    if args.save_image:
        out_img = pdf.parent / f"{pdf.stem}_p{args.page}_{args.dpi}dpi.png"
        cv2.imwrite(str(out_img), img)
        print(f"Saved: {out_img}")

    selected = [e.strip() for e in args.engines.split(',') if e.strip()]
    results = {}
    for engine in selected:
        if engine not in ENGINES:
            print(f"  skip unknown engine: {engine}")
            continue
        print(f"\n--- {engine.upper()} ---")
        t0 = time.time()
        try:
            words = ENGINES[engine](img)
        except ImportError as e:
            print(f"  not installed: {e}")
            continue
        except Exception as e:
            print(f"  failed: {e}")
            continue
        elapsed = time.time() - t0
        tags, raw_tokens = filter_tags(words)
        tag_shaped_toks = sorted({t for t, _ in raw_tokens if OCR_TAG_SHAPE.match(t)})
        results[engine] = {
            'time_s': round(elapsed, 1),
            'word_count': len(words),
            'token_count': len(raw_tokens),
            'tag_shaped': len(tag_shaped_toks),
            'tag_shaped_tokens': tag_shaped_toks,
            'normalized_tags': sorted(tags),
        }
        print(f"  time:       {elapsed:.1f}s")
        print(f"  word spans: {len(words)}")
        print(f"  tag-shaped tokens ({len(tag_shaped_toks)}): {tag_shaped_toks}")
        print(f"  normalized tags ({len(tags)}): {sorted(tags)}")

    print("\n=== SUMMARY ===")
    print(f"{'engine':<8} {'time':>6} {'words':>6} {'tags':>6}  matches")
    for engine, r in results.items():
        print(f"{engine:<8} {r['time_s']:>5}s {r['word_count']:>6} "
              f"{len(r['normalized_tags']):>6}  {r['normalized_tags']}")


if __name__ == '__main__':
    main()
