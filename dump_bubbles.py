"""Dump every detected+OCR'd bubble to CSV. Use to compare raw OCR output
against the team's takeoff PDF when the schedule parser fails and strict
mode drops everything in takeoff_cli."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import fitz
import cv2
import numpy as np

from tag_matcher import (
    detect_bubbles_on_page,
    ocr_bubble_crops,
    merge_split_bubbles,
    _normalize_for_match,
)
from takeoff_cli import render_page, find_mechanical_pages, DPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--out', default=None)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--all-pages', action='store_true')
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    out = Path(args.out) if args.out else pdf.parent / f"{pdf.stem}_bubbles.csv"

    if args.all_pages:
        doc = fitz.open(pdf)
        pages = list(range(doc.page_count))
        doc.close()
    else:
        pages = find_mechanical_pages(str(pdf))

    print(f"Pages: {[p+1 for p in pages]}")
    rows = []
    counts = Counter()
    for pi in pages:
        img, *_ = render_page(str(pdf), pi)
        bubbles = detect_bubbles_on_page(img, conf=args.conf)
        if not bubbles:
            print(f"  Page {pi+1}: 0 bubbles")
            continue
        bubbles = ocr_bubble_crops(img, bubbles)
        bubbles = merge_split_bubbles(bubbles)
        print(f"  Page {pi+1}: {len(bubbles)} bubbles+merges OCR'd")
        for b in bubbles:
            text = (b.get('text') or '').strip()
            norm = _normalize_for_match(text)
            rows.append({
                'page': pi + 1,
                'ocr_text': text,
                'normalized': norm,
                'cx': round(b['cx'], 1),
                'cy': round(b['cy'], 1),
                'bubble_conf': round(b.get('conf', 0), 3),
                'ocr_conf': round(b.get('ocr_conf', 0), 3),
                'merged_from': str(b.get('merged_from') or ''),
            })
            if norm and not b.get('merged_from'):
                counts[norm] += 1

    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['page', 'ocr_text', 'normalized',
                                          'cx', 'cy', 'bubble_conf',
                                          'ocr_conf', 'merged_from'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows → {out}")
    print(f"\nTop OCR tags (non-merged only):")
    for tag, c in counts.most_common(40):
        print(f"  {tag:<20} {c:>5}")


if __name__ == '__main__':
    main()
