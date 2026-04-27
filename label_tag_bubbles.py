import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
"""
label_tag_bubbles.py - Auto-label tag-bubble bounding boxes from PDF text layer.

For each count in ground_truth.jsonl we know the symbol center (cx, cy) and the
tag string (e.g. "A-1", "CU-3"). The tag is usually printed as text on the
drawing near the symbol. We search the PDF text layer for words matching the
tag within a radius of the symbol center and record the tightest bbox.

Output: tag_bubble_labels.jsonl — one row per count, with either a hit bbox or
a "miss" marker. Downstream dataset builder uses only hits.

Coord system: we return bubble bbox in the SAME display-pixel space that
build_tag_dataset already uses, so crop-relative coords are trivial.

Rotation transform is the same one proven out in extract_ground_truth.py.
"""
import argparse
import json
import re
import random
from pathlib import Path
from collections import defaultdict

import fitz


# ─── rotation transform (same as extract_ground_truth.native_rect_to_display) ──

def native_rect_to_display(x0, y0, x1, y1, mb_w, mb_h, rotation):
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    if rotation == 0:
        return lo_x, mb_h - hi_y, hi_x, mb_h - lo_y
    if rotation == 90:
        return lo_y, lo_x, hi_y, hi_x
    if rotation == 180:
        return mb_w - hi_x, lo_y, mb_w - lo_x, hi_y
    if rotation == 270:
        return mb_h - hi_y, mb_w - hi_x, mb_h - lo_y, mb_w - lo_x
    return lo_x, mb_h - hi_y, hi_x, mb_h - lo_y


# ─── tag normalization for matching ───────────────────────────────────────────

_NORM_RE = re.compile(r'[^A-Z0-9]+')

def norm(s: str) -> str:
    """Uppercase, strip all non-alphanumeric. 'A-1' == 'A1' == 'a 1'."""
    return _NORM_RE.sub('', (s or '').upper())


# ─── text-layer search ────────────────────────────────────────────────────────

def find_bubble(page, tag, cx, cy, mb_w, mb_h, rotation, radius=220):
    """Return (x0, y0, x1, y1) in display space for the closest text match,
    or None if no match within radius."""
    target = norm(tag)
    if not target or len(target) < 2:
        return None

    try:
        # get_text("words") returns tuples (x0, y0, x1, y1, word, block, line, wordno)
        # in PDF-native coords (pre-rotation), bottom-left origin y-up? actually
        # PyMuPDF's page.get_text returns coords already in the page's DISPLAY
        # space (top-left origin, y-down, rotation applied). But these specific
        # PDFs have rotated page boxes where PyMuPDF's rect handling was buggy
        # for annotations. For get_text("words") it's historically reliable —
        # try that first.
        words = page.get_text("words") or []
    except Exception:
        return None

    best = None
    best_d = radius * radius
    for w in words:
        x0, y0, x1, y1, text, *_ = w
        if norm(text) != target:
            # also accept multi-word: the tag might be split "A" "-" "1" or "A" "1"
            continue
        wcx = (x0 + x1) / 2
        wcy = (y0 + y1) / 2
        d = (wcx - cx) ** 2 + (wcy - cy) ** 2
        if d < best_d:
            best_d = d
            best = (x0, y0, x1, y1)

    if best is not None:
        return best

    # Fallback: tag might be split across words. Try concatenating adjacent
    # words on the same line and see if any concatenation matches.
    # Group by (block, line).
    lines = defaultdict(list)
    for w in words:
        x0, y0, x1, y1, text, block, line, wordno = w
        lines[(block, line)].append((wordno, x0, y0, x1, y1, text))

    for key, items in lines.items():
        items.sort()
        for i in range(len(items)):
            concat = ''
            xs0, ys0, xs1, ys1 = None, None, None, None
            for j in range(i, min(i + 4, len(items))):  # up to 4-word combo
                _, x0, y0, x1, y1, text = items[j]
                concat += text
                xs0 = x0 if xs0 is None else min(xs0, x0)
                ys0 = y0 if ys0 is None else min(ys0, y0)
                xs1 = x1 if xs1 is None else max(xs1, x1)
                ys1 = y1 if ys1 is None else max(ys1, y1)
                if norm(concat) == target:
                    wcx = (xs0 + xs1) / 2
                    wcy = (ys0 + ys1) / 2
                    d = (wcx - cx) ** 2 + (wcy - cy) ** 2
                    if d < best_d:
                        best_d = d
                        best = (xs0, ys0, xs1, ys1)
                    break
                if len(norm(concat)) > len(target):
                    break

    return best


# ─── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='ground_truth.jsonl')
    ap.add_argument('--out', required=True, help='output tag_bubble_labels.jsonl')
    ap.add_argument('--sample', type=int, default=None,
                    help='only process a random sample of N counts (for dry run)')
    ap.add_argument('--radius', type=float, default=220.0,
                    help='search radius around symbol center (PDF points)')
    ap.add_argument('--use-takeoff-pdf', action='store_true',
                    help='read text layer from takeoff PDF (has annotations) '
                         'instead of raw PDF — both should work; takeoff matches '
                         'the render used by build_tag_dataset')
    args = ap.parse_args()

    # Load all rows
    rows = []
    with open(args.jsonl, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} counts from {args.jsonl}")

    if args.sample:
        random.seed(42)
        rows = random.sample(rows, min(args.sample, len(rows)))
        print(f"Sampling {len(rows)}")

    # Group by (pdf, page) so we only open each PDF/page once
    by_page = defaultdict(list)
    for r in rows:
        pdf_key = r['takeoff_pdf'] if args.use_takeoff_pdf else r['raw_pdf']
        by_page[(pdf_key, r['page'])].append(r)

    hits = 0
    misses = 0
    errors = 0
    doc_cache = {}

    with open(args.out, 'w', encoding='utf-8') as of:
        for (pdf_path, page_num), items in by_page.items():
            if pdf_path not in doc_cache:
                # only cache up to 2 docs — these are big
                while len(doc_cache) >= 2:
                    k, d = doc_cache.popitem()
                    d.close()
                try:
                    doc_cache[pdf_path] = fitz.open(pdf_path)
                except Exception as e:
                    print(f"  open failed: {pdf_path}: {e}")
                    for r in items:
                        of.write(json.dumps({**r, 'bubble_rect': None, 'reason': 'open_failed'}) + '\n')
                        errors += 1
                    continue

            doc = doc_cache[pdf_path]
            if page_num - 1 >= len(doc):
                for r in items:
                    of.write(json.dumps({**r, 'bubble_rect': None, 'reason': 'page_oob'}) + '\n')
                    errors += 1
                continue

            page = doc[page_num - 1]
            mb = page.mediabox
            mb_w, mb_h = float(mb.width), float(mb.height)
            rotation = page.rotation

            # PyMuPDF get_text returns coords in DISPLAY space already (top-left,
            # y-down, post-rotation). Our stored cx/cy are also display space.
            # So no conversion needed for the search space — we compare directly.
            for r in items:
                tag = r.get('tag')
                if not tag:
                    of.write(json.dumps({**r, 'bubble_rect': None, 'reason': 'no_tag'}) + '\n')
                    misses += 1
                    continue
                rect = find_bubble(page, tag, r['cx'], r['cy'], mb_w, mb_h, rotation, args.radius)
                if rect is None:
                    of.write(json.dumps({**r, 'bubble_rect': None, 'reason': 'no_text_match'}) + '\n')
                    misses += 1
                else:
                    of.write(json.dumps({**r, 'bubble_rect': list(rect), 'reason': 'hit'}) + '\n')
                    hits += 1

    for d in doc_cache.values():
        d.close()

    total = hits + misses + errors
    print()
    print(f"{'='*60}")
    print(f"hits:    {hits:6d}  ({100*hits/max(1,total):5.1f}%)")
    print(f"misses:  {misses:6d}  ({100*misses/max(1,total):5.1f}%)")
    print(f"errors:  {errors:6d}  ({100*errors/max(1,total):5.1f}%)")
    print(f"total:   {total:6d}")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
