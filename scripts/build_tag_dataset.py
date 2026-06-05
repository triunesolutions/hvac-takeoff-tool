
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path (moved into scripts/)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
"""
build_tag_dataset.py - Turn ground_truth.jsonl into tag-recognition training data.

For every count annotation, render the raw PDF page at target DPI, crop a
neighborhood around the symbol center, and save (image, tag, subject, properties).

This produces a classification-style dataset: (crop_image -> tag_string).
The crop includes BOTH the symbol and its nearby tag bubble, so a downstream
model can be trained to either classify the tag or detect+read the bubble.

Output layout:
  <out>/
    images/<project_slug>/<idx>.png
    labels.jsonl                   # one row per crop

Each labels.jsonl row:
  {"img": "images/proj/00001.png",
   "tag": "A-1", "subject": "AD-GRD",
   "page": 3, "source_rect_in_crop": [x0,y0,x1,y1],
   "project": "...", "properties": {...}}
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import fitz
from PIL import Image
import numpy as np


def slugify(name):
    return "".join(c if c.isalnum() or c in '-_' else '_' for c in name)[:80]


def render_page_cached(cache, pdf_path, page_idx, dpi, with_annots=False):
    """Return (np_rgb_image, scale) for a page, with a small LRU cache.

    with_annots=False: render the underlying page without Bluebeam markup overlays.
    This keeps the annotation coordinates aligned to the rendered pixels.
    """
    key = (pdf_path, page_idx, dpi, with_annots)
    if key in cache:
        return cache[key]
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False, annots=with_annots)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    # scale = pixels per PDF point
    scale = dpi / 72.0
    doc.close()
    # Simple cache bound: keep only the last 3 page renders (memory)
    if len(cache) >= 3:
        cache.pop(next(iter(cache)))
    cache[key] = (img, scale)
    return img, scale


def pdf_rect_to_image(rect, page_w, page_h, rotation, scale):
    """Convert a PDF-coord bbox to image-pixel coords accounting for page rotation.

    PyMuPDF's annot.rect is already in the rotated display space (top-left origin
    in the visible page), so we just multiply by scale.
    """
    x0, y0, x1, y1 = rect
    return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)


def crop_center(img, cx_px, cy_px, size):
    """Crop `size`x`size` centered at (cx_px, cy_px), pad with white if near edges."""
    h, w, _ = img.shape
    half = size // 2
    x0 = int(cx_px - half)
    y0 = int(cy_px - half)
    x1 = x0 + size
    y1 = y0 + size

    # Compute source/destination ranges
    sx0, sx1 = max(0, x0), min(w, x1)
    sy0, sy1 = max(0, y0), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

    out = np.full((size, size, 3), 255, dtype=np.uint8)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='ground_truth.jsonl')
    ap.add_argument('--out', required=True, help='output dataset directory')
    ap.add_argument('--crop', type=int, default=320, help='crop side in pixels')
    ap.add_argument('--dpi', type=int, default=200, help='render DPI')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip crops that already exist (resume)')
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / 'labels.jsonl'

    # Group rows by (takeoff_pdf, page) so we render each page once.
    # We render from the TAKEOFF pdf (with annots=False) because the annotation
    # coords live in the takeoff PDF's coordinate space — rendering from raw
    # caused pixel-level misalignments.
    rows_by_page = defaultdict(list)
    with open(args.jsonl, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            rows_by_page[(r['takeoff_pdf'], r['page'])].append(r)

    # Sort for deterministic output
    keys = sorted(rows_by_page.keys())
    if args.limit:
        keys = keys[:args.limit]

    render_cache = {}
    total_rows = sum(len(rows_by_page[k]) for k in keys)
    print(f"Processing {len(keys)} unique (pdf,page) combos, {total_rows} total crops.")

    written = 0
    skipped = 0
    failed = 0
    per_proj_idx = defaultdict(int)

    with open(labels_path, 'w' if not args.skip_existing else 'a', encoding='utf-8') as lf:
        for i, key in enumerate(keys):
            takeoff_pdf, page_num = key
            rows = rows_by_page[key]
            project_slug = slugify(rows[0]['project'])
            proj_img_dir = out_dir / 'images' / project_slug
            proj_img_dir.mkdir(parents=True, exist_ok=True)

            if i % 50 == 0:
                print(f"  [{i}/{len(keys)}] {project_slug} page {page_num} ({len(rows)} counts)  "
                      f"written={written} skipped={skipped} failed={failed}")

            if not os.path.exists(takeoff_pdf):
                failed += len(rows)
                continue

            try:
                img, scale = render_page_cached(
                    render_cache, takeoff_pdf, page_num - 1, args.dpi, with_annots=False
                )
            except Exception as e:
                print(f"    ! render failed {takeoff_pdf} p{page_num}: {e}")
                failed += len(rows)
                continue

            for r in rows:
                idx = per_proj_idx[project_slug]
                per_proj_idx[project_slug] += 1
                rel_img_path = f"images/{project_slug}/{idx:06d}.png"
                abs_img_path = out_dir / rel_img_path

                if args.skip_existing and abs_img_path.exists():
                    skipped += 1
                    continue

                cx_px = r['cx'] * scale
                cy_px = r['cy'] * scale

                # If center lies outside page, skip
                if not (0 <= cx_px < img.shape[1] and 0 <= cy_px < img.shape[0]):
                    failed += 1
                    continue

                crop = crop_center(img, cx_px, cy_px, args.crop)

                try:
                    Image.fromarray(crop).save(abs_img_path, optimize=True)
                except Exception as e:
                    print(f"    ! save failed: {e}")
                    failed += 1
                    continue

                # Compute where the symbol bbox lies INSIDE the crop (in pixels)
                rx0, ry0, rx1, ry1 = pdf_rect_to_image(
                    r['rect'], r['page_w'], r['page_h'], r['rotation'], scale
                )
                half = args.crop // 2
                crop_x0 = cx_px - half
                crop_y0 = cy_px - half
                bb_in_crop = [
                    max(0.0, rx0 - crop_x0),
                    max(0.0, ry0 - crop_y0),
                    min(float(args.crop), rx1 - crop_x0),
                    min(float(args.crop), ry1 - crop_y0),
                ]

                label = {
                    'img': rel_img_path,
                    'tag': r.get('tag'),
                    'subject': r.get('subject'),
                    'count_style': r.get('count_style'),
                    'page': r['page'],
                    'project': r['project'],
                    'source_rect_in_crop': bb_in_crop,
                    'properties': r.get('properties', {}),
                }
                lf.write(json.dumps(label, ensure_ascii=False) + '\n')
                written += 1

    print(f"\n{'='*70}")
    print(f"DONE  written={written}  skipped={skipped}  failed={failed}")
    print(f"Labels: {labels_path}")
    print(f"Images: {out_dir / 'images'}")


if __name__ == '__main__':
    main()
