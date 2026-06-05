"""
build_v11_dataset.py — assemble the v11 YOLO training dataset.

Inputs:
  yolo_dataset/                                 (existing v10 base, ~29.5K train tiles)
  ground_truth/<project>/ls_ground_truth.json   (verified bboxes from 6 LS reviews)
  benchmark_output/<project>/*_detections.json  (source PDF path per project)

Output:
  yolo_dataset_v11/
    images/train/   <- v10 train + tiles from 5 ground_truth projects
    images/val/     <- v10 val + tiles from 1 held-out ground_truth project
    labels/train/
    labels/val/
    dataset.yaml    <- same class list as v10

ls_ground_truth.json bboxes are in display-pixel space at 200 DPI (the same
rendering the LS UI showed). We render the source PDF at 200 DPI here so the
coords line up directly — no scale conversion needed.

Usage:
  python build_v11_dataset.py                # full build
  python build_v11_dataset.py --dry-run      # plan only, no writes
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import argparse, json, os, shutil
from pathlib import Path
from collections import defaultdict

import cv2
import fitz
import numpy as np
import yaml

REPO = Path(__file__).parent
V10_DATASET = REPO / 'yolo_dataset'
V11_DATASET = REPO / 'yolo_dataset_v11'
GROUND_TRUTH_DIR = REPO / 'ground_truth'
BENCHMARK_DIR = REPO / 'benchmark_output'

DPI = 200
TILE_SIZE = 640
TILE_OVERLAP = 160

# Use the last ground_truth project as the new-data validation split. The v10
# val set is preserved so we can compare to the published v10 numbers.
VAL_PROJECT = '4.21.26 Anaheim 82'


def load_v10_class_map():
    cfg = yaml.safe_load((V10_DATASET / 'dataset.yaml').read_text())
    names = cfg['names']
    if isinstance(names, dict):
        return {names[i]: i for i in sorted(names.keys())}
    return {n: i for i, n in enumerate(names)}


def find_pdf_for_project(project_name):
    """Look up the source PDF path from the project's detections.json."""
    out_dir = BENCHMARK_DIR / project_name
    if not out_dir.is_dir():
        return None
    for det_path in out_dir.glob('*_detections.json'):
        try:
            data = json.loads(det_path.read_text())
            pdf = data.get('pdf')
            if pdf and Path(pdf).exists():
                return Path(pdf), data.get('dpi', DPI)
        except Exception:
            continue
    return None


def render_page(pdf_path, page_idx, dpi=DPI):
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


def tile_with_bboxes(img, bboxes, class_map, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Yield (tile_img, [yolo_label_lines]) for every tile that contains
    a bbox center. Empty tiles are skipped — ground_truth data is dense
    enough on plan pages that we don't need negative samples here.
    """
    h, w = img.shape[:2]
    step = tile_size - overlap

    # Pre-resolve each bbox to (cls_id, cx, cy, bw, bh).
    resolved = []
    for b in bboxes:
        cls = b['cls']
        if cls not in class_map:
            continue
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        resolved.append((class_map[cls], cx, cy, bw, bh))

    if not resolved:
        return

    for y_start in range(0, h, step):
        for x_start in range(0, w, step):
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)
            xs = max(0, x_end - tile_size)
            ys = max(0, y_end - tile_size)

            tile = img[ys:y_end, xs:x_end]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tile_lines = []
            for cls_id, cx, cy, bw, bh in resolved:
                tx = cx - xs
                ty = cy - ys
                if not (0 <= tx < tile_size and 0 <= ty < tile_size):
                    continue
                # Clip the box to the tile bounds, then convert to YOLO norm.
                bx1 = max(0.0, tx - bw / 2)
                by1 = max(0.0, ty - bh / 2)
                bx2 = min(float(tile_size), tx + bw / 2)
                by2 = min(float(tile_size), ty + bh / 2)
                cw = bx2 - bx1
                ch = by2 - by1
                if cw <= 0 or ch <= 0:
                    continue
                ncx = (bx1 + bx2) / 2 / tile_size
                ncy = (by1 + by2) / 2 / tile_size
                nw = cw / tile_size
                nh = ch / tile_size
                tile_lines.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

            if tile_lines:
                yield tile, tile_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not V10_DATASET.exists():
        print(f"ERROR: v10 dataset not found at {V10_DATASET}")
        sys.exit(1)

    class_map = load_v10_class_map()
    print(f"v10 class map: {len(class_map)} classes")

    projects = sorted(p.name for p in GROUND_TRUTH_DIR.iterdir() if p.is_dir())
    print(f"Ground-truth projects: {len(projects)}")
    for p in projects:
        print(f"  {p}{'  (VAL)' if p == VAL_PROJECT else ''}")

    if VAL_PROJECT not in projects:
        print(f"\nWARNING: VAL_PROJECT '{VAL_PROJECT}' not found, falling back to last project")

    if not args.dry_run:
        if V11_DATASET.exists():
            print(f"\nWiping existing {V11_DATASET}")
            shutil.rmtree(V11_DATASET)
        for split in ('train', 'val'):
            (V11_DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
            (V11_DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # Step 1: copy v10 dataset wholesale so v11 inherits all base data.
        print(f"\nCopying v10 dataset → v11 base...")
        for split in ('train', 'val'):
            src_img = V10_DATASET / 'images' / split
            src_lbl = V10_DATASET / 'labels' / split
            dst_img = V11_DATASET / 'images' / split
            dst_lbl = V11_DATASET / 'labels' / split
            n = 0
            for f in src_img.iterdir():
                shutil.copy2(f, dst_img / f.name)
                n += 1
            for f in src_lbl.iterdir():
                if f.suffix == '.txt':
                    shutil.copy2(f, dst_lbl / f.name)
            print(f"  {split}: copied {n} images")

    # Step 2: tile each ground_truth project and append.
    appended_train = 0
    appended_val = 0
    skipped_classes = defaultdict(int)
    bbox_total = 0

    for project in projects:
        gt_path = GROUND_TRUTH_DIR / project / 'ls_ground_truth.json'
        if not gt_path.exists():
            print(f"  ! {project}: no ls_ground_truth.json, skipping")
            continue
        gt = json.loads(gt_path.read_text())

        info = find_pdf_for_project(project)
        if info is None:
            print(f"  ! {project}: source PDF not found via detections.json, skipping")
            continue
        pdf_path, src_dpi = info
        if int(src_dpi) != DPI:
            print(f"  ! {project}: detections.json DPI={src_dpi} != {DPI} — coords won't align, skipping")
            continue

        split = 'val' if project == VAL_PROJECT else 'train'
        safe = project.replace(' ', '_').replace("'", '').replace('/', '_')

        proj_bbox = sum(len(b) for b in gt.values())
        proj_pages = sum(1 for b in gt.values() if b)
        print(f"\n[{split}] {project}: {proj_bbox} bboxes across {proj_pages} pages")
        bbox_total += proj_bbox

        for page_str, bboxes in gt.items():
            if not bboxes:
                continue
            page_idx = int(page_str)
            try:
                img = render_page(pdf_path, page_idx)
            except Exception as e:
                print(f"  ! page {page_idx+1} render failed: {e}")
                continue

            for c in bboxes:
                if c['cls'] not in class_map:
                    skipped_classes[c['cls']] += 1

            tiles_written = 0
            for ti, (tile_img, lines) in enumerate(tile_with_bboxes(img, bboxes, class_map)):
                tile_name = f"gt_{safe}_p{page_idx+1}_t{ti:04d}"
                if not args.dry_run:
                    cv2.imwrite(str(V11_DATASET / 'images' / split / f"{tile_name}.png"), tile_img)
                    (V11_DATASET / 'labels' / split / f"{tile_name}.txt").write_text('\n'.join(lines) + '\n')
                tiles_written += 1
                if split == 'train':
                    appended_train += 1
                else:
                    appended_val += 1
            print(f"  page {page_idx+1}: {len(bboxes)} bboxes → {tiles_written} tiles")

    if skipped_classes:
        print(f"\nClasses skipped (not in v10 class_map):")
        for c, n in sorted(skipped_classes.items(), key=lambda x: -x[1]):
            print(f"  {n:5d}  {c}")

    print(f"\nAppended tiles: train={appended_train} val={appended_val} (from {bbox_total} bboxes)")

    if not args.dry_run:
        # Step 3: write dataset.yaml mirroring v10 class order, point at new path.
        cfg_v10 = yaml.safe_load((V10_DATASET / 'dataset.yaml').read_text())
        cfg_v11 = dict(cfg_v10)
        cfg_v11['path'] = str(V11_DATASET)
        (V11_DATASET / 'dataset.yaml').write_text(yaml.dump(cfg_v11, default_flow_style=False))

        train_total = len(list((V11_DATASET / 'images' / 'train').iterdir()))
        val_total = len(list((V11_DATASET / 'images' / 'val').iterdir()))
        print(f"\n{'='*60}")
        print(f"v11 DATASET READY at {V11_DATASET}")
        print(f"  train: {train_total} images")
        print(f"  val:   {val_total} images")
        print(f"  classes: {len(class_map)}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
