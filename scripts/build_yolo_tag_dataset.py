
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path (moved into scripts/)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
"""
build_yolo_tag_dataset.py - Convert tag_dataset + tag_bubble_labels into
YOLO-format training data with two classes: symbol + tag_bubble.

Inputs:
  tag_dataset/
    images/<project>/000000.png
    labels.jsonl                    # has tag, source_rect_in_crop (symbol bbox)
  tag_bubble_labels.jsonl           # has bubble_rect_in_crop (or null)

Output:
  yolo_tag_dataset/
    images/train/*.png
    images/val/*.png
    labels/train/*.txt              # YOLO format: class cx cy w h (normalized)
    labels/val/*.txt
    data.yaml                        # Ultralytics dataset descriptor

Classes:
  0 = symbol       (always present, from source_rect_in_crop)
  1 = tag_bubble   (present only when OCR found the bubble)

Train/val split:
  per-project, seeded — 90% train / 10% val. All crops from one project stay
  together on the same side so we measure real generalization.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


def yolo_line(cls, x0, y0, x1, y1, crop_size):
    """Convert pixel bbox to YOLO (cls cx cy w h) normalized to [0,1]."""
    # Clip to [0, crop_size]
    x0 = max(0.0, min(crop_size, x0))
    y0 = max(0.0, min(crop_size, y0))
    x1 = max(0.0, min(crop_size, x1))
    y1 = max(0.0, min(crop_size, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    cx = ((x0 + x1) / 2) / crop_size
    cy = ((y0 + y1) / 2) / crop_size
    w = (x1 - x0) / crop_size
    h = (y1 - y0) / crop_size
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='tag_dataset')
    ap.add_argument('--bubble-labels', default='tag_bubble_labels.jsonl')
    ap.add_argument('--out', default='yolo_tag_dataset')
    ap.add_argument('--crop-size', type=int, default=320)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--copy-images', action='store_true',
                    help='copy images instead of symlinking (slower, portable)')
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)

    # Load crop labels (symbol bbox + tag + project)
    crops = {}
    with open(dataset / 'labels.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            crops[r['img']] = r

    # Load bubble labels
    bubble_rects = {}
    n_bubble_hits = 0
    if Path(args.bubble_labels).exists():
        with open(args.bubble_labels, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('bubble_rect_in_crop'):
                    bubble_rects[r['img']] = r['bubble_rect_in_crop']
                    n_bubble_hits += 1
        print(f"Loaded {n_bubble_hits} bubble bboxes from {args.bubble_labels}")
    else:
        print(f"WARNING: {args.bubble_labels} not found — only symbol class will be written")

    # Split per-project
    by_project = defaultdict(list)
    for img_path, row in crops.items():
        by_project[row['project']].append(img_path)

    random.seed(args.seed)
    projects = sorted(by_project.keys())
    random.shuffle(projects)
    n_val_proj = max(1, int(len(projects) * args.val_frac))
    val_projects = set(projects[:n_val_proj])
    train_projects = set(projects[n_val_proj:])
    print(f"Projects: {len(train_projects)} train / {len(val_projects)} val")
    print(f"Val projects: {sorted(val_projects)[:3]}...")

    # Build output dirs
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)

    for img_rel, row in crops.items():
        project = row['project']
        split = 'val' if project in val_projects else 'train'

        # symbol bbox
        symbol_rect = row.get('source_rect_in_crop')
        if not symbol_rect:
            stats['no_symbol_skip'] += 1
            continue

        lines = []
        sym_line = yolo_line(0, *symbol_rect, args.crop_size)
        if sym_line:
            lines.append(sym_line)
            stats['symbol_boxes'] += 1

        # tag bubble bbox (if we have one)
        bubble_rect = bubble_rects.get(img_rel)
        if bubble_rect:
            bub_line = yolo_line(1, *bubble_rect, args.crop_size)
            if bub_line:
                lines.append(bub_line)
                stats['bubble_boxes'] += 1

        if not lines:
            stats['empty_skip'] += 1
            continue

        # Flat filename: <project>__<orig>.png
        src_img = dataset / img_rel
        flat_name = img_rel.replace('/', '__').replace('\\', '__')
        dst_img = out / 'images' / split / flat_name
        dst_lbl = out / 'labels' / split / (flat_name.rsplit('.', 1)[0] + '.txt')

        if not dst_img.exists():
            if args.copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                try:
                    # Windows symlinks need privileges — fall back to copy if they fail
                    if dst_img.exists():
                        dst_img.unlink()
                    dst_img.symlink_to(src_img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src_img, dst_img)

        with open(dst_lbl, 'w') as lf:
            lf.write('\n'.join(lines) + '\n')

        stats[f'{split}_crops'] += 1

    # data.yaml
    yaml_path = out / 'data.yaml'
    with open(yaml_path, 'w') as yf:
        yf.write(f"""path: {out.resolve().as_posix()}
train: images/train
val: images/val

nc: 2
names:
  0: symbol
  1: tag_bubble
""")

    print()
    print('=' * 60)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == '__main__':
    main()
