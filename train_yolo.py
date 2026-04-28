"""
Prepare YOLO training data from annotated PDFs and train a model.
Reads from organized project folders: projects/{name}/raw/ + labeled/

Usage:
    python train_yolo.py                    # Train on Flex projects only
    python train_yolo.py --all              # Train on all projects with annotations
    python train_yolo.py --projects 01 02 03 04 08  # Train on specific projects
    python train_yolo.py --resume           # Resume interrupted training
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import fitz
import cv2
import numpy as np
import os
import shutil
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from class_aliases import normalize_class

PROJECTS_DIR = r"C:\Users\JFL\Downloads\Triune\data to train\projects"
YOLO_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\yolo_dataset"
OUTPUT_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool"
DPI = 200
TILE_SIZE = 640
TILE_OVERLAP = 160

# Equipment classes — will auto-expand as new types are found
KNOWN_CLASSES = [
    'AD-T-BAR SUPPLY',
    'AD-T-BAR RETURN',
    'AD-SURF SUPPLY',
    'AD-SURF RETURN',
    'AD-LINEAR SLOT DIFFUSER',
    'AD-LINEAR PLENUM',
    'LOUVERS',
]


def annot_to_display(ax, ay, rotation, mb_w, mb_h):
    """Convert annotation (mediabox) coords to display coords."""
    if rotation == 270:
        return ay, mb_w - ax
    elif rotation == 90:
        return mb_h - ay, ax
    elif rotation == 180:
        return mb_w - ax, mb_h - ay
    return ax, ay


def extract_annotations(pdf_path):
    """
    Extract all polygon annotations from a PDF.
    Returns dict: page_idx -> list of {class, cx, cy, ...} in display coords.
    """
    doc = fitz.open(pdf_path)
    pages = defaultdict(list)
    class_set = set()

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        rotation = page.rotation
        mb_w, mb_h = page.mediabox.width, page.mediabox.height

        for a in page.annots() or []:
            try:
                if a.type[1] != 'Polygon':
                    continue
            except:
                continue

            subject = a.info.get('subject', '').strip()
            content = a.info.get('content', '').strip()
            if not subject or not content:
                continue

            # Normalize class name (merge aliases, fix typos, collapse plurals)
            subject = normalize_class(subject)

            rect = a.rect
            acx = (rect.x0 + rect.x1) / 2
            acy = (rect.y0 + rect.y1) / 2
            dcx, dcy = annot_to_display(acx, acy, rotation, mb_w, mb_h)

            class_set.add(subject)
            pages[page_idx].append({
                'class': subject,
                'tag': content,
                'display_cx': dcx,
                'display_cy': dcy,
            })

    doc.close()
    return pages, class_set


def render_page(pdf_path, page_idx, dpi=DPI):
    """Render a PDF page to BGR numpy array."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def find_raw_pdf(project_dir, labeled_pdf_name, labeled_page_count):
    """
    Find the matching raw (unlabeled) PDF for a labeled PDF.
    For Flex projects: raw PDF is the one without 'Final' in the name.
    For other projects: raw PDF is in the raw/ folder.
    """
    raw_dir = os.path.join(project_dir, 'raw')
    if not os.path.exists(raw_dir):
        return None

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
    if not raw_files:
        return None

    # If there's only one raw PDF with same page count, that's it
    for rf in raw_files:
        rpath = os.path.join(raw_dir, rf)
        try:
            doc = fitz.open(rpath)
            pc = doc.page_count
            doc.close()
            if pc == labeled_page_count:
                return rpath
        except:
            continue

    # Fallback: return the first/largest raw PDF
    return os.path.join(raw_dir, raw_files[0])


def tile_image(img, annotations, class_map, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Split image into tiles, assign annotations to each tile."""
    h, w = img.shape[:2]
    step = tile_size - overlap
    scale = DPI / 72
    box_half = 45 * scale  # symbol bounding box radius in pixels

    tiles = []
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

            tile_annots = []
            for ann in annotations:
                px_cx = ann['display_cx'] * scale
                px_cy = ann['display_cy'] * scale
                cx = px_cx - xs
                cy = px_cy - ys

                if 0 <= cx < tile_size and 0 <= cy < tile_size:
                    cls_name = ann['class']
                    if cls_name not in class_map:
                        continue
                    tile_annots.append({
                        'class_id': class_map[cls_name],
                        'cx': max(0, min(1, cx / tile_size)),
                        'cy': max(0, min(1, cy / tile_size)),
                        'w': min(box_half * 2 / tile_size, 1.0),
                        'h': min(box_half * 2 / tile_size, 1.0),
                    })

            tiles.append((tile, tile_annots))

    return tiles


def prepare_dataset(project_ids=None, min_class_examples=30):
    """
    Build YOLO dataset from selected projects.
    Classes with fewer than `min_class_examples` total instances are dropped
    to prevent training instability from rare classes.
    """
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

    # Discover projects
    all_projects = sorted(os.listdir(PROJECTS_DIR))
    if project_ids:
        projects = [p for p in all_projects if any(p.startswith(pid) for pid in project_ids)]
    else:
        projects = all_projects

    print(f"Projects to process: {len(projects)}")
    for p in projects:
        print(f"  {p}")

    # First pass: discover all classes AND count instances
    class_counts = defaultdict(int)
    project_data = []

    for proj_name in projects:
        proj_dir = os.path.join(PROJECTS_DIR, proj_name)
        labeled_dir = os.path.join(proj_dir, 'labeled')
        if not os.path.isdir(labeled_dir):
            continue

        for f in os.listdir(labeled_dir):
            if not f.endswith('.pdf'):
                continue
            labeled_path = os.path.join(labeled_dir, f)
            pages, classes = extract_annotations(labeled_path)
            if not any(pages.values()):
                continue

            # Count instances per class
            for page_anns in pages.values():
                for ann in page_anns:
                    class_counts[ann['class']] += 1

            total = sum(len(v) for v in pages.values())
            print(f"  {proj_name}: {f[:50]} — {total} annotations, classes: {set(classes)}")
            project_data.append((proj_name, proj_dir, labeled_path, f, pages))

    # Filter out rare classes
    kept_classes = {c for c, n in class_counts.items() if n >= min_class_examples}
    dropped_classes = {c: n for c, n in class_counts.items() if n < min_class_examples}

    print(f"\n--- Class filtering (min {min_class_examples} examples) ---")
    print(f"  Kept: {len(kept_classes)} classes")
    print(f"  Dropped: {len(dropped_classes)} rare classes")
    if dropped_classes:
        for c, n in sorted(dropped_classes.items(), key=lambda x: -x[1]):
            print(f"    DROP {c}: {n} examples")

    # Build class map (keep known order, then add by frequency)
    class_list = [c for c in KNOWN_CLASSES if c in kept_classes]
    for c, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        if c in kept_classes and c not in class_list:
            class_list.append(c)
    class_map = {name: idx for idx, name in enumerate(class_list)}
    all_classes = kept_classes

    print(f"\nClasses ({len(class_list)}):")
    for idx, name in enumerate(class_list):
        marker = '*' if name in all_classes else ' '
        print(f"  {idx}: {name} {marker}")

    # Second pass: render, tile, save
    # Use last project as validation, rest for training
    total_tiles = 0
    total_annots = 0

    for pi, (proj_name, proj_dir, labeled_path, labeled_fname, pages) in enumerate(project_data):
        split = 'val' if pi == len(project_data) - 1 else 'train'
        safe_name = proj_name.replace(' ', '_')

        doc = fitz.open(labeled_path)
        labeled_page_count = doc.page_count
        doc.close()

        # Find raw PDF
        raw_path = find_raw_pdf(proj_dir, labeled_fname, labeled_page_count)

        for page_idx, annotations in pages.items():
            if not annotations:
                continue

            # Render from raw PDF if available, otherwise from labeled PDF directly
            # (labeled PDFs have polygon annotation overlays but the base drawing is intact)
            render_path = raw_path if raw_path else labeled_path
            try:
                img = render_page(render_path, page_idx)
            except Exception as e:
                # If raw fails, try labeled as fallback
                if render_path != labeled_path:
                    try:
                        img = render_page(labeled_path, page_idx)
                        print(f"  NOTE: using labeled PDF for {proj_name} page {page_idx+1} (no matching raw)")
                    except:
                        print(f"  SKIP page {page_idx+1} of {proj_name} — render failed: {e}")
                        continue
                else:
                    print(f"  SKIP page {page_idx+1} of {proj_name} — render failed: {e}")
                    continue

            tiles = tile_image(img, annotations, class_map)

            saved = 0
            for ti, (tile_img, tile_annots) in enumerate(tiles):
                save_empty = (ti % 10 == 0)
                if not tile_annots and not save_empty:
                    continue

                tile_name = f"{safe_name}_p{page_idx+1}_t{ti:04d}"
                img_path = os.path.join(YOLO_DIR, 'images', split, f"{tile_name}.png")
                lbl_path = os.path.join(YOLO_DIR, 'labels', split, f"{tile_name}.txt")

                cv2.imwrite(img_path, tile_img)
                with open(lbl_path, 'w') as f:
                    for ann in tile_annots:
                        f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

                if tile_annots:
                    saved += 1
                    total_annots += len(tile_annots)
                total_tiles += 1

            print(f"  [{split}] {proj_name} page {page_idx+1}: {len(annotations)} annots → {saved} labeled tiles")

    # Write dataset config
    config = {
        'path': YOLO_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_list)},
    }
    config_path = os.path.join(YOLO_DIR, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"DATASET READY")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Total annotations: {total_annots}")
    print(f"  Classes: {len(class_list)}")
    for split in ['train', 'val']:
        n = len(os.listdir(os.path.join(YOLO_DIR, 'images', split)))
        print(f"  {split}: {n} images")
    print(f"  Config: {config_path}")

    return config_path, class_list


def train_model(config_path, resume=False):
    """Train YOLOv8 on the prepared dataset."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("TRAINING YOLOv8")
    print(f"{'='*60}")

    if resume:
        weights = os.path.join(OUTPUT_DIR, 'runs', 'detect', 'runs', 'hvac_detect', 'weights', 'last.pt')
        if os.path.exists(weights):
            model = YOLO(weights)
            model.train(resume=True)
            return
        else:
            print(f"No checkpoint found at {weights}, starting fresh")

    model = YOLO('yolov8n.pt')
    model.train(
        data=config_path,
        epochs=50,
        imgsz=TILE_SIZE,
        batch=4,
        patience=15,
        device='cpu',
        workers=0,
        project=os.path.join(OUTPUT_DIR, 'runs'),
        name='hvac_v2',
        exist_ok=True,
    )

    # Copy best weights to models/
    best_src = os.path.join(OUTPUT_DIR, 'runs', 'hvac_v2', 'weights', 'best.pt')
    if os.path.exists(best_src):
        os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
        dst = os.path.join(OUTPUT_DIR, 'models', 'hvac_yolov8n_v2.pt')
        shutil.copy2(best_src, dst)
        print(f"\nBest model saved: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Train on all projects')
    parser.add_argument('--projects', nargs='+', help='Project ID prefixes (e.g., 01 02 03)')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted training')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Run dataset extraction/tiling only — skip training. '
                             'Use this locally before uploading the dataset to Kaggle.')
    args = parser.parse_args()

    if args.resume:
        train_model(None, resume=True)
    else:
        if args.all:
            project_ids = None
        elif args.projects:
            project_ids = args.projects
        else:
            # Default: Flex projects only
            project_ids = ['01', '02', '03', '04']

        config_path, classes = prepare_dataset(project_ids)
        if args.prepare_only:
            print(f"\n--prepare-only set. Skipping training.")
            print(f"Dataset config:  {config_path}")
            print(f"Classes ({len(classes)}): {sorted(classes)}")
            print(f"Next: zip yolo_dataset/ and upload to Kaggle.")
        else:
            train_model(config_path)
