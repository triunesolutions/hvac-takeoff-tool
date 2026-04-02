"""
Prepare YOLO training data from annotated PDFs and train a model.
Converts human annotations from 'Final' PDFs into YOLO format.
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
from pathlib import Path

DATA_DIR = r"C:\Users\JFL\Downloads\Triune\data to train\Plans_Specs"
YOLO_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\yolo_dataset"
DPI = 200
TILE_SIZE = 640  # YOLO default input size
TILE_OVERLAP = 160  # Overlap between tiles (pixels)

# Equipment classes
CLASSES = [
    'AD-T-BAR SUPPLY',      # 0 - Square diffuser (with circle-A tag)
    'AD-T-BAR RETURN',      # 1 - Square grille (with circle-B tag)
    'AD-SURF SUPPLY',       # 2 - Round diffuser (with circle-C tag)
    'AD-SURF RETURN',       # 3 - Square return (with circle-D tag)
    'AD-LINEAR SLOT DIFFUSER',  # 4 - Linear slot
    'AD-LINEAR PLENUM',     # 5 - Linear plenum
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}


def annot_to_display(ax, ay, rotation, mb_w, mb_h):
    if rotation == 270:
        return ay, mb_w - ax
    elif rotation == 90:
        return mb_h - ay, ax
    elif rotation == 180:
        return mb_w - ax, mb_h - ay
    return ax, ay


def extract_annotations_as_pixels(pdf_path, page_idx, dpi=DPI):
    """Extract annotations and convert to pixel coordinates in the rendered image."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rotation = page.rotation
    mb_w, mb_h = page.mediabox.width, page.mediabox.height
    scale = dpi / 72

    annotations = []
    annots = list(page.annots()) if page.annots() else []
    for a in annots:
        if a.type[1] == 'Highlight':
            continue
        subject = a.info.get('subject', '').strip()
        content = a.info.get('content', '').strip()
        if not subject or not content:
            continue
        if subject not in CLASS_MAP:
            # Skip types we don't train on (e.g., LOUVERS)
            continue

        rect = a.rect
        acx = (rect.x0 + rect.x1) / 2
        acy = (rect.y0 + rect.y1) / 2

        # Convert to display coords
        dcx, dcy = annot_to_display(acx, acy, rotation, mb_w, mb_h)

        # Convert to pixel coords
        px_cx = dcx * scale
        px_cy = dcy * scale

        # Annotation bounding box is small (just the tag bubble ~22x27 pts)
        # But the actual equipment symbol is larger — use a bigger bbox
        # that captures the whole symbol + tag
        # Equipment symbols are roughly 60x60 pts in display space
        box_half = 45 * scale  # 45 PDF pts = ~125px at 200dpi

        annotations.append({
            'class': subject,
            'class_id': CLASS_MAP[subject],
            'tag': content,
            'px_cx': px_cx,
            'px_cy': px_cy,
            'px_x1': px_cx - box_half,
            'px_y1': px_cy - box_half,
            'px_x2': px_cx + box_half,
            'px_y2': px_cy + box_half,
        })

    doc.close()
    return annotations


def render_page_image(pdf_path, page_idx, dpi=DPI):
    """Render a page to BGR numpy array."""
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


def tile_image_with_annotations(img, annotations, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """
    Split a large image into tiles and assign annotations to tiles.
    Returns list of (tile_image, tile_annotations) tuples.
    """
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles = []

    for y_start in range(0, h, step):
        for x_start in range(0, w, step):
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)
            x_start_actual = max(0, x_end - tile_size)
            y_start_actual = max(0, y_end - tile_size)

            tile = img[y_start_actual:y_end, x_start_actual:x_end]

            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            # Find annotations that fall within this tile
            tile_annots = []
            for ann in annotations:
                cx = ann['px_cx'] - x_start_actual
                cy = ann['px_cy'] - y_start_actual
                bw = ann['px_x2'] - ann['px_x1']
                bh = ann['px_y2'] - ann['px_y1']

                # Check if center is within tile
                if 0 <= cx < tile_size and 0 <= cy < tile_size:
                    # Convert to YOLO format: class cx cy w h (all normalized 0-1)
                    yolo_cx = cx / tile_size
                    yolo_cy = cy / tile_size
                    yolo_w = min(bw / tile_size, 1.0)
                    yolo_h = min(bh / tile_size, 1.0)

                    # Clamp
                    yolo_cx = max(0, min(1, yolo_cx))
                    yolo_cy = max(0, min(1, yolo_cy))

                    tile_annots.append({
                        'class_id': ann['class_id'],
                        'cx': yolo_cx,
                        'cy': yolo_cy,
                        'w': yolo_w,
                        'h': yolo_h,
                    })

            tiles.append((tile, tile_annots, x_start_actual, y_start_actual))

    return tiles


def prepare_dataset():
    """Convert all annotated PDFs to YOLO training format."""
    # Clean output directory
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

    # Find all Final PDFs
    files = os.listdir(DATA_DIR)
    final_pdfs = sorted([f for f in files if 'Final' in f and f.endswith('.pdf')])
    print(f"Found {len(final_pdfs)} annotated PDFs")

    # Use 3 for training, 1 for validation
    train_pdfs = final_pdfs[:3]
    val_pdfs = final_pdfs[3:]

    total_tiles = 0
    total_annotations = 0

    for split, pdf_list in [('train', train_pdfs), ('val', val_pdfs)]:
        for pdf_name in pdf_list:
            pdf_path = os.path.join(DATA_DIR, pdf_name)
            name = pdf_name.replace('3.11.26 EAG - 9530 Towne Center Drive ', '').replace(' - Final.pdf', '').replace('- Final.pdf', '')
            safe_name = name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')

            print(f"\n[{split}] Processing: {name}")

            # Find which pages have annotations
            doc = fitz.open(pdf_path)
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                annots = list(page.annots()) if page.annots() else []
                equip_annots = [a for a in annots if a.type[1] != 'Highlight'
                               and a.info.get('subject', '') in CLASS_MAP]
                if not equip_annots:
                    continue

                print(f"  Page {page_idx + 1}: {len(equip_annots)} annotations")
            doc.close()

            # For each annotated page, render + tile
            doc = fitz.open(pdf_path)
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                annots_check = list(page.annots()) if page.annots() else []
                equip_count = sum(1 for a in annots_check
                                 if a.type[1] != 'Highlight'
                                 and a.info.get('subject', '') in CLASS_MAP)
                if equip_count == 0:
                    continue

                # Extract annotations
                annotations = extract_annotations_as_pixels(pdf_path, page_idx)

                # Render the UNLABELED version (without annotations)
                unlabeled_name = pdf_name.replace(' - Final', '').replace('- Final', '')
                unlabeled_path = os.path.join(DATA_DIR, unlabeled_name)
                if os.path.exists(unlabeled_path):
                    img = render_page_image(unlabeled_path, page_idx)
                else:
                    img = render_page_image(pdf_path, page_idx)

                print(f"  Rendered page {page_idx+1}: {img.shape[1]}x{img.shape[0]}")
                print(f"  Annotations: {len(annotations)}")

                # Tile the image
                tiles = tile_image_with_annotations(img, annotations)
                print(f"  Tiles generated: {len(tiles)}")

                # Save tiles with annotations
                tiles_with_labels = 0
                for ti, (tile_img, tile_annots, tx, ty) in enumerate(tiles):
                    # Only save tiles that have at least one annotation
                    # (plus some empty tiles for negative samples)
                    save_empty = (ti % 8 == 0)  # Keep 1 in 8 empty tiles
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
                        tiles_with_labels += 1
                        total_annotations += len(tile_annots)

                    total_tiles += 1

                print(f"  Tiles with labels: {tiles_with_labels}")

            doc.close()

    # Write YOLO dataset config
    config = {
        'path': YOLO_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASSES)},
    }
    config_path = os.path.join(YOLO_DIR, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"DATASET PREPARED")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Config: {config_path}")

    # Count per split
    for split in ['train', 'val']:
        img_dir = os.path.join(YOLO_DIR, 'images', split)
        lbl_dir = os.path.join(YOLO_DIR, 'labels', split)
        n_img = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
        n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt') and os.path.getsize(os.path.join(lbl_dir, f)) > 0])
        print(f"  {split}: {n_img} images, {n_lbl} with labels")

    return config_path


def train_model(config_path):
    """Train YOLOv8 on the prepared dataset."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("TRAINING YOLOv8")
    print(f"{'='*60}")

    # Use YOLOv8 nano (smallest, fastest to train)
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=config_path,
        epochs=50,
        imgsz=TILE_SIZE,
        batch=8,
        patience=10,
        device='cpu',  # No GPU available
        workers=0,     # Windows compatibility
        project=os.path.join(os.path.dirname(YOLO_DIR), 'hvac-takeoff-tool', 'runs'),
        name='hvac_detect',
        exist_ok=True,
    )

    print(f"\nTraining complete!")
    return results


if __name__ == "__main__":
    config_path = prepare_dataset()
    print("\nDataset ready. Starting training...")
    train_model(config_path)
