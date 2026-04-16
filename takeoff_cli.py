"""
HVAC Takeoff CLI Tool — Demo Version

Takes a blueprint PDF, runs the v8 model, and outputs:
1. Annotated PDF with colored boxes around detected equipment
2. Excel takeoff with counts per equipment type
3. Summary report (printed to console)

Usage:
    python takeoff_cli.py path/to/blueprint.pdf
    python takeoff_cli.py path/to/blueprint.pdf --conf 0.5
    python takeoff_cli.py path/to/blueprint.pdf --output-dir results/
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import os
import argparse
import time
from pathlib import Path
from collections import defaultdict

import fitz
import cv2
import numpy as np


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL = 'models/hvac_yolov8s_v9.pt'
DPI = 200
TILE_SIZE = 640
TILE_OVERLAP = 100
NMS_DIST = 50
DEFAULT_CONF = 0.4

COLORS = [
    (0, 200, 0),     # green
    (200, 0, 0),     # blue (BGR)
    (0, 165, 255),   # orange
    (0, 0, 200),     # red
    (255, 200, 0),   # cyan
    (255, 0, 200),   # magenta
    (0, 255, 255),   # yellow
    (180, 180, 0),   # olive
    (128, 0, 128),   # purple
    (0, 128, 128),   # teal
]


# ─── PDF HANDLING ─────────────────────────────────────────────────────────────

def render_page(pdf_path, page_idx, dpi=DPI):
    """Render a PDF page to BGR numpy array."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rotation = page.rotation
    mb_w, mb_h = page.mediabox.width, page.mediabox.height
    doc.close()
    return img, rotation, mb_w, mb_h


def display_to_annot(dx, dy, rot, mb_w, mb_h):
    """Convert display pixel coords back to annotation (mediabox) coords for adding annotations."""
    # display image is rendered AFTER rotation, so display coords need to be inverted
    if rot == 270:
        # Display: (display_w, display_h) = (mb_h, mb_w)
        # Forward: dx = ay, dy = mb_w - ax => ax = mb_w - dy, ay = dx
        return mb_w - dy, dx
    elif rot == 90:
        return dy, mb_h - dx
    elif rot == 180:
        return mb_w - dx, mb_h - dy
    return dx, dy


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def run_inference(model, img, conf=DEFAULT_CONF):
    """Tile image, run YOLO, return deduplicated detections."""
    h, w = img.shape[:2]
    step = TILE_SIZE - TILE_OVERLAP

    # Build tile list, skip empty tiles
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            xe, ye = min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)
            xs, ys = max(0, xe - TILE_SIZE), max(0, ye - TILE_SIZE)
            tile = img[ys:ye, xs:xe]
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            if (gray < 200).mean() < 0.005:
                continue
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                p = np.ones((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) * 255
                p[:tile.shape[0], :tile.shape[1]] = tile
                tile = p
            tiles.append((tile, xs, ys))

    if not tiles:
        return []

    # Inference
    dets = []
    for tile, xs, ys in tiles:
        results = model.predict(tile, conf=conf, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    'cls': model.names[int(box.cls[0])],
                    'conf': float(box.conf[0]),
                    'cx': (x1 + x2) / 2 + xs,
                    'cy': (y1 + y2) / 2 + ys,
                    'x1': x1 + xs,
                    'y1': y1 + ys,
                    'x2': x2 + xs,
                    'y2': y2 + ys,
                })

    # NMS
    final = []
    for d in sorted(dets, key=lambda x: x['conf'], reverse=True):
        if not any(abs(d['cx'] - f['cx']) < NMS_DIST and abs(d['cy'] - f['cy']) < NMS_DIST for f in final):
            final.append(d)
    return final


# ─── ANNOTATE PDF ─────────────────────────────────────────────────────────────

def annotate_pdf(input_pdf, output_pdf, detections_per_page):
    """
    Add colored rectangles to the PDF for each detection.
    detections_per_page: dict of page_idx -> list of detections
    """
    doc = fitz.open(input_pdf)

    for page_idx, dets in detections_per_page.items():
        page = doc[page_idx]
        rot = page.rotation
        mb_w, mb_h = page.mediabox.width, page.mediabox.height
        scale = DPI / 72  # pixels per PDF point

        for d in dets:
            # Convert pixel coords back to PDF points
            px_x1, px_y1 = d['x1'], d['y1']
            px_x2, px_y2 = d['x2'], d['y2']

            # Pixel → display PDF points
            disp_x1 = px_x1 / scale
            disp_y1 = px_y1 / scale
            disp_x2 = px_x2 / scale
            disp_y2 = px_y2 / scale

            # Display PDF points → annotation (mediabox) coords
            ann_x1, ann_y1 = display_to_annot(disp_x1, disp_y1, rot, mb_w, mb_h)
            ann_x2, ann_y2 = display_to_annot(disp_x2, disp_y2, rot, mb_w, mb_h)

            # Make sure rect is in proper order
            rect = fitz.Rect(
                min(ann_x1, ann_x2),
                min(ann_y1, ann_y2),
                max(ann_x1, ann_x2),
                max(ann_y1, ann_y2)
            )

            # Color (RGB 0-1) by class
            color_idx = hash(d['cls']) % len(COLORS)
            bgr = COLORS[color_idx]
            rgb = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=rgb)
            annot.set_border(width=2)
            annot.set_info(content=f"{d['cls']} ({d['conf']:.0%})")
            annot.update()

    doc.save(output_pdf)
    doc.close()


# ─── EXCEL OUTPUT ─────────────────────────────────────────────────────────────

def write_excel(output_path, detections_per_page, project_name):
    """Write Excel takeoff with counts."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print("openpyxl not installed, skipping Excel output. Install with: pip install openpyxl")
        return False

    wb = openpyxl.Workbook()

    # Sheet 1: Summary by class
    ws = wb.active
    ws.title = 'Takeoff Summary'

    # Header
    ws['A1'] = f'HVAC Takeoff: {project_name}'
    ws['A1'].font = Font(size=14, bold=True)
    ws.merge_cells('A1:D1')

    ws['A3'] = 'Equipment Type'
    ws['B3'] = 'Quantity'
    ws['C3'] = 'Pages'
    ws['D3'] = 'Avg Confidence'
    for col in ['A3', 'B3', 'C3', 'D3']:
        ws[col].font = Font(bold=True)
        ws[col].fill = PatternFill('solid', fgColor='DDDDDD')

    # Aggregate counts
    class_counts = defaultdict(int)
    class_pages = defaultdict(set)
    class_confs = defaultdict(list)
    for page_idx, dets in detections_per_page.items():
        for d in dets:
            class_counts[d['cls']] += 1
            class_pages[d['cls']].add(page_idx + 1)
            class_confs[d['cls']].append(d['conf'])

    row = 4
    total = 0
    for cls in sorted(class_counts.keys(), key=lambda c: -class_counts[c]):
        ws[f'A{row}'] = cls
        ws[f'B{row}'] = class_counts[cls]
        ws[f'C{row}'] = ', '.join(str(p) for p in sorted(class_pages[cls]))
        ws[f'D{row}'] = f"{sum(class_confs[cls]) / len(class_confs[cls]):.0%}"
        total += class_counts[cls]
        row += 1

    # Total
    ws[f'A{row+1}'] = 'TOTAL'
    ws[f'B{row+1}'] = total
    ws[f'A{row+1}'].font = Font(bold=True)
    ws[f'B{row+1}'].font = Font(bold=True)

    # Column widths
    ws.column_dimensions['A'].width = 35
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 18

    # Sheet 2: Detail (every detection)
    ws2 = wb.create_sheet('Detail by Equipment')
    ws2['A1'] = 'Page'
    ws2['B1'] = 'Equipment'
    ws2['C1'] = 'Confidence'
    ws2['D1'] = 'X (px)'
    ws2['E1'] = 'Y (px)'
    for col in ['A1', 'B1', 'C1', 'D1', 'E1']:
        ws2[col].font = Font(bold=True)
        ws2[col].fill = PatternFill('solid', fgColor='DDDDDD')

    row = 2
    for page_idx in sorted(detections_per_page.keys()):
        for d in sorted(detections_per_page[page_idx], key=lambda x: x['cls']):
            ws2[f'A{row}'] = page_idx + 1
            ws2[f'B{row}'] = d['cls']
            ws2[f'C{row}'] = f"{d['conf']:.0%}"
            ws2[f'D{row}'] = int(d['cx'])
            ws2[f'E{row}'] = int(d['cy'])
            row += 1

    ws2.column_dimensions['A'].width = 8
    ws2.column_dimensions['B'].width = 35
    ws2.column_dimensions['C'].width = 12

    wb.save(output_path)
    return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def find_mechanical_pages(pdf_path):
    """
    Heuristic: scan all pages and pick the ones likely to be mechanical floor plans.
    Looks for HVAC keywords in the page text.
    """
    doc = fitz.open(pdf_path)
    candidate_pages = []
    for pi in range(doc.page_count):
        text = doc[pi].get_text().upper()
        # Skip pure schedule/legend pages
        if any(kw in text for kw in ['MECHANICAL PLAN', 'CEILING PLAN', 'HVAC PLAN', 'VENTILATION PLAN', 'FLOOR PLAN']):
            candidate_pages.append(pi)
    doc.close()

    # If no clear matches, return all pages
    if not candidate_pages:
        return list(range(doc.page_count if doc else 0))
    return candidate_pages


def main():
    parser = argparse.ArgumentParser(description='HVAC Takeoff CLI Tool')
    parser.add_argument('pdf', help='Path to blueprint PDF')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Confidence threshold (0-1)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as PDF)')
    parser.add_argument('--all-pages', action='store_true', help='Process all pages (not just mechanical)')
    parser.add_argument('--pages', type=int, nargs='+', help='Specific page numbers (1-indexed)')
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = pdf_path.parent / f"{pdf_path.stem}_takeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"HVAC TAKEOFF — {pdf_path.name}")
    print(f"{'='*70}")
    print(f"Model:    {args.model}")
    print(f"Conf:     {args.conf}")
    print(f"Output:   {out_dir}")
    print()

    # Load model
    print("Loading model...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"  Loaded with {len(model.names)} equipment classes\n")

    # Determine which pages to process
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    if args.pages:
        pages_to_process = [p - 1 for p in args.pages]
    elif args.all_pages:
        pages_to_process = list(range(total_pages))
    else:
        pages_to_process = find_mechanical_pages(pdf_path)
        if not pages_to_process:
            pages_to_process = list(range(total_pages))

    print(f"Processing {len(pages_to_process)} page(s) of {total_pages} total\n")

    # Process each page
    detections_per_page = {}
    t_start = time.time()
    for page_idx in pages_to_process:
        t0 = time.time()
        print(f"  Page {page_idx+1}: rendering...", end=' ', flush=True)
        try:
            img, rot, mb_w, mb_h = render_page(pdf_path, page_idx)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        print(f"detecting...", end=' ', flush=True)
        dets = run_inference(model, img, conf=args.conf)
        elapsed = time.time() - t0
        print(f"{len(dets)} equipment found ({elapsed:.0f}s)")
        if dets:
            detections_per_page[page_idx] = dets

    total_elapsed = time.time() - t_start
    print(f"\nDetection complete in {total_elapsed:.0f}s")
    print()

    # Aggregate
    total_count = sum(len(d) for d in detections_per_page.values())
    if total_count == 0:
        print("No HVAC equipment detected in the selected pages.")
        sys.exit(0)

    class_counts = defaultdict(int)
    for dets in detections_per_page.values():
        for d in dets:
            class_counts[d['cls']] += 1

    # Print summary
    print(f"{'='*70}")
    print(f"TAKEOFF SUMMARY")
    print(f"{'='*70}")
    print(f"  Total equipment detected: {total_count}")
    print(f"  Pages with equipment:     {len(detections_per_page)}")
    print()
    print(f"  {'Equipment Type':<35} {'Count':>8}")
    print(f"  {'-'*35} {'-'*8}")
    for cls in sorted(class_counts.keys(), key=lambda c: -class_counts[c]):
        print(f"  {cls:<35} {class_counts[cls]:>8}")
    print()

    # Output files
    annotated_pdf_path = out_dir / f"{pdf_path.stem}_annotated.pdf"
    excel_path = out_dir / f"{pdf_path.stem}_takeoff.xlsx"

    print(f"Writing outputs...")
    print(f"  Annotated PDF:  {annotated_pdf_path}")
    annotate_pdf(str(pdf_path), str(annotated_pdf_path), detections_per_page)

    print(f"  Excel takeoff:  {excel_path}")
    write_excel(str(excel_path), detections_per_page, pdf_path.stem)

    print(f"\n{'='*70}")
    print(f"DONE — open {out_dir} to see the results")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
