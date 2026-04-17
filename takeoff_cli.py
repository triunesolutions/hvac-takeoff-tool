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

from tag_extractor import summarize_detections_by_tag
from tag_inference import infer_tags
from schedule_parser import parse_pdf_schedules


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

    # Headers: PRODUCT | TAG | QTY | Pages | Avg Confidence
    ws['A3'] = 'PRODUCT'
    ws['B3'] = 'TAG'
    ws['C3'] = 'QTY'
    ws['D3'] = 'Pages'
    ws['E3'] = 'Avg Confidence'
    for col in ['A3', 'B3', 'C3', 'D3', 'E3']:
        ws[col].font = Font(bold=True)
        ws[col].fill = PatternFill('solid', fgColor='DDDDDD')

    # Group by (class, tag)
    grouped = defaultdict(lambda: {'count': 0, 'pages': set(), 'confs': []})
    for page_idx, dets in detections_per_page.items():
        for d in dets:
            cls = d['cls']
            tag = d.get('tag') or '(no-tag)'
            key = (cls, tag)
            grouped[key]['count'] += 1
            grouped[key]['pages'].add(page_idx + 1)
            grouped[key]['confs'].append(d['conf'])

    row = 4
    total = 0
    current_cls = None
    # Sort by class then by tag (mimics team's Excel grouping)
    for (cls, tag), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        # Print class name on first row of each group (like team's Excel)
        ws[f'A{row}'] = cls if cls != current_cls else ''
        current_cls = cls
        ws[f'B{row}'] = tag if tag != '(no-tag)' else ''
        ws[f'C{row}'] = data['count']
        ws[f'D{row}'] = ', '.join(str(p) for p in sorted(data['pages']))
        ws[f'E{row}'] = f"{sum(data['confs']) / len(data['confs']):.0%}"
        total += data['count']
        row += 1

    # Total
    ws[f'A{row+1}'] = 'TOTAL'
    ws[f'C{row+1}'] = total
    ws[f'A{row+1}'].font = Font(bold=True)
    ws[f'C{row+1}'].font = Font(bold=True)

    # Column widths
    ws.column_dimensions['A'].width = 32
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 8
    ws.column_dimensions['D'].width = 15
    ws.column_dimensions['E'].width = 18

    # Sheet 2: Detail (every detection)
    ws2 = wb.create_sheet('Detail by Equipment')
    ws2['A1'] = 'Page'
    ws2['B1'] = 'Equipment'
    ws2['C1'] = 'Tag'
    ws2['D1'] = 'Confidence'
    ws2['E1'] = 'X (px)'
    ws2['F1'] = 'Y (px)'
    for col in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1']:
        ws2[col].font = Font(bold=True)
        ws2[col].fill = PatternFill('solid', fgColor='DDDDDD')

    row = 2
    for page_idx in sorted(detections_per_page.keys()):
        for d in sorted(detections_per_page[page_idx], key=lambda x: (x['cls'], x.get('tag') or '')):
            ws2[f'A{row}'] = page_idx + 1
            ws2[f'B{row}'] = d['cls']
            ws2[f'C{row}'] = d.get('tag') or ''
            ws2[f'D{row}'] = f"{d['conf']:.0%}"
            ws2[f'E{row}'] = int(d['cx'])
            ws2[f'F{row}'] = int(d['cy'])
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

    # Parse schedule first — need tags for inference
    print("Parsing schedule...")
    try:
        schedules, marks, mark_details, legend, sched_summary = parse_pdf_schedules(str(pdf_path))
        print(f"  Found {len(marks)} GRD tags: {marks[:10]}{'...' if len(marks)>10 else ''}")
    except Exception as e:
        print(f"  Schedule parse failed: {e}")
        schedules, marks, mark_details = [], [], {}
    print()

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
        print(f"{len(dets)} found ({elapsed:.0f}s)")
        if dets:
            detections_per_page[page_idx] = dets

    total_elapsed = time.time() - t_start
    print(f"\nDetection complete in {total_elapsed:.0f}s")

    # Tag inference — 3-level system
    if detections_per_page:
        print("\nInferring tags...")
        detections_per_page, tag_stats = infer_tags(
            detections_per_page, schedules, marks, mark_details, str(pdf_path)
        )
        print(f"  Tagged: {tag_stats['tagged']}/{tag_stats['total']} ({tag_stats['tagged_pct']:.0f}%)")
        for ls in tag_stats.get('levels', []):
            if ls.get('tagged', 0) > 0 or ls.get('mapping'):
                print(f"  Level {ls.get('level', '?')}: {ls.get('method', '')} — {ls}")
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
    all_dets = [d for dets in detections_per_page.values() for d in dets]
    tag_summary = summarize_detections_by_tag(all_dets)
    tagged = sum(1 for d in all_dets if d.get('tag'))

    print(f"{'='*75}")
    print(f"TAKEOFF SUMMARY")
    print(f"{'='*75}")
    print(f"  Total equipment detected: {total_count}")
    print(f"  Tagged:                   {tagged} / {total_count} ({tagged/max(total_count,1)*100:.0f}%)")
    print(f"  Pages with equipment:     {len(detections_per_page)}")
    print()
    print(f"  {'Equipment Type':<30} {'Tag':<15} {'Count':>8}")
    print(f"  {'-'*30} {'-'*15} {'-'*8}")
    for row in tag_summary:
        tag_disp = row['tag'] if row['tag'] != '(no-tag)' else '—'
        print(f"  {row['class'][:29]:<30} {tag_disp:<15} {row['count']:>8}")
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
