"""
HVAC Blueprint Takeoff Tool - Phase 1 Prototype
Reads HVAC ventilation PDFs, parses legend, detects symbols, outputs counts.
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DPI = 200  # Render resolution
MATCH_THRESHOLD = 0.75  # Template matching confidence threshold
NMS_DISTANCE = 20  # Non-max suppression: min pixels between detections
LEGEND_PAGE_INDEX = 1  # 0-based index of legend page
FLOOR_PLAN_PAGES = [5, 6]  # 0-based indices of pages to scan (pages 6-7)


# ─── STEP 1: PDF TO IMAGES ───────────────────────────────────────────────────

def pdf_to_images(pdf_path, dpi=DPI):
    """Convert PDF pages to numpy arrays (BGR for OpenCV)."""
    doc = fitz.open(pdf_path)
    images = {}
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # RGBA → BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB → BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images[i] = img
    doc.close()
    return images


# ─── STEP 2: LEGEND PARSING ──────────────────────────────────────────────────

def extract_legend_symbols(legend_img, output_dir):
    """
    Extract individual symbols from the legend page.
    Strategy: Find the legend region, isolate symbol graphics,
    crop each one as a template for matching.
    """
    gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # The legend is typically in the left portion of the page
    # Crop to left ~40% and top ~70% where symbols live
    legend_region = gray[0:int(h * 0.75), 0:int(w * 0.40)]
    legend_color = legend_img[0:int(h * 0.75), 0:int(w * 0.40)]

    # Threshold to get black content on white background
    _, binary = cv2.threshold(legend_region, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours - these are the drawn elements
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    symbol_dir = os.path.join(output_dir, "legend_symbols")
    os.makedirs(symbol_dir, exist_ok=True)

    # Filter contours by size to find symbol-sized objects
    # Typical HVAC symbols on a blueprint at 200 DPI are roughly 30-120 px
    for i, cnt in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Filter: symbols are roughly square-ish and a specific size range
        area = cw * ch
        aspect = max(cw, ch) / max(min(cw, ch), 1)

        if 400 < area < 15000 and aspect < 4:
            # Add padding around the symbol
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(legend_region.shape[1], x + cw + pad)
            y2 = min(legend_region.shape[0], y + ch + pad)

            symbol_crop = legend_color[y1:y2, x1:x2]
            if symbol_crop.size > 0:
                symbols.append({
                    "id": i,
                    "bbox": (x1, y1, x2, y2),
                    "size": (x2 - x1, y2 - y1),
                    "area": area,
                })
                cv2.imwrite(
                    os.path.join(symbol_dir, f"symbol_{i:03d}.png"),
                    symbol_crop
                )

    print(f"  Extracted {len(symbols)} candidate symbols from legend")
    return symbols, symbol_dir


def extract_legend_text(pdf_path, page_idx=LEGEND_PAGE_INDEX):
    """Extract text blocks from legend page with positions."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    blocks = page.get_text("dict")["blocks"]
    doc.close()

    text_items = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"]).strip()
                if text:
                    bbox = line["bbox"]
                    text_items.append({
                        "text": text,
                        "x": bbox[0],
                        "y": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                    })
    return text_items


# ─── STEP 3: TEMPLATE MATCHING ───────────────────────────────────────────────

def match_template_multiscale(page_img, template, threshold=MATCH_THRESHOLD):
    """
    Match a template against a page image at multiple scales.
    Returns list of (x, y, w, h, confidence) matches.
    """
    gray_page = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    th, tw = gray_tmpl.shape[:2]
    if th < 10 or tw < 10:
        return []

    all_matches = []

    # Try multiple scales (symbols may vary slightly in size)
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        scaled_w = int(tw * scale)
        scaled_h = int(th * scale)
        if scaled_w < 10 or scaled_h < 10:
            continue

        scaled_tmpl = cv2.resize(gray_tmpl, (scaled_w, scaled_h))

        if scaled_h > gray_page.shape[0] or scaled_w > gray_page.shape[1]:
            continue

        result = cv2.matchTemplate(gray_page, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt_y, pt_x in zip(*locations):
            confidence = result[pt_y, pt_x]
            all_matches.append((pt_x, pt_y, scaled_w, scaled_h, float(confidence)))

    return all_matches


def non_max_suppression(matches, distance=NMS_DISTANCE):
    """Remove duplicate detections that are too close together."""
    if not matches:
        return []

    # Sort by confidence (descending)
    matches = sorted(matches, key=lambda m: m[4], reverse=True)
    kept = []

    for match in matches:
        mx, my = match[0] + match[2] // 2, match[1] + match[3] // 2
        too_close = False
        for kept_match in kept:
            kx, ky = kept_match[0] + kept_match[2] // 2, kept_match[1] + kept_match[3] // 2
            if abs(mx - kx) < distance and abs(my - ky) < distance:
                too_close = True
                break
        if not too_close:
            kept.append(match)

    return kept


# ─── STEP 4: ANNOTATE PDF ────────────────────────────────────────────────────

def annotate_pdf(pdf_path, output_path, detections_by_page, dpi=DPI):
    """
    Create a copy of the PDF with colored rectangles around detected symbols.
    """
    doc = fitz.open(pdf_path)
    scale = 72 / dpi  # Convert pixel coords back to PDF points

    colors = [
        (1, 0, 0),      # red
        (0, 0.7, 0),    # green
        (0, 0, 1),      # blue
        (1, 0.5, 0),    # orange
        (0.7, 0, 0.7),  # purple
        (0, 0.7, 0.7),  # teal
        (1, 0, 0.5),    # pink
        (0.5, 0.5, 0),  # olive
    ]

    for page_idx, detections in detections_by_page.items():
        page = doc[page_idx]
        for symbol_name, matches in detections.items():
            color_idx = hash(symbol_name) % len(colors)
            color = colors[color_idx]
            for (x, y, w, h, conf) in matches:
                # Convert pixel coords to PDF points
                rect = fitz.Rect(
                    x * scale, y * scale,
                    (x + w) * scale, (y + h) * scale
                )
                # Draw rectangle annotation
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color)
                annot.set_border(width=2)
                annot.set_opacity(0.7)
                annot.update()

    doc.save(output_path)
    doc.close()
    print(f"  Annotated PDF saved: {output_path}")


# ─── STEP 5: GENERATE REPORT ─────────────────────────────────────────────────

def generate_report(detections_by_page, output_dir):
    """Generate CSV/Excel summary of detected symbols."""
    rows = []
    for page_idx, detections in detections_by_page.items():
        for symbol_name, matches in detections.items():
            rows.append({
                "Page": page_idx + 1,
                "Symbol": symbol_name,
                "Count": len(matches),
                "Avg_Confidence": round(
                    sum(m[4] for m in matches) / len(matches), 3
                ) if matches else 0,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Summary pivot
        summary = df.groupby("Symbol").agg(
            Total_Count=("Count", "sum"),
            Pages_Found=("Page", lambda x: ", ".join(str(p) for p in sorted(x))),
            Avg_Confidence=("Avg_Confidence", "mean"),
        ).round(3).sort_values("Total_Count", ascending=False)

        csv_path = os.path.join(output_dir, "takeoff_counts.csv")
        xlsx_path = os.path.join(output_dir, "takeoff_report.xlsx")

        summary.to_csv(csv_path)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary")
            df.to_excel(writer, sheet_name="Detail_By_Page", index=False)

        print(f"  CSV report: {csv_path}")
        print(f"  Excel report: {xlsx_path}")
        print(f"\n{'='*60}")
        print("TAKEOFF SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string())
        return summary
    else:
        print("  No detections found.")
        return pd.DataFrame()


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_takeoff(pdf_path, legend_page=LEGEND_PAGE_INDEX,
                scan_pages=FLOOR_PLAN_PAGES, threshold=MATCH_THRESHOLD):
    """Full takeoff pipeline."""
    pdf_path = str(Path(pdf_path).resolve())
    output_dir = os.path.join(os.path.dirname(pdf_path), "hvac-takeoff-tool", "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"HVAC Takeoff Tool - Phase 1")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Legend page: {legend_page + 1}")
    print(f"Scanning pages: {[p + 1 for p in scan_pages]}")
    print(f"Match threshold: {threshold}")
    print()

    # Step 1: Convert PDF to images
    print("[1/5] Converting PDF to images...")
    images = pdf_to_images(pdf_path)
    print(f"  Converted {len(images)} pages")

    # Step 2: Extract legend symbols
    print("[2/5] Parsing legend...")
    legend_img = images[legend_page]
    symbols, symbol_dir = extract_legend_symbols(legend_img, output_dir)
    legend_text = extract_legend_text(pdf_path, legend_page)

    # Map symbols to their nearest text label
    # (symbols on the legend page sit next to their description)
    symbol_files = sorted(Path(symbol_dir).glob("symbol_*.png"))
    print(f"  Symbol templates saved to: {symbol_dir}")

    # Step 3: Template matching on floor plan pages
    print("[3/5] Scanning floor plans for symbols...")
    detections_by_page = {}

    for page_idx in scan_pages:
        if page_idx not in images:
            print(f"  Skipping page {page_idx + 1} (not found)")
            continue

        page_img = images[page_idx]
        page_detections = {}

        for sym_file in symbol_files:
            template = cv2.imread(str(sym_file))
            if template is None:
                continue

            sym_name = sym_file.stem
            matches = match_template_multiscale(page_img, template, threshold)
            matches = non_max_suppression(matches)

            if matches:
                page_detections[sym_name] = matches

        total = sum(len(m) for m in page_detections.values())
        print(f"  Page {page_idx + 1}: {total} detections across {len(page_detections)} symbol types")
        detections_by_page[page_idx] = page_detections

    # Step 4: Annotate PDF
    print("[4/5] Annotating PDF...")
    annotated_path = os.path.join(output_dir, "takeoff_annotated.pdf")
    annotate_pdf(pdf_path, annotated_path, detections_by_page)

    # Step 5: Generate report
    print("[5/5] Generating report...")
    summary = generate_report(detections_by_page, output_dir)

    print(f"\n{'='*60}")
    print("DONE. Output files in:", output_dir)
    return summary


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the sample file
        pdf_file = r"C:\Users\JFL\Downloads\Triune\PLANS VENTILAITON.pdf"
    else:
        pdf_file = sys.argv[1]

    run_takeoff(pdf_file)
