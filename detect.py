"""
HVAC Symbol Detection - Phase 1
Matches extracted legend templates against floor plan pages.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import fitz
import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PDF_PATH = r"C:\Users\JFL\Downloads\Triune\PLANS VENTILAITON.pdf"
TEMPLATE_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\templates"
OUTPUT_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\output"
DPI = 200
THRESHOLD = 0.70       # Match confidence threshold
NMS_DISTANCE = 25      # Min px between detections
SCAN_PAGES = [5, 6]    # 0-indexed (pages 6 & 7 = new construction floor plans)


def render_page(pdf_path, page_idx, dpi=DPI):
    """Render a single PDF page to BGR numpy array."""
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_idx].get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def match_template(page_gray, tmpl_gray, threshold=THRESHOLD):
    """Run template matching at multiple scales, return matches."""
    th, tw = tmpl_gray.shape[:2]
    if th < 8 or tw < 8:
        return []

    all_matches = []
    for scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
        sw = int(tw * scale)
        sh = int(th * scale)
        if sw < 8 or sh < 8:
            continue
        if sh >= page_gray.shape[0] or sw >= page_gray.shape[1]:
            continue

        scaled = cv2.resize(tmpl_gray, (sw, sh))
        result = cv2.matchTemplate(page_gray, scaled, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)

        for py, px in zip(*locs):
            conf = float(result[py, px])
            all_matches.append((int(px), int(py), sw, sh, conf))

    return all_matches


def nms(matches, min_dist=NMS_DISTANCE):
    """Non-maximum suppression by distance."""
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m[4], reverse=True)
    kept = []
    for m in matches:
        cx, cy = m[0] + m[2]//2, m[1] + m[3]//2
        skip = False
        for k in kept:
            kx, ky = k[0] + k[2]//2, k[1] + k[3]//2
            if abs(cx - kx) < min_dist and abs(cy - ky) < min_dist:
                skip = True
                break
        if not skip:
            kept.append(m)
    return kept


def run_detection():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load symbol name mapping
    map_path = os.path.join(TEMPLATE_DIR, "symbol_map.json")
    with open(map_path, "r", encoding="utf-8") as f:
        name_map = json.load(f)

    # Load templates
    templates = {}
    for fname in os.listdir(TEMPLATE_DIR):
        if not fname.endswith(".png"):
            continue
        key = fname.replace(".png", "")
        img = cv2.imread(os.path.join(TEMPLATE_DIR, fname))
        if img is not None:
            templates[key] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Loaded {len(templates)} templates")
    print(f"Scanning pages: {[p+1 for p in SCAN_PAGES]}")
    print(f"Threshold: {THRESHOLD}")
    print()

    # For each floor plan page, match all templates
    all_results = []
    all_detections = {}  # page_idx -> {symbol_name -> [(x,y,w,h,conf)]}

    for page_idx in SCAN_PAGES:
        print(f"--- Page {page_idx + 1} ---")
        page_img = render_page(PDF_PATH, page_idx)
        page_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

        page_detections = {}

        for key, tmpl in templates.items():
            display_name = name_map.get(key, key)
            matches = match_template(page_gray, tmpl, THRESHOLD)
            matches = nms(matches)

            if matches:
                page_detections[key] = matches
                print(f"  {display_name:40s} : {len(matches)} found  (best conf: {max(m[4] for m in matches):.3f})")
                all_results.append({
                    "Page": page_idx + 1,
                    "Symbol": display_name,
                    "Count": len(matches),
                    "Best_Confidence": round(max(m[4] for m in matches), 3),
                    "Avg_Confidence": round(sum(m[4] for m in matches) / len(matches), 3),
                })

        all_detections[page_idx] = page_detections
        total = sum(len(v) for v in page_detections.values())
        print(f"  TOTAL: {total} detections on page {page_idx + 1}")
        print()

    # ─── Annotate PDF ─────────────────────────────────────────────────────
    print("Annotating PDF...")
    doc = fitz.open(PDF_PATH)
    scale = 72 / DPI

    colors = [
        (1, 0, 0), (0, 0.7, 0), (0, 0, 1), (1, 0.5, 0),
        (0.7, 0, 0.7), (0, 0.7, 0.7), (1, 0, 0.5), (0.5, 0.5, 0),
        (0.8, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.8), (1, 0.7, 0),
    ]

    for page_idx, detections in all_detections.items():
        page = doc[page_idx]
        for sym_key, matches in detections.items():
            color = colors[hash(sym_key) % len(colors)]
            display_name = name_map.get(sym_key, sym_key)

            for (x, y, w, h, conf) in matches:
                rect = fitz.Rect(x * scale, y * scale, (x+w) * scale, (y+h) * scale)
                # Draw colored rectangle
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color)
                annot.set_border(width=1.5)
                annot.set_opacity(0.8)
                annot.set_info(content=f"{display_name} ({conf:.2f})")
                annot.update()

    out_pdf = os.path.join(OUTPUT_DIR, "takeoff_annotated.pdf")
    doc.save(out_pdf)
    doc.close()
    print(f"  Saved: {out_pdf}")

    # ─── Generate Reports ─────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)

        # Summary by symbol
        summary = df.groupby("Symbol").agg(
            Total=("Count", "sum"),
            Pages=("Page", lambda x: ", ".join(str(p) for p in sorted(set(x)))),
            Best_Conf=("Best_Confidence", "max"),
        ).sort_values("Total", ascending=False)

        csv_path = os.path.join(OUTPUT_DIR, "takeoff_counts.csv")
        xlsx_path = os.path.join(OUTPUT_DIR, "takeoff_report.xlsx")

        summary.to_csv(csv_path)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary")
            df.to_excel(writer, sheet_name="By_Page", index=False)

        print(f"  CSV:   {csv_path}")
        print(f"  Excel: {xlsx_path}")

        print(f"\n{'='*60}")
        print("TAKEOFF SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string())
    else:
        print("No detections found.")


if __name__ == "__main__":
    run_detection()
