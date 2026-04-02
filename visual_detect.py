"""
HVAC Visual Symbol Detection - Phase 1.5
Detects equipment by finding visual patterns (circles, squares) on floor plans,
then reads the tag letter inside each symbol using OCR.

Approach:
  1. Hough Circle detection → finds circle-with-letter tags (A, B types)
  2. Interior OCR → reads the letter (A/B/C/D) and CFM value inside each circle
  3. Square detection → finds square grille/diffuser symbols
  4. Scores against ground truth annotations
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import fitz
import cv2
import numpy as np
import easyocr
import os
import time
from collections import defaultdict


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR = r"C:\Users\JFL\Downloads\Triune\data to train\Plans_Specs"
OUTPUT_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\output"
DPI = 200  # Balance of detail vs speed


# ─── COORDINATE TRANSFORMS ───────────────────────────────────────────────────

def annot_to_display(ax, ay, rotation, mb_w, mb_h):
    """Convert annotation (mediabox) coords to display coords."""
    if rotation == 270:
        return ay, mb_w - ax
    elif rotation == 90:
        return mb_h - ay, ax
    elif rotation == 180:
        return mb_w - ax, mb_h - ay
    return ax, ay


def display_to_annot(dx, dy, rotation, mb_w, mb_h):
    """Convert display coords back to annotation (mediabox) coords."""
    if rotation == 270:
        return mb_w - dy, dx
    elif rotation == 90:
        return dy, mb_h - dx
    elif rotation == 180:
        return mb_w - dx, mb_h - dy
    return dx, dy


# ─── GROUND TRUTH EXTRACTION ─────────────────────────────────────────────────

def extract_ground_truth(pdf_path):
    """Extract human annotations, converting to display coordinates."""
    doc = fitz.open(pdf_path)
    annotations = []

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        rotation = page.rotation
        mb = page.mediabox
        mb_w, mb_h = mb.width, mb.height

        annots = list(page.annots()) if page.annots() else []
        for a in annots:
            if a.type[1] == 'Highlight':
                continue
            subject = a.info.get('subject', '').strip()
            content = a.info.get('content', '').strip()
            if not subject or not content:
                continue

            rect = a.rect
            # Convert annotation center to display coords
            acx = (rect.x0 + rect.x1) / 2
            acy = (rect.y0 + rect.y1) / 2
            dcx, dcy = annot_to_display(acx, acy, rotation, mb_w, mb_h)

            annotations.append({
                'page': page_idx + 1,
                'product_type': subject,
                'tag': content,
                'display_cx': dcx,
                'display_cy': dcy,
            })

    doc.close()
    return annotations


# ─── RENDER PAGE ──────────────────────────────────────────────────────────────

def render_page(pdf_path, page_idx, dpi=DPI):
    """Render page to BGR image, return image + page metadata."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    meta = {
        'rotation': page.rotation,
        'mb_w': page.mediabox.width,
        'mb_h': page.mediabox.height,
        'display_w': page.rect.width,
        'display_h': page.rect.height,
    }
    doc.close()
    return img, meta


# ─── CIRCLE DETECTION ─────────────────────────────────────────────────────────

def detect_circles(gray, dpi=DPI):
    """
    Find circles on the floor plan using Hough transform.
    Equipment tag circles are typically 0.15-0.4 inches in diameter.
    """
    # Circle radius in pixels at given DPI
    min_radius = int(0.10 * dpi)   # ~0.10 inch = 20px at 200dpi
    max_radius = int(0.35 * dpi)   # ~0.35 inch = 70px at 200dpi
    min_dist = int(0.3 * dpi)      # Min distance between circle centers

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    results = []
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        results.append((cx, cy, r))

    return results


# ─── CLASSIFY CIRCLE CONTENTS ─────────────────────────────────────────────────

def classify_circle(img_bgr, cx, cy, r, reader):
    """
    Crop the interior of a detected circle and OCR it to find:
    - Tag letter (A, B, C, D)
    - CFM value (number below the letter)
    """
    h, w = img_bgr.shape[:2]
    pad = int(r * 0.2)
    x1 = max(0, cx - r - pad)
    y1 = max(0, cy - r - pad)
    x2 = min(w, cx + r + pad)
    y2 = min(h, cy + r + pad)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    # Preprocess: convert to grayscale, threshold for clean text
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_crop, 160, 255, cv2.THRESH_BINARY)

    # Scale up for better OCR on small text
    scale = max(1, 80 // max(crop.shape[0], 1))
    if scale > 1:
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Run OCR on the cropped circle interior
    results = reader.readtext(binary, allowlist='ABCD0123456789', paragraph=False)

    tag_letter = None
    cfm_value = None

    for bbox, text, conf in results:
        text = text.strip().upper()
        if conf < 0.3:
            continue
        if text in ('A', 'B', 'C', 'D'):
            tag_letter = text
        elif text.isdigit():
            cfm_value = int(text)

    return tag_letter, cfm_value


# ─── SQUARE/RECTANGLE DETECTION ──────────────────────────────────────────────

def detect_squares(gray, dpi=DPI):
    """
    Find square symbols (diffusers, grilles) on the floor plan.
    These are typically squares with internal diagonal lines or X patterns.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Target size: equipment symbols are typically 0.15-0.5 inches
    min_side = int(0.12 * dpi)
    max_side = int(0.55 * dpi)

    squares = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Looking for roughly rectangular shapes (4 vertices)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = max(w, h) / max(min(w, h), 1)

            if min_side <= w <= max_side and min_side <= h <= max_side and aspect < 1.6:
                # Check if it has internal content (diagonals/X pattern)
                roi = gray[y:y+h, x:x+w]
                edge_roi = cv2.Canny(roi, 50, 150)
                edge_density = edge_roi.mean() / 255

                # Real equipment symbols have internal lines (X or diagonal)
                if edge_density > 0.08:
                    squares.append((x + w//2, y + h//2, w, h, edge_density))

    return squares


# ─── MAIN DETECTION PIPELINE ─────────────────────────────────────────────────

SCHEDULE_TAG_MAP = {
    'A': 'AD-T-BAR SUPPLY',
    'B': 'AD-T-BAR RETURN',
    'C': 'AD-SURF SUPPLY',
    'D': 'AD-SURF RETURN',
}


def detect_equipment(pdf_path, page_idx, reader):
    """Run full visual detection on a single page."""
    img, meta = render_page(pdf_path, page_idx)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = DPI / 72  # pixels per PDF point

    detections = []

    # 1. Detect circles (tag bubbles with A/B/C/D letters)
    circles = detect_circles(gray)
    print(f"  Circles found: {len(circles)}")

    for cx, cy, r in circles:
        tag_letter, cfm = classify_circle(img, cx, cy, r, reader)
        if tag_letter:
            # Convert pixel coords to display PDF coords
            pdf_cx = cx / scale
            pdf_cy = cy / scale
            product_type = SCHEDULE_TAG_MAP.get(tag_letter, f'UNKNOWN-{tag_letter}')
            detections.append({
                'method': 'circle+ocr',
                'tag': tag_letter,
                'product_type': product_type,
                'cfm': cfm,
                'display_cx': pdf_cx,
                'display_cy': pdf_cy,
                'radius_px': r,
                'px_cx': cx,
                'px_cy': cy,
            })

    # 2. Detect squares (diffuser/grille symbols)
    squares = detect_squares(gray)
    print(f"  Squares found: {len(squares)}")

    # Note: squares don't have tag letters inside them usually,
    # they're the visual symbol next to the circle tag.
    # We record them for cross-referencing but don't count as primary detections.

    print(f"  Tagged detections (circle+OCR): {len(detections)}")
    det_by_tag = defaultdict(int)
    for d in detections:
        det_by_tag[d['product_type']] += 1
    for pt, count in sorted(det_by_tag.items()):
        print(f"    {pt:30s} : {count}")

    return detections, circles, squares, img, meta


# ─── ACCURACY SCORING ─────────────────────────────────────────────────────────

def score_accuracy(detections, ground_truth, match_radius_pts=80):
    """Compare detections vs ground truth in display PDF coords."""
    matched_gt = set()
    matched_det = set()
    matches = []

    for di, det in enumerate(detections):
        best_dist = float('inf')
        best_gi = None

        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if det['product_type'] != gt['product_type']:
                continue

            dist = ((det['display_cx'] - gt['display_cx'])**2 +
                    (det['display_cy'] - gt['display_cy'])**2)**0.5

            if dist < best_dist and dist <= match_radius_pts:
                best_dist = dist
                best_gi = gi

        if best_gi is not None:
            matched_gt.add(best_gi)
            matched_det.add(di)
            matches.append({
                'det_idx': di,
                'gt_idx': best_gi,
                'distance': best_dist,
                'product_type': det['product_type'],
                'det_tag': det['tag'],
                'gt_tag': ground_truth[best_gi]['tag'],
            })

    fp = [detections[i] for i in range(len(detections)) if i not in matched_det]
    fn = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt]

    tp = len(matches)
    precision = tp / max(tp + len(fp), 1)
    recall = tp / max(tp + len(fn), 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': len(fp), 'fn': len(fn),
        'matches': matches, 'fp_list': fp, 'fn_list': fn,
    }


# ─── ANNOTATE OUTPUT ──────────────────────────────────────────────────────────

def annotate_image(img, detections, circles, squares, ground_truth, meta):
    """Draw detections and ground truth on the image for visual comparison."""
    vis = img.copy()
    scale = DPI / 72

    # Draw all detected circles in blue
    for cx, cy, r in circles:
        cv2.circle(vis, (cx, cy), r, (255, 180, 0), 2)

    # Draw tagged detections (circle+OCR matched) in green
    for det in detections:
        px = det['px_cx']
        py = det['px_cy']
        r = det['radius_px']
        cv2.circle(vis, (px, py), r + 5, (0, 255, 0), 3)
        label = f"{det['tag']}"
        if det['cfm']:
            label += f" {det['cfm']}CFM"
        cv2.putText(vis, label, (px + r + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw ground truth in red
    for gt in ground_truth:
        px = int(gt['display_cx'] * scale)
        py = int(gt['display_cy'] * scale)
        cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis, gt['tag'][:5], (px + 15, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw detected squares in yellow
    for sx, sy, sw, sh, _ in squares:
        cv2.rectangle(vis, (sx - sw//2, sy - sh//2), (sx + sw//2, sy + sh//2), (0, 255, 255), 2)

    return vis


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading EasyOCR model...")
    t0 = time.time()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    files = os.listdir(DATA_DIR)
    pdfs = sorted([f for f in files if f.endswith('.pdf') and 'Final' not in f])

    all_scores = []

    for pdf_name in pdfs:
        name = pdf_name.replace('3.11.26 EAG - 9530 Towne Center Drive ', '').replace('.pdf', '')
        final_name = pdf_name.replace('.pdf', ' - Final.pdf')
        if not os.path.exists(os.path.join(DATA_DIR, final_name)):
            final_name = pdf_name.replace('.pdf', '- Final.pdf')

        final_path = os.path.join(DATA_DIR, final_name)
        if not os.path.exists(final_path):
            print(f"SKIP {name}: no Final PDF")
            continue

        print(f"{'='*60}")
        print(f"PROJECT: {name}")
        print(f"{'='*60}")

        # Ground truth
        all_gt = extract_ground_truth(final_path)
        # Find main annotated page (page with most annotations)
        page_counts = defaultdict(int)
        for g in all_gt:
            page_counts[g['page']] += 1
        main_page = max(page_counts, key=page_counts.get)
        gt = [g for g in all_gt if g['page'] == main_page]
        print(f"Ground truth: {len(gt)} equipment on page {main_page}")

        # Detect
        pdf_path = os.path.join(DATA_DIR, pdf_name)
        print(f"Detecting on page {main_page}...")
        t0 = time.time()
        detections, circles, squares, img, meta = detect_equipment(
            pdf_path, main_page - 1, reader
        )
        elapsed = time.time() - t0
        print(f"  Detection time: {elapsed:.1f}s")

        # Score
        scores = score_accuracy(detections, gt)
        print(f"\n  ACCURACY:")
        print(f"    Precision: {scores['precision']:.1%}")
        print(f"    Recall:    {scores['recall']:.1%}")
        print(f"    F1:        {scores['f1']:.1%}")
        print(f"    TP={scores['tp']} FP={scores['fp']} FN={scores['fn']}")

        if scores['fn_list']:
            # Count missed by type
            fn_types = defaultdict(int)
            for m in scores['fn_list']:
                fn_types[m['product_type']] += 1
            print(f"    Missed by type: {dict(fn_types)}")

        # Save annotated visualization
        vis = annotate_image(img, detections, circles, squares, gt, meta)
        safe_name = name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
        vis_path = os.path.join(OUTPUT_DIR, f'visual_{safe_name}.png')
        # Downscale for reasonable file size
        vis_small = cv2.resize(vis, (vis.shape[1]//2, vis.shape[0]//2))
        cv2.imwrite(vis_path, vis_small)
        print(f"  Saved: {vis_path}")

        scores['project'] = name
        all_scores.append(scores)
        print()

    # Overall summary
    print(f"{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    total_tp = sum(s['tp'] for s in all_scores)
    total_fp = sum(s['fp'] for s in all_scores)
    total_fn = sum(s['fn'] for s in all_scores)
    p = total_tp / max(total_tp + total_fp, 1)
    r = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * p * r / max(p + r, 0.001)
    print(f"  Projects: {len(all_scores)}")
    print(f"  Precision: {p:.1%}")
    print(f"  Recall:    {r:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"\n  Previous (OCR-only): Precision=69.6% Recall=12.0% F1=20.5%")


if __name__ == "__main__":
    main()
