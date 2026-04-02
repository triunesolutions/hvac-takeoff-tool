"""
HVAC Takeoff - OCR-based Equipment Detection + Accuracy Scoring
Phase 1: Detect equipment by reading text tags from blueprints,
then score against human annotations in "Final" PDFs.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import fitz
import pandas as pd
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR = r"C:\Users\JFL\Downloads\Triune\data to train\Plans_Specs"
OUTPUT_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\output"


# ─── STEP 1: EXTRACT GROUND TRUTH FROM ANNOTATED PDFs ────────────────────────

def extract_ground_truth(pdf_path):
    """
    Extract human annotations from a 'Final' PDF.
    Returns list of dicts with: page, product_type, tag, x, y, w, h
    """
    doc = fitz.open(pdf_path)
    annotations = []

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        annots = list(page.annots()) if page.annots() else []
        for a in annots:
            atype = a.type[1]
            # Skip highlights (those mark schedule headers, not equipment)
            if atype == 'Highlight':
                continue

            subject = a.info.get('subject', '').strip()
            content = a.info.get('content', '').strip()
            rect = a.rect

            if subject and content:
                annotations.append({
                    'page': page_idx + 1,
                    'product_type': subject,
                    'tag': content,
                    'x': rect.x0,
                    'y': rect.y0,
                    'x2': rect.x1,
                    'y2': rect.y1,
                    'cx': (rect.x0 + rect.x1) / 2,
                    'cy': (rect.y0 + rect.y1) / 2,
                })

    doc.close()
    return annotations


# ─── STEP 2: OCR-BASED TAG DETECTION ─────────────────────────────────────────

# Tag patterns found in HVAC drawings
# Format: letter(s) followed by optional number, or specific known tags
TAG_PATTERNS = [
    # Single letter tags from schedule (A=supply diffuser, B=return grille, C=round supply, D=return)
    (r'\b([A-D])\b', 'schedule_tag'),
    # Linear diffuser/plenum tags
    (r'\b(LD-\d[\w\-]*)', 'linear_diffuser'),
    (r'\b(LR-\d[\w\-]*)', 'linear_return'),
    # VAV boxes
    (r'\bVAV[\s\-]*(\d+)\b', 'vav'),
    # Louver
    (r'\b(LVR)\b', 'louver'),
    # CFM values
    (r'\b(\d+)\s*CFM\b', 'cfm'),
    # Duct sizes
    (r'\b(\d+)"?\s*[xX]\s*(\d+)"?\b', 'duct_size'),
    # Round neck sizes
    (r'\b(\d+)"\b', 'neck_size'),
]

# Map schedule letter tags to product types based on typical conventions
SCHEDULE_TAG_MAP = {
    'A': 'AD-T-BAR SUPPLY',
    'B': 'AD-T-BAR RETURN',
    'C': 'AD-SURF SUPPLY',
    'D': 'AD-SURF RETURN',
}


def detect_tags_ocr(pdf_path, scan_pages=None):
    """
    Detect equipment tags using PyMuPDF text extraction.
    Returns list of dicts with: page, tag, product_type, x, y, text
    """
    doc = fitz.open(pdf_path)
    detections = []

    for page_idx in range(doc.page_count):
        if scan_pages and (page_idx + 1) not in scan_pages:
            continue

        page = doc[page_idx]
        # Get words with positions: (x0, y0, x1, y1, word, block, line, word_no)
        words = page.get_text('words')

        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]

            # Check each tag pattern
            for pattern, tag_type in TAG_PATTERNS:
                matches = re.findall(pattern, text)
                if not matches:
                    continue

                for match in matches:
                    if tag_type == 'schedule_tag':
                        # Single letter tags (A, B, C, D) — map to product type
                        tag = match
                        product_type = SCHEDULE_TAG_MAP.get(tag, f'UNKNOWN-{tag}')
                        detections.append({
                            'page': page_idx + 1,
                            'tag': tag,
                            'tag_type': tag_type,
                            'product_type': product_type,
                            'x': x0, 'y': y0, 'x2': x1, 'y2': y1,
                            'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2,
                            'raw_text': text,
                        })
                    elif tag_type in ('linear_diffuser', 'linear_return', 'louver'):
                        tag = match
                        if tag_type == 'linear_diffuser':
                            product_type = 'AD-LINEAR SLOT DIFFUSER'
                        elif tag_type == 'linear_return':
                            product_type = 'AD-LINEAR SLOT DIFFUSER'
                        else:
                            product_type = 'LOUVERS'
                        detections.append({
                            'page': page_idx + 1,
                            'tag': tag,
                            'tag_type': tag_type,
                            'product_type': product_type,
                            'x': x0, 'y': y0, 'x2': x1, 'y2': y1,
                            'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2,
                            'raw_text': text,
                        })
                    elif tag_type == 'vav':
                        detections.append({
                            'page': page_idx + 1,
                            'tag': f'VAV-{match}',
                            'tag_type': tag_type,
                            'product_type': 'VAV',
                            'x': x0, 'y': y0, 'x2': x1, 'y2': y1,
                            'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2,
                            'raw_text': text,
                        })

    doc.close()
    return detections


def filter_detections(detections, page):
    """
    Filter detections to only those on the main floor plan page,
    excluding schedule/legend areas (which have the same letters but aren't equipment).
    """
    # Only keep detections on the specified page
    page_dets = [d for d in detections if d['page'] == page]

    # Filter out detections that are likely in the title block or schedule area
    # Title block is typically on the right side (x > 80% of page width)
    # and bottom (y > 90% of page height)
    # We'll need to know the page dimensions to filter properly
    return page_dets


# ─── STEP 3: ACCURACY SCORING ────────────────────────────────────────────────

def score_accuracy(detections, ground_truth, match_radius=50):
    """
    Compare detections against ground truth annotations.
    A detection is "matched" if it's within match_radius PDF points of a ground truth annotation
    AND has the same product_type.

    Returns: precision, recall, matched pairs, unmatched detections, missed ground truth
    """
    matched_gt = set()
    matched_det = set()
    matches = []

    for di, det in enumerate(detections):
        best_dist = float('inf')
        best_gi = None

        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if det['page'] != gt['page']:
                continue
            if det['product_type'] != gt['product_type']:
                continue

            # Euclidean distance between centers
            dist = ((det['cx'] - gt['cx'])**2 + (det['cy'] - gt['cy'])**2)**0.5

            if dist < best_dist and dist <= match_radius:
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

    # Unmatched
    false_positives = [detections[i] for i in range(len(detections)) if i not in matched_det]
    missed = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt]

    precision = len(matches) / max(len(detections), 1)
    recall = len(matches) / max(len(ground_truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': len(matches),
        'false_positives': len(false_positives),
        'false_negatives': len(missed),
        'matches': matches,
        'fp_list': false_positives,
        'fn_list': missed,
    }


# ─── STEP 4: RUN ON ALL PROJECTS ─────────────────────────────────────────────

def analyze_project(unlabeled_pdf, labeled_pdf, excel_path, project_name):
    """Run full analysis on a single project."""
    print(f"\n{'='*60}")
    print(f"PROJECT: {project_name}")
    print(f"{'='*60}")

    # Extract ground truth
    gt = extract_ground_truth(labeled_pdf)
    gt_equipment = [g for g in gt if g['product_type'] != 'Highlight']
    print(f"\nGround truth: {len(gt_equipment)} equipment annotations")

    gt_by_type = defaultdict(int)
    for g in gt_equipment:
        gt_by_type[g['product_type']] += 1
    for ptype, count in sorted(gt_by_type.items()):
        print(f"  {ptype:30s} : {count}")

    # Detect tags via OCR on unlabeled PDF
    detections = detect_tags_ocr(unlabeled_pdf)
    print(f"\nOCR detections (all pages): {len(detections)}")

    # Figure out which page the annotations are on (the main floor plan)
    gt_pages = set(g['page'] for g in gt_equipment)
    main_page = max(gt_pages, key=lambda p: sum(1 for g in gt_equipment if g['page'] == p))
    print(f"Main annotated page: {main_page}")

    # Filter detections to main page only
    page_dets = [d for d in detections if d['page'] == main_page]
    print(f"OCR detections on page {main_page}: {len(page_dets)}")

    det_by_type = defaultdict(int)
    for d in page_dets:
        det_by_type[d['product_type']] += 1
    for ptype, count in sorted(det_by_type.items()):
        print(f"  {ptype:30s} : {count}")

    # Filter ground truth to main page too
    gt_main = [g for g in gt_equipment if g['page'] == main_page]

    # Score accuracy
    scores = score_accuracy(page_dets, gt_main, match_radius=60)

    print(f"\n--- ACCURACY (page {main_page}) ---")
    print(f"  Precision: {scores['precision']:.1%} ({scores['true_positives']}/{scores['true_positives']+scores['false_positives']})")
    print(f"  Recall:    {scores['recall']:.1%} ({scores['true_positives']}/{scores['true_positives']+scores['false_negatives']})")
    print(f"  F1 Score:  {scores['f1']:.1%}")
    print(f"  True Positives:  {scores['true_positives']}")
    print(f"  False Positives: {scores['false_positives']}")
    print(f"  False Negatives: {scores['false_negatives']}")

    if scores['fn_list']:
        print(f"\n  MISSED equipment:")
        for m in scores['fn_list']:
            print(f"    {m['product_type']:30s} tag={m['tag']:20s} at ({m['cx']:.0f},{m['cy']:.0f})")

    if scores['fp_list'][:10]:
        print(f"\n  FALSE POSITIVES (first 10):")
        for fp in scores['fp_list'][:10]:
            print(f"    {fp['product_type']:30s} tag={fp['tag']:10s} at ({fp['cx']:.0f},{fp['cy']:.0f}) text=\"{fp['raw_text']}\"")

    # Compare with Excel takeoff
    if excel_path and os.path.exists(excel_path):
        df = pd.read_excel(excel_path, sheet_name='RawData')
        print(f"\n--- EXCEL TAKEOFF ---")
        print(f"  {len(df)} line items")
        total_qty = df['QTY'].sum()
        print(f"  Total quantity: {total_qty}")
        print(f"  vs Ground truth annotations: {len(gt_main)} on page {main_page}")

    return scores


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all project sets
    projects = []
    files = os.listdir(DATA_DIR)
    pdfs = [f for f in files if f.endswith('.pdf') and 'Final' not in f]

    for pdf in sorted(pdfs):
        name = pdf.replace('3.11.26 EAG - 9530 Towne Center Drive ', '').replace('.pdf', '')
        final = pdf.replace('.pdf', ' - Final.pdf')
        # Handle the Flex 220 naming inconsistency
        if not os.path.exists(os.path.join(DATA_DIR, final)):
            final = pdf.replace('.pdf', '- Final.pdf')
        excel = f"Takeoff_EAG - 9530 Towne Center Drive {name}.xlsx"

        unlabeled_path = os.path.join(DATA_DIR, pdf)
        labeled_path = os.path.join(DATA_DIR, final)
        excel_path = os.path.join(DATA_DIR, excel)

        if os.path.exists(labeled_path):
            projects.append((unlabeled_path, labeled_path, excel_path, name))
        else:
            print(f"WARNING: No Final PDF found for {name} (tried {final})")

    print(f"Found {len(projects)} projects with ground truth")

    all_scores = []
    for unlabeled, labeled, excel, name in projects:
        scores = analyze_project(unlabeled, labeled, excel, name)
        scores['project'] = name
        all_scores.append(scores)

    # Summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    total_tp = sum(s['true_positives'] for s in all_scores)
    total_fp = sum(s['false_positives'] for s in all_scores)
    total_fn = sum(s['false_negatives'] for s in all_scores)
    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 0.001)

    print(f"  Projects analyzed: {len(all_scores)}")
    print(f"  Overall Precision: {overall_precision:.1%}")
    print(f"  Overall Recall:    {overall_recall:.1%}")
    print(f"  Overall F1:        {overall_f1:.1%}")
    print(f"  Total TP/FP/FN:    {total_tp}/{total_fp}/{total_fn}")


if __name__ == "__main__":
    main()
