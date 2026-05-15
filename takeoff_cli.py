"""
HVAC Takeoff CLI — Bubble-Only Pipeline (2026-05-13)

Replaces the symbol-YOLO pipeline. New flow:

  schedule_parser  → variables (tag → row properties + inferred class)
        │
        ▼
  find_mechanical_pages  → skip LEGEND / SCHEDULE / DETAILS sheets
        │
        ▼
  hvac_tag_detector_v1.pt  → bubble bboxes per page
        │
        ▼
  EasyOCR (tight bubble crops) → tag text
        │
        ▼
  match against schedule tag list (preferred) OR infer class from prefix
        │
        ▼
  count by (class, tag) → annotated PDF + Excel + detections.json

Drops: symbol YOLO (v9/v10), tag_inference 3-level system, tag_extractor.
Keeps: schedule_parser, tag_matcher bubble+OCR helpers, title-block extractor.

Usage:
    python takeoff_cli.py path/to/blueprint.pdf
    python takeoff_cli.py path/to/blueprint.pdf --conf 0.25 --max-distance 600
    python takeoff_cli.py path/to/blueprint.pdf --schedule-only
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import os
import re
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import fitz
import cv2
import numpy as np

from schedule_parser import parse_pdf_schedules, dump_variables
from tag_matcher import (
    detect_bubbles_on_page,
    ocr_bubble_crops,
    merge_split_bubbles,
    _normalize_for_match,
)


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_BUBBLE_MODEL = 'models/hvac_tag_detector_v1.pt'
DPI = 200
DEFAULT_BUBBLE_CONF = 0.25

COLORS = [
    (0, 200, 0), (200, 0, 0), (0, 165, 255), (0, 0, 200), (255, 200, 0),
    (255, 0, 200), (0, 255, 255), (180, 180, 0), (128, 0, 128), (0, 128, 128),
]


# Tag prefix → equipment class (lifted from tag_inference, with TA/LD/MD already
# present + a few additions noted on 2026-05-11). When OCR reads a bubble that
# doesn't appear in the schedule, this map gives us a class.
TAG_PREFIX_CLASS = {
    # Fans
    'EF': 'EXHAUST FAN', 'SF': 'FAN', 'CF': 'FAN', 'RF': 'FAN',
    'CEF': 'EXHAUST FAN', 'IEF': 'EXHAUST FAN',
    # Major equipment
    'CU': 'CONDENSING UNIT', 'AC': 'CONDENSING UNIT', 'OACU': 'CONDENSING UNIT',
    'AHU': 'AIR HANDLING UNIT', 'DOAS': 'AIR HANDLING UNIT',
    'RTU': 'PACKAGED ROOFTOP UNIT',
    'FCU': 'FAN COIL UNIT', 'FC': 'FAN COIL UNIT',
    'HP': 'HEAT PUMP',
    # Heaters
    'EUH': 'HEATER', 'UH': 'HEATER', 'EH': 'HEATER', 'BH': 'HEATER',
    'CUH': 'HEATER', 'DH': 'HEATER',
    # Terminals
    'VAV': 'VAV', 'VRF': 'VRF', 'ERV': 'CONDENSING UNIT',
    'TA': 'TRANSFER AIR',
    # Dampers
    'MD': 'MOTORIZED DAMPER', 'MVD': 'MANUAL VOLUME DAMPER', 'FD': 'FIRE DAMPER',
    'FSD': 'FIRE SMOKE DAMPER', 'BD': 'BACKDRAFT DAMPER', 'SD': 'SMOKE DAMPER',
    # Louvers
    'L': 'LOUVER', 'LVR': 'LOUVER',
    'EL': 'LOUVER', 'SL': 'LOUVER', 'IL': 'LOUVER',
    # Grilles / registers / diffusers
    'GR': 'AD-GRD', 'RG': 'AD-GRD', 'CD': 'AD-GRD',
    'SA': 'AD-GRD', 'RA': 'AD-GRD', 'EA': 'AD-GRD', 'SB': 'AD-GRD',
    'EG': 'AD-GRD', 'SG': 'AD-GRD',
    'RR': 'AD-GRD', 'SR': 'AD-GRD', 'ER': 'AD-GRD',
    # Linear diffusers
    'LD': 'AD-LINEAR PLENUM',
    # Single-letter air-device prefixes (Sola-style)
    'S': 'AD-T-BAR SUPPLY',
    'R': 'AD-T-BAR RETURN',
    'E': 'AD-T-BAR RETURN',
}


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
    if rot == 270:
        return mb_w - dy, dx
    elif rot == 90:
        return dy, mb_h - dx
    elif rot == 180:
        return mb_w - dx, mb_h - dy
    return dx, dy


# ─── PAGE FILTERING ───────────────────────────────────────────────────────────

NON_PLAN_TITLE_MARKERS = [
    'MECHANICAL LEGEND', 'HVAC LEGEND', 'PLUMBING LEGEND',
    'LEGEND AND ABBREVIATIONS', 'LEGENDS AND SCHEDULES', 'SCHEDULE AND LEGEND',
    'GENERAL NOTES', 'MECHANICAL NOTES', 'HVAC NOTES',
    'MECHANICAL SCHEDULE', 'HVAC SCHEDULE', 'EQUIPMENT SCHEDULE',
    'AIR DEVICE SCHEDULE', 'DIFFUSER SCHEDULE', 'FAN SCHEDULE',
    'MECHANICAL DETAILS', 'HVAC DETAILS', 'TYPICAL DETAILS',
    'PIPING DETAILS', 'INSTALLATION DETAILS',
    'MECHANICAL SPECIFICATIONS', 'HVAC SPECIFICATIONS',
    'TITLE SHEET', 'COVER SHEET', 'SHEET INDEX', 'DRAWING INDEX',
    'SYMBOLS AND ABBREVIATIONS',
]

# Plan sheets that LOOK like takeoff pages but aren't — counting them would
# double-count piping runs or take off equipment that is being removed.
# Demolition plan: existing equipment to be removed (don't count).
# Piping plan: shows refrigerant/water piping runs, not equipment.
# Zoning plan: shows VAV zone boundaries conceptually, not equipment locations.
NON_TAKEOFF_PLAN_MARKERS = [
    'DEMOLITION PLAN', 'DEMO PLAN',
    'PIPING PLAN',
    'ZONING PLAN',
]
PLAN_KEYWORDS = [
    'MECHANICAL PLAN', 'CEILING PLAN', 'HVAC PLAN',
    'VENTILATION PLAN', 'FLOOR PLAN', 'ROOF PLAN',
    'OVERALL PLAN', 'ENLARGED PLAN', 'PARTIAL PLAN',
]

# Pages with text length below this AND no non-plan markers are treated as
# CAD/raster plan pages (Flex projects: real plan pages have only 900-1700
# chars of room labels + bubble tags; legend/notes/schedule pages have 4500+).
SPARSE_TEXT_THRESHOLD = 2500


def _is_non_plan_sheet(text_upper):
    return any(m in text_upper for m in NON_PLAN_TITLE_MARKERS)


def find_mechanical_pages(pdf_path):
    doc = fitz.open(pdf_path)
    total = doc.page_count
    candidates = []
    skipped_non_plan = []
    skipped_non_takeoff = []
    skipped_no_signal = []
    for pi in range(total):
        text = doc[pi].get_text().upper()
        if _is_non_plan_sheet(text):
            skipped_non_plan.append(pi + 1)
            continue
        if any(m in text for m in NON_TAKEOFF_PLAN_MARKERS):
            skipped_non_takeoff.append(pi + 1)
            continue
        if any(kw in text for kw in PLAN_KEYWORDS):
            candidates.append(pi)
            continue
        # CAD/raster plan pages often have sparse text layers (room labels +
        # bubble tags only) with no explicit "...PLAN" string in the title
        # block. Include them if they survived the non-plan gate.
        if len(text) < SPARSE_TEXT_THRESHOLD:
            candidates.append(pi)
            continue
        skipped_no_signal.append(pi + 1)
    doc.close()
    if skipped_non_plan:
        print(f"  Skipping non-plan sheets: pages {skipped_non_plan}")
    if skipped_non_takeoff:
        print(f"  Skipping non-takeoff plans (demo/piping/zoning): pages {skipped_non_takeoff}")
    if skipped_no_signal:
        print(f"  Skipping text-heavy pages with no plan signal: {skipped_no_signal}")
    if not candidates:
        return list(range(total))
    return candidates


# ─── TAG NORMALIZATION & CLASS INFERENCE ─────────────────────────────────────

_TAG_SPLIT_RE = re.compile(r'^([A-Z]+)[-\s]?(\d{1,4}[A-Z]?)$')


def _split_prefix(tag):
    """('CU-1') → ('CU', '1'); ('A1') → ('A', '1'); else (tag, '')."""
    s = _normalize_for_match(tag)  # uppercase, strip punctuation
    if not s:
        return ('', '')
    # Find prefix run of letters then digits
    m = re.match(r'^([A-Z]+)(\d+[A-Z]?)$', s)
    if m:
        return (m.group(1), m.group(2))
    # All letters or all digits — no usable split
    return (s, '')


def _class_from_prefix(prefix):
    """Look up TAG_PREFIX_CLASS by longest matching prefix."""
    if not prefix:
        return None
    # Try full prefix, then back off one char at a time
    for n in range(len(prefix), 0, -1):
        sub = prefix[:n]
        if sub in TAG_PREFIX_CLASS:
            return TAG_PREFIX_CLASS[sub]
    return None


def resolve_bubble_text(bubble_text, schedule_tag_lookup, schedule_class_lookup):
    """Turn a raw OCR'd bubble string into (canonical_tag, class).

    Strict: only accept bubble text that matches a tag present in the parsed
    schedule. Everything else is dropped. Prevents OCR garbage like room
    labels ("STORAGE", "ELEV") and bare prefixes ("FC", "SD") from being
    counted as tags. Tradeoff: if the schedule parser misses a project's
    schedule entirely, we report zero counts for that project — by design,
    so we don't fabricate numbers.
    """
    if not bubble_text:
        return (None, None)
    n = _normalize_for_match(bubble_text)
    if not n or n not in schedule_tag_lookup:
        return (None, None)
    canonical = schedule_tag_lookup[n]
    cls = schedule_class_lookup.get(n) or _class_from_prefix(_split_prefix(canonical)[0])
    return (canonical, cls or 'UNKNOWN')


def build_schedule_lookups(variables):
    """Return (tag_lookup, class_lookup) keyed by normalized tag string."""
    tag_lookup = {}
    class_lookup = {}
    for v in variables:
        t = v.get('tag', '')
        if not t:
            continue
        n = _normalize_for_match(t)
        if not n:
            continue
        if n not in tag_lookup:
            tag_lookup[n] = t
            cls = v.get('inferred_yolo_class')
            if cls:
                class_lookup[n] = cls
    return tag_lookup, class_lookup


# ─── ANNOTATE PDF ─────────────────────────────────────────────────────────────

def annotate_pdf(input_pdf, output_pdf, bubbles_per_page):
    doc = fitz.open(input_pdf)
    for page_idx, bubbles in bubbles_per_page.items():
        page = doc[page_idx]
        rot = page.rotation
        mb_w, mb_h = page.mediabox.width, page.mediabox.height
        scale = DPI / 72
        for b in bubbles:
            if not b.get('tag'):
                continue
            disp_x1, disp_y1 = b['x1'] / scale, b['y1'] / scale
            disp_x2, disp_y2 = b['x2'] / scale, b['y2'] / scale
            ax1, ay1 = display_to_annot(disp_x1, disp_y1, rot, mb_w, mb_h)
            ax2, ay2 = display_to_annot(disp_x2, disp_y2, rot, mb_w, mb_h)
            rect = fitz.Rect(min(ax1, ax2), min(ay1, ay2),
                             max(ax1, ax2), max(ay1, ay2))
            bgr = COLORS[hash(b.get('cls', '')) % len(COLORS)]
            rgb = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=rgb)
            annot.set_border(width=2)
            annot.set_info(content=f"{b['tag']} ({b.get('cls', '?')})")
            annot.update()
    doc.save(output_pdf)
    doc.close()


# ─── EXCEL OUTPUT ─────────────────────────────────────────────────────────────

def _prop(details, keywords):
    if not details:
        return ''
    kw_upper = [k.upper() for k in keywords]
    for k, v in details.items():
        k_norm = ' '.join(str(k).upper().split())
        for kw in kw_upper:
            if kw in k_norm:
                return str(v)
    return ''


def write_excel(output_path, bubbles_per_page, project_name, mark_details):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print("openpyxl not installed, skipping Excel output.")
        return False

    if mark_details is None:
        mark_details = {}

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Triune Takeoff'
    ws['A1'] = f'HVAC Takeoff: {project_name}'
    ws['A1'].font = Font(size=14, bold=True)
    ws.merge_cells('A1:E1')

    HEADERS = ['PRODUCT', 'BRAND', 'MODEL', 'QTY', 'TAG', 'NECK SIZE',
               'MODULE SIZE', 'DUCT SIZE', 'TYPE', 'MOUNTING', 'REMARK']
    for ci, h in enumerate(HEADERS, 1):
        cell = ws.cell(row=3, column=ci, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='DDDDDD')

    grouped = defaultdict(lambda: {'count': 0, 'pages': set()})
    for page_idx, bubbles in bubbles_per_page.items():
        for b in bubbles:
            tag = b.get('tag')
            cls = b.get('cls', 'UNKNOWN')
            if not tag:
                continue
            grouped[(cls, tag)]['count'] += 1
            grouped[(cls, tag)]['pages'].add(page_idx + 1)

    row = 4
    total = 0
    current_cls = None
    for (cls, tag), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        details = mark_details.get(tag, {})
        brand_model = _prop(details, ['MANUFACTURER & MODEL', 'MAKE / MODEL', 'MAKE/MODEL'])
        if brand_model and ' / ' in brand_model:
            brand, model = brand_model.split(' / ', 1)
        elif brand_model and ' ' in brand_model:
            brand, model = brand_model.split(' ', 1)
        elif brand_model:
            brand, model = brand_model, ''
        else:
            brand = _prop(details, ['MANUFACTURER', 'BRAND', 'MAKE'])
            model = _prop(details, ['MODEL NUMBER', 'MODEL'])
        neck = _prop(details, ['NECK', 'SIZE (NECK)', 'SIZE'])
        etype = _prop(details, ['SERVICE', 'TYPE', 'DESCRIPTION'])
        mounting = _prop(details, ['MOUNTING', 'MOUNT'])

        ws.cell(row=row, column=1, value=cls if cls != current_cls else '')
        current_cls = cls
        ws.cell(row=row, column=2, value=brand)
        ws.cell(row=row, column=3, value=model)
        ws.cell(row=row, column=4, value=data['count'])
        ws.cell(row=row, column=5, value=tag)
        ws.cell(row=row, column=6, value=neck)
        ws.cell(row=row, column=9, value=etype)
        ws.cell(row=row, column=10, value=mounting)
        ws.cell(row=row, column=11, value=f"Pages: {', '.join(str(p) for p in sorted(data['pages']))}")
        total += data['count']
        row += 1

    for cls in sorted(set(c for c, t in grouped.keys())):
        cls_total = sum(d['count'] for (c, t), d in grouped.items() if c == cls)
        ws.cell(row=row, column=1, value=f'{cls} Total').font = Font(bold=True)
        ws.cell(row=row, column=4, value=cls_total).font = Font(bold=True)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value='GRAND TOTAL').font = Font(bold=True, size=12)
    ws.cell(row=row, column=4, value=total).font = Font(bold=True, size=12)

    widths = [30, 15, 15, 8, 18, 12, 12, 12, 25, 12, 25]
    for ci, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + ci)].width = w

    wb.save(output_path)
    return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='HVAC Takeoff CLI — bubble-only')
    parser.add_argument('pdf', help='Path to blueprint PDF')
    parser.add_argument('--bubble-model', default=DEFAULT_BUBBLE_MODEL,
                        help='Path to tag-bubble YOLO model')
    parser.add_argument('--conf', type=float, default=DEFAULT_BUBBLE_CONF,
                        help='Bubble-detector confidence threshold')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--all-pages', action='store_true')
    parser.add_argument('--pages', type=int, nargs='+')
    parser.add_argument('--schedule-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument(
        '--scanner', choices=['bubble', 'text', 'auto'], default='auto',
        help=("Detection backend. 'bubble' = YOLO tag-bubble detector + OCR "
              "(legacy). 'text' = text-layer-first scanner + OCR fallback "
              "(positioned tag finder, no YOLO). 'auto' (default) = text "
              "scanner, fall back to bubble detector if scanner yields 0."),
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    if not args.schedule_only and not Path(args.bubble_model).exists():
        print(f"ERROR: bubble model not found: {args.bubble_model}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent / f"{pdf_path.stem}_takeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"HVAC TAKEOFF (bubble-only) — {pdf_path.name}")
    print(f"{'='*70}")
    print(f"Bubble model: {args.bubble_model}")
    print(f"Conf:         {args.conf}")
    print(f"Output:       {out_dir}\n")

    # Parse schedule (best-effort — we can still count without it)
    print("Parsing schedule...")
    variables = []
    schedules, marks, mark_details = [], [], {}
    try:
        schedules, marks, mark_details, _legend, _summary, variables = parse_pdf_schedules(str(pdf_path))
        print(f"  {len(schedules)} schedule(s), {len(marks)} tag(s), {len(variables)} variable(s)")
        if marks:
            print(f"  Sample tags: {marks[:10]}{'...' if len(marks) > 10 else ''}")
    except Exception as e:
        print(f"  Schedule parse failed: {e}  (continuing without schedule)")

    variables_path = out_dir / f"{pdf_path.stem}_variables.json"
    try:
        with open(variables_path, 'w', encoding='utf-8') as f:
            json.dump(variables, f, indent=2, default=str, ensure_ascii=False)
        print(f"  Wrote {len(variables)} variables → {variables_path.name}")
    except Exception as e:
        print(f"  (JSON sidecar failed: {e})")

    if args.verify or args.schedule_only:
        dump_variables(variables)

    if args.schedule_only:
        print(f"\n--schedule-only: skipping detection.")
        return

    tag_lookup, class_lookup = build_schedule_lookups(variables)
    print(f"  Schedule lookup: {len(tag_lookup)} tag(s) indexed\n")

    # Pages to process
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()
    if args.pages:
        pages = [p - 1 for p in args.pages]
    elif args.all_pages:
        pages = list(range(total_pages))
    else:
        pages = find_mechanical_pages(pdf_path)
        if not pages:
            pages = list(range(total_pages))
    print(f"Processing {len(pages)} page(s) of {total_pages} total\n")

    use_scanner = args.scanner in ('text', 'auto')
    use_bubble = args.scanner in ('bubble', 'auto')

    if use_bubble:
        print("Loading bubble detector...")
        from tag_matcher import get_bubble_model
        if get_bubble_model(args.bubble_model) is None:
            print(f"ERROR: failed to load bubble model from {args.bubble_model}")
            sys.exit(1)
        print("  Loaded.\n")

    valid_tags = sorted(set(tag_lookup.values()))
    bubbles_per_page = {}
    t_start = time.time()
    for pi in pages:
        t0 = time.time()

        final = []
        scanner_used = False
        if use_scanner and valid_tags:
            print(f"  Page {pi+1}: text-scan...", end=' ', flush=True)
            try:
                from sliding_ocr import scan_page_for_tags
                hits = scan_page_for_tags(str(pdf_path), pi, valid_tags,
                                          dpi=200, ocr_fallback=False)
            except Exception as e:
                print(f"scanner failed: {e}")
                hits = []
            for h in hits:
                norm = _normalize_for_match(h['tag'])
                cls = class_lookup.get(norm) or _class_from_prefix(_split_prefix(h['tag'])[0])
                final.append({
                    'x1': h['x1'], 'y1': h['y1'],
                    'x2': h['x2'], 'y2': h['y2'],
                    'cx': h['cx'], 'cy': h['cy'],
                    'conf': h['conf'],
                    'text': h.get('raw_text', h['tag']),
                    'ocr_conf': h['conf'],
                    'tag': h['tag'],
                    'cls': cls,
                    'source': h.get('source', 'text_scan'),
                })
            scanner_used = True
            print(f"{len(final)} hit(s)", end=' ', flush=True)

        # Fall back to bubble detector if scanner found nothing (auto mode)
        if not final and use_bubble:
            print("→ bubble detect...", end=' ', flush=True)
            try:
                img, _rot, _mw, _mh = render_page(pdf_path, pi)
            except Exception as e:
                print(f"render FAILED: {e}")
                continue
            bubbles = detect_bubbles_on_page(img, conf=args.conf)
            if not bubbles:
                print("0 bubbles")
                continue
            print(f"{len(bubbles)} bubbles, OCR...", end=' ', flush=True)
            bubbles = ocr_bubble_crops(img, bubbles)
            if not bubbles:
                print("none readable")
                continue
            bubbles_with_merges = merge_split_bubbles(bubbles)
            resolved = []
            for b in bubbles_with_merges:
                tag, cls = resolve_bubble_text(b.get('text', ''), tag_lookup, class_lookup)
                if not tag:
                    continue
                b2 = dict(b)
                b2['tag'] = tag
                b2['cls'] = cls
                resolved.append(b2)
            for b in resolved:
                if b.get('merged_from'):
                    if any(abs(b['cx'] - f['cx']) < 30 and abs(b['cy'] - f['cy']) < 30
                           and not f.get('merged_from') for f in final):
                        continue
                final.append(b)

        bubbles_per_page[pi] = final
        elapsed = time.time() - t0
        suffix = ""
        if scanner_used and not final and use_bubble:
            suffix = " (after bubble fallback)"
        print(f"→ {len(final)} resolved{suffix} ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_start
    print(f"\nDetection complete in {total_elapsed:.0f}s")

    total_count = sum(len(bs) for bs in bubbles_per_page.values())
    if total_count == 0:
        print("\nNo tag bubbles resolved on the selected pages.")
        sys.exit(0)

    # Aggregate
    by_cls_tag = defaultdict(int)
    for bubbles in bubbles_per_page.values():
        for b in bubbles:
            by_cls_tag[(b['cls'], b['tag'])] += 1

    print(f"\n{'='*75}")
    print(f"TAKEOFF SUMMARY")
    print(f"{'='*75}")
    print(f"  Total tag bubbles resolved: {total_count}")
    print(f"  Pages with bubbles:         {len(bubbles_per_page)}")
    print()
    print(f"  {'Equipment Type':<30} {'Tag':<15} {'Count':>8}")
    print(f"  {'-'*30} {'-'*15} {'-'*8}")
    for (cls, tag), cnt in sorted(by_cls_tag.items(), key=lambda kv: (str(kv[0][0] or ''), str(kv[0][1] or ''))):
        print(f"  {(cls or 'UNKNOWN')[:29]:<30} {(tag or '?'):<15} {cnt:>8}")
    print()

    annotated_pdf_path = out_dir / f"{pdf_path.stem}_annotated.pdf"
    excel_path = out_dir / f"{pdf_path.stem}_takeoff.xlsx"
    detections_json_path = out_dir / f"{pdf_path.stem}_detections.json"

    print("Writing outputs...")
    print(f"  Annotated PDF:  {annotated_pdf_path}")
    annotate_pdf(str(pdf_path), str(annotated_pdf_path), bubbles_per_page)

    print(f"  Excel takeoff:  {excel_path}")
    write_excel(str(excel_path), bubbles_per_page, pdf_path.stem, mark_details)

    print(f"  Detections:     {detections_json_path}")
    det_dump = {
        'pdf': str(pdf_path),
        'pipeline': 'bubble-only',
        'dpi': DPI,
        'pages': {
            str(pi): [
                {
                    'cls': b['cls'],
                    'tag': b['tag'],
                    'ocr_text': b.get('text'),
                    'ocr_conf': b.get('ocr_conf'),
                    'bubble_conf': b.get('conf'),
                    'x1': b['x1'], 'y1': b['y1'], 'x2': b['x2'], 'y2': b['y2'],
                    'merged_from': b.get('merged_from'),
                }
                for b in bs
            ]
            for pi, bs in bubbles_per_page.items()
        },
    }
    with open(detections_json_path, 'w', encoding='utf-8') as f:
        json.dump(det_dump, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"DONE — open {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
