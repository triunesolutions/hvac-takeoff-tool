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
import re
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import fitz
import cv2
import numpy as np

from tag_extractor import summarize_detections_by_tag
from tag_inference import infer_tags
from schedule_parser import parse_pdf_schedules, dump_variables


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL = 'models/hvac_yolov8s_v10.pt'
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


# ─── PROJECT INFO (TITLE BLOCK) ───────────────────────────────────────────────

# Inline label-value patterns: value is on same line after LABEL: value
INLINE_PATTERNS = {
    'scale':       [r'\bSCALE\s*[:=]\s*([0-9/"\'\-=\s\.]{1,30}(?:\'|"|FT)[0-9\'\-"=\s\.]{0,20})',
                    r'\bSCALE\s*[:=]\s*(AS\s+NOTED|NTS|N\.T\.S\.|NONE)'],
    'date':        [r'\bDATE\s*[:=]\s*([0-9]{1,2}[\-/\.][0-9]{1,2}[\-/\.][0-9]{2,4})',
                    r'\b(?:ISSUE[D]?|PLOT)\s*DATE\s*[:=]\s*([0-9]{1,2}[\-/\.][0-9]{1,2}[\-/\.][0-9]{2,4})'],
    'sheet':       [r'\bSHEET\s*(?:NO\.?|NUMBER)\s*[:=]\s*([A-Z]{1,3}[\-\.]?[0-9]{1,4}(?:\.[0-9]+)?)',
                    r'\bDRAWING\s*(?:NO\.?|NUMBER)\s*[:=]\s*([A-Z]{1,3}[\-\.]?[0-9]{1,4})'],
    'project':     [r'\bPROJECT\s*(?:NAME)?\s*[:=]\s*([^\n\r]{3,80})',
                    r'\bJOB\s*(?:NAME|NO\.?)\s*[:=]\s*([^\n\r]{3,80})'],
    'engineer':    [r'\bENGINEER\s*(?:OF\s*RECORD)?\s*[:=]\s*([^\n\r]{3,60})',
                    r'\bDESIGNED\s*BY\s*[:=]\s*([^\n\r]{2,40})',
                    r'\bDRAWN\s*BY\s*[:=]\s*([^\n\r]{2,40})'],
    'firm':        [r'\b(?:MECHANICAL\s+ENGINEER|MEP\s+ENGINEER|CONSULTANT)\s*[:=]\s*([^\n\r]{3,80})'],
    'revision':    [r'\bREV(?:ISION)?\s*(?:NO\.?)?\s*[:=]\s*([0-9A-Z]{1,4})\b'],
}

# Bad-phrase prefixes that should never be treated as values
_BAD_VALUE_STARTS = re.compile(
    r'^(?:ISSUE(?:D)?\b|NOT\b|FOR\b|TO\s+BE\b|APPROVED\b|REVIEWED\b|SEE\b|DESCRIPTION\b|NO\.\b|NOTES?\b|BY\b)',
    re.IGNORECASE,
)

# Field-specific value validators (return True if val passes for that key)
_FIELD_VALIDATORS = {
    'scale':    lambda v: bool(re.search(r'[0-9/]["\']|NTS|N\.T\.S|AS\s+NOTED|NONE', v, re.I)),
    'date':     lambda v: bool(re.search(r'[0-9]{1,2}[\-/\.][0-9]{1,2}[\-/\.][0-9]{2,4}|[A-Z][a-z]+\s+\d{1,2}[,\s]+\d{4}', v)),
    'sheet':    lambda v: bool(re.match(r'^[A-Z]{1,3}[\-\.]?[0-9]{1,4}(?:\.[0-9]+)?$', v.strip())),
    'revision': lambda v: bool(re.match(r'^[0-9]{1,3}$|^[A-Z]$', v.strip())),
    'project':  lambda v: (len(v) >= 5 and not re.match(r'^(?:Project|Job|Sheet|Drawing|Date|Scale|Revision)\s*(?:Name|Number|Title|No\.?)?$', v, re.I)),
    'engineer': lambda v: (2 <= len(v) <= 40 and not re.search(r'\d{3,}', v)),
    'firm':     lambda v: len(v) >= 4,
}


def _valid_field(key, val):
    v = _FIELD_VALIDATORS.get(key)
    return v(val) if v else True

# Labels that commonly sit in a column with the value on the NEXT visible line
# (typical CAD title blocks: "Project Name" line, then the actual name below)
STACKED_LABELS = {
    'project':  [r'^PROJECT\s*(?:NAME|TITLE)?\s*:?$', r'^JOB\s*(?:NAME|TITLE)?\s*:?$'],
    'date':     [r'^DATE\s*:?$', r'^ISSUE(?:D)?\s*DATE\s*:?$'],
    'scale':    [r'^SCALE\s*:?$'],
    'sheet':    [r'^SHEET\s*(?:NO\.?|NUMBER)?\s*:?$', r'^DRAWING\s*(?:NO\.?|NUMBER)?\s*:?$'],
    'engineer': [r'^(?:DRAWN|DESIGNED|CHECKED)\s*BY\s*:?$'],
}


def _clean(val):
    val = val.strip().strip(':-=').strip()
    val = re.sub(r'\s+', ' ', val)
    return val


def _is_bad_value(val):
    if not val or len(val) < 2:
        return True
    if _BAD_VALUE_STARTS.match(val):
        return True
    return False


def extract_project_info(pdf_path, max_pages=3):
    """Best-effort extraction of title-block info from a PDF."""
    info = {}
    try:
        doc = fitz.open(str(pdf_path))
        chunks = []
        for i in range(min(max_pages, len(doc))):
            try:
                chunks.append(doc[i].get_text("text"))
            except Exception:
                pass
        doc.close()
    except Exception as e:
        return {'_error': str(e)}

    full_text = "\n".join(chunks)
    if not full_text.strip():
        return info

    # Pass 1 — inline LABEL: VALUE patterns
    for key, patterns in INLINE_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                val = _clean(m.group(1))
                if not _is_bad_value(val) and _valid_field(key, val):
                    info[key] = val
                    break

    # Pass 2 — stacked labels: value is on the next non-empty line
    lines = [ln.strip() for ln in full_text.split('\n')]
    for i, line in enumerate(lines):
        if not line:
            continue
        for key, patterns in STACKED_LABELS.items():
            if key in info:
                continue
            for pat in patterns:
                if re.match(pat, line, re.IGNORECASE):
                    # Next non-empty line within 3 lines = candidate value
                    for j in range(i + 1, min(i + 4, len(lines))):
                        cand = lines[j]
                        if not cand or len(cand) >= 80:
                            continue
                        if _is_bad_value(cand):
                            continue
                        # Skip if it's another label (ends with colon, or is a known label word)
                        if re.match(r'^(?:Project|Job|Sheet|Drawing|Date|Scale|Revision|Description|Drawn|Designed|Checked|By|No\.?|Number|Name|Title|Tel|Phone|Fax|Email)\s*(?:Name|Number|Title|By|No\.?|:)?\s*:?$', cand, re.I):
                            continue
                        if re.match(r'^[A-Z][A-Z\s]{2,30}:?$', cand):
                            continue
                        clean_cand = _clean(cand)
                        if not _valid_field(key, clean_cand):
                            continue
                        info[key] = clean_cand
                        break
                    break

    # Pass 3 — firm name heuristic (ALL CAPS + engineering/consulting suffix)
    if 'firm' not in info:
        firm_re = re.compile(
            r'(?m)^([A-Z][A-Z&\.\-\' ,]{2,60}\s+(?:ENGINEERING|ENGINEERS|CONSULTANTS?|ASSOCIATES|DESIGN(?:S)?|GROUP|ARCHITECTS?)(?:\s*,?\s*(?:INC|LLC|LLP|P\.?C\.?|PLLC)\.?)?)\s*$'
        )
        for m in firm_re.finditer(full_text):
            cand = _clean(m.group(1))
            if not _is_bad_value(cand):
                info['firm'] = cand
                break

    # Pass 4 — address (street line). Label-aware to avoid grabbing random "9530 Towne..." twice.
    if 'address' not in info:
        addr_re = re.compile(
            r'\b([0-9]{2,5}\s+[A-Z][A-Za-z0-9\.\'\-]+(?:\s+[A-Z][A-Za-z0-9\.\'\-]+){1,5}\s+(?:ST|STREET|AVE|AVENUE|RD|ROAD|BLVD|DR|DRIVE|WAY|LN|LANE|CT|COURT|PL|PLACE|CIR|CIRCLE|CTR|CENTER|CENTRE)\.?)\b'
        )
        m = addr_re.search(full_text)
        if m:
            info['address'] = _clean(m.group(1))

    # Pass 5 — spatial label/value pairing for CAD title blocks (Gensler etc.)
    try:
        spatial = extract_project_info_spatial(pdf_path, max_pages=max_pages)
        for k, v in spatial.items():
            if k not in info or len(info.get(k, '')) < 2:
                info[k] = v
    except Exception:
        pass

    return info


# Labels used by the spatial extractor. Each entry: (key, list of label-text variants).
# Match is case-insensitive, exact-text after stripping trailing colon.
SPATIAL_LABELS = [
    ('project',     ['Project Name', 'Job Name', 'Job Title', 'Project']),
    ('project_no',  ['Project Number', 'Project No.', 'Project No', 'Job Number', 'Job No.', 'Job No']),
    ('description', ['Description', 'Sheet Title', 'Drawing Title']),
    ('sheet',       ['Sheet Number', 'Sheet No.', 'Sheet No', 'Drawing Number', 'Drawing No.']),
    ('scale',       ['Scale']),
    ('date',        ['Date', 'Issue Date', 'Plot Date']),
    ('engineer',    ['Drawn By', 'Designed By', 'Checked By', 'Engineer']),
    ('firm',        ['Architect', 'Mechanical Engineer', 'MEP Engineer', 'Consultant']),
]

_LABEL_TEXTS_LC = {v.lower().rstrip(':') for _, vs in SPATIAL_LABELS for v in vs}


def _spans_in_display_space(page):
    """Return list of {bbox, cx, cy, w, h, text} spans in display coords."""
    rot = page.rotation
    mb_w, mb_h = float(page.mediabox.width), float(page.mediabox.height)
    out = []
    for block in page.get_text('dict').get('blocks', []):
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            for sp in line.get('spans', []):
                txt = sp.get('text', '').strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = sp['bbox']
                if rot == 270:
                    dx0, dy0, dx1, dy1 = mb_h - y1, x0, mb_h - y0, x1
                elif rot == 90:
                    dx0, dy0, dx1, dy1 = y0, mb_w - x1, y1, mb_w - x0
                elif rot == 180:
                    dx0, dy0, dx1, dy1 = mb_w - x1, mb_h - y1, mb_w - x0, mb_h - y0
                else:
                    dx0, dy0, dx1, dy1 = x0, y0, x1, y1
                out.append({
                    'bbox': (dx0, dy0, dx1, dy1),
                    'cx': (dx0 + dx1) / 2, 'cy': (dy0 + dy1) / 2,
                    'w': dx1 - dx0, 'h': dy1 - dy0,
                    'text': txt,
                })
    return out


def _find_value_for_label(label_span, spans):
    """Look for the value span paired with this label.
    CAD title blocks usually place the value DIRECTLY ABOVE the label
    (small label text under a larger value). Fall back to right-of-label.
    """
    lcx, lcy = label_span['cx'], label_span['cy']
    lh = max(label_span['h'], 6.0)
    lw = max(label_span['w'], 30.0)

    candidates = []
    for v in spans:
        if v is label_span:
            continue
        vt = v['text'].strip()
        if not vt or len(vt) < 1:
            continue
        if vt.lower().rstrip(':') in _LABEL_TEXTS_LC:
            continue
        if re.match(r'^[\W_]+$', vt):
            continue
        dx = v['cx'] - lcx
        dy = v['cy'] - lcy   # negative => above in display
        # ABOVE: CAD title blocks often have values that extend wider than the label.
        # Allow generous horizontal tolerance; weight pick by value height (taller = sheet title font).
        if abs(dx) <= max(lw * 3.0, 120) and -lh * 6 < dy < -lh * 0.2:
            score = abs(dy) - v['h'] * 2.0   # prefer closest-above + tallest value
            candidates.append((score, 0, v))
        # RIGHT-OF: value to the right on same baseline (LABEL: VALUE inline rendered as separate span)
        elif abs(dy) <= lh * 0.6 and 0 < dx < lw * 4:
            candidates.append((dx, 1, v))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], t[0]))
    return candidates[0][2]['text']


def _find_value_with_height(label_span, spans):
    """Same as _find_value_for_label but returns (text, height) so the caller
    can rank competing label instances by value typography size."""
    lcx, lcy = label_span['cx'], label_span['cy']
    lh = max(label_span['h'], 6.0)
    lw = max(label_span['w'], 30.0)
    candidates = []
    for v in spans:
        if v is label_span:
            continue
        vt = v['text'].strip()
        if not vt:
            continue
        if vt.lower().rstrip(':') in _LABEL_TEXTS_LC:
            continue
        if re.match(r'^[\W_]+$', vt):
            continue
        dx = v['cx'] - lcx
        dy = v['cy'] - lcy
        if abs(dx) <= max(lw * 3.0, 120) and -lh * 6 < dy < -lh * 0.2:
            score = abs(dy) - v['h'] * 2.0
            candidates.append((score, 0, v))
        elif abs(dy) <= lh * 0.6 and 0 < dx < lw * 4:
            candidates.append((dx, 1, v))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], t[0]))
    best = candidates[0][2]
    return best['text'], best['h']


def extract_project_info_spatial(pdf_path, max_pages=3):
    """Bbox-aware title-block extraction. Pairs each known label with the
    nearest value above it (CAD style) or to its right (form style)."""
    info = {}
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return info

    # Title-block fields live only on the cover sheet. After page 1, we'd be
    # picking up per-drawing detail callouts (e.g. SCALE: NONE under each detail
    # box) which is not the project scale.
    TITLEBLOCK_KEYS = {'project', 'project_no', 'description', 'sheet',
                        'scale', 'date', 'engineer', 'firm'}

    for page_idx in range(min(max_pages, len(doc))):
        try:
            page = doc[page_idx]
            spans = _spans_in_display_space(page)
        except Exception:
            continue
        if not spans:
            continue

        # Index labels by normalized exact text
        label_lc_to_key = {}
        for key, variants in SPATIAL_LABELS:
            for v in variants:
                label_lc_to_key.setdefault(v.lower(), key)

        # Collect ALL (label, value, value_height) candidates per key, pick best
        per_key = {}
        for sp in spans:
            t_norm = sp['text'].strip().rstrip(':').lower()
            key = label_lc_to_key.get(t_norm)
            if not key or key in info:
                continue
            # Only accept title-block fields on the cover sheet (page 0).
            # Otherwise we pick up "SCALE: NONE" stamps printed under each
            # detail drawing on later pages.
            if page_idx > 0 and key in TITLEBLOCK_KEYS:
                continue
            pair = _find_value_with_height(sp, spans)
            if not pair:
                continue
            val, vh = pair
            val = _clean(val)
            if _is_bad_value(val):
                continue
            if key in _FIELD_VALIDATORS and not _valid_field(key, val):
                continue
            # Keep the candidate with the tallest value text (real sheet titles
            # are big; column-header "Description" values are small).
            cur = per_key.get(key)
            if cur is None or vh > cur[1]:
                per_key[key] = (val, vh)

        for key, (val, _) in per_key.items():
            if key not in info:
                info[key] = val

    doc.close()
    return info


def print_project_info(info):
    """Pretty-print project info block."""
    if not info or (len(info) == 1 and '_error' in info):
        print("Project info: (none detected in title block)")
        return
    label_map = [
        ('project',     'Project'),
        ('project_no',  'Project No.'),
        ('description', 'Sheet Title'),
        ('firm',        'Firm'),
        ('engineer',    'Engineer'),
        ('address',     'Address'),
        ('sheet',       'Sheet'),
        ('scale',       'Scale'),
        ('date',        'Date'),
        ('revision',    'Revision'),
    ]
    print("Project info (best-effort from title block):")
    for key, label in label_map:
        if key in info:
            print(f"  {label:<10} {info[key]}")


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

def _prop(details, keywords):
    """Tolerant property lookup: any key containing any keyword (case-insensitive)."""
    if not details:
        return ''
    kw_upper = [k.upper() for k in keywords]
    for k, v in details.items():
        k_norm = ' '.join(str(k).upper().split())
        for kw in kw_upper:
            if kw in k_norm:
                return str(v)
    return ''


def write_excel(output_path, detections_per_page, project_name, schedule_details=None):
    """Write Excel takeoff matching team's format."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print("openpyxl not installed, skipping Excel output. Install with: pip install openpyxl")
        return False

    if schedule_details is None:
        schedule_details = {}

    wb = openpyxl.Workbook()

    # Sheet 1: Triune Takeoff (matches team's format)
    ws = wb.active
    ws.title = 'Triune Takeoff'

    # Header
    ws['A1'] = f'HVAC Takeoff: {project_name}'
    ws['A1'].font = Font(size=14, bold=True)
    ws.merge_cells('A1:E1')

    # Team's exact headers
    HEADERS = ['PRODUCT', 'BRAND', 'MODEL', 'QTY', 'TAG', 'NECK SIZE',
               'MODULE SIZE', 'DUCT SIZE', 'TYPE', 'MOUNTING', 'REMARK']
    for ci, h in enumerate(HEADERS, 1):
        cell = ws.cell(row=3, column=ci, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='DDDDDD')

    # Group detections by (class, tag) and fill in schedule details
    grouped = defaultdict(lambda: {'count': 0, 'pages': set()})
    for page_idx, dets in detections_per_page.items():
        for d in dets:
            cls = d['cls']
            tag = d.get('tag') or ''
            key = (cls, tag)
            grouped[key]['count'] += 1
            grouped[key]['pages'].add(page_idx + 1)

    row = 4
    total = 0
    current_cls = None
    for (cls, tag), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        # Lookup schedule details for this tag (tolerant of header variations)
        details = schedule_details.get(tag, {})
        # Prefer a combined column like "MANUFACTURER & MODEL" or "MAKE / MODEL".
        # Split on " / " if present (MAKE/MODEL convention), else first space.
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
        neck_size = _prop(details, ['NECK', 'SIZE (NECK)', 'SIZE'])
        etype = _prop(details, ['SERVICE', 'TYPE', 'DESCRIPTION'])
        mounting = _prop(details, ['MOUNTING', 'MOUNT'])

        ws.cell(row=row, column=1, value=cls if cls != current_cls else '')
        current_cls = cls
        ws.cell(row=row, column=2, value=brand)
        ws.cell(row=row, column=3, value=model)
        ws.cell(row=row, column=4, value=data['count'])
        ws.cell(row=row, column=5, value=tag)
        ws.cell(row=row, column=6, value=neck_size)
        ws.cell(row=row, column=9, value=etype)
        ws.cell(row=row, column=10, value=mounting)
        ws.cell(row=row, column=11, value=f"Pages: {', '.join(str(p) for p in sorted(data['pages']))}")
        total += data['count']
        row += 1

    # Product totals
    for cls in sorted(set(c for c, t in grouped.keys())):
        cls_total = sum(d['count'] for (c, t), d in grouped.items() if c == cls)
        ws.cell(row=row, column=1, value=f'{cls} Total').font = Font(bold=True)
        ws.cell(row=row, column=4, value=cls_total).font = Font(bold=True)
        row += 1

    # Grand total
    row += 1
    ws.cell(row=row, column=1, value='GRAND TOTAL').font = Font(bold=True, size=12)
    ws.cell(row=row, column=4, value=total).font = Font(bold=True, size=12)

    # Column widths
    widths = [30, 15, 15, 8, 18, 12, 12, 12, 25, 12, 25]
    for ci, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + ci)].width = w

    # Sheet 2: RawData (every detection, flat)
    ws2 = wb.create_sheet('RawData')
    for ci, h in enumerate(HEADERS, 1):
        cell = ws2.cell(row=1, column=ci, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='DDDDDD')

    row = 2
    for page_idx in sorted(detections_per_page.keys()):
        for d in sorted(detections_per_page[page_idx], key=lambda x: (x['cls'], x.get('tag') or '')):
            tag = d.get('tag') or ''
            details = schedule_details.get(tag, {})
            brand_model = _prop(details, ['MANUFACTURER & MODEL', 'MANUFACTURER'])
            if brand_model and ' ' in brand_model and not _prop(details, ['MODEL']):
                brand, model = brand_model.split(' ', 1)
            else:
                brand = _prop(details, ['MANUFACTURER', 'BRAND'])
                model = _prop(details, ['MODEL'])
            neck_size = _prop(details, ['NECK', 'SIZE (NECK)', 'SIZE'])
            etype = _prop(details, ['SERVICE', 'TYPE', 'DESCRIPTION'])

            ws2.cell(row=row, column=1, value=d['cls'])
            ws2.cell(row=row, column=2, value=brand)
            ws2.cell(row=row, column=3, value=model)
            ws2.cell(row=row, column=4, value=1)
            ws2.cell(row=row, column=5, value=tag)
            ws2.cell(row=row, column=6, value=neck_size)
            ws2.cell(row=row, column=9, value=etype)
            row += 1

    ws2.column_dimensions['A'].width = 8
    ws2.column_dimensions['B'].width = 35
    ws2.column_dimensions['C'].width = 12

    wb.save(output_path)
    return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

# Sheet-title phrases that mark a page as NOT a floor plan. Across the 6
# May-5 LS reviews, ~50 / 61 phantom detections came from pages of these
# types (legends, schedules, details, notes, cover sheets). Skipping them
# at inference time eliminates the largest single phantom source.
NON_PLAN_TITLE_MARKERS = [
    'MECHANICAL LEGEND', 'HVAC LEGEND', 'PLUMBING LEGEND',
    'LEGEND AND ABBREVIATIONS', 'LEGENDS AND SCHEDULES', 'SCHEDULE AND LEGEND',
    'GENERAL NOTES', 'MECHANICAL NOTES', 'HVAC NOTES',
    'MECHANICAL SCHEDULE', 'HVAC SCHEDULE', 'EQUIPMENT SCHEDULE',
    'AIR DEVICE SCHEDULE', 'DIFFUSER SCHEDULE', 'FAN SCHEDULE',
    'MECHANICAL DETAILS', 'HVAC DETAILS', 'TYPICAL DETAILS',
    'PIPING DETAILS', 'INSTALLATION DETAILS',
    'TITLE SHEET', 'COVER SHEET', 'SHEET INDEX', 'DRAWING INDEX',
    'SYMBOLS AND ABBREVIATIONS',
]

PLAN_KEYWORDS = [
    'MECHANICAL PLAN', 'CEILING PLAN', 'HVAC PLAN',
    'VENTILATION PLAN', 'FLOOR PLAN', 'ROOF PLAN',
]


def _is_non_plan_sheet(text_upper):
    """True if page is a legend / schedule / details / notes sheet.

    Match on compound title phrases (e.g., 'MECHANICAL SCHEDULE') rather
    than bare words so floor plans that happen to mention 'SCHEDULE' or
    'LEGEND' in a callout aren't filtered out.
    """
    return any(m in text_upper for m in NON_PLAN_TITLE_MARKERS)


def find_mechanical_pages(pdf_path):
    """
    Heuristic: scan all pages and pick the ones likely to be mechanical floor plans.
    Looks for HVAC keywords in the page text, then drops pages whose sheet
    title marks them as a legend / schedule / details / notes / cover sheet.
    """
    doc = fitz.open(pdf_path)
    total = doc.page_count
    candidate_pages = []
    skipped_non_plan = []
    for pi in range(total):
        text = doc[pi].get_text().upper()
        if not any(kw in text for kw in PLAN_KEYWORDS):
            continue
        if _is_non_plan_sheet(text):
            skipped_non_plan.append(pi + 1)
            continue
        candidate_pages.append(pi)
    doc.close()

    if skipped_non_plan:
        print(f"  Skipping non-plan sheets: pages {skipped_non_plan}")

    if not candidate_pages:
        return list(range(total))
    return candidate_pages


def main():
    parser = argparse.ArgumentParser(description='HVAC Takeoff CLI Tool')
    parser.add_argument('pdf', help='Path to blueprint PDF')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Confidence threshold (0-1)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as PDF)')
    parser.add_argument('--all-pages', action='store_true', help='Process all pages (not just mechanical)')
    parser.add_argument('--pages', type=int, nargs='+', help='Specific page numbers (1-indexed)')
    parser.add_argument('--verify', action='store_true',
                        help='Print full schedule variable dump and exit (no detection run)')
    parser.add_argument('--schedule-only', action='store_true',
                        help='Parse schedule and write variables JSON, skip YOLO detection')
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    if not args.schedule_only and not Path(args.model).exists():
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

    # Extract title-block info (best-effort)
    project_info = extract_project_info(pdf_path)
    print_project_info(project_info)
    print()

    # Parse schedule first — always, even in --schedule-only mode
    print("Parsing schedule...")
    variables = []
    try:
        schedules, marks, mark_details, legend, sched_summary, variables = parse_pdf_schedules(str(pdf_path))
        print(f"  {len(schedules)} schedule table(s), {len(marks)} unique tag(s), "
              f"{len(variables)} variable(s)")
        if marks:
            print(f"  Sample tags: {marks[:10]}{'...' if len(marks) > 10 else ''}")
    except Exception as e:
        print(f"  Schedule parse failed: {e}")
        schedules, marks, mark_details = [], [], {}

    # Always write variables JSON sidecar
    variables_path = out_dir / f"{pdf_path.stem}_variables.json"
    try:
        with open(variables_path, 'w', encoding='utf-8') as f:
            json.dump(variables, f, indent=2, default=str, ensure_ascii=False)
        print(f"  Wrote {len(variables)} variables to {variables_path.name}")
    except Exception as e:
        print(f"  (JSON sidecar failed: {e})")

    # Project info sidecar
    if project_info and not ('_error' in project_info and len(project_info) == 1):
        info_path = out_dir / f"{pdf_path.stem}_project_info.json"
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(project_info, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # Verification dump to stdout
    if args.verify or args.schedule_only:
        dump_variables(variables)

    # Early exit if schedule-only
    if args.schedule_only:
        print(f"\n--schedule-only: skipping detection. Output in {out_dir}")
        return

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

    # Process each page — keep rendered images so tag inference can OCR
    # bubbles next to each detection (Level 2b)
    detections_per_page = {}
    page_images = {}
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
            page_images[page_idx] = img

    total_elapsed = time.time() - t_start
    print(f"\nDetection complete in {total_elapsed:.0f}s")

    # Tag inference — 3-level system
    if detections_per_page:
        print("\nInferring tags...")
        detections_per_page, tag_stats = infer_tags(
            detections_per_page, schedules, marks, mark_details, str(pdf_path),
            variables=variables, page_images=page_images
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
    write_excel(str(excel_path), detections_per_page, pdf_path.stem, mark_details)

    detections_json_path = out_dir / f"{pdf_path.stem}_detections.json"
    print(f"  Detections:     {detections_json_path}")
    det_dump = {
        'pdf': str(pdf_path),
        'dpi': DPI,
        'pages': {
            str(page_idx): [
                {
                    'cls': d['cls'],
                    'tag': d.get('tag'),
                    'tag_method': d.get('tag_method'),
                    'conf': d.get('conf'),
                    'x1': d['x1'], 'y1': d['y1'], 'x2': d['x2'], 'y2': d['y2'],
                }
                for d in dets
            ]
            for page_idx, dets in detections_per_page.items()
        },
    }
    with open(detections_json_path, 'w', encoding='utf-8') as f:
        json.dump(det_dump, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"DONE — open {out_dir} to see the results")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
