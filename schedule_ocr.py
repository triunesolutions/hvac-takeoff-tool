"""
schedule_ocr.py — Stage 4 OCR fallback for raster schedules.

The text-layer schedule parser (schedule_parser.py) fails on CAD-exported
PDFs where the schedule LOOKS like a table but the cell contents are raster
graphics, not embedded text. Art Vascular is the canonical case:
variables.json came out as `[]` because pdfplumber couldn't read the cells.

This module:
  1. Renders the schedule page at high DPI (300+).
  2. Runs EasyOCR across the page.
  3. Reconstructs table structure by clustering OCR boxes into rows + columns.
  4. Identifies the tag column.
  5. Extracts TagVariable-shaped dicts.

It is invoked as a fallback when the text-layer parser returned no tags
for a page that the page classifier identified as 'schedule' or 'air_balance'.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# EasyOCR is expensive to import (~100 MB model load). Lazy.
_reader = None
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


# A tag looks like 1-4 letters + dash + 1-3 digits, optionally with trailing letter.
# Requires digits — avoids catching English words like CFM, THE, MUST, ON as tags.
TAG_PATTERN = re.compile(r'^[A-Z]{1,4}-?\d{1,3}[A-Z]?$')

# Common English words that look like potential tags but aren't. Defensive.
NON_TAG_WORDS = {'CFM', 'MUST', 'THE', 'ON', 'AND', 'FOR', 'NO', 'YES',
                'TO', 'OF', 'IN', 'BE', 'OR', 'WITH', 'BY', 'AT',
                'A', 'B', 'C', 'D', 'E',  # too short alone
                'NTS', 'NIC', 'TYP', 'EXIST', 'NEW'}

# Header keywords that mark the tag column. 'SYMBOL' is added for drawings
# where the schedule maps drawn symbols to specs (Art Vascular style).
TAG_HEADER_WORDS = {'MARK', 'TAG', 'UNIT', 'ID', 'NUMBER', 'NO', 'NO.', '#', 'SYMBOL'}

# Property column header keywords — used to identify "this is a schedule"
PROP_HEADER_WORDS = {
    'MANUFACTURER', 'MAKE', 'MODEL', 'TYPE', 'SIZE', 'CFM', 'AIR',
    'MCA', 'MOCP', 'VOLT', 'WT', 'WEIGHT', 'LBS', 'NECK',
    'MOUNTING', 'SERVICE', 'REMARK', 'NOTE',
}


def render_page(pdf_path: Path, page_index: int, dpi: int = 300) -> 'numpy.ndarray':
    """Render a page to a numpy RGB image."""
    import numpy as np
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi, annots=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        return img
    finally:
        doc.close()


def ocr_page(img) -> list[dict]:
    """Return list of {text, bbox, conf} where bbox is (x1,y1,x2,y2) in pixel coords."""
    reader = _get_reader()
    raw = reader.readtext(img, detail=1, paragraph=False)
    out = []
    for box, text, conf in raw:
        # box is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            'text': text.strip(),
            'bbox': (min(xs), min(ys), max(xs), max(ys)),
            'conf': float(conf),
            'cx': (min(xs) + max(xs)) / 2,
            'cy': (min(ys) + max(ys)) / 2,
        })
    return out


def cluster_rows(items: list[dict], row_tol_px: float = 25) -> list[list[dict]]:
    """Cluster OCR items into rows by Y center. Returns list of rows, each sorted by X."""
    if not items:
        return []
    sorted_by_y = sorted(items, key=lambda d: d['cy'])
    rows = [[sorted_by_y[0]]]
    last_y = sorted_by_y[0]['cy']
    for it in sorted_by_y[1:]:
        if abs(it['cy'] - last_y) <= row_tol_px:
            rows[-1].append(it)
        else:
            rows.append([it])
        last_y = it['cy']
    # Sort each row by X
    for r in rows:
        r.sort(key=lambda d: d['cx'])
    return rows


def cluster_columns(rows: list[list[dict]], col_tol_px: float = 50) -> list[float]:
    """Find shared column x-centers across rows. Returns sorted list of x positions."""
    all_cx = []
    for row in rows:
        for it in row:
            all_cx.append(it['cx'])
    if not all_cx:
        return []
    all_cx.sort()
    cols = [all_cx[0]]
    for cx in all_cx[1:]:
        if cx - cols[-1] > col_tol_px:
            cols.append(cx)
    return cols


def assign_to_columns(row: list[dict], col_centers: list[float],
                      col_tol_px: float = 80) -> dict[int, str]:
    """For each row, snap each item to the nearest column. Returns {col_idx: text}."""
    out: dict[int, str] = {}
    for it in row:
        # Nearest column
        best_i, best_d = -1, 1e9
        for i, cx in enumerate(col_centers):
            d = abs(it['cx'] - cx)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= col_tol_px:
            existing = out.get(best_i, '')
            out[best_i] = (existing + ' ' + it['text']).strip() if existing else it['text']
    return out


def _score_header_row(row: list[dict]) -> tuple[int, int]:
    """Score one row as a potential header. Returns (score, tag_col_index)."""
    score = 0
    tag_idx = -1
    for ci, it in enumerate(row):
        t = it['text'].upper().strip(' :,.-')
        if t in TAG_HEADER_WORDS:
            if tag_idx < 0:
                tag_idx = ci
            score += 3  # tag header is a strong signal
        elif any(p in t for p in PROP_HEADER_WORDS):
            score += 1
    return score, tag_idx


def find_header_row(rows: list[list[dict]]) -> tuple[int, int]:
    """Find the single best header row. Returns (row_index, tag_col_index_within_row),
    or (-1, -1) if none. Kept for backward compatibility / single-table pages."""
    best_score = 0
    best_row = -1
    best_tag_idx = -1
    for ri, row in enumerate(rows):
        score, tag_idx = _score_header_row(row)
        if score > best_score:
            best_score = score
            best_row = ri
            best_tag_idx = tag_idx
    return best_row, best_tag_idx


# Minimum header score to start a new schedule block. A real schedule header
# has a tag column (+3) AND several property columns, so >=4 is conservative.
HEADER_MIN_SCORE = 4


def find_all_header_rows(rows: list[list[dict]]) -> list[tuple[int, int]]:
    """Find EVERY row that looks like a schedule header. Large drawing sheets
    (E-size) stack multiple schedules vertically — RTU, air devices, ERV,
    louver, air balance — each with its own header. Returns a list of
    (row_index, tag_col_index) sorted by row, deduped so two adjacent
    header-ish rows (wrapped header text) don't both fire."""
    hits = []
    for ri, row in enumerate(rows):
        score, tag_idx = _score_header_row(row)
        if score >= HEADER_MIN_SCORE:
            hits.append((ri, score, tag_idx))
    # Collapse headers that are within 1 row of each other (wrapped header lines):
    # keep the higher-scoring one.
    collapsed: list[tuple[int, int]] = []
    for ri, score, tag_idx in hits:
        if collapsed and ri - collapsed[-1][0] <= 1:
            # adjacent to previous header — keep whichever scored higher
            if score > collapsed[-1][2]:
                collapsed[-1] = (ri, tag_idx, score)
        else:
            collapsed.append((ri, tag_idx, score))
    return [(ri, tag_idx) for (ri, tag_idx, _score) in collapsed]


# Characters EasyOCR commonly confuses inside equipment tags.
def _normalize_ocr_tag(raw: str) -> str:
    """Clean an OCR'd tag: uppercase, strip spaces/punctuation. Tags are
    LETTER(S) + optional dash + DIGITS, so fix digit/letter confusions
    positionally — 'SOF'->'50F'? No: only fix within the digit run. We keep
    this conservative: uppercase + strip, and map a trailing lone 'O' inside a
    numeric run to '0'. Aggressive fixes are left to downstream tag matching."""
    s = (raw or '').upper().strip()
    s = s.replace(' ', '').replace(',', '').strip('.:;-')
    return s


def _extract_block(rows: list[list[dict]], header_ri: int, tag_col_local: int,
                   body_end: int, page_index: int, schedule_name: str) -> list[dict]:
    """Extract tag-rows for ONE schedule block: rows (header_ri, body_end)."""
    header_row = rows[header_ri]
    col_centers = [it['cx'] for it in header_row]
    column_names = [it['text'] for it in header_row]

    # Canonical tag column within this block's header
    tag_col_idx = -1
    for i, name in enumerate(column_names):
        if name.upper().strip(' :,.-') in TAG_HEADER_WORDS:
            tag_col_idx = i
            break
    if tag_col_idx < 0:
        tag_col_idx = tag_col_local

    out = []
    for ri in range(header_ri + 1, body_end):
        row = rows[ri]
        cell_dict = assign_to_columns(row, col_centers, col_tol_px=80)
        tag_raw = cell_dict.get(tag_col_idx, '').strip()
        tag_norm = _normalize_ocr_tag(tag_raw)
        if tag_norm in NON_TAG_WORDS or not TAG_PATTERN.match(tag_norm):
            continue
        properties = {}
        for ci, name in enumerate(column_names):
            if ci == tag_col_idx:
                continue
            val = cell_dict.get(ci, '').strip()
            if val:
                properties[name.upper()] = val
        out.append({
            'tag': tag_norm,
            'schedule_name': schedule_name or f'OCR schedule p{page_index+1}',
            'page': page_index + 1,
            'properties': properties,
            'source_row_index': ri,
            'inferred_yolo_class': None,
            '_source': 'ocr',
            '_conf': sum(it['conf'] for it in row) / max(1, len(row)),
        })
    return out


def extract_variables_from_page(pdf_path: Path, page_index: int,
                                schedule_name: str = '',
                                dpi: int = 300,
                                img=None) -> list[dict]:
    """OCR + parse a schedule page, handling MULTIPLE stacked schedules per
    sheet (common on E-size drawings). Returns TagVariable-shaped dicts.

    Pass a pre-rendered ``img`` (numpy RGB) to avoid re-rendering when the
    caller already has the page raster.
    """
    if img is None:
        img = render_page(pdf_path, page_index, dpi=dpi)
    items = ocr_page(img)
    rows = cluster_rows(items)
    if not rows:
        return []

    headers = find_all_header_rows(rows)
    if not headers:
        # Fall back to single best header (lower bar) for simple pages
        hri, tci = find_header_row(rows)
        if hri < 0:
            return []
        headers = [(hri, tci)]

    variables: list[dict] = []
    for idx, (header_ri, tag_col_local) in enumerate(headers):
        # This block's body ends where the next schedule's header begins.
        body_end = headers[idx + 1][0] if idx + 1 < len(headers) else len(rows)
        variables.extend(
            _extract_block(rows, header_ri, tag_col_local, body_end,
                           page_index, schedule_name)
        )

    # Dedup: the same tag can appear if blocks overlap; keep the row with more props.
    by_tag: dict[str, dict] = {}
    for v in variables:
        k = v['tag']
        if k not in by_tag or len(v['properties']) > len(by_tag[k]['properties']):
            by_tag[k] = v
    return list(by_tag.values())


# ----------------------------------------------------------------------------
# Table-region segmentation (Path A) — for dense E-size sheets that pack several
# schedules (+ a legend column + a notes column) onto one page. Row/column
# clustering over the whole page scrambles such layouts; instead we detect each
# ruled table box with OpenCV, then OCR + parse each box independently.
# ----------------------------------------------------------------------------

def detect_table_regions(img, min_area_frac: float = 0.004,
                         max_area_frac: float = 0.55) -> list[tuple[int, int, int, int]]:
    """Detect ruled-table bounding boxes in an RGB page image.

    Returns a list of (x1,y1,x2,y2) pixel boxes, largest first. Empty list if
    OpenCV is unavailable or no usable grid is found (caller then falls back to
    whole-page clustering).

    Approach: isolate long H/V rulings, intersect them to find table *joints*
    (line crossings), then connected-component the joints. A real table has a
    dense block of joints; the page border (just a rectangle) has only corner
    joints, so it does not form a block. This separates side-by-side and
    stacked tables that a naive grid-contour would merge into one blob.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    page_area = float(h * w)
    # Simple fixed threshold: dark ink -> white on black. (Adaptive threshold
    # over-detects on anti-aliased CAD renders — ~60% false ink.)
    bw = (gray < 128).astype('uint8') * 255
    # Isolate long horizontal and vertical rulings.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 45), 1))
    horiz = cv2.dilate(cv2.erode(bw, hk), hk)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 45)))
    vert = cv2.dilate(cv2.erode(bw, vk), vk)
    grid = cv2.bitwise_or(horiz, vert)
    # Close small gaps so each table's rulings form one connected component;
    # tables separated by whitespace stay separate.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
                            iterations=2)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < min_area_frac * page_area or area > max_area_frac * page_area:
            continue
        if ww < w * 0.05 or hh < h * 0.02:   # too thin to be a table
            continue
        pad = int(0.004 * (w + h))   # include outer rulings + nearby labels
        boxes.append((max(0, x - pad), max(0, y - pad),
                      min(w, x + ww + pad), min(h, y + hh + pad)))
    boxes.sort(key=lambda b: (b[3] - b[1]) * (b[2] - b[0]), reverse=True)
    return boxes


def _accept_tag(tag_norm: str, known_tag_col: bool) -> bool:
    """Whether an OCR'd tag-column value is a real tag. When we KNOW the column
    is the MARK/TAG column (region-based parsing), accept single-letter marks
    (A, B, C, D — common for air devices) that the generic pattern rejects."""
    if not tag_norm:
        return False
    if TAG_PATTERN.match(tag_norm):
        return tag_norm not in NON_TAG_WORDS
    if known_tag_col:
        # Single letter, or short letter(+digit) mark inside a confirmed column.
        if re.match(r'^[A-Z]{1,3}\d{0,3}[A-Z]?$', tag_norm):
            return True
    return False


def extract_variables_from_region(img, x1: int, y1: int, x2: int, y2: int,
                                  page_index: int, schedule_name: str = '') -> list[dict]:
    """OCR + parse ONE cropped table region (clean single-table layout)."""
    crop = img[y1:y2, x1:x2]
    items = ocr_page(crop)
    rows = cluster_rows(items)
    if not rows:
        return []
    header_ri, tag_col_local = find_header_row(rows)
    if header_ri < 0:
        return []
    header_row = rows[header_ri]
    col_centers = [it['cx'] for it in header_row]
    column_names = [it['text'] for it in header_row]
    tag_col_idx = -1
    for i, name in enumerate(column_names):
        if name.upper().strip(' :,.-') in TAG_HEADER_WORDS:
            tag_col_idx = i
            break
    known_tag_col = tag_col_idx >= 0
    if tag_col_idx < 0:
        tag_col_idx = 0   # leftmost column is the mark column by convention

    out = []
    for ri in range(header_ri + 1, len(rows)):
        cell_dict = assign_to_columns(rows[ri], col_centers, col_tol_px=80)
        tag_norm = _normalize_ocr_tag(cell_dict.get(tag_col_idx, ''))
        if not _accept_tag(tag_norm, known_tag_col):
            continue
        properties = {}
        for ci, name in enumerate(column_names):
            if ci == tag_col_idx:
                continue
            val = cell_dict.get(ci, '').strip()
            if val:
                properties[name.upper()] = val
        out.append({
            'tag': tag_norm,
            'schedule_name': schedule_name or f'OCR region p{page_index+1}',
            'page': page_index + 1,
            'properties': properties,
            'source_row_index': ri,
            'inferred_yolo_class': None,
            '_source': 'ocr_region',
            '_conf': sum(it['conf'] for it in rows[ri]) / max(1, len(rows[ri])),
        })
    return out


def extract_variables_region_based(pdf_path: Path, page_index: int,
                                   dpi: int = 200, img=None) -> list[dict]:
    """Segment a page into ruled-table regions and parse each independently.
    Returns [] if no usable table regions were detected (caller falls back)."""
    if img is None:
        img = render_page(pdf_path, page_index, dpi=dpi)
    regions = detect_table_regions(img)
    if not regions:
        return []
    all_vars: list[dict] = []
    for (x1, y1, x2, y2) in regions:
        try:
            all_vars.extend(
                extract_variables_from_region(img, x1, y1, x2, y2, page_index)
            )
        except Exception as e:
            print(f'[schedule_ocr] region ({x1},{y1},{x2},{y2}) failed: {e}')
    # Dedup by tag, keep richest row.
    by_tag: dict[str, dict] = {}
    for v in all_vars:
        k = v['tag']
        if k not in by_tag or len(v['properties']) > len(by_tag[k]['properties']):
            by_tag[k] = v
    return list(by_tag.values())


def extract_all_schedules(pdf_path: Path,
                          schedule_page_nums: list[int],
                          dpi: int = 300) -> list[dict]:
    """Run OCR on every candidate schedule page, return aggregated variables.

    Strategy per page: render once, try table-region segmentation first (best
    for dense multi-schedule sheets); if that yields nothing, fall back to
    whole-page multi-block clustering.
    """
    out = []
    for p in schedule_page_nums:
        try:
            img = render_page(pdf_path, p - 1, dpi=dpi)
            vars_ = extract_variables_region_based(pdf_path, p - 1, dpi=dpi, img=img)
            if not vars_:
                vars_ = extract_variables_from_page(pdf_path, p - 1, dpi=dpi, img=img)
            out.extend(vars_)
        except Exception as e:
            print(f'[schedule_ocr] page {p}: {e}')
    return out


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--page', type=int, required=True, help='1-based page number to OCR')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--out', help='Output JSON file (default: stdout)')
    args = ap.parse_args()

    vars_ = extract_variables_from_page(Path(args.pdf), args.page - 1, dpi=args.dpi)
    print(f'Found {len(vars_)} tag variables on page {args.page}:')
    for v in vars_:
        print(f'  {v["tag"]:8s}  {len(v["properties"])} props  conf={v["_conf"]:.2f}')
        for k, val in list(v['properties'].items())[:5]:
            print(f'      {k[:25]:25s} = {val[:40]}')

    if args.out:
        Path(args.out).write_text(json.dumps(vars_, indent=2), encoding='utf-8')
        print(f'\nWrote {args.out}')
