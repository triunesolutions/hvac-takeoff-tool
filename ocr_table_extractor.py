"""OCR-based schedule-table reconstruction for vector/CAD schedule sheets.

Problem: some schedule sheets are exported from CAD with the cell text drawn as
vector line-art (curves/strokes), not as a text layer. pdfplumber / PyMuPDF then
extract ZERO cell text, so the normal `schedule_parser` path finds no tags and
no properties. The existing `extract_marks_via_ocr` fallback recovers the tag
list but leaves every property blank.

This module reconstructs the FULL table. CAD schedules are cleanly ruled, so we:
  1. render the page,
  2. detect the table ruling lines (morphology) -> column x's and row y's,
  3. OCR the page once with bounding boxes and drop each token into its grid cell,
  4. find the header row + MARK column + schedule title,
  5. emit TagVariable dicts WITH populated `properties` — identical shape to
     `schedule_parser.parse_pdf_schedules`, so tag inference and the Excel writer
     run unchanged.

Entry point: `extract_tables_via_ocr(pdf_path, pages=None, dpi=200, ...)`.
"""

import re

# Reused from the text-path parser so OCR output is normalized identically.
from schedule_parser import (
    normalize_tag, expand_tag_cell, TAG_COL_KEYWORDS, PROPERTY_KEYWORDS,
    NON_HVAC_SCHEDULE_KEYWORDS, _ocr_infer_class,
)

# Header tokens we expect in a schedule header row. Used to (a) locate the header
# row and (b) fuzzy-correct OCR noise in header labels back to a canonical word.
HEADER_VOCAB = sorted({
    "MARK", "TAG", "SYMBOL", "DESIGNATION", "UNIT", "TYPE", "SERVICE", "AREA",
    "MANUFACTURER", "MODEL", "BASIS OF DESIGN", "QTY", "QUANTITY",
    "CFM", "VOLUME", "AIRFLOW", "ESP", "STATIC", "RPM", "HP", "BHP",
    "MOTOR", "VOLTAGE", "VOLTS", "PHASE", "MCA", "MOCP", "WATTS", "KW",
    "WEIGHT", "MOUNTING", "LOCATION", "SIZE", "NECK", "DUCT", "COLOR",
    "MATERIAL", "DAMPER", "FINISH", "MBH", "BTUH", "TONS", "TONNAGE",
    "EAT", "LAT", "GPM", "REMARKS", "NOTES", "ELECTRICAL", "COOLING", "HEATING",
})

# Tokens that, appearing on a sheet, mark it as a schedule sheet worth reading.
_SCHED_PAGE_KW = ("SCHEDULE", "MANUFACTURER", "MODEL", "CFM", "DIFFUSER",
                  "GRILLE", "REGISTER", "EXHAUST", "CONDENSING", "ROOFTOP",
                  "MAKE-UP", "MAKEUP", "VENTILATOR", "LOUVER")

# Common OCR character confusions in CAD-rendered text (digits<->letters).
_CONFUSE = str.maketrans({})  # placeholder; value cleanup is conservative below


def _lazy_cv():
    import numpy as np
    import cv2
    return np, cv2


# Columns whose values are numeric measurements — safe to snap O->0 etc. Model
# and manufacturer columns must NOT be cleaned this way (272RS != 272R5).
_NUMERIC_COLS = ("CFM", "VOLUME", "AIRFLOW", "ESP", "STATIC", "RPM", "HP", "BHP",
                 "MCA", "MOCP", "WATTS", "KW", "WEIGHT", "MBH", "BTUH", "TONS",
                 "TONNAGE", "GPM", "EAT", "LAT", "VOLTAGE", "VOLTS")


def _clean_value(s, numeric=False):
    """Collapse whitespace + uppercase. For numeric-measurement columns, also
    snap digit<->letter OCR confusions (2OO0->2000); never for model/text cols."""
    if not s:
        return s
    out = []
    for tok in str(s).split():
        u = tok.upper()
        if numeric and any(c in 'OISBZ' for c in u):
            u = (u.replace('O', '0').replace('I', '1').replace('S', '5')
                   .replace('B', '8').replace('Z', '2'))
        out.append(u)
    return ' '.join(out)


# --------------------------------------------------------------------------- #
# Grid detection
# --------------------------------------------------------------------------- #
def _line_positions(mask, axis, min_frac, page_len):
    """Collapse a horizontal/vertical line mask into a sorted list of line
    centers. `axis=0` projects onto x (vertical lines -> column separators);
    `axis=1` projects onto y (horizontal lines -> row separators). A position is
    a line if its projection exceeds `min_frac` of the perpendicular extent."""
    proj = mask.sum(axis=axis) / 255.0
    thresh = min_frac * page_len
    on = proj > thresh
    centers = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            centers.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    return centers


def _line_masks(gray):
    np, cv2 = _lazy_cv()
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, -2)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 120), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 120)))
    horiz = cv2.dilate(cv2.erode(bw, hk), hk)
    vert = cv2.dilate(cv2.erode(bw, vk), vk)
    return horiz, vert


# Schedule-title detector: a text LINE that names a schedule. Stacked tables on
# one sheet are merged by the connected-components grid, so we instead anchor on
# each title and detect the ruled grid in the band beneath it.
_TITLE_RE = re.compile(
    r'\b(SCHEDULE|AIR DISTRIBUTION DEVICES?|DIFFUSER|GRILLE|REGISTER|'
    r'LOAD SUMMARY|EXHAUST FAN|MAKE.?UP AIR|ROOFTOP|ROOF TOP|CONDENSING|'
    r'SPLIT SYSTEM|UNIT HEATER|ENERGY RECOVERY|VENTILATOR|LOUVER|'
    r'AIR HOOD|AIR CURTAIN|FAN COIL|HEAT PUMP|PACKAGE)', re.IGNORECASE)


def _cluster_lines(tokens, y_tol=22, x_gap=140):
    """Group OCR tokens (cx, cy, h, text) into text lines, breaking a line when
    tokens are on a different y-row OR separated by a large horizontal gap (two
    titles sharing a row in different sheet columns must not merge)."""
    items = sorted(tokens, key=lambda t: (round(t[1] / y_tol), t[0]))
    lines = []
    cur = []
    last_y = None
    last_x = None
    for cx, cy, h, txt in items:
        new_row = last_y is not None and abs(cy - last_y) > y_tol
        big_gap = last_x is not None and (cx - last_x) > x_gap
        if (new_row or big_gap) and cur:
            lines.append(_mk_line(cur))
            cur = []
        cur.append((cx, cy, h, txt))
        last_y = cy
        last_x = cx
    if cur:
        lines.append(_mk_line(cur))
    return lines


def _mk_line(toks):
    toks = sorted(toks)
    return {
        'cy': sum(t[1] for t in toks) / len(toks),
        'x0': min(t[0] for t in toks),
        'x1': max(t[0] for t in toks),
        'h': max(t[2] for t in toks),
        'text': ' '.join(t[3] for t in toks).strip(),
    }


def _find_titles(tokens):
    """Title lines: match a schedule keyword AND are taller than the median
    token (titles are set larger than cell text)."""
    if not tokens:
        return []
    heights = sorted(t[2] for t in tokens)
    med_h = heights[len(heights) // 2]
    titles = []
    for ln in _cluster_lines(tokens):
        if _TITLE_RE.search(ln['text']) and ln['h'] >= 1.15 * med_h:
            titles.append(ln)
    titles.sort(key=lambda l: l['cy'])
    return titles


def _grid_in_band(horiz, vert, x0, x1, y0, y1):
    """Detect row/col lines for the ruled table inside a band."""
    x0, x1 = max(0, int(x0)), int(x1)
    y0, y1 = max(0, int(y0)), int(y1)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    sub_h = horiz[y0:y1, x0:x1]
    sub_v = vert[y0:y1, x0:x1]
    rows = _line_positions(sub_h, axis=1, min_frac=0.40, page_len=(x1 - x0))
    cols = _line_positions(sub_v, axis=0, min_frac=0.40, page_len=(y1 - y0))
    if len(rows) < 3 or len(cols) < 3:
        return None
    # Tighten the band to the actual ruled extent.
    rows = [y0 + r for r in rows]
    cols = [x0 + c for c in cols]
    return {'x0': cols[0] - 3, 'y0': rows[0] - 3, 'x1': cols[-1] + 3,
            'y1': rows[-1] + 3, 'rows': rows, 'cols': cols}


def _detect_tables(gray, tokens):
    """Cell-based detection. Find individual ruled cells (local rectangles,
    immune to the sheet's full-width frame lines), cluster spatially-adjacent
    cells into tables, then derive each table's row/col line positions from its
    cell edges. Schedule name = the OCR line just above the cluster."""
    np, cv2 = _lazy_cv()
    H, W = gray.shape
    horiz, vert = _line_masks(gray)
    grid = cv2.bitwise_or(horiz, vert)
    # Cell interiors = white regions enclosed by grid lines. The page background
    # is one huge component (filtered by max area); real cells are mid-sized.
    inv = cv2.bitwise_not(grid)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    page_area = H * W
    cells = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200 or area > 0.02 * page_area:
            continue
        if w < 12 or h < 8 or w > 0.5 * W or h > 0.2 * H:
            continue
        cells.append((x, y, w, h))
    if not cells:
        return []

    # Cluster cells into tables via union-find on adjacency (cells whose edges
    # nearly touch belong to the same table).
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    gap = max(10, W // 300)
    # sort by position for a near-linear adjacency sweep
    order = sorted(range(len(cells)), key=lambda i: (cells[i][1], cells[i][0]))
    for ii in range(len(order)):
        ax, ay, aw, ah = cells[order[ii]]
        for jj in range(ii + 1, len(order)):
            bx, by, bw, bh = cells[order[jj]]
            if by - (ay + ah) > 4 * ah + gap:
                break  # far below; nothing further can be adjacent
            # Same table only when two cells actually share a ruling edge:
            # edge-touching on one axis AND meaningful overlap on the other.
            # This keeps a table's cells together while the title-band gap
            # between stacked schedules (and the wide gap between sheet columns)
            # stays unmerged.
            x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
            y_overlap = min(ay + ah, by + bh) - max(ay, by)
            touch_h = abs(bx - (ax + aw)) < gap or abs((bx + bw) - ax) < gap
            touch_v = abs(by - (ay + ah)) < gap or abs((by + bh) - ay) < gap
            horiz_adj = touch_h and y_overlap > 0.3 * min(ah, bh)
            vert_adj = touch_v and x_overlap > 0.3 * min(aw, bw)
            if horiz_adj or vert_adj:
                union(order[ii], order[jj])

    groups = {}
    for i in range(len(cells)):
        groups.setdefault(find(i), []).append(cells[i])

    tables = []
    for g in groups.values():
        if len(g) < 6:           # a real schedule has many cells
            continue
        xs0 = [c[0] for c in g]
        ys0 = [c[1] for c in g]
        xs1 = [c[0] + c[2] for c in g]
        ys1 = [c[1] + c[3] for c in g]
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        if (x1 - x0) < 0.12 * W or (y1 - y0) < 0.03 * H:
            continue
        # Derive clean row/col lines from the actual ruling within this tight
        # bbox (avoids the over-segmentation of clustering raw cell edges, where
        # one tall visual row splits into several empty bands).
        g2 = _grid_in_band(horiz, vert, x0 - 4, x1 + 4, y0 - 4, y1 + 4)
        if g2 is None:
            cols = _cluster_edges([c[0] for c in g] + [c[0] + c[2] for c in g],
                                  tol=max(6, W // 400))
            rows = _cluster_edges([c[1] for c in g] + [c[1] + c[3] for c in g],
                                  tol=max(5, H // 500))
            if len(cols) < 3 or len(rows) < 3:
                continue
        else:
            rows, cols = g2['rows'], g2['cols']
        tables.append({'x0': x0 - 3, 'y0': y0 - 3, 'x1': x1 + 3, 'y1': y1 + 3,
                       'rows': rows, 'cols': cols,
                       'title': _title_above({'x0': x0, 'x1': x1, 'y0': y0},
                                              tokens)})
    tables.sort(key=lambda t: (t['y0'], t['x0']))
    return tables


def _cluster_edges(vals, tol):
    """Collapse a list of edge coordinates into representative line positions."""
    vals = sorted(vals)
    out = []
    cur = [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            out.append(sum(cur) // len(cur))
            cur = [v]
    out.append(sum(cur) // len(cur))
    return out


def _contained(inner, outer, frac=0.85):
    ix = max(0, min(inner['x1'], outer['x1']) - max(inner['x0'], outer['x0']))
    iy = max(0, min(inner['y1'], outer['y1']) - max(inner['y0'], outer['y0']))
    inter = ix * iy
    a = (inner['x1'] - inner['x0']) * (inner['y1'] - inner['y0'])
    return a > 0 and inter / a > frac


# --------------------------------------------------------------------------- #
# Cell assignment
# --------------------------------------------------------------------------- #
def _band_index(value, edges):
    """Return the band (0..len(edges)-2) that `value` falls into, or None."""
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def _build_cell_grid(table, tokens):
    """tokens: list of (cx, cy, text). Returns a 2D list grid[r][c] = joined
    text, using the table's row/col line positions as cell boundaries."""
    rows = sorted(set(table['rows']))
    cols = sorted(set(table['cols']))
    nR, nC = len(rows) - 1, len(cols) - 1
    if nR < 1 or nC < 1:
        return None, None, None
    cells = [[[] for _ in range(nC)] for _ in range(nR)]
    for cx, cy, _h, text in tokens:
        if not (table['x0'] <= cx <= table['x1'] and
                table['y0'] <= cy <= table['y1']):
            continue
        r = _band_index(cy, rows)
        c = _band_index(cx, cols)
        if r is None or c is None:
            continue
        cells[r][c].append((cx, text))
    grid = []
    for r in range(nR):
        row = []
        for c in range(nC):
            toks = sorted(cells[r][c])
            row.append(' '.join(t for _, t in toks).strip())
        grid.append(row)
    return grid, rows, cols


# --------------------------------------------------------------------------- #
# Header / mark-column / title resolution
# --------------------------------------------------------------------------- #
def _fix_header(label):
    """Snap an OCR'd header label to the closest HEADER_VOCAB term when close."""
    u = re.sub(r'[^A-Z ]', '', label.upper()).strip()
    if not u:
        return label.strip()
    if u in HEADER_VOCAB:
        return u
    best, bestd = None, 99
    for v in HEADER_VOCAB:
        d = _editdist(u, v)
        if d < bestd:
            best, bestd = v, d
    # accept a correction only when the edit distance is small relative to length
    if best is not None and bestd <= max(1, len(best) // 4):
        return best
    return u


def _editdist(a, b):
    m, n = len(a), len(b)
    if abs(m - n) > 4:
        return 99
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def _find_header_row(grid):
    """Return the index of the row most likely to be the column header:
    the row with the most cells matching HEADER_VOCAB (and at least 2)."""
    best, best_score = None, 0
    for r, row in enumerate(grid[:6]):  # header is near the top
        score = 0
        for cell in row:
            if not cell:
                continue
            fixed = _fix_header(cell)
            if fixed in HEADER_VOCAB:
                score += 1
        if score > best_score:
            best, best_score = r, score
    return best if best_score >= 2 else None


def _infer_class(tag, service, mounting, duty):
    """Best-effort YOLO class: tag prefix first, then schedule TYPE/DUTY/service.
    Single-letter air-device marks (A/B/C) carry no prefix signal, so fall back
    to the DUTY/TYPE column (SA=supply, RA=return, EXH=exhaust)."""
    cls = _ocr_infer_class(tag)
    if cls and cls != 'UNKNOWN':
        return cls
    try:
        from tag_inference import _infer_yolo_class_from_service
        txt = ' '.join(x for x in (service, duty, mounting) if x)
        c = _infer_yolo_class_from_service(txt, mounting or '')
        if c:
            return c
    except Exception:
        pass
    d = (duty or '').upper()
    if 'EXH' in d:
        return 'AD-GRD'
    if d.startswith('RA') or 'RETURN' in d:
        return 'AD-T-BAR RETURN'
    if d.startswith('SA') or 'SUPPLY' in d:
        return 'AD-T-BAR SUPPLY'
    return cls or 'UNKNOWN'


def _find_mark_col(header_cells):
    """Index of the MARK/TAG column, or 0 as a fallback."""
    for c, cell in enumerate(header_cells):
        u = _fix_header(cell)
        if u in ("MARK", "TAG", "SYMBOL", "DESIGNATION", "UNIT"):
            return c
    return 0


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_tables_via_ocr(pdf_path, pages=None, dpi=200, time_budget=None,
                           debug=False):
    """Reconstruct schedule tables from a vector/CAD sheet via OCR.

    pages: 0-indexed page list to scan (None = auto-detect schedule pages).
    Returns (variables, debug_info). `variables` are TagVariable dicts with
    populated `properties`, ready for tag inference / Excel."""
    import time as _time
    try:
        import fitz
        np, cv2 = _lazy_cv()
    except Exception as e:
        return [], {'error': f'imports failed: {e}'}

    _t0 = _time.time()
    doc = fitz.open(pdf_path)
    reader = None
    variables = []
    dbg = {'pages': [], 'tables': 0}
    try:
        page_idxs = pages if pages is not None else _auto_pages(doc)
        for pi in page_idxs:
            if time_budget and (_time.time() - _t0) > time_budget:
                dbg['stopped_at'] = pi
                break
            pix = doc[pi].get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if reader is None:
                from tag_matcher import get_ocr_reader
                reader = get_ocr_reader()
            ocr = reader.readtext(img, detail=1)
            tokens = []
            for box, txt, conf in ocr:
                if conf < 0.15:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                h = max(ys) - min(ys)
                tokens.append((sum(xs) / 4.0, sum(ys) / 4.0, h, str(txt).strip()))
            tables = _detect_tables(gray, tokens)
            if not tables:
                continue
            page_vars = 0
            for t in tables:
                vs = _table_to_variables(t, tokens, pi + 1, img=img, reader=reader)
                variables.extend(vs)
                page_vars += len(vs)
            dbg['tables'] += len(tables)
            dbg['pages'].append({'page': pi + 1, 'tables': len(tables),
                                 'variables': page_vars})
    finally:
        doc.close()
    # de-dup identical (tag, page) keeping the richest property dict
    variables = _dedup_variables(variables)
    return variables, dbg


def _auto_pages(doc):
    """Pick pages that look like vector schedule sheets: little/no text layer
    but schedule keywords present once rendered is expensive, so use a cheap
    proxy — pages whose text is sparse OR that contain a schedule keyword."""
    import fitz  # noqa
    out = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text")
        up = txt.upper()
        sparse = len(txt) < 400
        has_kw = any(k in up for k in _SCHED_PAGE_KW)
        # vector schedule sheet: sparse text, OR keyword present with few words
        if sparse or (has_kw and len(txt.split()) < 1200):
            out.append(i)
    return out


def _table_to_variables(table, tokens, page_no, img=None, reader=None):
    grid, rows, cols = _build_cell_grid(table, tokens)
    if not grid:
        return []
    hr = _find_header_row(grid)
    if hr is None:
        return []
    header = [_fix_header(c) for c in grid[hr]]
    title = table.get('title', '') or 'OCR SCHEDULE'
    if any(bad in title.upper() for bad in NON_HVAC_SCHEDULE_KEYWORDS):
        return []
    mark_c = _find_mark_col(grid[hr])
    # EasyOCR routinely drops isolated single-letter marks (A/B/C/D). If the mark
    # column came back mostly empty below the header, re-OCR that column strip
    # tightly (upscale + allowlist) and drop the recovered marks into their rows.
    if img is not None and reader is not None and mark_c < len(cols) - 1:
        filled = sum(1 for r in range(hr + 1, len(grid))
                     if mark_c < len(grid[r]) and grid[r][mark_c].strip())
        data_rows = len(grid) - hr - 1
        if data_rows > 0 and filled < 0.5 * data_rows:
            _recover_mark_column(img, reader, grid, rows, cols, mark_c, hr)
    out = []
    for r in range(hr + 1, len(grid)):
        cells = grid[r]
        if mark_c >= len(cells):
            continue
        mark_raw = cells[mark_c]
        tags = expand_tag_cell(mark_raw)
        if not tags:
            continue
        props = {}
        for c, val in enumerate(cells):
            if c == mark_c or not val:
                continue
            label = header[c] if c < len(header) and header[c] else f"COL_{c}"
            is_num = any(k in label for k in _NUMERIC_COLS)
            props[label] = _clean_value(val, numeric=is_num)
        svc = props.get('SERVICE', '') or props.get('TYPE', '')
        mnt = props.get('MOUNTING', '')
        duty = props.get('DUTY', '')
        for tag in tags:
            out.append({
                'tag': tag,
                'schedule_name': title or 'OCR SCHEDULE',
                'page': page_no,
                'properties': dict(props),
                'inferred_yolo_class': _infer_class(tag, svc, mnt, duty),
                'source_row_index': r,
                '_ocr': True,
                '_service': svc,
                '_mounting': mnt,
            })
    return out


def _recover_mark_column(img, reader, grid, rows, cols, mark_c, hr):
    """Re-OCR the mark column strip with upscale + an alnum allowlist to recover
    single-letter marks EasyOCR dropped on the full-page pass. Mutates `grid`
    in place, writing recovered marks into the mark cell of each data row."""
    np, cv2 = _lazy_cv()
    cx0, cx1 = cols[mark_c], cols[mark_c + 1]
    cy0, cy1 = rows[hr + 1] if hr + 1 < len(rows) else rows[hr], rows[-1]
    pad = 4
    x0 = max(0, cx0 - pad); x1 = min(img.shape[1], cx1 + pad)
    y0 = max(0, cy0 - pad); y1 = min(img.shape[0], cy1 + pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return
    crop = img[y0:y1, x0:x1]
    try:
        up = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        res = reader.readtext(
            bw, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            text_threshold=0.3, low_text=0.3, min_size=4)
    except Exception:
        return
    for box, txt, conf in res:
        if conf < 0.2 or not str(txt).strip():
            continue
        ys = [p[1] for p in box]
        cy_full = y0 + (sum(ys) / len(ys)) / 3.0   # undo the 3x upscale
        # which data row band does this y fall into?
        r = _band_index(cy_full, rows)
        if r is None or r <= hr or r >= len(grid):
            continue
        if mark_c < len(grid[r]) and not grid[r][mark_c].strip():
            grid[r][mark_c] = str(txt).strip()


def _title_above(table, tokens, band=320):
    """Schedule title = the best title-like text line in the band above the
    table. Prefers lines matching a schedule-title keyword and set in larger
    type; falls back to the nearest line, else 'OCR SCHEDULE'."""
    cand = [(cx, cy, h, txt) for (cx, cy, h, txt) in tokens
            if table['x0'] - 40 <= cx <= table['x1'] + 40
            and table['y0'] - band <= cy < table['y0']]
    if not cand:
        return 'OCR SCHEDULE'
    lines = _cluster_lines(cand)
    titled = [ln for ln in lines if _TITLE_RE.search(ln['text'])]
    if titled:
        # closest title line to the table top, breaking ties by larger height
        titled.sort(key=lambda ln: (table['y0'] - ln['cy'], -ln['h']))
        return titled[0]['text'].strip()
    lines.sort(key=lambda ln: table['y0'] - ln['cy'])
    return (lines[0]['text'].strip() if lines else 'OCR SCHEDULE') or 'OCR SCHEDULE'


def _dedup_variables(variables):
    best = {}
    for v in variables:
        key = (v['tag'], v['page'])
        cur = best.get(key)
        if cur is None or len(v['properties']) > len(cur['properties']):
            best[key] = v
    return list(best.values())


if __name__ == '__main__':
    import sys
    import json
    pdf = sys.argv[1]
    pages = [int(p) - 1 for p in sys.argv[2:]] or None
    vs, dbg = extract_tables_via_ocr(pdf, pages=pages, debug=True)
    print(json.dumps(dbg, indent=2))
    print(f"\n{len(vs)} variables:")
    for v in vs:
        print(f"  {v['tag']:10s} [{v['inferred_yolo_class']}] "
              f"{v['schedule_name'][:40]:40s} props={len(v['properties'])}")
        for k, val in list(v['properties'].items())[:6]:
            print(f"      {k}: {val}")
