"""
Diffuser / Grille / Register plan-label extractor.

Reads plan pages (M101-style) via pdfplumber text layer to extract
per-instance GRD data: mark, neck size, and CFM. Does not use YOLO or OCR.

When the text layer is unavailable (CID-encoded fonts or image-only pages),
an optional OCR fallback is invoked using EasyOCR.  The fallback is
graceful — if fitz or easyocr is not installed the page is silently skipped
with a warning, matching the original behaviour.
"""
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber


# ── Spatial search constants (PDF points; 1 pt = 1/72 inch) ──────────────────
# Tune these if CFM misses or false-matches for a specific drawing style.

CFM_WINDOW = 40   # points below label bottom to search for a CFM candidate
HSLOP      = 30   # horizontal tolerance: |label_cx − cfm_cx| must be ≤ this


# ── Neck-size normalization ───────────────────────────────────────────────────

_ROUND_RE = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*"\s*$')
# Rectangular neck: W/H  or  WxH  or  WXH  (CAD drawings use all three separators)
_RECT_RE  = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*[/xX]\s*(\d+(?:\.\d+)?)\s*$')


def normalize_neck_size(raw):
    """
    Canonicalize a neck-size string from a plan label.

    Round formats  → 'round:{diameter}'
      '12"'  → 'round:12'
      ' 8" ' → 'round:8'

    Rectangular formats  → 'rect:{width}x{height}'
      '10/10'    → 'rect:10x10'
      '22/10'    → 'rect:22x10'
      '22X10'    → 'rect:22x10'   (capital X separator)
      '22x10'    → 'rect:22x10'   (lowercase x separator)
      ' 14 / 12' → 'rect:14x12'

    Returns None for anything that does not match either pattern.
    """
    if not raw:
        return None

    s = str(raw).strip()
    if not s:
        return None

    m = _ROUND_RE.match(s)
    if m:
        d = m.group(1).rstrip('0').rstrip('.') if '.' in m.group(1) else m.group(1)
        return f'round:{d}'

    m = _RECT_RE.match(s)
    if m:
        def _fmt(n):
            return n.rstrip('0').rstrip('.') if '.' in n else n
        w, h = _fmt(m.group(1)), _fmt(m.group(2))
        return f'rect:{w}x{h}'

    return None


# ── GRD label regex ───────────────────────────────────────────────────────────
# Matches the first line of a plan-label like  S1-10"  or  E1-10/10  or TG1-22X10
#
# Capturing groups:
#   1 — mark      e.g. 'S1', 'E2', 'TG1', 'S10', 'TG10'
#   2 — neck_raw  e.g. '10"', '10/10', '22X10'
#
# Does NOT match bare equipment tags (RTU-1, EF-1) because their suffix is a
# plain integer with no " and no size separator. Does NOT match duct-size
# labels like "14/10 SA" because those have trailing text after the pair.
# Allows 1–2 letters (covers TG, EA, SA, ...) and 1–3 digit suffixes (S10, TG10).
# 3-letter prefixes like FCU/RTU are intentionally excluded to avoid matching
# equipment tags that appear on the same plan pages.

# ── GRD mark patterns ────────────────────────────────────────────────────────
#
# Three mark families are encountered across all project styles:
#
#   Family 1 — digit suffix (Busy Bees / hotel style)
#       S1, E2, TG1, SG-1, LD-1  →  [A-Z]{1,2}-?\d{1,3}
#
#   Family 2 — no digit (Haldeman / newer projects: CD, RG, SG, A, B, C)
#       CD, RG, SG, TG, A, B, C   →  [A-Z]{1,3}  (letters only)
#
#   Family 3 — compound (with additional segment)
#       LD-1-SLOT, SG-1-6/6 etc.  handled by suffix stripping
#
# Equipment tags (RTU, FCU, EF, …) are filtered at the CALLING site via
# _EQUIPMENT_PREFIXES; they are NOT excluded by the regexes themselves.

GRD_LABEL_RE = re.compile(
    r'^('
    r'[A-Z]{1,2}-?\d{1,3}'              # Family 1: S1, SG-1, TG1 (digit required)
    r'|[A-Z]{1,3}(?:-\d{1,3})?'         # Family 2+: CD, SG, A, or CD-1, SG-1
    r')'
    r'-'                                 # separator (neck size follows)
    r'(\d+(?:\.\d+)?"'                  # round neck: N" or N.N"
    r'|\d+(?:\.\d+)?[/xX]\d+(?:\.\d+)?' # rect neck: W/H, WxH, WXH
    r')$'
)
# Due to Python regex backtracking:
#   CD-10"   → tries Family-1 (fails, no digit), tries Family-2: CD + (nothing) + sep - + 10" ✓
#   SG-1-6/6 → Family-1: SG-1 + sep - + 6/6  ✓  (or Family-2 SG + (-1)? + sep -)
#   S1-10"   → Family-1: S + no-hyphen + 1 = S1, sep -, neck 10" ✓
#   A-6/6    → Family-2: A + (nothing) + sep - + 6/6 ✓

# Bare mark — tag only, neck is a separate nearby token.
#
# Extended to Family 2 (letter-only marks):
#   CD, RG, SG, TG  →  [A-Z]{1,3}
#   A, B, C, S, R   →  single uppercase letter
#   LD-1-SLOT etc.  →  optional compound word suffix stripped
_BARE_MARK_RE = re.compile(
    # Bare-mark extraction requires at LEAST one digit in the mark to avoid
    # false positives from single-letter or 2-letter abbreviations that appear
    # throughout architectural drawing text (CD, A, B, C etc. are too generic).
    # Combined-label format (GRD_LABEL_RE) already handles no-digit marks like
    # CD-10" or RG-8x8 as a single token — safer for high-precision extraction.
    r'^('
    r'[A-Z]{1,2}\d{1,3}'              # S1, E2, TG1 (letters + digits, no hyphen)
    r'|[A-Z]{1,3}-\d{1,3}'            # SG-1, LD-1 (letters + REQUIRED hyphen + digits)
    r')'
    r'(?:-[A-Z]{2,}(?:\s+[A-Z]{2,})*)?$',  # optional compound word suffix ≥2-letters
)

# Standalone size token next to a bare mark: 6/6  6X6  12"  24/12
# Does NOT include /CFM variants — those are handled by _extract_size_and_cfm().
_STANDALONE_SIZE_RE = re.compile(
    r'^(\d+(?:\.\d+)?"'                       # round: N"
    r'|\d+(?:\.\d+)?[/xX]\d+(?:\.\d+)?'      # rect: W/H WxH WXH
    r')$'
)

# Size+CFM tokens common on hotel/commercial drawings:  8x8/170  12x4/200  6/100
# Capturing group 1 = size string,  group 2 = CFM integer
_RECT_SIZE_CFM_RE = re.compile(
    r'^(\d+(?:\.\d+)?[xX]\d+(?:\.\d+)?)/(\d+)$'   # WxH/CFM or WXH/CFM
)
# Threshold: if the second number in W/H is > this, treat it as a CFM value
# (round neck + CFM) rather than a rectangular dimension (WxH).
# Rationale: realistic HVAC neck heights are ≤ 30"; CFM values for small grilles
# start around 50.  The 30" gap means no ambiguity in practice.
_ROUND_CFM_SPLIT_THRESHOLD = 30

# Inline-inch patterns like 6"x6" or 5"x5" (quotes inside dimension strings)
_INLINE_INCH_RE = re.compile(r'"')

# Equipment prefixes that should NOT be treated as GRD marks in bare-mark mode.
# These are the most common HVAC equipment tags whose symbols also appear on plan pages.
_EQUIPMENT_PREFIXES = frozenset({
    'FCU', 'RTU', 'AHU', 'CU', 'AC', 'HP', 'EF', 'SF', 'CF', 'RF',
    'EUH', 'UH', 'EH', 'BH', 'VAV', 'VRF', 'ERV', 'MD', 'MVD', 'FD',
    'FSD', 'BD', 'PTAC', 'ER', 'AIC', 'SR',
    # Additional non-GRD mark prefixes encountered in practice:
    'DN',   # door/diffuser number (drawing note identifier, e.g. DN17)
    'U',    # unit number / duct union label
    'SP',   # sprinkler head (fire protection)
    'FP',   # fire protection / floor plug
})

# Bare integer 2–5 digits — the CFM value that appears below a label.
# Lower bound 2 digits: smallest realistic CFM (≥10). Upper bound 5 digits
# keeps out room numbers / sheet numbers which tend to be longer.
_CFM_RE = re.compile(r'^\d{2,5}$')

# Fraction of word tokens that must contain '(cid:' for a page to be flagged
# as CID-encoded (unreadable with pdfplumber's text layer).
_CID_THRESHOLD = 0.30


def _page_diagnosis(words, page_num):
    """
    Inspect word list and return a short diagnostic string, or None if healthy.

    Returns one of:
      'cid'     — ≥30% of tokens are CID character references (font not embedded)
      'sparse'  — fewer than 10 words extracted (likely a raster/image page)
      None      — page looks fine
    """
    if not words:
        return 'sparse'
    n_cid = sum(1 for w in words if '(cid:' in str(w.get('text', '')))
    if n_cid / len(words) >= _CID_THRESHOLD:
        return 'cid'
    if len(words) < 10:
        return 'sparse'
    return None


# ── OCR fallback for CID/sparse pages ────────────────────────────────────────

def _ocr_page_for_grd(
    pdf_path: str,
    page_idx: int,
    dpi: int = 150,
    valid_marks: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    OCR fallback for pages whose text layer is unreadable.

    Renders the page with PyMuPDF, runs EasyOCR, and applies GRD_LABEL_RE to
    find GRD labels.  Adjacent tokens on the same horizontal line (within 8 px
    y-distance, within 40 px x-gap) are joined to reconstruct split labels
    like ``S1`` + ``-`` + ``12"`` → ``S1-12"``.  CFM values are found by the
    same spatial search used in the text-layer path.

    Returns
    -------
    (instances, sub_warnings)
        instances matches the format returned by the text-layer path.
        sub_warnings is a list of diagnostic strings (usually empty).

    Raises
    ------
    ImportError  if fitz or easyocr is not installed.
    """
    import fitz  # PyMuPDF — may raise ImportError
    from tag_matcher import get_ocr_reader  # lazy EasyOCR singleton

    scale = dpi / 72.0

    doc = fitz.open(str(pdf_path))
    page = doc[page_idx]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    doc.close()

    import numpy as np
    raw = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 4:
        img = raw.reshape(pix.height, pix.width, 4)
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = raw.reshape(pix.height, pix.width, 3)
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = raw.reshape(pix.height, pix.width, pix.n)

    reader = get_ocr_reader()
    ocr_results = reader.readtext(img)

    # Build a flat list of tokens with pixel coords, confidence, and text.
    # Coordinates are in *pixel* space; we'll convert to PDF points at the end.
    tokens: List[Dict[str, Any]] = []
    for bbox, text, conf in ocr_results:
        if conf < 0.35:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        tokens.append({
            'text':  text.strip(),
            'px0':   min(xs),
            'py0':   min(ys),
            'px1':   max(xs),
            'py1':   max(ys),
            'pcx':   (min(xs) + max(xs)) / 2,
            'pcy':   (min(ys) + max(ys)) / 2,
            'conf':  conf,
        })

    # ── Join adjacent tokens to reconstruct split labels ─────────────────────
    # Sort by (row, left-edge) so we can walk left-to-right per line.
    tokens.sort(key=lambda t: (round(t['pcy'] / 8), t['px0']))

    joined_texts: List[Dict[str, Any]] = []
    used_indices: set = set()

    for i, tok in enumerate(tokens):
        if i in used_indices:
            continue
        combined_text  = tok['text']
        combined_px0   = tok['px0']
        combined_py0   = tok['py0']
        combined_px1   = tok['px1']
        combined_py1   = tok['py1']
        used_indices.add(i)

        # Greedily absorb right-neighbours on the same line
        for j in range(i + 1, len(tokens)):
            if j in used_indices:
                continue
            nb = tokens[j]
            # Same line: y-centers within 8 px
            if abs(nb['pcy'] - tok['pcy']) > 8:
                break  # tokens are sorted by row; no point looking further
            # Close enough horizontally: left-edge of neighbour within 40 px of right-edge of accumulated span
            if nb['px0'] - combined_px1 > 40:
                continue
            # Concatenate (no space — reconstructing e.g. "S1" + "-" + "12\"")
            combined_text += nb['text']
            combined_px1   = max(combined_px1, nb['px1'])
            combined_py0   = min(combined_py0, nb['py0'])
            combined_py1   = max(combined_py1, nb['py1'])
            used_indices.add(j)

        joined_texts.append({
            'text': combined_text,
            'px0':  combined_px0,
            'py0':  combined_py0,
            'px1':  combined_px1,
            'py1':  combined_py1,
            'pcx':  (combined_px0 + combined_px1) / 2,
            'pcy':  (combined_py0 + combined_py1) / 2,
        })

    # Also include the individual tokens (some labels may not need joining)
    all_candidates = joined_texts + [
        {'text': t['text'],
         'px0':  t['px0'], 'py0': t['py0'],
         'px1':  t['px1'], 'py1': t['py1'],
         'pcx':  t['pcx'], 'pcy': t['pcy']}
        for t in tokens
    ]

    # ── Convert pixel coords back to PDF points ───────────────────────────────
    def _to_pts(px: float) -> float:
        return px / scale

    # ── Identify GRD anchors ──────────────────────────────────────────────────
    anchors: List[Dict[str, Any]] = []
    seen_positions: set = set()  # deduplicate by (rounded x0, rounded y0)

    for cand in all_candidates:
        text = cand['text'].strip()
        m = GRD_LABEL_RE.match(text)
        if not m:
            continue
        mark     = m.group(1)
        neck_raw = m.group(2)

        if valid_marks is not None and mark not in valid_marks:
            continue

        neck_canon = normalize_neck_size(neck_raw)
        if neck_canon is None:
            continue

        # Dedup: two candidates that resolve to the same pixel position are the
        # same physical label (once from a joined span, once from single token).
        pos_key = (round(cand['px0'] / 4), round(cand['py0'] / 4))
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        x0 = _to_pts(cand['px0'])
        y0 = _to_pts(cand['py0'])
        x1 = _to_pts(cand['px1'])
        y1 = _to_pts(cand['py1'])

        anchors.append({
            'mark':            mark,
            'neck_size_raw':   neck_raw,
            'neck_size_canon': neck_canon,
            'x0':  x0,
            'y0':  y0,
            'x1':  x1,
            'y1':  y1,
            'cx':  (x0 + x1) / 2,
            'cfm': None,
        })

    if not anchors:
        return [], []

    # ── CFM spatial association (same logic as text-layer path) ───────────────
    # Collect numeric CFM candidates from individual OCR tokens (not joined).
    cfm_candidates: List[Dict[str, Any]] = []
    for t in tokens:
        if _CFM_RE.match(t['text'].strip()):
            cfm_candidates.append({
                'value': int(t['text'].strip()),
                'x0':    _to_pts(t['px0']),
                'y0':    _to_pts(t['py0']),
                'cx':    _to_pts(t['pcx']),
                'top':   _to_pts(t['py0']),
            })

    consumed_cfm: set = set()
    for anchor in anchors:
        cx    = anchor['cx']
        y_top = anchor['y1']
        y_bot = y_top + CFM_WINDOW

        best_dist = float('inf')
        best_cfm  = None
        best_ci   = None

        for ci, cfm_tok in enumerate(cfm_candidates):
            if ci in consumed_cfm:
                continue
            if not (y_top <= cfm_tok['top'] <= y_bot):
                continue
            if abs(cfm_tok['cx'] - cx) > HSLOP:
                continue
            dist = (cfm_tok['cx'] - cx) ** 2 + (cfm_tok['top'] - y_top) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cfm  = cfm_tok['value']
                best_ci   = ci

        anchor['cfm'] = best_cfm
        if best_ci is not None:
            consumed_cfm.add(best_ci)

    instances: List[Dict[str, Any]] = []
    for a in anchors:
        instances.append({
            'mark':            a['mark'],
            'neck_size_raw':   a['neck_size_raw'],
            'neck_size_canon': a['neck_size_canon'],
            'cfm':             a['cfm'],
            'page':            page_idx + 1,
            'x0':              a['x0'],
            'y0':              a['y0'],
            'x1':              a['x1'],
            'y1':              a['y1'],
            'method':          'ocr_fallback',
            'confidence':      0.60,
        })

    return instances, []


# ── Global bipartite mark ↔ size matching ────────────────────────────────────
#
# Replaces the old positional-greedy loop.  On hotel-style drawings where many
# GRD symbols are packed in a narrow corridor and multiple marks see overlapping
# size tokens, positional greedy (process marks left-to-right) can "steal" the
# correct size token from a closer mark.
#
# The bipartite approach builds a cost matrix (marks × sizes) and finds the
# globally minimum-cost 1:1 assignment.  Marks that have no feasible size
# within the search window are left unmatched (optional via dummy columns).
#
# Falls back to *sorted* greedy when scipy is not installed — better than
# positional greedy because it processes marks in order of their nearest
# available size token, so marks with a unique close size grab it first.

_BIPARTITE_LARGE = 1e8   # sentinel for "infeasible pair"


def _build_cost_matrix(mark_cands, size_cands,
                        same_line_dy: float, same_line_hslop: float,
                        below_dy: float, below_hslop: float):
    """
    Build an (n_marks × n_sizes) cost matrix.

    cost[i][j] = Euclidean distance between mark i and size j if the pair
                 is within the search window, else _BIPARTITE_LARGE.

    Two spatial regimes:
      Same-baseline (|dy| ≤ same_line_dy) : generous horizontal window
      Below          (dy ≤ below_dy)       : tight horizontal window
    """
    n_m, n_s = len(mark_cands), len(size_cands)
    costs = [[_BIPARTITE_LARGE] * n_s for _ in range(n_m)]

    for i, mc in enumerate(mark_cands):
        cx    = mc['cx']
        y_mid = (mc['y0'] + mc['y1']) / 2

        for j, sc in enumerate(size_cands):
            dy = abs(sc['cy'] - y_mid)
            dx = abs(sc['cx'] - cx)

            if dy <= same_line_dy:
                feasible = (dx <= same_line_hslop)
            else:
                feasible = (dy <= below_dy and dx <= below_hslop)

            if feasible:
                costs[i][j] = (dx * dx + dy * dy) ** 0.5

    return costs


def _pair_marks_to_sizes(mark_cands, size_cands,
                          same_line_dy: float = 15.0,
                          same_line_hslop: float = 120.0,
                          below_dy: float = 50.0,
                          below_hslop: float = 40.0) -> list:
    """
    Find the globally optimal 1:1 assignment of mark candidates to size tokens.

    Returns a list of length ``len(mark_cands)`` where each entry is either:
      - an integer index into ``size_cands`` (the assigned size), or
      - ``None`` (no feasible size within the search window).

    Algorithm
    ---------
    1.  Build cost matrix (Euclidean distance, _BIPARTITE_LARGE if infeasible).
    2.  Try scipy ``linear_sum_assignment`` (Hungarian / Jonker–Volgenant).
        Dummy columns are appended so each mark can be left unmatched at cost
        _BIPARTITE_LARGE instead of being forced onto a far-away size token.
    3.  Fall back to *sorted greedy* when scipy is unavailable: process marks
        in ascending order of their minimum feasible cost so marks with unique
        close sizes claim them before competing marks arrive.
    """
    n_m = len(mark_cands)
    n_s = len(size_cands)

    if n_m == 0:
        return []
    if n_s == 0:
        return [None] * n_m

    costs = _build_cost_matrix(mark_cands, size_cands,
                                same_line_dy, same_line_hslop,
                                below_dy, below_hslop)

    # ── scipy Hungarian (optimal) ─────────────────────────────────────────────
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost_arr = np.array(costs, dtype=np.float64)  # [n_m × n_s]
        # Append n_m "dummy" size columns so marks can opt out.
        # A dummy assignment costs _BIPARTITE_LARGE, meaning "no match".
        dummy = np.full((n_m, n_m), _BIPARTITE_LARGE)
        cost_ext = np.hstack([cost_arr, dummy])        # [n_m × (n_s + n_m)]

        row_ind, col_ind = linear_sum_assignment(cost_ext)

        assignments: list = [None] * n_m
        for r, c in zip(row_ind, col_ind):
            if c < n_s and costs[r][c] < _BIPARTITE_LARGE:
                assignments[r] = c
        return assignments

    except (ImportError, Exception):
        pass

    # ── Sorted greedy fallback ────────────────────────────────────────────────
    min_costs = [min(row) for row in costs]
    mark_order = sorted(range(n_m), key=lambda i: min_costs[i])

    used: set = set()
    assignments = [None] * n_m
    for i in mark_order:
        if min_costs[i] >= _BIPARTITE_LARGE:
            continue
        best_c, best_j = _BIPARTITE_LARGE, None
        for j in range(n_s):
            if j not in used and costs[i][j] < best_c:
                best_c, best_j = costs[i][j], j
        if best_j is not None:
            assignments[i] = best_j
            used.add(best_j)

    return assignments


# ── Bare-mark fallback ────────────────────────────────────────────────────────

# Spatial constants for bare-mark pairing (PDF points)
_BARE_SIZE_WINDOW = 50   # vertical distance to look for a size token near a bare mark
_BARE_HSLOP       = 40   # horizontal tolerance


_MAX_NECK_DIM = 36   # inches — max realistic HVAC neck size. Keeps 22x22, 34x7 etc.
                     # Rejects 24x48, 48x24 (louver frame sizes, not neck openings).
                     # Previously 48 which let 24x48 pass at the exact boundary.


def _is_plausible_neck(canon: str) -> bool:
    """Return True if a normalised neck-size string represents a realistic dimension."""
    try:
        if canon.startswith('round:'):
            d = float(canon[6:])
            return 2 <= d <= _MAX_NECK_DIM
        if canon.startswith('rect:'):
            w, h = (float(x) for x in canon[5:].split('x'))
            return 2 <= w <= _MAX_NECK_DIM and 2 <= h <= _MAX_NECK_DIM
    except (ValueError, TypeError):
        pass
    return False


def _extract_size_and_cfm(raw: str):
    """
    Parse a plan-label size token and return (neck_canon, cfm_or_None).

    Handles all formats found on real HVAC drawings:
      '8x8/170'   → ('rect:8x8',  170)   rect WxH + CFM (lowercase x)
      '8X8/200'   → ('rect:8x8',  200)   rect WXH + CFM (uppercase X)
      '12x4/200'  → ('rect:12x4', 200)
      '6/100'     → ('round:6',   100)   round W" + CFM  (H > 30 → CFM)
      '8/200'     → ('round:8',   200)
      '6/6'       → ('rect:6x6',  None)  rect (both dims ≤ 30)
      '10x10'     → ('rect:10x10',None)  standard rect
      '12"'       → ('round:12',  None)  standard round
      '6"x6"'     → ('rect:6x6',  None)  inline inch-marks stripped
      '5"x5"'     → ('rect:5x5',  None)

    Returns None if the text cannot be parsed as a plausible neck size.
    """
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # ── Strip inline inch marks (e.g. '6"x6"' → '6x6') ──────────────────────
    # Keep trailing-only inch mark for round sizes; strip ones inside the string.
    if '"' in text:
        # If it ends with " and has no x/X, it's a standard round token ('12"')
        if text.endswith('"') and not re.search(r'[xX/]', text):
            # Standard round — let normalize_neck_size handle it below
            pass
        else:
            # Strip ALL inline quotes then re-evaluate
            text = text.replace('"', '')

    # ── WxH/CFM or WXH/CFM ───────────────────────────────────────────────────
    m = _RECT_SIZE_CFM_RE.match(text)
    if m:
        size_str = m.group(1).replace('X', 'x')
        cfm_val  = int(m.group(2))
        # Normalise to W/H format for normalize_neck_size
        parts    = re.split(r'[xX]', size_str)
        if len(parts) == 2:
            size_raw = f'{parts[0]}/{parts[1]}'
            canon    = normalize_neck_size(size_raw)
            if canon and _is_plausible_neck(canon):
                return (canon, cfm_val)
        return None

    # ── W/H — disambiguate rect vs round+CFM using threshold ─────────────────
    m = re.match(r'^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$', text)
    if m:
        w = float(m.group(1))
        h = float(m.group(2))
        if h > _ROUND_CFM_SPLIT_THRESHOLD:
            # High second value → round neck W" + CFM h
            canon = normalize_neck_size(f'{m.group(1)}"')
            if canon and _is_plausible_neck(canon):
                return (canon, int(h))
            return None
        else:
            # Both dims plausible → rectangular WxH
            canon = normalize_neck_size(text)
            if canon and _is_plausible_neck(canon):
                return (canon, None)
            return None

    # ── Standard token (round N" or rect WxH / WXH) ──────────────────────────
    canon = normalize_neck_size(text)
    if canon and _is_plausible_neck(canon):
        return (canon, None)

    return None


def _find_bare_mark_instances(words, valid_marks=None):
    """
    Second-pass extraction for drawings where mark and neck size are separate tokens.

    Finds pairs of (bare-mark token, nearby standalone-size token) on the same
    page.  The mark must not be a known equipment prefix.

    Returns a list of anchor dicts with the same shape as the combined-label path.
    """
    # Collect bare-mark candidates
    mark_cands = []
    for w in words:
        text = w['text'].strip()
        m = _BARE_MARK_RE.match(text)
        if not m:
            continue
        mark = m.group(1).upper()

        # Reject equipment prefixes (strip trailing digits/hyphen to get the prefix)
        prefix = re.match(r'^([A-Z]+)', mark)
        if prefix and prefix.group(1) in _EQUIPMENT_PREFIXES:
            continue

        if valid_marks is not None and mark not in valid_marks:
            continue

        mark_cands.append({
            'mark':           mark,
            'is_letter_only': False,
            'x0':   w['x0'],
            'y0':   w['top'],
            'x1':   w['x1'],
            'y1':   w['bottom'],
            'cx':   (w['x0'] + w['x1']) / 2,
        })

    # ── Letter-only marks from valid_marks whitelist ──────────────────────────
    # Marks like CD, RG, EG, SG (no digit suffix) appear in Haldeman/Gensler
    # drawings as standalone uppercase tokens but are safe to extract only when
    # a whitelist is available — without one they would produce many false
    # positives from abbreviations and other drawing text.
    if valid_marks:
        seen_positions = {(mc['mark'], round(mc['x0'], 0)) for mc in mark_cands}
        # Two-letter minimum for text-layer letter-only marks: single letters
        # (A, B, C, S, T, R) appear too frequently in plan text (column lines,
        # room labels, dimension leaders) and cause massive false positives.
        # Single-letter marks are handled by OCR rescue instead.
        lo_marks = {
            m for m in valid_marks
            if isinstance(m, str) and m.isalpha() and 2 <= len(m) <= 4
            and m not in _EQUIPMENT_PREFIXES
        }
        for w in words:
            text = w['text'].strip().upper()
            if text not in lo_marks:
                continue
            pos_key = (text, round(w['x0'], 0))
            if pos_key in seen_positions:
                continue  # already captured above
            mark_cands.append({
                'mark':           text,
                'is_letter_only': True,
                'x0':   w['x0'],
                'y0':   w['top'],
                'x1':   w['x1'],
                'y1':   w['bottom'],
                'cx':   (w['x0'] + w['x1']) / 2,
            })
            seen_positions.add(pos_key)

    if not mark_cands:
        return []

    # Collect size candidates using the full _extract_size_and_cfm parser.
    # This handles standard sizes (6X6, 10"), combined SIZE/CFM tokens (8x8/170,
    # 6/100), and inline-inch formats (6"x6").  Each candidate carries both
    # the canonical neck string AND the embedded CFM value (may be None).
    size_cands = []
    for wi, w in enumerate(words):
        text = w['text'].strip()
        result = _extract_size_and_cfm(text)
        if result is None:
            continue
        neck_canon, cfm_val = result
        size_cands.append({
            'wi':    wi,
            'text':  text,
            'x0':    w['x0'],
            'y0':    w['top'],
            'x1':    w['x1'],
            'y1':    w['bottom'],
            'cx':    (w['x0'] + w['x1']) / 2,
            'cy':    (w['top'] + w['bottom']) / 2,
            'canon': neck_canon,
            'cfm':   cfm_val,
        })

    if not size_cands:
        # Emit bare marks with null neck so the benchmark can count them
        return [{
            'mark':            mc['mark'],
            'is_letter_only':  mc.get('is_letter_only', False),
            'neck_size_raw':   '',
            'neck_size_canon': '',
            'cfm_from_size':   None,
            'x0': mc['x0'], 'y0': mc['y0'],
            'x1': mc['x1'], 'y1': mc['y1'],
        } for mc in mark_cands]

    # ── Global bipartite mark ↔ size assignment ───────────────────────────────
    # Uses _pair_marks_to_sizes() which wraps scipy Hungarian when available,
    # falling back to sorted greedy.  Both are better than the old positional
    # greedy (left-to-right order) because they respect global cost minimisation:
    # a mark with a uniquely close size token always gets it first, preventing
    # the "stolen size" problem that caused mis-pairings on dense floor plans.
    assignments = _pair_marks_to_sizes(
        mark_cands, size_cands,
        same_line_dy=15.0,
        same_line_hslop=120.0,
        below_dy=float(_BARE_SIZE_WINDOW),
        below_hslop=float(_BARE_HSLOP),
    )

    anchors = []
    for i, mc in enumerate(mark_cands):
        j = assignments[i]
        if j is None:
            anchors.append({
                'mark':            mc['mark'],
                'is_letter_only':  mc.get('is_letter_only', False),
                'neck_size_raw':   '',
                'neck_size_canon': '',
                'cfm_from_size':   None,
                'x0': mc['x0'], 'y0': mc['y0'],
                'x1': mc['x1'], 'y1': mc['y1'],
            })
        else:
            sc = size_cands[j]
            anchors.append({
                'mark':            mc['mark'],
                'is_letter_only':  mc.get('is_letter_only', False),
                'neck_size_raw':   sc['text'],
                'neck_size_canon': sc['canon'],
                'cfm_from_size':   sc['cfm'],
                'x0': mc['x0'], 'y0': mc['y0'],
                'x1': mc['x1'], 'y1': mc['y1'],
            })

    return anchors


# ── Extractor ─────────────────────────────────────────────────────────────────

def extract_diffuser_instances(pdf_path, plan_page_indices, valid_marks=None,
                               enable_ocr_supplement: bool = False):
    """
    Extract per-instance GRD data from plan-page text labels.

    Parameters
    ----------
    pdf_path : str or pathlib.Path
    plan_page_indices : iterable of int
        0-indexed page numbers to scan (only plan pages, not schedule pages).
    valid_marks : set of str or None
        If given, only emit instances whose mark appears in this set.
        Pass the set of marks returned by parse_pdf_schedules() to filter
        out any spurious regex hits that do not correspond to a schedule row.
        Pass None to accept every regex match (useful for dry-run debugging).
    enable_ocr_supplement : bool
        When True, run OCR as a supplementary pass on pages that have a healthy
        text layer (not CID/sparse) but where text-layer extraction found fewer
        labels than the word-count suggests there should be.  Adds any OCR
        instances not already covered by a text-layer instance (deduplicated by
        spatial proximity).  Default False — OCR is slow; enable explicitly when
        targeting undercount pages.

    Returns
    -------
    tuple (instances, warnings)

    instances : list of dict, one entry per plan-label found, each containing:
        mark            str         e.g. 'S1'
        neck_size_raw   str         e.g. '10"'
        neck_size_canon str         e.g. 'round:10'
        cfm             int | None  CFM from the label below; None if not found
        page            int         1-indexed page number
        x0, y0          float       top-left of label bounding box (PDF points)
        x1, y1          float       bottom-right of label bounding box

    warnings : list of str
        Human-readable diagnostics for pages that could not be parsed.
        Empty list when everything succeeded.
        Possible entries:
          'page N: CID-encoded font — text layer unreadable; OCR required'
          'page N: sparse text (N words) — may be a raster/image page'
          'page N: N words extracted but 0 GRD labels matched — check mark
           format or valid_marks filter'
    """
    instances = []
    warnings  = []
    _ocr_rescue_pages = 0   # per-project OCR rescue page counter

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx in plan_page_indices:
            if page_idx >= len(pdf.pages):
                continue

            page  = pdf.pages[page_idx]
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            )

            # ── Page-level diagnostics ────────────────────────────────────────
            diag = _page_diagnosis(words, page_idx + 1)
            if diag in ('cid', 'sparse'):
                # Attempt OCR fallback before giving up.
                try:
                    ocr_instances, _ocr_warns = _ocr_page_for_grd(
                        str(pdf_path), page_idx, dpi=150, valid_marks=valid_marks
                    )
                    warnings.append(
                        f'page {page_idx + 1}: {"CID-encoded" if diag == "cid" else "sparse"} '
                        f'— OCR fallback used, found {len(ocr_instances)} labels'
                    )
                    instances.extend(ocr_instances)
                except (ImportError, Exception) as _ocr_err:
                    # OCR not available or failed — fall back to original skip behaviour
                    if diag == 'cid':
                        warnings.append(
                            f'page {page_idx + 1}: CID-encoded font — text layer '
                            f'unreadable; OCR required for GRD extraction '
                            f'(OCR fallback failed: {_ocr_err})'
                        )
                    else:
                        warnings.append(
                            f'page {page_idx + 1}: sparse text '
                            f'({len(words) if words else 0} words) — may be raster/image '
                            f'(OCR fallback failed: {_ocr_err})'
                        )
                continue

            # ── Phase 1: label anchor detection ──────────────────────────────
            # Walk every word token; keep those that match GRD_LABEL_RE and
            # whose mark passes the valid_marks filter.

            anchors = []
            for w in words:
                text = w['text'].strip()
                m = GRD_LABEL_RE.match(text)
                if not m:
                    continue

                mark     = m.group(1)
                neck_raw = m.group(2)

                # Filter equipment prefixes — GRD_LABEL_RE now matches 3-letter
                # marks (Family 2) so we need this guard at the calling site.
                _pfx = re.match(r'^([A-Z]+)', mark)
                if _pfx and _pfx.group(1) in _EQUIPMENT_PREFIXES:
                    continue

                if valid_marks is not None and mark not in valid_marks:
                    continue

                neck_canon = normalize_neck_size(neck_raw)
                if neck_canon is None:
                    # neck_raw matched the label regex but normalize rejected it
                    # (should not happen for well-formed labels; guard anyway)
                    continue

                anchors.append({
                    'mark':            mark,
                    'neck_size_raw':   neck_raw,
                    'neck_size_canon': neck_canon,
                    'x0':  w['x0'],
                    'y0':  w['top'],
                    'x1':  w['x1'],
                    'y1':  w['bottom'],
                    'cx':  (w['x0'] + w['x1']) / 2,
                    'cfm': None,
                })

            if not anchors:
                # No combined MARK-NECK labels found.  Try the bare-mark + separate-
                # size fallback for drawings where neck size is a distinct token placed
                # next to the symbol tag (e.g. "SG-1" and "6X6" are separate words).
                bare_anchors = _find_bare_mark_instances(words, valid_marks)
                if bare_anchors:
                    warnings.append(
                        f'page {page_idx + 1}: 0 combined labels found but '
                        f'{len(bare_anchors)} bare-mark+size pairs detected — '
                        f'drawing uses separate mark/size tokens'
                    )
                    for a in bare_anchors:
                        has_size   = bool(a.get('neck_size_canon'))
                        is_lo      = a.get('is_letter_only', False)
                        if is_lo:
                            method     = 'letter_only_mark'     if has_size else 'letter_only_mark_no_sz'
                            confidence = 0.70                   if has_size else 0.55
                        else:
                            method     = 'bare_mark'            if has_size else 'bare_mark_no_size'
                            confidence = 0.75                   if has_size else 0.30
                        # Use CFM from size token (e.g. 8x8/170 → cfm=170)
                        # as fallback when no separate CFM number is below the mark.
                        cfm_val = a.get('cfm_from_size')
                        instances.append({
                            'mark':            a['mark'],
                            'neck_size_raw':   a['neck_size_raw'],
                            'neck_size_canon': a['neck_size_canon'],
                            'cfm':             cfm_val,
                            'page':            page_idx + 1,
                            'x0': a['x0'], 'y0': a['y0'],
                            'x1': a['x1'], 'y1': a['y1'],
                            'method':          method,
                            'confidence':      confidence,
                        })
                else:
                    # ── OCR rescue for healthy 0-label pages ─────────────────
                    # The page has a readable text layer but no GRD labels matched.
                    # Marks may be in CAD vector blocks (not in the PDF text stream).
                    #
                    # Guard: only attempt OCR rescue when the page looks like a plan
                    # page (few text words = sparse annotation).  Schedule / legend /
                    # detail pages contain 200+ text tokens and would waste 30-60 s of
                    # CPU with no GRD labels to find.  Also cap per-project OCR pages
                    # to avoid runaway runtimes on projects with many sparse pages.
                    _OCR_WORD_LIMIT   = 200   # pages with > this many words skip rescue
                    _OCR_PAGE_CAP     = 6     # max OCR rescue pages per project
                    if (enable_ocr_supplement
                            and len(words) <= _OCR_WORD_LIMIT
                            and _ocr_rescue_pages < _OCR_PAGE_CAP):
                        try:
                            ocr_res, _ = _ocr_page_for_grd(
                                str(pdf_path), page_idx, dpi=150,
                                valid_marks=valid_marks)
                            _ocr_rescue_pages += 1
                            if ocr_res:
                                for inst in ocr_res:
                                    inst['method']     = 'ocr_rescue'
                                    inst['confidence'] = 0.55
                                instances.extend(ocr_res)
                                warnings.append(
                                    f'page {page_idx + 1}: OCR rescue found '
                                    f'{len(ocr_res)} labels on healthy page with '
                                    f'0 text-layer matches'
                                )
                                continue
                        except (ImportError, Exception):
                            pass
                    warnings.append(
                        f'page {page_idx + 1}: {len(words)} words extracted but '
                        f'0 GRD labels matched — likely a schedule/legend/detail page, '
                        f'or plan uses a mark format outside [A-Z]{{1,2}}\\d{{1,3}}-<neck>'
                    )
                continue

            # ── Phase 2: CFM spatial association ─────────────────────────────
            # For each anchor, find the nearest bare integer in a vertical
            # window below it and within horizontal tolerance.
            # Each CFM word can only be claimed by one anchor (consumed set).

            consumed = set()   # word indices already matched

            for anchor in anchors:
                cx    = anchor['cx']
                y_top = anchor['y1']            # bottom edge of the label
                y_bot = y_top + CFM_WINDOW

                best_dist = float('inf')
                best_cfm  = None
                best_wi   = None

                for wi, w in enumerate(words):
                    if wi in consumed:
                        continue
                    if not _CFM_RE.match(w['text'].strip()):
                        continue

                    wcx = (w['x0'] + w['x1']) / 2

                    # Vertical: candidate top must fall inside the search window
                    if not (y_top <= w['top'] <= y_bot):
                        continue
                    # Horizontal: center must be within HSLOP of the label center
                    if abs(wcx - cx) > HSLOP:
                        continue

                    # Euclidean distance — prefer candidates that are close and
                    # directly below (penalise horizontal drift more than
                    # vertical closeness to capture the "immediately below" case)
                    dist = (wcx - cx) ** 2 + (w['top'] - y_top) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_cfm  = int(w['text'].strip())
                        best_wi   = wi

                anchor['cfm'] = best_cfm
                if best_wi is not None:
                    consumed.add(best_wi)

            # ── Emit ─────────────────────────────────────────────────────────
            for a in anchors:
                instances.append({
                    'mark':            a['mark'],
                    'neck_size_raw':   a['neck_size_raw'],
                    'neck_size_canon': a['neck_size_canon'],
                    'cfm':             a['cfm'],
                    'page':            page_idx + 1,
                    'x0':              a['x0'],
                    'y0':              a['y0'],
                    'x1':              a['x1'],
                    'y1':              a['y1'],
                    'method':          'combined_label',
                    'confidence':      1.0,
                })

            # ── Optional OCR supplement on combined-label pages ───────────────
            # When enable_ocr_supplement=True, run OCR and add any instances
            # whose mark is NOT already covered by a text-layer extraction within
            # 15 pts.  This catches labels embedded in CAD vector blocks that
            # pdfplumber does not extract from the text stream.
            if enable_ocr_supplement and len(anchors) > 0:
                try:
                    ocr_sup, _ = _ocr_page_for_grd(
                        str(pdf_path), page_idx, dpi=150, valid_marks=valid_marks)
                    if ocr_sup:
                        # Dedup: skip OCR instance if a text-layer instance of
                        # the same mark is already within 15 PDF pts.
                        tl_boxes = [
                            ((inst['x0']+inst['x1'])/2, (inst['y0']+inst['y1'])/2,
                             inst['mark'])
                            for inst in instances if inst['page'] == page_idx + 1
                        ]
                        added = 0
                        for oi in ocr_sup:
                            ocx = (oi['x0'] + oi['x1']) / 2
                            ocy = (oi['y0'] + oi['y1']) / 2
                            too_close = any(
                                abs(ocx - tx) < 15 and abs(ocy - ty) < 15
                                and oi['mark'] == tm
                                for tx, ty, tm in tl_boxes
                            )
                            if not too_close:
                                oi['confidence'] = 0.55
                                oi['method']     = 'ocr_supplement'
                                instances.append(oi)
                                tl_boxes.append((ocx, ocy, oi['mark']))
                                added += 1
                        if added:
                            warnings.append(
                                f'page {page_idx + 1}: OCR supplement added '
                                f'{added} label(s) not found in text layer')
                except (ImportError, Exception):
                    pass   # OCR not available — silently skip

    return instances, warnings


# ── Drawing-style detection ───────────────────────────────────────────────────

def detect_drawing_style(instances: list, warnings: list) -> str:
    """
    Classify the drawing style used in this PDF based on what extraction
    method was successful.

    Returns one of:
      'combined_label'   — all/most instances found via GRD_LABEL_RE (e.g. S1-12")
      'bare_mark'        — all/most instances from bare-mark fallback (e.g. SG-1 + 6X6)
      'mixed'            — both methods contributed substantially
      'ocr_only'         — only OCR fallback succeeded
      'no_grd_found'     — no instances extracted at all
      'cid_page'         — CID-encoded pages, OCR was required
    """
    if not instances:
        cid_warns = [w for w in warnings if 'CID' in w.upper() or 'OCR fallback' in w]
        return 'cid_page' if cid_warns else 'no_grd_found'

    methods = [inst.get('method', 'combined_label') for inst in instances]
    counts = {m: methods.count(m) for m in set(methods)}
    total = len(methods)

    ocr_n     = counts.get('ocr_fallback', 0) + counts.get('ocr_rescue', 0)
    combined  = counts.get('combined_label', 0)
    bare      = (counts.get('bare_mark', 0) + counts.get('bare_mark_no_size', 0)
                 + counts.get('letter_only_mark', 0) + counts.get('letter_only_mark_no_sz', 0))

    if ocr_n == total:
        return 'ocr_only'
    if combined >= 0.75 * total:
        return 'combined_label'
    if bare >= 0.75 * total:
        return 'bare_mark'
    if combined > 0 and bare > 0:
        return 'mixed'
    return 'combined_label'   # default — most projects


def filter_by_confidence(instances: list, min_confidence: float = 0.5) -> list:
    """
    Return only instances whose confidence meets the threshold.

    Use min_confidence=0.5 to drop bare-mark-no-size instances (confidence=0.3)
    while keeping combined_label (1.0), bare_mark (0.75), and OCR (0.6).
    """
    return [i for i in instances if i.get('confidence', 1.0) >= min_confidence]


# ── Schedule property lookup ──────────────────────────────────────────────────

def _sched_prop(props, keys):
    """
    Return the first value in props whose key contains any element of keys
    (case-insensitive substring match).  Returns '' if nothing matches.

    Mirrors takeoff_cli._prop() but lives here so diffuser_extractor has no
    dependency on takeoff_cli.
    """
    if not props:
        return ''
    keys_upper = [k.upper() for k in keys]
    for prop_key, val in props.items():
        k_norm = prop_key.upper()
        for ku in keys_upper:
            if ku in k_norm:
                return str(val).strip()
    return ''


# ── Display formatting ────────────────────────────────────────────────────────

def display_neck_size(neck_canon):
    """
    Convert a canonical neck size back to the team's takeoff display format.

      'round:12'   → '12"'
      'rect:10x10' → '10/10'

    This is the inverse of normalize_neck_size and is used when writing
    values to Excel or printing human-readable output.
    """
    if not neck_canon:
        return ''
    if neck_canon.startswith('round:'):
        return neck_canon[6:] + '"'
    if neck_canon.startswith('rect:'):
        return neck_canon[5:].replace('x', '/')
    return neck_canon


# ── BOM aggregation ───────────────────────────────────────────────────────────

def aggregate_diffuser_bom(instances, mark_details=None):
    """
    Group instances by (mark, neck_size_canon) and join with schedule data.

    Parameters
    ----------
    instances    : list of dict   from extract_diffuser_instances()
    mark_details : dict or None   {mark: {col_header: value, ...}}
                                  from parse_pdf_schedules(); pass None to
                                  skip the schedule join (BOM still produced,
                                  manufacturer/model/module_size will be '').

    Returns
    -------
    list of dict, sorted by (mark, neck_size_canon), each containing:
        mark            str   e.g. 'S1'
        neck_size_canon str   e.g. 'round:10'
        qty             int   number of instances in this group
        total_cfm       int   sum of cfm values (None-cfm instances add 0)
        cfm_missing     int   count of instances where CFM was not found
        manufacturer    str   from schedule row, '' if not available
        model           str   from schedule row
        module_size     str   from schedule row  e.g. '24"X24"'
        mounting        str   from schedule row  e.g. 'CEILING'
    """
    if mark_details is None:
        mark_details = {}

    grouped = defaultdict(lambda: {'qty': 0, 'total_cfm': 0, 'cfm_missing': 0})

    for inst in instances:
        key = (inst['mark'], inst['neck_size_canon'])
        grouped[key]['qty'] += 1
        if inst['cfm'] is not None:
            grouped[key]['total_cfm'] += inst['cfm']
        else:
            grouped[key]['cfm_missing'] += 1

    bom = []
    for (mark, neck_canon), data in sorted(grouped.items()):
        props = mark_details.get(mark, {})
        bom.append({
            'mark':            mark,
            'neck_size_canon': neck_canon,
            'qty':             data['qty'],
            'total_cfm':       data['total_cfm'],
            'cfm_missing':     data['cfm_missing'],
            'manufacturer':    _sched_prop(props, ('MANUFACTURER', 'MAKE', 'BRAND')),
            'model':           _sched_prop(props, ('MODEL',)),
            'module_size':     _sched_prop(props, ('MODULE SIZE', 'MODULE')),
            'mounting':        _sched_prop(props, ('MOUNTING', 'MOUNT')),
        })

    return bom


# ── Standalone test / debug ───────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # ── 1. normalize_neck_size unit tests ─────────────────────────────────────
    print('normalize_neck_size tests')
    print('-' * 40)
    _cases = [
        ('12"',      'round:12'),
        ('6"',       'round:6'),
        (' 8" ',     'round:8'),
        ('10"',      'round:10'),
        ('10.5"',    'round:10.5'),
        ('10/10',    'rect:10x10'),
        ('22/10',    'rect:22x10'),
        ('22X10',    'rect:22x10'),   # capital X separator
        ('22x10',    'rect:22x10'),   # lowercase x separator
        ('6/6',      'rect:6x6'),
        (' 14 / 12', 'rect:14x12'),
        ('24/18',    'rect:24x18'),
        ('10.5/8',   'rect:10.5x8'),
        ('',         None),
        (None,       None),
        ('12',       None),
        ('24"X24"',  None),           # inch marks break the pattern → rejected
        ('10x10',    'rect:10x10'),   # lowercase x is now accepted
        ('abc',      None),
        ('10/10/8',  None),
    ]
    _passed = _failed = 0
    for _raw, _expected in _cases:
        _result = normalize_neck_size(_raw)
        if _result == _expected:
            _passed += 1
        else:
            _failed += 1
            print(f'  FAIL  normalize_neck_size({_raw!r}) = {_result!r}  '
                  f'(expected {_expected!r})')
    print(f'{_passed} passed, {_failed} failed out of {len(_cases)} cases\n')

    # ── 2. display_neck_size unit tests ───────────────────────────────────────
    print('display_neck_size tests')
    print('-' * 40)
    _disp_cases = [
        ('round:12',   '12"'),
        ('round:6',    '6"'),
        ('round:10.5', '10.5"'),
        ('rect:10x10', '10/10'),
        ('rect:22x10', '22/10'),
        ('rect:6x6',   '6/6'),
        ('',           ''),
    ]
    _dp = _df = 0
    for _canon, _exp in _disp_cases:
        _res = display_neck_size(_canon)
        if _res == _exp:
            _dp += 1
        else:
            _df += 1
            print(f'  FAIL  display_neck_size({_canon!r}) = {_res!r}  (expected {_exp!r})')
    print(f'{_dp} passed, {_df} failed out of {len(_disp_cases)} cases\n')

    # ── 3. GRD_LABEL_RE sanity checks ─────────────────────────────────────────
    print('GRD_LABEL_RE sanity checks')
    print('-' * 40)
    _re_cases = [
        # existing patterns still work
        ('S1-12"',    True,  'S1',   '12"'),
        ('S2-6"',     True,  'S2',   '6"'),
        ('E1-10/10',  True,  'E1',   '10/10'),
        ('E2-6/6',    True,  'E2',   '6/6'),
        ('R1-22/10',  True,  'R1',   '22/10'),
        ('R2-12"',    True,  'R2',   '12"'),
        ('R3-24/24',  True,  'R3',   '24/24'),
        ('TG1-22/10', True,  'TG1',  '22/10'),
        # multi-digit mark suffixes (defensive — S10, TG10, etc.)
        ('S10-10"',   True,  'S10',  '10"'),
        ('TG10-22/10',True,  'TG10', '22/10'),
        ('E12-6/6',   True,  'E12',  '6/6'),
        # capital-X and lowercase-x neck separators (some CAD exports)
        ('TG1-22X10', True,  'TG1',  '22X10'),
        ('R2-22x22',  True,  'R2',   '22x22'),
        # negatives — must not match
        ('RTU-1',     False, None,   None),
        ('EF-1',      False, None,   None),
        ('450',       False, None,   None),
        ('14/10',     False, None,   None),
        ('S1',        False, None,   None),
        ('S1-',       False, None,   None),
        ('RTUA-12"',  False, None,   None),  # 4 letters — exceeds 2-letter cap
    ]
    _re_passed = _re_failed = 0
    for _text, _should, _exp_mark, _exp_neck in _re_cases:
        _m = GRD_LABEL_RE.match(_text)
        _matched = _m is not None
        _ok = _matched == _should
        if _ok and _matched:
            _ok = (_m.group(1) == _exp_mark) and (_m.group(2) == _exp_neck)
        if _ok:
            _re_passed += 1
        else:
            _re_failed += 1
            if _matched:
                print(f'  FAIL  {_text!r}: mark={_m.group(1)!r} neck={_m.group(2)!r}'
                      f'  (expected {_exp_mark!r}/{_exp_neck!r})')
            else:
                print(f'  FAIL  {_text!r}: no match (expected should_match={_should})')
    print(f'{_re_passed} passed, {_re_failed} failed out of {len(_re_cases)} cases\n')

    # ── 4. aggregate_diffuser_bom unit tests (synthetic) ──────────────────────
    print('aggregate_diffuser_bom tests (synthetic)')
    print('-' * 40)

    _synth_inst = [
        # mark   neck_raw  neck_canon       cfm   page  x0 y0 x1 y1
        {'mark': 'S1', 'neck_size_raw': '10"',   'neck_size_canon': 'round:10',  'cfm': 325, 'page': 1, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0},
        {'mark': 'S1', 'neck_size_raw': '10"',   'neck_size_canon': 'round:10',  'cfm': 325, 'page': 1, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0},
        {'mark': 'S1', 'neck_size_raw': '12"',   'neck_size_canon': 'round:12',  'cfm': 450, 'page': 1, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0},
        {'mark': 'E1', 'neck_size_raw': '10/10', 'neck_size_canon': 'rect:10x10','cfm': 240, 'page': 1, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0},
        {'mark': 'TG1','neck_size_raw': '22/10', 'neck_size_canon': 'rect:22x10','cfm': None,'page': 1, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0},
    ]
    _synth_details = {
        'S1':  {'MANUFACTURER': 'TITUS', 'MODEL': 'OMNI', 'MODULE SIZE': '24"X24"', 'MOUNTING': 'CEILING'},
        'E1':  {'MANUFACTURER': 'TITUS', 'MODEL': '50F',  'MODULE SIZE': '24"X24"', 'MOUNTING': 'CEILING'},
        'TG1': {'MANUFACTURER': 'TITUS', 'MODEL': '50F',  'MODULE SIZE': '24"X12"', 'MOUNTING': 'CEILING'},
    }
    _synth_bom = aggregate_diffuser_bom(_synth_inst, _synth_details)
    _bom_by_key = {(r['mark'], r['neck_size_canon']): r for r in _synth_bom}

    _bom_checks = [
        # (mark, neck_canon,    field,          expected)
        ('S1',  'round:10',  'qty',          2),
        ('S1',  'round:10',  'total_cfm',    650),
        ('S1',  'round:10',  'cfm_missing',  0),
        ('S1',  'round:10',  'manufacturer', 'TITUS'),
        ('S1',  'round:10',  'model',        'OMNI'),
        ('S1',  'round:10',  'module_size',  '24"X24"'),
        ('S1',  'round:12',  'qty',          1),
        ('S1',  'round:12',  'total_cfm',    450),
        ('E1',  'rect:10x10','qty',          1),
        ('E1',  'rect:10x10','total_cfm',    240),
        ('E1',  'rect:10x10','model',        '50F'),
        ('TG1', 'rect:22x10','qty',          1),
        ('TG1', 'rect:22x10','total_cfm',    0),
        ('TG1', 'rect:22x10','cfm_missing',  1),
        ('TG1', 'rect:22x10','module_size',  '24"X12"'),
    ]
    _bp = _bf = 0
    for _mark, _neck, _field, _exp in _bom_checks:
        _row = _bom_by_key.get((_mark, _neck))
        _got = _row.get(_field) if _row else None
        if _got == _exp:
            _bp += 1
        else:
            _bf += 1
            print(f'  FAIL  ({_mark}, {_neck}) [{_field}] = {_got!r}  (expected {_exp!r})')
    # Also verify sort order: E1 < R* < S1 < TG1
    _marks_order = [r['mark'] for r in _synth_bom]
    if _marks_order == sorted(_marks_order):
        _bp += 1
    else:
        _bf += 1
        print(f'  FAIL  BOM sort order wrong: {_marks_order}')
    print(f'{_bp} passed, {_bf} failed out of {len(_bom_checks) + 1} checks\n')

    # ── 5. Live extraction + BOM (requires PDF argument) ─────────────────────
    if len(sys.argv) < 2:
        print('Usage: python diffuser_extractor.py <blueprint.pdf> [page1 page2 ...]')
        print('       (page numbers are 1-indexed; omit to scan all pages)')
        sys.exit(0)

    _pdf_path = sys.argv[1]

    if len(sys.argv) > 2:
        _page_indices = [int(p) - 1 for p in sys.argv[2:]]
    else:
        with pdfplumber.open(_pdf_path) as _pdf:
            _page_indices = list(range(len(_pdf.pages)))

    # Optionally pull mark_details from schedule so the BOM gets manufacturer/model
    _mark_details = {}
    _variables    = []
    try:
        from schedule_parser import parse_pdf_schedules as _pps
        _, _, _mark_details, _, _, _variables = _pps(_pdf_path)
        _grd_marks = sorted(
            m for m in _mark_details
            if any(_mark_details[m].get(k, '') for k in ('MODULE SIZE', 'MOUNTING'))
        )
        print(f'Schedule: {len(_mark_details)} marks total, '
              f'GRD-like: {_grd_marks}')
    except Exception as _e:
        print(f'(Schedule parse skipped: {_e})')

    _valid = set(_mark_details) if _mark_details else None
    print(f'Scanning pages {[p + 1 for p in _page_indices]} of {_pdf_path}\n')

    _instances, _warns = extract_diffuser_instances(
        _pdf_path, _page_indices, valid_marks=_valid)

    if _warns:
        print('Page diagnostics:')
        for _w in _warns:
            print(f'  [warn] {_w}')
        print()

    if not _instances:
        print('No GRD labels found.')
        sys.exit(0)

    _bom = aggregate_diffuser_bom(_instances, _mark_details)

    # ── BOM table ──────────────────────────────────────────────────────────────
    _W = 60
    print('=' * _W)
    print('GRILLE / REGISTER / DIFFUSER — BILL OF MATERIALS')
    print('=' * _W)
    _hdr = f'{"MARK":<5} {"NECK SIZE":<10} {"QTY":>4} {"TOTAL CFM":>10} ' \
           f'{"MANUFACTURER":<14} {"MODEL":<8} {"MODULE":>10}'
    print(_hdr)
    print('-' * _W)
    for _row in _bom:
        _neck_disp  = display_neck_size(_row['neck_size_canon'])
        _miss_flag  = f"  *{_row['cfm_missing']} no-CFM" if _row['cfm_missing'] else ''
        _cfm_str    = str(_row['total_cfm']) if _row['total_cfm'] else '—'
        print(f"{_row['mark']:<5} {_neck_disp:<10} {_row['qty']:>4} "
              f"{_cfm_str:>10} {_row['manufacturer']:<14} "
              f"{_row['model']:<8} {_row['module_size']:>10}{_miss_flag}")
    print('=' * _W)

    # ── CFM cross-check by prefix ──────────────────────────────────────────────
    _prefix_cfm = defaultdict(int)
    _prefix_qty = defaultdict(int)
    for _row in _bom:
        # prefix = leading letters of mark (S, E, R, TG, ...)
        import re as _re
        _pfx = _re.match(r'^([A-Z]+)', _row['mark']).group(1)
        _prefix_cfm[_pfx] += _row['total_cfm']
        _prefix_qty[_pfx] += _row['qty']

    print('\nCFM summary by prefix:')
    for _pfx in sorted(_prefix_cfm):
        print(f'  {_pfx:<4}  qty={_prefix_qty[_pfx]:>3}  cfm={_prefix_cfm[_pfx]:>6}')

    # Supply vs RTU schedule cross-check
    _rtu_supply_total = 0
    for _var in _variables:
        if not _var['tag'].startswith('RTU'):
            continue
        _cfm_v = ''
        for _k, _v in _var['properties'].items():
            if 'SUPPLY' in _k.upper() and 'CFM' in _k.upper():
                _cfm_v = _v
                break
        if not _cfm_v:
            for _k, _v in _var['properties'].items():
                if _k.upper() in ('CFM', 'SUPPLY CFM'):
                    _cfm_v = _v
                    break
        if _cfm_v:
            try:
                _rtu_supply_total += int(str(_cfm_v).replace(',', '').strip())
            except ValueError:
                pass

    _supply_cfm = _prefix_cfm.get('S', 0)
    if _rtu_supply_total:
        _delta = _supply_cfm - _rtu_supply_total
        _pct   = abs(_delta) / _rtu_supply_total * 100 if _rtu_supply_total else 0
        print(f'\n  Supply diffusers CFM : {_supply_cfm:>6}')
        print(f'  RTU schedule total   : {_rtu_supply_total:>6}')
        print(f'  Delta                : {_delta:>+6}  ({_pct:.1f}%)')
        if _pct < 2.0:
            print('  -> Within 2% rounding tolerance  ✓')
        else:
            print('  -> Delta exceeds 2% — investigate missing or extra diffusers')

    print(f'\nTotal instances extracted: {len(_instances)}')
    _no_cfm = sum(1 for i in _instances if i['cfm'] is None)
    if _no_cfm:
        print(f'CFM not found: {_no_cfm}  '
              f'(CFM_WINDOW={CFM_WINDOW}, HSLOP={HSLOP})')
