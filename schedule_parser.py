"""
Schedule & Legend Parser v2

Extracts equipment schedules from HVAC blueprint PDFs.
v2 improvements:
- Reject pure numbers and noise (row indices, empty cells)
- Normalize multi-line tags ("VAV\n2" -> "VAV-2")
- Extract tags from description text, not just TAG column
- Split compound strings like "A, B, C" into individual tags
- Merge fragmented cells where tag spans two columns

Usage:
    from schedule_parser import parse_pdf_schedules
    schedules, marks, mark_details = parse_pdf_schedules("blueprint.pdf")
"""
import os
import re
import time
from collections import defaultdict
import pdfplumber

from tag_inference import _infer_yolo_class_from_service, _infer_class_from_tag, TAG_PREFIX_CLASS


# Opt-in debug logging. Many pdfplumber/OCR failures on messy PDFs are caught
# and skipped silently (correct — one bad table shouldn't sink a run), but that
# makes field failures invisible. Set TAKEOFF_DEBUG=1 to surface them.
_DEBUG = bool(os.environ.get('TAKEOFF_DEBUG'))


def _dbg(msg):
    if _DEBUG:
        print(f"  [debug] {msg}")


# Must contain at least 1 letter (rejects pure numbers).
# (The single source of truth for "is this a valid tag" is normalize_tag() —
# an older module-level TAG_REGEX was unused and removed to avoid drift.)
HAS_LETTER = re.compile(r'[A-Za-z]')
# Tags seen in description text — scan for things like "LD-1-PLENUM"
TAG_IN_DESC = re.compile(r'\b([A-Z]{1,4}-?\d+[A-Z]?(?:-[A-Z]+)?)\b')

# Known header keywords that indicate a TAG column
TAG_COL_KEYWORDS = ("MARK", "TAG", "DESIGNATION", "UNIT TAG", "EQUIPMENT TAG",
                     "SYMBOL", "ID", "NO.", "UNIT", "REF", "ITEM")

# Keywords that indicate a page likely contains equipment schedules
SCHEDULE_KEYWORDS = [
    "SCHEDULE", "EQUIPMENT", "DEVICE LIST",
    "AIR CURTAIN", "CONDENSING UNIT", "SPLIT SYSTEM",
    "FAN COIL", "DIFFUSER", "REGISTER", "GRILLE",
    "UNIT SCHEDULE", "TERMINAL", "ROOFTOP",
    "MECHANICAL SCHEDULE", "HVAC SCHEDULE",
]

# Schedule types that are NOT HVAC equipment — skip these tables entirely.
# Projects include plumbing, lighting, electrical schedules in the same
# drawing set. We only take HVAC off.
NON_HVAC_SCHEDULE_KEYWORDS = [
    "PLUMBING", "LIGHTING", "ELECTRICAL", "FIRE PROTECTION",
    "DATA", "TELECOM", "SECURITY", "FIRE ALARM",
    "WATER HEATER", "PLUMBING FIXTURE", "LIGHTING FIXTURE",
    "FIXTURE SCHEDULE", "PANEL SCHEDULE", "CIRCUIT",
    "SPRINKLER", "DOOR SCHEDULE", "WINDOW SCHEDULE",
    "FINISH SCHEDULE", "ROOM SCHEDULE",
]

# Property keywords that identify a schedule table even without "SCHEDULE" keyword
PROPERTY_KEYWORDS = {"MANUFACTURER", "MODEL", "AIRFLOW", "CFM", "CAPACITY",
                      "INLET", "OUTLET", "WEIGHT", "VOLTAGE", "WATTS",
                      "BTU", "TONNAGE", "MOUNTING", "SIZE"}

# Junk that should never count as a tag
JUNK_TAGS = {
    'TYPE', 'TYP', 'MARK', 'TAG', 'NO.', 'REF.', 'NOTE', 'NOTES',
    'N.T.S.', 'NTS', 'SEE', 'ALL', 'EACH', 'TOTAL', 'COL_1',
    'VAV', 'FCU', 'AHU', 'RTU',  # prefix-only (need number)
    'RR',  # junk fragment seen in Aritzia schedule
}

# Refrigerant designations commonly appear in schedules as a dedicated row
# (e.g., R-410A, R-454B, R-32) — they pass the tag regex but are NOT tags.
REFRIGERANT_PATTERN = re.compile(r'^R-?\d{2,4}[A-Z]?$', re.IGNORECASE)

# Cell values that match the tag regex shape but are clearly not equipment
# tags — typically labels from adjacent columns like "NOTES: 1" that got
# merged into the MARK cell during table extraction.
BANNED_TAG_PREFIXES = {
    'NOTES', 'NOTE', 'ROUTING', 'ROUTE', 'SCHEDULE', 'SCHED',
    'DETAIL', 'PAGE', 'SHEET', 'DWG', 'REF', 'SEE',
    'REV', 'DATE', 'ITEM',
}

# No equipment type filtering — extract ALL tags from all schedules.
# The team takes off everything: GRD, fans, heaters, dampers, etc.
EXCLUDE_PREFIXES = set()


def normalize_tag(raw):
    """Clean up tag string, return None if invalid."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Strip equipment-status prefixes used on drawings: "(E)" = existing,
    # "(R)" = relocated, "(N)" = new. They are not part of the tag.
    s = re.sub(r'^\(\s*[ERN]\s*\)\s*', '', s, flags=re.IGNORECASE).strip()
    if not s:
        return None

    # Handle multi-line cells like "VAV\n2" or "24\nVAV\n27" or "VAV\nN1"
    # Split by whitespace/newlines and find the most tag-like fragment(s)
    fragments = re.split(r'\s+', s)
    fragments = [f.strip('.-,;:|/') for f in fragments if f.strip('.-,;:|/')]

    if not fragments:
        return None

    # Valid tag shape: letters followed by optional digits, optionally hyphenated
    # Accept: A, A1, A-1, AHU-1, FCU-10, VAV-N1, LD-1, GR-2, SC1, SA1, etc.
    # Reject: 1-2-3, 24-VAV-27, pure numbers
    valid_tag_re = re.compile(r'^[A-Za-z][A-Za-z0-9\-]{0,15}$')

    # Case 1: single fragment — must look like a tag
    if len(fragments) == 1:
        s = fragments[0]
    # Case 2: multiple fragments — join meaningfully
    else:
        # If first fragment is pure digits and following starts with letters,
        # use only the letter+number parts (skip the leading digit)
        # e.g., ["24", "VAV", "27"] -> "VAV-27"
        # e.g., ["VAV", "N"] -> "VAV-N"
        # e.g., ["VAV", "2"] -> "VAV-2"
        letter_start_idx = None
        for i, f in enumerate(fragments):
            if re.match(r'^[A-Za-z]', f):
                letter_start_idx = i
                break
        if letter_start_idx is None:
            return None
        # Take letter fragment + next numeric/alpha fragment if present
        useful = fragments[letter_start_idx:]
        # Reject if we have more than 2 fragments (avoid "VAV-EXISTING-23" style)
        if len(useful) > 2:
            useful = useful[:2]
        s = '-'.join(useful)

    # Final cleanup
    s = s.strip('.-,;:|/ ')
    if not s:
        return None

    # Must start with a letter
    if not re.match(r'^[A-Za-z]', s):
        return None

    # Must match valid tag shape
    if not valid_tag_re.match(s.replace('-', '')):
        # allow a single hyphen separator — revalidate with hyphen
        parts = s.split('-')
        if len(parts) > 3:
            return None
        if not all(re.match(r'^[A-Za-z0-9]+$', p) for p in parts):
            return None

    if len(s) < 1 or len(s) > 20:
        return None

    if s.upper() in JUNK_TAGS:
        return None

    # Refrigerant codes (R-410A, R-454B, R-32) — never equipment tags
    if REFRIGERANT_PATTERN.match(s):
        return None

    # Drawing sheet numbers (M101, E202, P301) — single letter + 3 digits.
    # Equipment tags never use this pattern; sheet indexes always do.
    if re.match(r'^[A-Za-z]\d{3}$', s):
        return None

    # Model numbers are typically long, no-hyphen, alphanumeric strings
    # mixing 3+ letters with digits (e.g., RKF12AXVJU, FTKF12AXVJU, DAX0904A).
    # Real equipment tags are almost always hyphenated or short.
    if '-' not in s and len(s) > 8:
        # Count letter/digit alternations as a heuristic for model-number shape
        letters = sum(1 for c in s if c.isalpha())
        if letters >= 4:
            return None

    # Reject cell labels that leaked in from adjacent columns
    prefix_m = re.match(r'^([A-Z]+)', s.upper())
    if prefix_m and prefix_m.group(1) in BANNED_TAG_PREFIXES:
        return None

    # Must have digits OR be a short letter sequence (single-letter tags like A, B, C, D)
    has_digit = bool(re.search(r'\d', s))
    if not has_digit and len(s) > 2:
        # Allow KNOWN_PREFIX-LETTER_SUFFIX (e.g., "VAV-N", "FCU-A") where the
        # prefix is a recognized HVAC equipment prefix. This keeps legit tags
        # like VAV-N while rejecting abbreviations like "U-C" or "USE-NT".
        m = re.match(r'^([A-Za-z]{1,5})-[A-Za-z]{1,3}$', s)
        if not m or m.group(1).upper() not in TAG_PREFIX_CLASS:
            return None

    return s.upper()


def split_compound_cell(cell_value):
    """
    Split a cell like 'A, B, C' or 'D / E' into individual tags.
    Also handles the 'PREFIX-N1, N2, N3' shorthand common on drawings where
    one tag cell covers several equipment numbers sharing a row:
      'AC-1,2'   -> ['AC-1', 'AC-2']
      'CU-1,2,3' -> ['CU-1', 'CU-2', 'CU-3']
    """
    if not cell_value:
        return []
    s = str(cell_value).strip()
    parts = [p.strip() for p in re.split(r'[,;/&]', s) if p.strip()]
    if not parts:
        return []

    # Shorthand expansion: first part is PREFIX-N, following parts are bare numbers.
    # Strip any "(E)" / "(R)" / "(N)" status marker before the prefix.
    first_clean = re.sub(r'^\(\s*[ERN]\s*\)\s*', '', parts[0].upper().strip(),
                          flags=re.IGNORECASE).strip()
    m = re.match(r'^([A-Z]+)-(\d+)$', first_clean)
    if m and len(parts) > 1 and all(re.fullmatch(r'\d+', p.strip()) for p in parts[1:]):
        prefix = m.group(1)
        return [parts[0]] + [f"{prefix}-{p.strip()}" for p in parts[1:]]

    return parts


def expand_range(cell_value):
    """
    Expand range notation like 'CU-1 thru CU-6' to ['CU-1','CU-2',...,'CU-6'].
    Also handles 'CU-1 through CU-6' and 'CU-1 to CU-6'.
    Returns None if cell is not a range.
    """
    if not cell_value:
        return None
    s = ' '.join(str(cell_value).upper().split())
    # Match "PREFIX-N thru/through/to PREFIX-M" (prefix may repeat or be omitted)
    m = re.match(
        r'^([A-Z]{1,4})-?(\d+)\s*(?:THRU|THROUGH|TO|\-|\u2013|\u2014)\s*(?:([A-Z]{1,4})-?)?(\d+)$',
        s
    )
    if not m:
        return None
    prefix1, start, prefix2, end = m.groups()
    if prefix2 and prefix1 != prefix2:
        return None
    try:
        start_n, end_n = int(start), int(end)
    except ValueError:
        return None
    if end_n < start_n or end_n - start_n > 100:
        return None
    return [f"{prefix1}-{i}" for i in range(start_n, end_n + 1)]


def split_multi_number_cell(raw):
    """
    Detect cells where one letter prefix is paired with multiple numbers,
    e.g., '24\\nVAV\\n27' or '24 VAV 27' meaning BOTH VAV-24 and VAV-27
    share the same schedule row. Returns a list of tags, or None.
    """
    if not raw:
        return None
    s = str(raw).strip()
    fragments = [f for f in re.split(r'\s+', s) if f]
    fragments = [f.strip('.-,;:|/') for f in fragments if f.strip('.-,;:|/')]
    if len(fragments) < 3:
        return None
    digits = [f for f in fragments if re.fullmatch(r'\d{1,3}', f)]
    letters = [f for f in fragments if re.fullmatch(r'[A-Za-z]{1,5}', f)]
    if len(digits) >= 2 and len(letters) == 1:
        prefix = letters[0].upper()
        return [f"{prefix}-{d}" for d in digits]
    return None


def expand_tag_cell(raw):
    """
    Return a list of normalized tags from a cell. Handles:
      - single tag:   'A-1'             -> ['A-1']
      - compound:     'A, B, C'         -> ['A','B','C']
      - range:        'CU-1 thru CU-6'  -> ['CU-1','CU-2',...,'CU-6']
      - multi-number: '24\\nVAV\\n27'    -> ['VAV-24','VAV-27']
      - multi-line:   'CU-1\\nCU-2'      -> ['CU-1','CU-2'] (each line a tag)
    """
    if not raw:
        return []

    # Multi-line cell where each line is a complete tag on its own.
    # Seen on equipment-connection schedules where one row covers multiple
    # units: MARK cell is "CU-1\nCU-2" or "EF-2\nOACU-1".
    s = str(raw).strip()
    if '\n' in s:
        lines = [ln.strip() for ln in s.split('\n') if ln.strip()]
        # Each line must look like a complete tag: letters + (digit or hyphen)
        if len(lines) > 1 and all(
            re.match(r'^[A-Za-z]{1,5}-?\d', ln) or
            re.match(r'^[A-Za-z]{2,5}-[A-Za-z0-9]+$', ln)
            for ln in lines
        ):
            tags = [t for t in (normalize_tag(ln) for ln in lines) if t]
            if tags:
                return tags

    multi = split_multi_number_cell(raw)
    if multi:
        return [t for t in (normalize_tag(m) for m in multi) if t]
    ranged = expand_range(raw)
    if ranged:
        return [t for t in (normalize_tag(r) for r in ranged) if t]
    parts = split_compound_cell(raw)
    return [t for t in (normalize_tag(p) for p in parts) if t]


def _prop_lookup(props, keywords):
    """
    Find first value in props dict whose key contains any of the keywords.
    Case/whitespace-insensitive. Returns '' if nothing matches.
    """
    if not props:
        return ''
    kw_upper = [k.upper() for k in keywords]
    for k, v in props.items():
        k_norm = ' '.join(str(k).upper().split())
        for kw in kw_upper:
            if kw in k_norm:
                return v
    return ''


def extract_schedules_and_marks(pdf_path, time_budget=None):
    """
    Extract schedule tables and equipment marks from a PDF.
    v2: heavy validation, noise filtering.

    time_budget : float or None
        Wall-clock seconds allowed for the per-page scan. pdfplumber's
        extract_tables() costs ~4s/page on large-format plan sheets, so on big
        documents the unbounded scan can run for minutes. When set, the loop
        stops once the budget is exceeded and prints how many pages were skipped
        (schedules are usually front-loaded, so the early pages matter most).
        Default None = unlimited (unchanged behaviour for small files).

    Returns (schedule_tables, marks_list, mark_details, variables) where
    variables is a list of TagVariable dicts — one per (tag, source_row) — with
    the full row properties preserved and an inferred YOLO class attached.
    """
    import time
    schedule_tables = []
    marks_set = set()
    mark_details = {}
    variables = []

    _t0 = time.time()
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            if time_budget is not None and (time.time() - _t0) > time_budget:
                print(f"  [schedule] time budget ({time_budget:.0f}s) reached after "
                      f"page {page_index}/{n_pages} — skipping remaining "
                      f"{n_pages - page_index} page(s) for table scan")
                break
            try:
                text_upper = (page.extract_text() or "").upper()
                page_has_schedule = any(kw in text_upper for kw in SCHEDULE_KEYWORDS)
                tables = page.extract_tables()
            except Exception as e:
                _dbg(f"table scan failed on page {page_index + 1}: {e}")
                continue

            for t_index, table in enumerate(tables):
                if not table:
                    continue
                if all(all((cell is None or str(cell).strip() == "") for cell in row) for row in table):
                    continue

                # --- Detect table orientation ---
                # HORIZONTAL table = properties listed DOWN column 0
                # (MARK, MANUFACTURER, MODEL, CFM etc.) and tags as column headers.
                # Require MARK/TAG in column 0 AND 2+ property keywords in column 0.
                is_horizontal = False
                if len(table) >= 3 and len(table[0]) >= 3:
                    col0_cells = [str(row[0]).strip().upper() if row[0] else '' for row in table]
                    col0_has_mark = any(
                        c in ('MARK', 'TAG', 'DESIGNATION', 'SYMBOL')
                        for c in col0_cells
                    )
                    col0_prop_count = sum(
                        1 for c in col0_cells
                        if any(kw in c for kw in ('MODEL', 'MANUFACTURER', 'CFM',
                               'AIRFLOW', 'SIZE', 'CAPACITY', 'SERVICE', 'TYPE',
                               'VOLTAGE', 'WEIGHT'))
                        and len(c) < 30
                    )
                    # Both conditions: TAG label + properties in column 0
                    if col0_has_mark and col0_prop_count >= 2:
                        is_horizontal = True

                # Transpose horizontal tables
                if is_horizontal:
                    max_cols = max(len(row) for row in table)
                    transposed = []
                    for col_idx in range(max_cols):
                        new_row = []
                        for row in table:
                            new_row.append(row[col_idx] if col_idx < len(row) else None)
                        transposed.append(new_row)
                    table = transposed

                # --- Find header row containing a TAG keyword ---
                header_row_idx = None
                for r_idx, row in enumerate(table):
                    for cell in row:
                        if cell is None:
                            continue
                        cell_upper = str(cell).strip().upper()
                        if cell_upper in TAG_COL_KEYWORDS:
                            header_row_idx = r_idx
                            break
                        if ("MARK" in cell_upper or "TAG" in cell_upper) and len(cell_upper) < 20:
                            header_row_idx = r_idx
                            break
                    if header_row_idx is not None:
                        break

                # --- Property-based detection fallback ---
                # If no explicit MARK/TAG column found, look for a header row
                # identified by 3+ property keywords (TYPE, MODEL, SIZE, CFM, ...).
                # Require equipment-specific keywords (not just "NOTES") so we
                # don't mistake a drawing index for a schedule.
                if header_row_idx is None:
                    header_detection_kws = ('TYPE', 'MODEL', 'SIZE', 'CFM', 'MANUFACTURER',
                                             'DESCRIPTION', 'CAPACITY', 'NECK', 'SERVICE',
                                             'MAKE', 'MOUNTING', 'REMARK')
                    for r_idx, row in enumerate(table):
                        row_strs = [str(c).strip().upper() for c in row if c]
                        if not row_strs:
                            continue
                        prop_count = sum(
                            1 for s in row_strs
                            if any(kw in s for kw in header_detection_kws) and len(s) < 30
                        )
                        if prop_count >= 3:
                            header_row_idx = r_idx
                            page_has_schedule = True
                            break

                if header_row_idx is None and not page_has_schedule:
                    continue

                # Schedule name (from rows above header). Prefer rows that
                # contain "SCHEDULE" keyword — that's almost always the actual
                # title. Fall back to the first short non-prose row.
                schedule_name = ""
                fallback_name = ""
                if header_row_idx is not None:
                    for up in range(header_row_idx - 1, -1, -1):
                        cells = [str(c).strip() for c in table[up] if c not in [None, ""]]
                        if not cells:
                            continue
                        candidate = " ".join(cells[:3])
                        # Prefer a title row containing SCHEDULE
                        if 'SCHEDULE' in candidate.upper() and len(candidate) < 100:
                            schedule_name = candidate
                            break
                        # Otherwise remember first short non-prose row as fallback
                        if not fallback_name:
                            if len(candidate) <= 80 and candidate.count(' ') <= 10:
                                fallback_name = candidate
                    if not schedule_name:
                        schedule_name = fallback_name

                # Build header + data rows
                header = None
                data_rows = []

                if header_row_idx is not None:
                    header = [str(c).strip() if c is not None else "" for c in table[header_row_idx]]
                    for r in range(header_row_idx + 1, len(table)):
                        row = table[r]
                        if any(cell not in [None, ""] and str(cell).strip() != "" for cell in row):
                            data_rows.append([str(c).strip() if c is not None else "" for c in row])
                else:
                    for row in table:
                        if header is None and any(cell not in [None, ""] for cell in row):
                            header = [str(c).strip() if c is not None else "" for c in row]
                        elif any(cell not in [None, ""] for cell in row):
                            data_rows.append([str(c).strip() if c is not None else "" for c in row])

                if not header or not data_rows:
                    continue

                header_upper = [h.upper() for h in header]
                looks_like_schedule = page_has_schedule or ("SCHEDULE" in " ".join(header_upper))
                if not looks_like_schedule:
                    continue

                # Skip non-HVAC schedules (plumbing, lighting, electrical, etc.)
                # Check ONLY the schedule name — not column headers, because
                # legit HVAC schedules like RTU often have "ELECTRICAL" as a
                # column header under the electrical specs sub-section.
                name_text = (schedule_name or "").upper()
                if any(kw in name_text for kw in NON_HVAC_SCHEDULE_KEYWORDS):
                    # But don't reject if the name also contains an HVAC keyword
                    # (some combined "MECHANICAL AND PLUMBING" schedules exist)
                    hvac_keywords = ('HVAC', 'AIR HANDL', 'CONDENSING', 'FAN COIL',
                                       'DIFFUSER', 'GRILLE', 'DAMPER', 'VAV',
                                       'EXHAUST FAN', 'TERMINAL', 'ROOFTOP',
                                       'AIR CURTAIN', 'LOUVER',
                                       'UNIT HEATER', 'ELECTRIC HEATER',
                                       'CABINET HEATER', 'DUCT HEATER')
                    if not any(kw in name_text for kw in hvac_keywords):
                        continue

                # Identify columns
                mark_col_indices = []
                for i, h in enumerate(header_upper):
                    for kw in TAG_COL_KEYWORDS:
                        if kw in h and len(h) < 25:
                            mark_col_indices.append(i)
                            break

                # If no explicit MARK/TAG column found, check if column 0 holds
                # tag-shaped values (short alphanumeric like "A1", "B1", "C-1").
                # Schedules like AIR DEVICE SCHEDULE put tags in a "TYPE" column.
                if not mark_col_indices and data_rows:
                    first_val = ''
                    for row in data_rows:
                        if row and row[0] and str(row[0]).strip():
                            first_val = str(row[0]).strip()
                            break
                    # Accept a column-0 mark when it's tag-shaped (letters+digits,
                    # e.g. "A1", "C-1") OR a bare 1–2 letter mark that normalize_tag
                    # accepts ("A", "B"). Air-device schedules routinely use bare
                    # single-letter marks with no MARK/TAG header — the digit-
                    # requiring regex alone silently dropped those whole tables.
                    if first_val and (
                        re.match(r'^[A-Za-z]{1,4}-?\d{1,3}[A-Za-z]?$', first_val)
                        or (len(first_val) <= 2 and normalize_tag(first_val))
                    ):
                        mark_col_indices.append(0)

                # Description columns (for extracting embedded tags, and for details).
                # Exclude the mark column so we don't treat tag values as description.
                desc_col_indices = [
                    i for i, h in enumerate(header_upper)
                    if any(kw in h for kw in ["DESCRIPTION", "TYPE", "MODEL", "SIZE", "CAPACITY", "SERVICE", "REMARK"])
                    and i not in mark_col_indices
                ]

                # Convert rows to dicts
                table_dict_rows = []
                for row in data_rows:
                    row_dict = {}
                    for col_idx, col_name in enumerate(header):
                        key = col_name if col_name else f"COL_{col_idx+1}"
                        value = row[col_idx] if col_idx < len(row) else ""
                        row_dict[key] = value
                    table_dict_rows.append(row_dict)

                schedule_tables.append({
                    "page": page_index + 1,
                    "schedule_name": schedule_name,
                    "header": header,
                    "rows": table_dict_rows,
                })

                # Extract tags and build TagVariables
                for row_idx, row in enumerate(data_rows):
                    # Primary: tags from MARK/TAG columns (supports compound + range)
                    row_tags = []
                    for mark_col in mark_col_indices:
                        if mark_col >= len(row):
                            continue
                        for tag in expand_tag_cell(row[mark_col]):
                            if tag and tag not in row_tags:
                                row_tags.append(tag)

                    # Secondary: scan description columns for embedded tag patterns
                    # e.g., REMARKS = "LD-1-PLENUM installed" -> extract LD-1.
                    # Only scan description/type/remarks columns that are unlikely
                    # to contain model numbers or refrigerant codes.
                    # Require the tag prefix to be a known HVAC prefix — otherwise
                    # MODEL values like "MP-2-72", "DAX0904A", and refrigerants
                    # like "R-454B" get mistakenly extracted as tags.
                    desc_text_parts = []
                    for desc_col in desc_col_indices:
                        if desc_col >= len(row) or not row[desc_col]:
                            continue
                        header_name = header_upper[desc_col] if desc_col < len(header_upper) else ''
                        # Skip columns that explicitly hold model/part numbers
                        if any(skip in header_name for skip in
                               ('MODEL', 'PART', 'REFRIGERANT', 'SERIAL', 'MANUFACTURER')):
                            continue
                        desc_text_parts.append(str(row[desc_col]))
                    desc_text = " ".join(desc_text_parts)

                    for m in TAG_IN_DESC.finditer(desc_text.upper()):
                        tag = normalize_tag(m.group(1))
                        if not tag or tag in row_tags:
                            continue
                        # Prefix must be a known HVAC equipment prefix
                        prefix_match = re.match(r'^([A-Z]+)', tag)
                        if not prefix_match:
                            continue
                        prefix = prefix_match.group(1)
                        if prefix in TAG_PREFIX_CLASS or prefix in ('A', 'B', 'C', 'D'):
                            row_tags.append(tag)

                    if not row_tags:
                        continue

                    # Build full properties dict — EVERY column except the tag column(s).
                    # Normalize both keys (column headers) and values: collapse whitespace
                    # so multi-line headers like "MANUFACTURER\n& MODEL" become
                    # "MANUFACTURER & MODEL" — stable and readable downstream.
                    full_props = {}
                    for col_idx, col_name in enumerate(header):
                        if col_idx in mark_col_indices:
                            continue
                        raw_key = col_name if col_name else f"COL_{col_idx+1}"
                        key = ' '.join(str(raw_key).split()) or f"COL_{col_idx+1}"
                        value = row[col_idx] if col_idx < len(row) else ""
                        if value and str(value).strip():
                            full_props[key] = ' '.join(str(value).split())

                    # Infer YOLO class from the row (once per row, shared across tags)
                    service_text = _prop_lookup(full_props, ('SERVICE', 'TYPE', 'DESCRIPTION'))
                    mounting_text = _prop_lookup(full_props, ('MOUNTING', 'MOUNT'))
                    inferred_class = _infer_yolo_class_from_service(service_text, mounting_text)
                    if not inferred_class and row_tags:
                        inferred_class = _infer_class_from_tag(row_tags[0])

                    # One variable per tag, with the full row as properties
                    for tag in row_tags:
                        marks_set.add(tag)
                        # Legacy mark_details — first occurrence wins, now with ALL columns
                        if tag not in mark_details:
                            mark_details[tag] = dict(full_props)
                        variables.append({
                            'tag': tag,
                            'schedule_name': schedule_name,
                            'page': page_index + 1,
                            'properties': dict(full_props),
                            'inferred_yolo_class': inferred_class,
                            'source_row_index': row_idx,
                        })

    return schedule_tables, sorted(list(marks_set)), mark_details, variables


def extract_legend_info(pdf_path, time_budget=None):
    """Extract legend/abbreviation items.

    time_budget : float or None — wall-clock seconds for the scan (None =
    unlimited). Legend pages can be the same heavy schedule sheets, so this
    scan is bounded too on large plans.
    """
    import time
    legend_items = {}

    _t0 = time.time()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if time_budget is not None and (time.time() - _t0) > time_budget:
                break
            try:
                text = page.extract_text() or ""
            except Exception as e:
                _dbg(f"legend scan: text extract failed: {e}")
                continue
            text_upper = text.upper()

            if not any(kw in text_upper for kw in ["LEGEND", "ABBREVIATION", "SYMBOLS", "KEY NOTES"]):
                continue

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for table in tables:
                if not table:
                    continue
                for row in table:
                    clean = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if len(clean) == 2:
                        abbr, desc = clean[0], clean[1]
                        if len(abbr) < 15 and len(desc) > 3:
                            legend_items[abbr] = desc

            for line in text.split('\n'):
                m = re.match(r'^([A-Z]{1,6}(?:-\d+)?)\s*[-=:]\s*(.+)$', line.strip())
                if m:
                    abbr, desc = m.group(1), m.group(2).strip()
                    if len(desc) > 3:
                        legend_items[abbr] = desc

    return legend_items


def get_mark_type(mark):
    """FCU-10 -> FCU, L-1 -> L"""
    m = re.match(r'^([A-Z]+)', mark)
    return m.group(1) if m else mark.split("-")[0] if "-" in mark else mark


# ─── OCR fallback for schedules with no text layer ──────────────────────────
# Some plans (CAD/vector exports — common on retail-chain drawings) draw their
# schedule tables as line-art: every glyph is a vector path, not a text object.
# pdfplumber's text/table extraction then returns ZERO tags even though the
# schedule is plainly visible. parse_pdf_schedules detects that case and falls
# back to rendering the suspect pages and OCR-ing them, recovering at least the
# valid tag list so tag inference has something to match floor-plan bubbles
# against. (Full per-tag property reconstruction from OCR is a later step.)

OCR_FALLBACK_MIN_TAGS = 3  # below this, assume the text layer failed → try OCR
OCR_FALLBACK_MIN_BUDGET = 15.0  # don't start the OCR fallback with less than this
                                # many seconds left — better to keep detection's slice

_SCHED_KW = ('SCHEDULE', 'CFM', 'NECK', 'MODEL', 'GRILLE', 'DIFFUSER',
             'REGISTER', 'MANUFACTURER', 'MOUNTING', 'MBH', 'TONS', 'TYPE')


# Air-device marks are routinely used BARE (no number) on retail plans — e.g.
# AutoZone tags all 62 supply diffusers "CD" and all 16 return grilles "RG",
# differentiated only by neck size. So a digit cannot be required. These are the
# multi-letter diffuser/grille prefixes we accept unnumbered.
_BARE_AIR_DEVICE = {'CD', 'RG', 'SD', 'SA', 'RA', 'EA', 'GR', 'SB', 'EG',
                    'TG', 'RD', 'CG', 'LG', 'SG'}

# Supply vs return is encoded in the air-device prefix; mounting is NOT (it lives
# in the schedule table we can't reconstruct from OCR yet). We default mounting
# to lay-in / T-bar — the dominant retail ceiling type, and the same assumption
# the text-path already makes in _infer_yolo_class_from_service ("CEILING →
# likely T-BAR"). This lets OCR-recovered diffuser tags carry the SPECIFIC
# product the team's takeoff uses (AD-T-BAR SUPPLY/RETURN) instead of the generic
# AD-GRD, which would never match. Refine to mounting-aware once table OCR lands.
_OCR_SUPPLY_PREFIXES = {'CD', 'SD', 'SA', 'SG'}
_OCR_RETURN_PREFIXES = {'RG', 'RA', 'RD', 'CG', 'TG'}


def _ocr_infer_class(tag):
    """Class inference for the OCR path. Uses the diffuser/grille prefix's
    supply/return semantics to emit the specific AD-T-BAR product; falls back to
    the generic prefix map for everything else (RTU, EF, ...)."""
    m = re.match(r'^([A-Z]+)', tag)
    pfx = m.group(1) if m else ''
    if pfx in _OCR_SUPPLY_PREFIXES:
        return 'AD-T-BAR SUPPLY'
    if pfx in _OCR_RETURN_PREFIXES:
        return 'AD-T-BAR RETURN'
    return _infer_class_from_tag(tag)


def _ocr_tag_from_token(token):
    """Strict tag extractor for the noisy OCR path: normalize, then require the
    prefix to be a KNOWN HVAC equipment prefix. Accept the token only if it
    carries a digit (RTU-2, EF-1, TEF-3) OR is exactly a bare air-device prefix
    (CD, RG, ...). This keeps real marks — including the unnumbered diffuser/
    grille tags that dominate retail takeoffs — while dropping OCR garbage and
    word fragments (E, E-THE, RTU-DO, model bits like HSPF2). Precision over
    recall on purpose: a bad tag in valid_marks mis-tags real detections."""
    t = normalize_tag(token)
    if not t:
        return None
    m = re.match(r'^([A-Z]+)', t)
    if not m or m.group(1) not in TAG_PREFIX_CLASS:
        return None
    if re.search(r'\d', t):
        return t            # numbered mark — keep
    if t in _BARE_AIR_DEVICE:
        return t            # legitimately-bare diffuser/grille mark
    return None             # bare non-air-device → almost certainly OCR noise


def extract_marks_via_ocr(pdf_path, time_budget=None, dpi=150, max_pages=8, max_side=2600):
    """Render pages and OCR them to recover schedule tags when the text layer is
    empty (vector/CAD schedules). Returns (marks, variables). Variables carry
    tag + inferred class but EMPTY properties — the tag list alone unblocks tag
    inference. Bounded by time_budget (wall-clock seconds) and max_pages.

    max_side caps the rendered page's longest dimension (px) before OCR. A 36"
    sheet at 150 DPI is ~5400 px wide; EasyOCR on CPU then takes minutes for a
    single page — and the budget is only checked between pages, so one runaway
    page can blow the entire run budget (starving detection to zero). Downscaling
    to ~2600 px bounds per-page cost to seconds while keeping schedule-table text
    legible. Set max_side=0 to disable."""
    import time as _time
    try:
        import fitz
        import numpy as np
        import cv2
    except Exception:
        return [], []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return [], []

    # try/finally so the document handle is released even if rendering or OCR
    # raises mid-loop (previously the doc was never closed at all — a per-run
    # leak; small at one doc/run, but wrong).
    reader = None
    _t0 = _time.time()
    marks_set = set()
    variables = []
    try:
        n = min(len(doc), max_pages) if max_pages else len(doc)
        for i in range(n):
            if time_budget is not None and (_time.time() - _t0) > time_budget:
                print(f"[schedule-ocr] time budget reached after page {i}/{n}")
                break
            try:
                pix = doc[i].get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = img[:, :, :3]
                # Bound per-page OCR cost: downscale large-format sheets so a single
                # EasyOCR call can't run past the budget (checked only between pages).
                if max_side:
                    longest = max(img.shape[0], img.shape[1])
                    if longest > max_side:
                        scale = max_side / longest
                        img = cv2.resize(img, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
            except Exception as e:
                _dbg(f"schedule-ocr: render failed on page {i}: {e}")
                continue
            if reader is None:
                from tag_matcher import get_ocr_reader
                reader = get_ocr_reader()
            try:
                toks = [str(t).upper() for t in reader.readtext(img, detail=0)]
            except Exception as e:
                _dbg(f"schedule-ocr: readtext failed on page {i}: {e}")
                continue
            # Only harvest from schedule-looking pages — keeps floor-plan tag bubbles
            # and detail-callout text from polluting the valid-mark list.
            if sum(1 for t in toks if any(k in t for k in _SCHED_KW)) < 2:
                continue
            for t in toks:
                tag = _ocr_tag_from_token(t)
                if not tag or tag in marks_set:
                    continue
                marks_set.add(tag)
                variables.append({
                    'tag': tag,
                    'schedule_name': 'OCR SCHEDULE (no text layer)',
                    'page': i + 1,
                    'properties': {},
                    'inferred_yolo_class': _ocr_infer_class(tag),
                    'source_row_index': -1,
                })
    finally:
        doc.close()
    return sorted(marks_set), variables


def parse_pdf_schedules(pdf_path, exclude_prefixes=None, time_budget=None):
    """
    Main entry point.

    exclude_prefixes: set of equipment type prefixes to exclude (default: none).
    Pass exclude_prefixes=set() to get ALL tags.
    time_budget: wall-clock seconds for the page scan (None = unlimited). See
    extract_schedules_and_marks — guards against minute-long scans on big plans.

    Returns (schedules, marks, mark_details, legend, summary, variables).
    `variables` is a list of TagVariable dicts — one per (tag, source_row) pair —
    each with the full schedule row preserved in its 'properties' field and an
    inferred YOLO class. This is the recommended structure for downstream work.
    """
    if exclude_prefixes is None:
        exclude_prefixes = EXCLUDE_PREFIXES

    # Split the scan budget: schedules (the costly extract_tables loop) get the
    # bulk, legend gets a smaller slice so both stay bounded on large plans.
    # All three sub-scans (tables, legend, OCR fallback) share the ONE schedule
    # budget against a single start time — see the OCR fallback below.
    _sched_t0 = time.time()
    sched_tb = (time_budget * 0.8) if time_budget else None
    legend_tb = (time_budget * 0.2) if time_budget else None
    schedules, marks, mark_details, variables = extract_schedules_and_marks(
        pdf_path, time_budget=sched_tb)
    legend = extract_legend_info(pdf_path, time_budget=legend_tb)

    # OCR fallback: if the text layer yielded essentially no tags, the schedule
    # is almost certainly vector line-art (CAD export) with no readable text.
    # Render the pages and OCR them to recover the valid tag list so tag
    # inference isn't starved. Skipped entirely when the text path worked, so
    # text-layer projects pay zero cost and can't regress.
    used_ocr_fallback = False
    if len(variables) < OCR_FALLBACK_MIN_TAGS:
        # Give OCR only the time LEFT in the schedule budget — NOT a fresh slice.
        # Previously this passed sched_tb again, so the schedule phase could spend
        # 0.8x (tables) + 0.2x (legend) + 0.8x (OCR) and a single uninterruptible
        # EasyOCR page would blow the whole run deadline, starving detection to 0.
        if time_budget is not None:
            ocr_tb = time_budget - (time.time() - _sched_t0)
        else:
            ocr_tb = None
        if ocr_tb is None or ocr_tb >= OCR_FALLBACK_MIN_BUDGET:
            ocr_marks, ocr_vars = extract_marks_via_ocr(pdf_path, time_budget=ocr_tb)
        else:
            print(f"  [schedule-ocr] skipped — only {ocr_tb:.0f}s left in schedule "
                  f"budget (< {OCR_FALLBACK_MIN_BUDGET:.0f}s), preserving detection time")
            ocr_vars = []
        if ocr_vars:
            used_ocr_fallback = True
            existing = set(marks)
            for v in ocr_vars:
                if v['tag'] in existing:
                    continue
                existing.add(v['tag'])
                variables.append(v)
                mark_details.setdefault(v['tag'], {})
            marks = sorted(existing)

    # Filter out excluded equipment types (e.g., VAV boxes)
    if exclude_prefixes:
        filtered_marks = [m for m in marks if get_mark_type(m) not in exclude_prefixes]
        filtered_details = {m: d for m, d in mark_details.items() if get_mark_type(m) not in exclude_prefixes}
        variables = [v for v in variables if get_mark_type(v['tag']) not in exclude_prefixes]
        excluded_count = len(marks) - len(filtered_marks)
        marks = filtered_marks
        mark_details = filtered_details
    else:
        excluded_count = 0

    type_counts = defaultdict(int)
    for mark in marks:
        type_counts[get_mark_type(mark)] += 1

    summary = {
        'total_marks': len(marks),
        'total_schedules': len(schedules),
        'legend_items': len(legend),
        'excluded_count': excluded_count,
        'types': dict(type_counts),
        'marks': marks,
        'total_variables': len(variables),
        'used_ocr_fallback': used_ocr_fallback,
    }

    return schedules, marks, mark_details, legend, summary, variables


def dump_variables(variables, file=None):
    """
    Write a human-readable dump of all extracted variables, grouped by schedule.
    Pass file=None to print to stdout.
    """
    import sys
    out = file or sys.stdout

    if not variables:
        out.write("\nNo variables extracted.\n")
        return

    grouped = defaultdict(list)
    for v in variables:
        key = (v.get('page', 0), v.get('schedule_name') or '(unnamed schedule)')
        grouped[key].append(v)

    out.write("\n" + "=" * 70 + "\n")
    out.write("SCHEDULE EXTRACTION VERIFICATION\n")
    out.write("=" * 70 + "\n")

    for (page, name), vlist in sorted(grouped.items()):
        header_line = f"\nSchedule: {name} (page {page}) — {len(vlist)} variable(s)"
        out.write(header_line + "\n")
        out.write("-" * min(len(header_line), 70) + "\n")

        for v in vlist:
            tag = v.get('tag', '?')
            out.write(f"  {tag}\n")
            props = v.get('properties') or {}
            max_key = max((len(str(k)) for k in props.keys()), default=0)
            for k, val in props.items():
                clean_val = ' '.join(str(val).split())
                if not clean_val:
                    continue
                out.write(f"    {str(k).ljust(max_key)}   {clean_val}\n")
            ic = v.get('inferred_yolo_class')
            if ic:
                out.write(f"    -> Inferred class: {ic}\n")
            out.write("\n")

    out.write(f"TOTAL: {len(variables)} variables across {len(grouped)} schedule(s)\n")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python schedule_parser.py path/to/blueprint.pdf [--verify]")
        _sys.exit(1)

    pdf = _sys.argv[1]
    verify = '--verify' in _sys.argv[2:]
    print(f"Parsing: {pdf}")

    schedules, marks, details, legend, summary, variables = parse_pdf_schedules(pdf)

    print(f"\nSchedules found: {summary['total_schedules']}")
    print(f"Equipment marks: {summary['total_marks']}")
    print(f"Variables:       {summary['total_variables']}")
    print(f"Legend items:    {summary['legend_items']}")

    if summary['types']:
        print(f"\nBy type:")
        for t, n in sorted(summary['types'].items(), key=lambda x: -x[1]):
            print(f"  {t}: {n}")

    if schedules:
        print(f"\nSchedule tables:")
        for s in schedules:
            print(f"  Page {s['page']}: {s['schedule_name'][:50] or '(unnamed)'} - {len(s['rows'])} rows")

    if verify:
        dump_variables(variables)
