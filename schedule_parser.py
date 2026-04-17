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
from collections import defaultdict
import pdfplumber


# Tag validation regex — valid tag patterns we accept
TAG_REGEX = re.compile(r'^[A-Z]{1,5}(?:[-\s]?[A-Z0-9]{1,4})*$')
# Must contain at least 1 letter (rejects pure numbers)
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

# Property keywords that identify a schedule table even without "SCHEDULE" keyword
PROPERTY_KEYWORDS = {"MANUFACTURER", "MODEL", "AIRFLOW", "CFM", "CAPACITY",
                      "INLET", "OUTLET", "WEIGHT", "VOLTAGE", "WATTS",
                      "BTU", "TONNAGE", "MOUNTING", "SIZE"}

# Junk that should never count as a tag
JUNK_TAGS = {
    'TYPE', 'TYP', 'MARK', 'TAG', 'NO.', 'REF.', 'NOTE', 'NOTES',
    'N.T.S.', 'NTS', 'SEE', 'ALL', 'EACH', 'TOTAL', 'COL_1',
    'VAV', 'FCU', 'AHU', 'RTU',  # prefix-only (need number)
}

# Equipment types the team does NOT take off — filter from final output.
# Current focus: GRD (Grilles, Registers, Diffusers) only.
# Mechanical equipment (fans, AHUs, VAVs, etc.) is handled separately.
EXCLUDE_PREFIXES = {
    # Terminal / AHU equipment
    'VAV',   # Variable Air Volume boxes
    'AHU',   # Air Handling Units
    'FCU',   # Fan Coil Units
    'RTU',   # Rooftop Units
    'MUA',   # Make-Up Air units
    'ERV',   # Energy Recovery Ventilators
    'PTAC',  # Packaged Terminal AC
    'HRV',   # Heat Recovery Ventilators

    # Fans
    'EF',    # Exhaust Fans
    'SF',    # Supply Fans
    'KEF',   # Kitchen Exhaust Fans
    'CUH',   # Cabinet Unit Heaters

    # Heating equipment
    'EH',    # Electric Heaters
    'UH',    # Unit Heaters
    'DH',    # Duct Heaters
    'BH',    # Baseboard Heaters

    # Cooling / refrigeration
    'CU',    # Condensing Units
    'HP',    # Heat Pumps
    'SS',    # Split Systems
    'VRF',   # VRF systems
    'CR',    # Chillers / Refrigeration

    # Humidifiers / misc equipment
    'HUM',   # Humidifiers
    'BFC',   # Boxes with fin coil
    'BM',    # Mixing boxes
    'RC',    # Roof curbs
    'GV',    # Gravity ventilators
}

# What we DO want to keep — common GRD tag prefixes (for reference, not enforced):
# A, B, C, D (Flex single-letter)
# SC, SA, SB, RA, RB, EA, EB (St Elizabeth)
# GR, GA, GE (grilles)
# D (diffusers)
# LD, LR, LS (linear diffusers/returns/slots)


def normalize_tag(raw):
    """Clean up tag string, return None if invalid."""
    if not raw:
        return None
    s = str(raw).strip()
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

    # Must have digits OR be a short letter sequence (single-letter tags like A, B, C, D)
    has_digit = bool(re.search(r'\d', s))
    if not has_digit and len(s) > 2:
        return None

    return s.upper()


def split_compound_cell(cell_value):
    """Split a cell like 'A, B, C' or 'D / E' into individual tags."""
    if not cell_value:
        return []
    s = str(cell_value).strip()
    # Split on common separators
    parts = re.split(r'[,;/&]', s)
    return [p.strip() for p in parts if p.strip()]


def extract_schedules_and_marks(pdf_path):
    """
    Extract schedule tables and equipment marks from a PDF.
    v2: heavy validation, noise filtering.
    """
    schedule_tables = []
    marks_set = set()
    mark_details = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                text_upper = (page.extract_text() or "").upper()
                page_has_schedule = any(kw in text_upper for kw in SCHEDULE_KEYWORDS)
                tables = page.extract_tables()
            except Exception:
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
                # If no MARK/TAG found, check if table has enough property keywords
                # to be a schedule (then use first row as tags)
                if header_row_idx is None and not page_has_schedule:
                    # Check if any row has 3+ property keywords
                    for r_idx, row in enumerate(table):
                        row_props = sum(1 for cell in row if cell and
                                        any(kw in str(cell).upper() for kw in PROPERTY_KEYWORDS))
                        if row_props >= 3:
                            page_has_schedule = True
                            break

                if header_row_idx is None and not page_has_schedule:
                    continue

                # Schedule name (from rows above header)
                schedule_name = ""
                if header_row_idx is not None:
                    for up in range(header_row_idx - 1, -1, -1):
                        cells = [str(c).strip() for c in table[up] if c not in [None, ""]]
                        if cells:
                            schedule_name = " ".join(cells[:3])
                            break

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

                # Identify columns
                mark_col_indices = []
                for i, h in enumerate(header_upper):
                    for kw in TAG_COL_KEYWORDS:
                        if kw in h and len(h) < 25:
                            mark_col_indices.append(i)
                            break

                # Description columns (for extracting embedded tags, and for details)
                desc_col_indices = [
                    i for i, h in enumerate(header_upper)
                    if any(kw in h for kw in ["DESCRIPTION", "TYPE", "MODEL", "SIZE", "CAPACITY", "SERVICE", "REMARK"])
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

                # Extract tags
                for row_idx, row in enumerate(data_rows):
                    # Primary: tags from MARK/TAG columns
                    row_tags = []
                    for mark_col in mark_col_indices:
                        if mark_col >= len(row):
                            continue
                        raw = row[mark_col]
                        # Support compound cells "A, B, C"
                        parts = split_compound_cell(raw)
                        for p in parts:
                            tag = normalize_tag(p)
                            if tag:
                                row_tags.append(tag)

                    # Secondary: scan description columns for embedded tag patterns
                    # e.g., REMARKS = "LD-1-PLENUM installed" -> extract LD-1
                    desc_text_parts = []
                    for desc_col in desc_col_indices:
                        if desc_col < len(row) and row[desc_col]:
                            desc_text_parts.append(str(row[desc_col]))
                    desc_text = " ".join(desc_text_parts)

                    for m in TAG_IN_DESC.finditer(desc_text.upper()):
                        tag = normalize_tag(m.group(1))
                        if tag and tag not in row_tags:
                            row_tags.append(tag)

                    # Store tags + details
                    for tag in row_tags:
                        marks_set.add(tag)
                        details = {}
                        for desc_col in desc_col_indices:
                            if desc_col < len(row) and row[desc_col] and row[desc_col].strip():
                                details[header[desc_col]] = row[desc_col].strip()
                        if details and tag not in mark_details:
                            mark_details[tag] = details

    return schedule_tables, sorted(list(marks_set)), mark_details


def extract_legend_info(pdf_path):
    """Extract legend/abbreviation items."""
    legend_items = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
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


def parse_pdf_schedules(pdf_path, exclude_prefixes=None):
    """
    Main entry point.

    exclude_prefixes: set of equipment type prefixes to exclude (default: VAV).
    Pass exclude_prefixes=set() to get ALL tags including VAV.
    """
    if exclude_prefixes is None:
        exclude_prefixes = EXCLUDE_PREFIXES

    schedules, marks, mark_details = extract_schedules_and_marks(pdf_path)
    legend = extract_legend_info(pdf_path)

    # Filter out excluded equipment types (e.g., VAV boxes)
    if exclude_prefixes:
        filtered_marks = [m for m in marks if get_mark_type(m) not in exclude_prefixes]
        filtered_details = {m: d for m, d in mark_details.items() if get_mark_type(m) not in exclude_prefixes}
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
    }

    return schedules, marks, mark_details, legend, summary


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python schedule_parser.py path/to/blueprint.pdf")
        _sys.exit(1)

    pdf = _sys.argv[1]
    print(f"Parsing: {pdf}")

    schedules, marks, details, legend, summary = parse_pdf_schedules(pdf)

    print(f"\nSchedules found: {summary['total_schedules']}")
    print(f"Equipment marks: {summary['total_marks']}")
    print(f"Legend items:    {summary['legend_items']}")

    if marks:
        print(f"\nMarks ({len(marks)}):")
        for m in marks:
            detail_str = ""
            if m in details:
                detail_str = " | " + ", ".join(f"{k}={v[:30]}" for k, v in details[m].items())
            print(f"  {m}{detail_str}")

    if summary['types']:
        print(f"\nBy type:")
        for t, n in sorted(summary['types'].items(), key=lambda x: -x[1]):
            print(f"  {t}: {n}")

    if schedules:
        print(f"\nSchedule tables:")
        for s in schedules:
            print(f"  Page {s['page']}: {s['schedule_name'][:50] or '(unnamed)'} - {len(s['rows'])} rows")
