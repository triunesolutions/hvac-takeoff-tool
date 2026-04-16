"""
Schedule & Legend Parser

Extracts equipment schedules from HVAC blueprint PDFs.
Returns a list of expected equipment types and marks that
can be used to filter YOLO detections.

Ported from pdf-detection-main/mechanical_processor.py
and enhanced for our pipeline.

Usage:
    from schedule_parser import parse_pdf_schedules
    schedules, marks, mark_details = parse_pdf_schedules("blueprint.pdf")
"""
import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import re
from collections import defaultdict
import pdfplumber
import fitz


def extract_schedules_and_marks(pdf_path):
    """
    Extract schedule tables and equipment marks from a PDF using pdfplumber.

    Returns:
        schedule_tables: list of dicts with table data
        marks: sorted list of unique mark strings (e.g., ['AHU-1', 'FCU-1', 'FCU-2'])
        mark_details: dict mapping mark -> {description, size, cfm, etc.} from schedule
    """
    schedule_tables = []
    marks_set = set()
    mark_details = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text_upper = (page.extract_text() or "").upper()
            page_has_schedule = "SCHEDULE" in text_upper
            tables = page.extract_tables()

            for t_index, table in enumerate(tables):
                if not table or all(
                    all((cell is None or str(cell).strip() == "") for cell in row)
                    for row in table
                ):
                    continue

                # Find header row containing "MARK" or "TAG" or "DESIGNATION"
                header_row_idx = None
                for r_idx, row in enumerate(table):
                    for cell in row:
                        if cell is None:
                            continue
                        cell_upper = str(cell).strip().upper()
                        if cell_upper in ("MARK", "TAG", "DESIGNATION", "UNIT TAG", "EQUIPMENT TAG"):
                            header_row_idx = r_idx
                            break
                        if "MARK" in cell_upper and len(cell_upper) < 20:
                            header_row_idx = r_idx
                            break
                    if header_row_idx is not None:
                        break

                if header_row_idx is None and not page_has_schedule:
                    continue

                # Schedule name from rows above header
                schedule_name = ""
                if header_row_idx is not None:
                    for up in range(header_row_idx - 1, -1, -1):
                        cells = [str(c).strip() for c in table[up] if c not in [None, ""]]
                        if cells:
                            schedule_name = cells[0]
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

                # Collect marks from MARK/TAG columns
                mark_col_indices = [
                    i for i, h in enumerate(header_upper)
                    if any(kw in h for kw in ["MARK", "TAG", "DESIGNATION"])
                ]

                # Also look for description/type columns
                desc_col_indices = [
                    i for i, h in enumerate(header_upper)
                    if any(kw in h for kw in ["DESCRIPTION", "TYPE", "MODEL", "SIZE", "CAPACITY"])
                ]

                for row in data_rows:
                    for mark_col in mark_col_indices:
                        if mark_col < len(row):
                            mark_val = row[mark_col].strip()
                            if mark_val and len(mark_val) < 30:
                                marks_set.add(mark_val)

                                # Collect details from description columns
                                details = {}
                                for desc_col in desc_col_indices:
                                    if desc_col < len(row) and row[desc_col].strip():
                                        details[header[desc_col]] = row[desc_col].strip()
                                if details:
                                    mark_details[mark_val] = details

    return schedule_tables, sorted(list(marks_set)), mark_details


def extract_legend_info(pdf_path):
    """
    Try to find and extract legend/abbreviation information from the PDF.
    Looks for pages with 'LEGEND', 'ABBREVIATION', or 'SYMBOLS' in text.

    Returns:
        legend_items: dict mapping abbreviation -> description
    """
    legend_items = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_upper = text.upper()

            if not any(kw in text_upper for kw in ["LEGEND", "ABBREVIATION", "SYMBOLS", "KEY NOTES"]):
                continue

            # Try to extract tables from legend pages
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                for row in table:
                    clean = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if len(clean) == 2:
                        abbr, desc = clean[0], clean[1]
                        if len(abbr) < 15 and len(desc) > 3:
                            legend_items[abbr] = desc

            # Also try regex on raw text for "ABBR = Description" or "ABBR - Description"
            for line in text.split('\n'):
                m = re.match(r'^([A-Z]{1,6}(?:-\d+)?)\s*[-=:]\s*(.+)$', line.strip())
                if m:
                    abbr, desc = m.group(1), m.group(2).strip()
                    if len(desc) > 3:
                        legend_items[abbr] = desc

    return legend_items


def get_mark_type(mark):
    """Extract equipment type prefix from mark. FCU-10 -> FCU, L-1 -> L"""
    m = re.match(r'^([A-Z]+)', mark)
    return m.group(1) if m else mark.split("-")[0] if "-" in mark else mark


def parse_pdf_schedules(pdf_path):
    """
    Main entry point. Extracts all schedule info from a PDF.

    Returns:
        schedules: list of schedule table dicts
        marks: sorted list of unique mark strings
        mark_details: dict mapping mark -> details from schedule
        legend: dict mapping abbreviation -> description
        summary: dict with counts by equipment type
    """
    schedules, marks, mark_details = extract_schedules_and_marks(pdf_path)
    legend = extract_legend_info(pdf_path)

    # Build summary by type
    type_counts = defaultdict(int)
    for mark in marks:
        type_counts[get_mark_type(mark)] += 1

    summary = {
        'total_marks': len(marks),
        'total_schedules': len(schedules),
        'legend_items': len(legend),
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
    print(f"Parsing: {pdf}\n")

    schedules, marks, details, legend, summary = parse_pdf_schedules(pdf)

    print(f"Schedules found: {summary['total_schedules']}")
    print(f"Equipment marks: {summary['total_marks']}")
    print(f"Legend items:    {summary['legend_items']}")

    if marks:
        print(f"\nMarks ({len(marks)}):")
        for m in marks:
            detail_str = ""
            if m in details:
                detail_str = " | " + ", ".join(f"{k}={v}" for k, v in details[m].items())
            print(f"  {m}{detail_str}")

    if summary['types']:
        print(f"\nBy type:")
        for t, n in sorted(summary['types'].items(), key=lambda x: -x[1]):
            print(f"  {t}: {n}")

    if legend:
        print(f"\nLegend ({len(legend)}):")
        for abbr, desc in sorted(legend.items()):
            print(f"  {abbr} = {desc}")

    if schedules:
        print(f"\nSchedule details:")
        for s in schedules:
            print(f"  Page {s['page']}: {s['schedule_name'] or '(unnamed)'} — {len(s['rows'])} rows")
            print(f"    Columns: {s['header']}")
