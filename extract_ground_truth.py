import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

"""
extract_ground_truth.py - Parse Bluebeam counts from Triune's "Completed Takeoff"
PDFs into a structured JSONL ground-truth dataset.

Source schema decoded in:
  C:\\Users\\JFL\\Downloads\\Bluebeam Integration files\\triune_annotator.py
  C:\\Users\\JFL\\Downloads\\Bluebeam Integration files\\triune_tools_decoded.json

Key fields per Bluebeam count annotation:
  /IT = PolygonCount    → a real count (not line, text box, measurement)
  /Subj                 → subject  (AD-GRD, CONDENSING UNIT, FAN, ...)
  /Label                → the tag string (A-1, CU-1, MVD, ...)
  /Rect                 → bounding box (PyMuPDF Annot.rect handles rotation)
  /BSIColumnData        → 19-position array of properties

Output: one JSONL row per count. Each row pairs (raw_pdf, page, cx, cy, tag,
subject, properties) → the supervised training signal for tag detection.

Usage:
  python extract_ground_truth.py \\
    --root "C:\\Users\\JFL\\Downloads\\DATA FILES" \\
    --out  ground_truth.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import fitz

# ────────────────────────────────────────────────────────────────────────────
# Triune BSIColumnData schema (from triune_annotator.py).
# Positional — all 19 entries present, even deleted columns.
# ────────────────────────────────────────────────────────────────────────────
BSI_SCHEMA = [
    ("ACCESSORIES",   False),
    ("MODEL",         False),
    ("MANUFACTURER",  False),
    ("REMARK",        False),
    ("NECK SIZE",     False),
    ("TYPE",          False),
    ("UNIT",          True),   # deleted
    ("MOUNTING",      False),
    ("CFM",           False),
    ("R1",            True),   # deleted
    ("U1",            True),   # deleted
    ("DUCT SIZE",     False),
    ("UNITS",         False),
    ("TEST",          True),   # deleted
    ("FACE SIZE",     False),
    ("DESCRIPTION",   False),
    ("DAMPER TYPE",   False),
    ("LOCATION",      True),   # deleted
    ("HET OR DAM",    True),   # deleted
]

# Subjects that are noise — not real equipment counts
NOISE_SUBJECTS = {
    "", "Text Box", "Rectangle", "Line", "Length Measurement",
    "Snapshot", "Callout", "Polygon", "Cloud+", "Polyline", "Arrow",
    "Ellipse", "Square", "Highlight", "Underline", "Note",
}


# ────────────────────────────────────────────────────────────────────────────
# BSIColumnData parsing
# ────────────────────────────────────────────────────────────────────────────

_BSI_ITEM = re.compile(r'\(((?:[^()\\]|\\.)*)\)')


def parse_bsi_column_data(raw):
    """Parse a BSIColumnData array string into a dict keyed by column name.

    Input:  '[ () (SPD) (PRICE) (SEE SCHEDULE) (6") ... ]' or None
    Output: {"MODEL": "SPD", "MANUFACTURER": "PRICE", ...}  (only active columns)
    """
    if not raw:
        return {}
    items = _BSI_ITEM.findall(raw)
    # Unescape PDF string escapes
    items = [i.replace('\\(', '(').replace('\\)', ')').replace('\\\\', '\\') for i in items]
    out = {}
    for i, (name, deleted) in enumerate(BSI_SCHEMA):
        if i >= len(items):
            break
        if deleted:
            continue
        val = items[i].strip()
        if val:
            out[name] = val
    return out


# ────────────────────────────────────────────────────────────────────────────
# Project folder layout discovery
# ────────────────────────────────────────────────────────────────────────────

def find_projects(root: Path):
    """Yield (project_name, completed_takeoff_pdf, raw_pdf) tuples.

    Structure:
      root / "FILES FOR IT TEAM N" / "<Project Name>" /
          Completed Takeoff/*.pdf
          Plans_Specs/*.pdf
    """
    for batch in sorted(root.iterdir()):
        if not batch.is_dir():
            continue
        for proj in sorted(batch.iterdir()):
            if not proj.is_dir():
                continue
            completed_dir = proj / "Completed Takeoff"
            plans_dir = proj / "Plans_Specs"
            if not completed_dir.is_dir() or not plans_dir.is_dir():
                continue
            completed_pdfs = list(completed_dir.glob("*.pdf"))
            raw_pdfs = list(plans_dir.glob("*.pdf"))
            if not completed_pdfs or not raw_pdfs:
                continue
            yield proj.name, completed_pdfs[0], raw_pdfs[0]


# ────────────────────────────────────────────────────────────────────────────
# Per-PDF annotation extraction
# ────────────────────────────────────────────────────────────────────────────

_RAW_RECT_RE = re.compile(r'/Rect\s*\[\s*([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s*\]')


def native_rect_to_display(x0, y0, x1, y1, mb_w, mb_h, rotation):
    """Convert PDF native /Rect (bottom-left origin, y-up) to display pixel-style
    coords (top-left origin, y-down) in post-rotation space.

    Display dims:
      rotation 0, 180  -> (mb_w, mb_h)
      rotation 90, 270 -> (mb_h, mb_w)
    """
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)

    if rotation == 0:
        # display_x = x,        display_y = mb_h - y
        return lo_x, mb_h - hi_y, hi_x, mb_h - lo_y
    elif rotation == 90:
        # PDF /Rotate 90 = 90° clockwise display rotation
        # display_x = y,        display_y = x
        return lo_y, lo_x, hi_y, hi_x
    elif rotation == 180:
        # display_x = mb_w - x, display_y = y
        return mb_w - hi_x, lo_y, mb_w - lo_x, hi_y
    elif rotation == 270:
        # display_x = mb_h - y, display_y = mb_w - x
        return mb_h - hi_y, mb_w - hi_x, mb_h - lo_y, mb_w - lo_x
    return lo_x, mb_h - hi_y, hi_x, mb_h - lo_y


def extract_counts(takeoff_pdf: Path):
    """Yield one dict per Bluebeam count annotation."""
    try:
        doc = fitz.open(str(takeoff_pdf))
    except Exception as e:
        print(f"  ! open failed: {e}")
        return

    for pno in range(len(doc)):
        page = doc[pno]
        try:
            annots = list(page.annots() or [])
        except Exception:
            continue

        # True mediabox (unrotated) — rotation applied on top
        mb = page.mediabox
        mb_w = float(mb.width)
        mb_h = float(mb.height)
        rotation = page.rotation
        # Display dims (what the rendered image will be at scale=1)
        if rotation in (90, 270):
            disp_w, disp_h = mb_h, mb_w
        else:
            disp_w, disp_h = mb_w, mb_h

        for a in annots:
            info = a.info or {}
            subj = info.get('subject', '').strip()
            if subj in NOISE_SUBJECTS:
                continue

            raw_obj = ''
            try:
                raw_obj = doc.xref_object(a.xref)
            except Exception:
                pass

            # Must be a PolygonCount to count as a real tagged equipment count
            if '/PolygonCount' not in raw_obj:
                continue

            # Pull Label (the tag)
            label_m = re.search(r'/Label\s*\(((?:[^()\\]|\\.)*)\)', raw_obj)
            tag = None
            if label_m:
                tag = label_m.group(1).replace('\\(', '(').replace('\\)', ')').strip()

            # Fallback: content (newline-joined Label + count like "A6\n1")
            if not tag:
                c = info.get('content', '') or ''
                first_line = c.split('\n')[0].strip() if c else ''
                if first_line:
                    tag = first_line

            # BSIColumnData
            bsi_m = re.search(r'/BSIColumnData\s*\[([^\]]*)\]', raw_obj)
            bsi = parse_bsi_column_data(bsi_m.group(1) if bsi_m else None)

            # Count style
            cs_m = re.search(r'/CountStyle\s*/(\w+)', raw_obj)
            count_style = cs_m.group(1) if cs_m else None

            # Position — read raw /Rect and manually apply rotation transform.
            # PyMuPDF's a.rect mishandles rotation=90/270 for some PDFs, so we do it ourselves.
            rm = _RAW_RECT_RE.search(raw_obj)
            if not rm:
                continue
            rx0, ry0, rx1, ry1 = (float(rm.group(i)) for i in (1, 2, 3, 4))
            dx0, dy0, dx1, dy1 = native_rect_to_display(
                rx0, ry0, rx1, ry1, mb_w, mb_h, rotation
            )
            cx = (dx0 + dx1) / 2
            cy = (dy0 + dy1) / 2

            yield {
                'page': pno + 1,
                'page_w': disp_w,
                'page_h': disp_h,
                'rotation': rotation,
                'cx': cx,
                'cy': cy,
                'rect': [dx0, dy0, dx1, dy1],
                'subject': subj,
                'tag': tag,
                'count_style': count_style,
                'properties': bsi,
            }
    doc.close()


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Folder containing "FILES FOR IT TEAM N" batches')
    ap.add_argument('--out', default='ground_truth.jsonl')
    ap.add_argument('--limit', type=int, default=None, help='Process at most N projects (for quick verify)')
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    n_projects = 0
    n_counts = 0
    subj_totals = {}
    tag_totals = {}

    with open(out_path, 'w', encoding='utf-8') as f:
        for project_name, takeoff_pdf, raw_pdf in find_projects(root):
            if args.limit and n_projects >= args.limit:
                break
            n_projects += 1
            proj_counts = 0
            print(f"[{n_projects}] {project_name}")
            print(f"      takeoff: {takeoff_pdf.name}")
            print(f"      raw:     {raw_pdf.name}")
            try:
                for c in extract_counts(takeoff_pdf):
                    c['project'] = project_name
                    c['raw_pdf'] = str(raw_pdf)
                    c['takeoff_pdf'] = str(takeoff_pdf)
                    f.write(json.dumps(c, ensure_ascii=False) + '\n')
                    proj_counts += 1
                    subj_totals[c['subject']] = subj_totals.get(c['subject'], 0) + 1
                    if c.get('tag'):
                        tag_totals[c['tag']] = tag_totals.get(c['tag'], 0) + 1
            except Exception as e:
                print(f"      ! extract failed: {e}")
                continue
            n_counts += proj_counts
            print(f"      → {proj_counts} counts")

    print(f"\n{'='*70}")
    print(f"DONE — wrote {n_counts} counts from {n_projects} projects to {out_path}")
    print(f"{'='*70}")
    print(f"\nTop 20 subjects:")
    for s, c in sorted(subj_totals.items(), key=lambda x: -x[1])[:20]:
        print(f"  {c:5d}  {s}")
    print(f"\nTop 20 tags:")
    for t, c in sorted(tag_totals.items(), key=lambda x: -x[1])[:20]:
        print(f"  {c:5d}  {t!r}")
    print(f"\nUnique subjects: {len(subj_totals)}")
    print(f"Unique tags:     {len(tag_totals)}")


if __name__ == '__main__':
    main()
