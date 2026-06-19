"""
project_info.py — Stage 3, fixed.

Replacement for the legacy extract_project_info() in takeoff_cli.py.
Known issues with the legacy version:
  • date string ("9/01/25") was bleeding into the project_no field
  • picked oldest revision date instead of latest issue
  • no sheet-number heuristic (M001 etc.)

This version:
  • Walks page 1 only (title sheet)
  • Uses spatial layout — finds each labeled value by looking ABOVE the label
  • Distinguishes by field-name patterns (no value reuse across fields)
  • Picks latest revision date from any revision history grid
  • Finds sheet number via typography (tallest standalone span matching M-prefix pattern)
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF


# Label patterns we know how to find — each maps the canonical key to a list
# of label patterns that may appear on the title sheet.
LABELS = {
    'project':       [r'PROJECT\s+NAME', r'PROJECT:', r'^PROJECT$'],
    'project_no':    [r'PROJECT\s+(NO|NUMBER|#)', r'JOB\s+(NO|NUMBER)'],
    'description':   [r'DESCRIPTION', r'SHEET\s+TITLE', r'TITLE'],
    'sheet':         [r'^SHEET(\s+NO|\s+NUMBER|#)?$', r'^SHT$', r'^DWG\s+NO'],
    'scale':         [r'^SCALE:?$', r'SCALE\s+'],
    'date':          [r'^DATE:?$', r'ISSUE\s+DATE', r'DATE\s+OF\s+ISSUE'],
    'drawn_by':      [r'DRAWN\s+BY', r'^BY$', r'^DRAWN$'],
    'checked_by':    [r'CHECKED\s+BY', r'^CHECKED$'],
    'firm':          [r'^FIRM$', r'^ENGINEER$', r'ENGINEERING\s+FIRM'],
    'permit':        [r'PERMIT\s+(NO|NUMBER)'],
    'revision':      [r'^REV(ISION)?S?:?$', r'^REV$'],
}

# Field-value sanity checks: a value extracted for field X must look like X.
# A field with no validator accepts any nearby span — which is how dates, project
# numbers and note prose leaked into description / checked_by / firm. The added
# validators reject those: better to leave a field blank than fill it wrong.
VALUE_VALIDATORS = {
    'project_no':  re.compile(r'^[\w\.\-/]+$'),               # alphanumeric / dotted
    'sheet':       re.compile(r'^[A-Z]{1,2}-?\d{2,4}[A-Z]?$'),  # M001, MP-001
    'scale':       re.compile(r'(1["\']?\s*=\s*\d|\d/\d|NTS|N\.T\.S|NONE)', re.I),
    'date':        re.compile(r'\d{1,4}[\.\-/]\d{1,4}[\.\-/]\d{2,4}'),
    # Sheet title — a real phrase, never a bare date or number.
    'description': re.compile(r'^(?!\s*\d[\d\.\-/]*\s*$)(?=.*[A-Za-z]{3}).+', re.I),
    # Person initials / name — starts with a letter, no long digit runs
    # (rejects the project number "21295.00" and dates).
    'checked_by':  re.compile(r'^[A-Za-z][A-Za-z\.\-\s]{0,30}$'),
    'drawn_by':    re.compile(r'^[A-Za-z][A-Za-z\.\-\s]{0,30}$'),
    # Firm — a real name (has letters), short, not a spec sentence.
    'firm':        re.compile(
        r'^(?!.*\b(SHALL|COMPLIANCE|PROVIDE|REQUIRED|DEMONSTRATE|REFERENCE|COMPONENT|CONNECTION)\b)'
        r'(?=.*[A-Za-z]{3}).{2,50}$', re.I),
}

# Recognise dates in various formats. Used to find the LATEST revision date.
DATE_PATTERNS = [
    (re.compile(r'(\d{1,2})[\.\-/](\d{1,2})[\.\-/](\d{2,4})'), 'mdy'),
    (re.compile(r'(\d{4})[\.\-/](\d{1,2})[\.\-/](\d{1,2})'), 'ymd'),
]


def _parse_date(s: str) -> datetime | None:
    """Parse common date strings; return None if unrecognised."""
    s = s.strip()
    for pat, kind in DATE_PATTERNS:
        m = pat.fullmatch(s) or pat.search(s)
        if not m:
            continue
        a, b, c = m.group(1), m.group(2), m.group(3)
        try:
            if kind == 'ymd':
                return datetime(int(a), int(b), int(c))
            # mdy with 2-digit year normalisation
            year = int(c)
            if year < 100:
                year += 2000 if year < 70 else 1900
            return datetime(year, int(a), int(b))
        except ValueError:
            continue
    return None


def _get_spans(page) -> list[dict]:
    """Return every span on the page with its text, bbox, and font size."""
    spans = []
    blocks = page.get_text('dict')['blocks']
    for b in blocks:
        if 'lines' not in b:
            continue
        for line in b['lines']:
            for sp in line['spans']:
                text = (sp.get('text') or '').strip()
                if not text:
                    continue
                bbox = sp['bbox']
                spans.append({
                    'text': text,
                    'bbox': bbox,
                    'cx': (bbox[0] + bbox[2]) / 2,
                    'cy': (bbox[1] + bbox[3]) / 2,
                    'size': sp.get('size', 0),
                    'h': bbox[3] - bbox[1],
                    'w': bbox[2] - bbox[0],
                })
    return spans


_LABEL_LIKE = re.compile(r':\s*$|^(DATE|NAME|NO|NUMBER|SCALE|DRAWN|CHECKED|FIRM|REV(ISION)?|PROJECT|SHEET|DESCRIPTION|TITLE|BY)[:\s]*$', re.I)


def _looks_like_label(s: str) -> bool:
    """Heuristic: a span ending in ':' or matching a known label word
    is probably a label, not a value."""
    return bool(_LABEL_LIKE.search(s.strip()))


def _find_value_above_or_right(label_span: dict, all_spans: list[dict],
                              value_validator: re.Pattern | None = None
                              ) -> str | None:
    """Given a label span, find the value span ABOVE it (CAD title block) or
    RIGHT of it (form-style title block). Closest spatial match wins, with
    optional regex validation."""
    lx, ly = label_span['cx'], label_span['cy']
    lw, lh = label_span['w'], label_span['h']

    # 1) Look ABOVE: candidate y is less than label y, candidate x is roughly
    #    aligned with label x (within 3× label width)
    candidates_above = []
    for sp in all_spans:
        if sp is label_span:
            continue
        if sp['cy'] >= ly:
            continue
        if abs(sp['cx'] - lx) > max(60, lw * 3):
            continue
        # not too far vertically
        if (ly - sp['cy']) > lh * 6:
            continue
        if _looks_like_label(sp['text']):
            continue
        if value_validator and not value_validator.search(sp['text']):
            continue
        candidates_above.append(sp)

    if candidates_above:
        # Pick the closest above + tallest text (typographic)
        candidates_above.sort(key=lambda s: (ly - s['cy'], -s['h']))
        return candidates_above[0]['text']

    # 2) Look RIGHT on same baseline
    candidates_right = []
    for sp in all_spans:
        if sp is label_span:
            continue
        if abs(sp['cy'] - ly) > lh * 0.7:
            continue
        if sp['cx'] <= lx + lw:
            continue
        if sp['cx'] - (lx + lw) > lw * 4:
            continue
        if _looks_like_label(sp['text']):
            continue
        if value_validator and not value_validator.search(sp['text']):
            continue
        candidates_right.append(sp)
    if candidates_right:
        candidates_right.sort(key=lambda s: s['cx'])
        return candidates_right[0]['text']

    return None


def _find_sheet_number(spans: list[dict]) -> str | None:
    """Sheet number heuristic — tallest standalone span matching M-prefix pattern."""
    pat = re.compile(r'^[A-Z]{1,2}-?\d{2,4}[A-Z]?$')
    candidates = [sp for sp in spans if pat.match(sp['text'])]
    if not candidates:
        return None
    candidates.sort(key=lambda s: -s['h'])
    return candidates[0]['text']


def _find_latest_date(spans: list[dict]) -> str | None:
    """Find the latest date on page — typically picks the most recent revision."""
    dated = []
    for sp in spans:
        d = _parse_date(sp['text'])
        if d:
            dated.append((d, sp['text']))
    if not dated:
        return None
    dated.sort(key=lambda x: x[0], reverse=True)
    return dated[0][1]


def _find_firm(spans: list[dict]) -> str | None:
    """ALL CAPS span containing ENGINEERING / CONSULTANTS / ASSOCIATES / etc.

    Guarded against note prose: a spec sentence like "...DEMONSTRATE DESIGN
    COMPLIANCE..." also contains "DESIGN" and used to win. A real firm name is
    short (a few words) and free of spec/note vocabulary.
    """
    pat = re.compile(r'\b(ENGINEERING|CONSULTANTS|ASSOCIATES|ARCHITECTS|DESIGN)\b')
    prose = re.compile(r'\b(SHALL|COMPLIANCE|REFERENCE|COMPONENT|PROVIDE|CONNECTION|'
                       r'REQUIRED|CONTRACTOR|INSTALL|DEMONSTRATE|THESE|ABOVE)\b')
    candidates = []
    for sp in spans:
        t = sp['text'].strip()
        if not t or t.upper() != t:          # must be ALL CAPS
            continue
        up = t.upper()
        if not pat.search(up) or prose.search(up):
            continue
        if len(t.split()) > 6 or len(t) > 50:  # too long to be a firm name
            continue
        candidates.append(sp)
    if not candidates:
        return None
    candidates.sort(key=lambda s: -s['h'])
    return candidates[0]['text'].strip()


def _find_address(spans: list[dict]) -> str | None:
    """Street-address regex."""
    pat = re.compile(r'^\d{2,6}\s+[A-Z][A-Z\s\.]+(\s+(ST|AVE|BLVD|DR|RD|WAY|CT|LN|PL))\b', re.I)
    for sp in spans:
        if pat.search(sp['text']):
            return sp['text'].strip()
    return None


def extract_project_info(pdf_path: Path) -> dict:
    """Extract title-block metadata from page 1 of the PDF."""
    doc = fitz.open(str(pdf_path))
    try:
        if doc.page_count == 0:
            return {}
        page = doc[0]
        spans = _get_spans(page)
    finally:
        doc.close()

    out: dict[str, str | None] = {
        'project': None,
        'project_no': None,
        'description': None,
        'sheet': None,
        'scale': None,
        'date': None,
        'drawn_by': None,
        'checked_by': None,
        'firm': None,
        'address': None,
        'permit': None,
        'revision': None,
    }

    # Find label spans, then look for adjacent values
    used_value_spans: set[int] = set()
    for key, patterns in LABELS.items():
        validator = VALUE_VALIDATORS.get(key)
        for pat_str in patterns:
            pat = re.compile(pat_str, re.I)
            for sp in spans:
                if id(sp) in used_value_spans:
                    continue
                if not pat.search(sp['text']):
                    continue
                # Found a label — look for value
                value = _find_value_above_or_right(sp, spans, value_validator=validator)
                if value:
                    out[key] = value.strip()
                    break
            if out[key]:
                break

    # Independent heuristics that don't rely on label proximity
    if not out['sheet']:
        out['sheet'] = _find_sheet_number(spans)
    if not out['date']:
        out['date'] = _find_latest_date(spans)
    if not out['firm']:
        out['firm'] = _find_firm(spans)
    if not out['address']:
        out['address'] = _find_address(spans)

    # Final cleanup: if project_no looks like a date, blank it.
    if out.get('project_no') and VALUE_VALIDATORS['date'].search(out['project_no']):
        out['project_no'] = None

    # Drop None values to keep the JSON clean
    return {k: v for k, v in out.items() if v}


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--out')
    args = ap.parse_args()
    info = extract_project_info(Path(args.pdf))
    print(json.dumps(info, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=2), encoding='utf-8')
        print(f'Wrote {args.out}')
