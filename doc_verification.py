"""Step-1 document-set verification.

Mirrors what a human estimator does BEFORE counting anything: read the sheet
index off the cover, cross-check it against the pages actually present, and
surface the red flags (NOT FOR CONSTRUCTION watermark, missing sheets, no index,
no legend, etc.). See the team's "Step 1 — Gather Your Documents" reference.

Heuristic, not authoritative — every flag carries WHY so a human can confirm.
Pure-stdlib + PyMuPDF (already a backend dependency).
"""

import re

import fitz  # PyMuPDF

# Discipline prefix → trade. Longer keys are checked before single letters.
DISCIPLINE = {
    'MECH': 'Mechanical', 'HVAC': 'Mechanical', 'MH': 'Mechanical', 'ME': 'Mechanical',
    'MP': 'Mechanical', 'MD': 'Mechanical', 'HV': 'Mechanical', 'M': 'Mechanical',
    'ARCH': 'Architectural', 'AR': 'Architectural', 'A': 'Architectural',
    'STRUCT': 'Structural', 'ST': 'Structural', 'S': 'Structural',
    'PLMB': 'Plumbing', 'PL': 'Plumbing', 'P': 'Plumbing',
    'ELEC': 'Electrical', 'EL': 'Electrical', 'E': 'Electrical',
    'CIV': 'Civil', 'C': 'Civil',
    'FP': 'Fire Protection', 'FA': 'Fire Alarm', 'F': 'Fire Protection',
    'GEN': 'General', 'GN': 'General', 'G': 'General', 'T': 'General',
}

# Descriptors the page classifier appends after the sheet id in its evidence
# string, e.g. "sheet=M-400-alt-schedule" → sheet number is "M-400".
_PLAN_DESCRIPTORS = (
    'alt-roof-plan', 'alt-floor-plan', 'alt-misc-plan', 'alt-details', 'alt-schedule',
    'roof-plan', 'floor-plan', 'misc-plan', 'details', 'schedule',
)

# Watermarks that mean "do not take off from this yet". Only specific,
# unambiguous phrases — bare words like "PRELIMINARY" appear in normal notes
# and would false-positive, so they're deliberately excluded.
_WATERMARKS = [
    'NOT FOR CONSTRUCTION', 'NOT FOR BIDDING', 'NOT FOR BID', 'FOR REVIEW ONLY',
]
# Issue type — only strict, self-announcing phrases (an index/title block says
# "ISSUED FOR ..." or "... SET"). Bare "FOR CONSTRUCTION" is rejected because it
# shows up in body notes ("...prior to construction").
_ISSUE_PATTERNS = [
    ('For Construction', r'(ISSUED\s+FOR\s+CONSTRUCTION|CONSTRUCTION\s+SET)'),
    ('For Permit', r'(ISSUED\s+FOR\s+PERMIT|PERMIT\s+SET)'),
    ('For Bid', r'(ISSUED\s+FOR\s+BID|BID\s+SET)'),
]

# Content blocks that can live INSIDE any sheet (corner of a plan, top of a
# schedule), not only on a dedicated sheet. We scan every page's text for these
# so an embedded legend/notes block is found the way a human finds it.
CONTENT_BLOCKS = {
    'legend': ['LEGEND', 'SYMBOL', 'ABBREVIATION'],
    'general_notes': ['GENERAL NOTES', 'MECHANICAL NOTES', 'SHEET NOTES'],
    'keynotes': ['KEYNOTE'],
    'drawing_list': ['SHEET INDEX', 'DRAWING INDEX', 'DRAWING LIST', 'SHEET LIST',
                     'INDEX OF DRAWINGS', 'INDEX OF SHEETS'],
    'title_block': ['TITLE SHEET', 'COVER SHEET'],
}
_CONTENT_LABEL = {
    'legend': 'Legend / symbols',
    'general_notes': 'General notes',
    'keynotes': 'Keynotes',
    'drawing_list': 'Drawing list',
    'title_block': 'Title/cover block',
}

# A page only counts as a sheet index if it announces itself. This is the key
# guard against harvesting "<word> <number>" prose as fake sheet entries.
_INDEX_HEADERS = (
    'SHEET INDEX', 'DRAWING INDEX', 'DRAWING LIST', 'SHEET LIST',
    'INDEX OF DRAWINGS', 'INDEX OF SHEETS', 'LIST OF DRAWINGS', 'DRAWING SCHEDULE',
)
# Only real discipline prefixes are accepted at the start of an index entry,
# so 'AND 2', 'OF 1200', 'CO2', 'PER 250' can never parse as sheets.
_PREFIXES = sorted(DISCIPLINE.keys(), key=len, reverse=True)
_INDEX_LINE_RE = re.compile(
    r'^(' + '|'.join(_PREFIXES) + r')[-\s]?(\d{1,4}[A-Z]?)\b\s*[-:.–]?\s*(.{2,70})$'
)
# A line that is ONLY a sheet number (the common table-column index format,
# where titles sit in a separate column and linearize onto other lines).
_INDEX_STANDALONE_RE = re.compile(
    r'^(' + '|'.join(_PREFIXES) + r')[-\s]?(\d{1,4}[A-Z]?)$'
)


def _discipline(prefix: str) -> str:
    p = (prefix or '').upper()
    return DISCIPLINE.get(p) or DISCIPLINE.get(p[:2]) or DISCIPLINE.get(p[:1]) or 'Unknown'


def _canon(sheet: str) -> str:
    """Match key: uppercase, strip everything but letters+digits (so 'M-101'
    and 'M101' compare equal, 'M1' stays distinct)."""
    return re.sub(r'[^A-Z0-9]', '', (sheet or '').upper())


def _clean_sheet(ev_value: str) -> str:
    """Strip the classifier's descriptor suffix to recover the sheet number."""
    s = ev_value
    for d in _PLAN_DESCRIPTORS:
        if s.endswith('-' + d):
            return s[: -(len(d) + 1)]
    return s


def _sheet_of(row: dict) -> str:
    for ev in row.get('evidence') or []:
        if isinstance(ev, str) and ev.startswith('sheet='):
            return _clean_sheet(ev.split('=', 1)[1])
    return ''


def extract_sheet_index(doc, max_pages: int = 5):
    """Find a drawing/sheet index on the first few pages. Returns
    (entries, page_index) or ([], None). Each entry: {sheet, title, discipline}.

    Handles both real index formats, with guards against harvesting prose:
      A. same-line "<SHEET#>  <title>" rows — accepted when the page has an
         index header keyword (≥3 rows), and
      B. a dense column of standalone sheet numbers (≥6) — the common CAD
         table format where titles sit in a separate column; no header needed,
         the density itself is the signal.
    Both require a real discipline prefix, so 'AND 2'/'CO2'/'OF 1200' can't parse.
    """
    best: list = []
    best_page = None
    for pno in range(min(max_pages, doc.page_count)):
        text = doc[pno].get_text('text') or ''
        upper = text.upper()
        has_header = any(h in upper for h in _INDEX_HEADERS)
        lines = [' '.join(raw.split()) for raw in text.splitlines() if raw.strip()]

        # Path A — same-line entries with titles.
        same, seen_a = [], set()
        for line in lines:
            m = _INDEX_LINE_RE.match(line)
            if not m:
                continue
            title = m.group(3).strip(' -:.–')
            if not re.search(r'[A-Za-z]{3,}', title):
                continue
            sheet = f'{m.group(1)}-{m.group(2)}'
            k = _canon(sheet)
            if k in seen_a:
                continue
            seen_a.add(k)
            same.append({'sheet': sheet, 'title': title[:70], 'discipline': _discipline(m.group(1))})

        # Path B — standalone sheet-number column.
        col, seen_b = [], set()
        for line in lines:
            m = _INDEX_STANDALONE_RE.match(line)
            if not m:
                continue
            sheet = f'{m.group(1)}-{m.group(2)}'
            k = _canon(sheet)
            if k in seen_b:
                continue
            seen_b.add(k)
            col.append({'sheet': sheet, 'title': '', 'discipline': _discipline(m.group(1))})

        entries: list = []
        if has_header and len(same) >= 3:
            entries = same
        elif len(col) >= 6:
            entries = col
            titles = {_canon(e['sheet']): e['title'] for e in same if e['title']}
            for e in entries:
                e['title'] = titles.get(_canon(e['sheet']), '')
        elif has_header and len(col) >= 3:
            entries = col

        if len(entries) > len(best):
            best, best_page = entries, pno
    return best, best_page


# Words that mark the sheet-title line(s) inside a title block.
_TITLE_KEYWORDS = (
    'PLAN', 'SCHEDULE', 'NOTES', 'LEGEND', 'DETAIL', 'DIAGRAM', 'RISER',
    'COVER', 'TITLE', 'ABBREVIATION', 'SECTION', 'ENLARGED', 'SPECIFICATION',
)
_SHEET_TOKEN_RE = re.compile(r'^(' + '|'.join(_PREFIXES) + r')[-\s]?\d{1,4}[A-Z]?$')


def _type_from_title(title: str):
    """Page type inferred from the sheet title — the most reliable signal."""
    u = (title or '').upper()
    if 'LEGEND' in u or 'SYMBOL' in u or 'ABBREV' in u:
        return 'legend'
    if 'NOTES' in u:
        return 'notes'
    if 'SCHEDULE' in u:
        return 'schedule'
    if 'DETAIL' in u:
        return 'details'
    if 'RISER' in u or 'DIAGRAM' in u:
        return 'riser_diagram'
    if 'PLAN' in u:
        return 'plan'
    if 'COVER' in u or 'TITLE SHEET' in u:
        return 'cover'
    return None


def read_title_block(page) -> dict:
    """Read the sheet number + sheet title from a page's title block.

    Title blocks live in the bottom strip (most firms) or the right edge.
    Returns {'sheet', 'title'} or {} if no sheet token is found.
    """
    r = page.rect
    regions = [
        fitz.Rect(0, r.height * 0.82, r.width, r.height),   # bottom strip
        fitz.Rect(r.width * 0.76, 0, r.width, r.height),    # right edge
    ]
    for clip in regions:
        lines = [' '.join(l.split()) for l in (page.get_text('text', clip=clip) or '').splitlines() if l.strip()]
        sidx = next((i for i, l in enumerate(lines) if _SHEET_TOKEN_RE.match(l)), None)
        if sidx is None:
            continue
        sheet = re.sub(r'\s', '', lines[sidx]).upper()
        # Title: nearby lines carrying a title keyword (join multi-line titles).
        parts, seen = [], set()
        for j in range(max(0, sidx - 4), min(len(lines), sidx + 5)):
            lj = lines[j]
            if _SHEET_TOKEN_RE.match(lj) or len(lj) > 50:
                continue
            if any(k in lj.upper() for k in _TITLE_KEYWORDS) and lj not in seen:
                seen.add(lj)
                parts.append(lj)
        return {'sheet': sheet, 'title': ' '.join(parts)[:80]}
    return {}


def read_project_info(doc, pdf_name: str = '') -> dict:
    """Pull clean project info from the title-block region (bottom + right
    strips of the first few pages). Only emits fields it can extract
    confidently — leaves the rest out rather than guessing (the old extractor
    produced garbage like firm='LIGHTING CONSULTANT CONSULTANTS').
    """
    chunks = []
    for pno in range(min(3, doc.page_count)):
        pg = doc[pno]
        r = pg.rect
        for clip in (fitz.Rect(0, r.height * 0.82, r.width, r.height),
                     fitz.Rect(r.width * 0.76, 0, r.width, r.height)):
            chunks.append(pg.get_text('text', clip=clip) or '')
    blob = '\n'.join(chunks)
    up = blob.upper()
    info: dict[str, str] = {}

    # Project number — grouped form like 05.5444.000 (very title-block-specific).
    m = re.search(r'\b(\d{2,3}\.\d{3,5}\.\d{1,4})\b', blob)
    if m:
        info['project_no'] = m.group(1)
    # Date (issue/plot date).
    m = re.search(r'\b(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\b', blob)
    if m:
        info['date'] = m.group(1)
    # Scale.
    if 'NOT TO SCALE' in up:
        info['scale'] = 'NOT TO SCALE'
    else:
        m = re.search(r'(\d{1,2}/\d{1,2}"\s*=\s*[\d\'\-"]+)', blob)
        if m:
            info['scale'] = m.group(1)
    # Engineer — a "FIRST M. LAST" all-caps name (the stamping engineer).
    m = re.search(r'\b([A-Z][A-Z]+ [A-Z]\.? [A-Z][A-Z]+)\b', blob)
    if m:
        info['engineer'] = m.group(1)
    # NOTE: firm is deliberately NOT extracted. Title blocks list multiple firms
    # (architect + MEP/structural/lighting consultants) as label/value pairs, and
    # text-only parsing can't tell a firm NAME from a field LABEL ("ARCHITECT",
    # "MECHANICAL ENGINEER") — it produced garbage ("LIGHTING CONSULTANT
    # CONSULTANTS", "SUBMITTED TO ENGINEER", "ARCHITECT"). The engineer name
    # above is the useful, reliable signal for HVAC; firm would need spatial
    # title-block parsing to do correctly.

    # Lead sheet (first page) number + title.
    if doc.page_count:
        tb = read_title_block(doc[0])
        if tb.get('sheet'):
            info['lead_sheet'] = tb['sheet']
        if tb.get('title'):
            info['lead_sheet_title'] = tb['title']

    # Project name — from the filename (strip a leading date prefix); reliable
    # when the title block's project-name field can't be located spatially.
    base = re.sub(r'\.[Pp][Dd][Ff]$', '', pdf_name)
    base = re.sub(r'^[\d.\-_ ]+', '', base).strip()
    if base:
        info['project'] = base

    return info


def extract_index_from_pdf(pdf_path) -> list:
    """Extract a drawing index from a separately-uploaded cover/index PDF.
    Tries a formal index page first; falls back to reading every sheet's title
    block (works when the upload is the full bid set rather than just an index).
    Returns a list of {sheet, title, discipline}.
    """
    doc = fitz.open(str(pdf_path))
    try:
        entries, _ = extract_sheet_index(doc)
        if entries:
            return entries
        out, seen = [], set()
        for p in range(min(doc.page_count, 80)):
            tb = read_title_block(doc[p])
            s = tb.get('sheet')
            if not s or _canon(s) in seen:
                continue
            seen.add(_canon(s))
            pm = re.match(r'^([A-Z]{1,4})', s.upper())
            out.append({'sheet': s, 'title': tb.get('title', ''),
                        'discipline': _discipline(pm.group(1)) if pm else 'Unknown'})
        return out
    finally:
        doc.close()


def build_verification(pdf_path, classifications, uploaded_index=None) -> dict:
    """Produce the Step-1 verification report for one input PDF.

    `classifications` is the parsed *_page_classifications.json list (may be
    empty if the pipeline hasn't typed the pages yet).
    `uploaded_index` is an optional list of {sheet,title,discipline} extracted
    from a separately-uploaded cover/index PDF — when present it's the
    authoritative drawing index and enables a real completeness (missing) check.
    """
    doc = fitz.open(str(pdf_path))
    try:
        npages = doc.page_count
        scan_n = min(npages, 60)
        uppers = [(doc[p].get_text('text') or '').upper() for p in range(scan_n)]
        joined = '\n'.join(uppers)
        watermark = next((w for w in _WATERMARKS if w in joined), None)
        issue_type = next((label for label, pat in _ISSUE_PATTERNS if re.search(pat, joined)), None)
        in_pdf_index, index_page = extract_sheet_index(doc)
        title_blocks = [read_title_block(doc[p]) for p in range(npages)]
        pdf_name = str(pdf_path).replace('\\', '/').rsplit('/', 1)[-1]
        project = read_project_info(doc, pdf_name)
    finally:
        doc.close()

    # Content blocks found INSIDE pages (0-based page indices), e.g. a legend
    # printed in the corner of a roof plan.
    content: dict[str, list[int]] = {}
    for cat, phrases in CONTENT_BLOCKS.items():
        hits = [i for i, u in enumerate(uppers) if any(ph in u for ph in phrases)]
        if hits:
            content[cat] = hits

    # Pages present — driven by the authoritative title block (every page),
    # with the classifier's type folded in where available. Falls back to the
    # classifier's sheet evidence when a title block can't be read.
    cls_by_index = {}
    for row in classifications or []:
        cls_by_index[int(row.get('page', 0)) - 1] = row

    present = []
    types_present = set()
    disc_counts: dict[str, int] = {}
    for i in range(npages):
        tb = title_blocks[i] if i < len(title_blocks) else {}
        row = cls_by_index.get(i, {})
        sheet = tb.get('sheet') or _sheet_of(row)
        title = tb.get('title', '')
        typ = row.get('type') or _type_from_title(title)
        if typ:
            types_present.add(typ)
        pm = re.match(r'^([A-Z]{1,4})', sheet.upper()) if sheet else None
        disc = _discipline(pm.group(1)) if pm else 'Unknown'
        disc_counts[disc] = disc_counts.get(disc, 0) + 1
        present.append({'index': i, 'sheet': sheet, 'title': title, 'type': typ, 'discipline': disc})

    # The in-set drawing list, read from title blocks (sheet → title).
    drawing_list = [
        {'sheet': p['sheet'], 'title': p['title'], 'index': p['index'], 'discipline': p['discipline']}
        for p in present if p['sheet']
    ]

    # A FORMAL index proves what SHOULD be present. An uploaded cover/index PDF
    # wins; then an index page inside this PDF; else fall back to the title-block
    # drawing list (which only describes the sheets actually present).
    formal_index = uploaded_index or in_pdf_index
    index_source = 'uploaded' if uploaded_index else ('index_page' if in_pdf_index else None)
    if formal_index:
        drawing_list_source = index_source
        eff_index = formal_index
    elif len(drawing_list) >= 2:
        drawing_list_source = 'title_blocks'
        eff_index = drawing_list
    else:
        drawing_list_source = None
        eff_index = []

    present_keys = {_canon(p['sheet']) for p in present if p['sheet']}
    present_disc = {p['discipline'] for p in present if p['sheet'] and p['discipline'] != 'Unknown'}
    index_keys = {_canon(e['sheet']) for e in eff_index}
    # Completeness check, scoped to the disciplines actually in THIS PDF — so a
    # full-project index doesn't flag every A/S/P/E sheet as "missing" from a
    # mechanical-only set. Only a formal index (uploaded or in-PDF) can do this.
    scoped_index = [e for e in formal_index
                    if not present_disc or e.get('discipline') in present_disc] if formal_index else []
    missing = ([e for e in scoped_index if _canon(e['sheet']) not in present_keys]
               if (formal_index and present_keys) else [])
    unlisted = [p for p in present if p['sheet'] and index_keys and _canon(p['sheet']) not in index_keys]

    # ── Red flags (Part 8 of the reference) ──────────────────────────────────
    flags = []
    if watermark:
        flags.append({
            'level': 'error', 'code': 'watermark',
            'msg': f'"{watermark}" watermark detected — these drawings may be preliminary. '
                   f'Do not take off from them until you have a construction/bid set.',
        })
    if formal_index:
        # Formal index present — `missing` covers the gap case below. When all
        # the present-discipline sheets are accounted for, say so positively.
        if not missing:
            src = 'uploaded index' if uploaded_index else 'drawing index'
            flags.append({
                'level': 'info', 'code': 'index_complete',
                'msg': f'Cross-checked against the {src} — all {len(scoped_index)} sheet(s) for the '
                       f'discipline(s) in this PDF are present.',
            })
    elif drawing_list_source == 'title_blocks':
        flags.append({
            'level': 'info', 'code': 'index_from_title_blocks',
            'msg': f'No formal drawing-index page; built a {len(drawing_list)}-sheet list from the '
                   f'title blocks (the sheets present). Completeness vs. the full bid set can\'t be '
                   f'confirmed from this PDF alone. Upload the cover/index PDF to cross-check.',
        })
    else:
        flags.append({
            'level': 'warn', 'code': 'no_index',
            'msg': 'No sheet index / drawing list found — cannot confirm the set is complete '
                   '(the Golden Rule). Verify against the bid documents before counting.',
        })
    if missing:
        names = ', '.join(e['sheet'] for e in missing[:12]) + ('…' if len(missing) > 12 else '')
        flags.append({
            'level': 'error', 'code': 'missing_sheets',
            'msg': f'{len(missing)} sheet(s) listed in the index are NOT in this PDF: {names}. '
                   f'Get the missing sheets before starting.',
        })
    # Cover/legend are satisfied by a dedicated sheet, an embedded block, OR a
    # readable title block (which carries the project/sheet info a cover would).
    has_cover = 'cover' in types_present or bool(content.get('title_block')) or bool(drawing_list)
    has_legend = 'legend' in types_present or bool(content.get('legend'))
    if not has_cover:
        flags.append({
            'level': 'warn', 'code': 'no_cover',
            'msg': 'No cover/title sheet or title block detected — project info and the sheet index usually live here.',
        })
    if not has_legend:
        flags.append({
            'level': 'warn', 'code': 'no_legend',
            'msg': 'No legend/symbols found anywhere in the set — symbol meanings cannot be confirmed.',
        })

    return {
        'page_count': npages,
        'project': project,
        'issue_type': issue_type,
        'watermark': watermark,
        'index_found': bool(formal_index),
        'index_source': index_source,
        'index_page': index_page,
        'index': eff_index,
        'drawing_list': drawing_list,
        'drawing_list_source': drawing_list_source,
        'present': present,
        'missing': missing,
        'unlisted': unlisted,
        'disciplines': disc_counts,
        'content': content,
        'content_labels': _CONTENT_LABEL,
        'red_flags': flags,
    }


if __name__ == '__main__':  # pragma: no cover — quick manual smoke test
    import sys
    import json
    pdf = sys.argv[1]
    print(json.dumps(build_verification(pdf, []), indent=2))
