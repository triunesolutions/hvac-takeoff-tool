"""
sheet_filter.py

Find the M-series **plan** pages in a multi-discipline drawing set.

Why this exists:
  Architectural drawing sets ship across many disciplines:
    A   architectural        S   structural         C   civil
    M   mechanical (HVAC)    E   electrical         P   plumbing
    L   landscape            T   telecom            FP  fire protection
  Only the M-series sheets contain HVAC equipment, and even within
  M-series only PLAN sheets matter (not legend / schedule / details
  / notes / controls / schematic / riser).

  Filtering up-front turns a 38-page architectural set into the 5-8
  sheets we actually need, saves ~80% of inference time, and eliminates
  the #1 phantom-detection source (CLAUDE.md §19.5).

This module exposes:
  detect_sheet(page) -> {sheet_number, sheet_title, discipline, is_plan}
  is_m_series(discipline)
  pick_m_plan_pages(pdf_path) -> list[(page_idx, sheet_number, sheet_title)]

Usage from CLI for debugging:
    python sheet_filter.py "<plan.pdf>"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz


# ---- OCR fallback (lazy) ----

_OCR_READER = None
_OCR_AVAILABLE: Optional[bool] = None
_OCR_DPI = 300                       # higher than render DPI — small crop, want detail
_TITLE_BLOCK_W_FRAC = 0.20           # crop right 20% of page
_TITLE_BLOCK_H_FRAC = 0.15           # crop bottom 15% of page
_OCR_CONF_MIN = 0.35                 # accept a match if EasyOCR confidence >= this


def _get_ocr_reader():
    """Lazy-load EasyOCR. Returns None if EasyOCR isn't importable."""
    global _OCR_READER, _OCR_AVAILABLE
    if _OCR_AVAILABLE is False:
        return None
    if _OCR_READER is not None:
        return _OCR_READER
    try:
        import easyocr
        _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
        _OCR_AVAILABLE = True
        return _OCR_READER
    except Exception:
        _OCR_AVAILABLE = False
        return None


def detect_sheet_number_ocr(page) -> tuple[Optional[str], Optional[str]]:
    """Render the bottom-right title-block area and OCR for a sheet number.

    Used only when the text-layer extractor finds nothing.
    """
    reader = _get_ocr_reader()
    if reader is None:
        return None, None

    page_w_pt = page.rect.width
    page_h_pt = page.rect.height
    crop_w_pt = page_w_pt * _TITLE_BLOCK_W_FRAC
    crop_h_pt = page_h_pt * _TITLE_BLOCK_H_FRAC
    crop_rect = fitz.Rect(
        page_w_pt - crop_w_pt,
        page_h_pt - crop_h_pt,
        page_w_pt,
        page_h_pt,
    )

    matrix = fitz.Matrix(_OCR_DPI / 72.0, _OCR_DPI / 72.0)
    pix = page.get_pixmap(matrix=matrix, clip=crop_rect, annots=False)

    try:
        import numpy as np
    except Exception:
        return None, None

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]

    try:
        results = reader.readtext(img, detail=1)
    except Exception:
        return None, None

    # Score every OCR token against the sheet-number regex; pick the best one.
    best = None
    for entry in results:
        # easyocr returns (bbox, text, conf)
        if len(entry) < 3:
            continue
        _, text, conf = entry[0], entry[1], float(entry[2])
        text_clean = (text or '').strip().upper()
        if not text_clean:
            continue
        # Normalise common OCR errors that pop up in sheet numbers:
        # O → 0 (only when adjacent to digits), and remove embedded spaces
        normalized = text_clean.replace(' ', '')
        m = SHEET_NUMBER_RE.match(normalized)
        if not m:
            # Try O→0 normalization (e.g. "MOO1" -> "M001")
            o_to_0 = re.sub(r'(?<=[A-Z])O(?=\d)|(?<=\d)O(?=\d|$)', '0', normalized)
            m = SHEET_NUMBER_RE.match(o_to_0)
            if m:
                normalized = o_to_0
        if not m:
            continue
        disc = m.group('disc')
        if len(disc) == 1 and len(m.group('num')) <= 1:
            continue
        if conf < _OCR_CONF_MIN:
            continue
        # Prefer known disciplines, then high confidence
        score = conf + (0.5 if disc in KNOWN_DISCIPLINES else 0.0)
        if best is None or score > best[0]:
            best = (score, normalized, disc)

    if best is None:
        return None, None
    return best[1], best[2]


# ---- Discipline taxonomy ----

# Mechanical = HVAC. We include compound prefixes commonly used when a
# sheet covers mechanical + another trade.
M_DISCIPLINES = {'M', 'MEP', 'ME', 'MP', 'MH', 'HV'}

# Anything else we know — kept for reporting + future plumbing/electrical work.
KNOWN_DISCIPLINES = {
    'A', 'AD', 'AS',                      # architectural
    'S', 'SD',                            # structural
    'C', 'CD', 'CS',                      # civil / site
    'L', 'LP', 'LS',                      # landscape
    'P', 'PD', 'PP', 'PS',                # plumbing
    'E', 'ED', 'EE', 'EL', 'ES', 'EM',    # electrical
    'T', 'TC', 'TS',                      # telecom
    'FP', 'FA', 'FS',                     # fire protection / alarm
    'I', 'IT', 'AV',                      # IT / audio-visual
    'G',                                  # general / cover
} | M_DISCIPLINES


# ---- Patterns ----

# Sheet number: discipline letters + optional separator + digits + optional decimal/suffix.
# Examples matched:  M001   M-101   M0.01   M1.1   M101.1   M101A   MEP-201   HV-101
SHEET_NUMBER_RE = re.compile(
    r'^\s*(?P<disc>[A-Z]{1,3})[\s\-.]?\s*(?P<num>\d{1,3}(?:\.\d{1,3})?[A-Z]?)\s*$',
)

# Phrases that mark an M-series sheet as something OTHER than a floor plan.
# Order: check before is_plan keyword check so "MECHANICAL SCHEDULE" is skipped
# even though it contains "MECHANICAL".
NON_PLAN_TITLE_MARKERS = (
    'LEGEND',
    'SCHEDULE',
    'DETAILS', 'DETAIL ', 'DETAIL,',
    'NOTES',
    'CONTROLS', 'CONTROL ',
    'SCHEMATIC',
    'DIAGRAM',
    'RISER',
    'ISOMETRIC',
    'SYMBOLS', 'SYMBOL ',
    'ABBREVIATION',
    'TITLE SHEET', 'COVER SHEET',
    'SHEET INDEX', 'DRAWING INDEX',
    'SPECIFICATION', 'SPECIFICATIONS',
)

# Affirmative plan markers — if any of these appears in the title, we treat
# the sheet as a plan regardless of other content.
PLAN_TITLE_MARKERS = (
    'PLAN',  # 'MECHANICAL PLAN', 'CEILING PLAN', 'HVAC PLAN', 'ROOF PLAN', etc.
)


# ---- Data class ----

@dataclass
class SheetInfo:
    page_idx: int                       # 0-indexed
    sheet_number: Optional[str]         # 'M101' or None if not detected
    sheet_title: Optional[str]
    discipline: Optional[str]           # 'M', 'A', 'E', ...
    is_plan: bool                       # True if we think it's a floor plan
    reason: str                         # human-readable why we decided

    def as_dict(self) -> dict:
        return {
            'page': self.page_idx + 1,
            'sheet_number': self.sheet_number,
            'sheet_title': self.sheet_title,
            'discipline': self.discipline,
            'is_plan': self.is_plan,
            'reason': self.reason,
        }


# ---- Core extraction ----

def _spans(page) -> list[dict]:
    """Flatten page.get_text('dict') into a list of {text, bbox, size}."""
    out = []
    d = page.get_text('dict')
    for block in d.get('blocks', []):
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                txt = (span.get('text') or '').strip()
                if not txt:
                    continue
                bbox = span.get('bbox')
                size = float(span.get('size') or 0)
                out.append({'text': txt, 'bbox': bbox, 'size': size})
    return out


def _in_title_block(span, page_w, page_h) -> bool:
    """Title block typically lives in the right edge or bottom-right corner."""
    bbox = span['bbox']
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    # Right strip (75-100% width, any height) OR bottom strip (any width, 80-100% height)
    right_strip = cx >= 0.70 * page_w
    bottom_strip = cy >= 0.75 * page_h
    return right_strip or bottom_strip


def detect_sheet_number(page) -> tuple[Optional[str], Optional[str]]:
    """Return (sheet_number, discipline). Tries title-block area first."""
    page_w, page_h = page.rect.width, page.rect.height
    spans = _spans(page)

    title_block = [s for s in spans if _in_title_block(s, page_w, page_h)]

    # Title-block candidates: matches pattern + above-median font size
    candidates = []
    pool = title_block or spans
    sizes = sorted(s['size'] for s in pool)
    median_size = sizes[len(sizes) // 2] if sizes else 0
    for s in pool:
        m = SHEET_NUMBER_RE.match(s['text'])
        if not m:
            continue
        disc = m.group('disc')
        # Discard obvious tag-like things: 1-letter prefix + 1-digit number is too short
        # and very common as a callout (e.g. "A1" inside a detail bubble).
        if len(disc) == 1 and len(m.group('num')) <= 1:
            continue
        # Reject standalone numbers that are too small to be a sheet number
        candidates.append((s, disc, m.group('num')))

    if not candidates:
        return None, None

    # Prefer larger fonts (typographic emphasis), then prefer known disciplines
    def score(c):
        s, disc, _ = c
        size_score = s['size']
        disc_bonus = 5.0 if disc in KNOWN_DISCIPLINES else 0.0
        # Boost candidates that look like ours (M-family)
        m_bonus = 3.0 if disc in M_DISCIPLINES else 0.0
        # Penalize tiny fonts barely above noise floor
        if s['size'] < median_size:
            size_score *= 0.5
        return size_score + disc_bonus + m_bonus

    best = max(candidates, key=score)
    span, disc, num = best
    sheet_no = f'{disc}{num}' if disc else num
    # Reconstruct a clean form preserving the original separator pattern
    raw = span['text']
    return raw.strip(), disc


def detect_sheet_title(page) -> Optional[str]:
    """Find the sheet title — usually next to the sheet number in the title block."""
    page_w, page_h = page.rect.width, page.rect.height
    spans = _spans(page)
    title_block = [s for s in spans if _in_title_block(s, page_w, page_h)]

    # Heuristic: ALL-CAPS, > 6 chars, < 60 chars, not just digits
    def looks_like_title(t: str) -> bool:
        if not (6 <= len(t) <= 60):
            return False
        if not any(c.isalpha() for c in t):
            return False
        # Reject hyperbole-y short labels: "SCALE", "DATE", "PROJECT NO."
        if t.upper() in ('SCALE', 'DATE', 'PROJECT NO.', 'SHEET', 'DRAWN BY', 'CHECKED BY'):
            return False
        # Mostly uppercase letters
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(c.isupper() for c in letters) / len(letters)
        return uppercase_ratio >= 0.7

    # Group near-y spans, pick the largest plausible title
    titles = [s for s in title_block if looks_like_title(s['text'])]
    if not titles:
        return None
    # Score by font size — title is typically the tallest title-block text
    titles.sort(key=lambda s: -s['size'])
    return titles[0]['text']


def is_m_series(discipline: Optional[str]) -> bool:
    if not discipline:
        return False
    return discipline.upper() in M_DISCIPLINES


def is_plan_sheet(sheet_title: Optional[str]) -> tuple[bool, str]:
    """Return (is_plan, reason)."""
    if not sheet_title:
        return True, 'no title found — keeping by default'
    t = sheet_title.upper()
    for marker in NON_PLAN_TITLE_MARKERS:
        if marker in t:
            return False, f'title contains non-plan marker {marker!r}'
    for marker in PLAN_TITLE_MARKERS:
        if marker in t:
            return True, f'title contains plan marker {marker!r}'
    return True, 'no explicit non-plan markers — keeping'


def plan_by_sheet_number(sheet_no: Optional[str]) -> tuple[Optional[bool], str]:
    """Decide plan/non-plan from the M-series number's *series* (leading digit).

    Standard A/E numbering groups sheets by a leading "series" digit, in both
    the 3-digit form (M101) and the dotted form (M1.1):
        series 0  → cover / general / legend         → NOT a plan
        series 1  → floor plans                        → plan
        series 2  → roof / upper-level plans           → plan
        series 3–4→ enlarged / misc plans              → plan
        series 5+ → details, specs, schedules, controls→ NOT a plan
    So M101 and M1.1 both map to series 1 (plan); M501 and M5.2 both map to
    series 5 (details). This is the authoritative signal when the title text
    won't extract (common on CAD sets where the title is vector geometry).

    Handles: M101, M-201, M101A, M101.1 (canonical 3–4 digit) and M1.1, M2.11,
    M-2.1, M0.01 (dotted). Bare 1–2 digit numbers (M-01) are intentionally NOT
    classified — they're ambiguous (could be sheet #1 = a real plan) — so they
    return (None, '') and the caller keeps its title-based default. Never risk
    dropping a real plan on a guess.
    """
    if not sheet_no:
        return None, ''
    s = sheet_no.strip().upper()

    # Canonical 3–4 digit form (M101, M-201, M101A, M101.1) → series = hundreds.
    m = re.match(r'^[A-Z]{1,3}[\s\-.]?(\d{3,4})(?:[A-Z]|\.\d+)?$', s)
    if m:
        series = int(m.group(1)) // 100
    else:
        # Dotted form (M1.1, M2.11, M-2.1, M0.01) → series = integer part.
        m = re.match(r'^[A-Z]{1,3}[\s\-]?(\d{1,2})\.\d{1,3}[A-Z]?$', s)
        if not m:
            return None, ''
        series = int(m.group(1))

    if series == 0:
        return False, f'{sheet_no}: series 0 cover/general/legend by number'
    if series >= 5:
        return False, f'{sheet_no}: series {series} details/specs/schedule by number'
    return True, f'{sheet_no}: series {series} plan by number'


def detect_sheet(page, page_idx: int, use_ocr: bool = True) -> SheetInfo:
    sheet_no, disc = detect_sheet_number(page)
    title = detect_sheet_title(page)
    sheet_source = 'text'

    if not sheet_no and use_ocr:
        sheet_no, disc = detect_sheet_number_ocr(page)
        if sheet_no:
            sheet_source = 'ocr'

    if not sheet_no:
        # Fall back to page-text keyword scan to keep behavior compatible
        text = page.get_text('text').upper()
        if 'MECHANICAL PLAN' in text or 'HVAC PLAN' in text:
            return SheetInfo(
                page_idx=page_idx,
                sheet_number=None,
                sheet_title=title,
                discipline=None,
                is_plan=True,
                reason='no sheet number; page text mentions mechanical/hvac plan',
            )
        return SheetInfo(
            page_idx=page_idx,
            sheet_number=None,
            sheet_title=title,
            discipline=None,
            is_plan=False,
            reason='no sheet number (text or OCR) and no mechanical-plan keywords',
        )

    if not is_m_series(disc):
        return SheetInfo(
            page_idx=page_idx,
            sheet_number=sheet_no,
            sheet_title=title,
            discipline=disc,
            is_plan=False,
            reason=f'{disc}-series (not mechanical)',
        )

    is_plan, why = is_plan_sheet(title)
    num_plan, num_why = plan_by_sheet_number(sheet_no)
    # When the title gave only a default "keeping" verdict (unreadable or no
    # markers — common on CAD sets), refine with the sheet-number convention so
    # cover/details/schedule sheets aren't blindly kept and real plans aren't at
    # the mercy of a missing title.
    if 'keeping' in why:
        if num_plan is not None:
            is_plan, why = num_plan, num_why
    elif is_plan is False and num_plan is True:
        # Conflict: a title marker demoted this sheet, but the M-series NUMBER says
        # it's a core plan (series 1–4). Title text on CAD sets is frequently an
        # OCR'd room label / detail callout (e.g. "RISER ROOM" stamped on a floor
        # plan) rather than the real sheet title — far less reliable than the sheet
        # number. Cost asymmetry: dropping a real plan misses ALL its equipment,
        # while keeping a borderline page only risks a few phantoms that the
        # schedule-class filter + QA gating already catch. Trust the number.
        is_plan, why = True, f'{num_why} (number overrides title marker: {why})'
    src_note = f' (via {sheet_source})' if sheet_source == 'ocr' else ''
    return SheetInfo(
        page_idx=page_idx,
        sheet_number=sheet_no,
        sheet_title=title,
        discipline=disc,
        is_plan=is_plan,
        reason=f'{disc}-series{src_note}; {why}',
    )


# ---- Public API ----

def survey_pdf(pdf_path: str | Path, use_ocr: bool = True) -> list[SheetInfo]:
    """Return a SheetInfo for every page. Set use_ocr=False to skip the
    OCR fallback (faster, useful for debugging)."""
    doc = fitz.open(str(pdf_path))
    try:
        return [detect_sheet(doc[i], i, use_ocr=use_ocr) for i in range(doc.page_count)]
    finally:
        doc.close()


def pick_m_plan_pages(pdf_path: str | Path) -> list[int]:
    """Return 0-indexed list of M-series PLAN pages — for YOLO inference.

    Excludes legend / schedule / details / notes pages even when they're
    M-series, since YOLO would fire phantoms on those.

    Conservative fallback: if NO pages classify as M-plan, return everything
    (so we don't silently produce empty results on edge-case PDFs).
    """
    survey = survey_pdf(pdf_path)
    keep = [s.page_idx for s in survey if s.is_plan and is_m_series(s.discipline)]
    if not keep:
        keep = [s.page_idx for s in survey if s.is_plan]
    if not keep:
        keep = list(range(len(survey)))
    return keep


def pick_m_series_pages(pdf_path: str | Path) -> list[int]:
    """Return 0-indexed list of ALL M-family pages — for schedule parsing.

    Unlike pick_m_plan_pages, this INCLUDES legend / schedule / details /
    notes pages, because that's where equipment schedules actually live
    (e.g. M0.03 'SPLIT SYSTEM OUTDOOR UNIT SCHEDULE'). YOLO should skip
    them but schedule_parser needs them.

    Conservative fallback: if NO M-family pages found, return all pages.
    """
    survey = survey_pdf(pdf_path)
    keep = [s.page_idx for s in survey if is_m_series(s.discipline)]
    if not keep:
        keep = list(range(len(survey)))
    return keep


def survey_summary(pdf_path: str | Path) -> dict:
    """Single pass: return BOTH plan and full M-series page lists plus the
    raw survey, so callers don't have to invoke survey_pdf twice (saves
    duplicating OCR work).
    """
    survey = survey_pdf(pdf_path)
    m_plan = [s.page_idx for s in survey if s.is_plan and is_m_series(s.discipline)]
    m_all = [s.page_idx for s in survey if is_m_series(s.discipline)]
    if not m_plan:
        m_plan = [s.page_idx for s in survey if s.is_plan]
    if not m_plan:
        m_plan = list(range(len(survey)))
    if not m_all:
        m_all = list(range(len(survey)))
    return {
        'survey': survey,
        'm_plan_pages': m_plan,
        'm_series_pages': m_all,
    }


# ---- CLI for debugging ----

def main():
    ap = argparse.ArgumentParser(description='Survey M-series plan sheets in a PDF')
    ap.add_argument('pdf')
    ap.add_argument('--keep-only', action='store_true',
                    help='Print only the pages we would process')
    ap.add_argument('--no-ocr', action='store_true',
                    help='Skip OCR fallback (faster, but raster pages will be missed)')
    args = ap.parse_args()

    survey = survey_pdf(args.pdf, use_ocr=not args.no_ocr)
    if args.keep_only:
        for s in survey:
            if s.is_plan and is_m_series(s.discipline):
                print(f'{s.page_idx+1:>4d}  {s.sheet_number or "?":12s}  {s.sheet_title or ""}')
        return

    print(f'{"page":>4s}  {"sheet#":12s}  {"disc":5s}  {"plan?":5s}  title  ::  reason')
    print('-' * 100)
    for s in survey:
        flag = 'YES' if (s.is_plan and is_m_series(s.discipline)) else '-'
        title = (s.sheet_title or '')[:40]
        print(f'{s.page_idx+1:>4d}  {(s.sheet_number or "-"):12s}  {(s.discipline or "-"):5s}  '
              f'{flag:5s}  {title}  ::  {s.reason}')

    keepers = [s for s in survey if s.is_plan and is_m_series(s.discipline)]
    print(f'\nWould process {len(keepers)} / {len(survey)} pages.')


if __name__ == '__main__':
    main()
