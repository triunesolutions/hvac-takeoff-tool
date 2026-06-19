"""
page_classifier.py — Stage 2 of the upload pipeline.

Classify every page in a blueprint PDF as one of:
  cover, schedule, legend, mechanical_plan, roof_plan, details,
  riser_diagram, air_balance, other

Used by the stamper to skip non-plan pages (eliminates ~80% of phantom
detections per CLAUDE.md section 19.5) and by downstream stages to know
which pages to scan for schedules vs equipment.

Heuristics, in order:
  1. Keyword scan of the page's text layer.
  2. Table-density check (many small ruled tables → schedule).
  3. Detection-density check (handed in from YOLO output if available).
  4. Default to 'other'.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


PAGE_TYPES = (
    'cover', 'schedule', 'legend', 'notes', 'mechanical_plan', 'roof_plan',
    'details', 'riser_diagram', 'air_balance', 'other',
)

# Pages we DO NOT want to stamp equipment on (phantoms come from these)
NON_PLAN_TYPES = frozenset({'cover', 'schedule', 'legend', 'notes', 'details',
                            'riser_diagram', 'air_balance'})


def _type_from_title_block(title: str) -> Optional[str]:
    """Map a sheet's title-block TITLE to a page type. The title block is the
    most authoritative signal we have ("MECHANICAL SCHEDULES" → schedule,
    "HVAC ROOF PLAN" → roof_plan). Order matters: legend wins over notes so
    "GENERAL NOTES AND LEGEND" reads as legend; schedule/detail/riser before
    plan; roof before floor."""
    u = (title or '').upper()
    if not u:
        return None
    if 'LEGEND' in u or 'SYMBOL' in u or 'ABBREV' in u:
        return 'legend'
    if 'SCHEDULE' in u:
        return 'schedule'
    if 'DETAIL' in u:
        return 'details'
    if 'RISER' in u or 'DIAGRAM' in u or 'ONE LINE' in u:
        return 'riser_diagram'
    if 'NOTES' in u:
        return 'notes'
    if 'PLAN' in u:
        return 'roof_plan' if 'ROOF' in u else 'mechanical_plan'
    if 'COVER' in u or 'TITLE SHEET' in u:
        return 'cover'
    return None

# Keywords that strongly indicate a page type. Matched case-insensitive,
# against the first ~5000 chars of the page text.
KEYWORDS = {
    'schedule': [
        'SCHEDULE', 'MANUFACTURER', 'MODEL', 'CFM', 'MCA', 'MOCP',
        'AIR BALANCE', 'EQUIPMENT SCHEDULE', 'UNIT TAG', 'AHU SCHEDULE',
    ],
    'legend': [
        'LEGEND', 'SYMBOL LEGEND', 'ABBREVIATIONS', 'GENERAL NOTES',
        'MECHANICAL SYMBOLS', 'KEYED NOTES',
    ],
    'cover': [
        'TITLE SHEET', 'COVER SHEET', 'DRAWING INDEX', 'SHEET INDEX',
        'PROJECT DIRECTORY', 'VICINITY MAP',
    ],
    'mechanical_plan': [
        'MECHANICAL PLAN', 'HVAC PLAN', 'MECHANICAL FLOOR PLAN',
        'HVAC FLOOR PLAN', 'AIR DISTRIBUTION PLAN',
    ],
    'roof_plan': [
        'ROOF PLAN', 'MECHANICAL ROOF PLAN', 'HVAC ROOF PLAN',
    ],
    'details': [
        'DETAILS', 'TYPICAL DETAILS', 'MECHANICAL DETAILS',
        'INSTALLATION DETAILS', 'MOUNTING DETAILS', 'SECTION ',
    ],
    'riser_diagram': [
        'RISER DIAGRAM', 'PIPING DIAGRAM', 'ONE LINE', 'SCHEMATIC',
    ],
    'air_balance': [
        'AIR BALANCE DATA', 'ROOM AIR BALANCE', 'CFM BALANCE',
    ],
}


@dataclass
class PageClassification:
    page: int
    type: str
    confidence: float           # 0..1
    evidence: list[str]         # what tipped us off
    text_word_count: int
    table_count: int
    is_plan: bool               # True iff equipment detection should run

    def to_dict(self) -> dict:
        return {
            'page': self.page,
            'type': self.type,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'text_word_count': self.text_word_count,
            'table_count': self.table_count,
            'is_plan': self.is_plan,
        }


def _count_tables_quick(pdf_path: Path, page_index: int) -> int:
    """Quick table count via pdfplumber (returns 0 if anything fails)."""
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_index >= len(pdf.pages):
                return 0
            return len(pdf.pages[page_index].find_tables() or [])
    except Exception:
        return 0


def classify_page(pdf_path: Path, page_index: int,
                  detection_count: Optional[int] = None) -> PageClassification:
    """Classify a single page.

    detection_count: if YOLO has already run, the number of detections on
    this page. Pages with many detections but no plan-keyword still get
    classified as plan; pages with zero detections look more like cover/details.
    """
    title_block = {}
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        text = page.get_text('text') or ''
        # Vector-path density: a real CAD plan typically has 5k-30k paths
        # with relatively little text. Schedules have thousands of words and
        # only a few hundred to a few thousand paths from grid lines.
        try:
            path_count = len(page.get_drawings())
        except Exception:
            path_count = 0
        # Authoritative signal: the sheet's own title block (sheet # + title).
        try:
            from doc_verification import read_title_block
            title_block = read_title_block(page) or {}
        except Exception:
            title_block = {}
    finally:
        doc.close()

    text_upper = text.upper()[:5000]
    word_count = len(text.split())
    table_count = _count_tables_quick(pdf_path, page_index)

    # ── Title-block override — the most reliable pre-detection signal ─────────
    # The sheet title states what the sheet IS, which beats keyword races and
    # the vector-density / sheet-number heuristics (those mislabel a Notes &
    # Legend sheet as a roof_plan when its linetype list mentions ROOF/PLAN).
    tb_type = _type_from_title_block(title_block.get('title', ''))
    if tb_type:
        is_plan = tb_type in ('mechanical_plan', 'roof_plan')
        # Safety net: trust YOLO over the title only when the title says a
        # non-plan type yet the page is dense with equipment (rare title misread
        # on an actual plan).
        if not is_plan and detection_count is not None and detection_count >= 20:
            pass  # fall through to the heuristic scorer below
        else:
            sheet = title_block.get('sheet', '')
            ev = f'title-block={sheet} "{title_block.get("title", "")}"'.strip()
            return PageClassification(
                page=page_index + 1,
                type=tb_type,
                confidence=0.95,
                evidence=[ev],
                text_word_count=word_count,
                table_count=table_count,
                is_plan=is_plan,
            )

    # Score each candidate type by keyword hits
    scores: dict[str, list[str]] = {t: [] for t in PAGE_TYPES}
    for ptype, kws in KEYWORDS.items():
        for kw in kws:
            if kw in text_upper:
                scores[ptype].append(kw)

    # M-series sheet number — strong signal that this is a mechanical plan.
    # Standard convention: M001 (title), M101 (floor plan), M201 (roof plan),
    # M501 (details), M601 (specs).  Pacific Palisades-style: MH3/MH4/MH5
    # (mechanical HVAC), M-400, etc.  We try the canonical M### form first,
    # then fall back to letter-prefix or hyphenated variants and route them
    # via keyword presence (since their numbering doesn't follow the
    # M000/M100/M200 buckets).
    # The page's actual sheet number always appears multiple times (title
    # block, footer, file metadata). Cross-references to OTHER sheets
    # (e.g. "SEE M-500 FOR DETAILS") appear once or twice. So when several
    # M-series numbers are present, pick the most frequent — that's
    # almost certainly THIS page's number, not a cross-reference.
    import re as _re
    from collections import Counter as _Cnt
    # Canonical form (M101, M201, M500) is the strongest signal.
    canonical_matches = _re.findall(r'\b(M[0-9]{3,4})\b', text_upper)
    if canonical_matches:
        sheet_id = _Cnt(canonical_matches).most_common(1)[0][0]
        sheet_num = int(sheet_id[1:])
        if sheet_num == 0 or sheet_id.startswith('M00') or sheet_num < 100:
            scores['cover'].append(f'sheet={sheet_id}')
        elif 100 <= sheet_num < 200:
            scores['mechanical_plan'].append(f'sheet={sheet_id}-floor-plan')
        elif 200 <= sheet_num < 300:
            scores['roof_plan'].append(f'sheet={sheet_id}-roof-plan')
        elif 300 <= sheet_num < 500:
            scores['mechanical_plan'].append(f'sheet={sheet_id}-misc-plan')
        elif sheet_num >= 500:
            scores['details'].append(f'sheet={sheet_id}-details')
    else:
        # Fallback: letter-prefixed or hyphenated M-series (MH3, ME4, M-201).
        # Also pick most frequent — title-block sheet number wins over
        # body-text cross-references.
        alt_matches = _re.findall(r'\b(M[A-Z]?-?[0-9]{1,4})\b', text_upper)
        if alt_matches:
            # Prefer the most frequent; tie-break by preferring plan-range
            # numbers (100-499) over details/specs (500+).
            counts = _Cnt(alt_matches)
            def _rank(item):
                tag, n = item
                # Extract numeric part
                digits = _re.sub(r'\D', '', tag)
                num = int(digits) if digits else 0
                in_plan_range = 100 <= num < 500
                return (n, 1 if in_plan_range else 0)
            sheet_id = max(counts.items(), key=_rank)[0]
            has_roof = 'ROOF' in text_upper
            has_plan = 'PLAN' in text_upper
            has_sched = 'SCHEDULE' in text_upper and not has_plan
            has_det = 'DETAILS' in text_upper and not has_plan
            # IMPORTANT: include "plan" in roof/floor labels so the
            # has_plan_sheet_signal check downstream recognizes them and
            # suppresses the dense-tables boost on the schedule bucket.
            if has_roof and has_plan:
                scores['roof_plan'].append(f'sheet={sheet_id}-alt-roof-plan')
            elif has_plan:
                scores['mechanical_plan'].append(f'sheet={sheet_id}-alt-floor-plan')
            elif has_sched:
                scores['schedule'].append(f'sheet={sheet_id}-alt-schedule')
            elif has_det:
                scores['details'].append(f'sheet={sheet_id}-alt-details')

    # Schedule boost — require BOTH many tables AND a schedule-type keyword
    # OR a very dense table layout (>=5 tables) regardless of keywords.
    # Just having 3 tables is too permissive — most floor plans have 2-4
    # small callout boxes that pdfplumber sees as tables. The M-series
    # sheet check above must dominate, so we drop this if the sheet number
    # already classified the page as a plan.
    has_plan_sheet_signal = any(
        s.startswith('sheet=') and ('plan' in s)
        for s in scores['mechanical_plan'] + scores['roof_plan']
    )
    if not has_plan_sheet_signal:
        if table_count >= 5:
            scores['schedule'].append(f'dense-tables={table_count}')
        elif table_count >= 3 and scores['schedule']:
            scores['schedule'].append(f'tables+kw={table_count}')

    # Mechanical plan boost — many detections is the strongest signal for a plan
    if detection_count is not None:
        if detection_count >= 10:
            scores['mechanical_plan'].append(f'detections={detection_count}')
        elif detection_count >= 3 and table_count <= 4 and word_count < 100:
            scores['mechanical_plan'].append(f'few-detections-no-text={detection_count}')

    # Air-balance is a kind of schedule — keep both signals
    if 'AIR BALANCE' in text_upper:
        scores['air_balance'].append('AIR BALANCE')
        if not scores['schedule']:
            scores['schedule'].append('AIR BALANCE (schedule-like)')

    # Vector-path density: CAD plan sheets are mostly geometry, very little
    # text. Schedules are the opposite. If a page has >=8000 paths AND the
    # word count is sparse (≤ ~800 words AND fewer than 10 words per path),
    # it's a plan even if pdfplumber sees its title-block grid as "tables".
    # This catches PDFs like Pacific Palisades where the sheet number uses
    # non-standard prefixes (MH3/MH4/MH5) so the keyword route misses it.
    if path_count >= 8000 and word_count <= 800:
        paths_per_word = path_count / max(1, word_count)
        if paths_per_word >= 10:
            roof_signal = 'ROOF' in text_upper
            ptype = 'roof_plan' if roof_signal else 'mechanical_plan'
            scores[ptype].append(
                f'vector-density={path_count}paths/{word_count}words'
            )

    # Definite-plan override. EITHER strong plan signal alone is enough:
    #
    #   (a) plan-range sheet number — the page's OWN dominant M-series number is
    #       a plan bucket (M1xx floor, M2xx roof, M3xx misc-plan). This is the
    #       most authoritative pre-detection signal we have. Floor plans very
    #       commonly embed air-device schedules and keyed-note blocks, so a pure
    #       keyword race lets 'schedule'/'legend' bury the plan and silently drop
    #       the whole sheet (e.g. Busy Bees M101 carries 150+ embedded schedule
    #       tables). Dropping a plan misses ALL its equipment — far costlier than
    #       the few phantoms a kept schedule-heavy page adds, which the schedule-
    #       class filter, reconciliation, and QA gating all catch downstream.
    #   (b) vector-path density — mostly geometry, little text (catches non-
    #       standard sheet numbers like Pacific Palisades MH3/MH4/MH5).
    #
    # The sheet number is taken as the MOST FREQUENT M-number on the page, so a
    # cross-reference like "SEE M-101 FOR PLAN" on a details sheet won't trip
    # this — the details sheet's own number dominates.
    plan_sheet_signal = any(
        s.startswith('sheet=') and 'plan' in s
        for s in scores['mechanical_plan'] + scores['roof_plan']
    )
    vector_density_signal = any(
        s.startswith('vector-density=')
        for s in scores['mechanical_plan'] + scores['roof_plan']
    )
    if plan_sheet_signal or vector_density_signal:
        plan_bucket = 'roof_plan' if scores['roof_plan'] else 'mechanical_plan'
        trigger = 'sheet-number' if plan_sheet_signal else 'vector-density'
        return PageClassification(
            page=page_index + 1,
            type=plan_bucket,
            confidence=1.0,
            evidence=scores[plan_bucket] + [f'definite-plan-override({trigger})'],
            text_word_count=word_count,
            table_count=table_count,
            is_plan=True,
        )

    # Pick winner by score length (= number of unique keyword hits)
    ranked = sorted(scores.items(), key=lambda kv: -len(kv[1]))
    top_type, top_evidence = ranked[0]
    runner_type, runner_evidence = ranked[1] if len(ranked) > 1 else ('other', [])

    # If no keywords hit at all, fall back based on word count + detections
    if not top_evidence:
        if word_count < 30 and (detection_count or 0) > 0:
            top_type = 'mechanical_plan'
            top_evidence = ['no-text-but-has-detections']
        elif word_count < 30:
            top_type = 'other'
            top_evidence = ['no-text']
        else:
            top_type = 'other'
            top_evidence = ['no-keyword-match']

    # Confidence — fraction of "win" vs runner-up
    n_top = len(top_evidence)
    n_runner = len(runner_evidence)
    conf = 1.0 if n_runner == 0 else min(1.0, max(0.0, (n_top - n_runner) / max(1, n_top)))

    # Safety override: if a page has many detections, NEVER classify it as
    # cover/legend/riser_diagram. These types are inherently low-detection.
    # If the AI sees 10+ pieces of equipment, the page is functionally a plan.
    if detection_count is not None and detection_count >= 10:
        if top_type in ('cover', 'legend', 'riser_diagram'):
            top_type = 'mechanical_plan'
            top_evidence = [f'override-by-detection-count={detection_count}'] + top_evidence

    # Stronger override: if a page has 20+ equipment detections it is almost
    # certainly a floor or roof plan, even if the text suggests "schedule".
    # CAD-export PDFs often have many small ruled boxes (callouts, room labels)
    # that pdfplumber misreads as tables — but real schedule pages have only
    # a handful of visual symbol references, never 20+ detected equipment.
    if detection_count is not None and detection_count >= 20:
        if top_type in ('schedule', 'air_balance', 'details', 'other'):
            top_type = 'mechanical_plan'
            top_evidence = [f'override-many-detections={detection_count}'] + top_evidence

    return PageClassification(
        page=page_index + 1,
        type=top_type,
        confidence=conf,
        evidence=top_evidence,
        text_word_count=word_count,
        table_count=table_count,
        is_plan=top_type in ('mechanical_plan', 'roof_plan'),
    )


def classify_pdf(pdf_path: Path,
                 detections_per_page: Optional[dict[int, int]] = None
                ) -> list[PageClassification]:
    """Classify every page in the PDF.

    detections_per_page: optional {1-based page no: detection count}.
    """
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    doc.close()

    out = []
    for i in range(n):
        det_n = (detections_per_page or {}).get(i + 1)
        out.append(classify_page(pdf_path, i, detection_count=det_n))
    return out


def plan_pages(classifications: list[PageClassification]) -> set[int]:
    """1-based page numbers that are mechanical or roof plans."""
    return {c.page for c in classifications if c.is_plan or c.type == 'other'
            # 'other' is permissive — if we're not sure, we don't strip the page
            }


def non_plan_pages(classifications: list[PageClassification]) -> set[int]:
    """1-based page numbers that should NOT be stamped (phantom sources)."""
    return {c.page for c in classifications if c.type in NON_PLAN_TYPES}


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--detections', help='Optional detections.json to incorporate counts')
    args = ap.parse_args()

    dets_per_page = None
    if args.detections:
        d = json.loads(Path(args.detections).read_text(encoding='utf-8'))
        dets_per_page = {int(k): len(v) for k, v in d.get('pages', {}).items()}

    classifications = classify_pdf(Path(args.pdf), detections_per_page=dets_per_page)
    print(f'{"page":>4} {"type":18s} {"conf":>4s} {"words":>6s} {"tables":>6s} {"plan?":>5s}  evidence')
    print('-' * 100)
    for c in classifications:
        flag = 'YES' if c.is_plan else ('NO' if c.type in NON_PLAN_TYPES else '?')
        ev = ', '.join(c.evidence[:4])
        print(f'{c.page:>4} {c.type:18s} {c.confidence:>4.2f} {c.text_word_count:>6d} {c.table_count:>6d} {flag:>5s}  {ev[:80]}')
