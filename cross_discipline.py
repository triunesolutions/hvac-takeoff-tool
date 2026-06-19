"""
cross_discipline.py — Stage 6.

Per Deck 1 slide 4, HVAC equipment tags are sometimes shown only on plumbing
or piping plans instead of mechanical plans. This module scans EVERY page's
text layer for any tag from the schedule and flags pages where mechanical
tags appear but no mechanical detection landed there.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


def scan_pages_for_tags(pdf_path: Path, tags: list[str]) -> dict[str, list[int]]:
    """Return {tag: [page_numbers_where_it_appears]}.
    Matches as whole words; case-insensitive.
    """
    if not tags:
        return {}

    # Build a single regex for efficiency
    escaped = [re.escape(t) for t in tags]
    pattern = re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', re.IGNORECASE)

    found: dict[str, list[int]] = {t.upper(): [] for t in tags}

    doc = fitz.open(str(pdf_path))
    try:
        for pno in range(doc.page_count):
            text = doc[pno].get_text('text') or ''
            for m in pattern.finditer(text):
                tag_norm = m.group(0).upper()
                if tag_norm in found:
                    if (pno + 1) not in found[tag_norm]:
                        found[tag_norm].append(pno + 1)
    finally:
        doc.close()

    return found


def find_orphan_tags(pdf_path: Path, schedule_tags: list[str],
                     detected_tags_by_page: dict[int, set[str]]) -> list[dict]:
    """For each schedule tag, find any page that mentions it AND has no
    matching detection. These are candidates for cross-discipline placement
    (the tag is on a plumbing/piping sheet, not the mechanical sheet).

    detected_tags_by_page: {page_no: {tag1, tag2, ...}} — tags the AI placed
                          via Stage 9 tag inference.
    """
    appearances = scan_pages_for_tags(pdf_path, schedule_tags)
    orphans = []
    for tag in schedule_tags:
        tag_u = tag.upper()
        pages_with_tag = appearances.get(tag_u, [])
        pages_with_detection = [p for p, ts in detected_tags_by_page.items()
                                if tag_u in ts]
        if pages_with_tag and not pages_with_detection:
            orphans.append({
                'tag': tag_u,
                'appears_on_pages': pages_with_tag,
                'detected_on_pages': pages_with_detection,
                'severity': 'orphan' if not pages_with_detection else 'cross_discipline',
            })
        elif pages_with_tag and pages_with_detection:
            mismatch = set(pages_with_tag) - set(pages_with_detection)
            if mismatch:
                orphans.append({
                    'tag': tag_u,
                    'appears_on_pages': pages_with_tag,
                    'detected_on_pages': pages_with_detection,
                    'unmatched_text_pages': sorted(mismatch),
                    'severity': 'cross_discipline',
                })
    return orphans


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--variables', help='variables.json — uses tags from here')
    args = ap.parse_args()

    if not args.variables:
        print('Provide --variables to know which tags to look for.')
        raise SystemExit(1)

    vars_ = json.loads(Path(args.variables).read_text(encoding='utf-8'))
    tags = sorted({v['tag'] for v in vars_})
    print(f'Scanning {len(tags)} schedule tags across PDF...')

    appearances = scan_pages_for_tags(Path(args.pdf), tags)
    found_count = sum(1 for v in appearances.values() if v)
    print(f'Tags found at least once: {found_count} / {len(tags)}')
    for tag, pages in appearances.items():
        if pages:
            print(f'  {tag:8s}  pages: {pages}')
