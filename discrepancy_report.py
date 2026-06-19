"""
discrepancy_report.py — assembles a human-readable QA report for the estimator.

Pulls together outputs of:
  - page_classifier   (which pages are what)
  - quality_checks    (tag mismatches, quantity warnings, scale issues)
  - keynote_extractor (unreferenced notes, undefined callouts)
  - cross_discipline  (orphan tags)

Produces both:
  - Markdown report (for the SaaS download page)
  - Structured JSON (for programmatic consumption by the frontend)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def build_report(
    job_id: str,
    pdf_name: str,
    classifications: list[dict] | None = None,
    quality_warnings: dict | None = None,
    keynotes: dict | None = None,
    cross_discipline_orphans: list[dict] | None = None,
    variables: list[dict] | None = None,
    detections: dict | None = None,
    fill_stats: dict | None = None,
    enrich_stats: dict | None = None,
    project_info: dict | None = None,
    room_data: dict | None = None,
) -> dict:
    """Return {markdown, json_structured}."""

    lines = []
    structured = {}

    lines.append(f'# Takeoff QA Report')
    lines.append(f'')
    lines.append(f'**Job:** `{job_id}`  ·  **File:** `{pdf_name}`')
    lines.append(f'')

    # --- Project info ---
    if project_info:
        lines.append('## Project')
        for k in ('project', 'project_no', 'firm', 'address', 'date',
                  'sheet', 'description', 'scale', 'drawn_by', 'checked_by'):
            v = project_info.get(k)
            if v:
                lines.append(f'- **{k.replace("_", " ").title()}:** {v}')
        lines.append('')
        structured['project_info'] = project_info

    # --- Detection summary ---
    if detections:
        total = sum(len(v) for v in detections.get('pages', {}).values())
        by_cls = Counter()
        for v in detections.get('pages', {}).values():
            for d in v:
                by_cls[d.get('cls', '?')] += 1
        lines.append('## Detections')
        lines.append(f'- **{total} total** detections across {len(detections.get("pages", {}))} pages')
        lines.append(f'- Top classes:')
        for cls, n in by_cls.most_common(10):
            lines.append(f'  - `{n}× {cls}`')
        lines.append('')
        structured['detections'] = {'total': total, 'by_class': dict(by_cls)}

    # --- Page classification ---
    if classifications:
        lines.append('## Pages')
        lines.append(f'| Page | Type | Plan? | Detections |')
        lines.append(f'|------|------|-------|------------|')
        for c in classifications:
            plan_str = '✓' if c.get('is_plan') else ' '
            lines.append(f'| {c["page"]} | `{c["type"]}` | {plan_str} | — |')
        lines.append('')
        structured['classifications'] = classifications

    # --- Variables ---
    if variables is not None:
        lines.append('## Schedules')
        if variables:
            sched_groups = Counter()
            for v in variables:
                sched_groups[v.get('schedule_name', '?')] += 1
            lines.append(f'- **{len(variables)} tag(s) extracted** from {len(sched_groups)} schedule(s):')
            for sname, n in sched_groups.most_common():
                lines.append(f'  - `{n}× {sname}`')
        else:
            lines.append('- **No schedule variables extracted.** Drawing may have raster-only schedules — run OCR fallback.')
        lines.append('')
        structured['variables_count'] = len(variables)

    # --- Inferences (data filler) ---
    if fill_stats:
        lines.append('## Inferred Data')
        for k, v in fill_stats.items():
            lines.append(f'- `{k}`: **{v}**')
        lines.append('')
        structured['fill_stats'] = fill_stats

    # --- Enrichment (Deck 2 rules) ---
    if enrich_stats:
        lines.append('## Context Rules Applied (Deck 2)')
        for k, v in enrich_stats.items():
            if v:
                lines.append(f'- `{k}`: **{v}**')
        lines.append('')
        structured['enrich_stats'] = enrich_stats

    # --- Tag-by-tag breakdown ---
    # (We don't embed the full table — it's a separate downloadable file.
    # Show the top-discrepancy rows here so estimators see issues at a glance.)
    # Note: tag_report data is loaded from the artifact file in post_takeoff,
    # not passed in directly to keep this builder simple.

    # --- Per-room breakdown ---
    if room_data and room_data.get('breakdown'):
        lines.append('## Per-Room Breakdown')
        lines.append(f'- **{room_data.get("n_rooms_found", 0)} rooms** identified via OCR')
        lines.append('')
        breakdown = room_data['breakdown']
        # Top rooms by total equipment count
        room_totals = sorted(breakdown.items(),
                            key=lambda kv: -sum(kv[1].values()))
        lines.append('| Room | Total | Breakdown |')
        lines.append('|------|-------|-----------|')
        for room, counts in room_totals[:20]:
            total = sum(counts.values())
            details = ', '.join(f'{n}×{c}' for c, n in
                              sorted(counts.items(), key=lambda x: -x[1])[:5])
            lines.append(f'| `{room}` | {total} | {details} |')
        lines.append('')
        structured['room_breakdown'] = breakdown

    # --- Keynotes ---
    if keynotes:
        lines.append('## Keynotes')
        lines.append(f'- {keynotes.get("total_notes", 0)} keynote(s) defined')
        lines.append(f'- {keynotes.get("total_callouts", 0)} callout(s) placed on plans')
        if keynotes.get('unreferenced_notes'):
            lines.append(f'- ⚠️ **{len(keynotes["unreferenced_notes"])} note(s) defined but never referenced on plan:** {keynotes["unreferenced_notes"]}')
        if keynotes.get('undefined_callouts'):
            lines.append(f'- ⚠️ **{len(keynotes["undefined_callouts"])} callout(s) on plan reference notes that aren\'t defined:** {keynotes["undefined_callouts"]}')
        lines.append('')
        structured['keynotes'] = {k: keynotes[k] for k in ('total_notes', 'total_callouts',
                                                            'unreferenced_notes', 'undefined_callouts')}

    # --- Cross-discipline ---
    if cross_discipline_orphans:
        lines.append('## Cross-Discipline Tag Flags')
        lines.append('Tags that appear in the schedule but only on non-mechanical sheets:')
        for o in cross_discipline_orphans[:20]:
            lines.append(f'- `{o["tag"]}` — appears on pages {o.get("appears_on_pages")}, detected on {o.get("detected_on_pages") or "none"}')
        lines.append('')
        structured['cross_discipline'] = cross_discipline_orphans

    # --- Quality warnings ---
    if quality_warnings:
        lines.append('## Quality Warnings')
        by_sev = quality_warnings.get('by_severity', {})
        lines.append(f'- **{quality_warnings.get("count", 0)} warning(s)**  ·  {by_sev}')
        for w in quality_warnings.get('warnings', [])[:25]:
            sev = w.get('severity', '?')
            tag = w.get('tag') or w.get('page') or ''
            lines.append(f'- _[{sev}]_ {w.get("message", "")}')
        lines.append('')
        structured['quality_warnings'] = quality_warnings

    # --- Footer ---
    lines.append('---')
    lines.append('Generated by the HVAC AI Takeoff Tool · Bluebeam-stamped PDF available for direct correction.')

    return {
        'markdown': '\n'.join(lines),
        'json_structured': structured,
    }


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--pdf-name', default='unknown.pdf')
    ap.add_argument('--detections')
    ap.add_argument('--variables')
    ap.add_argument('--out', help='Output .md file (or .json with --json)')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    dets = json.loads(Path(args.detections).read_text(encoding='utf-8')) if args.detections else None
    vars_ = json.loads(Path(args.variables).read_text(encoding='utf-8')) if args.variables else None
    report = build_report(args.job, args.pdf_name, variables=vars_, detections=dets)

    out = args.out or f'qa_report_{args.job}.{"json" if args.json else "md"}'
    content = json.dumps(report['json_structured'], indent=2) if args.json else report['markdown']
    Path(out).write_text(content, encoding='utf-8')
    print(f'Wrote {out}')
