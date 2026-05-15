"""Survey project PDFs: characterize schedule style, tag conventions, page mix.
Output one JSON line per project to stdout."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import json
import argparse
import re
from pathlib import Path
from collections import Counter

import fitz
from schedule_parser import parse_pdf_schedules


def find_main_pdf(project_dir):
    # Prefer raw/, but fall back to labeled/ for projects whose raw/ wasn't
    # captured (Burlington, Hope Chapel, Perch, Optum, Planet Fitness, Vista
    # Murrieta). Labeled PDFs are the same plan content with team annotations.
    for sub in ('raw', 'labeled'):
        d = project_dir / sub
        if not d.exists():
            continue
        pdfs = [p for p in d.glob('*.pdf') if 'working' not in p.stem.lower()]
        # Skip pure takeoff PDFs only if there's a non-takeoff alternative
        non_takeoff = [p for p in pdfs if 'takeoff' not in p.stem.lower()]
        if non_takeoff:
            return max(non_takeoff, key=lambda p: p.stat().st_size)
        if pdfs:
            return max(pdfs, key=lambda p: p.stat().st_size)
    return None


def survey(pdf_path):
    out = {'pdf': str(pdf_path), 'size_mb': round(pdf_path.stat().st_size / 1024 / 1024, 2)}
    try:
        doc = fitz.open(pdf_path)
        out['pages'] = doc.page_count
        # Page title heuristics: collect first 80 chars of each page text
        titles = []
        for i in range(min(doc.page_count, 40)):
            t = doc[i].get_text().strip().split('\n')
            titles.append(' '.join(t[:3])[:100])
        out['titles_sample'] = titles[:5]
        doc.close()
    except Exception as e:
        out['error'] = f"fitz: {e}"
        return out

    try:
        schedules, marks, mark_details, _legend, _summary, variables = parse_pdf_schedules(str(pdf_path))
        out['schedule_count'] = len(schedules)
        out['schedule_names'] = sorted({s.get('schedule_name', '?') for s in (schedules or [])})[:10]
        out['tag_count'] = len(marks)
        out['sample_tags'] = list(marks)[:15]
        # Tag-prefix histogram (informs class coverage)
        prefixes = Counter()
        for t in marks:
            m = re.match(r'^([A-Z]+)', t)
            if m:
                prefixes[m.group(1)] += 1
        out['tag_prefixes'] = prefixes.most_common(10)
        out['variable_count'] = len(variables)
    except Exception as e:
        out['schedule_error'] = str(e)[:200]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('project_dir', help='Path to one project directory (parallel-friendly mode)')
    ap.add_argument('--max-mb', type=float, default=30.0,
                    help='Skip the PDF if larger than this many MB')
    args = ap.parse_args()

    pd = Path(args.project_dir)
    result = {'project': pd.name}
    pdf = find_main_pdf(pd)
    if pdf is None:
        result['error'] = 'no main PDF'
        print(json.dumps(result, default=str))
        return
    size_mb = pdf.stat().st_size / 1024 / 1024
    if size_mb > args.max_mb:
        result.update({'pdf': str(pdf), 'size_mb': round(size_mb, 2),
                       'skipped': f'> {args.max_mb} MB'})
        print(json.dumps(result, default=str))
        return
    info = survey(pdf)
    result.update(info)
    print(json.dumps(result, default=str))


if __name__ == '__main__':
    main()
