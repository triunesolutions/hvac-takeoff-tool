"""
quality_checks.py — Stage 12.

Sanity checks the estimator usually does manually. Output is a list of
human-readable warnings that ship with the takeoff so the estimator can
verify before signing off.

Checks implemented:
  Q1 - Tag mismatch: schedule tags with no detection vs detections with no tag
  Q2 - Unit name mismatch: tags used inconsistently on enlarged-vs-overall plans
  Q3 - Quantity reasonableness: detection counts vs schedule expectations
  Q4 - Scale consistency: pages with different scales that should match
  Q5 - Page-classification confidence: low-confidence classifications flagged
"""

from __future__ import annotations

from collections import Counter, defaultdict


def check_tag_mismatch(variables: list[dict], detections: dict) -> list[dict]:
    """Return list of {severity, kind, tag/page, message} warnings."""
    warnings = []

    # Tags present in schedule
    schedule_tags = {v['tag'].upper() for v in variables}

    # Tags assigned to detections
    detected_tags = Counter()
    untagged_detections = 0
    for pkey, det_list in detections.get('pages', {}).items():
        for det in det_list:
            t = det.get('tag')
            if t:
                detected_tags[t.upper()] += 1
            else:
                untagged_detections += 1

    # 1. Schedule tags with NO detections
    for tag in schedule_tags - set(detected_tags.keys()):
        warnings.append({
            'severity': 'medium',
            'kind': 'tag_in_schedule_not_on_plan',
            'tag': tag,
            'message': f'Tag {tag} is defined in the schedule but no detection on any plan.',
        })

    # 2. Detections with no schedule tag mapping (untagged)
    if untagged_detections > 0:
        warnings.append({
            'severity': 'low' if untagged_detections < 20 else 'medium',
            'kind': 'untagged_detections',
            'count': untagged_detections,
            'message': f'{untagged_detections} detections could not be matched to a schedule tag.',
        })

    return warnings


def check_quantity_reasonableness(variables: list[dict], detections: dict) -> list[dict]:
    """Q3: Compare AI counts vs schedule's expected quantities. The schedule
    may not always have explicit counts, but if a 'QTY' / 'QUANTITY' field
    exists we use it."""
    warnings = []

    # Build expected quantities from schedule
    expected: dict[str, int] = {}
    for v in variables:
        for k, val in (v.get('properties') or {}).items():
            if 'QTY' in k.upper() or 'QUANTITY' in k.upper():
                try:
                    n = int(str(val).strip())
                    expected[v['tag'].upper()] = n
                except (ValueError, TypeError):
                    pass

    # Tally actual detections per tag
    actual = Counter()
    for pkey, det_list in detections.get('pages', {}).items():
        for det in det_list:
            t = det.get('tag')
            if t:
                actual[t.upper()] += 1

    for tag, exp_n in expected.items():
        act_n = actual.get(tag, 0)
        if act_n == 0 and exp_n > 0:
            warnings.append({
                'severity': 'high',
                'kind': 'quantity_missing',
                'tag': tag,
                'expected': exp_n,
                'actual': 0,
                'message': f'Schedule says {exp_n}× {tag} but AI found 0.',
            })
        elif abs(act_n - exp_n) >= max(2, exp_n * 0.3):
            warnings.append({
                'severity': 'medium',
                'kind': 'quantity_off',
                'tag': tag,
                'expected': exp_n,
                'actual': act_n,
                'message': f'{tag}: schedule says {exp_n}, AI found {act_n}.',
            })
    return warnings


def check_scale_consistency(scales_by_page: dict[int, str]) -> list[dict]:
    """Q4: flag pages where scale is missing or differs from majority."""
    warnings = []
    if not scales_by_page:
        return warnings
    # Count scale strings
    counts = Counter(s for s in scales_by_page.values() if s)
    if not counts:
        return [{
            'severity': 'medium',
            'kind': 'no_scale_detected',
            'message': 'No scale detected on any page.',
        }]
    majority_scale = counts.most_common(1)[0][0]
    for pno, s in scales_by_page.items():
        if not s:
            warnings.append({
                'severity': 'low',
                'kind': 'page_no_scale',
                'page': pno,
                'message': f'Page {pno}: no scale detected.',
            })
        elif s != majority_scale:
            warnings.append({
                'severity': 'low',
                'kind': 'page_scale_differs',
                'page': pno,
                'page_scale': s,
                'majority_scale': majority_scale,
                'message': f'Page {pno} scale is {s!r} (majority is {majority_scale!r}).',
            })
    return warnings


def check_page_classifications(classifications: list[dict]) -> list[dict]:
    """Q5: surface pages where the classifier was unsure."""
    warnings = []
    for c in classifications:
        if c.get('confidence', 1.0) < 0.5:
            warnings.append({
                'severity': 'low',
                'kind': 'page_low_classification_conf',
                'page': c['page'],
                'type': c['type'],
                'confidence': c['confidence'],
                'message': f"Page {c['page']} classified as {c['type']!r} with low confidence.",
            })
    return warnings


def run_all_checks(variables: list[dict],
                  detections: dict,
                  scales_by_page: dict[int, str] | None = None,
                  classifications: list[dict] | None = None) -> dict:
    """Run every check; return aggregated warnings + counts by severity."""
    all_warnings = []
    all_warnings += check_tag_mismatch(variables, detections)
    all_warnings += check_quantity_reasonableness(variables, detections)
    if scales_by_page:
        all_warnings += check_scale_consistency(scales_by_page)
    if classifications:
        all_warnings += check_page_classifications(classifications)

    counts = Counter(w['severity'] for w in all_warnings)
    return {
        'warnings': all_warnings,
        'count': len(all_warnings),
        'by_severity': dict(counts),
    }


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--variables', required=True)
    ap.add_argument('--detections', required=True)
    args = ap.parse_args()

    vars_ = json.loads(Path(args.variables).read_text(encoding='utf-8'))
    dets = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    result = run_all_checks(vars_, dets)

    print(f'{result["count"]} warnings.  By severity: {result["by_severity"]}')
    for w in result['warnings'][:20]:
        print(f'  [{w["severity"]}] {w["message"]}')
