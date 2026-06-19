"""
validation_engine.py — schedule ↔ detection reconciliation (the closed loop).

The pipeline already parses every scheduled tag into TagVariables and already
detects equipment with YOLO. Until now those two halves never checked each
other: a drawing where YOLO missed half the units still produced an Excel that
*looked* complete. This module is the self-check that was missing.

For every equipment class it compares:
  • what the SCHEDULE says should be on the drawings (expected), against
  • what YOLO actually DETECTED (detected),
and emits an explicit verdict — match / under / over — plus a tag-level
cross-reference (which scheduled tags never showed up on a plan, and which
detected tags aren't in any schedule) and a project-level trust score.

It introduces no new dependencies and reuses class_normalization so both sides
are compared on the same canonical class names.

Design notes
------------
* COUNT reconciliation only applies to "unique-instance" classes — major
  equipment where one schedule tag corresponds to exactly one physical unit
  (CU-1, AHU-2, EF-3 …). For those, detected count *should* equal the number of
  distinct scheduled tags.
* Air devices (diffusers/grilles, AD-*) and ducts are NOT count-reconcilable:
  one tag (A1) legitimately repeats dozens of times across a floor plan, and the
  schedule never states how many. For those we report counts as INFO and rely on
  the tag presence cross-reference instead.
* The trust score is a transparent heuristic, not a calibrated probability. It is
  labelled as such everywhere it surfaces. Real calibration (isotonic regression
  on ground truth) is a later step — see PLAN.md §5.
"""

from collections import Counter, defaultdict

try:
    from class_normalization import normalize_class as _normalize_class
except Exception:  # pragma: no cover - normalization is best-effort
    def _normalize_class(raw, fold_rare=True):
        return (raw or '').upper().strip()


# Major equipment where one scheduled tag == one physical unit on the drawing.
# Detected count of these classes should equal the number of distinct scheduled
# tags. Everything else (air devices, ducts) repeats per tag and is presence-
# checked only.
UNIQUE_INSTANCE_CLASSES = {
    'FAN', 'EXHAUST FAN', 'CONDENSING UNIT', 'AIR HANDLING UNIT',
    'PACKAGED ROOFTOP UNIT', 'FAN COIL UNIT', 'HEAT PUMP', 'VAV', 'VRF',
    'HEATER', 'GAS UNIT HEATER', 'ERV',
    'INDOOR UNIT', 'OUTDOOR UNIT', 'SPLIT SYSTEM INDOOR', 'SPLIT SYSTEM OUTDOOR',
    'HUMIDIFIER', 'DEHUMIDIFIER',
}

# Trust tiers mirror PLAN.md §5 so the UX is consistent end-to-end.
TIER_HIGH = 0.80
TIER_MEDIUM = 0.50


def _norm(raw):
    return _normalize_class(raw or '')


def _scheduled_by_class(variables):
    """class -> set(tags) and tag -> class, from schedule TagVariables."""
    by_class = defaultdict(set)
    tag_class = {}
    for v in (variables or []):
        tag = (v.get('tag') or '').strip()
        if not tag:
            continue
        cls = _norm(v.get('inferred_yolo_class'))
        if not cls:
            continue
        by_class[cls].add(tag)
        tag_class.setdefault(tag, cls)
    return by_class, tag_class


def _detected_by_class(detections_per_page, conf_threshold):
    """class -> {'count', 'tags' Counter, 'confs' list} from YOLO detections."""
    out = defaultdict(lambda: {'count': 0, 'tags': Counter(), 'confs': []})
    for dets in (detections_per_page or {}).values():
        for d in dets:
            if d.get('conf') is not None and d['conf'] < conf_threshold:
                continue
            cls = _norm(d.get('cls'))
            rec = out[cls]
            rec['count'] += 1
            if d.get('conf') is not None:
                rec['confs'].append(float(d['conf']))
            tag = (d.get('tag') or '').strip()
            if tag:
                rec['tags'][tag] += 1
    return out


def reconcile(variables, detections_per_page, conf_threshold=0.0):
    """Compare scheduled expectations against detections.

    Returns a plain-data dict (JSON-serializable) so callers can render it to
    stdout, a sidecar, or an Excel sheet without re-deriving anything.
    """
    sched_by_class, tag_class = _scheduled_by_class(variables)
    det_by_class = _detected_by_class(detections_per_page, conf_threshold)

    all_classes = set(sched_by_class) | set(det_by_class)

    class_rows = []
    for cls in sorted(all_classes):
        expected_tags = sched_by_class.get(cls, set())
        expected = len(expected_tags)
        det = det_by_class.get(cls, {'count': 0, 'tags': Counter(), 'confs': []})
        detected = det['count']
        reconcilable = cls in UNIQUE_INSTANCE_CLASSES

        if reconcilable and expected > 0:
            delta = detected - expected
            if delta == 0:
                status = 'match'
            elif delta < 0:
                status = 'under'      # missed equipment — the costly error
            else:
                status = 'over'       # phantom / overcount
        elif reconcilable and expected == 0 and detected > 0:
            status = 'orphan_class'   # detected a class the schedule never lists
            delta = detected
        else:
            status = 'info'           # air devices / ducts: count not reconcilable
            delta = detected - expected if expected else 0

        # Which scheduled tags of this class never appear on a plan?
        detected_tags = set(det['tags'].keys())
        missing_tags = sorted(t for t in expected_tags if t not in detected_tags)

        confs = det['confs']
        class_rows.append({
            'class': cls,
            'expected': expected,
            'detected': detected,
            'delta': delta,
            'status': status,
            'reconcilable': reconcilable,
            'expected_tags': sorted(expected_tags),
            'missing_tags': missing_tags,
            'mean_conf': round(sum(confs) / len(confs), 3) if confs else None,
        })

    # ── Tag-level cross-reference ──────────────────────────────────────────
    scheduled_tags = set(tag_class)
    detected_tag_counts = Counter()
    for det in det_by_class.values():
        detected_tag_counts.update(det['tags'])
    detected_tags = set(detected_tag_counts)

    missing_on_plan = sorted(scheduled_tags - detected_tags)         # scheduled, never detected
    orphan_tags = sorted(detected_tags - scheduled_tags)             # detected, not in schedule

    # ── Project trust score (transparent heuristic, NOT calibrated) ────────
    components = {}

    recon_rows = [r for r in class_rows if r['reconcilable'] and r['expected'] > 0]
    if recon_rows:
        components['count_match_rate'] = (
            sum(1 for r in recon_rows if r['status'] == 'match') / len(recon_rows)
        )

    if scheduled_tags:
        components['tag_presence_rate'] = len(scheduled_tags & detected_tags) / len(scheduled_tags)

    all_confs = [c for det in det_by_class.values() for c in det['confs']]
    if all_confs:
        components['mean_detection_conf'] = sum(all_confs) / len(all_confs)

    weights = {'count_match_rate': 0.45, 'tag_presence_rate': 0.35, 'mean_detection_conf': 0.20}
    used = {k: w for k, w in weights.items() if k in components}
    if used:
        wsum = sum(used.values())
        score = sum(components[k] * w for k, w in used.items()) / wsum
    else:
        score = None

    if score is None:
        tier = 'UNKNOWN'
    elif score >= TIER_HIGH:
        tier = 'HIGH'
    elif score >= TIER_MEDIUM:
        tier = 'MEDIUM'
    else:
        tier = 'LOW'

    n_under = sum(1 for r in class_rows if r['status'] == 'under')
    n_over = sum(1 for r in class_rows if r['status'] in ('over', 'orphan_class'))

    return {
        'classes': class_rows,
        'missing_on_plan': missing_on_plan,
        'orphan_tags': orphan_tags,
        'project_confidence': round(score, 3) if score is not None else None,
        'tier': tier,
        'confidence_components': {k: round(v, 3) for k, v in components.items()},
        'summary': {
            'classes_compared': len(class_rows),
            'classes_under': n_under,
            'classes_over': n_over,
            'scheduled_tags': len(scheduled_tags),
            'scheduled_tags_found': len(scheduled_tags & detected_tags),
            'missing_on_plan': len(missing_on_plan),
            'orphan_tags': len(orphan_tags),
            'has_schedule': bool(scheduled_tags),
        },
    }


# Excel cell fills (hex, no leading '#') keyed by status, for the caller's sheet.
STATUS_FILL = {
    'match': 'C6EFCE',        # green
    'under': 'FFC7CE',        # red — missed equipment
    'over': 'FFEB9C',         # amber — overcount / phantom
    'orphan_class': 'FFEB9C',
    'info': 'FFFFFF',
}

STATUS_LABEL = {
    'match': 'OK',
    'under': 'UNDER (missed?)',
    'over': 'OVER (phantom?)',
    'orphan_class': 'NOT IN SCHEDULE',
    'info': 'count n/a',
}


def format_report(result):
    """Human-readable reconciliation report for stdout and the .txt sidecar."""
    s = result['summary']
    lines = []
    lines.append('=' * 75)
    lines.append('SCHEDULE ↔ DETECTION RECONCILIATION')
    lines.append('=' * 75)
    if not s['has_schedule']:
        lines.append('  No schedule tags parsed — nothing to reconcile against.')
        lines.append('  (Raster-only schedule? Detection counts are unverified.)')
        return '\n'.join(lines)

    conf = result['project_confidence']
    conf_disp = f'{conf:.0%}' if conf is not None else 'n/a'
    lines.append(f"  Project trust: {result['tier']} ({conf_disp})   "
                 f"[heuristic, not calibrated]")
    comp = result['confidence_components']
    if comp:
        bits = ', '.join(f'{k}={v:.0%}' for k, v in comp.items())
        lines.append(f"    components: {bits}")
    lines.append(f"  Scheduled tags found on plan: "
                 f"{s['scheduled_tags_found']}/{s['scheduled_tags']}")
    lines.append(f"  Classes under-detected: {s['classes_under']}   "
                 f"over-detected: {s['classes_over']}")
    lines.append('')

    # Per-class table (reconcilable classes first, by severity)
    order = {'under': 0, 'over': 1, 'orphan_class': 2, 'match': 3, 'info': 4}
    rows = sorted(result['classes'], key=lambda r: (order.get(r['status'], 9), r['class']))
    lines.append(f"  {'Class':<28} {'Sched':>6} {'Found':>6} {'Δ':>5}  Verdict")
    lines.append(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*5}  {'-'*16}")
    for r in rows:
        delta = r['delta']
        delta_disp = f'{delta:+d}' if r['status'] not in ('match', 'info') else ('0' if r['status'] == 'match' else '')
        lines.append(f"  {r['class'][:27]:<28} {r['expected']:>6} {r['detected']:>6} "
                     f"{delta_disp:>5}  {STATUS_LABEL.get(r['status'], r['status'])}")

    if result['missing_on_plan']:
        lines.append('')
        lines.append(f"  ⚠ Scheduled but NOT found on any plan ({len(result['missing_on_plan'])}):")
        lines.append('    ' + ', '.join(result['missing_on_plan'][:40])
                     + (' …' if len(result['missing_on_plan']) > 40 else ''))
    if result['orphan_tags']:
        lines.append('')
        lines.append(f"  ⚠ Detected tags not in any schedule ({len(result['orphan_tags'])}):")
        lines.append('    ' + ', '.join(result['orphan_tags'][:40])
                     + (' …' if len(result['orphan_tags']) > 40 else ''))
    return '\n'.join(lines)


if __name__ == '__main__':
    # Tiny self-test with synthetic data — no PDF needed.
    variables = [
        {'tag': 'CU-1', 'inferred_yolo_class': 'CONDENSING UNIT'},
        {'tag': 'CU-2', 'inferred_yolo_class': 'CONDENSING UNIT'},
        {'tag': 'EF-1', 'inferred_yolo_class': 'EXHAUST FAN'},
        {'tag': 'A1', 'inferred_yolo_class': 'AD-GRD'},
    ]
    detections_per_page = {
        0: [
            {'cls': 'CONDENSING UNIT', 'tag': 'CU-1', 'conf': 0.9},
            # CU-2 missed entirely → UNDER
            {'cls': 'EXHAUST FAN', 'tag': 'EF-1', 'conf': 0.8},
            {'cls': 'EXHAUST FAN', 'tag': None, 'conf': 0.7},  # extra fan → OVER
            {'cls': 'AD-GRD', 'tag': 'A1', 'conf': 0.6},
            {'cls': 'AD-GRD', 'tag': 'A1', 'conf': 0.6},       # repeats OK (info)
            {'cls': 'AD-GRD', 'tag': 'Z9', 'conf': 0.5},       # orphan tag
        ],
    }
    res = reconcile(variables, detections_per_page)
    print(format_report(res))
    assert any(r['class'] == 'CONDENSING UNIT' and r['status'] == 'under' for r in res['classes'])
    assert any(r['class'] == 'EXHAUST FAN' and r['status'] == 'over' for r in res['classes'])
    assert 'CU-2' in res['missing_on_plan']
    assert 'Z9' in res['orphan_tags']
    print('\nself-test OK')
