"""
line_items.py — unified, evidence-carrying output contract + agreement gating.

This is the precision spine the technical reviews ask for. Every detection
becomes one LineItem that records the *independent* signals supporting it and a
status derived from how many of those signals agree:

  confirmed     ≥2 independent signals agree (read a real tag, class matches)
  needs_review  exactly 1 reliable signal, no contradiction
  flagged       signals contradict (e.g. a tag was read that the schedule
                doesn't contain, or a class the schedule never lists)

The three independent signal sources:
  • vision    — YOLO detected the symbol (always present)
  • text      — a tag was READ off the drawing near the symbol (OCR / text
                layer / fingerprint). NOT 'direct': a 'direct' tag is inferred
                from schedule structure, not an independent reading, so it does
                not corroborate vision on its own.
  • schedule  — the read tag (and its class) actually exists in the parsed
                schedule (closed-world confirmation)

Policy: ship `confirmed` as the high-precision output; surface the rest
honestly rather than guessing. No new dependencies.
"""

from collections import Counter

try:
    from validation_engine import _scheduled_by_class, _norm
except Exception:  # pragma: no cover
    def _norm(c):
        return (c or '').upper().strip()

    def _scheduled_by_class(variables):
        from collections import defaultdict
        by_class, tag_class = defaultdict(set), {}
        for v in (variables or []):
            t = (v.get('tag') or '').strip()
            c = _norm(v.get('inferred_yolo_class'))
            if t and c:
                by_class[c].add(t)
                tag_class.setdefault(t, c)
        return by_class, tag_class

try:
    from tag_inference import YOLO_CLASS_ALIASES
except Exception:  # pragma: no cover
    YOLO_CLASS_ALIASES = {}


# tag_method values that represent a tag actually READ from the drawing (an
# independent corroboration of the vision signal). 'direct' is excluded — it is
# inferred from schedule structure (class has exactly one tag), not a reading.
TEXT_METHODS = {
    'bubble_detect', 'bubble_ocr', 'text_layer_callout', 'fingerprint', 'size_cfm',
}

STATUS_CONFIRMED = 'confirmed'
STATUS_REVIEW = 'needs_review'
STATUS_FLAGGED = 'flagged'


def _classes_compatible(a, b):
    """True if two normalized class names refer to the same equipment, directly
    or via the YOLO↔schedule alias map."""
    if not a or not b:
        return False
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        alias = YOLO_CLASS_ALIASES.get(x)
        if alias == y:
            return True
        if isinstance(alias, list) and y in alias:
            return True
    return False


def agreement_gate(det, scheduled_tags, tag_class, class_set):
    """Return (status, confidence, flags, evidence) for one detection.

    Pure function over a detection dict + schedule lookups — no side effects.
    """
    cls = _norm(det.get('cls'))
    tag = (det.get('tag') or '').strip() or None
    method = det.get('tag_method')
    vision_conf = float(det.get('conf') or 0.0)
    tag_conf = float(det.get('tag_confidence') or 0.0)

    evidence = [{'source': 'yolo', 'value': cls, 'conf': round(vision_conf, 3)}]

    text_signal = bool(tag and method in TEXT_METHODS)
    if tag:
        evidence.append({'source': method or 'tag', 'value': tag,
                         'conf': round(tag_conf, 3)})

    tag_in_sched = bool(tag and tag in scheduled_tags)
    # Alias-aware: AD-GRD counts as "in schedule" if the schedule lists any of
    # its alias classes (AD-T-BAR SUPPLY/RETURN, …) — avoids false phantom flags.
    class_in_sched = cls in class_set or any(_classes_compatible(cls, c) for c in class_set)
    class_agrees = bool(tag and tag in tag_class
                        and _classes_compatible(cls, tag_class[tag]))
    if tag_in_sched:
        evidence.append({'source': 'schedule', 'value': tag, 'conf': 1.0})

    flags = []
    if tag and not tag_in_sched:
        flags.append('tag_not_in_schedule')
    if not class_in_sched:
        flags.append('class_not_in_schedule')
    if tag and tag_in_sched and not class_agrees:
        flags.append('class_tag_mismatch')

    # ── Gate ───────────────────────────────────────────────────────────────
    if text_signal and tag_in_sched and class_agrees:
        # read a real scheduled tag whose class matches → ≥2 signals agree
        status = STATUS_CONFIRMED
        confidence = min(0.99, 0.80 + 0.19 * ((vision_conf + tag_conf) / 2))
    elif tag and not tag_in_sched:
        # read a tag the schedule does not contain → contradiction
        status = STATUS_FLAGGED
        confidence = round(0.30 * vision_conf, 3)
    elif tag and tag_in_sched and not class_agrees:
        # tag is real but the detected class disagrees with the schedule's class
        status = STATUS_FLAGGED
        confidence = round(0.40 * vision_conf, 3)
    else:
        # detected, possibly class-plausible, but no corroborating read tag
        status = STATUS_REVIEW
        # plausible class → moderate; off-schedule class → lower
        base = vision_conf if class_in_sched else 0.6 * vision_conf
        confidence = round(0.35 + 0.45 * base, 3)

    return status, round(confidence, 3), flags, evidence


def build_line_items(detections_per_page, variables, annotate=True):
    """Build LineItems for every detection and (optionally) annotate each
    detection dict in place with qa_status / qa_confidence / qa_flags so the
    existing detections.json + Excel writers can surface them for free.
    """
    by_class, tag_class = _scheduled_by_class(variables)
    scheduled_tags = set(tag_class)
    class_set = set(by_class)

    items = []
    for page_idx, dets in (detections_per_page or {}).items():
        for d in dets:
            status, conf, flags, evidence = agreement_gate(
                d, scheduled_tags, tag_class, class_set)
            if annotate:
                d['qa_status'] = status
                d['qa_confidence'] = conf
                d['qa_flags'] = flags
            items.append({
                'page': page_idx,
                'class': _norm(d.get('cls')),
                'tag': (d.get('tag') or '').strip() or None,
                'bbox': [d.get('x1'), d.get('y1'), d.get('x2'), d.get('y2')],
                'status': status,
                'confidence': conf,
                'flags': flags,
                'evidence': evidence,
            })
    return items


def summarize(items):
    """Counts by status + a confirmed-rate, for the stdout QA block."""
    by_status = Counter(i['status'] for i in items)
    flags = Counter(f for i in items for f in i['flags'])
    total = len(items)
    return {
        'total': total,
        'confirmed': by_status.get(STATUS_CONFIRMED, 0),
        'needs_review': by_status.get(STATUS_REVIEW, 0),
        'flagged': by_status.get(STATUS_FLAGGED, 0),
        'confirmed_rate': round(by_status.get(STATUS_CONFIRMED, 0) / total, 3) if total else 0.0,
        'top_flags': flags.most_common(6),
    }


def format_summary(items):
    s = summarize(items)
    lines = [
        '=' * 75,
        'QA STATUS (agreement-gated — ship `confirmed`, review the rest)',
        '=' * 75,
        f"  Total line items: {s['total']}",
        f"  ✅ confirmed:    {s['confirmed']:>5}  ({s['confirmed_rate']:.0%})  ≥2 signals agree",
        f"  🟡 needs_review: {s['needs_review']:>5}         1 signal, no conflict",
        f"  🔴 flagged:      {s['flagged']:>5}         signals contradict",
    ]
    if s['top_flags']:
        lines.append('  flags: ' + ', '.join(f'{k}×{n}' for k, n in s['top_flags']))
    return '\n'.join(lines)


if __name__ == '__main__':
    variables = [
        {'tag': 'CU-1', 'inferred_yolo_class': 'CONDENSING UNIT'},
        {'tag': 'A1', 'inferred_yolo_class': 'AD-GRD'},
    ]
    dets = {
        0: [
            # read CU-1 via bubble OCR, exists in schedule, class matches → confirmed
            {'cls': 'CONDENSING UNIT', 'tag': 'CU-1', 'tag_method': 'bubble_detect',
             'tag_confidence': 0.9, 'conf': 0.92, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
            # detected, class in schedule, but no tag read → needs_review
            {'cls': 'AD-GRD', 'tag': None, 'tag_method': 'none',
             'tag_confidence': 0, 'conf': 0.7, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
            # read a tag that isn't in the schedule → flagged
            {'cls': 'AD-GRD', 'tag': 'Z9', 'tag_method': 'bubble_ocr',
             'tag_confidence': 0.6, 'conf': 0.6, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        ],
    }
    items = build_line_items(dets, variables)
    print(format_summary(items))
    assert dets[0][0]['qa_status'] == 'confirmed'
    assert dets[0][1]['qa_status'] == 'needs_review'
    assert dets[0][2]['qa_status'] == 'flagged'
    assert 'tag_not_in_schedule' in dets[0][2]['qa_flags']
    print('\nself-test OK')
