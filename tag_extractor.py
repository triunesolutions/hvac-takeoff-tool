"""
Detection summary helper.

`summarize_detections_by_tag()` groups YOLO detections by (class, tag) for the
Excel/JSON output in takeoff_cli.py. The earlier PyMuPDF text-layer tag
extraction that lived here (normalize_tag / extract_text_with_positions /
find_tag_candidates_near / assign_tags_to_detections) was superseded by the
schedule-driven `variables` path in tag_inference.py and has been removed.
"""
from collections import defaultdict


def summarize_detections_by_tag(detections):
    """
    Group detections by (class, tag) and return counts.
    Returns list of {class, tag, count, avg_confidence}.
    """
    groups = defaultdict(lambda: {'count': 0, 'confidences': []})

    for det in detections:
        cls = det.get('cls', 'UNKNOWN')
        tag = det.get('tag') or '(no-tag)'
        key = (cls, tag)
        groups[key]['count'] += 1
        groups[key]['confidences'].append(det.get('conf', 0))

    result = []
    for (cls, tag), data in sorted(groups.items(), key=lambda x: (-x[1]['count'], x[0])):
        avg_conf = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
        result.append({
            'class': cls,
            'tag': tag,
            'count': data['count'],
            'avg_confidence': avg_conf,
        })

    return result
