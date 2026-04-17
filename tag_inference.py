"""
Tag Inference System — 3-level approach to assign tags to YOLO detections.

Level 1: Direct mapping
  If schedule has exactly 1 tag per YOLO class → auto-assign.
  Works for: Flex projects (A→T-BAR SUPPLY, B→T-BAR RETURN, etc.)

Level 2: CFM/size matching
  Read nearby text (CFM, dimensions) via PyMuPDF text layer.
  Match to schedule row by size/CFM values.
  Works for: Shamrock (CD-1→10", CD-2→8", etc.)

Level 3: Class counts fallback
  Output class-level counts when tags can't be determined.
  User assigns tags manually from schedule knowledge.

Usage:
    from tag_inference import infer_tags
    infer_tags(detections_per_page, schedules, marks, mark_details, pdf_path)
"""
import re
from collections import defaultdict
import fitz


# Common tag prefix → YOLO class mapping
TAG_PREFIX_CLASS = {
    'EF': 'EXHAUST FAN', 'SF': 'FAN', 'CF': 'FAN', 'RF': 'FAN',
    'CU': 'CONDENSING UNIT', 'AC': 'CONDENSING UNIT',
    'AHU': 'AIR HANDLING UNIT', 'RTU': 'PACKAGED ROOFTOP UNIT',
    'FCU': 'FAN COIL UNIT', 'HP': 'HEAT PUMP',
    'EUH': 'HEATER', 'UH': 'HEATER', 'EH': 'HEATER', 'BH': 'HEATER',
    'VAV': 'VAV', 'VRF': 'VRF', 'ERV': 'CONDENSING UNIT',
    'MD': 'MOTORIZED DAMPER', 'MVD': 'MANUAL VOLUME DAMPER', 'FD': 'FIRE DAMPER',
    'FSD': 'FIRE SMOKE DAMPER', 'BD': 'BACKDRAFT DAMPER',
    'L': 'LOUVER', 'LVR': 'LOUVER',
    'GR': 'AD-GRD', 'RG': 'AD-GRD', 'SD': 'AD-GRD', 'CD': 'AD-GRD',
    'SA': 'AD-GRD', 'RA': 'AD-GRD', 'EA': 'AD-GRD', 'SB': 'AD-GRD',
    'LD': 'AD-LINEAR PLENUM',
}


def _infer_class_from_tag(tag):
    """Infer YOLO class from tag prefix. E.g., EF-1 → EXHAUST FAN."""
    if not tag:
        return None
    tag_upper = tag.upper()
    # Try longest prefix first (EUH before E)
    for prefix_len in range(4, 0, -1):
        prefix = tag_upper[:prefix_len]
        if prefix in TAG_PREFIX_CLASS:
            return TAG_PREFIX_CLASS[prefix]
    return None


def _infer_yolo_class_from_service(service_text, mounting_text=''):
    """
    Map schedule SERVICE/TYPE + MOUNTING to a YOLO class name.
    Handles multi-line text like "CEILING\nSUPPLY AIR".
    """
    if not service_text:
        return None
    # Collapse multi-line, normalize
    s = ' '.join(service_text.upper().split())
    m = ' '.join(mounting_text.upper().split()) if mounting_text else ''
    combined = f"{s} {m}".strip()

    # LAY-IN / T-BAR mounted → AD-T-BAR class
    if 'LAY-IN' in combined or 'T-BAR' in combined or 'TBAR' in combined:
        if 'SUPPLY' in combined:
            return 'AD-T-BAR SUPPLY'
        if 'RETURN' in combined:
            return 'AD-T-BAR RETURN'

    # CEILING diffuser without explicit mounting → likely T-BAR
    if 'CEILING' in combined:
        if 'SUPPLY' in combined:
            return 'AD-T-BAR SUPPLY'
        if 'RETURN' in combined:
            return 'AD-T-BAR RETURN'

    # Surface / exposed mounted
    if 'SURFACE' in combined or 'EXPOSED' in combined:
        if 'ROUND' in combined or ('SUPPLY' in combined and 'GRILLE' not in combined):
            return 'AD-SURF SUPPLY'
        if 'RETURN' in combined:
            return 'AD-SURF RETURN'

    # Linear
    if 'LINEAR' in combined:
        if 'SLOT' in combined:
            return 'AD-LINEAR SLOT DIFFUSER'
        if 'PLENUM' in combined:
            return 'AD-LINEAR PLENUM'

    # Generic GRD fallback
    if any(kw in combined for kw in ['DIFFUSER', 'GRILLE', 'REGISTER', 'SUPPLY', 'RETURN', 'EXHAUST']):
        return 'AD-GRD'

    return None


def build_class_to_tags(mark_details, schedules):
    """
    From schedule tables in the PDF, build a mapping: YOLO_class -> {tag -> details}.
    Infers the YOLO class from the schedule's TYPE/SERVICE column.
    """
    class_tags = defaultdict(dict)

    for sched in schedules:
        header_upper = [h.upper() for h in sched.get('header', [])]
        header = sched['header']

        # Find key columns
        tag_idx = None
        for i, h in enumerate(header_upper):
            if any(kw in h for kw in ('TAG', 'MARK', 'DESIGNATION')):
                tag_idx = i
                break
        if tag_idx is None:
            continue

        size_idx = next((i for i, h in enumerate(header_upper) if 'SIZE' in h or 'NECK' in h), None)
        cfm_idx = next((i for i, h in enumerate(header_upper) if 'CFM' in h or 'CAPACITY' in h), None)
        type_idx = next((i for i, h in enumerate(header_upper) if any(kw in h for kw in ('TYPE', 'SERVICE', 'DESCRIPTION'))), None)
        mount_idx = next((i for i, h in enumerate(header_upper) if 'MOUNT' in h), None)

        def _get(row_dict, idx):
            if idx is None:
                return ''
            key = header[idx] if idx < len(header) else ''
            return row_dict.get(key, '').strip() if key else ''

        for row in sched['rows']:
            tag = _get(row, tag_idx)
            if not tag or 'Total' in tag or len(tag) > 25:
                continue

            # Skip VAV and other non-GRD tags
            tag_upper = tag.upper().replace('\n', ' ')
            if any(tag_upper.startswith(p) for p in ('VAV', 'AHU', 'FCU', 'RTU', 'EF-', 'SF-')):
                continue
            # Skip pure multi-line garbage
            if '\n' in tag:
                continue

            size = _get(row, size_idx)
            cfm = _get(row, cfm_idx)
            etype = _get(row, type_idx)
            mounting = _get(row, mount_idx)

            # Try service text first, then tag prefix, then fallback
            yolo_class = _infer_yolo_class_from_service(etype, mounting)
            if not yolo_class:
                yolo_class = _infer_class_from_tag(tag)
            if not yolo_class:
                yolo_class = 'AD-GRD'

            class_tags[yolo_class][tag] = {'size': size, 'cfm': cfm, 'type': etype}

    return dict(class_tags)


def build_class_to_tags_from_marks(marks, mark_details):
    """
    Simpler version: from parsed marks + details, infer class→tags.
    Since schedule parser doesn't always know the PRODUCT class,
    we group tags by their prefix pattern.
    """
    # Group by likely class
    tag_groups = defaultdict(list)
    for mark in marks:
        details = mark_details.get(mark, {})
        tag_groups[mark] = details

    return tag_groups


# ─── LEVEL 1: Direct class→tag mapping (from THIS project's schedule) ────────

def level1_direct_mapping(detections, schedule_tags, class_to_tags=None):
    """
    If a YOLO class maps to exactly 1 schedule tag → auto-assign.
    The mapping comes from THIS project's schedule, not hardcoded patterns.

    Example:
      Flex schedule says AD-SURF RETURN has only tag "D" → all AD-SURF RETURN = D
      But St Elizabeth has AD-GRD with 17 tags → can't auto-assign, skip to Level 2.

    Returns (detections_with_tags, stats).
    """
    auto_map = {}

    if class_to_tags:
        for cls, tags_dict in class_to_tags.items():
            if len(tags_dict) == 1:
                # This class has exactly 1 tag in the schedule → safe to auto-assign
                auto_map[cls] = list(tags_dict.keys())[0]

    tagged = 0
    for det in detections:
        cls = det.get('cls', '')
        if cls in auto_map:
            det['tag'] = auto_map[cls]
            det['tag_method'] = 'direct'
            det['tag_confidence'] = 1.0
            tagged += 1

    return detections, {
        'level': 1,
        'method': 'direct_mapping',
        'tagged': tagged,
        'total': len(detections),
        'mapping': auto_map,
    }


# ─── LEVEL 2: CFM/size text matching ────────────────────────────────────────

def extract_nearby_text(pdf_path, page_idx, det, radius_pts=60):
    """
    Get text near a detection from the PDF text layer.
    Returns list of text strings found within radius.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    words = page.get_text("words")
    doc.close()

    # Detection center in PDF points (convert from pixels)
    # Assume DPI=200 → scale = 200/72
    scale = 200 / 72
    det_cx_pts = det.get('cx', 0) / scale
    det_cy_pts = det.get('cy', 0) / scale

    nearby = []
    for w in words:
        wx = (w[0] + w[2]) / 2
        wy = (w[1] + w[3]) / 2
        dist = ((wx - det_cx_pts) ** 2 + (wy - det_cy_pts) ** 2) ** 0.5
        if dist <= radius_pts:
            nearby.append(w[4])

    return nearby


def find_size_cfm_in_text(texts):
    """
    Extract size and CFM values from nearby text.
    Returns dict with found values.
    """
    result = {}

    for text in texts:
        t = text.strip().upper()

        # CFM pattern: "260" or "260 CFM" or "260CFM"
        cfm_match = re.match(r'^(\d{2,4})\s*(CFM|L/S)?$', t)
        if cfm_match:
            result['cfm'] = cfm_match.group(1)

        # Size pattern: "8"" or "10"" or "22X22" or "24X12"
        size_match = re.match(r'^(\d{1,3})"?$', t) or re.match(r'^(\d{1,3}[xX]\d{1,3})$', t)
        if size_match:
            result['size'] = size_match.group(1)

        # Round duct: "8"Ø" or "10"Ø"
        round_match = re.match(r'^(\d{1,3})"?[ØO]?$', t)
        if round_match:
            result['neck'] = round_match.group(1)

    return result


def level2_size_cfm_matching(detections, schedule_tags, mark_details, pdf_path, page_idx):
    """
    For untagged detections, read nearby CFM/size text and match to schedule.

    Only attempts for detections that weren't tagged by Level 1.
    """
    if not mark_details:
        return detections, {'level': 2, 'method': 'size_cfm', 'tagged': 0, 'total': 0}

    # Build reverse index: (size or cfm) → tag
    value_to_tag = {}
    for tag, details in mark_details.items():
        for key in ('SIZE', 'NECK SIZE', 'SIZE\n(NECK)', 'CFM', 'CFM\nRANGE'):
            val = details.get(key, '').strip().upper().replace('"', '').replace("'", '')
            if val and val not in ('.', '-', 'N/A'):
                value_to_tag[val] = tag

    tagged = 0
    for det in detections:
        if det.get('tag'):
            continue  # Already tagged by Level 1

        try:
            nearby = extract_nearby_text(pdf_path, page_idx, det, radius_pts=80)
        except:
            continue

        found = find_size_cfm_in_text(nearby)

        # Try to match found values to a schedule tag
        for key, val in found.items():
            clean_val = val.upper().replace('"', '').replace("'", '')
            if clean_val in value_to_tag:
                det['tag'] = value_to_tag[clean_val]
                det['tag_method'] = 'size_cfm'
                det['tag_confidence'] = 0.7
                tagged += 1
                break

    return detections, {'level': 2, 'method': 'size_cfm', 'tagged': tagged}


# ─── LEVEL 3: Class counts fallback ─────────────────────────────────────────

def level3_class_fallback(detections):
    """
    For any still-untagged detections, mark as class-only (no specific tag).
    """
    for det in detections:
        if not det.get('tag'):
            det['tag'] = None
            det['tag_method'] = 'none'
            det['tag_confidence'] = 0

    untagged = sum(1 for d in detections if not d.get('tag'))
    return detections, {'level': 3, 'method': 'class_fallback', 'untagged': untagged}


# ─── MAIN ENTRY POINT ───────────────────────────────────────────────────────

def infer_tags(detections_per_page, schedules, marks, mark_details, pdf_path):
    """
    Run all 3 levels of tag inference on all detections.

    Returns:
        detections_per_page (mutated with 'tag' fields)
        stats: dict with per-level results
    """
    # Build class→tags mapping from schedule data
    class_to_tags = build_class_to_tags(mark_details, schedules)

    all_stats = []

    for page_idx, detections in detections_per_page.items():
        # Level 1: Direct mapping
        detections, stats1 = level1_direct_mapping(detections, marks, class_to_tags)
        all_stats.append(stats1)

        # Level 2: CFM/size matching (only for untagged)
        untagged_count = sum(1 for d in detections if not d.get('tag'))
        if untagged_count > 0 and mark_details:
            detections, stats2 = level2_size_cfm_matching(
                detections, marks, mark_details, pdf_path, page_idx
            )
            all_stats.append(stats2)

        # Level 3: Fallback
        detections, stats3 = level3_class_fallback(detections)
        all_stats.append(stats3)

    # Aggregate stats
    total = sum(len(d) for d in detections_per_page.values())
    tagged = sum(1 for dets in detections_per_page.values()
                 for d in dets if d.get('tag'))

    return detections_per_page, {
        'total': total,
        'tagged': tagged,
        'tagged_pct': tagged / max(total, 1) * 100,
        'levels': all_stats,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tag_inference.py path/to/blueprint.pdf")
        sys.exit(1)

    pdf = sys.argv[1]
    print(f"Testing tag inference on: {pdf}")

    from schedule_parser import parse_pdf_schedules
    schedules, marks, mark_details, legend, summary = parse_pdf_schedules(pdf)
    print(f"\nSchedule: {len(marks)} tags found: {marks}")

    # Simulate some detections (in real use, YOLO provides these)
    fake_dets = {
        5: [
            {'cls': 'AD-T-BAR SUPPLY', 'cx': 4000, 'cy': 2000, 'conf': 0.9},
            {'cls': 'AD-T-BAR RETURN', 'cx': 3500, 'cy': 2500, 'conf': 0.8},
            {'cls': 'AD-SURF SUPPLY', 'cx': 5000, 'cy': 3000, 'conf': 0.85},
            {'cls': 'AD-SURF RETURN', 'cx': 4500, 'cy': 3500, 'conf': 0.75},
            {'cls': 'AD-GRD', 'cx': 3000, 'cy': 1500, 'conf': 0.7},
        ]
    }

    print(f"\nRunning 3-level tag inference on {sum(len(d) for d in fake_dets.values())} detections...")
    _, stats = infer_tags(fake_dets, schedules, marks, mark_details, pdf)

    print(f"\nResults:")
    print(f"  Tagged: {stats['tagged']}/{stats['total']} ({stats['tagged_pct']:.0f}%)")

    for dets in fake_dets.values():
        for d in dets:
            tag = d.get('tag', '-')
            method = d.get('tag_method', '-')
            print(f"  {d['cls']:<30} -> tag={tag:<15} method={method}")
