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
    # Fans
    'EF': 'EXHAUST FAN', 'SF': 'FAN', 'CF': 'FAN', 'RF': 'FAN',
    'CEF': 'EXHAUST FAN',   # ceiling exhaust fan
    'IEF': 'EXHAUST FAN',   # inline exhaust fan
    # Major equipment
    'CU': 'CONDENSING UNIT', 'AC': 'CONDENSING UNIT', 'OACU': 'CONDENSING UNIT',
    'AHU': 'AIR HANDLING UNIT', 'DOAS': 'AIR HANDLING UNIT',
    'RTU': 'PACKAGED ROOFTOP UNIT',
    'FCU': 'FAN COIL UNIT', 'FC': 'FAN COIL UNIT',
    'HP': 'HEAT PUMP',
    # Heaters
    'EUH': 'HEATER', 'UH': 'HEATER', 'EH': 'HEATER', 'BH': 'HEATER',
    'CUH': 'HEATER', 'DH': 'HEATER',
    # Terminals / specialty
    'VAV': 'VAV', 'VRF': 'VRF', 'ERV': 'CONDENSING UNIT',
    # Dampers
    'MD': 'MOTORIZED DAMPER', 'MVD': 'MANUAL VOLUME DAMPER', 'FD': 'FIRE DAMPER',
    'FSD': 'FIRE SMOKE DAMPER', 'BD': 'BACKDRAFT DAMPER', 'SD': 'SMOKE DAMPER',
    # Louvers
    'L': 'LOUVER', 'LVR': 'LOUVER',
    'EL': 'LOUVER',        # exhaust louver
    'SL': 'LOUVER',        # supply louver
    'IL': 'LOUVER',        # intake louver
    # Grilles / registers / diffusers (AD-GRD family)
    'GR': 'AD-GRD', 'RG': 'AD-GRD', 'CD': 'AD-GRD',
    'SA': 'AD-GRD', 'RA': 'AD-GRD', 'EA': 'AD-GRD', 'SB': 'AD-GRD',
    'EG': 'AD-GRD',        # exhaust grille
    'SG': 'AD-GRD',        # supply grille
    'RR': 'AD-GRD',        # return register
    'SR': 'AD-GRD',        # supply register
    'ER': 'AD-GRD',        # exhaust register
    # Linear diffusers
    'LD': 'AD-LINEAR PLENUM',
}

# Map YOLO detection class names to the schedule-inferred class family.
# YOLO sometimes outputs a specific variant (SPLIT SYSTEM) while the schedule
# classifies equipment more broadly (CONDENSING UNIT). This lets Level 1 and
# Level 2b still match when the names differ.
YOLO_CLASS_ALIASES = {
    'SPLIT SYSTEM': 'CONDENSING UNIT',
    'PACKAGED ROOFTOP UNIT': 'PACKAGED ROOFTOP UNIT',
    'MANUAL VOLUME DAMPER': 'MOTORIZED DAMPER',  # YOLO often can't tell them apart
    'VENT CAP': 'EXHAUST FAN',                     # vent caps sit atop exhaust fans
}


def _resolve_class(yolo_class, class_to_tags):
    """
    Look up a YOLO class in class_to_tags, trying direct match first then
    aliases. Returns the key under which the class's tags live.
    """
    if yolo_class in class_to_tags:
        return yolo_class
    alias = YOLO_CLASS_ALIASES.get(yolo_class)
    if alias and alias in class_to_tags:
        return alias
    return None


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

    # Generic GRD fallback — broader keyword list for air device schedules
    # that describe diffusers as "PERFORATED FACE", "PLAQUE", etc. rather than
    # using "DIFFUSER" explicitly.
    if any(kw in combined for kw in ['DIFFUSER', 'GRILLE', 'REGISTER',
                                       'SUPPLY', 'RETURN', 'EXHAUST',
                                       'PERFORATED', 'PLAQUE', 'FACE',
                                       'LOUVERED', 'DROP', 'MOUNTED']):
        return 'AD-GRD'

    return None


def build_class_to_tags_from_variables(variables):
    """
    Build YOLO_class -> {tag -> properties} mapping directly from TagVariable
    list. Uses the `inferred_yolo_class` each variable already carries, which
    is cleaner than re-inferring from raw schedule rows.
    """
    class_tags = defaultdict(dict)
    for v in (variables or []):
        cls = v.get('inferred_yolo_class')
        tag = v.get('tag')
        if not cls or not tag:
            continue
        class_tags[cls][tag] = v.get('properties') or {}
    return dict(class_tags)


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
    Also applies YOLO_CLASS_ALIASES so detections like SPLIT SYSTEM can match
    schedule tags inferred as CONDENSING UNIT.

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
        resolved = cls if cls in auto_map else YOLO_CLASS_ALIASES.get(cls)
        if resolved and resolved in auto_map:
            det['tag'] = auto_map[resolved]
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


# ─── LEVEL 2A: Fingerprint matching using TagVariable properties ────────────

# Property values too generic to discriminate tags
_GENERIC_VALUES = {
    '', '-', '--', '---', '.', 'N/A', 'NA', 'NONE', 'TBD', 'NOTES',
    'SURFACE', 'LAY-IN', 'CEILING', 'WALL', 'INLINE', 'FLOOR', 'ROOF',
    'SUPPLY', 'RETURN', 'EXHAUST', 'OUTSIDE', 'MIXED', 'AIR',
    'YES', 'NO', 'VARIES', 'ALL', 'TYP', 'SEE NOTES',
    'ELECTRIC', 'GAS', 'HEAT PUMP', 'DX', 'HVAC',
}


def _clean_value(val):
    """Normalize a property value for text matching."""
    if not val:
        return ''
    return ' '.join(str(val).upper().replace('"', '').replace("'", '').split())


def build_tag_fingerprints(variables):
    """
    For each tag, build a set of distinctive value tokens that can be matched
    against nearby text on the drawing. A value is "distinctive" if it appears
    on <=2 tags (so it can disambiguate detections of the same class).

    Breaks compound values into tokens (e.g., "480V/3PH 28.7" -> {"480V","3PH","28.7"}).
    """
    from collections import defaultdict

    # Collect all tokens per tag
    tag_tokens = defaultdict(set)
    token_tags = defaultdict(set)  # reverse: which tags use each token?

    for v in variables:
        tag = v.get('tag')
        if not tag:
            continue
        for key, val in (v.get('properties') or {}).items():
            clean = _clean_value(val)
            if not clean or clean in _GENERIC_VALUES:
                continue
            # Break into tokens on whitespace/slashes — each token evaluated separately
            for tok in re.split(r'[\s/,]+', clean):
                tok = tok.strip('.()')
                if len(tok) < 2 or tok in _GENERIC_VALUES:
                    continue
                # Must contain at least one digit to be a useful discriminator
                # (purely verbal tokens like "CARRIER" would match every CU)
                if not re.search(r'\d', tok):
                    continue
                tag_tokens[tag].add(tok)
                token_tags[tok].add(tag)

    # Fingerprint = tokens that are distinctive (shared by <=2 tags)
    fingerprints = {}
    for tag, tokens in tag_tokens.items():
        distinctive = {t for t in tokens if len(token_tags[t]) <= 2}
        fingerprints[tag] = distinctive
    return fingerprints


def level2_fingerprint_matching(detections, variables, pdf_path, page_idx,
                                  class_to_tags, radius_pts=100):
    """
    Match untagged detections to specific tags using property fingerprints.

    For each multi-tag class:
      1. Build fingerprints (distinctive value tokens) per candidate tag.
      2. For each untagged detection, read text near it from the PDF text layer.
      3. Score (detection, tag) pairs by fingerprint overlap.
      4. Greedy 1:1 assignment — highest-scoring pairs win first, each tag
         claimed at most once per page (most equipment has one instance).
    """
    if not variables:
        return detections, {'level': '2a', 'method': 'fingerprint', 'tagged': 0}

    fingerprints = build_tag_fingerprints(variables)
    if not fingerprints:
        return detections, {'level': '2a', 'method': 'fingerprint', 'tagged': 0}

    from collections import defaultdict
    by_class = defaultdict(list)
    for i, det in enumerate(detections):
        if det.get('tag'):
            continue
        by_class[det.get('cls', '')].append(i)

    tagged = 0
    for cls, det_indices in by_class.items():
        resolved_cls = _resolve_class(cls, class_to_tags or {})
        if not resolved_cls:
            continue
        candidates = list(class_to_tags[resolved_cls].keys())
        if len(candidates) < 2:
            continue  # Single-tag classes handled by Level 1

        # Score every (detection, candidate_tag) pair
        scores = []  # (score, det_idx, tag)
        for di in det_indices:
            try:
                nearby_words = extract_nearby_text(pdf_path, page_idx,
                                                     detections[di],
                                                     radius_pts=radius_pts)
            except Exception:
                continue
            nearby_tokens = set()
            for w in nearby_words:
                clean = _clean_value(w)
                for tok in re.split(r'[\s/,]+', clean):
                    tok = tok.strip('.()')
                    if len(tok) >= 2 and re.search(r'\d', tok):
                        nearby_tokens.add(tok)

            for tag in candidates:
                fp = fingerprints.get(tag, set())
                if not fp:
                    continue
                overlap = fp & nearby_tokens
                if overlap:
                    scores.append((len(overlap), di, tag))

        # Greedy 1:1 assignment
        scores.sort(key=lambda x: -x[0])
        assigned_dets = set()
        used_tags = set()
        for score, di, tag in scores:
            if di in assigned_dets or tag in used_tags:
                continue
            detections[di]['tag'] = tag
            detections[di]['tag_method'] = 'fingerprint'
            detections[di]['tag_confidence'] = min(0.5 + 0.15 * score, 0.95)
            assigned_dets.add(di)
            used_tags.add(tag)
            tagged += 1

    return detections, {'level': '2a', 'method': 'fingerprint', 'tagged': tagged}


# ─── LEVEL 2B: Schedule-guided bubble OCR ───────────────────────────────────
# Most HVAC drawings print a small tag label (e.g., "CU-1", "FCU-7") in a
# bubble next to each equipment symbol. OCR the region around each detection
# and match against the valid tag list for that YOLO class.

def level2b_bubble_detect(detections, class_to_tags, img, max_distance=140):
    """Use the trained tag-bubble detector to find tight bubble bboxes, OCR
    each one, then assign the closest matching valid tag to each untagged
    detection. Higher precision than the windowed OCR fallback because we
    OCR a tight bubble crop instead of a 150 px window of mixed content.

    Returns (detections, stats). If the bubble detector model is missing
    or no bubbles fire on this page, returns 0 tagged so the caller can
    fall back to level2b_bubble_ocr.
    """
    if img is None or not class_to_tags:
        return detections, {'level': '2b\'', 'method': 'bubble_detect', 'tagged': 0}

    try:
        from tag_matcher import detect_bubbles_on_page, ocr_bubble_crops, _normalize_for_match
    except Exception as e:
        return detections, {'level': '2b\'', 'method': 'bubble_detect',
                              'tagged': 0, 'error': str(e)}

    bubbles = detect_bubbles_on_page(img)
    if not bubbles:
        return detections, {'level': '2b\'', 'method': 'bubble_detect',
                              'tagged': 0, 'bubbles': 0}

    bubbles = ocr_bubble_crops(img, bubbles)
    if not bubbles:
        return detections, {'level': '2b\'', 'method': 'bubble_detect',
                              'tagged': 0, 'bubbles_ocr': 0}

    tagged = 0
    for det in detections:
        if det.get('tag'):
            continue
        cls = det.get('cls', '')
        resolved_cls = _resolve_class(cls, class_to_tags)
        if not resolved_cls:
            continue
        valid_tags = list(class_to_tags[resolved_cls].keys())
        if not valid_tags:
            continue
        # normalized tag lookup
        tag_lookup = {}
        for t in valid_tags:
            n = _normalize_for_match(t)
            if n:
                tag_lookup[n] = t

        dcx, dcy = det.get('cx', 0), det.get('cy', 0)
        best = None
        best_dist = float('inf')
        for b in bubbles:
            n = _normalize_for_match(b.get('text', ''))
            if not n or n not in tag_lookup:
                continue
            dist = ((b['cx'] - dcx) ** 2 + (b['cy'] - dcy) ** 2) ** 0.5
            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best = tag_lookup[n]
        if best:
            det['tag'] = best
            det['tag_method'] = 'bubble_detect'
            det['tag_confidence'] = 1.0 - min(best_dist / max_distance, 1.0)
            tagged += 1

    return detections, {'level': '2b\'', 'method': 'bubble_detect',
                          'tagged': tagged, 'bubbles': len(bubbles)}


def level2b_bubble_ocr(detections, class_to_tags, img, crop_size=150,
                         max_distance=140):
    """
    For each untagged detection in a multi-tag class, OCR a small crop around
    it and match tokens against the valid tags for that class. Greedy 1:1
    assignment by proximity (closest matched word to detection center wins).

    Requires `img` — a BGR numpy array of the rendered page (the same image
    used for YOLO inference, at the same DPI).
    """
    if img is None or not class_to_tags:
        return detections, {'level': '2b', 'method': 'bubble_ocr', 'tagged': 0}

    # Import lazily — pulls in EasyOCR which is heavy
    try:
        from tag_matcher import ocr_near_detection, match_valid_tags
    except Exception as e:
        return detections, {'level': '2b', 'method': 'bubble_ocr',
                              'tagged': 0, 'error': str(e)}

    by_class = defaultdict(list)
    for i, det in enumerate(detections):
        if det.get('tag'):
            continue
        by_class[det.get('cls', '')].append(i)

    tagged = 0
    for cls, det_indices in by_class.items():
        resolved_cls = _resolve_class(cls, class_to_tags)
        if not resolved_cls:
            continue
        valid_tags = list(class_to_tags[resolved_cls].keys())
        if len(valid_tags) < 2:
            continue

        # For each detection, pick the closest matching valid tag.
        # No 1:1 constraint — the same tag can be assigned to many detections
        # (air devices like A1 commonly repeat across a floor plan).
        for di in det_indices:
            det = detections[di]
            try:
                words = ocr_near_detection(img, det, crop_size=crop_size,
                                             conf_threshold=0.3)
            except Exception:
                continue
            matches = match_valid_tags(words, valid_tags)
            if not matches:
                continue
            dcx = det.get('cx', 0)
            dcy = det.get('cy', 0)
            best = None
            best_dist = float('inf')
            for tag, word in matches:
                dist = ((word['cx'] - dcx) ** 2 + (word['cy'] - dcy) ** 2) ** 0.5
                if dist < best_dist and dist <= max_distance:
                    best_dist = dist
                    best = tag
            if best is not None:
                detections[di]['tag'] = best
                detections[di]['tag_method'] = 'bubble_ocr'
                detections[di]['tag_confidence'] = 1.0 - min(best_dist / max_distance, 1.0)
                tagged += 1

    return detections, {'level': '2b', 'method': 'bubble_ocr', 'tagged': tagged}


# ─── LEVEL 2C: CFM/size text matching (legacy fallback) ─────────────────────

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

    # Build reverse index: any distinctive value → tag
    value_to_tag = {}
    for tag, details in mark_details.items():
        for key, val in details.items():
            val = str(val).strip().upper().replace('"', '').replace("'", '')
            # Skip generic/empty values
            if not val or val in ('.', '-', 'N/A', 'NONE', '') or len(val) > 30:
                continue
            # Skip common non-distinctive words
            if val in ('SURFACE', 'LAY-IN', 'CEILING', 'SUPPLY', 'RETURN', 'YES', 'NO'):
                continue
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

def infer_tags(detections_per_page, schedules, marks, mark_details, pdf_path,
               variables=None, page_images=None):
    """
    Run all levels of tag inference on all detections.

    Levels:
      1.   Direct class->tag mapping when schedule has a single tag per class
      2a.  Fingerprint matching using TagVariable properties (rich)
      2b.  Legacy CFM/size matching from mark_details (fallback)
      3.   Mark anything still untagged as no-tag

    Returns:
        detections_per_page (mutated with 'tag' fields)
        stats: dict with per-level results
    """
    # Build class→tags mapping — prefer variables (clean, single source) when
    # available, otherwise fall back to legacy schedule/mark_details inference.
    if variables:
        class_to_tags = build_class_to_tags_from_variables(variables)
    else:
        class_to_tags = build_class_to_tags(mark_details, schedules)

    all_stats = []

    for page_idx, detections in detections_per_page.items():
        # Level 1: Direct mapping
        detections, stats1 = level1_direct_mapping(detections, marks, class_to_tags)
        all_stats.append(stats1)

        # Level 2a: Fingerprint matching using variables (from PDF text layer)
        untagged_count = sum(1 for d in detections if not d.get('tag'))
        if untagged_count > 0 and variables:
            detections, stats2a = level2_fingerprint_matching(
                detections, variables, pdf_path, page_idx, class_to_tags
            )
            all_stats.append(stats2a)

        # Level 2b': Bubble DETECT — run the trained tag-bubble YOLO across
        # the page, OCR each tight bubble crop, match against valid tags.
        # Higher precision than the 150 px windowed OCR fallback below.
        untagged_count = sum(1 for d in detections if not d.get('tag'))
        if (untagged_count > 0 and variables and page_images
                and page_idx in page_images):
            detections, stats2bp = level2b_bubble_detect(
                detections, class_to_tags, page_images[page_idx]
            )
            all_stats.append(stats2bp)

        # Level 2b: Windowed bubble OCR — fallback when the bubble detector
        # didn't find anything. Crops a fixed 150 px window around each
        # detection and runs EasyOCR. Lower precision but higher recall on
        # drawings the bubble model wasn't trained for.
        untagged_count = sum(1 for d in detections if not d.get('tag'))
        if (untagged_count > 0 and variables and page_images
                and page_idx in page_images):
            detections, stats2b = level2b_bubble_ocr(
                detections, class_to_tags, page_images[page_idx]
            )
            all_stats.append(stats2b)

        # Level 2c: Legacy CFM/size matching — only when we have no
        # variables. Lacks class filtering; cross-class matching risk.
        untagged_count = sum(1 for d in detections if not d.get('tag'))
        if untagged_count > 0 and mark_details and not variables:
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
    schedules, marks, mark_details, legend, summary, variables = parse_pdf_schedules(pdf)
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
