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
    # Split-system indoor / outdoor units (Fujitsu / Daikin / Mitsubishi style)
    'IU': 'INDOOR UNIT', 'OU': 'OUTDOOR UNIT',
    # Humidifiers, heat recovery, dehumidifiers
    'HUM': 'HUMIDIFIER', 'DHU': 'DEHUMIDIFIER',
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
    'TA': 'AD-GRD',        # transfer air grille
    # Linear diffusers
    'LD': 'AD-LINEAR PLENUM',
    # Single-letter air-device tag prefixes (Sola-style schedules: S-1, R-1, E-1).
    # These are checked LAST in _infer_class_from_tag (shortest prefixes lose to
    # longer matches like EF/SF/RF/SD), so they only fire for bare single-letter
    # tags. Class is broad — refined by SERVICE/MOUNTING inference at parse time.
    'S': 'AD-T-BAR SUPPLY',
    'R': 'AD-T-BAR RETURN',
    'E': 'AD-T-BAR RETURN',   # exhaust grilles use the same return-grille product
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
    # AD-GRD is the YOLO 'air device grille' generic class. Bubble OCR
    # disambiguates between supply/return/surface/linear by reading the tag
    # text. Matching against a list of candidate classes keeps Level 2b'
    # working when YOLO can't tell which AD-* sub-class a symbol belongs to.
    'AD-GRD': [
        'AD-T-BAR SUPPLY', 'AD-T-BAR RETURN',
        'AD-SURF SUPPLY', 'AD-SURF RETURN',
        'AD-LINEAR SLOT DIFFUSER', 'AD-LINEAR PLENUM',
    ],
    # v10 outputs these subclasses natively. Schedules typically file
    # everything under the generic AD-GRD class, so let bubble matching
    # fall back to AD-GRD's tag pool when the YOLO sub-class has no
    # direct schedule entries.
    'AD-T-BAR SUPPLY': 'AD-GRD',
    'AD-T-BAR RETURN': 'AD-GRD',
    'AD-SURF SUPPLY': 'AD-GRD',
    'AD-SURF RETURN': 'AD-GRD',
    'AD-LINEAR SLOT DIFFUSER': 'AD-GRD',
    'AD-LINEAR PLENUM': 'AD-GRD',
}


def _expand_class_for_bubble(yolo_class, class_to_tags):
    """Return list of class keys whose tags should be considered for a bubble
    detection of this YOLO class. Used by Level 2b' (bubble_detect) where the
    OCR'd bubble text disambiguates between candidate sub-classes."""
    candidates = []
    if yolo_class in class_to_tags:
        candidates.append(yolo_class)
    alias = YOLO_CLASS_ALIASES.get(yolo_class)
    if isinstance(alias, list):
        for a in alias:
            if a in class_to_tags and a not in candidates:
                candidates.append(a)
    elif alias and alias in class_to_tags and alias not in candidates:
        candidates.append(alias)
    return candidates


def _resolve_class(yolo_class, class_to_tags):
    """
    Look up a YOLO class in class_to_tags, trying direct match first then
    aliases. Returns the key under which the class's tags live.
    """
    if yolo_class in class_to_tags:
        return yolo_class
    alias = YOLO_CLASS_ALIASES.get(yolo_class)
    # List-form aliases mean "ambiguous between these candidates" — only the
    # bubble-OCR level can disambiguate. Bail here so Level 1 doesn't pick one.
    if isinstance(alias, list):
        return None
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

    # Fan family — must come BEFORE the generic AD-GRD fallback because that
    # bucket includes the word "EXHAUST" (for "EXHAUST GRILLE") and would
    # otherwise swallow "EXHAUST FAN" schedules.
    if 'FAN' in combined:
        if 'EXHAUST' in combined:
            return 'EXHAUST FAN'
        if 'SUPPLY' in combined or 'TRANSFER' in combined or 'CEILING' in combined:
            return 'FAN'
        return 'FAN'

    # Louver — explicit so the GRD fallback doesn't catch a service text like
    # "LOUVERED RETURN" and silently map LV-1 to AD-GRD.
    if 'LOUVER' in combined and 'LOUVERED' not in combined:
        return 'LOUVER'

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
        if cls in auto_map:
            resolved = cls
        else:
            alias = YOLO_CLASS_ALIASES.get(cls)
            # Skip list-form aliases — those mean "ambiguous, needs bubble OCR".
            resolved = alias if isinstance(alias, str) else None
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

def level1b_text_layer_callouts(detections, class_to_tags, pdf_path,
                                  page_idx_0based,
                                  emit_synthetic_for_unmatched=True,
                                  max_pair_distance_pt=8.0,
                                  yolo_match_distance_pt=80.0,
                                  dpi=200):
    """Scan the PDF text layer for tag callouts drawn as two stacked words —
    a prefix on top, a number directly below — typical of hexagonal callout
    conventions. This is how Pacific Palisades-style plans label EF-1, RTU-2,
    etc. The hexagon shape itself is a constellation of disconnected line
    segments that the bubble detector can't see; but the text is intact in
    the PDF text layer and unambiguous.

    For each callout found:
      - Assign to the nearest untagged YOLO detection of compatible class
        within ``yolo_match_distance_pt``.
      - If no nearby detection and emit_synthetic_for_unmatched=True, emit
        a synthetic detection at the callout's center so downstream counting
        still reports QTY > 0. Marked tag_method='text_layer_callout' so
        the Excel writer can flag them.

    All coordinates are converted to the 200 DPI pixel space YOLO uses so
    proximity tests are apples-to-apples.

    Returns (detections_or_extended_list, stats).
    """
    import fitz
    import re as _re

    if not class_to_tags:
        return detections, {'level': '1b', 'method': 'text_layer',
                            'tagged': 0, 'synthetic_emitted': 0}

    # Build a normalized valid-tag pool with the class each tag belongs to.
    # tag_to_class['EF-1'] → 'EXHAUST FAN'.  When the same tag shows up under
    # multiple classes (Pacific Palisades schedules: indoor-unit schedule
    # lists both IU and OU columns, and the outdoor-unit schedule does too),
    # prefer the class whose tag-prefix mapping matches — that's the
    # unambiguous truth from TAG_PREFIX_CLASS.
    tag_to_class: dict[str, str] = {}
    for cls, tags in class_to_tags.items():
        for t in tags.keys():
            if not t:
                continue
            tu = t.upper()
            prefix_class = _infer_class_from_tag(tu)
            existing = tag_to_class.get(tu)
            if existing is None:
                tag_to_class[tu] = cls
            elif prefix_class:
                # If we have a tag-prefix-derived class, only override the
                # existing entry when it doesn't match the prefix.
                if existing != prefix_class and cls == prefix_class:
                    tag_to_class[tu] = cls

    # Open the PDF for this single page only
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_idx_0based]
        words = page.get_text('words')
        doc.close()
    except Exception as e:
        return detections, {'level': '1b', 'method': 'text_layer',
                            'tagged': 0, 'synthetic_emitted': 0,
                            'error': f'open-failed: {e}'}

    if not words:
        return detections, {'level': '1b', 'method': 'text_layer',
                            'tagged': 0, 'synthetic_emitted': 0,
                            'reason': 'no-text-layer'}

    # Find PREFIX word + NUMBER word pairs that are vertically stacked
    # within tolerance. Same x-center within max_pair_distance_pt; the
    # number's top is within (text_height * 0.3, text_height * 1.5) below
    # the prefix's top — accommodates the slight overlap CAD exports
    # introduce due to text bounding-box padding.
    number_re = _re.compile(r'\d{1,3}')
    callouts = []
    for w in words:
        x0, y0, x1, y1, txt, *_ = w
        u = str(txt).upper().strip().replace('-', '').replace(',', '')
        if u not in TAG_PREFIX_CLASS:
            continue
        cx, ytop = (x0 + x1) / 2, y0
        h = max(y1 - y0, 6.0)
        for w2 in words:
            x0b, y0b, x1b, y1b, txtb, *_ = w2
            num = str(txtb).strip()
            if not number_re.fullmatch(num):
                continue
            cxb = (x0b + x1b) / 2
            dy = y0b - ytop
            if abs(cxb - cx) >= max_pair_distance_pt:
                continue
            if not (h * 0.3 < dy < h * 1.5):
                continue
            full = f'{u}-{num}'
            if full not in tag_to_class:
                continue
            callouts.append({
                'tag': full,
                'class': tag_to_class[full],
                'cx_pt': cx,
                'cy_pt': (ytop + y1b) / 2,
                'x0_pt': min(x0, x0b),
                'y0_pt': y0,
                'x1_pt': max(x1, x1b),
                'y1_pt': y1b,
            })
            break  # one number per prefix

    if not callouts:
        return detections, {'level': '1b', 'method': 'text_layer',
                            'tagged': 0, 'synthetic_emitted': 0,
                            'callouts': 0}

    # Convert callouts from points to 200-DPI pixels (where YOLO bboxes live)
    pt_to_px = dpi / 72.0
    for c in callouts:
        c['cx_px'] = c['cx_pt'] * pt_to_px
        c['cy_px'] = c['cy_pt'] * pt_to_px

    # Phase 1: assign each callout to nearest untagged YOLO det of a compatible
    # class. Track which detections we've consumed so the same one doesn't
    # get two tags.
    yolo_match_distance_px = yolo_match_distance_pt * pt_to_px
    tagged_via_yolo = 0
    consumed = set()
    callouts_unmatched = []
    for c in callouts:
        compatible_classes = {c['class']}
        # Also accept reverse-aliased YOLO labels (AD-GRD ↔ subclasses)
        for yolo_cls, schedule_cls in YOLO_CLASS_ALIASES.items():
            if isinstance(schedule_cls, list):
                if c['class'] in schedule_cls:
                    compatible_classes.add(yolo_cls)
            elif schedule_cls == c['class']:
                compatible_classes.add(yolo_cls)

        best_det = None
        best_dist = float('inf')
        for i, det in enumerate(detections):
            if i in consumed or det.get('tag'):
                continue
            if det.get('cls') not in compatible_classes:
                continue
            dcx = det.get('cx') or (det.get('x1', 0) + det.get('x2', 0)) / 2
            dcy = det.get('cy') or (det.get('y1', 0) + det.get('y2', 0)) / 2
            dist = ((dcx - c['cx_px']) ** 2 + (dcy - c['cy_px']) ** 2) ** 0.5
            if dist <= yolo_match_distance_px and dist < best_dist:
                best_dist = dist
                best_det = i
        if best_det is not None:
            detections[best_det]['tag'] = c['tag']
            detections[best_det]['tag_method'] = 'text_layer_callout'
            detections[best_det]['tag_confidence'] = 1.0
            tagged_via_yolo += 1
            consumed.add(best_det)
        else:
            callouts_unmatched.append(c)

    # Phase 2: emit synthetic detections for callouts that didn't find a
    # nearby YOLO box. The tag location IS the equipment location for
    # these plans (the symbol may exist but YOLO missed it, or it's a piece
    # of equipment YOLO wasn't trained to recognize). Use a small bbox at
    # the callout position so it has a coordinate footprint downstream.
    synthetic_emitted = 0
    if emit_synthetic_for_unmatched:
        for c in callouts_unmatched:
            cx, cy = c['cx_px'], c['cy_px']
            # 40-pixel synthetic bbox centered on the callout
            bbox_half = 20
            detections.append({
                'cls': c['class'],
                'tag': c['tag'],
                'tag_method': 'text_layer_callout',
                'tag_confidence': 1.0,
                'conf': 0.80,           # text-layer match = high confidence
                'x1': cx - bbox_half,
                'y1': cy - bbox_half,
                'x2': cx + bbox_half,
                'y2': cy + bbox_half,
                'cx': cx,
                'cy': cy,
                'synthetic': True,
                'source': 'text_layer_callout',
            })
            synthetic_emitted += 1

    return detections, {
        'level': '1b', 'method': 'text_layer',
        'callouts': len(callouts),
        'tagged': tagged_via_yolo,
        'synthetic_emitted': synthetic_emitted,
    }


def level2b_bubble_detect(detections, class_to_tags, img, max_distance=600):
    """Use the trained tag-bubble detector to find tight bubble bboxes, OCR
    each one, then assign the closest matching valid tag to each untagged
    detection. Higher precision than the windowed OCR fallback because we
    OCR a tight bubble crop instead of a 150 px window of mixed content.

    Returns (detections, stats). If the bubble detector model is missing
    or no bubbles fire on this page, returns 0 tagged so the caller can
    fall back to level2b_bubble_ocr.
    """
    import sys as _sys
    print(f"  [bubble_detect] called: img.shape={img.shape if img is not None else None}, "
          f"class_to_tags keys={list((class_to_tags or {}).keys())[:5]}, "
          f"detections={len(detections)}", file=_sys.stderr, flush=True)
    if img is None or not class_to_tags:
        return detections, {'level': '2b\'', 'method': 'bubble_detect', 'tagged': 0}

    try:
        from tag_matcher import (detect_bubbles_on_page, ocr_bubble_crops,
                                  merge_split_bubbles, _normalize_for_match)
    except Exception as e:
        return detections, {'level': '2b\'', 'method': 'bubble_detect',
                              'tagged': 0, 'error': str(e)}

    import sys as _sys2
    bubbles = detect_bubbles_on_page(img)
    print(f"  [bubble_detect] found {len(bubbles)} raw bubbles", file=_sys2.stderr, flush=True)
    if not bubbles:
        return detections, {'level': '2b\'', 'method': 'bubble_detect',
                              'tagged': 0, 'bubbles': 0}

    bubbles = ocr_bubble_crops(img, bubbles)
    print(f"  [bubble_detect] {len(bubbles)} bubbles after OCR. Sample texts: {[b.get('text', '') for b in bubbles[:8]]}", file=_sys2.stderr, flush=True)
    if not bubbles:
        return detections, {'level': '2b\'', 'method': 'bubble_detect',
                              'tagged': 0, 'bubbles_ocr': 0}
    # Some drawings draw tags as two stacked bubbles ("CD" + "A"); add
    # synthetic merged bubbles so pair-text like "CD-A" can match the schedule.
    bubbles_with_merges = merge_split_bubbles(bubbles)

    tagged = 0
    reclassified = 0
    for det in detections:
        if det.get('tag'):
            continue
        cls = det.get('cls', '')
        candidate_classes = _expand_class_for_bubble(cls, class_to_tags)
        if not candidate_classes:
            continue
        # Build a normalized tag lookup over the union of candidate classes,
        # remembering which class each tag came from so we can reclassify the
        # detection when the bubble disambiguates a generic AD-GRD prediction.
        tag_lookup = {}        # normalized → original tag string
        tag_to_class = {}      # normalized → class key
        for cc in candidate_classes:
            for t in class_to_tags[cc].keys():
                n = _normalize_for_match(t)
                if n and n not in tag_lookup:
                    tag_lookup[n] = t
                    tag_to_class[n] = cc

        # cx/cy may not be set on detections coming from run_inference; fall
        # back to bbox center. (Without this, all detections looked like they
        # sat at (0,0) and no bubble ever matched within max_distance.)
        dcx = det.get('cx') or (det.get('x1', 0) + det.get('x2', 0)) / 2
        dcy = det.get('cy') or (det.get('y1', 0) + det.get('y2', 0)) / 2
        best = None
        best_norm = None
        best_score = float('inf')
        best_dist = float('inf')
        # Score = distance − 80 px per extra normalized char. A specific
        # tag like "CD-A" (3 chars) beats a generic "CD" (2 chars) match
        # within ~80 px, but a much-closer "CD" still wins over a far
        # "CD-A". Tuned to handle Harbor Freight's mixed legend+schedule
        # tag pool without dropping prefix-only legend tags entirely.
        for b in bubbles_with_merges:
            raw_text = b.get('text', '')
            n = _normalize_for_match(raw_text)
            # Match strategies, tried in order:
            #   1) exact normalized match (e.g. "S1" == schedule "S1")
            #   2) prefix-before-dash (e.g. "S1-84" → "S1" because the schedule
            #      tag is the equipment mark; "-84" is a duct/neck size suffix)
            matched_norm = None
            if n and n in tag_lookup:
                matched_norm = n
            else:
                # Try splitting raw_text on common separators and matching the
                # prefix. Bubble texts like "S1-84", "RTU-2/3", "TG1-2" use a
                # dash, slash, or space between the tag mark and a size suffix.
                for sep in ('-', '/', ' ', '_'):
                    if sep in raw_text:
                        prefix_raw = raw_text.split(sep, 1)[0]
                        prefix_n = _normalize_for_match(prefix_raw)
                        if prefix_n and prefix_n in tag_lookup:
                            matched_norm = prefix_n
                            break
            if not matched_norm:
                continue
            dist = ((b['cx'] - dcx) ** 2 + (b['cy'] - dcy) ** 2) ** 0.5
            if dist > max_distance:
                continue
            score = dist - 80 * len(matched_norm)
            if score < best_score:
                best_score = score
                best_dist = dist
                best = tag_lookup[matched_norm]
                best_norm = matched_norm
        if best:
            det['tag'] = best
            det['tag_method'] = 'bubble_detect'
            det['tag_confidence'] = 1.0 - min(best_dist / max_distance, 1.0)
            # Reclassify the detection to the resolved class so the Excel
            # row groups under the right product (AD-T-BAR SUPPLY etc.).
            resolved = tag_to_class.get(best_norm)
            if resolved and resolved != cls:
                det['cls'] = resolved
                det['original_yolo_cls'] = cls
                reclassified += 1
            tagged += 1

    return detections, {'level': '2b\'', 'method': 'bubble_detect',
                          'tagged': tagged, 'bubbles': len(bubbles),
                          'reclassified': reclassified}


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
            dcx = det.get('cx') or (det.get('x1', 0) + det.get('x2', 0)) / 2
            dcy = det.get('cy') or (det.get('y1', 0) + det.get('y2', 0)) / 2
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

def _discover_tags_from_bubbles(detections_per_page, page_images,
                                schedules, variables, max_pages=3):
    """Run bubble detection + OCR across plan pages to harvest tag patterns
    that look like real HVAC tags (PREFIX-NUMBER). Returns a dict
    {tag → discovered_class}.

    Used when schedule parsing came back too thin (≤2 useful tags) to assign
    detections reliably — typically PDFs where the schedule is on a separate
    sheet or uses an unusual layout that pdfplumber's table detector misses.
    Confidence: tags only count when they appear ≥2 times across pages.
    """
    if not page_images:
        return {}
    import re as _re
    from collections import Counter as _Counter
    try:
        from tag_matcher import detect_bubbles_on_page, ocr_bubble_crops
    except Exception:
        return {}

    tag_pat = _re.compile(r'^([A-Z]{1,3})-?(\d{1,2})$')
    seen = _Counter()
    pages_done = 0
    for pno, img in page_images.items():
        if pages_done >= max_pages:
            break
        if img is None:
            continue
        try:
            bubbles = detect_bubbles_on_page(img)
            if not bubbles:
                continue
            bubbles = ocr_bubble_crops(img, bubbles)
        except Exception:
            continue
        for b in bubbles:
            t = (b.get('text') or '').upper().strip().replace(' ', '')
            m = tag_pat.match(t)
            if m:
                normalized = f'{m.group(1)}-{m.group(2)}'
                seen[normalized] += 1
        pages_done += 1

    # Context disambiguation: SD prefix maps to SMOKE DAMPER in the
    # dictionary, but on a diffuser plan (where we see RG, EG, TG, or SR
    # alongside it), SD almost certainly means "Supply Diffuser". Detect
    # the diffuser-context and override SD → AD-GRD for the discovery.
    prefixes_seen = {tag.split('-')[0] for tag in seen.keys() if seen[tag] >= 2}
    diffuser_context = bool(prefixes_seen & {'RG', 'EG', 'TG', 'SR', 'CD'})

    DIFFUSER_2LETTER = {'SD', 'RG', 'EG', 'TG', 'SR', 'CD'}

    discovered = {}
    for tag, count in seen.items():
        if count < 2:
            continue
        prefix = tag.split('-')[0]
        if diffuser_context and prefix in DIFFUSER_2LETTER:
            cls = 'AD-GRD'
        else:
            cls = TAG_PREFIX_CLASS.get(prefix)
            if not cls and prefix in DIFFUSER_2LETTER:
                cls = 'AD-GRD'
        if cls:
            discovered[tag] = cls
    return discovered


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

    # Schedule fallback: when extraction came back thin (≤2 tags total or any
    # class has just 1 generic-looking tag), harvest the real tags from the
    # plan via bubble OCR. This catches PDFs where the schedule lives on a
    # separate sheet (HLPUSD-style) or uses a layout pdfplumber misses.
    total_tags = sum(len(v) for v in class_to_tags.values())
    is_thin = total_tags <= 2 or any(
        len(tags) == 1 and len(list(tags.keys())[0]) <= 2
        for tags in class_to_tags.values()
    )
    if is_thin and page_images:
        discovered = _discover_tags_from_bubbles(
            detections_per_page, page_images, schedules, variables or [],
        )
        if discovered:
            print(f'  [tag-discovery] schedule was thin ({total_tags} tags); '
                  f'harvested {len(discovered)} tag(s) from plan bubbles: '
                  f'{", ".join(sorted(discovered.keys())[:10])}')
            # Extend class_to_tags with discovered tags. Use empty
            # properties dict — neck size will come from the waterfall.
            for tag, cls in discovered.items():
                if cls not in class_to_tags:
                    class_to_tags[cls] = {}
                if tag not in class_to_tags[cls]:
                    class_to_tags[cls][tag] = {'_source': 'bubble_discovery'}
                    # Also extend variables so downstream Excel writer sees them
                    if variables is not None:
                        variables.append({
                            'tag': tag,
                            'schedule_name': '<discovered-from-plan>',
                            'page': None,
                            'properties': {},
                            'inferred_yolo_class': cls,
                            'source_row_index': -1,
                        })

    all_stats = []

    for page_idx, detections in detections_per_page.items():
        # Level 1: Direct mapping
        detections, stats1 = level1_direct_mapping(detections, marks, class_to_tags)
        all_stats.append(stats1)

        # Level 1b: Text-layer hex-callout scanner — catches plans where tags
        # are drawn as hexagonal callouts with stacked "PREFIX" + "NUMBER"
        # words. Also emits synthetic detections for callouts where YOLO
        # missed the symbol entirely (RTU rooftop plans, IU/OU split-system
        # plans). Reliable when the PDF has a real text layer, near-zero
        # cost otherwise.
        if variables and pdf_path:
            detections, stats1b = level1b_text_layer_callouts(
                detections, class_to_tags, pdf_path, page_idx,
            )
            all_stats.append(stats1b)
            # detections list may have grown — write back so downstream sees it
            detections_per_page[page_idx] = detections

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
