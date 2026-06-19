"""
neck_size_waterfall.py — per-detection neck-size extraction.

Implements the 5-level cascade defined in PLAN.md §4. Each level returns
(neck_size, confidence, source, evidence) or None. First level that returns
a value wins; later levels don't run.

This module is STANDALONE — it doesn't modify any existing pipeline. To
integrate, call extract_neck_size_for_detection() from post_takeoff.py
after detections and variables are loaded.

Levels:
  1. Text-layer tag-size labels ("S1-8\"")           [implemented]
  2. Schedule NECK SIZE column + tag bubble OCR        [stub — Slice 2]
  3. OCR a crop around detection, look for size        [stub — Slice 2]
  4. CFM-range table lookup                            [stub — Slice 3]
  5. Explicit unknown                                  [implemented]

Confidence scoring follows the formulas in PLAN.md §5.
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


# ── Configuration ──────────────────────────────────────────────────────────

DETECTIONS_DPI = 200          # detections.json bbox coords are at this DPI
DPI_TO_PT = 72.0 / DETECTIONS_DPI

# Level 1 proximity: 400 px at 200 DPI = 2 inches. In PDF points = 144.
# Widened from 72pt (1 inch) on 2026-06-04 — Busy Bees showed valid labels
# sitting at 50-100pt from the detection center, just outside the old window.
LEVEL1_PROXIMITY_PT = 400 * DPI_TO_PT


# ── Regex patterns for tag-size labels ─────────────────────────────────────

# Combined tag + round size: "S1-8\"", "S2-6\"", "RG-12\""
TAG_ROUND_PATTERN = re.compile(
    r'^([A-Z]{1,3}\d{1,3})\s*[-\s]?\s*(\d+(?:\.\d+)?)\s*["\'″]$'
)

# Combined tag + rect size: "S1-10X8", "RG-12/12", "R1-22X10"
TAG_RECT_PATTERN = re.compile(
    r'^([A-Z]{1,3}\d{1,3})\s*[-\s]?\s*(\d+)\s*[xX/]\s*(\d+)$'
)

# Bare tag: "S1", "RG-2", "CU-1"
BARE_TAG_PATTERN = re.compile(r'^([A-Z]{1,3}-?\d{1,3})$')

# Bare round size: "8\"", "10\"", "6\""
BARE_ROUND_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)\s*["\'″]$')

# Bare rect size: "10X8", "12/12", "22X10"
BARE_RECT_PATTERN = re.compile(r'^(\d+)\s*[xX/]\s*(\d+)$')


def _normalize_round(size_str: str) -> str:
    """Normalize a round neck size string. '8' → '8"', '10.5' → '10.5"'."""
    try:
        v = float(size_str)
        if v == int(v):
            return f'{int(v)}"'
        return f'{v:g}"'
    except ValueError:
        return size_str.strip()


def _normalize_rect(w_str: str, h_str: str) -> str:
    """Normalize a rectangular neck size. ('10', '8') → '10x8'."""
    try:
        w, h = int(w_str), int(h_str)
        return f'{w}x{h}'
    except ValueError:
        return f'{w_str}x{h_str}'


# ── Level 1 — Text-layer tag-size label ────────────────────────────────────

def level1_textlayer_tag_size(
    det: dict,
    page_words: list,
    valid_tags: set[str] | None = None,
) -> dict | None:
    """Look for a tag-size combined label near the detection.

    Returns a result dict or None if no match.

    Args:
        det: detection dict with x1, y1, x2, y2 in pixel coords at DETECTIONS_DPI
        page_words: PyMuPDF's page.get_text("words") output —
                    list of (x0, y0, x1, y1, text, block, line, word_no)
        valid_tags: optional set of known tags from schedule. If provided,
                    only matches with these tags are accepted.
    """
    # Detection center in PDF points
    det_cx_pt = (det['x1'] + det['x2']) / 2 * DPI_TO_PT
    det_cy_pt = (det['y1'] + det['y2']) / 2 * DPI_TO_PT

    # Collect words within proximity, sorted by distance
    nearby = []
    for w in page_words:
        word_cx = (w[0] + w[2]) / 2
        word_cy = (w[1] + w[3]) / 2
        dist = math.hypot(word_cx - det_cx_pt, word_cy - det_cy_pt)
        if dist <= LEVEL1_PROXIMITY_PT:
            nearby.append({
                'text': w[4],
                'bbox': (w[0], w[1], w[2], w[3]),
                'distance_pt': dist,
            })
    nearby.sort(key=lambda x: x['distance_pt'])

    # Strategy A — try combined patterns (highest confidence)
    for word in nearby:
        text = word['text'].strip().upper()

        # Try round: "S1-8\""
        m = TAG_ROUND_PATTERN.match(text)
        if m:
            tag = m.group(1)
            if valid_tags and tag not in valid_tags:
                continue
            neck = _normalize_round(m.group(2))
            conf = _level1_confidence(
                distance_pt=word['distance_pt'],
                pattern_strength='exact_combined',
                source_type='text',
            )
            return {
                'neck_size': neck,
                'tag': tag,
                'confidence': conf,
                'source': 'level1-plan-text-combined-round',
                'evidence': f'"{word["text"]}" at {word["distance_pt"]:.0f}pt from detection',
                'distance_pt': word['distance_pt'],
            }

        # Try rect: "S1-10X8"
        m = TAG_RECT_PATTERN.match(text)
        if m:
            tag = m.group(1)
            if valid_tags and tag not in valid_tags:
                continue
            neck = _normalize_rect(m.group(2), m.group(3))
            conf = _level1_confidence(
                distance_pt=word['distance_pt'],
                pattern_strength='exact_combined',
                source_type='text',
            )
            return {
                'neck_size': neck,
                'tag': tag,
                'confidence': conf,
                'source': 'level1-plan-text-combined-rect',
                'evidence': f'"{word["text"]}" at {word["distance_pt"]:.0f}pt',
                'distance_pt': word['distance_pt'],
            }

    # Strategy B — separated tag + size in adjacent words
    # Find bare tag, then look for size in the next 3 nearby words
    for i, word in enumerate(nearby):
        text = word['text'].strip().upper()
        m = BARE_TAG_PATTERN.match(text)
        if not m:
            continue
        tag = m.group(1)
        if valid_tags and tag not in valid_tags:
            continue

        # Look for size in the next 3 nearby words (closest first, since sorted)
        for next_word in nearby[i+1:i+4]:
            next_text = next_word['text'].strip().upper()

            # Don't pair across large distance gaps
            if next_word['distance_pt'] > word['distance_pt'] + 20:
                continue

            # Try round
            m2 = BARE_ROUND_PATTERN.match(next_text)
            if m2:
                neck = _normalize_round(m2.group(1))
                conf = _level1_confidence(
                    distance_pt=word['distance_pt'],
                    pattern_strength='separated',
                    source_type='text',
                )
                return {
                    'neck_size': neck,
                    'tag': tag,
                    'confidence': conf,
                    'source': 'level1-plan-text-paired-round',
                    'evidence': f'tag "{tag}" + size "{next_word["text"]}" '
                                f'at {next_word["distance_pt"]:.0f}pt',
                    'distance_pt': word['distance_pt'],
                }

            # Try rect
            m2 = BARE_RECT_PATTERN.match(next_text)
            if m2:
                neck = _normalize_rect(m2.group(1), m2.group(2))
                conf = _level1_confidence(
                    distance_pt=word['distance_pt'],
                    pattern_strength='separated',
                    source_type='text',
                )
                return {
                    'neck_size': neck,
                    'tag': tag,
                    'confidence': conf,
                    'source': 'level1-plan-text-paired-rect',
                    'evidence': f'tag "{tag}" + size "{next_word["text"]}"',
                    'distance_pt': word['distance_pt'],
                }

    return None  # fall through to next level


def _level1_confidence(distance_pt: float, pattern_strength: str,
                     source_type: str) -> float:
    """Compute raw confidence for Level 1. Calibration applied later.

    Proximity curve was retuned 2026-06-04 after Busy Bees test showed
    the linear falloff was too aggressive: labels at typical distance of
    30-50pt got penalized to ~0.30 when they were clearly correct matches.
    New curve: flat near the detection, gentle falloff to LEVEL1_PROXIMITY_PT.
    """
    base = 0.95 if source_type == 'text' else 0.75

    # Two-segment proximity: full credit within bbox-typical distance, then
    # linear falloff. Empirically AD-GRD detections are ~36pt wide so a
    # label at <50pt is essentially "right next to" the detection.
    if distance_pt < 50:
        proximity_factor = 1.0
    else:
        # Linear from 1.0 at 50pt → 0.5 at LEVEL1_PROXIMITY_PT (72pt by default)
        proximity_factor = max(0.5, 1.0 - (distance_pt - 50) / (LEVEL1_PROXIMITY_PT - 50) * 0.5)

    pattern_factors = {
        'exact_combined': 1.0,    # "S1-8\""
        'separated': 0.85,        # "S1" then "8\""
        'inferred': 0.65,         # tag alone, size guessed
    }
    pattern_factor = pattern_factors.get(pattern_strength, 0.6)

    return round(base * proximity_factor * pattern_factor, 3)


# ── Level 2 — Schedule NECK SIZE column via tag bubble ────────────────────

# Common variants for the neck-size column name across HVAC schedules
NECK_SIZE_KEYS = (
    'NECK SIZE', 'NECK', 'INLET SIZE', 'INLET',
    'SIZE (NECK)', 'CONNECTION SIZE', 'CONN SIZE',
    'CONNECTION', 'INLET DIA', 'DIA',
)

# Distance within which a tag bubble can be associated with a detection.
# 350 px at 200 DPI = ~1.7 inches. Matches the existing bubble matcher in
# tag_inference.py's level2b_bubble_detect.
LEVEL2_BUBBLE_PROXIMITY_PX = 350


def _normalize_for_match(s: str) -> str:
    """Strip everything but alphanumerics, uppercase. Same as tag_matcher."""
    if not s:
        return ''
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def _match_bubble_to_schedule(bubble_text: str, valid_tags: set[str]) -> tuple[str, str] | None:
    """Try to match a bubble's OCR text to a known schedule tag.
    Returns (matched_tag, match_strength) where strength is 'exact'/'prefix'/'fuzzy'.
    """
    if not bubble_text or not valid_tags:
        return None

    norm = _normalize_for_match(bubble_text)
    if not norm:
        return None

    # Exact match
    for tag in valid_tags:
        if _normalize_for_match(tag) == norm:
            return (tag, 'exact')

    # Prefix match (bubble text like "S1-84" matches schedule tag "S1")
    for sep in ('-', '/', ' '):
        if sep in bubble_text:
            prefix = bubble_text.split(sep, 1)[0]
            prefix_norm = _normalize_for_match(prefix)
            if prefix_norm:
                for tag in valid_tags:
                    if _normalize_for_match(tag) == prefix_norm:
                        return (tag, 'prefix')

    # Substring fuzzy (bubble OCR'd noisily but contains the tag)
    for tag in valid_tags:
        tag_norm = _normalize_for_match(tag)
        if tag_norm and tag_norm in norm and len(tag_norm) >= 2:
            return (tag, 'fuzzy')

    return None


def _neck_size_from_props(props: dict) -> str | None:
    """Extract neck size from a schedule row's properties dict."""
    if not props:
        return None
    norm = {(k or '').upper().strip(): v for k, v in props.items()}
    for key in NECK_SIZE_KEYS:
        v = norm.get(key)
        if not v:
            continue
        s = str(v).strip()
        if not s or s in ('-', '—', '–', 'N/A', 'NA'):
            continue
        return s
    return None


def level2_schedule_neck_via_bubble(
    det: dict,
    bubbles_on_page: list[dict] | None = None,
    variables_by_tag: dict[str, dict] | None = None,
    valid_tags: set[str] | None = None,
) -> dict | None:
    """Find nearest tag bubble, match to schedule, return schedule's NECK SIZE.

    Args:
        det: detection dict
        bubbles_on_page: list of {x1,y1,x2,y2,text} bubble dicts from tag_matcher
        variables_by_tag: {tag_upper: {properties: {...}}} from variables.json
        valid_tags: set of schedule tag names (uppercase)
    """
    if not bubbles_on_page or not variables_by_tag:
        return None

    if valid_tags is None:
        valid_tags = set(variables_by_tag.keys())

    # Detection center in pixel space (same as bubble coords)
    dcx = (det['x1'] + det['x2']) / 2
    dcy = (det['y1'] + det['y2']) / 2

    # Find nearest matching bubble (closest distance, valid tag match)
    best = None
    for b in bubbles_on_page:
        bcx = (b.get('x1', 0) + b.get('x2', 0)) / 2 if 'x1' in b else b.get('cx', 0)
        bcy = (b.get('y1', 0) + b.get('y2', 0)) / 2 if 'y1' in b else b.get('cy', 0)
        dist = math.hypot(bcx - dcx, bcy - dcy)
        if dist > LEVEL2_BUBBLE_PROXIMITY_PX:
            continue

        bubble_text = (b.get('text') or '').strip()
        if not bubble_text:
            continue

        match = _match_bubble_to_schedule(bubble_text, valid_tags)
        if not match:
            continue

        matched_tag, match_strength = match
        if best is None or dist < best['distance']:
            best = {
                'tag': matched_tag,
                'match_strength': match_strength,
                'distance': dist,
                'bubble_text': bubble_text,
                'bubble_ocr_conf': float(b.get('conf') or b.get('confidence') or 0.9),
            }

    if not best:
        return None

    # Look up schedule entry
    schedule_entry = variables_by_tag.get(best['tag'])
    if not schedule_entry:
        return None

    props = schedule_entry.get('properties') or {}
    neck = _neck_size_from_props(props)
    if not neck:
        return None  # tag found but no neck size in schedule

    # Special case: schedule says <varies> — multiple sizes for one tag.
    # Surface as low-confidence; estimator must verify per-instance.
    if 'VARIES' in neck.upper() or '<' in neck or 'SEE PLAN' in neck.upper():
        return {
            'neck_size': None,
            'tag': best['tag'],
            'confidence': 0.30,
            'source': 'level2-schedule-varies',
            'evidence': f'bubble "{best["bubble_text"]}" → tag {best["tag"]} → schedule says "{neck}"',
            'flag': f'schedule says <varies> — needs plan label per instance',
        }

    conf = _level2_confidence(
        bubble_ocr_conf=best['bubble_ocr_conf'],
        tag_match_strength=best['match_strength'],
    )

    return {
        'neck_size': neck,
        'tag': best['tag'],
        'confidence': conf,
        'source': 'level2-schedule-via-bubble',
        'evidence': f'bubble "{best["bubble_text"]}" at {best["distance"]:.0f}px → tag {best["tag"]} → schedule neck "{neck}"',
        'match_strength': best['match_strength'],
    }


def _level2_confidence(bubble_ocr_conf: float, tag_match_strength: str) -> float:
    """Compute raw confidence for Level 2."""
    base = 0.90  # schedule match is high-quality when bubble OCR is clean

    match_factors = {
        'exact': 1.0,
        'prefix': 0.95,
        'fuzzy': 0.65,
    }
    match_factor = match_factors.get(tag_match_strength, 0.5)

    return round(base * max(0.4, bubble_ocr_conf) * match_factor, 3)


# ── Level 3 — Targeted OCR around detection ──────────────────────────────

# OCR lazy-loaded
_easyocr_reader = None
def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


# Size patterns OCR'd from a crop
LEVEL3_SIZE_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+)?)\s*["\'″]$'),         # 8" round
    re.compile(r'^(\d+)\s*[xX/]\s*(\d+)$'),            # 12X10 / 12/12 rect
    re.compile(r'^(\d+(?:\.\d+)?)\s*RD$', re.I),       # 8 RD
    re.compile(r'^(\d+(?:\.\d+)?)\s*INCH$', re.I),     # 8 INCH
    re.compile(r'^(\d+(?:\.\d+)?)\s*Ø$'),              # 8Ø
]


def level3_ocr_crop_for_size(
    det: dict,
    page_image_300dpi=None,
    **kwargs,
) -> dict | None:
    """Render a tight crop around the detection, OCR it, look for size patterns.

    Args:
        det: detection bbox at 200 DPI (px coords)
        page_image_300dpi: numpy RGB array of the page rendered at 300 DPI
    """
    if page_image_300dpi is None:
        return None

    try:
        import numpy as np
        import cv2
    except ImportError:
        return None

    # Scale detection coords from 200 DPI to 300 DPI
    scale = 1.5
    PADDING_PX_300 = 150  # ~0.5" padding at 300 DPI

    img = page_image_300dpi
    h, w = img.shape[:2]

    x1 = max(0, int(det['x1'] * scale - PADDING_PX_300))
    y1 = max(0, int(det['y1'] * scale - PADDING_PX_300))
    x2 = min(w, int(det['x2'] * scale + PADDING_PX_300))
    y2 = min(h, int(det['y2'] * scale + PADDING_PX_300))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Upscale + binarize for OCR
    try:
        crop_up = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        crop_gray = cv2.cvtColor(crop_up, cv2.COLOR_RGB2GRAY)
        # Otsu binarization
        _, crop_bin = cv2.threshold(crop_gray, 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    except Exception:
        crop_bin = crop

    try:
        reader = _get_ocr_reader()
        ocr_results = reader.readtext(crop_bin)
    except Exception:
        return None

    # det center in crop coords (300 DPI px, then scaled by 1.5 again)
    det_cx_in_crop = (det['x1'] + det['x2']) / 2 * scale - x1
    det_cy_in_crop = (det['y1'] + det['y2']) / 2 * scale - y1
    det_cx_scaled = det_cx_in_crop * 1.5
    det_cy_scaled = det_cy_in_crop * 1.5

    candidates = []
    for box, text, ocr_conf in ocr_results:
        text_clean = text.strip().upper()
        for pat in LEVEL3_SIZE_PATTERNS:
            m = pat.match(text_clean)
            if not m:
                continue

            # Parse size
            groups = m.groups()
            if len(groups) == 2:
                # rect like 12X10
                size = f'{int(float(groups[0]))}x{int(float(groups[1]))}'
            else:
                v = float(groups[0])
                size = f'{int(v) if v == int(v) else v}"'

            # OCR bbox center
            ocr_cx = (box[0][0] + box[2][0]) / 2
            ocr_cy = (box[0][1] + box[2][1]) / 2
            dist = math.hypot(ocr_cx - det_cx_scaled, ocr_cy - det_cy_scaled)

            # Classify pattern strength
            if pat is LEVEL3_SIZE_PATTERNS[0]:
                strength = 'standard_round'
            elif pat is LEVEL3_SIZE_PATTERNS[1]:
                strength = 'standard_rect'
            else:
                strength = 'variant'

            candidates.append({
                'size': size,
                'ocr_conf': float(ocr_conf),
                'distance': dist,
                'text': text,
                'strength': strength,
            })
            break  # one pattern match per OCR word

    if not candidates:
        return None

    # Pick best: prefer high OCR confidence, close to detection
    candidates.sort(key=lambda c: (-c['ocr_conf'] * (1.0 - min(c['distance'] / 600, 0.8))))
    best = candidates[0]

    conf = _level3_confidence(
        ocr_conf=best['ocr_conf'],
        distance_px=best['distance'],
        pattern_strength=best['strength'],
    )

    if conf < 0.20:  # too weak to surface
        return None

    return {
        'neck_size': best['size'],
        'tag': det.get('tag'),
        'confidence': conf,
        'source': 'level3-ocr-near-detection',
        'evidence': f'OCR "{best["text"]}" at {best["distance"]:.0f}px (300dpi), ocr_conf={best["ocr_conf"]:.2f}',
        'ocr_text': best['text'],
    }


def _level3_confidence(ocr_conf: float, distance_px: float,
                      pattern_strength: str) -> float:
    """Compute raw confidence for Level 3 (OCR crop). Capped at 0.80."""
    base = 0.80

    # Proximity: full credit within 200px of detection (~0.5" at 300 DPI)
    if distance_px < 200:
        proximity_factor = 1.0
    else:
        proximity_factor = max(0.3, 1.0 - (distance_px - 200) / 400 * 0.5)

    pattern_factors = {
        'standard_round': 1.0,
        'standard_rect': 0.95,
        'variant': 0.8,
        'ambiguous': 0.55,
    }
    pattern_factor = pattern_factors.get(pattern_strength, 0.6)

    return round(base * ocr_conf * proximity_factor * pattern_factor, 3)


# ── Level 4 — CFM range lookup ────────────────────────────────────────────

def _parse_cfm(value) -> float | None:
    """Parse a CFM value from various string formats."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Strip parentheses, units
    s = re.sub(r'[()CFM\s]', '', s.upper())
    try:
        return float(s)
    except ValueError:
        return None


def detect_cfm_range_table(variables: list[dict]) -> list[dict]:
    """Look for a CFM-range neck-size lookup table in the variables list.

    Returns a list of {min_cfm, max_cfm, neck_size} entries. Empty if no
    such table is present.

    Heuristic: a row whose 'tag' field is a CFM range like "50-100" and
    whose properties contain a neck size.
    """
    table = []
    range_re = re.compile(r'^(\d+)\s*[-–to]+\s*(\d+)$')
    for v in variables or []:
        tag = (v.get('tag') or '').strip()
        m = range_re.match(tag)
        if not m:
            # Also check schedule_name or first property for the range
            props = v.get('properties') or {}
            for k in ('CFM RANGE', 'RANGE', 'CFM'):
                val = props.get(k) or props.get(k.title())
                if val:
                    m = range_re.match(str(val).strip())
                    if m:
                        break
            if not m:
                continue
        cfm_min, cfm_max = float(m.group(1)), float(m.group(2))
        neck = _neck_size_from_props(v.get('properties') or {})
        if neck:
            table.append({
                'min_cfm': cfm_min,
                'max_cfm': cfm_max,
                'neck_size': neck,
            })
    table.sort(key=lambda x: x['min_cfm'])
    return table


def level4_cfm_range_lookup(
    det: dict,
    tag: str | None = None,
    variables_by_tag: dict[str, dict] | None = None,
    cfm_range_table: list[dict] | None = None,
    **kwargs,
) -> dict | None:
    """For a tagged detection, look up its CFM in the range table.

    Args:
        det: detection
        tag: resolved tag (str). If absent, falls back to det['tag'].
        variables_by_tag: schedule data — used to read CFM for the tag
        cfm_range_table: output of detect_cfm_range_table()
    """
    if not cfm_range_table or not variables_by_tag:
        return None

    tag = tag or det.get('tag')
    if not tag:
        return None

    schedule_entry = variables_by_tag.get(tag.upper())
    if not schedule_entry:
        return None

    props = schedule_entry.get('properties') or {}
    # Try multiple CFM column names
    cfm_value = None
    for k in ('CFM', 'AIRFLOW', 'AIR FLOW', 'FLOW', 'TOTAL CFM'):
        for pk in props:
            if pk and k in pk.upper():
                cfm_value = _parse_cfm(props[pk])
                if cfm_value is not None:
                    break
        if cfm_value is not None:
            break

    if cfm_value is None:
        return None

    # Find the matching range
    for entry in cfm_range_table:
        if entry['min_cfm'] <= cfm_value <= entry['max_cfm']:
            conf = _level4_confidence(
                cfm=cfm_value,
                range_min=entry['min_cfm'],
                range_max=entry['max_cfm'],
                table_size=len(cfm_range_table),
            )
            return {
                'neck_size': entry['neck_size'],
                'tag': tag,
                'confidence': conf,
                'source': 'level4-cfm-range-lookup',
                'evidence': f'CFM={cfm_value:.0f} in range {entry["min_cfm"]:.0f}-{entry["max_cfm"]:.0f} → {entry["neck_size"]}',
            }

    return None


def _level4_confidence(cfm: float, range_min: float, range_max: float,
                      table_size: int) -> float:
    """Compute raw confidence for Level 4. Capped at 0.55."""
    base = 0.55

    # Penalize edge-of-range CFM (less certain than mid-range)
    range_mid = (range_min + range_max) / 2
    range_half = max(1.0, (range_max - range_min) / 2)
    edge_factor = max(0.5, 1.0 - abs(cfm - range_mid) / range_half * 0.5)

    # Tiny tables (only a couple buckets) are less trustworthy
    if table_size < 4:
        table_factor = 0.75
    elif table_size >= 8:
        table_factor = 1.0
    else:
        table_factor = 0.85

    return round(base * edge_factor * table_factor, 3)


# ── Level 5 — Explicit unknown ────────────────────────────────────────────

def level5_unknown(det: dict, tag: str | None = None) -> dict:
    """Final fallback. Marks detection as needing manual review."""
    return {
        'neck_size': None,
        'tag': tag,
        'confidence': 0.0,
        'source': 'level5-unknown',
        'evidence': 'no neck size found via any method',
        'flag': 'manual_review_needed',
    }


# ── Waterfall driver ──────────────────────────────────────────────────────

def extract_neck_size_for_detection(
    det: dict,
    page_words: list | None = None,
    valid_tags: set[str] | None = None,
    bubbles_on_page: list[dict] | None = None,
    variables_by_tag: dict[str, dict] | None = None,
    page_image_300dpi=None,
    cfm_range_table: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Run the waterfall on a single detection. Return best result.

    Args:
        det: detection dict
        page_words: PyMuPDF words from the detection's page (Level 1)
        valid_tags: set of known tags from schedule (validation)
        bubbles_on_page: tag bubbles from tag_matcher (Level 2)
        variables_by_tag: schedule data {tag: {properties: {...}}} (Levels 2, 4)
        page_image_300dpi: rendered page numpy array (Level 3)
        cfm_range_table: extracted CFM-range lookup table (Level 4)
    """
    # Level 1 — text-layer tag-size labels
    if page_words is not None:
        result = level1_textlayer_tag_size(det, page_words, valid_tags)
        if result and result.get('neck_size'):
            return result

    # Level 2 — schedule via bubble
    if bubbles_on_page and variables_by_tag:
        result = level2_schedule_neck_via_bubble(
            det,
            bubbles_on_page=bubbles_on_page,
            variables_by_tag=variables_by_tag,
            valid_tags=valid_tags,
        )
        if result and result.get('neck_size'):
            return result

    # Level 3 — OCR crop
    if page_image_300dpi is not None:
        result = level3_ocr_crop_for_size(det, page_image_300dpi=page_image_300dpi)
        if result and result.get('neck_size'):
            return result

    # Level 4 — CFM range lookup
    if cfm_range_table and variables_by_tag:
        result = level4_cfm_range_lookup(
            det,
            tag=det.get('tag') or kwargs.get('tag'),
            variables_by_tag=variables_by_tag,
            cfm_range_table=cfm_range_table,
        )
        if result and result.get('neck_size'):
            return result

    # Level 5 — explicit unknown
    return level5_unknown(det, tag=det.get('tag') or kwargs.get('tag'))


# ── Page-level helper ─────────────────────────────────────────────────────

def extract_for_page(pdf_path: Path, page_index_0based: int,
                    detections_on_page: list[dict],
                    valid_tags: set[str] | None = None) -> list[dict]:
    """Convenience: extract neck size for every detection on a single page.

    Pulls page words once, then runs the waterfall per detection.
    Returns a list of (detection, waterfall_result) tuples.
    """
    doc = fitz.open(str(pdf_path))
    try:
        if page_index_0based < 0 or page_index_0based >= doc.page_count:
            return [(d, level5_unknown(d)) for d in detections_on_page]
        page = doc[page_index_0based]
        page_words = page.get_text('words')  # list of tuples
    finally:
        doc.close()

    out = []
    for det in detections_on_page:
        result = extract_neck_size_for_detection(
            det, page_words=page_words, valid_tags=valid_tags
        )
        out.append((det, result))
    return out


# ── CLI for testing ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import json
    from collections import Counter

    ap = argparse.ArgumentParser(
        description='Test the neck-size waterfall on an existing job.'
    )
    ap.add_argument('--pdf', required=True, help='Path to source PDF')
    ap.add_argument('--detections', required=True,
                    help='Path to detections.json')
    ap.add_argument('--variables',
                    help='Path to variables.json (for tag validation)')
    ap.add_argument('--out', help='Write per-detection results to JSON')
    args = ap.parse_args()

    dets = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    valid_tags = None
    if args.variables:
        v = json.loads(Path(args.variables).read_text(encoding='utf-8'))
        valid_tags = {x['tag'].upper() for x in v if x.get('tag')}
        print(f'Loaded {len(valid_tags)} valid tags from schedule')

    pdf_path = Path(args.pdf)
    all_results = []
    by_source = Counter()
    by_tag = Counter()

    for page_key, page_dets in dets.get('pages', {}).items():
        if not page_dets:
            continue
        page_idx = int(page_key)
        results = extract_for_page(
            pdf_path, page_idx, page_dets, valid_tags=valid_tags
        )
        for det, result in results:
            all_results.append({
                'page': page_idx + 1,
                'cls': det.get('cls'),
                'bbox': [det.get('x1'), det.get('y1'),
                         det.get('x2'), det.get('y2')],
                'result': result,
            })
            by_source[result['source']] += 1
            if result.get('tag'):
                by_tag[result['tag']] += 1

    print()
    print(f'{"="*70}')
    print(f'Neck-size waterfall results')
    print(f'{"="*70}')
    print(f'Total detections processed: {len(all_results)}')
    print()
    print(f'By source:')
    for src, n in by_source.most_common():
        print(f'  {n:5d}  {src}')

    if by_tag:
        print()
        print(f'Top tags with neck sizes:')
        for tag, n in by_tag.most_common(10):
            # Find a sample result for this tag
            sample = next(
                r for r in all_results if r['result'].get('tag') == tag
            )
            print(f'  {n:4d}× {tag:6s} → {sample["result"]["neck_size"]} '
                  f'(conf={sample["result"]["confidence"]:.2f})')

    # Quick confidence histogram
    confs = [r['result']['confidence'] for r in all_results
             if r['result']['confidence'] > 0]
    if confs:
        print()
        print(f'Confidence histogram (where neck size found):')
        buckets = [0.0, 0.15, 0.50, 0.80, 1.01]
        labels = ['drop', '?+flag', 'verify', 'auto']
        for i in range(len(buckets) - 1):
            count = sum(1 for c in confs
                       if buckets[i] <= c < buckets[i+1])
            print(f'  {buckets[i]:.2f}-{buckets[i+1]:.2f} '
                  f'[{labels[i]:8s}]: {count}')

    if args.out:
        Path(args.out).write_text(
            json.dumps(all_results, indent=2, default=str),
            encoding='utf-8',
        )
        print()
        print(f'Wrote {args.out}')
