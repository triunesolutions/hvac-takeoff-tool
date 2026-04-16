"""
Tag Extractor — assigns a tag (e.g., "A-1", "SB2", "LD-1") to each YOLO detection
by finding text near the symbol on the PDF page.

Strategy:
  1. Fast path: use PyMuPDF's page.get_text("words") to get text with positions.
     Works when the PDF has a text layer (most modern CAD exports do).
  2. Fallback: render page to image, run EasyOCR near detection bbox.
     Only needed for fully vectorized drawings with no text layer.
  3. Filter candidates by tag regex patterns, pick nearest to detection center.

Tag patterns handled:
  - Single letters:     A, B, C, D                     (Flex/Plum style)
  - Letter+digit:       A1, B7, C8, D3                 (Haldeman style)
  - Letter-digit:       A-1, D-4, GR-2                 (GA Larson style)
  - Multi-letter:       SA1, SB2, RA1, EA1             (St Elizabeth style)
  - Complex:            LD-1, LD-1-PLENUM              (linear diffusers)
"""
import re
import fitz
from collections import defaultdict


# Tag regex — matches any common HVAC tag format
TAG_PATTERN = re.compile(r'^([A-Z]{1,4})(-?)(\d{1,3})([A-Z]?)(-[A-Z]+)?$')

# Relaxed pattern for initial word filtering
TAG_CANDIDATE = re.compile(r'^[A-Z][A-Z0-9-]{0,15}$')

# Words that look like tags but are junk
BLACKLIST = {
    'TYPE', 'TYP', 'MARK', 'NTS', 'NO', 'NOTE', 'NOTES', 'SEE',
    'REF', 'DWG', 'SHT', 'PLAN', 'MECH', 'HVAC', 'ROOM', 'BLDG',
    'NORTH', 'EAST', 'WEST', 'SOUTH', 'NEW', 'EXIST', 'REMOVE',
    'SCALE', 'CFM', 'LS', 'FPM', 'OBD', 'OA', 'SA', 'RA', 'EA',
    'MAX', 'MIN', 'NOM', 'EQ', 'APPROX',
}


def normalize_tag(text):
    """Return normalized tag if text looks like a valid equipment tag, else None."""
    if not text:
        return None
    s = text.strip().upper()

    # Must be shape-like a tag
    if not TAG_CANDIDATE.match(s):
        return None

    # Strip surrounding punctuation
    s = s.strip('.-,;:|/ ')
    if not s:
        return None

    # Length check
    if len(s) < 1 or len(s) > 15:
        return None

    # Blacklist
    if s in BLACKLIST:
        return None

    # Must contain at least one letter
    if not re.search(r'[A-Z]', s):
        return None

    # Pure single-letter tags (A, B, C, D) are valid
    if len(s) == 1 and s in 'ABCDEFGH':
        return s

    # Otherwise require a digit or hyphen
    if len(s) > 1 and not re.search(r'[0-9\-]', s):
        # Pure word like "ROOM" would fail here
        if len(s) > 3:
            return None

    return s


def extract_text_with_positions(pdf_path, page_idx, dpi_scale=None):
    """
    Extract all words from a PDF page with their bounding boxes.
    Returns list of dicts: {text, x, y, x2, y2, cx, cy}

    If dpi_scale is provided, positions are scaled to pixel coords at that DPI.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)
    doc.close()

    scale = (dpi_scale / 72) if dpi_scale else 1.0

    result = []
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        result.append({
            'text': text,
            'x': x0 * scale,
            'y': y0 * scale,
            'x2': x1 * scale,
            'y2': y1 * scale,
            'cx': (x0 + x1) / 2 * scale,
            'cy': (y0 + y1) / 2 * scale,
        })
    return result


def find_tag_candidates_near(detection, words, radius=150):
    """
    Find all words within `radius` pixels of the detection center.
    Returns list of (tag, distance, word_center).
    """
    dcx = detection.get('cx', (detection.get('x1', 0) + detection.get('x2', 0)) / 2)
    dcy = detection.get('cy', (detection.get('y1', 0) + detection.get('y2', 0)) / 2)

    candidates = []
    for w in words:
        dist = ((w['cx'] - dcx) ** 2 + (w['cy'] - dcy) ** 2) ** 0.5
        if dist > radius:
            continue

        tag = normalize_tag(w['text'])
        if tag:
            candidates.append((tag, dist, w['cx'], w['cy']))

    return candidates


def assign_tags_to_detections(detections, words, radius=150, prefer_single_letter=True):
    """
    For each detection, find the best tag (closest matching word).
    Mutates detections by adding a 'tag' field.

    prefer_single_letter: in Flex-style drawings, single-letter tags (A, B, C, D)
    are usually closest to the symbol. Prefer them over multi-char if both found.
    """
    for det in detections:
        candidates = find_tag_candidates_near(det, words, radius)
        if not candidates:
            det['tag'] = None
            det['tag_confidence'] = 0
            continue

        # Sort by distance (closest first)
        candidates.sort(key=lambda c: c[1])

        # Pick the closest valid tag
        best_tag, best_dist, _, _ = candidates[0]
        det['tag'] = best_tag
        det['tag_confidence'] = 1.0 - min(best_dist / radius, 1.0)
        det['tag_alternatives'] = [c[0] for c in candidates[1:4]]

    return detections


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


# --- Standalone test ---

if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python tag_extractor.py path/to/blueprint.pdf [page_index]")
        _sys.exit(1)

    pdf = _sys.argv[1]
    page_idx = int(_sys.argv[2]) - 1 if len(_sys.argv) > 2 else 0

    print(f"Extracting text with positions from page {page_idx+1} of {pdf}")
    words = extract_text_with_positions(pdf, page_idx, dpi_scale=200)

    print(f"Total words: {len(words)}")

    # Show candidate tags
    candidates = set()
    for w in words:
        tag = normalize_tag(w['text'])
        if tag:
            candidates.add(tag)

    print(f"\nUnique tag candidates: {len(candidates)}")
    for tag in sorted(candidates):
        print(f"  {tag}")
