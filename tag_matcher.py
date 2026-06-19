"""
Schedule-guided tag matcher.

Given:
  - A list of valid tags from the project's schedule (e.g., ["A", "B", "C", "D"])
  - A rendered page image
  - YOLO detections with positions

Does:
  1. Runs EasyOCR on the page (one pass)
  2. Filters OCR results to ONLY tokens matching the valid tag list
  3. For each detection, assigns the closest matching tag

This is more accurate than unrestricted text extraction because it only
looks for tags we KNOW exist in the project's schedule.
"""
import re
import numpy as np
from collections import defaultdict


_easyocr_reader = None


def get_ocr_reader():
    """Lazy-init EasyOCR (loads ~100MB model on first call)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


def _normalize_for_match(s):
    """Normalize for fuzzy matching: uppercase, strip punctuation/spaces."""
    if not s:
        return ''
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def ocr_page(img, conf_threshold=0.4):
    """
    Run EasyOCR on a page image. Returns list of
    {text, cx, cy, bbox, confidence}
    """
    reader = get_ocr_reader()
    results = reader.readtext(img)

    words = []
    for bbox, text, conf in results:
        if conf < conf_threshold:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        words.append({
            'text': text.strip(),
            'cx': (min(xs) + max(xs)) / 2,
            'cy': (min(ys) + max(ys)) / 2,
            'x1': min(xs), 'y1': min(ys),
            'x2': max(xs), 'y2': max(ys),
            'conf': conf,
        })
    return words


_bubble_model = None


def get_bubble_model(model_path='models/hvac_tag_detector_v1.pt'):
    """Lazy-init the YOLO tag-bubble detector. Returns None if not available."""
    global _bubble_model
    if _bubble_model is False:
        return None  # cached miss
    if _bubble_model is None:
        try:
            from ultralytics import YOLO
            from pathlib import Path
            # Resolve relative paths against this file's directory (repo root)
            # so the model loads correctly when called from a subprocess
            # whose cwd is somewhere else (e.g. saas/backend/).
            p = Path(model_path)
            if not p.is_absolute():
                here = Path(__file__).resolve().parent
                p = here / model_path
            if not p.exists():
                import sys as _s
                print(f"  [tag_matcher] bubble model not found at {p}", file=_s.stderr, flush=True)
                _bubble_model = False
                return None
            _bubble_model = YOLO(str(p))
        except Exception as e:
            import sys as _s
            print(f"  [tag_matcher] failed to load bubble model: {e}", file=_s.stderr, flush=True)
            _bubble_model = False
            return None
    return _bubble_model


def detect_bubbles_on_page(img, conf=0.25, tile=320, overlap=80):
    """Run the tag-bubble detector across a full page (tiled). Returns list of
    {x1,y1,x2,y2,cx,cy,conf} for each detected tag_bubble (cls=1 only)."""
    model = get_bubble_model()
    if model is None:
        return []
    h, w = img.shape[:2]
    step = tile - overlap
    bubbles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            xe, ye = min(x + tile, w), min(y + tile, h)
            xs, ys = max(0, xe - tile), max(0, ye - tile)
            crop = img[ys:ye, xs:xe]
            if crop.size == 0:
                continue
            results = model.predict(crop, conf=conf, imgsz=tile, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != 1:   # only tag_bubble class
                        continue
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    bubbles.append({
                        'x1': bx1 + xs, 'y1': by1 + ys,
                        'x2': bx2 + xs, 'y2': by2 + ys,
                        'cx': (bx1 + bx2) / 2 + xs,
                        'cy': (by1 + by2) / 2 + ys,
                        'conf': float(box.conf[0]),
                    })
    # Simple NMS — drop bubble bboxes whose center is within 8 px of a higher-conf one
    bubbles.sort(key=lambda b: -b['conf'])
    keep = []
    for b in bubbles:
        if any(abs(b['cx'] - k['cx']) < 8 and abs(b['cy'] - k['cy']) < 8 for k in keep):
            continue
        keep.append(b)
    return keep


def ocr_bubble_crops(img, bubbles, pad=12, upscale=2.0, conf_threshold=0.2):
    """OCR the tight crop for each bubble bbox (with small padding + upscale).
    Returns each bubble enriched with {'text', 'ocr_conf'}."""
    if not bubbles:
        return bubbles
    import cv2
    reader = get_ocr_reader()
    out = []
    h, w = img.shape[:2]
    for b in bubbles:
        x1 = max(0, int(b['x1']) - pad)
        y1 = max(0, int(b['y1']) - pad)
        x2 = min(w, int(b['x2']) + pad)
        y2 = min(h, int(b['y2']) + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if upscale != 1.0:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            results = reader.readtext(
                crop,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-',
                text_threshold=0.4, low_text=0.2,
            )
        except Exception:
            continue
        # Concatenate words in this single bubble (handles "CU-1" split as "CU" "-" "1")
        toks = [(t, c) for _, t, c in results if c >= conf_threshold]
        if not toks:
            continue
        text = ''.join(t for t, _ in toks).strip()
        if not text:
            continue
        b2 = dict(b)
        b2['text'] = text
        b2['ocr_conf'] = sum(c for _, c in toks) / len(toks)
        out.append(b2)
    return out


def merge_split_bubbles(bubbles, max_dx=60, max_dy=80):
    """Some drawings draw tags as two stacked bubbles — prefix on top
    ("CD"), suffix below ("A") — instead of a single "CD-A" bubble. The
    detector finds both halves; OCR reads each half correctly; matching
    against schedule tags fails because neither half alone is a valid tag.

    This helper appends synthetic merged bubbles for every pair of OCR'd
    bubbles whose centers are within a small neighborhood. The synthetic
    bubble carries the concatenation of the two texts (both orders) at the
    midpoint, with conf set to the min of the pair so longer-distance pairs
    are penalized. Real (single-bubble) tags still match first because
    they're closer to the equipment center.

    The synthetic bubbles never *replace* the originals — they're appended,
    so single-bubble matches still work. Bubbles whose OCR text is already
    a multi-character tag (contains a digit or dash) are not paired —
    pairing only triggers for short alpha prefixes that can't stand alone.
    """
    if not bubbles or len(bubbles) < 2:
        return bubbles
    out = list(bubbles)
    for i, b1 in enumerate(bubbles):
        t1 = (b1.get('text') or '').strip()
        if not t1 or len(t1) > 4:
            continue
        # Pair only when at least one side is alpha-only (the prefix half).
        # If t1 already contains a digit/dash it's likely a complete tag.
        t1_alpha = t1.isalpha()
        for j, b2 in enumerate(bubbles):
            if i == j:
                continue
            t2 = (b2.get('text') or '').strip()
            if not t2 or len(t2) > 4:
                continue
            if not (t1_alpha or t2.isalpha()):
                continue
            dx = abs(b1['cx'] - b2['cx'])
            dy = abs(b1['cy'] - b2['cy'])
            if dx > max_dx or dy > max_dy:
                continue
            if dx == 0 and dy == 0:
                continue
            # Generate both concatenation orders so we don't depend on which
            # half the OCR got first. The match-lookup is normalized so
            # "CD"+"A" and "CD-A" collapse the same way.
            for combined in (f"{t1}-{t2}", f"{t2}-{t1}"):
                merged = {
                    'x1': min(b1['x1'], b2['x1']),
                    'y1': min(b1['y1'], b2['y1']),
                    'x2': max(b1['x2'], b2['x2']),
                    'y2': max(b1['y2'], b2['y2']),
                    'cx': (b1['cx'] + b2['cx']) / 2,
                    'cy': (b1['cy'] + b2['cy']) / 2,
                    'conf': min(b1.get('conf', 1.0), b2.get('conf', 1.0)),
                    'text': combined,
                    'ocr_conf': min(b1.get('ocr_conf', 1.0), b2.get('ocr_conf', 1.0)),
                    'merged_from': (t1, t2),
                }
                out.append(merged)
    return out


def ocr_near_detection(img, det, crop_size=180, conf_threshold=0.3):
    """
    Crop a region around a detection and OCR just that crop.
    Works better for single-character tags (A, B, C, D) that get missed
    in full-page OCR because they're too small relative to the page.

    Returns list of words with coords in the ORIGINAL image coordinate system.
    """
    reader = get_ocr_reader()

    dcx = det.get('cx', (det.get('x1', 0) + det.get('x2', 0)) / 2)
    dcy = det.get('cy', (det.get('y1', 0) + det.get('y2', 0)) / 2)

    h, w = img.shape[:2]
    x1 = max(0, int(dcx - crop_size))
    y1 = max(0, int(dcy - crop_size))
    x2 = min(w, int(dcx + crop_size))
    y2 = min(h, int(dcy + crop_size))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    results = reader.readtext(crop)
    words = []
    for bbox, text, conf in results:
        if conf < conf_threshold:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        # Translate back to page coords
        words.append({
            'text': text.strip(),
            'cx': (min(xs) + max(xs)) / 2 + x1,
            'cy': (min(ys) + max(ys)) / 2 + y1,
            'x1': min(xs) + x1, 'y1': min(ys) + y1,
            'x2': max(xs) + x1, 'y2': max(ys) + y1,
            'conf': conf,
        })
    return words


def tag_detections_by_cropped_ocr(img, detections, valid_tags, crop_size=180,
                                    max_distance=150):
    """
    For each detection, OCR a crop around it and match against valid tags.
    Returns stats dict.

    This is the recommended path for single-letter tags that get missed
    by full-page OCR.
    """
    if not valid_tags:
        for d in detections:
            d['tag'] = None
            d['tag_confidence'] = 0
        return {'tagged': 0, 'total': len(detections)}

    tagged_count = 0
    for det in detections:
        words = ocr_near_detection(img, det, crop_size=crop_size)
        matches = match_valid_tags(words, valid_tags)

        if not matches:
            det['tag'] = None
            det['tag_confidence'] = 0
            continue

        # Closest match
        dcx = det.get('cx', 0)
        dcy = det.get('cy', 0)
        best = None
        best_dist = float('inf')
        for tag, w in matches:
            d = ((w['cx'] - dcx) ** 2 + (w['cy'] - dcy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = (tag, w)

        if best and best_dist <= max_distance:
            det['tag'] = best[0]
            det['tag_confidence'] = 1.0 - min(best_dist / max_distance, 1.0)
            tagged_count += 1
        else:
            det['tag'] = None
            det['tag_confidence'] = 0

    return {'tagged': tagged_count, 'total': len(detections)}


def match_valid_tags(ocr_words, valid_tags):
    """
    Filter OCR results to tokens matching valid tags from the schedule.
    Uses fuzzy matching (uppercase, ignores non-alphanumeric).

    Returns a list of (tag, word_dict) pairs.
    """
    if not valid_tags:
        return []

    # Build normalized lookup
    tag_lookup = {}
    for tag in valid_tags:
        normalized = _normalize_for_match(tag)
        if normalized:
            tag_lookup[normalized] = tag

    # Also add "A" -> "A" for single-letter tags in case OCR reads them
    # standalone without the circle bubble context
    matches = []
    for w in ocr_words:
        normalized = _normalize_for_match(w['text'])
        if not normalized:
            continue
        if normalized in tag_lookup:
            matches.append((tag_lookup[normalized], w))
            continue

        # Partial match: sometimes OCR reads "A 240" as one word — split and try
        parts = re.findall(r'[A-Z0-9]+', w['text'].upper())
        for part in parts:
            if part in tag_lookup:
                matches.append((tag_lookup[part], w))
                break

    return matches


def assign_tags_to_detections(detections, tag_matches, max_distance=200):
    """
    For each detection, find the closest tag match within max_distance pixels.
    Mutates detections by setting 'tag'.
    """
    if not tag_matches:
        for d in detections:
            d['tag'] = None
            d['tag_confidence'] = 0
        return detections

    # Track which matches have been used (each tag instance matches at most once)
    used = set()

    # Pair each detection with closest unused tag match
    pairs = []  # (detection_idx, match_idx, distance)
    for di, d in enumerate(detections):
        dcx = d.get('cx', (d.get('x1', 0) + d.get('x2', 0)) / 2)
        dcy = d.get('cy', (d.get('y1', 0) + d.get('y2', 0)) / 2)
        for mi, (tag, w) in enumerate(tag_matches):
            dist = ((w['cx'] - dcx) ** 2 + (w['cy'] - dcy) ** 2) ** 0.5
            if dist <= max_distance:
                pairs.append((di, mi, dist, tag, w))

    # Greedy assignment: closest pairs first
    pairs.sort(key=lambda p: p[2])
    assigned_det = set()
    for di, mi, dist, tag, w in pairs:
        if di in assigned_det or mi in used:
            continue
        detections[di]['tag'] = tag
        detections[di]['tag_confidence'] = 1.0 - min(dist / max_distance, 1.0)
        assigned_det.add(di)
        used.add(mi)

    # Mark unassigned
    for di, d in enumerate(detections):
        if di not in assigned_det:
            d['tag'] = None
            d['tag_confidence'] = 0

    return detections


def run_schedule_guided_tagging(img, detections, valid_tags, max_distance=200):
    """
    Full pipeline:
    1. OCR the page image
    2. Filter to valid tags from schedule
    3. Assign tags to detections by proximity

    Returns (detections_with_tags, ocr_word_count, match_count)
    """
    if not valid_tags:
        return detections, 0, 0

    ocr_words = ocr_page(img)
    matches = match_valid_tags(ocr_words, valid_tags)
    assign_tags_to_detections(detections, matches, max_distance)

    return detections, len(ocr_words), len(matches)


if __name__ == "__main__":
    # Quick test
    import sys
    import fitz
    import cv2
    from schedule_parser import parse_pdf_schedules

    if len(sys.argv) < 2:
        print("Usage: python tag_matcher.py path/to/blueprint.pdf [page_index]")
        sys.exit(1)

    pdf = sys.argv[1]
    page_idx = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 5

    print(f"Step 1: Parsing schedule from {pdf}")
    schedules, marks, details, legend, summary = parse_pdf_schedules(pdf)
    print(f"  Found {len(marks)} valid tags: {marks}")

    if not marks:
        print("No tags in schedule — can't do schedule-guided matching")
        sys.exit(0)

    print(f"\nStep 2: Rendering page {page_idx+1}")
    doc = fitz.open(pdf)
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()

    print(f"\nStep 3: Running OCR on full page ({img.shape[1]}x{img.shape[0]} px)...")
    ocr_words = ocr_page(img)
    print(f"  OCR found {len(ocr_words)} text regions")

    print(f"\nStep 4: Matching to schedule tags...")
    matches = match_valid_tags(ocr_words, marks)
    print(f"  Matched {len(matches)} tag instances on the drawing")

    tag_counts = defaultdict(int)
    for tag, w in matches:
        tag_counts[tag] += 1

    print(f"\n  Tag instances found on drawing:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"    {tag}: {count}")
