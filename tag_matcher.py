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


_easyocr_reader = None


def get_ocr_reader():
    """Lazy-init EasyOCR (loads ~100MB model on first call).

    Auto-uses the GPU when CUDA is available (Kaggle/Colab T4) — EasyOCR is the
    pipeline's heaviest stage, so this is the bulk of the cloud speedup. Falls
    back to CPU locally (cuda.is_available() is False), so behaviour is unchanged
    on a CPU-only machine."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        try:
            import torch
            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False
        _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
    return _easyocr_reader


def _normalize_for_match(s):
    """Normalize for fuzzy matching: uppercase, strip punctuation/spaces."""
    if not s:
        return ''
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


# Restrict OCR to tag-relevant characters — cuts noise tokens on busy crops.
_TAG_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'


def preprocess_crop_for_ocr(crop, upscale=3.0, binarize=True):
    """Upscale + Otsu-binarize a small crop to maximize OCR recall on tiny tag
    bubbles. Ports the preprocessing proven in label_tag_bubbles_ocr.py (3x
    upscale + Otsu lifted the bubble-labeling hit rate past plain upscaling),
    using OpenCV's built-in Otsu rather than the hand-rolled histogram loop.

    Returns (processed_bgr_image, effective_scale). effective_scale is the total
    pixel scaling applied, so callers that read OCR bbox coords off the processed
    crop can divide by it to get back to original-crop pixels.
    """
    import cv2
    if crop is None or getattr(crop, 'size', 0) == 0:
        return crop, 1.0
    scale = float(upscale) if upscale else 1.0
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    if not binarize:
        return crop, scale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), scale


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
            if not Path(model_path).exists():
                _bubble_model = False
                return None
            _bubble_model = YOLO(model_path)
        except Exception:
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


def ocr_bubble_crops(img, bubbles, pad=12, upscale=3.0, conf_threshold=0.2,
                     binarize=True):
    """OCR the tight crop for each bubble bbox (with small padding + upscale +
    Otsu binarize). Returns each bubble enriched with {'text', 'ocr_conf'}.
    Coords come from the bubble bbox (page space), so the preprocessing only
    affects the read text, not positions."""
    if not bubbles:
        return bubbles
    h, w = img.shape[:2]
    boxes = [(max(0, int(b['x1']) - pad), max(0, int(b['y1']) - pad),
              min(w, int(b['x2']) + pad), min(h, int(b['y2']) + pad))
             for b in bubbles]
    # One batched OCR pass over all bubble crops instead of a readtext per crop.
    word_lists = ocr_crops_batched(img, boxes, upscale=upscale,
                                   conf_threshold=conf_threshold,
                                   text_threshold=0.4, low_text=0.2)
    out = []
    for b, words in zip(bubbles, word_lists):
        if not words:
            continue
        # Concatenate words in this single bubble (handles "CU-1" split as "CU" "-" "1"),
        # left-to-right by x so the characters keep their order.
        words = sorted(words, key=lambda wd: wd['cx'])
        text = ''.join(wd['text'] for wd in words).strip()
        if not text:
            continue
        b2 = dict(b)
        b2['text'] = text
        b2['ocr_conf'] = sum(wd['conf'] for wd in words) / len(words)
        out.append(b2)
    return out


def ocr_crops_batched(img, boxes, upscale=3.0, conf_threshold=0.25,
                      text_threshold=0.3, low_text=0.3, allowlist=None,
                      batch_size=16, deadline=None):
    """OCR many small crops in batched EasyOCR passes instead of one readtext
    per crop. `boxes` is a list of (x1,y1,x2,y2) in page space; returns a list
    aligned with `boxes`, each entry a list of word dicts (text + page-space
    coords). Crops are upscaled + Otsu-binarized (the proven single-char prep),
    padded to a uniform size per batch so EasyOCR can process them together.

    This replaces the per-detection readtext loop that made dense plans (200+
    detections) take tens of minutes — one batched pass per ~16 crops."""
    import numpy as np
    import cv2
    reader = get_ocr_reader()
    if allowlist is None:
        allowlist = _TAG_ALLOWLIST
    h, w = img.shape[:2]
    out = [[] for _ in boxes]

    # Pre-crop + preprocess; remember offset/scale per box.
    prepped = []   # (idx, processed_bgr, scale, ox, oy) for non-empty crops
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        if x2i - x1i < 4 or y2i - y1i < 4:
            continue
        crop = img[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue
        proc, scale = preprocess_crop_for_ocr(crop, upscale=upscale, binarize=True)
        prepped.append((idx, proc, scale, x1i, y1i))

    def _run_batch(batch):
        if not batch:
            return
        # Pad each crop to the batch's max H×W (top-left aligned, white bg so
        # binarized black text stays legible) → uniform size for batched OCR.
        maxh = max(p[1].shape[0] for p in batch)
        maxw = max(p[1].shape[1] for p in batch)
        imgs = []
        for _, proc, _, _, _ in batch:
            ph, pw = proc.shape[:2]
            canvas = np.full((maxh, maxw, 3), 255, dtype=np.uint8)
            canvas[:ph, :pw] = proc
            imgs.append(canvas)
        try:
            results = reader.readtext_batched(
                imgs, n_width=maxw, n_height=maxh, allowlist=allowlist,
                text_threshold=text_threshold, low_text=low_text)
        except Exception:
            # Fall back to per-crop readtext if batched API misbehaves.
            results = []
            for canvas in imgs:
                try:
                    results.append(reader.readtext(
                        canvas, allowlist=allowlist,
                        text_threshold=text_threshold, low_text=low_text))
                except Exception:
                    results.append([])
        for (idx, _proc, scale, ox, oy), res in zip(batch, results):
            for bbox, text, conf in res:
                if conf < conf_threshold or not str(text).strip():
                    continue
                xs = [p[0] / scale for p in bbox]
                ys = [p[1] / scale for p in bbox]
                out[idx].append({
                    'text': str(text).strip(),
                    'cx': (min(xs) + max(xs)) / 2 + ox,
                    'cy': (min(ys) + max(ys)) / 2 + oy,
                    'conf': float(conf),
                })

    import time as _time
    for i in range(0, len(prepped), batch_size):
        if deadline is not None and _time.time() > deadline:
            break   # time-capped: leave remaining crops un-OCR'd (partial result)
        _run_batch(prepped[i:i + batch_size])
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

    # Upscale + Otsu-binarize before OCR (WS1.4) — single-char tags inside small
    # bubbles are the dominant miss; this is the same preprocessing proven on the
    # bubble-labeling corpus. The allowlist drops noise tokens.
    proc, scale = preprocess_crop_for_ocr(crop, upscale=3.0, binarize=True)
    try:
        results = reader.readtext(proc, allowlist=_TAG_ALLOWLIST)
    except Exception:
        return []
    words = []
    for bbox, text, conf in results:
        if conf < conf_threshold:
            continue
        # bbox coords are in upscaled-crop space → divide by scale, then offset
        # back into the original page coordinate system.
        xs = [p[0] / scale for p in bbox]
        ys = [p[1] / scale for p in bbox]
        words.append({
            'text': text.strip(),
            'cx': (min(xs) + max(xs)) / 2 + x1,
            'cy': (min(ys) + max(ys)) / 2 + y1,
            'x1': min(xs) + x1, 'y1': min(ys) + y1,
            'x2': max(xs) + x1, 'y2': max(ys) + y1,
            'conf': conf,
        })
    return words


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


