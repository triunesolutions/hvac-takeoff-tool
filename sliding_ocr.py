"""
Positioned tag scanner.

Finds every occurrence of every schedule tag on a plan page, with pixel
coordinates. Two paths:

  1. Text-layer pass (zero-cost, exact): PyMuPDF `get_text("words")` gives
     us every text span with its bbox. We assemble adjacent spans into tag
     candidates and strict-match against the schedule tag list. Handles
     multi-line bubbles where "VAV" and "23" are on separate lines.
  2. Sliding-window OCR fallback (slow, fuzzy): for raster pages with no
     text layer, render and tile-scan with RapidOCR.

Public API:
    scan_page_for_tags(pdf_path, page_index, valid_tags, dpi=200)
      → list of {tag, cx, cy, x1, y1, x2, y2, conf, source}
"""
import re
from collections import defaultdict

import fitz
import numpy as np
import cv2


TAG_TOKEN_RE = re.compile(r'^[A-Za-z]{1,5}-?\d{0,3}[A-Za-z]?(?:-\d{1,3})?$')


def _normalize_tag(s):
    if not s:
        return ''
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def _build_tag_lookup(valid_tags):
    """{normalized: original}. Also expand multi-part tags to their
    component pieces so we can match split-line bubbles."""
    lookup = {}
    for t in valid_tags:
        n = _normalize_tag(t)
        if n:
            lookup[n] = t
    return lookup


def _is_tag_shape(token):
    if not token or len(token) > 12:
        return False
    return bool(TAG_TOKEN_RE.match(token))


# ───────────────────────── TEXT-LAYER PATH ─────────────────────────────────


def _words_with_bboxes(pdf_path, page_index):
    """Return list of (text, x1, y1, x2, y2) in PDF point units."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    words = page.get_text("words")
    doc.close()
    # PyMuPDF "words" tuple: (x0, y0, x1, y1, text, block, line, word)
    return [(w[4], w[0], w[1], w[2], w[3]) for w in words]


def _scan_text_layer(pdf_path, page_index, valid_tags, dpi=200,
                     max_join_dist=40, verbose=False):
    """Find tag occurrences in the text layer. Returns hits in image pixels
    (caller renders the page at the same DPI to overlay).

    Two passes:
      1. Single-word tokens that already match a schedule tag.
      2. Two-word adjacency: combine an alpha word with a nearby digit word
         when their concatenation matches a schedule tag. Adjacency is any
         neighbour within max_join_dist pt — bubbles place pieces left/right
         OR above/below depending on the drawing.
    """
    words = _words_with_bboxes(pdf_path, page_index)
    if not words:
        return []

    lookup = _build_tag_lookup(valid_tags)
    if not lookup:
        return []

    scale = dpi / 72.0
    hits = []

    def _to_px(x1, y1, x2, y2):
        return x1 * scale, y1 * scale, x2 * scale, y2 * scale

    # Pass 1: single-token match — "CU-1", "EF-1A", "A", etc.
    for text, x1, y1, x2, y2 in words:
        clean = text.strip().strip('.,;:()[]')
        if not _is_tag_shape(clean):
            continue
        norm = _normalize_tag(clean)
        if norm in lookup:
            px1, py1, px2, py2 = _to_px(x1, y1, x2, y2)
            hits.append({
                'tag': lookup[norm],
                'x1': px1, 'y1': py1, 'x2': px2, 'y2': py2,
                'cx': (px1 + px2) / 2, 'cy': (py1 + py2) / 2,
                'conf': 1.0,
                'source': 'text_layer_single',
                'raw_text': clean,
            })

    # Pass 2: two-token combos with proximity-based pairing.
    alpha_words = []
    digit_words = []
    mixed_words = []  # like "1A", "N" — short letter+digit fragments
    for text, x1, y1, x2, y2 in words:
        clean = text.strip().strip('.,;:()[]')
        if not clean or len(clean) > 5:
            continue
        if clean.isalpha():
            alpha_words.append((clean, x1, y1, x2, y2))
        elif clean.isdigit():
            digit_words.append((clean, x1, y1, x2, y2))
        elif re.match(r'^[A-Za-z0-9]+$', clean):
            mixed_words.append((clean, x1, y1, x2, y2))

    # Build a combined "candidate-suffix" list = digits + mixed (so "VAV-N"
    # is reachable as alpha "VAV" + alpha "N", but also alpha "VAV" + mixed
    # things). For "VAV-N", "N" is alpha — pair alpha with alpha too.
    suffix_pool = digit_words + mixed_words + alpha_words

    for atext, ax1, ay1, ax2, ay2 in alpha_words:
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        a_w = ax2 - ax1
        a_h = ay2 - ay1
        for stext, sx1, sy1, sx2, sy2 in suffix_pool:
            if stext == atext and ax1 == sx1 and ay1 == sy1:
                continue  # skip self
            scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            dx = abs(scx - acx)
            dy = abs(scy - acy)
            # Within bubble-sized neighbourhood: ~3× the larger word's width
            # horizontally, and within max_join_dist vertically.
            if dx > max(a_w, sx2 - sx1) * 3 + 5:
                continue
            if dy > max_join_dist:
                continue
            if dx == 0 and dy == 0:
                continue
            combined = atext + '-' + stext
            norm = _normalize_tag(combined)
            if norm not in lookup:
                continue
            jx1 = min(ax1, sx1); jy1 = min(ay1, sy1)
            jx2 = max(ax2, sx2); jy2 = max(ay2, sy2)
            px1, py1, px2, py2 = _to_px(jx1, jy1, jx2, jy2)
            hits.append({
                'tag': lookup[norm],
                'x1': px1, 'y1': py1, 'x2': px2, 'y2': py2,
                'cx': (px1 + px2) / 2, 'cy': (py1 + py2) / 2,
                'conf': 1.0,
                'source': 'text_layer_paired',
                'raw_text': combined,
            })

    if verbose:
        from collections import Counter
        c = Counter(h['tag'] for h in hits)
        print(f"  [text-layer] {len(hits)} hits across {len(c)} tags: {dict(c)}")
    return _dedup_hits(hits, dedup_dist=10)


# ───────────────────────── OCR FALLBACK PATH ───────────────────────────────

_rapidocr_reader = None


def _get_reader():
    global _rapidocr_reader
    if _rapidocr_reader is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapidocr_reader = RapidOCR()
    return _rapidocr_reader


def _tile_coords(w, h, tile, overlap):
    step = tile - overlap
    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, min(x0 + tile, w), min(y0 + tile, h)


def _bbox_from_quad(quad):
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2), float((x1 + x2) / 2), float((y1 + y2) / 2)


def _sliding_window_ocr(img, valid_tags, *, tile=480, overlap=120,
                        upscale=2.0, conf_threshold=0.4, verbose=False):
    if not valid_tags:
        return []
    h, w = img.shape[:2]
    lookup = _build_tag_lookup(valid_tags)
    reader = _get_reader()
    hits = []
    tiles = list(_tile_coords(w, h, tile, overlap))
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        if upscale != 1.0:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            res, _ = reader(crop)
        except Exception:
            continue
        if not res:
            continue
        tile_matches = 0
        for entry in res:
            try:
                quad, text, conf = entry[0], str(entry[1]), float(entry[2])
            except Exception:
                continue
            if conf < conf_threshold:
                continue
            for raw_tok in re.split(r'[\s|/,;]+', text.strip()):
                if not _is_tag_shape(raw_tok):
                    continue
                norm = _normalize_tag(raw_tok)
                if norm not in lookup:
                    continue
                bx1, by1, bx2, by2, cx, cy = _bbox_from_quad(quad)
                if upscale != 1.0:
                    bx1, by1 = bx1 / upscale, by1 / upscale
                    bx2, by2 = bx2 / upscale, by2 / upscale
                    cx, cy = cx / upscale, cy / upscale
                hits.append({
                    'tag': lookup[norm],
                    'x1': bx1 + x0, 'y1': by1 + y0,
                    'x2': bx2 + x0, 'y2': by2 + y0,
                    'cx': cx + x0, 'cy': cy + y0,
                    'conf': conf,
                    'source': 'ocr_tile',
                    'raw_text': raw_tok,
                })
                tile_matches += 1
        if verbose and tile_matches:
            print(f"  tile {ti+1}/{len(tiles)}: {tile_matches} match")
    return _dedup_hits(hits, dedup_dist=40)


# ───────────────────────── DEDUP + UNIFIED API ─────────────────────────────


def _dedup_hits(hits, dedup_dist):
    if not hits:
        return hits
    by_tag = defaultdict(list)
    for h in hits:
        by_tag[h['tag']].append(h)
    out = []
    for group in by_tag.values():
        group.sort(key=lambda h: -h['conf'])
        kept = []
        for h in group:
            keep = True
            for k in kept:
                if (abs(h['cx'] - k['cx']) < dedup_dist
                        and abs(h['cy'] - k['cy']) < dedup_dist):
                    keep = False
                    break
            if keep:
                kept.append(h)
        out.extend(kept)
    return out


def scan_page_for_tags(pdf_path, page_index, valid_tags, dpi=200,
                       ocr_fallback=True, verbose=False):
    """Find every occurrence of every schedule tag on a single page.

    Text-layer-first: if the page has a usable text layer, use it (zero-cost,
    exact). Falls back to sliding-window RapidOCR for raster pages.

    Returns: list of {tag, cx, cy, x1, y1, x2, y2, conf, source, raw_text}.
    All coordinates in image pixels at the given DPI.
    """
    # First try the text layer
    hits = _scan_text_layer(pdf_path, page_index, valid_tags, dpi=dpi,
                            verbose=verbose)
    if hits:
        return hits

    if not ocr_fallback:
        return []

    # Fall back to rendering + sliding-window OCR
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    if verbose:
        print(f"  [ocr fallback] page rendered {img.shape[1]}x{img.shape[0]}")
    return _sliding_window_ocr(img, valid_tags, verbose=verbose)


if __name__ == '__main__':
    import argparse
    from collections import Counter
    from schedule_parser import parse_pdf_schedules

    ap = argparse.ArgumentParser()
    ap.add_argument('pdf')
    ap.add_argument('--page', type=int, required=True, help='1-indexed')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--no-ocr', action='store_true', help='Text-layer only')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    _, _, _, _, _, variables = parse_pdf_schedules(args.pdf)
    tags = sorted({v['tag'] for v in variables})
    print(f"Schedule tags ({len(tags)}): {tags}\n")

    hits = scan_page_for_tags(args.pdf, args.page - 1, tags, dpi=args.dpi,
                              ocr_fallback=not args.no_ocr, verbose=args.verbose)
    counts = Counter(h['tag'] for h in hits)
    sources = Counter(h['source'] for h in hits)
    print(f"=== {len(hits)} hits, {len(counts)} unique tags ===")
    for tag, n in counts.most_common():
        confs = [h['conf'] for h in hits if h['tag'] == tag]
        print(f"  {tag:<12} {n:>3}× (conf {min(confs):.2f}-{max(confs):.2f})")
    print(f"\nsources: {dict(sources)}")
    missing = [t for t in tags if t not in counts]
    if missing:
        print(f"\nNot found on this page: {missing}")
