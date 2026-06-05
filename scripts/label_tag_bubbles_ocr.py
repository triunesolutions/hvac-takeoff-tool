import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
"""
label_tag_bubbles_ocr.py - Auto-label tag bubbles by OCR'ing the existing crops.

Approach: for each crop in tag_dataset/labels.jsonl we already know the true
tag string. Run EasyOCR on the crop, normalize every detected token, and find
the one whose normalization equals the tag. Record its bbox in crop pixel
coords.

Advantages over text-layer search:
  - Works even when tags are CAD vector paths (the common case)
  - Small OCR area (320x320) → much higher accuracy than full-page OCR
  - Known target → tolerant to OCR misreads via normalization

Output: tag_bubble_labels.jsonl — appended mode so Ctrl-C / resume is safe.
Each row:
  {img: "images/proj/000123.png",
   tag: "A-1",
   bubble_rect_in_crop: [x0, y0, x1, y1] | None,
   reason: "hit" | "no_text" | "no_match",
   ocr_tokens: [[text, [x0,y0,x1,y1], conf], ...]   (only on miss, for debug)}
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image


_NORM_RE = re.compile(r'[^A-Z0-9]+')

# Common OCR confusions on CAD/stamp fonts — used for edit-distance 1 matching
_OCR_EQUIV = {
    'I': '1', 'L': '1', 'l': '1',
    'O': '0', 'Q': '0', 'D': '0',
    'S': '5', 'Z': '2', 'B': '8',
    'G': '6', 'T': '7',
}


def norm(s: str) -> str:
    return _NORM_RE.sub('', (s or '').upper())


def _edit_dist_le_1(a: str, b: str) -> bool:
    """Return True if a and b differ by at most 1 edit (sub/ins/del).
    Also considers OCR-confusion substitutions as zero-cost."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                continue
            # allow OCR-confusion substitutions without counting a diff
            if _OCR_EQUIV.get(ca) == cb or _OCR_EQUIV.get(cb) == ca:
                continue
            diffs += 1
            if diffs > 1:
                return False
        return True
    # one insertion/deletion
    short, long = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            j += 1
        else:
            return False
    return True


def preprocess_for_ocr(pil_img, upscale=3.0):
    """Upscale + Otsu binarize to maximize OCR recall on small bubbles."""
    w, h = pil_img.size
    if upscale != 1.0:
        pil_img = pil_img.resize((int(w * upscale), int(h * upscale)), Image.LANCZOS)
    arr = np.array(pil_img.convert('L'))  # grayscale
    # Otsu threshold
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sumB, wB, max_var, thresh = 0.0, 0, 0.0, 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    bw = (arr > thresh).astype(np.uint8) * 255
    rgb = np.stack([bw, bw, bw], axis=-1)
    return rgb


def best_match(ocr_results, target_norm):
    """Find the OCR token whose normalized text equals target.
    ocr_results: list of (bbox_polygon, text, conf) from EasyOCR.
    Returns (x0, y0, x1, y1) or None.
    """
    # Pass 1: exact normalized match
    # Pass 2: edit-distance ≤ 1 fuzzy match (only if exact fails)
    exact = []
    fuzzy = []
    for poly, text, conf in ocr_results:
        n = norm(text)
        if not n:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = (min(xs), min(ys), max(xs), max(ys), conf)
        if n == target_norm:
            exact.append(bbox)
        elif len(target_norm) >= 2 and _edit_dist_le_1(n, target_norm):
            fuzzy.append(bbox)

    if exact:
        exact.sort(key=lambda b: -b[4])
        return exact[0][:4]
    if fuzzy:
        fuzzy.sort(key=lambda b: -b[4])
        return fuzzy[0][:4]

    # Fallback: try concatenating adjacent tokens (tag split across words/bubbles)
    # Sort tokens by left-x then by top-y and try pairs/triples.
    toks = []
    for poly, text, conf in ocr_results:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        toks.append((min(xs), min(ys), max(xs), max(ys), norm(text), conf))
    toks.sort(key=lambda t: (round(t[1] / 10), t[0]))  # rough line grouping
    best = None
    best_conf = -1
    for i in range(len(toks)):
        concat = ''
        xs0, ys0, xs1, ys1 = None, None, None, None
        c_sum, c_n = 0.0, 0
        for j in range(i, min(i + 3, len(toks))):
            x0, y0, x1, y1, n, conf = toks[j]
            concat += n
            xs0 = x0 if xs0 is None else min(xs0, x0)
            ys0 = y0 if ys0 is None else min(ys0, y0)
            xs1 = x1 if xs1 is None else max(xs1, x1)
            ys1 = y1 if ys1 is None else max(ys1, y1)
            c_sum += conf
            c_n += 1
            if concat == target_norm:
                avg = c_sum / c_n
                if avg > best_conf:
                    best_conf = avg
                    best = (xs0, ys0, xs1, ys1)
                break
            if len(concat) > len(target_norm):
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='tag_dataset', help='crop dataset dir')
    ap.add_argument('--out', default='tag_bubble_labels.jsonl')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--random-sample', type=int, default=None,
                    help='sample N rows uniformly (for representative hit-rate measurement)')
    ap.add_argument('--resume', action='store_true',
                    help='skip crops already present in --out (match by img path)')
    ap.add_argument('--upscale', type=float, default=3.0,
                    help='upscale factor before OCR (tiny bubbles need this)')
    ap.add_argument('--no-binarize', action='store_true',
                    help='disable Otsu binarization preprocessing')
    ap.add_argument('--gpu', action='store_true', help='use CUDA if available')
    ap.add_argument('--save-debug-misses', type=int, default=0,
                    help='save N miss crops to debug_misses/ for inspection')
    args = ap.parse_args()

    # Lazy import so --help stays snappy
    import easyocr
    print(f"Initializing EasyOCR (gpu={args.gpu})...")
    reader = easyocr.Reader(['en'], gpu=args.gpu, verbose=False)
    print("Ready.")

    dataset = Path(args.dataset)
    labels_path = dataset / 'labels.jsonl'

    rows = []
    with open(labels_path, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} crop labels from {labels_path}")

    done = set()
    if args.resume and Path(args.out).exists():
        with open(args.out, encoding='utf-8') as f:
            for line in f:
                try:
                    done.add(json.loads(line)['img'])
                except Exception:
                    pass
        print(f"Resume: skipping {len(done)} already-done crops")

    if args.random_sample:
        import random
        random.seed(42)
        rows = random.sample(rows, min(args.random_sample, len(rows)))
        print(f"Random-sampled {len(rows)} rows")
    if args.limit:
        rows = rows[:args.limit]

    debug_dir = Path('debug_misses')
    if args.save_debug_misses:
        debug_dir.mkdir(exist_ok=True)

    hits, misses, errors, skipped = 0, 0, 0, 0
    t0 = time.time()
    debug_saved = 0

    mode = 'a' if args.resume else 'w'
    # buffering=1 → line-buffered; each json row flushes to disk on newline.
    # Critical: 24-hour run must survive a kill without losing progress.
    with open(args.out, mode, encoding='utf-8', buffering=1) as of:
        for idx, r in enumerate(rows):
            if r['img'] in done:
                skipped += 1
                continue

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                processed = hits + misses + errors
                rate = processed / max(1e-9, elapsed)
                rem = (len(rows) - skipped - processed) / max(1e-9, rate)
                print(f"  [{idx+1}/{len(rows)}] hits={hits} misses={misses} "
                      f"errors={errors}  {rate:.1f}/s  ~{rem/60:.1f}min left")

            tag = r.get('tag')
            if not tag:
                of.write(json.dumps({'img': r['img'], 'tag': None,
                                     'bubble_rect_in_crop': None,
                                     'reason': 'no_tag'}) + '\n')
                misses += 1
                continue

            img_path = dataset / r['img']
            try:
                im = Image.open(img_path).convert('RGB')
                if args.no_binarize:
                    if args.upscale != 1.0:
                        w, h = im.size
                        im = im.resize((int(w * args.upscale), int(h * args.upscale)),
                                       Image.LANCZOS)
                    arr = np.array(im)
                else:
                    arr = preprocess_for_ocr(im, upscale=args.upscale)
            except Exception as e:
                of.write(json.dumps({'img': r['img'], 'tag': tag,
                                     'bubble_rect_in_crop': None,
                                     'reason': f'img_error:{e}'}) + '\n')
                errors += 1
                continue

            try:
                # allowlist: restrict OCR to tag-relevant chars; reduces noise tokens
                # Thresholds lowered — we match to known targets so recall > precision
                ocr = reader.readtext(
                    arr,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-',
                    text_threshold=0.4,
                    low_text=0.2,
                    link_threshold=0.2,
                )
            except Exception as e:
                of.write(json.dumps({'img': r['img'], 'tag': tag,
                                     'bubble_rect_in_crop': None,
                                     'reason': f'ocr_error:{e}'}) + '\n')
                errors += 1
                continue

            target_n = norm(tag)
            rect = best_match(ocr, target_n)

            if rect is not None:
                # undo upscale so bbox is in original 320x320 crop space
                if args.upscale != 1.0:
                    rect = tuple(v / args.upscale for v in rect)
                of.write(json.dumps({'img': r['img'], 'tag': tag,
                                     'bubble_rect_in_crop': list(rect),
                                     'reason': 'hit'}) + '\n')
                hits += 1
            else:
                out = {'img': r['img'], 'tag': tag,
                       'bubble_rect_in_crop': None,
                       'reason': 'no_match' if ocr else 'no_text'}
                if args.save_debug_misses and debug_saved < args.save_debug_misses:
                    # Dump OCR tokens to help diagnose
                    out['ocr_tokens'] = [
                        [text, [[float(p[0]), float(p[1])] for p in poly], float(conf)]
                        for poly, text, conf in ocr
                    ]
                    # Copy the crop to debug_misses/
                    try:
                        Image.open(img_path).save(
                            debug_dir / f"{debug_saved:04d}_{norm(tag)}.png")
                    except Exception:
                        pass
                    debug_saved += 1
                of.write(json.dumps(out) + '\n')
                misses += 1

    total = hits + misses + errors
    print()
    print('=' * 60)
    print(f"hits:    {hits:6d}  ({100*hits/max(1,total):5.1f}%)")
    print(f"misses:  {misses:6d}  ({100*misses/max(1,total):5.1f}%)")
    print(f"errors:  {errors:6d}  ({100*errors/max(1,total):5.1f}%)")
    print(f"total:   {total:6d}  (+ {skipped} resumed)")
    print(f"elapsed: {time.time()-t0:.1f}s")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
