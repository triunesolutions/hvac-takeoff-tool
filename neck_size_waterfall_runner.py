"""
neck_size_waterfall_runner.py — orchestrates the neck-size waterfall over
all detections in a job.

Builds per-page inputs once (text words, bubbles, rendered 300 DPI image,
CFM range table), then runs the waterfall per detection. Mutates the
detections dict in place so downstream stages (tag_report, Excel writer)
can read neck_size + confidence_tier + source.

Confidence tier mapping (from PLAN.md §5):
  ≥ 0.80 → HIGH    (green)
  0.50 – 0.80 → MEDIUM  (yellow)
  0.15 – 0.50 → LOW     (red, manual fill)
  < 0.15  → drop, do not emit
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from typing import Any

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from neck_size_waterfall import (
    extract_neck_size_for_detection,
    detect_cfm_range_table,
)
try:
    from confidence_calibration import calibrate_one, load_curves
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    def calibrate_one(raw, source, curves=None):
        return raw
    def load_curves():
        return {}


def confidence_to_tier(conf: float) -> str:
    """Map raw confidence to display tier."""
    if conf >= 0.80:
        return 'HIGH'
    if conf >= 0.50:
        return 'MEDIUM'
    if conf >= 0.15:
        return 'LOW'
    return 'DROP'


def _build_per_page_inputs(input_pdf: Path, page_indexes_0based: list[int],
                          want_300dpi: bool = True) -> dict[int, dict]:
    """Render each page once, extract text words once, lazy-OCR bubbles.

    Returns {page_idx_0based: {words, image_300dpi, bubbles}}.
    bubbles is left as None — populated lazily on first L2 call to save
    work when L1 already nailed it.
    """
    out: dict[int, dict] = {}
    doc = fitz.open(str(input_pdf))
    try:
        for pidx in page_indexes_0based:
            if pidx < 0 or pidx >= doc.page_count:
                continue
            page = doc[pidx]
            words = list(page.get_text('words'))

            image_300 = None
            if want_300dpi:
                try:
                    import numpy as np
                    pix = page.get_pixmap(dpi=300, annots=False)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        arr = arr[:, :, :3]
                    image_300 = arr
                except Exception as e:
                    print(f'[neck_runner] could not render p{pidx+1}: {e}')

            out[pidx] = {
                'words': words,
                'image_300dpi': image_300,
                'bubbles': None,  # lazy
                'page_idx': pidx,
            }
    finally:
        doc.close()
    return out


def _ensure_bubbles_for_page(page_inputs: dict, input_pdf: Path,
                            page_idx_0based: int) -> list[dict]:
    """Lazily detect + OCR tag bubbles for a page. Cached in page_inputs."""
    cached = page_inputs.get('bubbles')
    if cached is not None:
        return cached

    bubbles: list[dict] = []
    try:
        from tag_matcher import detect_bubbles_on_page, ocr_bubble_crops
        import numpy as np

        # Render page at 200 DPI (the model's expected scale)
        doc = fitz.open(str(input_pdf))
        try:
            page = doc[page_idx_0based]
            pix = page.get_pixmap(dpi=200, annots=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = img[:, :, :3]
        finally:
            doc.close()

        raw_bubbles = detect_bubbles_on_page(img)
        bubbles = ocr_bubble_crops(img, raw_bubbles) if raw_bubbles else []
    except Exception as e:
        print(f'[neck_runner] bubble detection failed on p{page_idx_0based+1}: {e}')
        bubbles = []

    page_inputs['bubbles'] = bubbles
    return bubbles


def enrich_detections_with_neck_size(
    detections: dict,
    variables: list[dict],
    input_pdf: Path,
    plan_pages_1based: list[int] | None = None,
) -> dict:
    """Run the waterfall over every detection. Mutates detections in place.

    Returns a summary dict of {by_source, by_tier, total, with_neck_size}.
    """
    # Build inputs
    valid_tags = {(v.get('tag') or '').upper() for v in (variables or [])
                  if v.get('tag')}
    valid_tags.discard('')

    variables_by_tag = {(v.get('tag') or '').upper(): v
                       for v in (variables or [])
                       if v.get('tag')}

    cfm_range_table = detect_cfm_range_table(variables or [])

    # Render the pages we'll touch (detection pages only)
    page_idxs_to_render = set()
    for pkey, det_list in detections.get('pages', {}).items():
        if not det_list:
            continue
        # detections.json keys are 0-indexed page indices (after the
        # 2026-06-04 indexing fix). Use as-is.
        try:
            page_idxs_to_render.add(int(pkey))
        except ValueError:
            continue

    per_page = _build_per_page_inputs(
        input_pdf, sorted(page_idxs_to_render),
        want_300dpi=True,
    )

    # Calibration curves (empty until corrections have accumulated)
    curves = load_curves() if CALIBRATION_AVAILABLE else {}

    # Iterate detections
    stats = {
        'total': 0,
        'with_neck_size': 0,
        'by_source': Counter(),
        'by_tier': Counter(),
        'cfm_range_buckets': len(cfm_range_table),
        'calibration_curves_active': len(curves),
    }

    for pkey, det_list in detections.get('pages', {}).items():
        try:
            pidx = int(pkey)
        except ValueError:
            continue
        page_inputs = per_page.get(pidx)
        if not page_inputs:
            continue

        # Bubbles: lazy-loaded if L1 misses
        bubbles = None

        for det in det_list:
            stats['total'] += 1

            # Trigger L1 first
            result = extract_neck_size_for_detection(
                det,
                page_words=page_inputs['words'],
                valid_tags=valid_tags,
                bubbles_on_page=None,  # don't run L2 yet
                variables_by_tag=variables_by_tag,
                page_image_300dpi=None,  # don't OCR yet
                cfm_range_table=None,
                tag=det.get('tag'),
            )

            # If L1 didn't get a result, escalate by adding bubbles
            if not result or not result.get('neck_size'):
                if bubbles is None:
                    bubbles = _ensure_bubbles_for_page(
                        page_inputs, input_pdf, pidx
                    )
                result = extract_neck_size_for_detection(
                    det,
                    page_words=page_inputs['words'],
                    valid_tags=valid_tags,
                    bubbles_on_page=bubbles,
                    variables_by_tag=variables_by_tag,
                    page_image_300dpi=None,
                    cfm_range_table=None,
                    tag=det.get('tag'),
                )

            # If L2 didn't get a result, try L3 (OCR crop)
            if not result or not result.get('neck_size'):
                result = extract_neck_size_for_detection(
                    det,
                    page_words=page_inputs['words'],
                    valid_tags=valid_tags,
                    bubbles_on_page=bubbles,
                    variables_by_tag=variables_by_tag,
                    page_image_300dpi=page_inputs['image_300dpi'],
                    cfm_range_table=None,
                    tag=det.get('tag'),
                )

            # If still nothing, try L4 (CFM range — only useful for tagged dets)
            if (not result or not result.get('neck_size')) and cfm_range_table:
                result = extract_neck_size_for_detection(
                    det,
                    page_words=page_inputs['words'],
                    valid_tags=valid_tags,
                    bubbles_on_page=bubbles,
                    variables_by_tag=variables_by_tag,
                    page_image_300dpi=page_inputs['image_300dpi'],
                    cfm_range_table=cfm_range_table,
                    tag=det.get('tag'),
                )

            # Attach to detection
            if result:
                raw_conf = result.get('confidence', 0.0)
                source = result.get('source', 'unknown')
                # Apply calibration curve (no-op until data accumulates)
                cal_conf = calibrate_one(raw_conf, source, curves=curves)

                det['neck_size'] = result.get('neck_size')
                det['neck_confidence'] = cal_conf
                det['neck_confidence_raw'] = raw_conf
                det['neck_source'] = source
                det['neck_tier'] = confidence_to_tier(cal_conf)
                if result.get('flag'):
                    det['neck_flag'] = result['flag']

                # Promote the resolved tag onto the detection if waterfall
                # discovered one and the detection wasn't tagged yet
                if not det.get('tag') and result.get('tag'):
                    det['tag'] = result['tag']
                    det['tag_method'] = 'neck-waterfall'

                if result.get('neck_size'):
                    stats['with_neck_size'] += 1
                    stats['by_source'][result.get('source', 'unknown')] += 1
                    stats['by_tier'][det['neck_tier']] += 1

    stats['by_source'] = dict(stats['by_source'])
    stats['by_tier'] = dict(stats['by_tier'])
    return stats
