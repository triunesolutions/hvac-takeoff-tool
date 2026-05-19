"""Shared bubble-crop preprocessing used by both training data prep and
inference-time OCR. Train/inference parity is critical — any change here
must be re-applied on both sides before re-running the benchmark.

The fine-tuned PP-OCRv4 rec head expects 3x upscale + Otsu binarize.
This matches the preprocessing used by label_tag_bubbles_ocr.py when the
ground-truth bubble bboxes were extracted, so training crops produced via
this helper are pixel-identical to what inference will see.
"""
import numpy as np
from PIL import Image


UPSCALE = 3.0


def _otsu_threshold(arr_u8: np.ndarray) -> int:
    """Otsu threshold on a uint8 grayscale array. Returns the cutoff."""
    hist, _ = np.histogram(arr_u8, bins=256, range=(0, 256))
    total = arr_u8.size
    sum_total = float(np.dot(np.arange(256), hist))
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
    return thresh


def preprocess_bubble_crop(crop, upscale: float = UPSCALE) -> np.ndarray:
    """Take a bubble crop (PIL.Image or HxWx3 BGR/RGB ndarray) and return a
    3x upscaled, Otsu-binarized RGB ndarray (HxWx3, uint8) ready for OCR.

    The output is what the PP-OCRv4 rec head was fine-tuned on. Do not
    diverge inference preprocessing from this function without re-training.
    """
    if isinstance(crop, np.ndarray):
        pil = Image.fromarray(crop[..., ::-1] if crop.ndim == 3 else crop)
    else:
        pil = crop
    w, h = pil.size
    if upscale != 1.0:
        pil = pil.resize((max(1, int(w * upscale)), max(1, int(h * upscale))),
                         Image.LANCZOS)
    arr = np.array(pil.convert('L'))
    thresh = _otsu_threshold(arr)
    bw = (arr > thresh).astype(np.uint8) * 255
    return np.stack([bw, bw, bw], axis=-1)
