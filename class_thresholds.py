"""
class_thresholds.py

Per-class YOLO confidence thresholds.

Why these exist:
  YOLO defaults to a single global confidence threshold (we use 0.4).
  In practice each equipment class behaves differently:

  - Classes with high false-positive rates (AD-GRD on ductwork joints)
    need a HIGHER threshold to suppress noise.
  - Rare classes the model has only seen a few times (FIRE DAMPER, ROOFTOP
    UNIT) need a LOWER threshold or the model never fires above 0.4.
  - Classes with very distinct visual signatures (VAV, RAIN CAP) can go
    LOWER without picking up noise — gain recall for free.

  These values are educated initial estimates based on:
   - Music Academy + CUSD + United Mechanical detection behavior
   - Class sample counts from yolo_dataset_v11 (12 projects, 1,392 boxes)
   - CLAUDE.md known limitations (AD-GRD over-detection, rare-class miss)

  v2: replace with values learned from a threshold sweep on a labelled
  validation set (precision-recall curve per class, pick F1-optimal point).
"""

from class_normalization import normalize_class

# Global default — used for any class not listed below
DEFAULT_CONF = 0.40

# Per-canonical-class thresholds. Lower = more permissive = more recall.
# Higher = stricter = more precision.
CLASS_CONF: dict[str, float] = {
    # ----- High-false-positive classes — RAISE threshold -----
    'AD-GRD':               0.55,   # over-detected on ductwork joints / ceiling grids
    'AD-SURF SUPPLY':       0.50,   # similar geometry to AD-GRD
    'AD-SURF RETURN':       0.50,
    'AD-T-BAR SUPPLY':      0.45,
    'AD-T-BAR RETURN':      0.45,
    'MANUAL VOLUME DAMPER': 0.50,   # often confused with duct fittings
    'DAMPER':               0.50,
    'MOTORIZED DAMPER':     0.50,

    # ----- Visually distinct classes — LOWER threshold (free recall) -----
    'VAV':                  0.30,
    'FAN':                  0.30,   # rooftop fans, distinctive
    'AIR HANDLING UNIT':    0.30,
    'CONDENSING UNIT':      0.30,
    'ROOFTOP UNIT':         0.30,
    'FAN COIL UNIT':        0.30,
    'AIR COOLED CONDENSING UNIT': 0.30,
    'SPLIT SYSTEM INDOOR':  0.30,
    'SPLIT SYSTEM OUTDOOR': 0.30,
    'GAS UNIT HEATER':      0.30,
    'RAIN CAP':             0.30,
    'RELIEF HOOD':          0.30,
    'VENT CAP':             0.30,
    'LOUVER':               0.30,

    # ----- Rare classes (few training samples) — LOW threshold -----
    'FIRE DAMPER':          0.25,
    'FIRE SMOKE DAMPER':    0.25,
    'BACKDRAFT DAMPER':     0.30,

    # ----- Linear devices — moderate -----
    'AD-LINEAR PLENUM':         0.40,
    'AD-LINEAR SLOT DIFFUSER':  0.40,
    'AD-LINEAR':                0.40,

    # ----- Catch-all -----
    'OTHER MECHANICAL':     0.40,
}


def threshold_for(class_name: str, default: float = DEFAULT_CONF) -> float:
    """Return the confidence threshold to use for a given YOLO class name.

    Looks up the model's RAW class name first — the YOLO head emits names like
    'VAV', 'CONDENSING UNIT', 'SPLIT SYSTEM' that ``normalize_class`` (built for
    a different reduced taxonomy) would remap to the wrong bucket
    (e.g. VAV → 'OTHER MECHANICAL' → 0.40 instead of its tuned 0.30). Only when
    the raw name isn't tuned do we fall back to the normalized name, so we still
    get 'FANS' → 'FAN', 'AD-LINEAR PLENUM 1" SLOT' → 'AD-LINEAR PLENUM', etc.
    """
    raw = (class_name or '').upper().strip()
    if raw in CLASS_CONF:
        return CLASS_CONF[raw]
    canonical = normalize_class(class_name or '')
    return CLASS_CONF.get(canonical, default)


def filter_by_class_threshold(detections: list[dict],
                              default: float = DEFAULT_CONF) -> tuple[list[dict], dict]:
    """Filter a list of detection dicts by per-class threshold.

    Each detection must have 'cls' (or 'class') and 'conf' fields.
    Returns (filtered_detections, drop_counts_per_class).
    """
    out = []
    drops: dict[str, int] = {}
    for d in detections:
        cls = d.get('cls') or d.get('class') or ''
        thresh = threshold_for(cls, default=default)
        if d.get('conf', 1.0) >= thresh:
            out.append(d)
        else:
            drops[cls] = drops.get(cls, 0) + 1
    return out, drops


if __name__ == '__main__':
    # Smoke-test the threshold lookup
    cases = [
        ('AD-GRD', 0.55),
        ('FAN', 0.30),
        ('FANS', 0.30),                       # normalized
        ('FIRE DAMPER', 0.25),
        ('AD-LINEAR PLENUM 1" SLOT', 0.40),   # normalized
        ('SOMETHING NEW', 0.40),              # falls back to default
        ('vav', 0.30),                        # case-insensitive
    ]
    failures = 0
    for raw, expected in cases:
        got = threshold_for(raw)
        ok = abs(got - expected) < 1e-6
        if not ok:
            failures += 1
        flag = 'OK ' if ok else 'FAIL'
        print(f'  [{flag}] {raw!r:40s} -> {got}  (expected {expected})')
    print(f'\n{failures} failure(s)')
