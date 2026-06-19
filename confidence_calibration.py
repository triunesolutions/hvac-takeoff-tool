"""
confidence_calibration.py — turn raw waterfall confidences into honest numbers.

A confidence score should say "if I claim 0.85, I'm correct 85% of the time."
Raw scores from the waterfall formulas don't satisfy this — they're loose
heuristics. This module:

  1. Logs (predicted_confidence, was_correct) per row when ground truth
     is available (estimator submitted a correction).
  2. Fits a per-source calibration curve (isotonic regression).
  3. Applies the curve when surfacing confidence to the estimator.
  4. Tracks Brier score over time to verify the calibration is improving.

Per-source calibration is important: Level 1 (text-layer) needs a
different curve than Level 4 (CFM lookup). Each source's raw scoring
has different systematic biases.

Inputs are stored in JSONL under ~/.gstack/ or saas/data/calibration/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Any

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / 'data' / 'calibration'
EVENTS_FILE = CALIBRATION_DIR / 'events.jsonl'
CURVES_FILE = CALIBRATION_DIR / 'curves.json'


def _ensure_dir():
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


def log_calibration_event(predicted_confidence: float,
                         was_correct: bool,
                         source: str,
                         tag: str | None = None,
                         job_id: str | None = None,
                         metadata: dict | None = None) -> None:
    """Append a calibration event for later curve fitting.

    Call this from the correction-submission endpoint: for each row in the
    estimator's corrected takeoff, we know what the AI predicted and what
    was actually right. Use that to log calibration data.
    """
    _ensure_dir()
    event = {
        'predicted_confidence': round(float(predicted_confidence), 4),
        'was_correct': bool(was_correct),
        'source': source,
        'tag': tag,
        'job_id': job_id,
        'metadata': metadata or {},
    }
    with EVENTS_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')


def load_all_events() -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    events = []
    for line in EVENTS_FILE.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def brier_score(events: list[dict]) -> float:
    """Brier score: mean squared error between predicted prob and actual outcome.
    Lower is better. 0 = perfect calibration. 0.25 = always predicts 0.5.
    """
    if not events:
        return float('nan')
    total = sum(
        (e['predicted_confidence'] - (1.0 if e['was_correct'] else 0.0)) ** 2
        for e in events
    )
    return round(total / len(events), 4)


def calibration_reliability(events: list[dict], n_buckets: int = 10) -> list[dict]:
    """Bucket by predicted_confidence, report actual accuracy per bucket.
    Returns a list of {bucket_min, bucket_max, n_samples, mean_predicted,
    mean_actual, gap}.
    """
    buckets = []
    for i in range(n_buckets):
        bmin = i / n_buckets
        bmax = (i + 1) / n_buckets
        in_bucket = [e for e in events
                    if bmin <= e['predicted_confidence'] < bmax
                    or (i == n_buckets - 1 and e['predicted_confidence'] == bmax)]
        if not in_bucket:
            continue
        mean_pred = sum(e['predicted_confidence'] for e in in_bucket) / len(in_bucket)
        mean_actual = sum(1.0 if e['was_correct'] else 0.0 for e in in_bucket) / len(in_bucket)
        buckets.append({
            'bucket_min': round(bmin, 2),
            'bucket_max': round(bmax, 2),
            'n_samples': len(in_bucket),
            'mean_predicted': round(mean_pred, 3),
            'mean_actual': round(mean_actual, 3),
            'gap': round(mean_pred - mean_actual, 3),
        })
    return buckets


def fit_isotonic_calibration(events: list[dict]) -> list[tuple[float, float]]:
    """Fit a simple per-bucket calibration mapping.

    Real Platt scaling uses sklearn.isotonic.IsotonicRegression. We use
    a lighter-weight per-bucket lookup here that doesn't require sklearn
    as a dependency: bucket by predicted, compute mean actual accuracy
    per bucket, interpolate between bucket centers at query time.

    Returns a sorted list of (predicted_bucket_center, actual_accuracy)
    pairs. apply_calibration() looks up via linear interpolation.
    """
    if not events:
        return []
    buckets = calibration_reliability(events, n_buckets=10)
    curve = [(b['mean_predicted'], b['mean_actual'])
            for b in buckets if b['n_samples'] >= 5]
    curve.sort(key=lambda p: p[0])
    return curve


def apply_calibration(raw_confidence: float, curve: list[tuple[float, float]]) -> float:
    """Map a raw confidence to a calibrated confidence using the fitted curve.

    If the curve is empty or has <2 points, returns raw_confidence unchanged.
    Otherwise interpolates between the nearest bucket centers.
    """
    if not curve or len(curve) < 2:
        return raw_confidence

    raw = max(0.0, min(1.0, raw_confidence))

    # Clip to curve range
    if raw <= curve[0][0]:
        return round(curve[0][1], 3)
    if raw >= curve[-1][0]:
        return round(curve[-1][1], 3)

    # Linear interpolation
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        if x1 <= raw <= x2:
            if x2 == x1:
                return round((y1 + y2) / 2, 3)
            t = (raw - x1) / (x2 - x1)
            return round(y1 + t * (y2 - y1), 3)
    return raw


def fit_all_curves() -> dict[str, list[tuple[float, float]]]:
    """Fit one calibration curve per source. Writes curves.json."""
    events = load_all_events()
    if not events:
        return {}

    by_source: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_source[e['source']].append(e)

    curves = {}
    for source, evs in by_source.items():
        if len(evs) >= 30:
            curves[source] = fit_isotonic_calibration(evs)

    _ensure_dir()
    CURVES_FILE.write_text(
        json.dumps({k: v for k, v in curves.items()}, indent=2),
        encoding='utf-8',
    )
    return curves


def load_curves() -> dict[str, list[tuple[float, float]]]:
    if not CURVES_FILE.exists():
        return {}
    try:
        raw = json.loads(CURVES_FILE.read_text(encoding='utf-8'))
        # JSON loads tuples as lists; convert back
        return {k: [tuple(p) for p in v] for k, v in raw.items()}
    except json.JSONDecodeError:
        return {}


def calibrate_one(raw_confidence: float, source: str,
                 curves: dict | None = None) -> float:
    """Apply the calibration curve for a specific source.
    Falls back to identity if no curve available.
    """
    if curves is None:
        curves = load_curves()
    curve = curves.get(source) or []
    if not curve:
        # Try a generic fallback: source prefix (level1-*, level2-*)
        for src_key, src_curve in curves.items():
            if source.startswith(src_key.split('-')[0]):
                return apply_calibration(raw_confidence, src_curve)
        return raw_confidence
    return apply_calibration(raw_confidence, curve)


def calibration_report() -> dict:
    """Summary of the current calibration state."""
    events = load_all_events()
    curves = load_curves()
    by_source = defaultdict(list)
    for e in events:
        by_source[e['source']].append(e)
    return {
        'total_events': len(events),
        'sources': {
            src: {
                'n_samples': len(evs),
                'brier': brier_score(evs),
                'has_curve': src in curves,
            }
            for src, evs in by_source.items()
        },
        'overall_brier': brier_score(events),
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('fit', help='Fit calibration curves from logged events')
    sub.add_parser('report', help='Show calibration state')
    sub.add_parser('list', help='Show last 20 events')

    apply = sub.add_parser('apply', help='Apply calibration to a raw confidence')
    apply.add_argument('--raw', type=float, required=True)
    apply.add_argument('--source', required=True)

    args = ap.parse_args()
    if args.cmd == 'fit':
        curves = fit_all_curves()
        print(f'Fit {len(curves)} per-source curve(s)')
        for src, curve in curves.items():
            print(f'  {src}: {len(curve)} points')
    elif args.cmd == 'report':
        print(json.dumps(calibration_report(), indent=2))
    elif args.cmd == 'list':
        events = load_all_events()
        for e in events[-20:]:
            print(f"  {e['predicted_confidence']:.2f} "
                  f"{'✓' if e['was_correct'] else '✗'}  "
                  f"src={e['source']}  tag={e.get('tag')}")
    elif args.cmd == 'apply':
        result = calibrate_one(args.raw, args.source)
        print(f'raw={args.raw:.3f} → calibrated={result:.3f}')
