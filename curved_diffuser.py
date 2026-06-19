"""
curved_diffuser.py — Deck 2 slide 13.

For curved linear slot diffusers the team measures CHORD length and RISE,
then looks up ARC LENGTH (for slot diffuser) and RADIUS (for plenum) in
the "GRD Curve Calculator" Excel.

We replicate that calculator in pure math.

Curved slot diffusers connected by a curve are SEPARATE components when
they curve in different directions — unlike straight linear diffusers,
which can be merged into one run.
"""

from __future__ import annotations

import math


def arc_length_from_chord_rise(chord: float, rise: float) -> dict:
    """Given the chord (straight-line distance between curve endpoints) and
    the rise (perpendicular height of the curve at midpoint), compute the
    arc length and radius of the underlying circle.

    Geometry:
        chord = 2 R sin(θ)
        rise  = R - R cos(θ) = R (1 - cos(θ))
        →  R = (chord² / 4 + rise²) / (2 rise)
        →  θ = arcsin(chord / (2 R))
        →  arc_length = 2 R θ

    Returns: {radius, arc_length, theta_rad, theta_deg}
    """
    if rise <= 0:
        # Straight line — degenerate case
        return {
            'radius': float('inf'),
            'arc_length': chord,
            'theta_rad': 0.0,
            'theta_deg': 0.0,
        }
    R = (chord ** 2 / 4 + rise ** 2) / (2 * rise)
    # Half-angle subtended by chord at the center
    theta = math.asin(min(1.0, chord / (2 * R)))
    arc = 2 * R * theta
    return {
        'radius': R,
        'arc_length': arc,
        'theta_rad': theta,
        'theta_deg': math.degrees(theta),
    }


def is_separate_curve_direction(det_a: dict, det_b: dict,
                               tol_deg: float = 30) -> bool:
    """Two curved diffusers should NOT be merged if their curve directions differ.
    Direction is approximated by the angle from bbox center to the midpoint
    of the longest side (heuristic; CAD would have actual curve vectors).
    """
    def midline_angle(det):
        # Approximate orientation by aspect ratio
        w = det['x2'] - det['x1']
        h = det['y2'] - det['y1']
        if w == 0 and h == 0:
            return 0
        return math.degrees(math.atan2(h, w))
    a = midline_angle(det_a)
    b = midline_angle(det_b)
    return abs(a - b) > tol_deg


def annotate_curved_segments(dets_on_page: list[dict]) -> int:
    """For each AD-LINEAR SLOT DIFFUSER detection flagged as curved
    (det.get('curved', False)), compute the arc length using the bbox
    dimensions as a rough chord+rise estimate.
    """
    n = 0
    for det in dets_on_page:
        cls = det.get('cls', '')
        if 'LINEAR' not in cls.upper():
            continue
        if not det.get('curved'):
            continue
        w = det['x2'] - det['x1']
        h = det['y2'] - det['y1']
        # Use the longer side as chord, shorter as rise
        chord = max(w, h)
        rise = min(w, h)
        result = arc_length_from_chord_rise(chord, rise)
        det['arc_length_px'] = result['arc_length']
        det['radius_px'] = result['radius']
        det['curve_angle_deg'] = result['theta_deg']
        n += 1
    return n


if __name__ == '__main__':
    # Quick sanity check — match values estimators expect for typical cases
    tests = [
        (chord, rise, 'expected radius, arc')
        for chord, rise in [(48, 4), (96, 8), (24, 2), (192, 16)]
    ]
    for chord, rise, _ in tests:
        r = arc_length_from_chord_rise(chord, rise)
        print(f'chord={chord:>5.1f}  rise={rise:>5.1f}  →  R={r["radius"]:>6.1f}  arc={r["arc_length"]:>6.2f}  θ={r["theta_deg"]:.2f}°')
