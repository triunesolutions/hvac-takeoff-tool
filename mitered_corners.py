"""
mitered_corners.py — Deck 2 slide 15.

When 4 linear slot diffusers (LD-2 / LR-2) are joined at right angles to form
a rectangle around a space, the team adds 4 mitered corners as ACCESSORIES.

Detection:
  1. Find groups of 4 AD-LINEAR SLOT DIFFUSER (or LD-prefixed) detections
     that form a rectangle: 2 horizontal + 2 vertical, ends near-coincident.
  2. For each detected rectangle, emit 4 corner accessories.

Constraints from the deck:
  - Connecting diffusers must be the same MODEL (same length spec)
  - Same number of SLOTS and SLOT WIDTH for compatibility
  - Plenums on SUPPLY diffusers (with duct connection)
  - Sight baffles on RETURN diffusers (no duct connection)
"""

from __future__ import annotations

from typing import Iterable


LINEAR_CLASSES = {'AD-LINEAR PLENUM', 'AD-LINEAR SLOT DIFFUSER'}


def _orient(det: dict) -> str:
    """Return 'H' (horizontal run) or 'V' (vertical run)."""
    w = det['x2'] - det['x1']
    h = det['y2'] - det['y1']
    return 'H' if w >= h else 'V'


def _near(a: float, b: float, tol: float = 50) -> bool:
    return abs(a - b) < tol


def detect_mitered_rectangle(dets: list[dict], tol_px: float = 80) -> list[list[int]]:
    """Find groups of 4 linear diffusers forming a closed rectangle.
    Returns list of [i,j,k,l] index quadruples (one quadruple per rectangle).
    """
    linears = [(i, d) for i, d in enumerate(dets) if d.get('cls') in LINEAR_CLASSES]
    h_list = [(i, d) for i, d in linears if _orient(d) == 'H']
    v_list = [(i, d) for i, d in linears if _orient(d) == 'V']
    if len(h_list) < 2 or len(v_list) < 2:
        return []

    rectangles = []
    used = set()

    for i_top, top in h_list:
        if i_top in used:
            continue
        for i_bot, bot in h_list:
            if i_bot == i_top or i_bot in used:
                continue
            if not _near(top['x1'], bot['x1'], tol_px):
                continue
            if not _near(top['x2'], bot['x2'], tol_px):
                continue
            # top.y < bot.y (top is higher)
            if top['y1'] >= bot['y1']:
                continue
            # Now find two vertical sides connecting them
            for i_left, left in v_list:
                if i_left in used:
                    continue
                if not _near(left['x1'], top['x1'], tol_px):
                    continue
                if not _near(left['y1'], top['y1'], tol_px) or not _near(left['y2'], bot['y2'], tol_px):
                    continue
                for i_right, right in v_list:
                    if i_right == i_left or i_right in used:
                        continue
                    if not _near(right['x2'], top['x2'], tol_px):
                        continue
                    if not _near(right['y1'], top['y1'], tol_px) or not _near(right['y2'], bot['y2'], tol_px):
                        continue
                    rectangles.append([i_top, i_bot, i_left, i_right])
                    used |= {i_top, i_bot, i_left, i_right}
                    break
                if i_top in used: break
            if i_top in used: break
    return rectangles


def annotate_mitered_corners(dets: list[dict]) -> int:
    """In-place: tag each detection in a detected rectangle with 'mitered_group'
    and 'mitered_corners' = 4. Returns number of groups found.
    """
    rects = detect_mitered_rectangle(dets)
    for group_idx, members in enumerate(rects):
        for mi in members:
            dets[mi]['mitered_group'] = group_idx
            dets[mi]['mitered_corners'] = 4  # 4 corner accessories per rectangle
    return len(rects)


if __name__ == '__main__':
    # Smoke test with a synthetic rectangle
    dets = [
        {'cls': 'AD-LINEAR SLOT DIFFUSER', 'x1': 100, 'y1': 100, 'x2': 500, 'y2': 130},  # top
        {'cls': 'AD-LINEAR SLOT DIFFUSER', 'x1': 100, 'y1': 470, 'x2': 500, 'y2': 500},  # bottom
        {'cls': 'AD-LINEAR SLOT DIFFUSER', 'x1': 100, 'y1': 130, 'x2': 130, 'y2': 470},  # left
        {'cls': 'AD-LINEAR SLOT DIFFUSER', 'x1': 470, 'y1': 130, 'x2': 500, 'y2': 470},  # right
    ]
    n = annotate_mitered_corners(dets)
    print(f'Found {n} mitered rectangle(s)')
    for i, d in enumerate(dets):
        print(f'  det {i}: mitered_group={d.get("mitered_group")} corners={d.get("mitered_corners")}')
