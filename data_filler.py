"""
data_filler.py — Stage 11.

Per Deck 1 slides 5, 6, 7 and Deck 2 slide 8, estimators manually fill in
missing equipment data. This module automates the tractable parts:

  Rule N1 — Neck size from CFM range table (Deck 1 slide 7)
            If a tag has CFM but no NECK SIZE, look up the GRD schedule's
            CFM-range row and assign the matching neck size.

  Rule M1 — Slot width from model name (Deck 2 slide 8)
            "AS22O" → 2", "AS35" → 3"-class, etc. Regex on the MODEL field.

  Rule D1 — Damper size = duct size (Deck 1 slide 6)
            For dampers without a size, copy the adjacent duct size if
            one was detected/labeled.
            (Requires duct vector parsing — not implemented; flagged for future)

The fill-ins are written back into each TagVariable's 'properties' dict
with a '_inferred_*' key so downstream consumers can tell what was
inferred vs originally present.
"""

from __future__ import annotations

import re


# Pattern: model names like "AS22O", "AS35", "5LD" — capture the numeric part
# that indicates slot width. Empirically the team's nomenclature uses N-inch
# slots encoded as the leading 1-2 digit number after letters.
SLOT_WIDTH_PATTERNS = [
    # "AS22O" → 2 (the slot is the FIRST digit-pair / 10 then round? simpler: 2.2" rounds to 2")
    # The deck says "AS22O" → 2 inches. Treat first digit as slot inches.
    (re.compile(r'\bAS(\d)', re.I), lambda m: float(m.group(1))),
    # "AS35" same logic (3 inches)
    # "1-SLOT", "2-SLOT" etc.
    (re.compile(r'\b(\d)\s*[-/]\s*SLOT', re.I), lambda m: float(m.group(1))),
    # "1.5\" SLOT"
    (re.compile(r'\b(\d+(?:\.\d+)?)["\'\s]+SLOT', re.I), lambda m: float(m.group(1))),
]


def _normalize_props(props: dict) -> dict[str, str]:
    """Lower-case key map for case-insensitive lookup."""
    return {k.upper().strip(): str(v).strip() for k, v in props.items()}


def _find_value(norm_props: dict[str, str], keywords: list[str]) -> str | None:
    for kw in keywords:
        for k, v in norm_props.items():
            if kw in k:
                return v
    return None


def _to_float(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r'\d+(?:\.\d+)?', s)
    return float(m.group(0)) if m else None


# ---- Rule N1: neck size from CFM range table ----

def build_cfm_range_table(grd_schedule_rows: list[dict]) -> list[tuple]:
    """If any rows in the GRD schedule represent a CFM-RANGE → NECK-SIZE table,
    return a list of (cfm_min, cfm_max, neck_size) tuples. Each row needs both
    a CFM range and a neck size to be useful.
    """
    table = []
    for v in grd_schedule_rows:
        norm = _normalize_props(v.get('properties', {}))
        cfm_field = _find_value(norm, ['CFM RANGE', 'CFM', 'RANGE'])
        neck_field = _find_value(norm, ['NECK', 'NECK SIZE', 'SIZE'])
        if not cfm_field or not neck_field:
            continue
        # Try to parse "100-200" or "100 to 200"
        m = re.match(r'\s*(\d+)\s*[-–to]+\s*(\d+)', cfm_field)
        if m:
            table.append((float(m.group(1)), float(m.group(2)), neck_field.strip()))
    # Sort by CFM range start
    table.sort(key=lambda r: r[0])
    return table


def apply_neck_size_fill(variables: list[dict], cfm_range_table: list[tuple]) -> int:
    """For variables missing NECK SIZE but having CFM, look up the table.
    Returns count of inferences made."""
    if not cfm_range_table:
        return 0
    n = 0
    for v in variables:
        norm = _normalize_props(v.get('properties', {}))
        # Has neck already?
        if _find_value(norm, ['NECK', 'SIZE']):
            continue
        cfm = _to_float(_find_value(norm, ['CFM']))
        if cfm is None:
            continue
        # Lookup
        for cmin, cmax, neck in cfm_range_table:
            if cmin <= cfm <= cmax:
                v.setdefault('properties', {})['NECK SIZE'] = neck
                v.setdefault('properties', {})['_inferred_neck'] = (
                    f'from CFM-range table ({cmin:.0f}-{cmax:.0f})'
                )
                n += 1
                break
    return n


# ---- Rule M1: slot width from model name ----

def apply_slot_width_fill(variables: list[dict]) -> int:
    """For linear-diffuser tags missing slot width, derive from MODEL.
    Returns count of inferences made."""
    n = 0
    for v in variables:
        # Only apply to linear diffuser classes
        cls = (v.get('inferred_yolo_class') or '').upper()
        tag = (v.get('tag') or '').upper()
        if not any(k in cls for k in ('LINEAR', 'SLOT')) and not tag.startswith(('LD', 'LSD', 'L-')):
            continue
        norm = _normalize_props(v.get('properties', {}))
        # Already has?
        if _find_value(norm, ['SLOT', 'SLOT WIDTH']):
            continue
        model = _find_value(norm, ['MODEL', 'MAKE', 'MAKE & MODEL', 'MANUFACTURER & MODEL'])
        if not model:
            continue
        for pat, extractor in SLOT_WIDTH_PATTERNS:
            m = pat.search(model)
            if m:
                width = extractor(m)
                v.setdefault('properties', {})['SLOT WIDTH'] = f'{width:g}"'
                v.setdefault('properties', {})['_inferred_slot_width'] = (
                    f'from model {model!r}'
                )
                n += 1
                break
    return n


# ---- Top-level entry point ----

def fill_missing_data(variables: list[dict]) -> dict:
    """Apply all inference rules to a variables list (mutates in place).
    Returns a stats dict."""
    # Find the GRD schedule rows
    grd_rows = [v for v in variables
                if (v.get('inferred_yolo_class') or '').upper() in
                ('AD-GRD', 'AD-T-BAR SUPPLY', 'AD-T-BAR RETURN',
                 'AD-SURF SUPPLY', 'AD-SURF RETURN')
                or (v.get('schedule_name') or '').upper().find('GRD') >= 0]
    cfm_table = build_cfm_range_table(grd_rows)

    n_neck = apply_neck_size_fill(variables, cfm_table)
    n_slot = apply_slot_width_fill(variables)

    return {
        'cfm_range_table_size': len(cfm_table),
        'neck_sizes_inferred': n_neck,
        'slot_widths_inferred': n_slot,
    }


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('variables', help='Path to variables.json')
    ap.add_argument('--out', help='Output JSON file (overwrite the variables, with fills)')
    args = ap.parse_args()

    vars_ = json.loads(Path(args.variables).read_text(encoding='utf-8'))
    stats = fill_missing_data(vars_)
    print(f'Inferences:')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    if args.out:
        Path(args.out).write_text(json.dumps(vars_, indent=2), encoding='utf-8')
        print(f'\nWrote {args.out}')
