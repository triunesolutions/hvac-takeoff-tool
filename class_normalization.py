"""
class_normalization.py

Canonical class mapping for Bluebeam markup subjects → YOLO training classes.

Two layers:
  1. EXACT_SYNONYMS: case-insensitive 1:1 mapping for plurals and spelling variants.
  2. PATTERN_RULES: regex collapses — strip size/slot suffixes that vary per project
     but represent the same visual class.

The original (un-normalized) subject is preserved as `subclass` in the rich
annotations.jsonl so downstream tag-inference + Excel writing can still
recover MODULE SIZE, NECK SIZE, etc.
"""

import re


# Exact 1:1 normalizations. Keys are matched UPPERCASE-stripped.
# Value is the canonical class name we want the model to learn.
EXACT_SYNONYMS = {
    # Plurals
    'FANS': 'FAN',
    'LOUVERS': 'LOUVER',
    'DAMPERS': 'DAMPER',

    # Fire / smoke damper variants — all the same physical equipment
    'SMOKE/FIRE DAMPER': 'FIRE SMOKE DAMPER',
    'COMBINATION FIRE/SMOKE DAMPER': 'FIRE SMOKE DAMPER',
    'FIRE/SMOKE DAMPER': 'FIRE SMOKE DAMPER',

    # Manual-damper variants
    'VOLUME DAMPER': 'MANUAL VOLUME DAMPER',

    # Roof terminations
    'ROOF CAP': 'RAIN CAP',
    'ROOF HOOD - BAROMETRIC RELIEF': 'RELIEF HOOD',

    # Ductless split system normalization
    'DUCTLESS SPLIT SYSTEM - INDOOR UNIT': 'SPLIT SYSTEM INDOOR',
    'DUCTLESS SPLIT SYSTEM - OUTDOOR UNIT': 'SPLIT SYSTEM OUTDOOR',
    'SPLIT SYSTEM HEAT PUMP INDOOR UNIT': 'SPLIT SYSTEM INDOOR',
    'SPLIT SYSTEM HEAT PUMP OUTDOOR UNIT': 'SPLIT SYSTEM OUTDOOR',

    # v14-retrain folds (2026-06-08): subjects found in the 27-project Bluebeam
    # audit that fell outside the 25-class taxonomy. Folded into existing classes
    # so the fine-tune from v10 stays at 25 classes. (~2.8% of all boxes.)
    # Fans — user decision: exhaust/ceiling/inline all train as the generic FAN
    # class; tag inference still distinguishes EF tags downstream via the schedule.
    'EXHAUST FAN': 'FAN',
    'CEILING FAN': 'FAN',
    'INLINE FAN': 'FAN',
    'FLY FAN': 'FAN',
    'POWER VENTILATOR': 'FAN',
    'FSD': 'FIRE SMOKE DAMPER',                 # abbreviation
    'FAN COIL': 'FAN COIL UNIT',
    'WALL CAP': 'VENT CAP',
    'CONTROL DAMPERS': 'MOTORIZED DAMPER',
    'CONTROL DAMPER': 'MOTORIZED DAMPER',
    'RELIFE AIR HOOD': 'RELIEF HOOD',            # misspelling of RELIEF
    'RELIEF AIR HOOD': 'RELIEF HOOD',
    'AIR CONDITIONING UNIT': 'ROOFTOP UNIT',     # packaged AC
    'DOAS ERV UNIT': 'ROOFTOP UNIT',
    # genuinely-other small equipment → OTHER MECHANICAL (keeps singletons out of
    # the taxonomy as their own classes)
    'RANGE HOOD': 'OTHER MECHANICAL',
    'DRYER BOX': 'OTHER MECHANICAL',
    'AIR CURTAIN': 'OTHER MECHANICAL',
    'CLOUD': 'OTHER MECHANICAL',                 # stray revision-cloud markup
    'AIRFLOW REGULATOR': 'MANUAL VOLUME DAMPER',  # regulates airflow ≈ volume damper
    'AREA MEASUREMENT': 'OTHER MECHANICAL',       # backstop — usually dropped as a non-equipment markup
}


# Pattern-based collapses. Regex is matched after uppercase-strip; if it
# matches, the listed canonical name replaces the raw subject.
# Order matters — first match wins.
PATTERN_RULES = [
    # AD-LINEAR variants — size info (1" SLOT, 1.5" SLOT, 1-SLOT, 1-2.5" SLOT)
    # all collapse to the underlying visual class. Size lives in the content
    # field (preserved in jsonl).
    (re.compile(r'^AD-LINEAR PLENUM\b.*'), 'AD-LINEAR PLENUM'),
    (re.compile(r'^AD-LINEAR SLOT DIFFUSER\b.*'), 'AD-LINEAR SLOT DIFFUSER'),
    (re.compile(r'^AD-LINEAR\b(?!.*PLENUM)(?!.*SLOT DIFFUSER).*'), 'AD-LINEAR'),

    # Catch-all damper variants we didn't list explicitly
    (re.compile(r'^.*FIRE.*SMOKE.*DAMPER.*$|^.*SMOKE.*FIRE.*DAMPER.*$'), 'FIRE SMOKE DAMPER'),

    # v14-retrain folds: any electric heater variant (duct/unit/wall/-FSC) → the
    # heater bucket. Checked before generic rules; GAS UNIT HEATER is unaffected
    # (starts with GAS, not ELECTRIC).
    (re.compile(r'^ELECTRIC.*HEATER.*'), 'GAS UNIT HEATER'),
    # '-FSC' is an annotation suffix (for-shop-coordination), not a class —
    # strip it onto the fan bucket where it appears (FANS-FSC).
    (re.compile(r'^FANS?\b.*-FSC.*'), 'FAN'),

    # ── Multifamily / detailed-subject folds (v14, 2026-06-09) ──────────────
    # The 7-XX archive (Vale, Encore, Liberty, …) uses richer equipment names
    # than the 25-class taxonomy. Collapse them onto an existing class so the
    # fine-tune from v10 stays at 25 classes. ORDER MATTERS (first match wins):
    # specific equipment keywords are tested before the broad fan/split fallback.
    (re.compile(r'.*FAN COIL.*'), 'FAN COIL UNIT'),                 # incl. VRF / mini-split / DX fan coil
    (re.compile(r'.*CONDENS.*'), 'AIR COOLED CONDENSING UNIT'),     # split/VRF/mini-split condensing units
    (re.compile(r'.*ROOFTOP.*'), 'ROOFTOP UNIT'),                   # rooftop heat-pump / packaged units
    (re.compile(r'.*RADIATION DAMPER.*'), 'FIRE DAMPER'),           # ceiling radiation damper ≈ fire damper
    (re.compile(r'.*MOTORIZED\s+DAMPER.*'), 'MOTORIZED DAMPER'),    # collapses double-space variant
    (re.compile(r'.*\bVAV\b.*|.*TERMINAL UNIT.*'), 'OTHER MECHANICAL'),  # no VAV class in the 25
    (re.compile(r'.*HUMIDIFIER.*'), 'OTHER MECHANICAL'),
    (re.compile(r'.*(ENERGY RECOVERY|MAKEUP AIR|MAKE-UP AIR|PENTHOUSE).*'), 'OTHER MECHANICAL'),
    (re.compile(r'.*(GRAVITY VENTILATOR|ROOF VENT|HOOD CAP).*'), 'VENT CAP'),
    (re.compile(r'.*WALL LOUVER.*'), 'LOUVER'),
    (re.compile(r'.*(UNIT\s+HEATER|RADIANT.*HEATER).*'), 'GAS UNIT HEATER'),
    # Split-system / heat-pump family. OUTDOOR keyword → outdoor unit; everything
    # else (mini-split, ductless, cassette, dwelling/corridor/common-area indoor
    # units, generic "split system heat pump unit") → the indoor unit class.
    (re.compile(r'.*OUTDOOR.*'), 'SPLIT SYSTEM OUTDOOR'),
    (re.compile(r'.*(MINI.?SPLIT|DUCTLESS|CASSETTE|WALL MOUNTED SPLIT|VERTICAL TERMINAL HEAT PUMP|SPLIT SYSTEM|HEAT PUMP|VARIABLE REFRIGERANT|A/C UNIT).*'), 'SPLIT SYSTEM INDOOR'),
    # Remaining fans (circulation / dwelling-unit / per-building exhaust fans).
    # Runs last so it can't steal FAN COIL or split-system subjects above.
    (re.compile(r'.*\bFAN\b.*'), 'FAN'),
]


# Classes with too few samples — optionally folded into a parent or dropped.
# Set value to None to drop (label written with class_id = -1, ignored at train time).
RARE_CLASS_FOLD = {
    'DEHUMIDIFIER': 'OTHER MECHANICAL',
    'CO SENSOR': 'OTHER MECHANICAL',
    'ELECTRIC HEATER': 'GAS UNIT HEATER',  # both small heaters; visually similar
    'DX FAN COIL UNIT': 'FAN COIL UNIT',
}


def normalize_class(raw: str, fold_rare: bool = True) -> str:
    """Return the canonical class name for a raw Bluebeam subject."""
    if not raw:
        return ''
    cls = raw.upper().strip()

    if cls in EXACT_SYNONYMS:
        cls = EXACT_SYNONYMS[cls]

    for pat, target in PATTERN_RULES:
        if pat.match(cls):
            cls = target
            break

    if fold_rare and cls in RARE_CLASS_FOLD:
        folded = RARE_CLASS_FOLD[cls]
        if folded is not None:
            cls = folded

    return cls


if __name__ == '__main__':
    # Tiny self-test
    cases = [
        ('FANS', 'FAN'),
        ('Fan', 'FAN'),
        ('LOUVERS', 'LOUVER'),
        ('AD-LINEAR PLENUM', 'AD-LINEAR PLENUM'),
        ('AD-LINEAR PLENUM 1" SLOT', 'AD-LINEAR PLENUM'),
        ('AD-LINEAR PLENUM 1-2.5" SLOT', 'AD-LINEAR PLENUM'),
        ('AD-LINEAR SLOT DIFFUSER 1.5" SLOT', 'AD-LINEAR SLOT DIFFUSER'),
        ('COMBINATION FIRE/SMOKE DAMPER', 'FIRE SMOKE DAMPER'),
        ('SMOKE/FIRE DAMPER', 'FIRE SMOKE DAMPER'),
        ('ROOF CAP', 'RAIN CAP'),
        ('AD-GRD', 'AD-GRD'),  # unchanged
    ]
    failures = 0
    for raw, expected in cases:
        got = normalize_class(raw)
        ok = got == expected
        if not ok:
            failures += 1
        flag = 'OK ' if ok else 'FAIL'
        print(f'  [{flag}] {raw!r:50s} -> {got!r:25s}  expected {expected!r}')
    print(f'\n{failures} failure(s)')
