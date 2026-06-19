"""
v10_class_map.py — map Bluebeam annotation subjects onto v10's EXACT 33-class
taxonomy, for fine-tuning models/hvac_yolov8s_v10.pt.

This is SEPARATE from class_normalization.py (which reduces classes for the
inference/reconciliation layer). For training we must match the model head
exactly, so the targets here are v10's own class names (model.names).

map_subject(subject) -> one of V10_CLASSES, or None to DROP (non-equipment or
no sensible home in the 33).
"""
import re

# v10's head, in order (from YOLO('models/hvac_yolov8s_v10.pt').names).
V10_CLASSES = [
    'AD-T-BAR SUPPLY', 'AD-T-BAR RETURN', 'AD-SURF SUPPLY', 'AD-SURF RETURN',
    'AD-LINEAR SLOT DIFFUSER', 'AD-LINEAR PLENUM', 'AD-GRD',
    'MANUAL VOLUME DAMPER', 'FIRE DAMPER', 'DAMPER WITH TAP', 'VENT CAP',
    'FAN', 'MOTORIZED DAMPER', 'LOUVER', 'EXHAUST FAN', 'FIRE SMOKE DAMPER',
    'VAV', 'VRF', 'CONDENSING UNIT', 'SPLIT SYSTEM', 'AIR HANDLING UNIT',
    'PTAC', 'HEATER', 'INLINE DAMPER', 'DESTRATIFICATION FAN',
    'SPLIT SYSTEM HEAT PUMP', 'HOOD', 'AD-MISC/LINEAR', 'PACKAGED ROOFTOP UNIT',
    'HEAT PUMP', 'JET VENT FAN', 'FAN COIL UNIT', 'HIGH EFFICENCY TAKE-OFF',
]
_V10 = set(V10_CLASSES)

# Exact 1:1 (UPPER-stripped). Identity for most; variants/plurals/typos mapped.
EXACT = {
    'FANS': 'FAN', 'FANS-FSC': 'FAN', 'CEILING FAN': 'FAN', 'INLINE FAN': 'FAN',
    'FLY FAN': 'FAN', 'CIRCULATION FAN': 'FAN', 'POWER VENTILATOR': 'EXHAUST FAN',
    'LOUVERS': 'LOUVER', 'WALL LOUVER': 'LOUVER',
    'FSD': 'FIRE SMOKE DAMPER',
    'DAMPER': 'MANUAL VOLUME DAMPER', 'DAMPERS': 'MANUAL VOLUME DAMPER',
    'VOLUME DAMPER': 'MANUAL VOLUME DAMPER', 'CONTROL DAMPER': 'MOTORIZED DAMPER',
    'CONTROL DAMPERS': 'MOTORIZED DAMPER', 'AIRFLOW REGULATOR': 'MANUAL VOLUME DAMPER',
    'BACKDRAFT DAMPER': 'MANUAL VOLUME DAMPER',          # no backdraft class in v10
    'CEILING RADIATION DAMPER': 'FIRE DAMPER',
    'WALL CAP': 'VENT CAP', 'ROOF VENT': 'VENT CAP', 'GRAVITY VENTILATOR': 'VENT CAP',
    'RAIN CAP': 'VENT CAP', 'ROOF CAP': 'VENT CAP', 'HOOD CAP': 'HOOD',
    'RELIEF HOOD': 'HOOD', 'RELIFE AIR HOOD': 'HOOD', 'RELIEF AIR HOOD': 'HOOD',
    'ROOF HOOD - BAROMETRIC RELIEF': 'HOOD', 'RANGE HOOD': 'HOOD',
    'GAS UNIT HEATER': 'HEATER', 'ELECTRIC HEATER': 'HEATER', 'ELECTRIC HEATERS': 'HEATER',
    'UNIT HEATER': 'HEATER',
    'AIR COOLED CONDENSING UNIT': 'CONDENSING UNIT',
    'SPLIT SYSTEM CONDENSING UNITS': 'CONDENSING UNIT',
    'SPLIT SYSTEM A/C UNITS': 'SPLIT SYSTEM', 'SPLIT SYSTEM INDOOR': 'SPLIT SYSTEM',
    'SPLIT SYSTEM INDOOR UNITS': 'SPLIT SYSTEM', 'SPLIT SYSTEM OUTDOOR': 'SPLIT SYSTEM',
    'AIR CONDITIONING UNIT': 'SPLIT SYSTEM',
    'FAN COIL': 'FAN COIL UNIT', 'DX FAN COIL UNIT': 'FAN COIL UNIT',
    'TERMINAL UNIT': 'VAV', 'SINGLE DUCT VAV BOX': 'VAV',
    'VERTICAL TERMINAL HEAT PUMP A/C UNIT': 'PTAC',
    'ENERGY RECOVERY VENTILATOR': 'AIR HANDLING UNIT', 'DOAS ERV UNIT': 'AIR HANDLING UNIT',
    'ELECTRIC MAKEUP AIR UNIT': 'AIR HANDLING UNIT', 'MAKEUP AIR UNIT': 'AIR HANDLING UNIT',
}

# Pattern rules (UPPER-stripped, first match wins). Tested AFTER EXACT.
PATTERNS = [
    (re.compile(r'.*FAN COIL.*'), 'FAN COIL UNIT'),
    (re.compile(r'.*CONDENS.*'), 'CONDENSING UNIT'),
    (re.compile(r'.*ROOFTOP.*'), 'PACKAGED ROOFTOP UNIT'),
    (re.compile(r'.*RADIATION DAMPER.*'), 'FIRE DAMPER'),
    (re.compile(r'.*MOTORIZED\s+DAMPER.*'), 'MOTORIZED DAMPER'),
    (re.compile(r'.*INLINE DAMPER.*'), 'INLINE DAMPER'),
    (re.compile(r'.*DAMPER WITH TAP.*'), 'DAMPER WITH TAP'),
    (re.compile(r'.*(VAV|TERMINAL UNIT).*'), 'VAV'),
    (re.compile(r'.*ELECTRIC.*HEATER.*|.*RADIANT.*HEATER.*|.*\bUNIT\s+HEATER.*'), 'HEATER'),
    (re.compile(r'.*\bAD-LINEAR PLENUM.*'), 'AD-LINEAR PLENUM'),
    (re.compile(r'.*\bAD-LINEAR SLOT DIFFUSER.*'), 'AD-LINEAR SLOT DIFFUSER'),
    (re.compile(r'.*\bAD-LINEAR.*'), 'AD-MISC/LINEAR'),
    (re.compile(r'.*(FIRE.*SMOKE|SMOKE.*FIRE).*DAMPER.*'), 'FIRE SMOKE DAMPER'),
    (re.compile(r'.*ENERGY RECOVERY.*|.*MAKE.?UP AIR.*'), 'AIR HANDLING UNIT'),
    (re.compile(r'.*VERTICAL TERMINAL.*|.*\bPTAC\b.*'), 'PTAC'),
    # split-system / heat-pump family
    (re.compile(r'.*SPLIT SYSTEM HEAT PUMP.*|.*MINI.?SPLIT.*HEAT PUMP.*'), 'SPLIT SYSTEM HEAT PUMP'),
    (re.compile(r'.*(MINI.?SPLIT|DUCTLESS|CASSETTE|WALL MOUNTED SPLIT|SPLIT SYSTEM|\bA/C UNIT|VARIABLE REFRIGERANT).*'), 'SPLIT SYSTEM'),
    (re.compile(r'.*HEAT PUMP.*'), 'HEAT PUMP'),
    (re.compile(r'.*(GRAVITY VENTILATOR|ROOF VENT|VENT CAP|WALL CAP).*'), 'VENT CAP'),
    (re.compile(r'.*HOOD.*'), 'HOOD'),
    (re.compile(r'.*LOUVER.*'), 'LOUVER'),
    (re.compile(r'.*EXHAUST FAN.*'), 'EXHAUST FAN'),
    (re.compile(r'.*DESTRATIFICATION.*'), 'DESTRATIFICATION FAN'),
    (re.compile(r'.*JET.*FAN.*'), 'JET VENT FAN'),
    (re.compile(r'.*\bFAN\b.*'), 'FAN'),
    (re.compile(r'.*AIR HANDL.*|.*\bAHU\b.*'), 'AIR HANDLING UNIT'),
    (re.compile(r'.*HIGH EFF.*TAKE.?OFF.*'), 'HIGH EFFICENCY TAKE-OFF'),
]

# Non-equipment / no-home subjects to DROP entirely.
DROP = {
    'AREA MEASUREMENT', 'LENGTH MEASUREMENT', 'PERIMETER MEASUREMENT',
    'CLOUD', 'TEXT BOX', 'NOTE', 'CALLOUT', 'HUMIDIFIER', 'ELECTRIC HUMIDIFIER',
    'HUMIDIFIER -ELECTRIC', 'PENTHOUSE', 'DRYER BOX', 'AIR CURTAIN',
}


def map_subject(subject):
    """Return a v10 class name, or None to drop. None means the box is excluded."""
    if not subject:
        return None
    s = re.sub(r'\s+', ' ', subject.upper().strip())
    if s in DROP:
        return None
    if s in _V10:
        return s                      # identity — already a v10 class name
    if s in EXACT:
        return EXACT[s]
    for pat, tgt in PATTERNS:
        if pat.match(s):
            return tgt
    if 'HUMIDIFIER' in s:
        return None
    return None                       # unmapped -> drop (audited to be ~0)
