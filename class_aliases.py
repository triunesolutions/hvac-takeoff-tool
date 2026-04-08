"""
Class consolidation mapping — merges duplicate/equivalent class names.
Used by train_yolo.py and benchmark.py to normalize annotations.

Why: We had 75 classes from raw annotations, but many were:
1. Whitespace/plural variants (MOTORIZED  DAMPER vs MOTORIZED DAMPER, FANS vs FAN)
2. Synonymous types (different naming conventions for same equipment)
3. Too rare to learn (only 1-3 examples)

Result: ~50 effective classes that the model can actually disambiguate.
"""

CLASS_ALIASES = {
    # Whitespace/plural normalization
    'MOTORIZED  DAMPER': 'MOTORIZED DAMPER',
    'FANS': 'FAN',
    'SUPPLY FANS': 'SUPPLY FAN',
    'UNIT HEATERS': 'UNIT HEATER',
    'LOUVERS': 'LOUVER',
    'FAN COIL UNITS': 'FAN COIL UNIT',
    'SPLIT SYSTEM CEILING CONCEALED HEAT PUMP UNITS': 'SPLIT SYSTEM HEAT PUMP',
    'SPLIT SYSTEM COOLING ONLY UNITS': 'SPLIT SYSTEM',
    'SPLIT SYSTEM DUCTLESS AIR CONDITIONING UNIT': 'SPLIT SYSTEM',
    'SPLIT SYSTEM INDOOR UNIT': 'SPLIT SYSTEM',
    'SPLIT SYSTEM OUTDOOR UNIT': 'SPLIT SYSTEM',
    'SPLIT SYSTEM CONDENSING UNIT': 'SPLIT SYSTEM',
    'SPLIT SYSTEM FURNACE': 'FURNACE',

    # Damper synonyms
    'DAMPER WTH TAP': 'DAMPER WITH TAP',  # typo fix
    'COMBINATION FIRE/SMOKE DAMPER': 'FIRE SMOKE DAMPER',
    'SMOKE FIRE DAMPER': 'FIRE SMOKE DAMPER',
    'SMOKE DAMPER': 'FIRE SMOKE DAMPER',  # close enough for our purposes

    # Fan synonyms
    'GREASE EXHAUST FAN': 'EXHAUST FAN',
    'CIRCULATION FAN': 'FAN',
    'DOAS/RTU FAN': 'FAN',
    'MUA FAN': 'FAN',
    'DRYER BOOSTER FAN': 'FAN',
    'JET VENT FAN': 'JET VENT FAN',  # keep — visually distinct
    'DESTRATIFICATION FAN': 'DESTRATIFICATION FAN',  # keep — visually distinct

    # Heater consolidation
    'GAS UNIT HEATER': 'UNIT HEATER',
    'PLENUM RATED ELECTRIC UNIT HEATER': 'ELECTRIC HEATER',
    'ELECTRIC DUCT HEATER': 'DUCT HEATER',
    'GAS FIRED RADIANT HEATER': 'UNIT HEATER',
    'ELECTRIC WALL HEATER': 'ELECTRIC HEATER',

    # Hood/vent consolidation
    'WALL CAP': 'VENT CAP',
    'ROOF HOOD': 'HOOD',
    'KITCHEN HOOD': 'HOOD',
    'EXHAUST HOOD': 'HOOD',

    # AHU/RTU consolidation
    'AIR HANDLING UNIT-DX': 'AIR HANDLING UNIT',
    'AIR HANDLER UNIT': 'AIR HANDLING UNIT',
    'ROOFTOP AIR HANDLING UNIT': 'PACKAGED ROOFTOP UNIT',
    'ROOFTOP UNIT WITH GAS HEAT': 'PACKAGED ROOFTOP UNIT',
    'ROOFTOP UNIT': 'PACKAGED ROOFTOP UNIT',
    'MAKE UP AIR UNIT': 'PACKAGED ROOFTOP UNIT',

    # Condensing unit consolidation
    'AIR COOLED CONDENSING UNIT': 'CONDENSING UNIT',
    'CONDENSER': 'CONDENSING UNIT',
    'CONDENSER UNIT': 'CONDENSING UNIT',

    # VRF consolidation
    'VRF UNIT': 'VRF',
    'VRF INDOOR UNIT': 'VRF',
    'VRF OUTDOOR UNIT': 'VRF',
    'VRF HEAT RECOVERY BRANCH CIRCUIT CONTROLLER': 'VRF',
    'VARIABLE REFRIGERANT FLOW': 'VRF',
    'BRANCH CIRCUIT CONTROLLER': 'VRF',

    # Energy recovery
    'ENERGY RECOVERY VENTILATOR': 'ENERGY RECOVERY',

    # Heat pump
    'HEAT PUMP UNIT': 'HEAT PUMP',
    'WATER SOURCE HEAT PUMP': 'HEAT PUMP',
    'HORIZONTAL FAN COIL': 'FAN COIL UNIT',

    # VAV
    'VAV UNIT': 'VAV',
    'VARIABLE AIR VOLUME': 'VAV',

    # Larchmont outliers — map to closest match
    'AD-MISC/LINEAR': 'AD-LINEAR PLENUM',  # rough mapping
    'AD-SURF EXHAUST': 'AD-SURF RETURN',   # rough mapping
}


def normalize_class(name):
    """Apply aliases. Strip extra whitespace."""
    name = name.strip()
    # Collapse multiple spaces
    name = ' '.join(name.split())
    return CLASS_ALIASES.get(name, name)


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print(f"Total aliases: {len(CLASS_ALIASES)}")
    targets = set(CLASS_ALIASES.values())
    print(f"Target classes (after merging): {len(targets)}")
    for src, tgt in sorted(CLASS_ALIASES.items()):
        print(f"  {src:50s} -> {tgt}")
