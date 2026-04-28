"""
Class consolidation mapping — merges duplicate/equivalent class names.
Used by train_yolo.py and benchmark.py to normalize annotations.

v8 changes (from confusion matrix analysis 2026-04-09):
- ELECTRIC HEATER + UNIT HEATER + ELECTRIC WALL HEATER → HEATER
  (model was 100% confusing them — visually identical floor-mount units)
- FAN COIL UNIT separated from SPLIT SYSTEM HEAT PUMP
  (was 47% misclassified as split system)
- Removed CIRCULATION FAN/MUA FAN/DRYER BOOSTER FAN → FAN merge
  (caused FAN to be over-predicted for AD-GRD diffusers)
- Removed AD-MISC/LINEAR → AD-LINEAR PLENUM (kept distinct)
- Removed AD-SURF EXHAUST → AD-SURF RETURN (kept distinct)
"""

CLASS_ALIASES = {
    # Whitespace/plural normalization (always safe)
    'MOTORIZED  DAMPER': 'MOTORIZED DAMPER',
    'FANS': 'FAN',
    'SUPPLY FANS': 'SUPPLY FAN',
    'UNIT HEATERS': 'HEATER',
    'LOUVERS': 'LOUVER',
    'FAN COIL UNITS': 'FAN COIL UNIT',

    # Split system consolidation
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
    'SMOKE DAMPER': 'FIRE SMOKE DAMPER',

    # Fan synonyms — only merge truly synonymous
    'GREASE EXHAUST FAN': 'EXHAUST FAN',
    # NOTE: removed CIRCULATION FAN/DOAS/MUA/DRYER BOOSTER → FAN
    # because it caused FAN to be over-predicted as a "catch-all"

    # Heater consolidation — merged based on confusion matrix
    # ELECTRIC HEATER and UNIT HEATER were 93% confused → merge to HEATER
    'GAS UNIT HEATER': 'HEATER',
    'PLENUM RATED ELECTRIC UNIT HEATER': 'HEATER',
    'ELECTRIC UNIT HEATER': 'HEATER',
    'ELECTRIC HEATER': 'HEATER',
    'UNIT HEATER': 'HEATER',
    'ELECTRIC WALL HEATER': 'HEATER',
    'GAS FIRED RADIANT HEATER': 'HEATER',
    'ELECTRIC DUCT HEATER': 'DUCT HEATER',  # keep duct heater separate (visually different)
    'DUCT HEATER': 'DUCT HEATER',

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

    # VAV — all variants
    'VAV UNIT': 'VAV',
    'VARIABLE AIR VOLUME': 'VAV',
    'VARIABLE AIR VOLUME BOX': 'VAV',
    'VARIABLE AIR VOLUME TERMINAL UNIT': 'VAV',
    'VARIABLE AIR VOLUME TERMINAL UNIT FLOOR - FIRST FLOOR': 'VAV',
    'VARIABLE AIR VOLUME TERMINAL UNIT FLOOR - SECOND FLOOR': 'VAV',
    'TERMINAL BOXES, VARIABLE AIR VOLUME, HOT WATER HEAT': 'VAV',
    'SINGLE DUCT AIR TERMINAL UNIT': 'VAV',
    'FAN POWERED TERMINAL UNIT': 'VAV',

    # Split system — all variants
    'AIR-COOLED SPLIT SYSTEM / CONDESING UNIT': 'SPLIT SYSTEM',
    'SPLIT SYSTEM A/C UNIT': 'SPLIT SYSTEM',
    'DUCTLESS SPLIT SYSTEM': 'SPLIT SYSTEM',
    'VERTICAL DX SPLIT-SYSTEM': 'SPLIT SYSTEM',
    'SPLIT SYSTEM HEAT PUMP - DWELLING UNITS': 'SPLIT SYSTEM HEAT PUMP',

    # VRF variants
    'VRV INDOOR UNIT ERV': 'VRF',
    'MULTI VRF INDOOR UNIT': 'VRF',

    # Fan variants
    'EXHAUST FANS': 'EXHAUST FAN',

    # Heater variants
    'ELECTRIC HEATERS': 'HEATER',

    # Terminal/misc
    'PACKAGED TERMINAL AIR CONDITIONING': 'PTAC',
    'VERTICAL TERMINAL AIR CONDITIONING UNITS': 'PTAC',
    'ZONE REGISTER TERMINAL': 'AD-GRD',
    'AIR TERMINAL DEVICE': 'AD-GRD',

    # Damper variants
    'MANUAL BALANCING DAMPER': 'MANUAL VOLUME DAMPER',
    'CABLE OPERATED BALANCING DAMPER': 'MANUAL VOLUME DAMPER',
    'CEILING RADIATION DAMPER': 'FIRE DAMPER',
    'VOLUME DAMPER': 'MANUAL VOLUME DAMPER',

    # Misc
    'ROOF CAP': 'VENT CAP',
    'DRYER BOX': 'VENT CAP',
    'UNIT': 'CONDENSING UNIT',

    # v10 — additions from the 36 sample-project corpus (April 2026)
    # Linear plenum slot-count variants → single class
    'AD-LINEAR PLENUM 1 SLOT': 'AD-LINEAR PLENUM',
    'AD-LINEAR PLENUM 2 SLOT': 'AD-LINEAR PLENUM',
    'AD-LINEAR PLENUM 2" SLOT': 'AD-LINEAR PLENUM',
    'AD-LINEAR PLENUM 1-2" SLOT': 'AD-LINEAR PLENUM',
    'AD-LINEAR PLENUM 2-1" SLOT': 'AD-LINEAR PLENUM',
    # Linear slot diffuser variants
    'AD-LINEAR SLOT DIFFUSER 1 SLOT': 'AD-LINEAR SLOT DIFFUSER',
    'AD-LINEAR SLOT DIFFUSER 2 SLOT': 'AD-LINEAR SLOT DIFFUSER',
    'AD-LINEAR SLOT DIFFUSER 2" SLOT': 'AD-LINEAR SLOT DIFFUSER',
    'AD-LINEAR SLOT DIFFUSER 1-2" SLOT': 'AD-LINEAR SLOT DIFFUSER',
    'AD-LINEAR SLOT DIFFUSER 2-1" SLOT': 'AD-LINEAR SLOT DIFFUSER',
    # Fire/smoke damper combo shorthand
    'FD/FSD': 'FIRE SMOKE DAMPER',
    # Exhaust fan variants — small TI projects label by location/use
    'EXHAUST FAN-COMMON AREA': 'EXHAUST FAN',
    'EXHAUST FAN-JANITOR/RESTROOM': 'EXHAUST FAN',
    'OUTSIDE AIR FAN-COMMON AREA': 'EXHAUST FAN',
    'TRANSFER FAN': 'EXHAUST FAN',
    'KITCHEN EXHAUST FAN': 'EXHAUST FAN',
    'HVLS FAN': 'EXHAUST FAN',
    'HVLS CEILING FANS': 'EXHAUST FAN',
    'FOG FAN': 'EXHAUST FAN',
    # Damper variants → motorized damper umbrella
    'OPPOSED BLADE DAMPER': 'MOTORIZED DAMPER',
    'BACKDRAFT DAMPER': 'MOTORIZED DAMPER',
    'ELECTRONIC REMOTE DAMPER': 'MOTORIZED DAMPER',
    'DAMPERS': 'MOTORIZED DAMPER',
    # VAV variants
    'VARIABLE VOLUME BOX': 'VAV',
    # Heater typo / variants
    'ELECTRIC CABINATE UNIT HEATER': 'HEATER',
    'ELECTRIC CABINET UNIT HEATER': 'HEATER',
    # Roof gravity vent looks like a hood
    'GRAVITY VENTILATOR': 'HOOD',
    'GOOSENECK': 'VENT CAP',
    # Air curtain — keep family together
    'AIR CURTAIN-AMBIENT': 'AIR CURTAIN',
}


def normalize_class(name):
    """Apply aliases. Strip extra whitespace."""
    name = name.strip()
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
