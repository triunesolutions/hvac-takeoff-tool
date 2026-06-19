"""
toolbox_mapping.py

Maps AI detection classes (YOLO v10/v11) to NSW Toolbox subjects + colors.

Source files:
  - AI classes:     yolo_dataset_v11/data.yaml (25 classes)
                  + v10 extras observed in saas/data/jobs/ (EXHAUST FAN, CONDENSING UNIT)
  - Toolbox tools:  C:/Users/TriuneTakeoff/Downloads/NSW-ToolBox.btx
                    (95 ToolChestItems, ~36 unique subjects)

How the team's toolbox represents equipment:
  - 'AD-GRD' is ONE subject; color encodes supply (yellow) vs return (green).
  - 'FAN' is ONE subject; color variants distinguish supply/exhaust/relief.
  - Damper family all use red.
  - Caps/hoods/CU outdoor equipment use blue.

The mapping below picks the closest toolbox subject + color for each AI class.
Where the AI is finer-grained than the toolbox (e.g. AD-SURF SUPPLY), we collapse
into the toolbox's coarser bucket — the estimator's correction will preserve
the toolbox subject anyway.

Confidence:
  HIGH    direct match, no human review needed
  MEDIUM  reasonable substitute; estimator may want to re-stamp some
  LOW     no good match; emitted as a placeholder rather than dropped

Color format matches the toolbox XML: 'R G B' floats in [0,1].
"""

# Toolbox colors (so each tool reads one consistent name)
YELLOW          = '1 1 0'                        # AD-GRD supply variant
GREEN_SUPPLY    = '0 1 0.2509804'                # AD-GRD return variant
BLUE_FAN_1      = '0 0.5019608 1'                # FAN supply variant
BLUE_FAN_2      = '0 0.5019608 1'                # FAN exhaust variant (toolbox uses same color; subject differs by position in list)
RED_DAMPER      = '1 0 0'                        # Damper family
BLUE_CAP        = '0 0.5019608 1'                # Caps / hoods / outdoor units
ORANGE_EQUIP    = '1 0.5019608 0'                # Louvers, sensors, heaters, ERVs, condensers
PURPLE_INDOOR   = '0.5019608 0 1'                # Fan coil / VAV / duct silencer / indoor units
GRAY_FALLBACK   = '0.5 0.5 0.5'                  # any AI class without a curated toolbox subject


# (ai_class) -> (toolbox_subject, color, confidence, note)
AI_TO_TOOLBOX = {
    # ---- Air-distribution diffusers/grilles ----
    'AD-GRD':                       ('AD-GRD',                YELLOW,        'HIGH',   'Direct'),
    'AD-T-BAR SUPPLY':              ('AD-GRD',                YELLOW,        'HIGH',   'Team uses AD-GRD yellow for T-bar supply'),
    'AD-T-BAR RETURN':              ('AD-GRD',                GREEN_SUPPLY,  'HIGH',   'Team uses AD-GRD green for T-bar return'),
    'AD-SURF SUPPLY':               ('AD-GRD',                YELLOW,        'MEDIUM', 'Surface mounting not in toolbox; collapse to AD-GRD'),
    'AD-SURF RETURN':               ('AD-GRD',                GREEN_SUPPLY,  'MEDIUM', 'Surface mounting not in toolbox; collapse to AD-GRD'),
    'AD-LINEAR PLENUM':             ('AD-GRD',                YELLOW,        'LOW',    'No linear-plenum tool; emit as AD-GRD, estimator can re-stamp'),
    'AD-LINEAR SLOT DIFFUSER':      ('AD-GRD',                YELLOW,        'LOW',    'No linear-slot tool; emit as AD-GRD, estimator can re-stamp'),

    # ---- Fans ----
    'FAN':                          ('FAN',                   BLUE_FAN_1,    'HIGH',   'Direct'),
    'EXHAUST FAN':                  ('FAN',                   BLUE_FAN_2,    'HIGH',   'Team uses one FAN tool with color variants'),

    # ---- Dampers ----
    'FIRE SMOKE DAMPER':            ('FIRE SMOKE DAMPER',     RED_DAMPER,    'HIGH',   'Direct'),
    'FIRE DAMPER':                  ('FIRE DAMPER',           RED_DAMPER,    'HIGH',   'Direct'),
    'BACKDRAFT DAMPER':             ('BACKDRAFT DAMPER',      RED_DAMPER,    'HIGH',   'Direct'),
    'MOTORIZED DAMPER':             ('MOTORIZED  DAMPER',     RED_DAMPER,    'HIGH',   'Direct (toolbox subject has double space — preserved)'),
    'MANUAL VOLUME DAMPER':         ('MANUAL VOLUME DAMPER',  RED_DAMPER,    'HIGH',   'Direct'),
    'DAMPER':                       ('MANUAL VOLUME DAMPER',  RED_DAMPER,    'MEDIUM', 'Generic damper -> most common toolbox damper'),

    # ---- Caps / hoods ----
    'RAIN CAP':                     ('ROOF CAP',              BLUE_CAP,      'HIGH',   'Toolbox name is ROOF CAP; class_normalization already merges these'),
    'VENT CAP':                     ('VENT CAP',              BLUE_CAP,      'HIGH',   'Direct'),
    'RELIEF HOOD':                  ('ROOF HOOD',             BLUE_CAP,      'MEDIUM', 'Close — roof-mounted relief device'),

    # ---- Louvers ----
    'LOUVER':                       ('LOUVER',                ORANGE_EQUIP,  'HIGH',   'Direct'),

    # ---- Heaters ----
    'GAS UNIT HEATER':              ('ELECTRIC HEATERS',      ORANGE_EQUIP,  'LOW',    'Toolbox does not split gas vs electric; estimator may re-class'),

    # ---- Split systems / condensers / rooftops ----
    'SPLIT SYSTEM INDOOR':          ('FAN COIL UNIT',         PURPLE_INDOOR, 'MEDIUM', 'Indoor unit of split system is functionally a fan coil'),
    'SPLIT SYSTEM OUTDOOR':         ('CONDENSER UNIT',        ORANGE_EQUIP,  'HIGH',   'Outdoor unit = condensing unit'),
    'INDOOR UNIT':                  ('FAN COIL UNIT',         PURPLE_INDOOR, 'HIGH',   'IU prefix = mini-split indoor unit ≈ fan coil'),
    'OUTDOOR UNIT':                 ('CONDENSER UNIT',        ORANGE_EQUIP,  'HIGH',   'OU prefix = mini-split outdoor condenser'),
    'AIR COOLED CONDENSING UNIT':   ('CONDENSER UNIT',        ORANGE_EQUIP,  'HIGH',   'Direct'),
    'CONDENSING UNIT':              ('CONDENSER UNIT',        ORANGE_EQUIP,  'HIGH',   'Spelling difference only'),
    'FAN COIL UNIT':                ('FAN COIL UNIT',         PURPLE_INDOOR, 'HIGH',   'Direct'),
    'ROOFTOP UNIT':                 ('CONDENSER UNIT',        ORANGE_EQUIP,  'MEDIUM', 'RTU bundles condenser + AHU; closest toolbox match'),
    'PACKAGED ROOFTOP UNIT':        ('CONDENSER UNIT',        ORANGE_EQUIP,  'HIGH',   'RTU prefix = packaged rooftop unit, stamp as condenser'),
    'HEAT PUMP':                    ('CONDENSER UNIT',        ORANGE_EQUIP,  'MEDIUM', 'HP prefix; closest toolbox match is condenser'),
    'AIR HANDLING UNIT':            ('FAN COIL UNIT',         PURPLE_INDOOR, 'MEDIUM', 'AHU; closest toolbox match is fan coil'),
    'HUMIDIFIER':                   ('HUMIDIFIER',            PURPLE_INDOOR, 'HIGH',   'Direct — was in TOOLBOX_NOT_IN_AI, now AI predicts via text-layer'),

    # ---- v10 model-native classes the old (v11-era) table missed ----
    'VAV':                          ('SINGLE DUCT VAV BOX',   PURPLE_INDOOR, 'MEDIUM', 'VAV terminal box'),
    'VRF':                          ('VRF FAN COIL UNIT',     ORANGE_EQUIP,  'MEDIUM', 'VRF unit (toolbox color: orange); estimator re-classes indoor/outdoor'),
    'SPLIT SYSTEM':                 ('FAN COIL UNIT',         PURPLE_INDOOR, 'MEDIUM', 'Split system indoor ~ fan coil'),
    'SPLIT SYSTEM HEAT PUMP':       ('CONDENSER UNIT',        ORANGE_EQUIP,  'MEDIUM', 'Split HP outdoor ~ condenser'),
    'PTAC':                         ('FAN COIL UNIT',         PURPLE_INDOOR, 'MEDIUM', 'PTAC ~ fan coil'),
    'HEATER':                       ('ELECTRIC HEATERS',      ORANGE_EQUIP,  'MEDIUM', 'Generic heater'),
    'HOOD':                         ('ROOF HOOD',             BLUE_CAP,      'MEDIUM', 'Hood'),
    'DESTRATIFICATION FAN':         ('FAN',                   BLUE_FAN_1,    'MEDIUM', 'Destratification fan ~ FAN'),
    'JET VENT FAN':                 ('FAN',                   BLUE_FAN_2,    'MEDIUM', 'Jet/vent fan ~ FAN'),
    'INLINE DAMPER':               ('MANUAL VOLUME DAMPER',  RED_DAMPER,    'MEDIUM', 'Inline damper ~ volume damper'),
    'DAMPER WITH TAP':              ('MANUAL VOLUME DAMPER',  RED_DAMPER,    'MEDIUM', 'Damper with tap ~ volume damper'),
    'AD-MISC/LINEAR':               ('AD-GRD',                YELLOW,        'LOW',    'Linear/misc diffuser; emit as AD-GRD'),
    'HIGH EFFICENCY TAKE-OFF':      ('HIGH EFFICENCY TAKE-OFF', RED_DAMPER,  'MEDIUM', 'Duct take-off fitting (toolbox color: red)'),

    # ---- Catch-all ----
    'OTHER MECHANICAL':             (None,                    None,          'LOW',    'No good toolbox match — skip emitting'),
}


# Toolbox subjects the AI does NOT currently predict.
# These are equipment the estimator stamps that we would learn from
# corrections (each one stamped becomes training data for a future model).
TOOLBOX_NOT_IN_AI = [
    'CONSTANT AIRFLOW REGULATOR',
    'ZONE REGISTER TERMINAL',
    'REMOTE DAMPER w/COD',
    'REMOTE/CABLE DAMPER',
    'HIGH EFFICENCY TAKE-OFF',
    'MANUAL BALANCING DAMPER',
    'SMOKE DAMPER',                       # separate from FIRE SMOKE DAMPER
    'CEILING RADIATION DAMPER',
    'DUCT RADIATION DAMPER',
    'MOTORIZED/CONTROL DAMPER',
    'MOTORIZED ZONE DAMPER',
    'DRYER BOX',
    'CO/NO SENSOR',
    'WALL CAP',                           # separate from VENT CAP
    'AIR CURTAIN',
    'ENERGY RECOVERY VENTILATOR',
    'EVAPORATIVE COOLING UNIT',
    'HUMIDIFIER',
    'DUCT SILENCERS',
    'SINGLE DUCT VAV BOX',
    'SINGLE DUCT VAR. AIR VOL. UNIT',
    'VRF INDOOR HEAT PUMP UNIT',
    'VRF FAN COIL UNIT',
    'VRF OUTDOOR HEAT PUMP UNIT',
    'VRF AIR COOLED CONDENSER',
    'VRF INDOOR BRANCH CONTROLLER',
    'VRF DISTRIBUTION BOX',
    'VARIABLE REFRIGERANT VOLUME',
    'FLOOR UNIT',
]


def map_ai_class(ai_class: str):
    """Return (toolbox_subject, color, confidence, note) for an AI class.

    If the AI class is unmapped, returns (None, None, 'UNMAPPED', '...').
    If the toolbox_subject is None, the caller should skip emitting the stamp.
    """
    if ai_class in AI_TO_TOOLBOX:
        return AI_TO_TOOLBOX[ai_class]
    # Fallback: never silently drop a real detection. Stamp it with its own
    # class name + a neutral color so it shows in Bluebeam's Markups List (the
    # estimator can re-class or delete it). Matches the annotated PDF, which
    # draws every detection regardless of mapping.
    if ai_class:
        return (ai_class, GRAY_FALLBACK, 'FALLBACK',
                f'AI class {ai_class!r} not curated — stamped with class name')
    return (None, None, 'UNMAPPED', 'empty AI class')


if __name__ == '__main__':
    # Sanity check: every v11 class is covered
    V11_CLASSES = [
        'AD-T-BAR RETURN', 'AD-SURF SUPPLY', 'AD-SURF RETURN', 'FAN',
        'DAMPER', 'FIRE SMOKE DAMPER', 'AD-LINEAR PLENUM',
        'AD-LINEAR SLOT DIFFUSER', 'AD-T-BAR SUPPLY', 'RAIN CAP',
        'FIRE DAMPER', 'BACKDRAFT DAMPER', 'RELIEF HOOD', 'LOUVER',
        'GAS UNIT HEATER', 'VENT CAP', 'AD-GRD', 'SPLIT SYSTEM INDOOR',
        'AIR COOLED CONDENSING UNIT', 'SPLIT SYSTEM OUTDOOR',
        'MOTORIZED DAMPER', 'OTHER MECHANICAL', 'MANUAL VOLUME DAMPER',
        'FAN COIL UNIT', 'ROOFTOP UNIT',
    ]
    V10_EXTRAS = ['EXHAUST FAN', 'CONDENSING UNIT']

    missing = [c for c in V11_CLASSES + V10_EXTRAS if c not in AI_TO_TOOLBOX]
    if missing:
        print(f'!! MISSING coverage for: {missing}')
    else:
        print(f'OK: all {len(V11_CLASSES)} v11 classes + {len(V10_EXTRAS)} v10 extras mapped.')

    by_conf = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    skips = 0
    for ai_cls, (subj, color, conf, note) in AI_TO_TOOLBOX.items():
        by_conf[conf] = by_conf.get(conf, 0) + 1
        if subj is None:
            skips += 1
    print()
    print(f'Confidence breakdown:')
    for c in ('HIGH', 'MEDIUM', 'LOW'):
        print(f'  {c:6s} {by_conf[c]:3d}')
    print(f'  SKIP  {skips:3d}  (no toolbox match — detection dropped)')
    print()
    print(f'Toolbox subjects AI does not predict (correction-loop opportunities):')
    print(f'  {len(TOOLBOX_NOT_IN_AI)} subjects — each one stamped by an estimator')
    print(f'  becomes training data for a future model version.')
