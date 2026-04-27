# Plan: YOLOv8s v10 Retrain — Add the 36 Sample Projects

*Drafted April 27, 2026 while v9 benchmark is running.*

## Context

The team gave us 36 small commercial TI projects under `SAMPLE FILES 27.04.26/`, each with a `Completed Takeoff/Takeoff_*.pdf` containing **Polygon annotations** with the YOLO class as the `subject` field. **3,219 total polygons** of human-verified equipment labels — exactly the format `train_yolo.py` already consumes. This is essentially free training data for v10.

The existing v9 model was trained on 124 larger projects (mostly schools, hospitals, larger commercial). v9 benchmark on the small-commercial sample set (Sola/Imperial/Alliance/Krispy Kreme so far) shows recall in the 0–33% range — confirming v9 has a domain gap on small-TI-style drawings.

## Goal

Train **v10 = v9's 124 projects + these 36 new ones = 160 total**. Expected outcome: meaningfully better recall on Sola-shape commercial TIs, modest gain on the existing test bench (close to v9 performance, no regression).

## Class distribution in the new data (top 15)

```
AD-T-BAR SUPPLY                  745
AD-SURF SUPPLY                   582
AD-T-BAR RETURN                  522
AD-SURF RETURN                   402
AD-LINEAR PLENUM                 130
LOUVER                           107
EXHAUST FAN                       75
FANS                              72
AD-LINEAR PLENUM 1 SLOT           68
LOUVERS                           59
AD-LINEAR SLOT DIFFUSER           57
FIRE SMOKE DAMPER                 50
AD-LINEAR SLOT DIFFUSER 1 SLOT    39
FD/FSD                            20
WALL CAP                          25
```

Heavy on air devices (which is exactly where v9 over-detects in benchmark) — adding these as positives + their backgrounds as negatives should sharpen the AD-* classes.

## Class aliasing additions needed

`class_aliases.py` will need entries for the variant labels in this corpus:

- `LOUVERS` → `LOUVER`
- `FANS` → `EXHAUST FAN` (or split if ambiguous — check sample annotations)
- `AD-LINEAR PLENUM 1 SLOT` / `AD-LINEAR PLENUM 2" SLOT` / `AD-LINEAR PLENUM 2-1" SLOT` → `AD-LINEAR PLENUM`
- `AD-LINEAR SLOT DIFFUSER 1 SLOT` / `2" SLOT` → `AD-LINEAR SLOT DIFFUSER`
- `FD/FSD` → `FIRE SMOKE DAMPER`
- `EXHAUST FAN-COMMON AREA` / `EXHAUST FAN-JANITOR/RESTROOM` / `OUTSIDE AIR FAN-COMMON AREA` / `TRANSFER FAN` → `EXHAUST FAN` (or keep separate if model can learn the distinction with enough samples — review with team)
- `ELECTRIC HEATERS` → `HEATER`
- `WALL CAP` → keep as new class (it's a distinct visual)
- `UNIT` → ambiguous; either drop or alias to AHU/RTU after spot-check

## Approach

`train_yolo.py` already expects a `projects/{name}/raw/` + `projects/{name}/labeled/` structure. Two viable paths:

**Path A (preferred, no train_yolo.py change):** Build a one-shot prep script that creates a sister `data to train/projects_sample/` folder mirroring the expected layout via symlinks (or copies on Windows where symlinks need privileges):

```
data to train/projects_sample/
  4.10.26 Woodbridge HS Modernization - Bldg J & K/
    raw/
      4.13.26 Woodbridge HS Modernization - Bldg J & K (RCP).pdf  ← from Plans_Specs/
    labeled/
      Takeoff_Woodbridge HS Modernization - Bldg J & K.pdf         ← from Completed Takeoff/
```

Then point `train_yolo.py --projects-dir data\ to\ train/projects_sample` (we'd need to add a flag — easier: temporarily symlink the 36 sample dirs into the existing PROJECTS_DIR with a `sample_` prefix).

**Path B (cleaner long-term):** Modify `train_yolo.py` to accept multiple `--projects-dir` flags so the existing 124-project corpus + new 36-project corpus combine cleanly.

Recommend Path A for v10 (lower risk, smaller change). Path B is correct but can wait.

## Steps

1. **Spot-check 3 PDFs by class.** Open Woodbridge HS, OSI Irvine, UCLA Health Oncology in PyMuPDF and confirm the polygons' `subject` fields are clean. Look for any class names that should be aliased.

2. **Update `class_aliases.py`** with the variants listed above. Make sure `normalize_class()` handles them.

3. **Drop projects with too few annotations** from the training set: Lancaster (6 polys), MBUSD MS (1), MBUSD HS (2). These are probably incomplete annotations from the team — including them adds label noise. Keep them as eval-only.

4. **Build `prep_sample_training.py`:** for each project in SAMPLE_ROOT (skip the 3 above), copy/symlink files into `data to train/projects/<sample_NN_name>/{raw,labeled}/`. Use `sample_` prefix to keep them grouped. ~33 new project entries.

5. **Run `train_yolo.py --all`** on Kaggle T4 (same notebook as `kaggle_train_tag_detector.ipynb`, swap dataset). Expected duration: 2-3 hrs at 60 epochs. Output `hvac_yolov8s_v10.pt`.

6. **Evaluate v10** by running `benchmark_samples.py` with `--model models/hvac_yolov8s_v10.pt` (need to add `--model` passthrough — small CLI change). Compare per-project recall/precision deltas vs v9.

7. **Hold-out validation set:** before training, set 4 sample projects aside (Sola, Imperial, Alliance, Krispy Kreme — the ones we've manually analyzed) as eval-only. Don't include them in training. Their post-train numbers are the regression bar.

## Files to create / modify

- **New:** `prep_sample_training.py` — one-shot copy-into-projects-dir script.
- **Modify:** `class_aliases.py` — add the new class mappings.
- **Modify:** `benchmark_samples.py` — add `--model` arg passthrough to subprocess CLI.
- **No change** to `train_yolo.py` (Path A).

## Risks

- **Annotation noise.** Some projects have 1-6 polygons, suggesting incomplete labeling. Already addressed by dropping those 3.
- **Class drift.** `WALL CAP`, `UNIT`, `OUTSIDE AIR FAN-COMMON AREA` etc. could create class explosion if we add them all as new YOLO classes. Solution: aggressive aliasing into the existing 35 v9 classes; only add new classes if there are 50+ polygons (WALL CAP qualifies).
- **Holdout discipline.** It's tempting to include Sola in training because it's our reference project. Don't — keep it as a regression check.
- **Page selection.** `train_yolo.py` tiles all pages. Some Takeoff PDFs have annotations on schedule pages (Highlights). The existing extractor filters `Polygon` only and ignores `Highlight`, so this is fine.

## Verification

Post-training:
1. v10's mAP on the existing test bench shouldn't regress (v9 was 79% position recall) — it should match or beat.
2. Sola: v9 product_recall = 33%. v10 target ≥ 50%.
3. The other 3 holdout projects (Imperial, Alliance, Krispy Kreme): each should improve. Even Alliance (where v9 only detected 5 of 17 truth) should jump if exhaust fans / louvers learn from the new corpus.
4. Full benchmark median product_recall should rise from ~3% (current 4-project median) to >25% across all 33 projects.

## Out of scope for v10

- Adding the bubble detector's training data corrections (separate effort).
- Refactoring `train_yolo.py` to multi-source dirs (Path B). Defer until we want a third dataset.
- Class hierarchy refactor. Stay flat for now.
