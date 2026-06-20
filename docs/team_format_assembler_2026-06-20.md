# Team-format takeoff assembler (2026-06-20)

**Problem this solves:** the pipeline was emitting **counts only** (one aggregate row
per tag). The team's takeoff is far richer — **one row per (tag, neck size)** with
BRAND, MODEL, NECK SIZE, MODULE SIZE, CFM, TYPE, MOUNTING, ACCESSORIES. An estimator
prices per neck size, so aggregate counts are not usable. This assembler produces the
team's column layout.

## Key insight: a team takeoff is TWO data streams joined

| Column | Source |
|---|---|
| PRODUCT | YOLO class |
| BRAND, MODEL, TYPE, MOUNTING, MODULE SIZE, ACCESSORIES | **schedule table** (keyed by tag) |
| TAG | plan — bubble next to the symbol |
| **NECK SIZE, CFM** | **plan — callout next to the tag bubble** (`(C) 14X6 / 335 CFM`) |
| QTY | count of detections grouped by (product, tag, neck size), × any `TYP n` |

Neck size is **not in the schedule** — it is annotated per-instance on the plan, which
is why the same tag (e.g. C) splits into 6X6, 8X8, 14X6, 18X8 rows. Reading it is what
turns "counts" into a real takeoff.

## Why PaddleOCR (new dependency, isolated venv)

EasyOCR garbles these CAD callouts (`COMPRESSED`→`COHPRESSED`); pdfplumber/img2table
miss the borderless/CID schedules entirely (see `benchmark_newcorpus_2026-06-20.md`).
PaddleOCR reads both cleanly:

- **Schedule tables** via `TableRecognitionPipelineV2` — read the Cityvet CID-font
  AIR DISTRIBUTION DEVICES table that pdfplumber/img2table/EasyOCR all failed on
  (`A·TITUS·TMS·LOUVER FACE`, … all correct).
- **Plan callouts** via `PaddleOCR.predict` on each detection crop — returned
  `14"x6"`, `8"ø`, `6"x6"`, `170 CFM`, `75 CFM`, `TYP.2` cleanly.

**Paddle does not support Python 3.14** (the main interpreter). It runs in a separate
**Python 3.12 venv** (`ppenv/`): `paddlepaddle 3.3.1`, `paddleocr 3.7.0`,
`paddlex[ocr] 3.7.1`, `PyMuPDF`. Two flags are required on this CPU build:
`FLAGS_use_mkldnn=0` (oneDNN PIR bug) and `use_layout_detection=False` for the table
pipeline (the layout model segfaults — feed pre-cropped tables instead).

## scripts/team_takeoff_assemble.py

Consumes an existing pipeline output dir (`*_detections.json` + `*_variables.json`),
renders each plan page, OCRs a padded crop around every detection, binds the nearest
plausible neck-size + CFM token (distance-gated; dims capped 3–24" to exclude duct
runs), joins schedule props by tag, groups by (product, tag, neck size), and writes the
team's columns.

```
ppenv/Scripts/python.exe scripts/team_takeoff_assemble.py <benchmark_output/<project>> <out.xlsx>
```

## Result — SPE Sinton flagship

Per-(tag, neck-size) rows with model/type/CFM matching the team:

| Tag·Neck | Team qty | Ours qty | |
|---|--:|--:|---|
| E · 6" | 3 | 3 | exact |
| B · 10X10 | 4 | 4*/6 | close |
| C · 6X6 | 3 | 3 | exact |
| C · 18X8 | 3 | 9 | over (bind noise) |
| D · 8X8 | 3 | 3 | exact |
| D · 18X8 | 3 | 3 | exact |

MODEL (TITUS 300FL/350FL/PAR-AA/TDC-AA) and TYPE match exactly; we also fill CFM, which
the team's own SPE sheet left blank.

## Remaining work

1. **MOUNTING / MODULE SIZE / ACCESSORIES** — present in the schedule but our text
   parser only extracted 2 columns on SPE (`MFR. & MODEL`, `TYPE`). Use PP-Structure
   (`TableRecognitionPipelineV2`) on the schedule sheet to recover all columns.
2. **Neck-bind noise** — some rows over/under (C·18X8=9 vs 3). Bind to the token
   directly *below* the bubble, not just nearest; reject duct-run sizes on the duct line.
3. **Tagging gaps** — VAV/VRF/FAN and some AD-GRD untagged (upstream tag inference).
4. **Fold into the pipeline** — currently a post-process over pipeline output; once
   stable, move the plan-stream OCR into `tag_inference.py` Level 2b (it already crops
   the same region to read the tag) and the team layout into `write_excel`.

## Status

Architecture proven end-to-end: both streams read correctly, the join produces the
team's row structure, and many (tag, neck, qty) rows match the team exactly. The piece
the team flagged as missing ("only counts") is solved.
