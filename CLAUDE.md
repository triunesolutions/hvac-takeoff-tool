# HVAC AI Takeoff Tool — Engineering Reference

**Last updated:** April 27, 2026
**Purpose:** Technical reference for engineers (and future Claude Code sessions) working on the codebase. Read this before making changes.

> **Resuming a Claude Code session?** Read this file end-to-end first. Sections 14–17 cover everything that happened after April 21, 2026: tag-bubble detector (Kaggle training in progress), spatial title-block extractor, current bottlenecks, and the "small files first" rule. The rest of the doc (sections 1–13) is still accurate as of April 21 — minor extensions are noted inline.

---

## 1. What This Project Is

An end-to-end pipeline that reads HVAC blueprint PDFs and produces an Excel takeoff (Bill of Materials) matching the team's existing format. Built by Triune Solutions, an HVAC industry company with an in-house takeoff team.

**Primary users (Phase 1):** Triune's internal takeoff estimators — the tool assists them; they verify and correct.
**Eventual users (Phase 4+):** HVAC suppliers via a SaaS product (Rebar competitor).

**Communication style for this codebase:** Blunt, honest, push back when ideas are wrong. Don't sugarcoat accuracy numbers or limitations.

---

## 2. End-to-End Pipeline

```
  blueprint.pdf
       │
       ▼
┌─────────────────────────┐
│ schedule_parser.py      │   Parse every schedule table on every page
│   parse_pdf_schedules() │   → TagVariable list (tag + full row props)
└────────────┬────────────┘
             │  schedules, marks, mark_details, legend, summary, variables
             ▼
┌─────────────────────────┐
│ find_mechanical_pages() │   Heuristic: keywords like MECHANICAL PLAN
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ render_page() + YOLO    │   Tile each page 640×640 @ 200 DPI, detect
│ run_inference()         │   → per-page detection list (cls, cx, cy, box)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ tag_inference.py        │   Three-level assignment:
│   infer_tags()          │     1.  direct class → single-tag auto-assign
│                         │     2a. fingerprint match (text layer + variables)
│                         │     2b. bubble OCR (crop EasyOCR + valid tags)
│                         │     3.  mark untagged as no-tag
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ takeoff_cli.py          │   Write outputs:
│   write_excel()         │     {pdf}_takeoff.xlsx   (team's format)
│   annotate_pdf()        │     {pdf}_annotated.pdf  (boxes + labels)
│   json sidecar          │     {pdf}_variables.json (all variables)
└─────────────────────────┘
```

Every step can be run standalone — each module has its own `__main__` for debugging.

---

## 3. Core Data Model — `TagVariable`

This is the central data structure introduced April 2026 when we pivoted to a schedule-first workflow. Every row in every schedule table becomes one `TagVariable`:

```python
{
    'tag': 'CU-1',                              # the tag string
    'schedule_name': 'HVAC - SPLIT DX CONDENSING UNIT SCHEDULE',
    'page': 2,                                   # 1-indexed PDF page
    'properties': {                              # ENTIRE row — all columns
        'MANUFACTURER': 'CARRIER',
        'MODEL': '40RUQA12',
        'SUPPLY AIR (CFM)': '3650',
        'MCA': '28.7',
        'VOLT/PH': '480V/3PH',
        'INDOOR UNIT (AHU) WT. (LBS)': '427',
        ...                                       # every non-tag column kept
    },
    'inferred_yolo_class': 'CONDENSING UNIT',    # inferred once at parse time
    'source_row_index': 2,                       # for debug traceability
}
```

**Key properties of this structure:**
- **All columns preserved.** Keys are normalized (whitespace collapsed) so multi-line headers like `'MANUFACTURER\n& MODEL'` become `'MANUFACTURER & MODEL'`.
- **One entry per (tag, row).** If a row lists `CU-1,2,3` or `CU-1 thru CU-6`, each tag gets its own `TagVariable` sharing the same property dict.
- **Schedule name attached** so downstream code always knows which schedule a tag came from.
- **YOLO class inferred once** at parse time (not lazily computed later).

**Output artifact:** `{pdf_basename}_variables.json` — written next to the Excel on every CLI run. This is the single source of truth for every downstream step.

**Verification:** `python takeoff_cli.py <pdf> --verify` prints a human-readable dump of every variable with every property, grouped by schedule.

---

## 4. Module Reference

### `schedule_parser.py`
Parses every PDF page for schedule tables. Handles vertical layouts (tags as rows), horizontal layouts (tags as columns — auto-transposed), multi-tag cells, range shorthand, and a pile of noise filters.

**Key functions:**
- `parse_pdf_schedules(pdf_path, exclude_prefixes=None)` → `(schedules, marks, mark_details, legend, summary, variables)`
  - Returns 6-tuple. `variables` is the structured per-tag list.
  - `exclude_prefixes=set()` by default — extracts ALL equipment types.
- `normalize_tag(raw)` → normalized tag string, or None. Applies:
  - Strip `(E)` / `(R)` / `(N)` equipment-status prefixes (Existing / Relocated / New)
  - Multi-line cell handling (`"VAV\n2"` → `"VAV-2"`)
  - Reject refrigerant codes (`R-410A`, `R-454B`, `R-32`)
  - Reject drawing sheet numbers (`M102`, `E301`, `P201`)
  - Reject banned prefixes (`NOTES`, `ROUTING`, `REV`, `SHEET`, etc.)
  - For letter-suffix tags like `VAV-N`: only allow when the prefix is a known HVAC prefix from `TAG_PREFIX_CLASS`.
- `expand_tag_cell(raw)` → list of tags. Handles:
  - single tag: `'A-1'` → `['A-1']`
  - compound: `'A, B, C'` → `['A','B','C']`
  - range: `'CU-1 thru CU-6'` → `['CU-1','CU-2',...,'CU-6']`
  - shorthand: `'AC-1,2'` → `['AC-1','AC-2']` (first prefix carried to following bare numbers)
  - multi-number: `'24\nVAV\n27'` → `['VAV-24','VAV-27']`
- `dump_variables(variables, file=None)` → human-readable dump of every variable + properties, grouped by (page, schedule).

**Header detection:**
Schedule tables are identified in this order:
1. Row containing a `TAG_COL_KEYWORDS` cell (`MARK`, `TAG`, `UNIT TAG`, etc.)
2. Fallback: row with 3+ property keywords (`TYPE`, `MODEL`, `SIZE`, `CFM`, `MANUFACTURER`, etc.) — handles schedules like `AIR DEVICE SCHEDULE` that use `TYPE` as the tag column.
3. If column 0 holds tag-shaped values (`[A-Z]{1,4}-?\d{1,3}[A-Z]?`), treat column 0 as the mark column.

**Horizontal-table detection:** If column 0 contains `MARK`/`TAG` label AND 2+ property keywords, the table is transposed before parsing.

**Schedule name extraction:** Walks up from the header row until it finds a short title (<80 chars, <10 words) — this skips prose rows like contractor notes.

### `tag_inference.py`
Three-level system for assigning tags to YOLO detections.

**Levels (in order):**
1. **`level1_direct_mapping`** — for each YOLO class, if the schedule has exactly 1 tag for that class, auto-assign every detection of that class to that tag. Works for Flex-style projects (single A/B/C/D diffuser per class).
2. **`level2_fingerprint_matching`** — for multi-tag classes (e.g., CU-1…CU-6), build per-tag fingerprints of distinctive value tokens from `variables`, scan the PDF text layer near each detection, score (det, tag) pairs by fingerprint overlap. Greedy 1:1 assignment. Good when CFM/model values are printed on the drawing.
3. **`level2b_bubble_ocr`** — for multi-tag classes, crop a ~150px region around each untagged detection, run EasyOCR on the crop, match tokens against the valid tag list for that class. Each detection picks its closest match — **no 1:1 cap** (tags like `A1` repeat many times across a floor plan). This is the highest-yield level for drawings that show tag bubbles next to symbols.
4. **`level2_size_cfm_matching`** — legacy path; only runs when no `variables` were provided (for back-compat with older callers). Lacks class filtering; superseded by 2a+2b.
5. **`level3_class_fallback`** — sets `tag=None`, `tag_method='none'` for anything still untagged.

**Class-to-tags mapping:** `build_class_to_tags_from_variables(variables)` is the current source of truth — uses each variable's `inferred_yolo_class` directly. The legacy `build_class_to_tags(mark_details, schedules)` only runs when no variables are available.

**Class inference from schedule text:** `_infer_yolo_class_from_service(service_text, mounting_text)` checks for LAY-IN/T-BAR, CEILING, SURFACE/EXPOSED, LINEAR, and a broader generic-GRD fallback covering `PERFORATED`, `PLAQUE`, `FACE`, `MOUNTED`. If that returns None, `_infer_class_from_tag(tag)` looks up the tag prefix in `TAG_PREFIX_CLASS`.

**`TAG_PREFIX_CLASS`** is the canonical prefix → YOLO class map:
```python
{
    'EF','SF','CF','RF': 'FAN' family
    'CU','AC','AHU','RTU','FCU','HP': major equipment
    'EUH','UH','EH','BH': heaters
    'VAV','VRF','ERV'
    'MD','MVD','FD','FSD','BD': dampers
    'L','LVR': louvers
    'GR','RG','SD','CD','SA','RA','EA','SB': diffusers/grilles
    'LD': linear plenum
}
```
Extend this when new prefixes show up in projects.

### `tag_matcher.py`
EasyOCR utilities used by Level 2b. Key functions:
- `ocr_near_detection(img, det, crop_size, conf_threshold)` — OCR a small crop around a detection, return words with coords translated back to page space.
- `match_valid_tags(words, valid_tags)` — filter OCR tokens to those that normalize to a valid schedule tag (case-insensitive, punctuation-tolerant).

EasyOCR is lazy-loaded on first call (the model is ~100MB).

### `tag_extractor.py`
Earlier iteration — PyMuPDF text-layer tag extraction + summary helpers. Still used by `takeoff_cli.py` for `summarize_detections_by_tag()`. Superseded by the variables path for the extraction side.

### `takeoff_cli.py`
The CLI entry point. Flags:
- `--verify` — print full variable dump to stdout alongside normal run.
- `--schedule-only` — parse schedule + write `variables.json`, skip YOLO detection. Fast iteration for schedule debugging.
- `--pages 1 2 3` — only process specific pages (1-indexed).
- `--all-pages` — bypass the MECHANICAL-PLAN keyword filter.
- `--conf 0.4` — YOLO confidence threshold (default 0.4).
- `--model <path>` — override the default model (`models/hvac_yolov8s_v9.pt`).

**Key detail:** takeoff_cli keeps rendered page images in a `page_images` dict after YOLO inference and passes them to `infer_tags` so Level 2b can OCR without re-rendering.

**Excel output format** (matches the team's Bluebeam takeoff):
```
PRODUCT | BRAND | MODEL | QTY | TAG | NECK SIZE | MODULE SIZE | DUCT SIZE | TYPE | MOUNTING | REMARK
```
Two sheets: `Triune Takeoff` (grouped by class+tag) and `RawData` (every detection flat).

**Property lookups in Excel** use the tolerant `_prop(details, [keywords])` helper — finds any key containing any keyword, case-insensitive. This handles both `MANUFACTURER & MODEL` (combined) and separate `MANUFACTURER` / `MODEL` columns, including `MAKE / MODEL` (split on `' / '`).

### `class_aliases.py`
Merges duplicate class names from training-data annotation typos. Applied during dataset prep in `train_yolo.py`.

### `train_yolo.py`
End-to-end training pipeline: extracts Polygon annotations from labeled PDFs → tiles images 640×640 → emits YOLO dataset → trains. Designed to run on Google Colab / Kaggle T4 GPU. Run on local CPU only for smoke tests.

### `benchmark.py`
Detection accuracy scorer against ground-truth labeled PDFs.

---

## 5. Test Projects & Accuracy

Three projects tested end-to-end as of April 2026:

| Project | Schedule style | Variables | Tagged detections | Notes |
|---|---|---|---|---|
| **Flex 230** | Vertical, simple | 19 (A×6, B, C×2, D, 9 VAVs) | 38 / 42 (90%) | Level 1 handles everything (each class has one tag) |
| **Aritzia Americana** | Horizontal stacked schedules | 36 (AC, AHU, CU, FCU, FC, VAV, ERV + OUTSIDE AIR) | Not benchmarked end-to-end | Horizontal transpose, status prefix stripping, refrigerant filter all working |
| **United Mechanical** | Mixed 8-schedule page | 31 (CU-1..6, EF-1..8, FCU-1..7, EUH-1/2, CF-1..3, L-1, A1/B1/C1) | 107 / 185 (58%) | All Levels fire; bubble OCR does the heavy lifting on A1/B1/C1 counts |

**Breakdown of the United gap (78 untagged out of 185):**
- ~35 detections in classes with no schedule match (`MANUAL VOLUME DAMPER`, `VENT CAP`, `SPLIT SYSTEM`). These are class-aliasing gaps — `SPLIT SYSTEM` YOLO detections likely correspond to `CONDENSING UNIT` schedule entries but the class names don't match.
- ~26 `AD-GRD` detections where no readable bubble text sits within 140px.
- ~13 likely YOLO over-detections (more heaters than the schedule suggests).

**YOLOv8s v9 (production model):**
- Trained on Kaggle T4 GPU
- 124 projects, ~25K tiles, 35 classes
- 66% full recall, 79% position recall on 12-project benchmark set
- Production model at `models/hvac_yolov8s_v9.pt`

---

## 6. Known Limitations

1. **Class-name mismatches** between YOLO (`SPLIT SYSTEM`, `AD-GRD`) and schedule-inferred class (`CONDENSING UNIT`, `AD-T-BAR SUPPLY`). No alias layer yet.
2. **Tag case sensitivity** — `A1` and `a1` (e.g., United AIR DEVICE SCHEDULE) collapse to the same uppercase tag. Different sizes remain distinguishable via `properties`, but the tag string is identical.
3. **Large PDFs crash pdfplumber** — St Elizabeth (4+ GB) can't be loaded. Need a streaming/chunked alternative.
4. **EasyOCR unreliable on very small bubbles** — letters inside circle stamps often fail. Level 2b works best when the tag bubble is at least 15-20px tall in the rendered image (at 200 DPI).
5. **Horizontal schedules with multiple stacked sub-sections** (e.g., Aritzia's combined AHU+CU schedule) sometimes pdfplumber fragments into many sub-tables; schedule names can be lost per fragment.
6. **Fingerprint matching is limited** to drawings where distinctive property values (CFM, model numbers) are printed near each detection. Many drawings print only the tag bubble.

---

## 7. CLI Usage Cookbook

**Quick schedule sanity check (no YOLO):**
```bash
python takeoff_cli.py path/to/blueprint.pdf --schedule-only
```
Parses schedules, writes `variables.json`, prints summary. ~30-60s for typical project.

**Full schedule dump to verify extraction:**
```bash
python takeoff_cli.py path/to/blueprint.pdf --schedule-only
# inspect:
cat <pdf_stem>_takeoff/<pdf_stem>_variables.json | python -m json.tool
```
Or use `--verify` for the human-readable stdout version.

**Full pipeline (YOLO + tag inference + Excel):**
```bash
python takeoff_cli.py path/to/blueprint.pdf
```
Outputs to `<pdf_stem>_takeoff/`. Takes ~2-10 min depending on page count (YOLO ~20s/page, Level 2b OCR ~0.5-1s/detection).

**Single-module debug:**
```bash
python schedule_parser.py path/to/blueprint.pdf --verify
python tag_inference.py path/to/blueprint.pdf     # fake detections for smoke test
python tag_matcher.py path/to/blueprint.pdf 5     # OCR test on page 5
```

---

## 8. Setup & Dependencies

```bash
git clone https://github.com/triunesolutions/hvac-takeoff-tool.git
cd hvac-takeoff-tool
pip install PyMuPDF Pillow opencv-python-headless pandas openpyxl easyocr ultralytics pdfplumber
```
Python 3.12+ required (dev on 3.14).

**Training data** is NOT in the repo. Lives at `C:\Users\JFL\Downloads\Triune\data to train\projects\` — ~130 projects as of April 2026. Ask JFL for access.

**Production model:** `models/hvac_yolov8s_v9.pt` (committed). 35 classes.

---

## 9. Development Workflow

**Adding a new schedule parsing rule:**
1. Find a failing project. Run `python takeoff_cli.py <pdf> --schedule-only`.
2. Dump raw pdfplumber tables to see what's going in (see the raw dump snippet in `schedule_parser.py.__main__`).
3. Add the fix in `schedule_parser.py` (parser logic) or `normalize_tag()` (value validation).
4. Re-run with `--schedule-only` — fast iteration.
5. Regression-check Flex 230 + one other project.

**Adding a new tag-inference technique:**
1. Add a new `levelX_*` function in `tag_inference.py`.
2. Wire into `infer_tags()` after existing levels.
3. Write its signature to accept `detections`, `class_to_tags`, and whatever extras it needs (variables, page_images, etc.).
4. Each level should only tag currently-untagged detections, and attach `tag_method` + `tag_confidence` for traceability.

**Testing locally without running full YOLO:**
- Use `--schedule-only` for schedule changes
- Use `tag_inference.py __main__` with fake detections for inference changes
- Use `tag_matcher.py __main__` for OCR changes

**Commit style:** The team prefers commits that describe the BEHAVIOR change + WHY, with project-specific verification notes (e.g., "Flex 230: 19 variables unchanged; United: 3% → 58% tagged"). See recent git log for examples.

---

## 10. What Failed (so we don't repeat it)

1. **Template matching from legend crops** — too many false positives from ductwork.
2. **Hough circle detection** — 783 circles per page (ceiling grid, duct connections).
3. **Full-page EasyOCR for tag detection** — 12% recall. Most tags are CAD vector graphics, not text objects.
4. **Cross-class tag matching** (original Level 2) — matched AD-GRD detections to FCU tags because it didn't filter by YOLO class. Fixed by requiring `class_to_tags[det.cls]` to contain the candidate tag.
5. **Single-letter + 3-digit regex accepting drawing sheet numbers** (`M102`, `E301`). Explicitly rejected now.
6. **Trusting "MANUFACTURER\n& MODEL" literal key lookups** — multi-line headers vary across schedules. Now normalized at parse time.

---

## 11. Roadmap

**Phase 1 (DONE):**
- YOLOv8 detection model trained and in production (`hvac_yolov8s_v9.pt`)
- Accuracy benchmarking pipeline
- Schedule parser handling 3+ distinct schedule styles
- TagVariable extraction with full property preservation
- 3-level tag inference (direct / fingerprint / bubble OCR)
- Excel + annotated PDF output matching team format
- JSON sidecar for programmatic access

**Phase 2 (NEXT):**
- Class aliasing layer (YOLO class → schedule class family)
- Retrain model on more projects (especially French Beaconsfield + SouthVAC styles)
- Improve Level 2b with crop preprocessing (upscale + binarize before OCR)
- Handle large PDFs (stream pdfplumber or chunk by page)
- Review UI where the team corrects mistakes → corrections become training data

**Phase 3:**
- Value extraction (CFM, dimensions from plan text) for richer BOM
- Larger YOLO variant (s → m or RT-DETR) for higher recall
- Multi-PDF project support (combine spec book + drawings)

**Phase 4:** Public SaaS product.

**Phase 5:** Expand to plumbing and electrical takeoffs.

---

## 12. File Organization

```
hvac-takeoff-tool/
├── CLAUDE.md                     ← this file
├── PRD.md                        ← full product requirements
├── WHAT_WE_ARE_BUILDING.md       ← plain-English status
│
├── takeoff_cli.py                ← main CLI entry point
├── schedule_parser.py            ← schedule table parsing + TagVariable
├── tag_inference.py              ← 3-level tag→detection matching
├── tag_matcher.py                ← EasyOCR helpers (Level 2b)
├── tag_extractor.py              ← legacy text-layer tag extraction
├── class_aliases.py              ← training-data class merging
│
├── train_yolo.py                 ← YOLO training pipeline
├── benchmark.py                  ← detection accuracy scorer
├── colab_train.ipynb             ← Colab training notebook
│
├── models/
│   └── hvac_yolov8s_v9.pt        ← production model (35 classes)
├── templates/                    ← legend symbol templates (reference)
├── output/                       ← per-project takeoff outputs (gitignored)
├── runs/                         ← YOLO training runs (gitignored)
└── yolo_dataset/                 ← training tiles (gitignored)
```

---

## 13. Rules & Guardrails

- **Don't over-engineer infrastructure before core detection is >80% recall.**
- **Don't build a web UI until the model is good enough for the team to use daily.**
- **Don't try to generalize across all drawing styles yet** — nail the current training projects first, then expand.
- **Don't use Claude Vision API as primary detection** — cost/latency at scale. Save it for spec-book parsing.
- **Don't commit training data (PDFs) or large artifacts** (yolo_dataset/, runs/).
- **Always regression-check Flex 230** after parser changes — it's the known-good baseline.
- **Keep the Excel output format identical to the team's Bluebeam format** — column order, header text, row grouping. They re-use the Excel downstream.
- **Use small PDFs (≤15–20 MB) for dev/test.** Team is sourcing more small files. Get accurate on those first; large files (St Elizabeth 4+ GB) are deferred until the small-file pipeline is solid.

---

## 14. Tag-Bubble Detector (Phase 2 — IN PROGRESS as of April 27, 2026)

**Goal:** A second YOLO model trained specifically to detect **tag bubbles** (the small `A1` / `CU-1` labels next to symbols). Used as a stronger Level-2b signal: instead of OCR-ing arbitrary 150 px crops, we first detect bubble bboxes, then OCR only those tight crops.

**Why a separate model:** The production v9 model detects equipment classes (diffusers, fans, CUs). It does *not* localize tag bubbles. Tag bubbles are a different visual entity — small circles/ovals/rectangles with 1–4 chars of text inside, drawn near (but not on) the symbol they identify. Treating them as their own detection class lets us crop tightly for OCR and get higher recall than the current 150 px window heuristic.

### 14.1 Pipeline (already built)

```
ground_truth.jsonl  +  90 Bluebeam takeoff PDFs
        │
        ▼
build_tag_dataset.py            → tag_dataset/ (26,722 PNG crops, 320×320)
                                  labels.jsonl with symbol bbox (source_rect_in_crop)
        │                         per (project, page, count) sample
        ▼
label_tag_bubbles_ocr.py        → tag_bubble_labels.jsonl (one row per crop)
  (Colab GPU, EasyOCR + fuzzy)    9,908 hits / 26,722 (37.1%) — realistic ceiling
        │                         line-buffered + --resume (crash-safe)
        ▼
build_yolo_tag_dataset.py       → yolo_tag_dataset/ (YOLO 2-class)
                                  23,306 train / 3,416 val crops
                                  per-project split (9 val / 81 train, seed 42)
                                  classes: 0=symbol  1=tag_bubble
        │
        ▼
kaggle_train_tag_detector.ipynb → hvac_tag_detector_v1.pt
  (Kaggle T4 ×2)                  YOLOv8s · 60 epochs · imgsz=320 · batch=64
                                  ~2–3 hr full run
```

### 14.2 Files added in this phase

| File | Role |
|---|---|
| `label_tag_bubbles.py` | Earlier text-layer approach (PyMuPDF `get_text("words")`). Got 17.8% on N=500 — abandoned. Tags are CAD vector paths, not text objects. Kept for reference. |
| `label_tag_bubbles_ocr.py` | **Production OCR labeler.** Runs EasyOCR on each crop with the known target tag, fuzzy-matches with edit-distance ≤1 + OCR-confusion table (I↔1, O↔0, S↔5, Z↔2, B↔8, G↔6, T↔7, L↔1, Q↔0, D↔0). Otsu-binarized + 3× upscale before OCR. Line-buffered output (`buffering=1`) + `--resume` for crash safety. |
| `colab_label_bubbles.ipynb` | Colab notebook: GPU check → upload script + dataset zip via `google.colab.files.upload()` → unzip → install easyocr → run labeler → download `tag_bubble_labels.jsonl`. **Used for the actual run.** |
| `tag_bubble_labels.jsonl` | 26,722 rows. Each: `{img, tag, bubble_rect_in_crop, reason}`. 9,908 hits, 16,169 no_match, 645 no_text. Committed (3.7 MB) — feeds dataset builder. |
| `build_yolo_tag_dataset.py` | Convert `tag_dataset/` + `tag_bubble_labels.jsonl` → YOLO-format. Per-project train/val split. Outputs `yolo_tag_dataset/{images,labels}/{train,val}/` + `data.yaml`. |
| `kaggle_train_tag_detector.ipynb` | Kaggle T4 training notebook. Detects auto-extracted dataset path, regenerates `data.yaml` with Kaggle paths, trains YOLOv8s. Outputs `hvac_tag_detector_v1.pt`. |

### 14.3 Status (as of April 27, 2026)

- **Dataset prep:** ✅ Complete. Zipped `yolo_tag_dataset.zip` (574 MB, 53,445 entries) gitignored.
- **Kaggle training:** 🔄 In progress. ~42 / 60 epochs at last check.
- **Next:** When training completes, download `hvac_tag_detector_v1.pt` → integrate into `tag_inference.py` Level 2b. New flow: tag_bubble model on each page → for each detected bubble, OCR the bubble's tight crop → match against valid tags for the nearest equipment-detection class.

### 14.4 Why 37.1% OCR hit rate is the floor (not a bug)

- **17.3% of crops are MVD/symbol-only tags** with no readable bubble (just shapes).
- Many tags are tiny inside small circles → EasyOCR can't read 8 px text reliably.
- Improvements tried (allowlist, Otsu binarize, lower text_threshold, fuzzy match) plateaued at ~42% — accepted as good enough for ~10K bubble bboxes, which is plenty for a 2-class detector.

### 14.5 Crash-safety lessons (carry forward)

1. **Line-buffer file writes** when running on Colab: default Python text mode is block-buffered → 0 bytes on disk after 28 min of running, would lose hours on disconnect. Always pass `buffering=1` to `open()` and `python -u` to the entrypoint.
2. **Use `python zipfile` not PowerShell `Compress-Archive`** for big trees — Compress-Archive stalled at 0 bytes for 1 min on 26K files; Python zipfile with `ZIP_STORED` did 580 MB in 11 s.
3. **Colab disconnect banner ≠ session dead.** Backend keeps running. Don't close the tab.

---

## 15. Title-Block Metadata Extractor (April 27, 2026)

`takeoff_cli.py` now extracts and prints project metadata at the top of every CLI run, alongside the existing schedule parsing.

### 15.1 Two-pass approach

1. **`extract_project_info(pdf)`** — regex-based passes over `page.get_text("text")`:
   - Pass 1: inline `LABEL: VALUE` patterns (`SCALE:`, `DATE:`, `PROJECT NAME:`, etc.)
   - Pass 2: stacked labels (label on one line, value on the next non-empty line — for simple title blocks)
   - Pass 3: firm-name regex (ALL CAPS + `ENGINEERING|CONSULTANTS|ASSOCIATES|...`)
   - Pass 4: street-address regex
   - Pass 5: **calls** `extract_project_info_spatial()` and merges results.

2. **`extract_project_info_spatial(pdf)`** — bbox-aware extraction for CAD title blocks (e.g., Gensler) where labels and values are placed in a graphic grid. For each known label span (`Project Name`, `Project Number`, `Description`, `Sheet`, `Scale`, `Date`, `Drawn By`, etc.):
   - Convert all spans from mediabox → display coords (handles 270° rotation).
   - Find the value span **directly above** the label within ~3× label width and ~6× label height (CAD layout). Fallback: span to the right on same baseline (form-style layout).
   - When multiple instances of the same label exist on a page (e.g., a `Description` column header in the Mechanical Legend table AND the real sheet description), pick the candidate whose value span has the **largest text height** — sheet titles are typographically larger than column header values.
   - **Title-block fields are restricted to page 0** to avoid picking up per-drawing scale callouts (`SCALE: NONE` printed under each detail box on later pages).

### 15.2 Output keys

`project`, `project_no`, `description` (sheet title), `sheet` (sheet number), `scale`, `date`, `engineer`, `firm`, `address`, `revision`. Printed in `print_project_info()`. Written to `{pdf_stem}_project_info.json` sidecar.

### 15.3 Verified on Flex 230

```
Project     FLEX+ 220
Project No. 55.4020.332
Sheet Title MECHANICAL TITLE SHEET
Firm        PLUM ENGINEERING, INC.
Address     9530 TOWNE CENTRE DRIVE
Date        06.02.25
```

### 15.4 Known gaps

- **Sheet number (e.g., `M001`)** not labeled in Gensler title blocks — sits as a bare span. Needs a typography heuristic (find the tallest standalone span matching `^[A-Z]\d{2,4}$` near the title-block region).
- **Date is the *oldest* issue** from the revision history table, not the latest. Need to pick the largest date or the date in the last "ISSUE" row.
- **Engineer initials** (e.g., `JE | MG | RW`) live in the issue history grid, not a single labeled cell. Treated as low priority.
- **Confirmed:** Flex 230 has no title-block scale at all. The previous `Scale: NONE` was bleeding in from per-drawing scale stamps on later pages — fixed by restricting title-block fields to page 0.

---

## 16. Memory and User Preferences (cross-session)

These have been saved into the user's `~/.claude/projects/.../memory/` and persist across Claude Code sessions:

- **Communication style:** Blunt, honest, push back when wrong. Don't sugarcoat accuracy numbers.
- **Repo sync:** Commit and push to `triunesolutions/hvac-takeoff-tool` regularly. Implicit authorization for routine commits.
- **Small files first:** Use PDFs ≤15–20 MB for dev/test until pipeline is accurate. Big PDFs (St Elizabeth) are Phase-2.
- **Bluebeam ground-truth calibration:** Team is preparing a CSV export to calibrate detection accuracy. Not yet delivered.

---

## 17. Pending / Next-Up

1. **Train + integrate tag-bubble detector** — Kaggle run finishing today. After download, wire `hvac_tag_detector_v1.pt` into `tag_inference.py` Level 2b.
2. **Title-block extractor improvements** — sheet number heuristic, latest-date selection, engineer-initials parsing.
3. **Class aliasing layer** — `SPLIT SYSTEM` (YOLO) → `CONDENSING UNIT` (schedule), `AD-GRD` family alignment. Listed in Phase 2 roadmap; needed to close the United Mechanical 78-untagged gap.
4. **Larger small-file test set** — team is sourcing more PDFs in the ≤20 MB band.
