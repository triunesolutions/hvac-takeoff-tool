# HVAC AI Takeoff Tool — Engineering Reference

**Last updated:** May 5, 2026
**Purpose:** Technical reference for engineers (and future Claude Code sessions) working on the codebase. Read this before making changes.

> **Resuming a Claude Code session?** Read this file end-to-end first. Section 19 (added May 5) covers the Label Studio review loop and the 6-project ground-truth dataset feeding v11. Sections 14–17 cover post-April-21 work (tag-bubble detector, title-block extractor). Sections 1–13 are still accurate as of April 21 — minor extensions noted inline.

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

1. **Train + integrate tag-bubble detector** — DONE. `hvac_tag_detector_v1.pt` wired into `tag_inference.py` as Level 2b' (`level2b_bubble_detect`). Old windowed OCR remains as a fallback.
2. **Class aliasing layer** — DONE for the AD-GRD family. `YOLO_CLASS_ALIASES['AD-GRD']` is now a list `[AD-T-BAR SUPPLY, AD-T-BAR RETURN, AD-SURF SUPPLY, ...]` with `_expand_class_for_bubble()` letting the bubble-OCR text disambiguate. Single-letter tag prefixes (`S`/`R`/`E`) added to `TAG_PREFIX_CLASS`.
3. **Title-block extractor improvements** — sheet number heuristic, latest-date selection, engineer-initials parsing.
4. **Benchmark across the 35-project sample set** — DONE. `benchmark_samples.py` runs the CLI on every project under `SAMPLE FILES 27.04.26/`, scores our generated xlsx vs the team's `Completed Takeoff/*.xlsx` by per-product and per-(product, tag) QTY overlap. See section 18.

---

## 18. Sample-Project Benchmark

`benchmark_samples.py` is the regression bar for any pipeline change.

**Input dataset:** `C:\Users\JFL\Downloads\SAMPLE FILES 27.04.26\SAMPLE FILES 27.04.26\` — ~35 small commercial projects (median 2.77 MB, 87/141 PDFs <5 MB), each with both `Plans_Specs/<plan>.pdf` and `Completed Takeoff/<takeoff>.xlsx`.

**Run:**
```
python benchmark_samples.py                              # all projects (~50–60 min)
python benchmark_samples.py --projects "Sola Salons"     # subset
python benchmark_samples.py --cache                      # skip projects with existing xlsx
```

**Outputs (in `benchmark_output/`, gitignored):**
- `benchmark_results.csv` — one row per project: status, team_total, our_total, product_recall, product_precision, tag_recall, tag_precision, schedule_tags_found, runtime_s.
- `benchmark_per_product.csv` — one row per (project, product): team_qty, our_qty, match_qty, over, under.
- `benchmark_summary.md` — leaderboard + status breakdown + top-10/bottom-10.

**Score model:**
- `match_qty = min(team_qty, our_qty)` per product (or per product+tag).
- `recall = sum(match) / sum(team)` — what fraction of the team's count we caught.
- `precision = sum(match) / sum(ours)` — what fraction of our detections were real.
- Per-tag is secondary because some teams use manufacturer-as-tag (Krispy Kreme: `KRUEGER`/`MARS`/`BERNER`) — tag_recall will be low there even when product_recall is fine.

**What scores tell you:**
- **`status == 'crashed'`** → the CLI subprocess died. Look at `error` column for the stderr tail.
- **`status == 'no_detections'`** → CLI ran but YOLO found zero equipment. Probably wrong page selection.
- **`schedule_tags_found == 0`** → `parse_pdf_schedules` couldn't find any tags. Schedule parser issue, fix in `schedule_parser.py`.
- **`product_recall > 0` but `tag_recall == 0`** → detection works, tag inference doesn't. Probably the `class_to_tags` map is empty for the YOLO classes we found — check `_resolve_class` and `TAG_PREFIX_CLASS`.
- **`our_total >> team_total` (low precision)** → YOLO over-detection. Bump confidence threshold or retrain.
- **`our_total << team_total` (low recall)** → YOLO under-detection. Need v10 with these projects in training data.

**Reference numbers (April 2026 baseline, post bubble-detector + class-aliasing):**
- Sola Salons: product_recall ≈ 32% (32 / 100 truth caught) — best case for current pipeline shape.
- 677 Imperial / Alliance / Krispy Kreme: 0% — three different failure modes documented in commit history.

**v10 baseline (April 30, 2026):** median full-recall jumped from 22% → 44% on scored projects after retraining with the expanded sample dataset. v10 lives at `models/hvac_yolov8s_v10.pt` and is the default for new runs.

---

## 19. Label Studio Review Loop & Ground-Truth Dataset (May 5, 2026)

A human-in-the-loop pipeline for verifying every detection v10 makes on a project, capturing corrections as ground truth, and feeding them straight into v11 retraining.

### 19.1 Pipeline

```
takeoff_cli.py <pdf>                               # produces detections.json sidecar
        │
        ▼
export_to_label_studio.py "<project_dir>"          # renders pages 200 DPI → base64 PNG,
        │                                          # uploads tasks with v10 boxes as predictions
        ▼
Human / Claude-in-Chrome reviews in LS UI          # delete phantoms, relabel wrong-class
        │                                          # do NOT add new boxes (separate workflow)
        ▼
import_from_label_studio.py "<project_dir>"        # IoU-joins truth back against detections.json
        │                                          # writes ls_ground_truth.json,
        │                                          # ls_discrepancy_report.csv, ls_summary.txt
        ▼
ground_truth/<project>/                            # tracked in git, feeds v11 retraining
```

### 19.2 Files

| File | Role |
|---|---|
| `takeoff_cli.py` | Now also writes `<pdf_stem>_detections.json` next to the Excel (per-page list of `{cls, tag, conf, x1,y1,x2,y2}`). External tools key off this rather than re-running inference. |
| `export_to_label_studio.py` | Reads detections.json, renders each PDF page at 200 DPI as base64 PNG, builds an LS labeling-config XML with all 38 model+alias classes (HSL palette), uploads tasks. Auto-exchanges the JWT refresh token from `~/.label_studio_token` for a 5-min access token before each API call. |
| `import_from_label_studio.py` | Pulls verified annotations from LS, IoU≥0.4 joins truth bboxes against v10 detections.json. Emits per-detection status (`accepted` / `relabeled` / `deleted` / `added`), phantom-class counts, and v10→truth class confusion totals. |
| `batch_prepare_review.sh` | Runs `takeoff_cli.py` then `export_to_label_studio.py` for a list of projects sequentially. |
| `ground_truth/<project>/` | Tracked output: `ls_summary.txt` (counts), `ls_ground_truth.json` (verified bboxes per page), `ls_discrepancy_report.csv` (per-detection status). |

### 19.3 Auth gotchas

- LS 1.23 only exposes JWT-style refresh tokens as PATs. The legacy `Authorization: Token` header returns 401. Both export/import scripts call `POST /api/token/refresh/` and use `Authorization: Bearer <access>`.
- Project title is **capped at 50 chars**. Long folder names (BMO, Saint Mary's) require `--ls-project-name "HVAC Review — Short Name"`. The default title is `HVAC Review — <project_dir.name>`. **TODO:** clamp this in `export_to_label_studio.py` so the batch script doesn't fail silently on long names.
- Run `import_from_label_studio.py` with `python -X utf8` on Windows — the summary's `→` arrow crashes the cp1252 console (files are written first, so a crash here is cosmetic).

### 19.4 First batch results (6 projects, May 5)

| Project | Accepted | Relabeled | Deleted (phantom) | LS proj id |
|---|---:|---:|---:|---:|
| Sola Salons | 112 | 72 | 12 | (deleted, archived) |
| Erewhon - Pacific Palisades | 80 | 0 | 14 | 5 |
| The Bungalow - San Diego | 88 | 6 | 7 | 6 |
| Anaheim 82 | 72 | 1 | 2 | 7 |
| BMO Santee 2026 Reno | 54 | 10 | 0 | 8 |
| Saint Mary's Stadium | 69 | 0 | 26 | 9 |
| **Total** | **475** | **89** | **61** | |

### 19.5 Two dominant signals for v11

1. **Legend / schedule / details sheets are the #1 phantom source** (~50 of 61 phantoms across the 6 projects). v10 fires equipment boxes onto every legend symbol row, every schedule table title, and every detail-drawing fitting (pipe hangers, anchor bolts, mounting brackets). A simple sheet-type filter — skip pages whose title block contains `LEGEND`, `SCHEDULE`, `DETAILS`, or `NOTES` — would knock out most phantoms before training even starts. Cheapest win on the table.
2. **AD-GRD vs AD-T-BAR SUPPLY is the #1 class confusion** (89 relabels; 72 of those on Sola alone). v10 collapses square T-bar lay-in supply diffusers into the generic AD-GRD class. v11 retraining with the corrections in `ground_truth/` should fix this directly — the `class_aliases.py` workaround can probably go away.

### 19.6 Other patterns flagged but not yet acted on

- **Returns vs supplies on identical-looking square diffusers** — supply/return discrimination is being driven by surrounding context (CFM tags, hatch direction) more than the symbol itself. Worth a future spot-check.
- **FAN vs EXHAUST FAN vs PACKAGED ROOFTOP UNIT** — recurring on roof plans (BMO, Anaheim, Saint Mary's). v10 frequently labels rooftop units as FAN or EXHAUST FAN. Not in our explicit relabel rule yet — extend on the next review pass.
- **Localization noise** — overlapping duplicate boxes on the same diffuser (Erewhon, Bungalow). Bug 3 (page-level NMS) was already queued; this confirms the need.
- **AD-GRD on louvers and exhaust risers** — they're real HVAC equipment but the wrong class. Currently kept as AD-GRD because the relabel rule only covers AD-T-BAR SUPPLY.

### 19.7 Workflow per new project

```bash
# 1. Run takeoff (produces detections.json)
python takeoff_cli.py "<plan>.pdf" --model models/hvac_yolov8s_v10.pt --output-dir "benchmark_output/<project>"

# 2. Push to Label Studio
python export_to_label_studio.py "<project>" \
    [--ls-project-name "HVAC Review — Short Name"]   # only if folder name > ~30 chars

# 3. Review in browser at http://localhost:8080/projects/<id>/data
#    (Claude in Chrome works well — see prompt template in commit history)

# 4. Pull verified annotations back
python -X utf8 import_from_label_studio.py "<project>" \
    [--ls-project-name "HVAC Review — Short Name"]

# 5. Copy ground-truth files into the tracked dir
cp benchmark_output/<project>/ls_*.{txt,json,csv} ground_truth/<project>/
git add ground_truth/<project>/ && git commit -m "Add ground truth for <project>"
```
