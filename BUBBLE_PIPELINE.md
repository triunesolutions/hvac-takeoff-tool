# Bubble-Only Pipeline — Experimental Branch

**Branch:** `bubble-pipeline`
**Started:** 2026-05-14
**Origin:** copied from `hvac-takeoff-tool` master at `d483846`
**Goal:** Remove the YOLO symbol detection loop. Detect tags only — via the
existing tag-bubble detector and (eventually) full-page OCR + schedule match.

---

## 1. Architectural change vs. master

| Step | master (`hvac-takeoff-tool`) | this branch (`hvac-takeoff-bubble`) |
|---|---|---|
| Schedule parse | pdfplumber → TagVariables | same |
| Plan inference | YOLOv8s_v10 (symbols) + 3-level inference (direct → fingerprint → bubble OCR) | tag_bubble detector + EasyOCR crop → strict schedule match |
| Tag → count | per-equipment-class count from symbol detections | direct tag count from OCR'd bubble text |

The symbol-detection model (`hvac_yolov8s_v10.pt`) is no longer loaded in
`takeoff_cli.py`. Only `hvac_tag_detector_v1.pt` (bubble detector) is used.

**Why we did this:** symbol detection was carrying compounding error — wrong
class, wrong position, wrong tag — and we were trying to fix every layer.
Going tag-direct gives us a count we can verify against the team's Bluebeam
takeoff one tag at a time.

---

## 2. Strict mode (the win)

Earlier `takeoff_cli.py` accepted any bare-prefix OCR read as a tag (`VAV` →
`VAV-?`). That introduced lump-counted noise like "VAV: 4" on Flex 230.

The current `resolve_bubble_text()` (in `takeoff_cli.py`) is strict:

```python
def resolve_bubble_text(bubble_text, schedule_tag_lookup, schedule_class_lookup):
    n = _normalize_for_match(bubble_text)
    if not n or n not in schedule_tag_lookup:
        return (None, None)
    canonical = schedule_tag_lookup[n]
    cls = schedule_class_lookup.get(n) or _class_from_prefix(_split_prefix(canonical)[0])
    return (canonical, cls or 'UNKNOWN')
```

Only tags present in the parsed schedule are counted. Bubble text that
doesn't match drops to None.

**Results (strict mode):**

| Project | Bubble pipeline count | Notes |
|---|---:|---|
| Acushnet | 174 clean | SD-1:21, SD-2:7, SD-3:30, SD-4:37, EG-1..6, RG-1..4, TA-1..3, FCU subset. Was 285 with prefix-fallback (60% garbage). |
| Capital One Minden | 164 | 9 mech pages |
| Anaheim, Free People | 0 resolved | **Not a bubble problem — pdfplumber returned 0 schedule tags.** Bubble detection + OCR found the right strings; strict mode dropped them because nothing in the schedule to match against. |

The bubble + OCR + strict-match pipeline works. **The bottleneck is the
schedule parser.** Without a tag list, strict mode has nothing to compare to.

---

## 3. Project-style survey (1–30)

`survey_projects.py` was run in parallel batches (`xargs -P 4 --max-mb 30`) over
projects 1–30 in `data to train/projects/`. Results in `survey_*.jsonl`.

| Style | Count | Examples | Bubble-pipeline status |
|---|---:|---|---|
| A — EAG template clones (VAV-heavy) | 4 | Flex 200/210/220/230 | works |
| B — Dense multi-schedule | 6 | Capitol Complex (66/75/118), STEM (189/17/23), Aritzia (70/24/36), Mission Bay (41/125/176), ARE Campus Point (17/60/72), Yucaipa B (15/34/36) | works |
| C — Single-sheet, no schedule in PDF | 1 | Fort Totten | needs companion schedule sheet |
| D — Oversized (>30 MB) | 8 | Dickinson, Acadiana, Gonzales, Grand Calilou, Sheraton, St Elizabeth, St Wenteslaus, Mygrant | need page-range / higher cap |
| E — pdfplumber hang | 1 | Basin Electric | needs timeout wrapper |
| F — Spec book, no equipment | 1 | LUS Admin (336 pp) | wrong PDF |
| G — Empty raw/ | 6 | Burlington, Hope Chapel, Perch, Optum, Planet Fitness, Vista Murrieta | re-source from team |
| H — Schedule parser misses (raster) | 3 | Larchmont, Shamrock, Yucaipa A | needs schedule OCR fallback |

**13 of 30 projects are bubble-pipeline-testable today.** The remaining 17
are blocked by: oversized PDFs (8), missing PDFs (6), raster schedules (3).

---

## 4. OCR engine A/B test — Yucaipa A page 4, 300 DPI

| Engine | Time | Word spans | Tag-shaped tokens | Real tags (after filter) | Read quality |
|---|---:|---:|---:|---:|---|
| EasyOCR | 72 s | 719 | 48 | 4 | DU-036, DU-091, DU-01**b**, EF-0 (case/digit errors) |
| RapidOCR (= PaddleOCR ONNX) | 145 s | 694 | 43 | 6 | DU-01A, DU-03E, IDU-05B, KEF-01 (clean) |
| Surya | n/a | — | — | — | Pillow build fails on Python 3.14 |

**Decision: use RapidOCR** for schedule pages. PaddleOCR's recognition model
under ONNXRuntime — works on Python 3.14, no `paddlepaddle` dependency
(`pip install rapidocr-onnxruntime`).

EasyOCR is faster but its case errors break strict matching. Paddle's clean
reads are worth the 2× time cost on the rare projects that need OCR fallback.

---

## 5. Two filter findings (also from Yucaipa A page 4)

### 5a. The `TAG_PREFIX_CLASS` whitelist was the real bottleneck

RapidOCR read **43 tag-shaped tokens** on page 4. My initial filter accepted
only those whose prefix was in `TAG_PREFIX_CLASS` (CU, EF, VAV, ...). It
rejected 37 of them because `DU`, `IDU`, `DBF`, `KEF` weren't in the static
whitelist.

**Fixed in `schedule_ocr_fallback.extract_tags_from_ocr`:** dropped the
prefix gate. Schedule pages should *discover* the project's tag universe.
False positives like sheet refs (`M101`) or room labels (`X171`) are kept here
— they'll be filtered downstream when plan-page bubble OCR matches against
this captured list. Strict at plan-match, permissive at schedule-parse.

### 5b. Yucaipa A doesn't *have* a schedule sheet — the tags are in the floor-plan text layer

Inspecting `page.get_text()`:

| Page | Text-layer hits |
|---|---|
| p4 floor plan | `DBF-01`, `EF-01`, `IDU-01A`..`IDU-10D` (~40 tags) |
| p5 floor plan | `EF-01`, `IDU-11A`..`IDU-20D` (~40 tags) |
| p6 floor plan | `EF-01`, `IDU-21A`..`IDU-30D` (~40 tags) |
| p7 roof plan | `ODU-01`..`ODU-29` |

The schedule survey returned 0 tags because pdfplumber's table extraction
couldn't find a structured schedule — but **the tags are right there in the
PDF text layer**, just scattered across the floor plans.

**This invalidates the "schedule-only ground truth" assumption for some
projects.** Some plans have no separate schedule; the plan IS the schedule.

---

## 6. Text-layer-first extractor (LANDED 2026-05-14)

`text_layer_tag_extractor.py` walks every page, regex-pulls tag-shaped tokens
from `page.get_text()`, filters out sheet refs (`M101`), code refs
(`CMC-303`, `CEC-150`, `T-24`), refrigerants, and single-letter+3-digit room
labels (`A101`). Returns TagVariable dicts (same shape as pdfplumber output).

Wired into `parse_pdf_schedules()` as an **augment** layered between
pdfplumber and OCR fallback. Existing variables (with property rows) are
never overwritten — text-layer only ADDS tags pdfplumber missed.

**Combined-parser comparison (post-text-layer):**

| Project | pdfplumber alone | + text_layer | Delta | Total time |
|---|---:|---:|---:|---:|
| Yucaipa A | 0 | **122** | +122 | 17 s |
| Aritzia | 36 | **64** | +28 | 90 s |
| Capitol Complex | 118 | **153** | +35 | 42 s |
| LUS Admin | 290 | **418** | +128 | 87 s |
| Flex 200 | 22 | **28** | +6 | 15 s |
| Larchmont | 0 | 0 | 0 (pure raster) | 44 s |

5/6 projects gain meaningfully from the text-layer path. Larchmont is the
only one still blocked — pure raster PDF, needs the OCR fallback (also
wired now, just behind a `len(variables) < 3` gate so it never duplicates).

### Discovery: Yucaipa A doesn't have a schedule sheet at all

`page.get_text()` reveals the real layout: tags are scattered directly on
the floor plans, not collected into a schedule table.

| Page | Text-layer hits |
|---|---|
| p4 floor plan | `DBF-01`, `EF-01`, `IDU-01A`..`IDU-10D` (~40 tags) |
| p5 floor plan | `EF-01`, `IDU-11A`..`IDU-20D` (~40 tags) |
| p6 floor plan | `EF-01`, `IDU-21A`..`IDU-30D` (~40 tags) |
| p7 roof plan | `ODU-01`..`ODU-29` |

The "schedule-only ground truth" assumption fails for projects like this.
Some plans have no separate schedule; the plan IS the schedule. Text-layer
extraction handles this case for free.

## 7. Recommended next step — sliding-window OCR scan on plan pages

```python
for each page:
    text = page.get_text()
    if text has tag-shaped tokens:
        emit them                 # free, instant
    elif text is empty:
        OCR the page (RapidOCR)   # slow, only when needed
    else:
        leave alone               # text exists but no tags → not a tag page
```

This handles:
- Style H (Larchmont, Shamrock) — empty text layer → OCR path.
- Yucaipa A (style A-ish but tags-on-plan) — text-layer path, zero OCR cost.
- Style A/B (Flex, Aritzia, etc.) — already work via pdfplumber, this just
  augments with floor-plan tags pdfplumber missed.

Filter universe to remove obvious non-tags:
- Sheet refs (`M101`, `P201`, `E301`) — single letter + 3 digits.
- Code refs (`CMC-303`, `CEC-150`) — known regulation prefixes.
- Refrigerants (`R-410A`, `R-32`) — already in `REFRIGERANT_PATTERN`.
- Lengths (`T-24`) — title 24 reference.

---

## 8. Files added on this branch

| File | Role |
|---|---|
| `survey_projects.py` | Single-project schedule/style profiler. Used with `xargs -P 4 --max-mb 30` for parallel batches. |
| `survey_1-10.jsonl`, `survey_11-20.jsonl`, `survey_21-30.jsonl` | Survey output, three batches. |
| `dump_bubbles.py` | Bypass strict mode, dump every OCR'd bubble to CSV. Diagnostic for projects where strict mode dropped everything. |
| `schedule_ocr_fallback.py` | RapidOCR-based fallback when pdfplumber returns <3 variables. Wired into `parse_pdf_schedules()` via the `ocr_fallback=True` kwarg. |
| `ocr_engine_benchmark.py` | A/B harness for OCR engines. Run with `--engines easy,paddle` to compare on any single page. |
| `text_layer_tag_extractor.py` | Text-layer-first tag scanner. `page.get_text()` regex + filters. Free, instant, zero OCR cost. Now layered into `parse_pdf_schedules()`. |
| `BUBBLE_PIPELINE.md` | This file. |

Modified:
| `schedule_parser.py` | Added `ocr_fallback` kwarg to `parse_pdf_schedules()`. |
| `takeoff_cli.py` | Removed symbol-YOLO loop; pipeline is bubble-detect → OCR → strict schedule match. |

Not committed (gitignored or noisy):
- `_smoke_*/` — per-project smoke-test output dirs (Excel + annotated PDFs).
- `survey_*.err`, `_*.log` — run logs.

---

## 8. Pending follow-ups (in priority order)

1. **Text-layer-first tag scanner** (section 6). Implementation ~1 hour. Unblocks Yucaipa A and likely several "0-tag" projects whose tags are in the text layer.
2. **Better candidate-page logic in `schedule_ocr_fallback.find_candidate_pages`** — currently misses pages that have text but no SCHEDULE keyword (and the tags are on those pages).
3. **Sliding-window OCR scan on plan pages** (user proposal). Once schedule tag list is solid, scan every plan tile, match each OCR token to the schedule, count by tag. This replaces the bubble detector entirely.
4. **Survey batches 4–N** (31+) — same `xargs -P 4 --max-mb 30` command.
5. **Re-source missing PDFs** for Burlington, Hope Chapel, Perch, Optum, Planet Fitness, Vista Murrieta — empty `raw/` dirs.
6. **pdfplumber timeout wrapper** so Basin Electric and similar don't hang the surveyor.
7. **Surya retry on Python 3.12 venv** — only if RapidOCR isn't enough.

---

## 9. How to test locally

```bash
# Schedule parse with OCR fallback
python takeoff_cli.py "<pdf>" --schedule-only

# OCR fallback alone (debugging)
python schedule_ocr_fallback.py "<pdf>" -v

# OCR engine A/B on a single page
python ocr_engine_benchmark.py "<pdf>" --page 4 --dpi 300 --engines easy,paddle

# Full bubble pipeline
python takeoff_cli.py "<pdf>"

# Diagnostic: dump every bubble bypassing strict mode
python dump_bubbles.py "<pdf>" --all-pages
```
