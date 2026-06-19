# HVAC AI Takeoff Tool — Engineering Specification

**Audience:** An engineering team building this system from scratch (or extending the
current build). Read this end-to-end before writing code. It defines **what** we are
building, **why**, the **exact inputs and outputs**, every **processing stage**, the
**data model**, the **hard problems**, and the **acceptance criteria**.

**Owner:** Triune Solutions (HVAC industry; in-house takeoff team).
**Status of this document:** Authoritative product + engineering spec. Where the
current code already implements a stage, it is noted; where a stage is unsolved, it is
called out honestly as an open problem.

---

## 1. The one-paragraph summary

We receive a **commercial HVAC construction blueprint as a PDF**. A human estimator
today reads that PDF and hand-builds a **"takeoff"** — a spreadsheet (Bill of Materials)
that lists every piece of HVAC equipment (diffusers, grilles, fans, rooftop units, etc.),
how many of each, and each item's properties (size, CFM, model, manufacturer, mounting).
That takeoff is what the company prices and orders from. **We are automating the
estimator.** The software ingests the PDF and produces an Excel takeoff in the **same
format the team already uses**, as accurately as possible, so an estimator only has to
verify and correct rather than build from zero.

---

## 2. Domain background (so non-HVAC engineers can follow)

### 2.1 What a "takeoff" is
A *quantity takeoff* is the count of materials needed for a job. For HVAC, that means:
how many supply diffusers, return grilles, exhaust fans, rooftop units (RTUs),
condensing units (CUs), VAV boxes, dampers, louvers, etc., plus each item's spec.
The estimator produces this from the drawings; the company uses it to quote the job and
order equipment.

### 2.2 How HVAC equipment is represented on a drawing
Two linked things on every set of drawings:

1. **Symbols on the plan** — a small CAD glyph drawn on the floor/roof plan at the
   physical location of each device (a square with an X for a lay-in diffuser, a circle
   for a round diffuser, a rectangle with a fan symbol, etc.).

2. **A schedule (a table)** — a table elsewhere in the set that defines each **tag**.
   Example *Air Device Schedule*:

   | TAG | TYPE | NECK SIZE | CFM | MOUNTING | MODEL | MANUFACTURER |
   |-----|------|-----------|-----|----------|-------|--------------|
   | A | Lay-in supply | 10x6 | 255 | Ceiling T-bar | TMS | Titus |
   | B | Lay-in supply | 12x6 | 300 | Ceiling T-bar | TMS | Titus |
   | C | Return | 24x24 | 600 | Ceiling | 50F | Titus |
   | D | Sidewall supply | 8x6 | 150 | Wall | 300RL | Titus |

   The plan shows the **symbol**; next to it a **tag marker** prints the tag letter
   (e.g. "A" in a pentagon/circle) and usually the **neck size** ("10x6") and **CFM**
   ("255"). The schedule tells you what "A" *is*.

   **The fundamental link:** a symbol on the plan is meaningless until you read its tag
   and look that tag up in the schedule. **Counting the symbols gives you a total;
   reading the tags gives you the per-type breakdown the takeoff needs.**

### 2.3 Tag naming conventions you will encounter
- **Air devices (diffusers/grilles/registers):** single letters — `A`, `B`, `C`, `D`
  (sometimes `A1`, `B2`, or `S-1`/`R-1` for supply/return). These dominate the counts.
- **Major equipment:** prefix + number — `RTU-1`, `CU-2`, `AHU-1`, `FCU-3`, `EF-4`,
  `VAV-12`, `ERV-1`, `MAU-1`.
- **Dampers / accessories:** `FD` (fire), `FSD` (fire/smoke), `MVD`/`MD` (volume),
  `BD` (backdraft).
- **Louvers:** `L-1`, `LVR-1`.
- Drawing sheet numbers (`M1.10`, `E301`, `P201`) and refrigerant codes (`R-410A`) look
  like tags but are **NOT** tags — they must be rejected.

### 2.4 Sheet numbering convention (mechanical "M" series)
- `M0xx` — cover / general notes / legend
- `M1xx–M4xx` — **plans** (floor plans, roof plans) ← where symbols live
- `M5xx+` — details, schedules, specifications ← where schedule tables often live

This convention is the most reliable signal for classifying pages when the title text
can't be read (see §6).

---

## 3. The input: what kind of PDFs we get

This is the single biggest source of difficulty. **The same logical drawing arrives in
very different digital forms**, and the pipeline must detect which form it is dealing
with and adapt.

### 3.1 Three broad PDF types

| Type | What it is | Text layer? | Geometry | How to read it |
|---|---|---|---|---|
| **Vector / CAD export** | Exported straight from AutoCAD/Revit. Crisp at any zoom. | **Sometimes** — title-block and notes may be real text; **drawing labels are often vector paths, not text** | True vector paths (`get_drawings()`) | Text layer where present; rasterize + detect symbols; OCR only where there is no text |
| **Raster / scanned ("sticks")** | A scan or flattened image of a printed drawing. May be skewed, speckled, lower-DPI. | **No** (image only) | None — just pixels | Must rasterize and **OCR everything**; deskew first |
| **Hybrid** | Vector geometry with raster stamps, or a vector plan whose tag text is drawn as vector line-art (looks like text but isn't selectable) | Partial | Mixed | Treat per-region: text layer where it exists, OCR/vision where it doesn't |

**Critical, non-obvious fact:** even in a "nice" vector PDF, the **tag letters and
sizes printed on the plan are frequently CAD line-art, not selectable text.** You cannot
assume that because the PDF is vector you can extract the tags as text. This is why naive
text extraction fails and why we need rasterization + detection + OCR/vision.

### 3.2 File-size range and the working rule
- Real files range from **<1 MB** (e.g. a small retail vet clinic) to **4+ GB** (large
  hospital sets that crash naive PDF loaders).
- **Working rule for development:** build and tune on **small files (≤15–20 MB)** first.
  Get accuracy solid there before touching the giant files. Large files are a separate
  performance workstream (streaming/chunking), not an accuracy workstream.

### 3.3 Page composition of a typical set
A mechanical PDF (the "M" pages, sometimes bundled with the full architectural set) has:
- 1 cover / general-notes sheet
- 1–2 legend sheets (symbol key)
- **1–N mechanical plan sheets** (floor plans, roof plans) ← **symbols here**
- **1–N schedule sheets** (equipment + air-device tables) ← **tag definitions here**
- Several detail sheets (how to mount things — NOT counted)

Schedules are sometimes on **their own sheet**, and sometimes **embedded in the corner
of a plan sheet**. The pipeline must handle both.

### 3.4 Input contract (what the system accepts)
- A single PDF path (multi-page). Future: multiple PDFs per project (drawings + spec
  book) merged into one logical project.
- Optional CAD source (`.dwg`/`.dxf`) if a client ever provides it — out of scope today
  but the architecture should leave room (see §12 tooling note on `ezdxf`).

---

## 4. The output: the Excel takeoff

The output must match **the format the team already produces by hand**, because they
reuse the spreadsheet downstream. Do not invent a new format.

### 4.1 Primary sheet — `Triune Takeoff` (grouped BOM)
One row per (product type + tag), with quantity and properties:

| Column | Meaning | Example |
|---|---|---|
| PRODUCT | Equipment type / class | `Supply Diffuser`, `Exhaust Fan`, `RTU` |
| BRAND | Manufacturer | `Titus` |
| MODEL | Model number | `TMS` |
| QTY | Count of that tag on the plans | `26` |
| TAG | The tag string | `A` |
| NECK SIZE | Neck/inlet size | `10x6` |
| MODULE SIZE | Face/module size | `24x24` |
| DUCT SIZE | Connected duct size | `8"Ø` |
| TYPE | Device type/service | `Lay-in supply` |
| MOUNTING | How it mounts | `Ceiling T-bar` |
| REMARK | Notes | `Typical`, `See detail 3/M5.01` |

Rows are **grouped by product + tag** and counted. Column order, header text, and
grouping must stay **byte-identical** to the team's template (it's a hard guardrail).

### 4.2 Secondary sheet — `RawData` (audit trail)
One row per **individual detected symbol** (flat, ungrouped): class, tag, confidence,
page, bounding box, QA status. This is the diagnostic sheet — used to verify and debug,
and where per-detection QA color-coding lives. It does **not** need to match any team
template (it's ours).

### 4.3 Sidecar artifacts (machine-readable, written every run)
- `{stem}_variables.json` — the parsed schedule (every tag + all its properties). Single
  source of truth for downstream steps.
- `{stem}_detections.json` — every detection (class, tag, conf, page, box, qa_status).
- `{stem}_reconciliation.json` / `.txt` — schedule-vs-detected count check (§9).
- `{stem}_line_items.json` — per-item evidence + QA status (§9).
- `{stem}_annotated.pdf` — the plan with boxes + labels drawn on, for human review.

---

## 5. The pipeline, end to end

```
                ┌───────────────────────────────────────────────────────────┐
   PDF  ───────▶│ 0. INGEST & NORMALIZE                                      │
                │    detect vector vs raster; deskew scans; build page index │
                └───────────────┬───────────────────────────────────────────┘
                                ▼
                ┌───────────────────────────────────────────────────────────┐
                │ 1. PAGE CLASSIFICATION                                     │
                │    label each page: cover / legend / PLAN / SCHEDULE /     │
                │    details. Drives where each later stage runs.            │
                └───────┬───────────────────────────────┬───────────────────┘
                        ▼                                ▼
        ┌───────────────────────────┐     ┌─────────────────────────────────┐
        │ 2. SCHEDULE TABLE EXTRACT │     │  (PLAN pages held for stage 3)  │
        │    (SCHEDULE pages)       │     └─────────────────────────────────┘
        │    build tag → properties │
        │    table = "variables"    │
        └───────────┬───────────────┘
                    │ variables (the tag dictionary)
                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ 3. SYMBOL DETECTION (PLAN pages only)                              │
        │    rasterize plan, run object detector → equipment symbol boxes    │
        └───────────┬───────────────────────────────────────────────────────┘
                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ 4. TAG ASSOCIATION                                                │
        │    for each symbol, read the tag near it and match to a schedule  │
        │    tag (by letter, OR by neck-size+CFM fingerprint)               │
        └───────────┬───────────────────────────────────────────────────────┘
                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ 5. PROPERTY CAPTURE                                               │
        │    read neck size / CFM / type printed next to each symbol;       │
        │    fill remaining properties from the schedule row                │
        └───────────┬───────────────────────────────────────────────────────┘
                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ 6. COUNT & AGGREGATE  →  7. WRITE EXCEL + SIDECARS                │
        └───────────┬───────────────────────────────────────────────────────┘
                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ 8. RECONCILE & QA   (cross-check counts vs schedule; flag gaps)   │
        └───────────────────────────────────────────────────────────────────┘
```

The rest of this section specifies each stage.

---

### Stage 0 — Ingest & normalize

**Goal:** turn an arbitrary PDF into a clean, classified working set.

Requirements:
- Open the PDF; if a naive loader fails (huge files), fall back to streaming page-by-page.
- For each page determine: **does it have a usable text layer?** (`len(get_text("words"))`),
  **is it vector or raster?** (`len(get_drawings())` and image XObject inspection),
  page size, and **rotation** (CAD sheets are often rotated 90°/270° — must be corrected
  before any coordinate math or OCR).
- **Deskew** scanned/raster pages (small rotation correction) before OCR.
- Choose a working render DPI (default **200 DPI**; raise to 300 for small/dense text).
- Emit a **page index**: `[{page, type_hint, has_text, is_vector, rotation, size}]`.

Honest note: vector/raster detection is per-page, not per-file — a single PDF can mix
both. Decisions must be made per page.

---

### Stage 1 — Page classification

**Goal:** label every page so later stages run **only where they should**. Detecting
symbols on a schedule sheet, or trying to parse a table on a floor plan, both produce
garbage.

Classes: `cover`, `legend`, `plan` (floor/roof — has symbols), `schedule` (has the tag
tables), `details`, `notes`.

Signals, in priority order:
1. **Sheet number** (most reliable on CAD): the `M0xx/M1xx../M5xx+` convention (§2.4).
   Read it from the title block (text layer; OCR fallback).
2. **Title-block text** keywords: `MECHANICAL PLAN`, `ROOF PLAN`, `SCHEDULE`, `LEGEND`,
   `DETAILS`, `NOTES`.
3. **Page content**: a page dense with table grid lines + the words `SCHEDULE`/`CFM`/
   `MARK`/`TAG` is a schedule page even if titled oddly.

**Hard-won rule (already a fixed bug in the current build):** when the **sheet number**
says a page is a core plan (M1xx–M4xx) but a noisy **title-OCR** marker says otherwise,
**trust the number.** Title OCR on CAD is unreliable (e.g. a "RISER ROOM" room label was
misread as a non-plan marker and dropped the main floor plan). **Cost asymmetry:**
dropping a real plan loses *all* its equipment; keeping a borderline page only risks a
few phantoms that later QA catches. Bias toward keeping plans.

Output: each page tagged with a class; `plan` pages go to Stage 3, `schedule` pages go to
Stage 2. **Note:** a schedule can be embedded on a plan sheet — a page can be *both*.

---

### Stage 2 — Schedule table extraction (build the tag dictionary)

**Goal:** turn every schedule table into structured data: **one record per tag, holding
all of that tag's properties.** This is the dictionary the whole takeoff is built on.
We call each record a **`TagVariable`** (see §7).

This stage is the **#1 accuracy bottleneck** today (see §10). It must be built as a
**cascade of strategies**, strongest-cheapest first, because schedules arrive in wildly
different forms.

**Required capabilities:**
- **Ruled tables in the text layer** — the easy case. Extract cells directly; map header
  row → columns; each body row → a `TagVariable`.
- **Borderless / whitespace-aligned tables** — no grid lines; columns separated by
  spacing. Needs a stream/alignment-based extractor.
- **Vector-drawn tables with no text layer** — the grid and text are CAD line-art. Needs
  a **visual table-structure model** (reconstruct the cell grid from the image) plus OCR
  of each cell.
- **Pure raster/scanned tables** — OCR everything, then reconstruct rows/columns by
  coordinate clustering.
- **Layout variants the parser must handle:**
  - *Vertical* tables — tags down the rows (most common).
  - *Horizontal* tables — tags across the columns; must be **transposed** before parsing.
  - *Stacked multi-section* tables (e.g. a combined AHU+CU schedule) — split correctly so
    schedule names aren't lost.
  - *Multi-tag cells* — `CU-1,2,3` or `CU-1 thru CU-6` → expand to one record per tag.
  - *Status prefixes* — strip `(E)` existing / `(R)` relocated / `(N)` new.

**Header detection** (how to find which row is the header and which column is the tag):
1. A row containing a tag-column keyword (`MARK`, `TAG`, `UNIT TAG`, `TYPE`).
2. Fallback: a row with ≥3 property keywords (`TYPE`, `MODEL`, `SIZE`, `CFM`,
   `MANUFACTURER`, `MOUNTING`).
3. Fallback: column 0 holds tag-shaped values (`[A-Z]{1,4}-?\d{0,3}[A-Z]?`).

**Validation (reject non-tags):** drawing sheet numbers (`M102`), refrigerant codes
(`R-410A`), and banned words (`NOTES`, `REV`, `SHEET`, `ROUTING`) must be rejected by the
tag normalizer. **Garbage-in here poisons everything downstream**, so be strict.

**Class inference:** each `TagVariable` is assigned an equipment **class** at parse time
(e.g. tag `A` of type "lay-in supply" → `AD-T-BAR SUPPLY`; tag `CU-1` → `CONDENSING
UNIT`). Inference uses the row's service/mounting text first, then falls back to the tag
prefix. This class is what links a schedule row to a detected symbol class in Stage 4.

**Output:** `variables` — a list of `TagVariable`, written to `{stem}_variables.json`.
**Acceptance for this stage:** on a project whose schedule lists tags A/B/C/D + the major
equipment, `variables` contains those tags with non-null class and the right
neck/CFM/type — *not* rows scraped out of a notes paragraph.

> **Tooling guidance for this stage is in `OPEN_SOURCE_TOOLING_RECOMMENDATIONS.md`:**
> Camelot (`stream` mode) and img2table for borderless tables; PaddleOCR **PP-Structure**
> for vector/raster table-structure reconstruction; a document VLM (e.g. Qwen2.5-VL) as
> the last-resort fallback for schedules nothing else can segment. Validate every
> returned tag through the normalizer before trusting it.

---

### Stage 3 — Symbol detection (plan pages only)

**Goal:** find every equipment symbol on each plan page and put a box + class on it.

Requirements:
- Run **only on `plan` pages** (Stage 1). Never on schedules/legends/details — they are
  the #1 source of phantom detections.
- Rasterize the plan (200 DPI default), tile it (e.g. 640×640) so small symbols aren't
  lost, run the object detector, then merge tiles and **de-duplicate** overlapping boxes
  (non-max suppression) so one diffuser isn't counted twice.
- Output: per page, a list of `{class, confidence, box}`.

The current build uses a trained YOLO model (`hvac_yolov8s_v10.pt`, 33 classes). Detection
is **already the strong part** of the pipeline (≈86% of devices located on the test
project). Treat the detector as a component with a measurable recall/precision; it can be
swapped/retrained later (see tooling doc), but **detection is not the current bottleneck.**

Honest note: the detector under-detects some roof equipment (RTUs on roof plans) and can
confuse look-alike diffusers (supply vs return T-bar). Those are training-data gaps that
Stage 8 reconciliation surfaces, not silently hides.

---

### Stage 4 — Tag association (the crux)

**Goal:** for each detected symbol, determine **which schedule tag it is**, so the count
can be broken down per tag (26 of "A", 6 of "B", …) — not just "62 diffusers total".

This is the **hardest unsolved problem** and must be designed carefully. There are two
independent ways to identify a symbol's tag; **the system should use both and prefer
whichever is reliable per drawing:**

**Method A — read the tag marker directly.**
Near each symbol there is a **tag marker**: the tag letter inside a small shape
(commonly a **pentagon** or circle), e.g. "A". Read that glyph.
- **Honest limitation (proven):** reading a **single CAD-drawn letter inside a stamp** via
  OCR tops out around **15–20% reliability** on dense linework. Do **not** build the
  product on this alone. It is a *secondary* signal. (We spent real effort confirming this
  dead end; don't repeat it.)

**Method B — fingerprint by the printed neck-size + CFM (preferred).**
Next to almost every air-device symbol the drawing prints the **neck size** (e.g.
`10x6`) and **CFM** (e.g. `255`). These are **multi-character digit strings — far more
reliably read** than a single letter, whether from the text layer or by OCR. The schedule
(Stage 2) maps each tag to its neck size + CFM. So:

> read `10x6` / `255` beside a symbol  →  look it up in the schedule  →  that's tag **A**.

This turns an unreliable single-glyph read into a reliable multi-token lookup. **Method B
is the primary path; Method A is a tie-breaker / fallback.**

- **Feasibility caveat that must be checked per project:** Method B only uniquely
  resolves a tag if each tag has a **distinct** (neck size, CFM) combination. If two tags
  share the same size+CFM, the lookup is ambiguous and you fall back to Method A (the
  letter) or to spatial reasoning. The pipeline must detect this ambiguity and handle it,
  not silently mis-assign.

**Association levels (apply in order, each only tags still-untagged symbols):**
1. **Direct** — if a class has exactly **one** tag in the schedule, assign every symbol of
   that class to it (common in small jobs: one "A" diffuser type).
2. **Fingerprint (Method B)** — match printed neck/CFM near the symbol to a schedule row.
3. **Marker read (Method A)** — read the tag letter/number from the stamp; match to the
   class's valid tag list.
4. **Unresolved** — leave `tag = null`, mark for human review. **Never guess.**

**Where to read text near a symbol:** prefer the **text layer** within a small radius of
the symbol box (cheap, exact); fall back to OCR of a tight crop only where there is no
text layer. Always constrain matches to the **valid tags for that symbol's class** (don't
match an air-device symbol to an `RTU` tag).

---

### Stage 5 — Property capture

**Goal:** fill each line item's properties (neck size, module size, duct size, type,
mounting, model, brand, remark).

Two sources, merged:
1. **Read from the plan** what is printed beside the symbol — primarily **neck size** and
   **CFM** (already read in Stage 4 Method B). These are the most job-specific and should
   win when present, because a plan can override the schedule for a specific location.
2. **Inherit from the schedule row** for everything not printed on the plan (model,
   manufacturer, mounting, type, module size).

Property lookups from schedule rows must be **tolerant** — match a column by any keyword
it contains, case-insensitive, and handle combined headers (`MANUFACTURER & MODEL`,
`MAKE / MODEL`) by splitting.

---

### Stage 6 — Count & aggregate

**Goal:** produce the grouped BOM.
- Group detected, tagged symbols by **(class, tag)**; the group size is `QTY`.
- Attach the merged properties (Stage 5) to each group.
- Map internal class names to the team's **PRODUCT** wording (e.g. `AD-T-BAR SUPPLY` →
  `Supply Diffuser`). Maintain this mapping explicitly.
- Untagged symbols (Stage 4 level 4) are still counted at the **class** level (so the
  total isn't understated) but listed with a blank/`?` tag and flagged for review.

---

### Stage 7 — Output generation

Write the `Triune Takeoff` sheet (grouped, team format — guardrail) + `RawData` sheet
(flat, ours) + all JSON sidecars + the annotated PDF (§4). Excel formatting (column order,
headers, grouping) on the team sheet must match the template exactly.

---

### Stage 8 — Reconciliation & QA (don't ship a confident wrong answer)

**Goal:** cross-check the takeoff against the schedule and surface uncertainty, instead of
reporting a possibly-incomplete count as if it were complete.

- **Count reconciliation:** for unique-instance equipment (1 tag = 1 unit, e.g. RTUs,
  CUs), compare **scheduled count vs detected count** per class → verdict `match` /
  `under` (missed) / `over` (phantom) / `orphan`. Air devices repeat per tag, so they are
  presence-checked, not count-checked.
- **Agreement gating (per item):** each line item carries which signals agreed —
  `vision` (detector), `text` (a tag/size read off the plan), `schedule` (tag+class exist
  in the parsed schedule). `confirmed` = ≥2 agree (ship); `needs_review` = 1 signal;
  `flagged` = contradiction. Color the `RawData` rows green/yellow/red accordingly.
- **Honest reporting:** if the schedule didn't parse, say *"no schedule parsed —
  detections unverified"* rather than implying a clean takeoff. The whole point of this
  stage is to make the tool **trustworthy to an estimator** — it tells them exactly which
  rows to double-check.

---

## 6. Page-routing rules (summary table)

| Page class | Stage 2 (schedule) | Stage 3 (detect) | Notes |
|---|---|---|---|
| cover / notes | no | no | skip |
| legend | no | no | (legend reading is a separate optional enrichment) |
| **plan** (M1xx–M4xx) | only if a schedule is embedded | **yes** | symbols live here |
| **schedule** (M5xx+) | **yes** | no | tag tables live here |
| details | no | no | mounting details, not counted |

---

## 7. Data model

### 7.1 `TagVariable` (one per schedule tag) — the source of truth
```python
{
  "tag": "A",
  "schedule_name": "AIR DEVICE SCHEDULE",
  "page": 7,                       # 1-indexed page the schedule was on
  "properties": {                  # ENTIRE row, all columns, normalized keys
    "TYPE": "LAY-IN SUPPLY",
    "NECK SIZE": "10x6",
    "CFM": "255",
    "MOUNTING": "CEILING T-BAR",
    "MODEL": "TMS",
    "MANUFACTURER": "TITUS"
  },
  "inferred_yolo_class": "AD-T-BAR SUPPLY",   # links to detection class
  "source_row_index": 2
}
```

### 7.2 `Detection` (one per symbol found on a plan)
```python
{
  "page": 4, "cls": "AD-T-BAR SUPPLY", "conf": 0.86,
  "x1":…, "y1":…, "x2":…, "y2":…,
  "tag": "A",                    # filled by Stage 4 (null if unresolved)
  "tag_method": "fingerprint",   # direct | fingerprint | marker_ocr | none
  "tag_confidence": 0.0-1.0,
  "neck_size": "10x6", "cfm": "255",   # read from plan in Stage 5
  "qa_status": "confirmed"       # confirmed | needs_review | flagged
}
```

### 7.3 Line item (one per grouped BOM row) — what becomes an Excel row
`(PRODUCT, BRAND, MODEL, QTY, TAG, NECK SIZE, MODULE SIZE, DUCT SIZE, TYPE, MOUNTING,
REMARK)` + the evidence/QA fields backing it.

---

## 8. Accuracy: how we measure success

We do **not** claim accuracy without measuring it against the team's own takeoffs.

- **Ground truth** = the team's completed Excel takeoff for the same project (per-product
  and per-(product, tag) quantities).
- **Metrics:**
  - `product_recall  = Σ min(team_qty, our_qty) / Σ team_qty` — fraction of the team's
    items we caught (at the product level).
  - `product_precision = Σ min(team, ours) / Σ our_qty` — fraction of ours that were real.
  - `tag_recall` / `tag_precision` — same, but per **(product, tag)** — the harder bar,
    because it requires Stage 4 to have worked.
- **Regression harness:** a benchmark script runs the full pipeline over a folder of
  projects (each with `Plans_Specs/<pdf>` + `Completed Takeoff/<xlsx>`) and reports these
  per project. **Any pipeline change must be measured on this harness before merge** — a
  prior model "upgrade" silently regressed and had to be rolled back; measurement is
  non-negotiable.

**Current measured reality (test project "Cityvet", ground truth = 71 devices):**
- `product_recall ≈ 86%` (symbols are found well — Stage 3 works).
- `tag_recall ≈ 0%` (Stage 2 produced **0 air-device rows** → Stage 4 had nothing to
  match against). **This is the gap the spec above is designed to close, in this order:
  fix Stage 2 (schedule table), then Stage 4 Method B (neck/CFM fingerprint).**

---

## 9. The hard problems (read before estimating effort)

1. **Schedule parsing is the bottleneck (~56% of all failures).** Symbols are found; the
   *table that defines the tags* often isn't extracted. Everything downstream is starved
   when Stage 2 fails. **Invest here first.** (§2, §5-Stage2, tooling doc.)
2. **Single-letter tag OCR does not work** on CAD stamps (~15–20% ceiling). Build Stage 4
   on the **neck/CFM fingerprint (Method B)**, not the letter. Spending more effort on
   pentagon/circle letter OCR is a known dead end.
3. **Vector ≠ readable text.** A crisp vector PDF can still have its tags as line-art.
   Decide per-region whether to use the text layer, OCR, or a vision model.
4. **Per-page heterogeneity.** One PDF mixes vector and raster pages, rotated sheets,
   embedded vs standalone schedules. Decisions are per page, never per file.
5. **Large files.** 4 GB sets crash naive loaders. Deferred — solve accuracy on small
   files first; large-file streaming is a separate performance track.
6. **Never silently guess or silently cap.** An unresolved tag is flagged for review, not
   assigned a plausible-but-wrong value. A dropped page / truncated scan is logged, not
   hidden. The tool's value is that an estimator can *trust* the confident rows.

---

## 10. Recommended tech stack

- **PDF / render / text layer / vector paths:** PyMuPDF (`get_text`, `get_drawings`,
  rasterize).
- **Table extraction cascade:** pdfplumber (ruled) → Camelot `stream` (borderless) →
  img2table → PaddleOCR **PP-Structure** (visual structure) → document VLM (last resort).
  See `OPEN_SOURCE_TOOLING_RECOMMENDATIONS.md` for the ranked rationale + licenses.
- **OCR:** start with EasyOCR; evaluate PaddleOCR / docTR for small dense text (allowlist
  the tag alphabet). OCR is a *supporting* engine, not the answer to Stage 4.
- **Symbol detection:** Ultralytics YOLO (current v10); RT-DETR / newer YOLO as a measured
  upgrade only. Optionally **vector path-clustering via `get_drawings()`** to count
  byte-identical CAD glyphs exactly (strong for counting; see tooling doc §3).
- **Output:** openpyxl (Excel), Pillow/PyMuPDF (annotated PDF).
- **License note for the eventual SaaS:** PyMuPDF and Ultralytics are AGPL — fine
  internally, but resolve commercial licensing before a hosted product ships. Camelot/
  img2table (MIT) and PaddleOCR/docTR/VLM weights (Apache-2.0) are SaaS-friendly.

---

## 11. Definition of done (acceptance criteria)

A build is "working" when, on the small-file benchmark set:
1. **Page classification** correctly routes plans vs schedules (no real plan dropped; no
   schedule fed to the detector) on every benchmark project.
2. **Schedule extraction** produces a correct tag dictionary (right tags, right
   neck/CFM/type, valid class) for the air-device schedule **and** the major-equipment
   schedules — verified against the team xlsx, not just "non-empty".
3. **Detection** locates ≥ (target) % of devices (`product_recall`), measured.
4. **Tag association** moves **`tag_recall` materially off 0%** via Method B, with
   ambiguous (shared size+CFM) cases correctly flagged rather than mis-assigned.
5. **Output** opens in Excel in the team's exact format; `RawData` + sidecars present.
6. **Reconciliation/QA** flags every unverified/under/over case honestly.
7. **Every number above is produced by the benchmark harness**, with a before/after vs the
   prior build — no accuracy claim without a measured comparison.

---

## 12. Phasing (suggested delivery order)

1. **Stage 2 schedule cascade** (Camelot/PP-Structure) — unblocks everything. *Highest ROI.*
2. **Stage 4 Method B** (neck/CFM fingerprint association) on the now-parsed schedules.
3. **Stage 8 reconciliation/QA polish** so output is trustworthy.
4. **Raster/scanned ("sticks") path** — deskew + full-OCR ingest for non-vector files.
5. **Large-file streaming** — performance track for the 100 MB–GB sets.
6. **Detector upgrade / retrain** — only after 1–4, and only if it beats the current model
   on the harness.

---

## 13. Glossary

- **Takeoff / BOM** — the counted list of equipment + specs used to price/order a job.
- **Schedule** — a table on the drawings defining each tag's properties.
- **Tag / mark** — the label identifying a device type (`A`, `RTU-1`).
- **Air device (AD)** — diffuser/grille/register that moves air; tagged with letters.
- **Neck size** — the inlet/duct-connection size of an air device (e.g. `10x6`).
- **CFM** — cubic feet per minute; the airflow rating printed by each device.
- **RTU / CU / AHU / FCU / VAV / EF / ERV** — rooftop unit / condensing unit / air
  handler / fan-coil / variable-air-volume box / exhaust fan / energy-recovery ventilator.
- **Plan sheet** — a floor or roof plan (M1xx–M4xx); where symbols are drawn.
- **Vector PDF** — geometry stored as paths (crisp, scalable). **Raster PDF** — stored as
  pixels (a scan/image). **Hybrid** — a mix.
- **Phantom** — a false-positive detection. **Under/over** — fewer/more than scheduled.

---

*This document is the single source of truth for what to build. The companion
`OPEN_SOURCE_TOOLING_RECOMMENDATIONS.md` ranks the specific libraries per stage. The
current code in this branch implements Stages 0–3, 5–8 to varying degrees; Stage 2
(schedule table) and Stage 4 (tag association via Method B) are the active gaps.*
