# HVAC AI Takeoff Tool — A Plain-English Guide

*For non-technical readers. Last updated: May 5, 2026.*

> **Latest progress (May 5):** Trained **v10** of the equipment-detection model — median full-recall jumped from 22% → 44% on our scored sample projects. Built a **Label Studio review loop** so we (and Claude in Chrome) can verify every box and feed corrections back as training data. Reviewed 6 projects end-to-end: 475 boxes confirmed correct, 89 mislabels caught (mostly AD-GRD that should be AD-T-BAR SUPPLY), 61 phantom boxes deleted (almost all on legend/schedule/details sheets). Ground truth saved under `ground_truth/` and is now ready to feed v11 retraining.
>
> **Earlier (April 27):** Trained a second AI — a "tag-bubble detector" — that lets us read tag labels (`A1`, `CU-1`) more reliably. The CLI also prints project info (Name, Number, Sheet Title, Firm, Address, Date) at the top of every run.

---

## What We're Building

A tool that reads HVAC blueprint PDFs and produces a takeoff (a list of all equipment with counts) — the same thing your team currently does manually in Bluebeam, but in seconds instead of hours.

**Goal:** Internal tool first. Public SaaS product later.

---

## How It Works (Simple Version)

Three separate jobs run together:

**Job 1 — Read the schedule tables.**
Every blueprint has schedule tables (like a spec sheet) listing every piece of equipment: its tag, manufacturer, model, size, CFM, etc. We parse every table on every page and turn each row into a "variable" — a structured record with the tag and every property from that row.

**Job 2 — Spot equipment on the drawing.**
We use a visual AI (YOLO) that's been trained on labeled blueprints to look at each floor plan and draw a box around every piece of equipment it sees — diffusers, grilles, condensing units, fans, dampers.

**Job 3 — Match boxes to tags.**
For each box the AI drew, we figure out which schedule variable it corresponds to. We use three strategies in order:

1. **Direct match** — if there's only one CU tag in the schedule, every CU-shaped box gets that tag.
2. **Property match** — read the text near each box (CFM, model number). If it matches a tag's distinctive properties, we know that's the tag.
3. **Bubble OCR** — read the little label bubble next to each symbol (like "CU-1") using OCR, matched against the valid tag list for that equipment type.

The result is a filled Excel takeoff in your team's exact format + an annotated PDF with colored boxes showing what was detected.

---

## The Three Outputs

Every time you run the tool on a PDF, you get a folder with:

1. **`{project}_takeoff.xlsx`** — Excel in your team's format (PRODUCT, BRAND, MODEL, QTY, TAG, NECK SIZE, MODULE SIZE, DUCT SIZE, TYPE, MOUNTING, REMARK).
2. **`{project}_annotated.pdf`** — Your original PDF with colored boxes drawn around every detected symbol.
3. **`{project}_variables.json`** — A machine-readable file listing every schedule variable (for verification, debugging, or feeding into other tools).

---

## The Words People Throw Around

| Term | What it means in plain English |
|---|---|
| **Model** | The "brain" of the system. It's just a file that knows how to spot HVAC equipment. We're on version 10 (v10), with v11 in prep. |
| **Training** | Teaching the brain. Takes ~45 minutes on a Kaggle/Colab GPU. |
| **Inference** | Asking the brain to do a job. Takes ~20 seconds per page. |
| **Annotation** | When your team draws a box around a diffuser in Bluebeam, that's an annotation. |
| **Schedule** | The spec table on a drawing listing every piece of equipment and its properties. |
| **Variable** | Our term for one tag + all the properties from its schedule row. E.g., "CU-1" variable = {MANUFACTURER: CARRIER, MODEL: 40RUQA12, CFM: 3650, ...}. |
| **Tag** | The equipment identifier on the drawing — like "CU-1", "A1", "EF-3". |
| **YOLO** | The specific AI technique we use — it does both detection (where) and classification (what type) in one pass. |
| **OCR** | Optical Character Recognition — reading text from an image (like reading a tag bubble next to a symbol). |
| **Class** | A type of equipment (e.g., "T-bar supply diffuser" or "condensing unit"). |
| **Confidence** | How sure the computer is about a detection (0-100%). We typically accept anything above 40%. |

### The Three Numbers That Matter

**Recall = "Did we find everything?"**
- If a drawing has 100 diffusers and the model finds 79, that's **79% recall**.
- Currently: ~**79% position recall** (we find 4 out of 5 pieces of equipment).

**Precision = "Are our answers correct?"**
- If the model says it found 100 diffusers and 88 actually are, that's **88% precision**.
- Currently: ~**88% precision** (about 9 out of 10 answers are correct).

**Tagging rate = "Did we label each box with the right tag?"**
- Out of all detected boxes, what percent got matched to a specific schedule tag?
- This varies by project style: Flex 230 = 90%, United = 58%, others untested.

---

## Where We Are Right Now (May 5, 2026)

| What | Status |
|---|---|
| Model finds equipment positions | ✅ **79%** — works |
| Model labels equipment correctly | ✅ **88%** precision on things it finds |
| Parse schedule tables | ✅ Works on 3 tested styles (Flex, Aritzia, United) |
| Extract schedule variables with all properties | ✅ **Done** — every row → one variable with every column preserved |
| Match detections to schedule tags | 🟡 **Variable** — 90% on simple projects, 58% on complex multi-tag projects |
| Output annotated PDF | ✅ Works |
| Output Excel in team's format | ✅ Works |
| Output JSON sidecar for verification | ✅ Works |
| Has a user interface | ⏳ Not yet |
| Has review/correction workflow | ✅ Label Studio loop — Claude-in-Chrome reviews each box, corrections saved to `ground_truth/` |
| Has been tested on all common project styles | 🟡 Partial — Flex, Aritzia, United, more pending |

### The Honest Score

For a typical project the tool now does the following:
- Finds about **4 out of 5** pieces of equipment in the drawing
- Labels about **9 out of 10** of those correctly with equipment type
- Extracts **100% of schedule variables** on supported schedule styles
- Matches detections to specific tags somewhere between **58% and 90%** of the time depending on project style

**We are at "genuinely useful internal helper" quality, not yet production.** A team member using this tool today gets most of the grunt work done automatically but still needs to review and fill in the gaps.

---

## Why Some Projects Work Better Than Others

Three variables drive accuracy:

**1. Drawing style.** Different engineering firms draw HVAC symbols differently. We've trained on Flex/Plum, Haldeman, and Larson/Micah styles. Styles we haven't seen yet (like French Beaconsfield) will have lower accuracy until we add training data.

**2. Schedule layout.** Schedules vary wildly:
- **Vertical simple** (Flex 230): one table, one tag column, clean rows. Extraction is perfect.
- **Horizontal stacked** (Aritzia): properties listed down column 0, tags as column headers. We auto-transpose.
- **Multi-section combined** (Aritzia AHU+CU): multiple sub-schedules in one table. Sometimes fragments.

**3. How tags appear on the drawing.**
- **Single-tag per class** (Flex A/B/C/D): trivial to match. 90% tagging.
- **Multi-tag per class with visible bubbles** (United CU-1..6): bubble OCR works well. 58% tagging.
- **No tag bubbles, just tiny text**: hard; needs better OCR or trained per-tag model.

---

## What's New Since April 8

### Schedule parsing got a lot smarter
- Now handles horizontal tables (auto-transpose)
- Handles multi-tag cells: `"A, B, C"`, `"CU-1 thru CU-6"`, `"24\nVAV\n27"`, `"AC-1,2"`
- Strips equipment-status prefixes `(E)`, `(R)`, `(N)`
- Filters out drawing sheet numbers (`M102`, `E301`) and refrigerant codes (`R-454B`, `R-32`)
- Works on schedules where the tag column is labeled "TYPE" instead of "TAG" (AIR DEVICE SCHEDULE style)

### Variable extraction added
- Every schedule row is now one "variable" with ALL its columns preserved
- Inferred equipment type attached to each variable
- Written to `variables.json` next to every takeoff for verification
- `--verify` flag prints human-readable dump to terminal

### Tag matching went from naive to 3-level
- Level 1: direct class → single tag assignment (works for Flex)
- Level 2a: fingerprint match using PDF text layer
- Level 2b: schedule-guided bubble OCR (the big win for complex projects)
- United went from **3% tagged to 58% tagged** after Level 2b bubble OCR

---

## What's Next

### Immediate (this week / next)
- **Class aliasing** — YOLO sometimes detects "SPLIT SYSTEM" when the schedule has "CONDENSING UNIT". Need a map so they count together.
- **Crop preprocessing for OCR** — upscale + binarize crops before EasyOCR for better bubble reading.
- **Test on 3-5 more project styles** (SmithGroup, French Beaconsfield, SouthVAC files).

### Short term (2-4 weeks)
- More training data + retrain YOLO on broader style mix
- Handle the "big PDF crashes pdfplumber" case (streaming parser)
- Start on a review UI so the team can correct mistakes → feed into training

### Medium term (1-2 months)
- Simple web tool the team uses daily (Phase 2)
- Human-in-the-loop correction loop (every fix = training data)
- Target: 90%+ accuracy across all common drawing styles

### Long term (3-6 months)
- Public SaaS launch
- Plumbing and electrical takeoffs

---

## How To Talk About This in One Sentence

> "We're building an AI that reads HVAC blueprints and produces equipment takeoffs in seconds. With v10 we find 4 out of 5 pieces of equipment, ~89% of the boxes are placed correctly, and we now have a human-in-the-loop review tool feeding corrections directly back into the next training run."

---

## How We Compare to Rebar

Rebar (withrebar.ai) is the AI HVAC takeoff company that raised $14M and bootstrapped from Triune's files. Same fundamental approach (visual pattern recognition on labeled blueprints). They have:
- More training data (millions vs our ~25K tiles)
- Production GPUs
- A polished web app

We have:
- Real domain expertise (in-house takeoff team)
- A data flywheel from the team's daily Bluebeam corrections
- Control of the pipeline (we can tune schedule parsing for our specific project styles)

We're several engineering months behind on the product side but only a labeled-data gap behind on accuracy.

---

## Glossary of Files in the Repo

| File | What it does |
|---|---|
| `takeoff_cli.py` | The main command — feed it a PDF, get Excel + annotated PDF + variables JSON |
| `schedule_parser.py` | Reads every schedule table in the PDF and builds variables |
| `tag_inference.py` | Three-level system for matching AI detections to schedule tags |
| `tag_matcher.py` | OCR helpers for reading tag bubbles next to detections |
| `train_yolo.py` | Teaches a new model from your team's labeled projects |
| `benchmark.py` | Tests how accurate the model is on real projects |
| `class_aliases.py` | Fixes typos and merges duplicate equipment names in training data |
| `colab_train.ipynb` | The notebook we run on Google Colab to train (free GPU) |
| `models/hvac_yolov8s_v10.pt` | The current best model — the "brain" |
| `export_to_label_studio.py` | Push a project's detections into Label Studio for human/Chrome review |
| `import_from_label_studio.py` | Pull verified annotations back, write `ground_truth/` files |
| `ground_truth/` | Verified bbox + class data per project — feeds v11 retraining |
| `data to train/projects/` | All ~130 labeled projects (not in the repo, lives on JFL's machine) |
| `PRD.md` | The full product roadmap |
| `CLAUDE.md` | Technical context for engineers |
| `WHAT_WE_ARE_BUILDING.md` | This document |
