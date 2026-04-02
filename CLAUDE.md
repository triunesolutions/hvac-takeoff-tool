# HVAC AI Takeoff Tool

## What This Project Is

An AI-powered tool that reads HVAC blueprint PDFs, detects equipment (diffusers, grilles, VAV boxes, dampers, etc.), and produces a structured Bill of Materials (takeoff). Built by Triune Solutions — an HVAC industry company with an in-house takeoff team.

**Goal:** Internal tool first (assist human estimators), then public SaaS product.

## Communication Style

Be blunt and honest. Push back when ideas are wrong. Suggest better approaches proactively. No sugarcoating. This is a serious long-term product — give real engineering counsel.

## Project Context

- Triune provided the original 50 blueprint files that Rebar (withrebar.ai) used to bootstrap their product. We're building our own competing solution.
- We have a human takeoff team that uses Bluebeam Revu + Excel. They are the domain experts AND the source of training data.
- The team is preparing Bluebeam markup CSV exports as ground truth data for accuracy calibration.
- Blueprints are primarily Quebec (French) HVAC projects, but must support English too.

## Architecture Decisions Made

1. **OCR-first, not vision-first.** Detecting text tags (D-1, GR-2, VM-05) via OCR is far more reliable than template matching symbol graphics. Visual symbol detection is Phase 3, not Phase 1.
2. **Template matching was tried and found insufficient.** Legend symbols don't match floor plan symbols well — too much surrounding clutter (text labels, ductwork, dimensions). See `detect.py` for the prototype.
3. **Bluebeam export data is the unlock.** Can't properly tune accuracy without ground truth. Waiting on team to provide CSV exports.
4. **Data flywheel from day 1.** Every project processed generates training data. Human corrections feed back into model improvement.

## Current State (as of April 2026)

### What Exists
- `PRD.md` — Full product requirements document with phased roadmap
- `takeoff.py` — Initial prototype (PDF → legend extraction → template matching). Partially working.
- `detect.py` — Refined detection script with per-symbol thresholds. Runs but accuracy is low (~30 detections found vs ~200 real).
- `templates/` — 37 named HVAC symbol templates extracted from legend page
- `symbols_auto/` — 78 auto-detected symbol candidates from legend
- `legend_crops/` — Cropped sections of legend page for reference
- `floor_crops/` — High-DPI crops of floor plan areas showing actual symbol appearance
- `output/` — Annotated PDFs and detection results from test runs

### What's Proven
- PyMuPDF renders blueprints well at 150-300 DPI
- Text extraction finds 203 equipment tag instances and 52 unique tags on the sample project
- Equipment tags follow predictable patterns: `D-\d+`, `GR-\d+`, `GE-\d+`, `GA-\d+`, `VM-\d+`, etc.
- Airflow values (L/s) and dimensions (WxH, ø) are extractable from page text
- Template matching works for some symbols but false positive rate is too high for production use

### What Failed
- Template matching from legend crops → floor plan detection: symbols on drawings are embedded in clutter (text, ductwork, dimensions) unlike clean legend graphics
- Generic/small templates (registre, grille murale retour) match ductwork lines everywhere
- Blind threshold tuning without ground truth data is guesswork

## Roadmap (from PRD.md)

- **Phase 1 (current):** OCR-based tag detection + Bluebeam ground truth parser + accuracy scoring
- **Phase 2:** Schedule parsing + human-in-the-loop correction UI
- **Phase 3:** Train custom YOLO/DETR object detection model on accumulated labeled data
- **Phase 4:** Public SaaS product
- **Phase 5:** Expand to plumbing and electrical

## Key Technical Details

### Equipment Tag Patterns (from real Beaconsfield project)
```
D-{n}   → Diffuser (supply)         | Read: airflow (L/s), neck size (ø)
GR-{n}  → Grille (return)           | Read: size (WxH), airflow
GE-{n}  → Grille (exhaust)          | Read: size, airflow
GA-{n}  → Grille (supply)           | Read: size, airflow
VM-{n}  → Volet motorisé (damper)   | Read: size
VAV-{n} → VAV box                   | Read: min/max airflow, inlet size
TA-{n}  → Transfer air              | Read: size
BV      → Fire/bypass damper        | Read: size, rating
SE      → Serpentin électrique       | Read: capacity
SC-{n}  → Serpentin chauffage        | Read: type, capacity
HUM-{n} → Humidifier                | Read: capacity
```

### Sample Project Stats (PLANS VENTILAITON.pdf — Beaconsfield)
- 15 pages, French, Revit-generated
- Page 2: Legend (V001)
- Pages 6-7: New construction floor plans (the money pages)
- Pages 12-13: Equipment schedules (the answer key)
- 203 tag instances, 52 unique tags, 135 airflow values, 226 dimensions

### Tech Stack
- Python 3.14, PyMuPDF (fitz), OpenCV, pandas, openpyxl
- Future: PaddleOCR/EasyOCR, PyTorch + Ultralytics (YOLOv8), FastAPI + React, Claude API

## File Organization

```
hvac-takeoff-tool/
├── CLAUDE.md          ← You are here
├── PRD.md             ← Full product requirements document
├── detect.py          ← Symbol detection via template matching (Phase 1 prototype)
├── takeoff.py         ← Initial pipeline prototype
├── templates/         ← 37 named HVAC symbol templates from legend
│   └── symbol_map.json
├── symbols_auto/      ← 78 auto-detected symbol crops
├── legend_crops/      ← Legend page analysis crops
├── floor_crops/       ← High-DPI floor plan crops for reference
└── output/            ← Detection results and annotated PDFs
```

## Next Steps (Priority Order)

1. **When Bluebeam CSV arrives:** Build parser, establish ground truth, measure current accuracy
2. **Build OCR tag detector:** Use PyMuPDF text extraction (already proven) + PaddleOCR for tags embedded in graphics
3. **Build value extractor:** For each detected tag, find nearby airflow (L/s) and dimension values using spatial proximity
4. **Build accuracy scorer:** Compare tool output vs. Bluebeam ground truth
5. **Iterate on accuracy** using real data, not guesswork

## Don't Do

- Don't over-engineer infrastructure before core detection works
- Don't build a web UI until Phase 2
- Don't train ML models until we have 50+ annotated projects (Phase 3)
- Don't try to generalize across all drawing styles yet — nail Triune's common engineering firms first
- Don't use Claude Vision API as primary detection (cost/latency at scale) — use it for spec book parsing and as a fallback
