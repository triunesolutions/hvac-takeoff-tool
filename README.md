# HVAC Takeoff — Accuracy-Only Standalone Build

This is the **accuracy-focused** branch of the Triune HVAC takeoff tool. It merges
the accuracy-improving parts of Micah's fork into the research pipeline, with
**no SaaS web app, no Bluebeam stamping, and no training infrastructure** — the
single goal is the most accurate takeoff (Bill of Materials) from a blueprint PDF.

Pipeline: **schedule parse → mechanical-page filter → YOLO equipment detection
(v10) → tag inference → enrichment → Excel / PDF / JSON output.**

## What this build adds over `master`

Merged from Micah's fork (`MMicah-Git/triune`), accuracy code only:
- **Per-class detection thresholds** (`class_thresholds.py`)
- **Schedule-class post-filter** (`class_normalization.py`)
- **QA agreement gating** (`line_items.py`) — ship `confirmed`, surface the rest
- **Schedule↔detection reconciliation** (`validation_engine.py`)
- **OCR schedule fallback** (`schedule_ocr.py`) for vector/CAD schedule tables
- **Enrichment set** (neck-size waterfall, legend reader, data filler, quality
  checks, context enrich) wired in after detection
- **`sheet_filter.py` fix** — M-series sheet *number* overrides a noisy title-OCR
  marker, so real floor plans (e.g. Cityvet `M1.10`) are no longer dropped.

## Quick start

```bash
pip install -r requirements.txt
python takeoff_cli.py "path/to/blueprint.pdf" --output-dir out/
```

## Current status / next (2026-06-19)

- **Detection works:** Cityvet product_recall ≈ 86% (we locate the grilles).
- **Open bottleneck — schedule parsing.** `tag_recall` is still ~0% because the
  air-device schedule (the A/B/C/D tag table) does not parse: on Cityvet the
  parser returns **0 air-device rows** (only 3 junk rows scraped from a notes
  paragraph). Without the tag table there is nothing to correlate the detected
  symbols against, so per-tag counts can't be produced.
- **The lever, not pentagon OCR.** Reading the single tag-letter off the plan via
  OCR is a proven dead end (~15–20% ceiling on CAD stamps). The correct fix is to
  (1) extract the schedule table, then (2) match each symbol to a row via the
  **neck-size + CFM** printed beside it (readable digits), not the letter.
- **Tooling shortlist** for the table extractor is in
  `OPEN_SOURCE_TOOLING_RECOMMENDATIONS.md` (Camelot `stream`, PaddleOCR
  PP-Structure, img2table; PyMuPDF `get_drawings()` for exact vector counting).
  Paused here pending the laptop migration.

## Related Repositories

Part of the Triune HVAC takeoff effort. Full map for picking this up on a new machine:

| Repo / branch | What it is |
|---|---|
| [hvac-takeoff-tool `master`](https://github.com/triunesolutions/hvac-takeoff-tool) | **Canonical** research-grade pipeline: v10 YOLO, schedule parser, 3-level tag inference, benchmark harness. |
| [hvac-takeoff-tool `accuracy-standalone`](https://github.com/triunesolutions/hvac-takeoff-tool/tree/accuracy-standalone) | **(this branch)** Accuracy-only build — Micah's 4 accuracy layers + OCR schedule fallback + enrichment, **no** SaaS/Bluebeam/training. Active accuracy work lives here. |
| [MMicah-Git/triune](https://github.com/MMicah-Git/triune) | Micah's productized fork: SaaS web app + v14 model + Bluebeam pipeline. Separate account (read-only to us). |
| [mandeeps1nghh/triunebackup](https://github.com/mandeeps1nghh/triunebackup) | Automated daily snapshots of the working tree. Recovery point, not a dev line. |

**Canonical:** `triunesolutions/hvac-takeoff-tool` — `master` (research) + `accuracy-standalone` (accuracy build). Micah's fork is ahead on productization; this build takes only his accuracy code.
