# HVAC Takeoff Tool

End-to-end pipeline that reads commercial HVAC blueprint PDFs and produces an
Excel takeoff (Bill of Materials), an annotated PDF, and JSON sidecars.

Input: a multi-page blueprint PDF →
**schedule parse → mechanical-page filter → YOLO equipment detection → 3-level tag inference → Excel/PDF/JSON output.**

## Related Repositories

Part of the Triune HVAC takeoff effort. Full map for picking this up on a new machine:

| Repo / branch | What it is |
|---|---|
| [hvac-takeoff-tool `master`](https://github.com/triunesolutions/hvac-takeoff-tool) | **(this repo)** Canonical research-grade pipeline: v10 YOLO, schedule parser, 3-level tag inference, benchmark harness. |
| [hvac-takeoff-tool `accuracy-standalone`](https://github.com/triunesolutions/hvac-takeoff-tool/tree/accuracy-standalone) | Accuracy-only build — Micah's 4 accuracy layers + OCR schedule fallback + enrichment, **no** SaaS/Bluebeam/training. Active accuracy work lives here. |
| [MMicah-Git/triune](https://github.com/MMicah-Git/triune) | Micah's productized fork: SaaS web app + v14 model + Bluebeam pipeline. Separate account (read-only to us). |
| [mandeeps1nghh/triunebackup](https://github.com/mandeeps1nghh/triunebackup) | Automated daily snapshots of the working tree. Recovery point, not a dev line. |

**Canonical:** `triunesolutions/hvac-takeoff-tool` — `master` (research) + `accuracy-standalone` (accuracy build). Micah's fork is ahead on productization; the accuracy build takes only his accuracy code.

## Quick start

```bash
pip install -r requirements.txt

# Run a takeoff (outputs to <pdf_stem>_takeoff/)
python takeoff_cli.py "path/to/blueprint.pdf"

# Schedule-only sanity check (fast, no YOLO)
python takeoff_cli.py "path/to/blueprint.pdf" --schedule-only

# Score the pipeline against the team's truth xlsx
python benchmark_samples.py --layout projects --out benchmark_output
```

Models are committed under `models/` (`hvac_yolov8s_v10.pt` is the production
default; `--model` overrides it). Python 3.12+.

## Layout

| Path | What |
|---|---|
| `takeoff_cli.py` | CLI entry point — orchestrates the full pipeline |
| `schedule_parser.py` | Schedule-table extraction → `TagVariable` list |
| `tag_inference.py` | 3-level tag→detection matching (direct / fingerprint / bubble-OCR) |
| `tag_matcher.py` | EasyOCR helpers for the bubble-OCR level |
| `tag_extractor.py` | Detection-summary helper for the Excel/JSON output |
| `class_aliases.py` | YOLO class-name normalization (shared by the benchmark) |
| `benchmark_samples.py` | Regression benchmark — scores generated xlsx vs team truth |
| `models/` | Trained YOLO detectors |
| `templates/` | Legend symbol references |
| `scripts/` | Manual training / dataset-prep / eval / Label-Studio scripts (honor `HVAC_*` env vars for paths) |
| `notebooks/` | Colab/Kaggle training notebooks |
| `docs/` | Engineering notes and benchmark reports |
| `tests/` | Unit tests (`python tests/test_normalize_tag.py`) |
| `ground_truth/` | Label Studio review output per project (tracked; feeds v11 retraining) |
| `batch_datatrain_local/` | Full data-train batch report (heavy artifacts gitignored; `batch_report.md` kept) |
| `sample_*.csv` | Class-survey outputs produced by `scripts/survey_sample_classes.py` |

Dev/eval scripts default to JFL's local paths but accept env-var overrides on
other machines: `HVAC_PROJECTS_DIR`, `HVAC_SAMPLE_ROOT`, `HVAC_YOLO_DIR`,
`HVAC_OUTPUT_DIR`, `HVAC_BENCHMARK_OUT`.

## Reference

- **`CLAUDE.md`** — full engineering reference (pipeline internals, data model,
  parser rules, known limitations, what failed). Read it before changing code.
- **`PRD.md`** / **`docs/WHAT_WE_ARE_BUILDING.md`** — product vision and status.
