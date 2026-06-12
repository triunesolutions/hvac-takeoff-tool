# Session State / Handoff — 2026-06-12

**This laptop is being wiped after this session.** This repo (pushed to
`triunesolutions/hvac-takeoff-tool`) is the source of truth. Everything below
is what survives and where to find it. Written for whoever (incl. a fresh
Claude session) picks this up on a new machine.

## What survives the wipe (and where)

| Asset | Location | Status |
|---|---|---|
| All source code + pipeline | git (`master`) | ✅ safe |
| Trained YOLO models (`models/hvac_*.pt`, v9/v10/tag_detector) | git-tracked | ✅ safe |
| `ground_truth.jsonl` (26,844 recs), `tag_bubble_labels.jsonl` | git-tracked | ✅ safe |
| Strategy + benchmark + this session's docs | `docs/` (git) | ✅ safe |
| Datasets (`tag_dataset.zip`, `yolo_dataset*.zip`, v10/v11 parts) | GH release `datasets-2026-05-11` | ✅ safe |
| Claude memory (cross-session context) | `docs/memory_snapshot_2026-06-12/` | ✅ snapshotted into repo |
| WS1 batch numbers (157 projects) | `docs/batch_ws1_results_2026-06-12.csv` + `..._report_2026-06-12.md` | ✅ preserved |
| `images/` (595 MB raw training imgs), `labels.jsonl` (11 MB) | **see Open Items** | ⚠️ verify before wipe |

## Current verdict (the important part)

**Tag inference is the accuracy gate; within it, the SCHEDULE PARSER (WS2) is
the dominant bottleneck** — not class families, not (primarily) OCR. Full-batch
disaggregation (157/157, 16.4% tagged) in
`docs/tag_inference_disaggregation_2026-06-12.md`:

- WS2 schedule parser — **56.4%** of untagged mass (priority 1)
- OCR / distance (WS1.3/1.4) — **31.0%** (priority 2)
- TRUE class-gap (WS1.2/WS2.4) — **12.7%** (priority 3)

This **reverses** the original priority in
`docs/architecture_strategy_2026-06-10.md` (which led with class families).
The interim 58/157 read was wrong; see the disaggregation doc for why.

## Where the pipeline stands

- Production model: `models/hvac_yolov8s_v10.pt` (`DEFAULT_MODEL` in
  `takeoff_cli.py`). Do NOT retrain — v11 already regressed and was rolled back
  (criteria for a v12 in `docs/architecture_strategy_2026-06-10.md` §2).
- WS1 shipped to master: instrumentation, conservative class-families,
  `--bubble-max-dist` flag, bubble-OCR upscale+Otsu preprocessing.
- Run a takeoff: `python -X utf8 takeoff_cli.py "<plan.pdf>"` → xlsx + annotated
  PDF + json. Regression bar: `python benchmark_samples.py` (Flex 230 must hold).
- Batch runner: `scripts/batch_run_all.py` (one process at a time, resumable via
  `_done.json`, writes CSV + report after every project).

## Open items / next session TODO

1. **WS2 — schedule parser (the lever).** Attack zero-parse projects (43 with
   sched=0: Roots Church, Residence Inn El Paso, Aaron Packaging, Texas
   Roadhouse Bastrop, Erewhon, Total Wine #1149) and sparse-parse (Yucaipa
   BldgA sched=4/det=620, Rice Moody, Air Products).
2. **WS2.4 bug** — `_infer_yolo_class_from_service` lists `'EXHAUST'` in its
   generic-diffuser keyword set, so fan rows (EF/KEF) get filed under `AD-GRD`
   and no FAN/EXHAUST FAN detection can match. Poster child: Aritzia Americana
   (sched=50, 0 tagged). Prefer the tag-prefix's specific class over the generic
   AD-GRD fallback. Core class-inference change — validate on benchmark_samples
   + batch first.
3. **Schedule-OCR cap leak (#1 runtime debt)** — the `--time-budget` does not
   bound the EasyOCR fallback render loop, so small plans burn the whole budget
   on schedule-OCR before detection runs. Still unfixed.
4. **⚠️ Verify `images/` + `labels.jsonl` are preserved** before wiping — they
   are untracked and not in a release as-is. They may be redundant with
   `yolo_dataset.zip` in `datasets-2026-05-11`; confirm, or upload them to a new
   release. (See the closing note in this session's chat.)

## Conventions (carry forward)

- Blunt/honest feedback; push back when wrong; don't sugarcoat accuracy.
- Commit + push to `triunesolutions/hvac-takeoff-tool` regularly; direct-to-master.
- Dev/test on small PDFs (≤15–20 MB) until accuracy is solid.
- Heavy artifacts (`batch_*/`, `images/`, `labels.jsonl`) stay untracked/gitignored.
- Run heavy Python processes ONE AT A TIME (PC lags on parallel loads).
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
