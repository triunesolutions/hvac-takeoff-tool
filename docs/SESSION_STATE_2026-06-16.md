# Session State / Handoff — 2026-06-16 (PC formatted this day)

**This laptop is being reformatted 2026-06-16.** The repo (pushed to
`triunesolutions/hvac-takeoff-tool`) is the source of truth. Supersedes the
2026-06-12 handoff for the latest work. Read this first on the new machine.

## What survives the format (and where)

| Asset | Location | Status |
|---|---|---|
| All source code (incl. `ocr_table_extractor.py`) | git `master` | ✅ |
| Models `models/hvac_*.pt`, `ground_truth.jsonl`, `tag_bubble_labels.jsonl` | git-tracked | ✅ |
| Datasets `tag_dataset.zip`, `yolo_dataset*.zip` | GH release `datasets-2026-05-11` | ✅ |
| `images/` (595MB) + `labels.jsonl` | GH release `datasets-2026-06-16` | ✅ |
| Claude memory (cross-session context) | `docs/memory_snapshot_2026-06-16/` | ✅ snapshotted |
| **Investor-demo outputs + source rars** | `Downloads/Triune/investor_demo` (OUTSIDE repo) | ⚠️ **WILL BE WIPED** — regenerate from the team's rars; outputs are reproducible from code |

## This session's work — OCR table extraction for vector/CAD schedules

For two investor-demo projects (Cityvet Verrado, CCV Buckeye) whose schedule
tables are vector line-art (no text layer), built and shipped:

- **`ocr_table_extractor.py`** — reconstructs full schedule tables from vector
  sheets (grid detection → OCR-into-cells → MARK-column re-OCR). Auto-fires in
  `parse_pdf_schedules` when the text layer yields < 3 tags. Emits TagVariables
  with populated properties, so the Excel writer fills brand/model/CFM unchanged.
- **Air-device class inference** — Diffuser/Grille/Register marks now map into the
  AD-GRD family (device type lives in DESCRIPTION, tag in SYMBOL).
- **`takeoff_cli` detail columns** — MODULE/DUCT SIZE populated, TYPE composed to
  the team's "SUPPLY DIFFUSER" style, MATERIAL/DAMPER → remark.
- **Level-2b bubble OCR** — batched + tight 90px/2× crop, ~8× faster, 180s cap.
- Env var `HVAC_OCR_SCHED_PAGES="8,9,10"` targets known schedule pages (skips the
  slow blind full-document OCR sweep).

Commits: `a02d8c4`, `7544d78`, `dbd85b8` (+ this doc).

## Honest capability boundary (proven this session — do not re-litigate)

**Works:** vector-schedule reconstruction (mark + brand/model/type/CFM/material on
sheets with zero text layer) and full equipment detection + counts. This is the
demo headline.

**Does NOT work reliably:** assigning each detected diffuser to its exact mark
(A/B/C/D) or reading per-symbol neck sizes.
- Cityvet labels diffusers with a **pentagon letter** + size + CFM, but EasyOCR
  can't read the single letter inside the pentagon (5/15, mostly wrong).
- CCV diffusers carry **no mark letter** on the RCP at all.
- Per-diffuser size reading: ~15% recall, some misread.
- The old "Cityvet 87% tagged" was an artifact — blind direct-mapping put all 81
  diffusers under "A" (team's real split is A=26/B=6/C=19/D=11). Correct class
  inference exposed this (dropped to 0% without a readable per-mark signal).
- **Root cause:** small isolated text on dense CAD linework = low OCR recall.
  **Real fix = model investment** (train YOLO to distinguish diffuser sub-types
  and/or detect+read pentagon tag-bubbles). Weeks of work — post-demo.

## Demo plan (agreed 2026-06-16)

Lead with schedule reconstruction + detection counts; present per-mark assignment
as an "assisted takeoff the estimator confirms." Do NOT show the 87%-all-A number
or 15%-accurate sizes — the HVAC team/investors would catch them.

## Conventions (carry forward)

Blunt/honest feedback; push back when wrong. Commit+push to
`triunesolutions/hvac-takeoff-tool` regularly (direct-to-master). Dev/test on
small PDFs (≤15–20 MB). Heavy artifacts (`batch_*/`, `images/`, `labels.jsonl`,
`investor_demo/`) stay untracked. Run heavy Python ONE AT A TIME. Commit messages
end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
