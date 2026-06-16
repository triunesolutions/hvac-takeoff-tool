---
name: project_schedule_ocr_budget_fix
description: "Schedule-OCR budget-starvation fixed 2026-06-08; static 40/60 split still wastes detection's idle time"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

2026-06-08: Fixed the schedule-OCR fallback starving detection to zero (commit `06ba375`). The raster-schedule EasyOCR fallback could eat the whole run budget in one uninterruptible page call → YOLO got 0s → no takeoff even on tiny plans (Cy-Fair 0.5 MB was 0 detections / no xlsx). Two fixes in `schedule_parser.py`: (1) OCR fallback gets only the *remaining* schedule budget (shared `_sched_t0`), skips if <15s left; (2) downscale each rendered page to 2600px long-side before OCR (a 36" sheet at 150 DPI is ~5400px → minutes/page on CPU). After: Cy-Fair runs end-to-end, 18 detections, 5 tagged, valid xlsx. `write_excel` path now live-verified.

**Remaining lever (not done):** the run budget is split STATICALLY ~40% schedule / ~60% detection. On Cy-Fair detection used only 18s of its slice while schedule-OCR was capped at ~90s and stopped at page 2/4, leaving schedule tags unrecovered despite tons of idle budget. Next improvement = let schedule-OCR borrow detection's unused time, or run OCR *after* detection with leftover budget. This is the targeted piece of the broader "streaming pdfplumber" todo. AD-GRD untagged count on Cy-Fair (10) still reflects the AD-GRD taxonomy gap. See [[project_v10_newcorpus_benchmark]].
