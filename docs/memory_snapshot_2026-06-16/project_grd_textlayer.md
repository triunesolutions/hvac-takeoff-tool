---
name: project_grd_textlayer
description: GRD text-layer extractor ported into our repo; honest benchmark shows ~5% ceiling on retail corpus
metadata: 
  node_type: memory
  type: project
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

We brought GRD (Grille/Register/Diffuser) extraction in-house, independent of the colleague's fork, on 2026-06-02. User's call: "let the team member do his thing, we do our own."

Ported `diffuser_extractor.py` (pure pdfplumber text-layer extractor) into the canonical repo `Downloads/Triune/hvac-takeoff-tool`, plus two new tools we own:
- `grd_extract.py` — production runner; derives valid_marks from our `parse_pdf_schedules()`, NOT from the answer-key takeoff (the fork cheated by reading truth marks).
- `grd_benchmark.py` — honest scorer. Handles all 3 truth formats (DATA / RawData / TAKEOFF), reports precision N/A (not fake 100%) on zero extraction, scores unit (mark+neck) AND mark-only.

**Honest baseline (14-project 26.05 sample corpus, truth-marks CEILING mode):** micro UNIT recall 4.7% (65/1389), median 0%, median MARK recall 3.6%. Only Emanate Health scores well (82% unit — combined-label drawing).

**Two failure modes, both fatal at unit level:**
1. Detection gap — 7/14 projects have an empty text layer (0% mark recall): Crunch, Sephora, Tesla, CURO, AutoZone, Nike, Les Schwab. GRD labels are CAD vector graphics, not text → only vision/OCR can read them.
2. Sizing gap — Clearwater (78% mark, 0% unit), T&T, Five Below: bare mark IS in text but neck size is drawn elsewhere, not co-located. Can't borrow size from schedule (one mark → many sizes).

**Conclusion:** pure text-layer GRD only works on combined-label drawings (Emanate, Busy Bees `S1-10"` style). The colleague's "92% Busy Bees" was a best-case outlier; his status-report numbers were hardcoded in `generate_status_report.py` and didn't match his own benchmark history. The general solution needs the vision path (YOLO + bubble OCR), consistent with CLAUDE.md §10's lesson that most tags are vector graphics. See [[project_hvac_takeoff_tool]] and [[project_ocr_finetune_scaffold]].

Sample corpus on this machine: `C:/Users/JFL/Downloads/SAMPLE FILES 26.05.2026 EXTRACT/SAMPLE FILES 26.05.2026/`. Known issue: schedule parsing hangs/times out on large PDFs (38MB 1755 Ontario, 90-page Clearwater) — same timeout the fork hit.

**ROLLED BACK 2026-06-02:** User chose to reset master back to before the OCR fine-tune scaffold (to commit `51bb8fb`), discarding the GRD work AND the PP-OCRv4 scaffold from master, keeping the v10 pipeline (81% median product recall) as the most-accurate stage. All discarded work is preserved on remote branch `archive/experiments-2026-06-02` and tag `pre-rollback-2026-06-02` — recover from there if GRD or the vision-path benchmark is revisited. The ~5% text-layer ceiling finding above is why; it remains valid.
