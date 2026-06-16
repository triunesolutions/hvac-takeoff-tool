---
name: project_ocr_vector_schedules
description: "OCR table extractor for vector/CAD schedule sheets (investor demo 2026-06-16); schedule reconstruction works, per-symbol mark/size tagging does NOT"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

2026-06-16 investor-demo work (presenting ~2026-06-22 to potential investors) on
two projects with **vector/CAD schedule tables** (text drawn as line-art, zero
text layer): **Cityvet Verrado** and **CCV Buckeye**. Builds on [[project_ws1_tag_inference]].

**Shipped to master (`triunesolutions/hvac-takeoff-tool`):**
- `ocr_table_extractor.py` — reconstructs full schedule tables from vector sheets:
  cell-grid detection (cell-clustering, not connected-components — stacked tables
  merge otherwise) → OCR tokens into cells → MARK-column re-OCR (upscale+allowlist)
  to recover single-letter marks EasyOCR drops. Emits TagVariables w/ populated
  properties, same shape as the text parser. Auto-fires in `parse_pdf_schedules`
  when the text layer yields <3 tags. Commits a02d8c4, 7544d78.
- Air-device class inference: Diffuser/Grille/Register schedules put device type
  in DESCRIPTION (not TYPE) + tag in SYMBOL; marks now map into the AD-GRD family
  so they connect to AD-GRD detections (7544d78).
- `takeoff_cli` Phase A: populate MODULE/DUCT SIZE columns, compose TYPE to team's
  "SUPPLY DIFFUSER" style, MATERIAL+DAMPER into remark (7544d78).
- Level-2b bubble OCR: batched + tight 90px/2x crop, **~8x faster** + 180s cap
  (was per-detection readtext, uncapped → 50-min runs). `ocr_crops_batched` in
  tag_matcher (dbd85b8). PARKED (uncommitted→committed): `read_diffuser_annotations`.

**WHAT WORKS (demo headline):** vector-schedule reconstruction — every mark with
brand/model/type/CFM/material off sheets with NO text layer. Cityvet + CCV both.
Plus full equipment detection + counts.

**WHAT DOES NOT WORK (proven, do not re-litigate):** per-symbol mark/size tagging
on these dense CAD plans.
- Cityvet diffusers ARE labeled (pentagon bubble = letter B/A/C + size "10x6" +
  CFM "260") but EasyOCR can't read the single letter inside the pentagon (5/15,
  mostly WRONG — reads pentagon edges as D/K). The old "87% tagged" was an
  ARTIFACT: blind direct-mapping tagged all 81 diffusers "A" (team's real answer
  is A=26/B=6/C=19/D=11). Correct class inference dropped it to 0% — exposing
  that per-mark needs the letter.
- CCV diffusers carry NO mark letter on the RCP at all (15% tagged, all from
  direct-map). The team assigns marks by symbol-TYPE + neck-size matched to the
  schedule — not by reading the plan.
- Per-diffuser size reading also unreliable: 13/83 = 15%, some misread (8x6→80x6).
- ROOT CAUSE: small isolated text on busy CAD linework = low EasyOCR recall.
  The real fix is a MODEL investment (YOLO to distinguish diffuser sub-types /
  detect+read pentagon tag-bubbles), weeks not days — POST-DEMO.

**Demo recommendation (agreed):** lead with schedule reconstruction + detection
counts; frame per-mark as "assisted takeoff, estimator confirms." Do NOT show the
old 87%-all-A number or 15% sizes — JFL's HVAC team would catch it.

**Source files (in Downloads, NOT repo — regenerate from team's rars):**
`Downloads/@01-05 Cityvet...rar`, `@01-02 CCV Buckeye...rar`. Cityvet schedule =
plan p2 (M0.10). CCV schedules = p8/9/10 (M002-4, zero-text). `HVAC_OCR_SCHED_PAGES`
env var targets schedule pages to skip the slow blind page sweep.
