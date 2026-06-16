---
name: Laptop Handoff postponed to ~2026-06-13
description: Laptop submission deferred ~1 month from 2026-05-13; staying on current PC, focus back on accuracy work
type: project
originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---
Laptop handoff was originally planned for 2026-05-11 but on 2026-05-13 user postponed it by ~1 month (target ~2026-06-13). Continuing on the current PC for now.

Why: The user needed everything Triune-related preserved before submitting the device. Non-Triune folders (jhaveri-leads, reassurance-site) were intentionally deleted.

How to apply: When the user resumes work, follow the New-PC bootstrap section at the top of `CLAUDE.md` in `triunesolutions/hvac-takeoff-tool`. Specifically:
- The repo holds source + 3 models (v9, v10, tag_detector_v1) + ground_truth.jsonl.
- Training dataset zips live in GH release `datasets-2026-05-11`. v10/v11 zips were split (1900 MB chunks) for the 2 GB GH asset limit; reassemble with `cat *.part-* > file.zip` (Linux/Git Bash) or `copy /b a + b out.zip` (Windows cmd).
- The team's `SAMPLE FILES 27.04.26/` benchmark corpus is NOT in the repo — re-source from team Drive.
- Open follow-ups diagnosed but not implemented 2026-05-11: schedule-page OCR fallback for raster schedules (Krispy Kreme returned 0 variables); bump `level2b_bubble_detect.max_distance` 350 → 600 (Acushnet only 13% tagged); add `TA`/`LD`/`MD` to `TAG_PREFIX_CLASS`; page-level NMS in `takeoff_cli.py`; skip LEGEND/SCHEDULE/DETAILS pages from YOLO inference.
