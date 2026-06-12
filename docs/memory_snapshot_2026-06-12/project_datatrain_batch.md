---
name: project_datatrain_batch
description: "COMPLETE local end-to-end batch over the 158-project data-to-train corpus — 157/157 done, 14.5% tagged, tagging cliffs at ~7MB"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

2026-06-09: **COMPLETE — 157/157 finished** (resumed the paused run; 82 skipped via --resume, 75 run fresh). ~22,957s total wall-clock across both days. **Zero crashes** across 0.7 MB → 167 MB.

**Final results:** 133/157 produced xlsx (85%), 24 no_xlsx (0 detections), 0 crashes. **8,604 detections, 1,244 tagged = 14.5% overall.** Report: `batch_datatrain_local/batch_report.md`.

**Decisive finding — tagging has a hard size cliff at ~7 MB.** EVERY project tagging ≥50% is ≤6.7 MB (clean schedule-driven small plans: Flex/Knape/United/Optum/Mavis/BMO/Woodbridge, several 90–100%). Above ~7 MB tagging collapses to ~0% (best large: JFK ITB 38.7%, Parallel Waco 30.6%, Collins 17%); the 30 MB+ tail is 0% bar 3 low-single-digit exceptions. Detection fires fine on big plans (Strathmore 182, Quest 254, Rice 246) — tag-inference dies, not detection. **This is TRAINING data, so 14.5% on memorized plans is a damning floor.** Tag-inference is unambiguously THE bottleneck.

**Two cap leaks** (240s `--time-budget` doesn't bound the schedule-OCR render loop): 05_Basin_Electric 2645s, 43_Bethesda_Health 978s → need a hard kill-switch on the OCR-fallback path.

**Several no_xlsx are picker misfires, not detection failures** (parsed schedule fine, 0 detections = wrong sheet): Perch 83/0, ScanHealth 91/0, Sheraton 81/0, UCLA 55/0. Redo list: Copper_Ranch, Basin_Electric (electrical), Vista_Murrieta (Takeoff_-prefixed output), Fort_Totten, Terrebonne, Woodbridge, Autry.

---
ORIGINAL (2026-06-08, superseded by completion note above):
Started a full end-to-end pipeline run over the entire `data to train` corpus (`C:\Users\JFL\Downloads\Triune\data to train\projects`, 158 project folders). **Paused at 82/157** to shut down for the day — fully resumable.

**Tool:** `scripts/batch_run_all.py` (new this session). `--recursive` picks one plan PDF per project (skips _annotated/Takeoff/Schedule/Specification, prefers mechanical/drawings keyword then largest; every pick logged to `plan_selection.csv`). Checkpoints via `_done.json` per project + `_run.log` terminal-banner detection. Output → `batch_datatrain_local/` (gitignored). Report (`batch_report.md` + `batch_results.csv`) rewrites after every project.

**Resume command:**
`python -X utf8 scripts/batch_run_all.py --input "C:\Users\JFL\Downloads\Triune\data to train\projects" --out batch_datatrain_local --recursive --time-budget 240 --resume`

**Headline finding (82 projects):** 69/82 produced xlsx, 13 zero-detection, 0 crashes. **4,532 detections but only 23.1% tagged** → detection is fine at scale, **tag-inference is the bottleneck** (consistent with [[project_v10_newcorpus_benchmark]] and [[project_schedule_ocr_budget_fix]]). Pipeline is robust (no crashes).

**Two caveats to remember:**
1. **This is TRAINING data** — v10 trained on these 158 projects, so scores are inflated (memorization). Valid as a robustness/floor check ONLY, not a real accuracy number. For real accuracy need held-out plans + team ground truth.
2. **~7 picker misfires** to re-run with corrected picks: 120_Copper_Ranch (picked an architectural sheet), 05_Basin_Electric (1 of 78 sheets), 08_Fort_Totten (one floor), Terrebonne, Woodbridge, Autry (partial sheets), and a "Takoeff_" misspelling that slipped the skip-filter. Audit `plan_selection.csv`.

**Faster path available:** `notebooks/kaggle_batch_inference.ipynb` + `get_ocr_reader` now auto-uses GPU — running this corpus on a Kaggle T4 would be ~2-4 hr vs ~10 hr local (local no-cap was ~27 min for a 1.2 MB plan). Needs the 6 GB corpus uploaded as a Kaggle dataset.
