# Repo Review — 2026-06-10

Reviewed at commit `0fcee71` on `claude/repo-review-f5203g`. Three passes: documented
follow-ups vs. actual code, core pipeline code review (findings verified by direct
read), and repo hygiene. Findings below are ordered by what's worth acting on first.

**Overall: healthy and improving.** The last two weeks (time budgets, schedule OCR
fallback, dead-code removal, repo reorg) attacked the right bottleneck — the June
benchmarks confirm recall is gated by schedule tag extraction, not the YOLO model.
No secrets committed, .gitignore solid, core/periphery split sensible. Main debt:
stale documentation, a few resource-handling bugs, machine-specific tooling.

---

## 1. CLAUDE.md is materially stale (highest-impact issue)

CLAUDE.md is the navigation reference for every engineer and future session, and it
is wrong in three ways:

### 1.1 The "open follow-ups" list (2026-05-11) — 3 of 5 items are now DONE

| Item | Status | Evidence |
|---|---|---|
| Schedule-page OCR fallback for raster schedules | **DONE** | `extract_marks_via_ocr()` at `schedule_parser.py:696-780` (commit 66f56cb) |
| Page-level NMS in takeoff_cli | **DONE** | `takeoff_cli.py:505-509`, `NMS_DIST = 50` (per-page only; no cross-page dedup) |
| Skip LEGEND/SCHEDULE/DETAILS sheets from YOLO | **DONE** | `NON_PLAN_TITLE_MARKERS` + `_is_non_plan_sheet()` at `takeoff_cli.py:741-766` |
| `level2b_bubble_detect` max_distance 350 → 600 | NOT DONE | `tag_inference.py:459` still defaults to 350 |
| Add `TA` to `TAG_PREFIX_CLASS` | NOT DONE | `MD`/`LD` present; `TA` missing (`tag_inference.py:27-68`) |

Also still open from §19.3 / §15.4: the Label Studio 50-char title clamp
(`scripts/export_to_label_studio.py:178` builds the title with no truncation), and
the title-block sheet-number heuristic + latest-date selection.

### 1.2 §12 file layout predates the cf24da2 reorg

`train_yolo.py` and `benchmark.py` are documented at root but live in `scripts/`;
`colab_train.ipynb` is in `notebooks/`; `PRD.md` and `WHAT_WE_ARE_BUILDING.md` are in
`docs/`.

### 1.3 Undocumented directories

`ground_truth/` (6 reviewed projects), `batch_datatrain_local/` (157-project batch
report), and the root `sample_*.csv` files appear in neither CLAUDE.md §12 nor
README.md. The `~/.label_studio_token` setup step is also undocumented in §19.

---

## 2. Core pipeline code findings

All verified by direct read, not just automated review.

### Medium

1. **`schedule_parser.py:785` — PDF handle leak.** `fitz.open(pdf_path)` in
   `extract_marks_via_ocr()` is never closed (returns at :838 without close). One
   doc per run so impact is small, but wrap in try/finally.
2. **`tag_inference.py:622-634` — `extract_nearby_text()` three issues.**
   (a) Opens the PDF fresh *per detection* inside the Level-2a fingerprint loop —
   measurable perf hit on big plans; (b) no try/finally around `get_text` before
   `doc.close()`, so an exception leaks the handle; (c) hardcodes `scale = 200/72`
   instead of sharing the `DPI = 200` constant from `takeoff_cli.py:39` — silently
   wrong if DPI ever changes.
3. **`takeoff_cli.py:654-659` — brand/model split.** The fallback splits the
   combined MANUFACTURER+MODEL string on the *first* space, so multi-word brands
   ("UNITED COOLAIR") split incorrectly in the Excel.
4. **`schedule_parser.py:523-530` — headerless mark-column fallback.** The
   tag-shape regex requires letters+digits, so bare single-letter tags (`A`, `B`)
   that `normalize_tag` accepts are rejected — schedules with bare-letter marks and
   no MARK/TAG header row get missed.

### Low

5. Broad silent `except Exception: continue/pass` throughout `schedule_parser.py`
   (e.g., :361-366, :812, :819). Appropriate for messy PDFs, but a one-line debug
   log would make field failures diagnosable.
6. Two parallel "what's a valid tag" regexes: module-level `TAG_REGEX`
   (`schedule_parser.py:26`) vs. the function-local validation inside
   `normalize_tag`. Two sources of truth that can drift.
7. `takeoff_cli.py:1062` — the `detections.json` write has no try/except while the
   sibling JSON writes do.
8. `normalize_tag` is ~100 lines with ~10 return paths and no unit tests — works,
   but hard to extend safely. Candidate for a staged-validator refactor with a small
   test file of known-good/known-bad tags pulled from the benchmark corpus.

### Checked and cleared (not bugs)

- The OCR-fallback budget at `schedule_parser.py:881` can go negative, but the
  guard at :884 (`ocr_tb >= OCR_FALLBACK_MIN_BUDGET`) correctly skips OCR in that
  case.
- The detection-deadline reserve at `takeoff_cli.py:950` capping against
  `args.time_budget` (not remaining time) matches the documented intent — reserve
  scales with detection count from zero and is capped vs. the total budget.

---

## 3. Repo hygiene

1. **`.claude/sessions/*.jsonl` — 14.5 MB of session transcripts tracked in git**
   (`current_session.jsonl` 6.5 MB, `previous_session.jsonl` 8 MB). Not source, not
   data — should be gitignored and `git rm --cached`'d. `.claude/memory/` is fine.
2. **`requirements.txt` missing `requests` and `PyYAML`** — used by
   `scripts/export_to_label_studio.py`, `scripts/import_from_label_studio.py`,
   `scripts/train_yolo.py`, `scripts/build_v11_dataset.py`.
3. **Hardcoded `C:\Users\JFL\...` paths** in six scripts (`scripts/benchmark.py`,
   `scripts/train_yolo.py`, `scripts/prep_sample_training.py`,
   `scripts/survey_sample_classes.py`, `scripts/test_parser_accuracy.py`, root
   `benchmark_samples.py`) plus a Windows `cd` in
   `scripts/batch_prepare_review.sh:5`. Fine as dev-local tooling; an env-var or
   `--root` override would make them portable.
4. README.md layout section omits `ground_truth/`, `batch_datatrain_local/`, and
   `sample_*.csv`.

Strengths worth keeping: no credentials in the repo (LS token correctly read from
`~/.label_studio_token`); models and frozen data files committed deliberately at
reasonable sizes; .gitignore covers outputs, datasets, and runs.

---

## 4. Recommended actions, in priority order

Quick wins (low-risk, no parser-behavior change):

1. Untrack `.claude/sessions/` (gitignore + `git rm --cached`).
2. Add `requests`, `PyYAML` to `requirements.txt`.
3. Update CLAUDE.md: mark the 3 done follow-ups DONE, fix §12 layout, document
   `ground_truth/` / `batch_datatrain_local/` / LS token setup.
4. try/finally around the two `fitz.open` calls (`schedule_parser.py:785`,
   `tag_inference.py:627`).
5. Clamp the LS project title to 50 chars in `scripts/export_to_label_studio.py:178`.
6. Add `TA` to `TAG_PREFIX_CLASS`.

Needs owner judgment / benchmark evidence first:

- `max_distance` 350 → 600 — run the sample benchmark before/after to confirm it
  helps more than it hurts (a wider radius can mis-tag dense plans).
- Cross-page NMS (per-page NMS exists; CLAUDE.md §19.6 localization noise was the
  motivation — confirm it still reproduces on Erewhon/Bungalow).
- Parameterize the six Windows-path scripts.
- `normalize_tag` refactor + unit tests.
- Hoist `extract_nearby_text` PDF open out of the per-detection loop (measure first;
  only matters on plans where Level 2a fires a lot).
- Title-block extractor: sheet-number heuristic, latest-date selection.
