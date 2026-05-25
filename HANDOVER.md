# HVAC Takeoff Tool — Project Handover

**Date:** 2026-05-25
**Owner:** Triune Solutions — `tech@triunesolutions.com`
**Repo:** `https://github.com/triunesolutions/hvac-takeoff-tool` (single branch: `master`)
**Working dir on this PC:** `C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool`

## Context

This plan file is itself the deliverable. On approval I'll commit its content as `HANDOVER.md` at the repo root so any new contributor can pick up cold. The doc consolidates every non-obvious fact about the codebase, models, datasets, training, benchmarks, and open work so the next operator does not have to reverse-engineer it from `CLAUDE.md` + git history.

---

## 1. What the project does (90-second version)

End-to-end takeoff tool for commercial HVAC blueprints. Input: a multi-page PDF. Output: an Excel file listing every piece of equipment (with tag, manufacturer, model, properties) + an annotated PDF + JSON sidecars.

Pipeline stages (all in `takeoff_cli.py:main()`):
1. **Schedule parse** — `schedule_parser.parse_pdf_schedules()` extracts every schedule table → `variables` list (tag + properties + page).
2. **Page filter** — keyword-scan for "MECHANICAL PLAN" pages (override with `--all-pages`).
3. **YOLO detection** — tile each page (640×640, overlap 100, DPI 200) and run `hvac_yolov8s_v10.pt` (35-class) at conf≥0.4.
4. **Tag inference** — `tag_inference.infer_tags()` runs 3 levels:
   - **L1 direct**: if a YOLO class has exactly 1 tag in the schedule, assign it.
   - **L2a fingerprint**: read PyMuPDF text near each detection, match CFM/dim/model tokens.
   - **L2b bubble OCR**: optional pre-crop via `hvac_tag_detector_v1.pt`, then OCR (EasyOCR default, PaddleOCR fine-tune via `--ocr-engine paddleocr_hvac`), match against schedule-valid tags.
   - **L3**: leave untagged.
5. **Outputs** — `{stem}_takeoff.xlsx`, `{stem}_annotated.pdf`, `{stem}_variables.json`, `{stem}_detections.json`, `{stem}_project_info.json`.

Current accuracy: median product recall **81%**, max **95%** on the 7-project benchmark suite (`benchmark_output_v10/benchmark_summary.md`). Precision varies 57–84% by project (worst on Flex corridors).

---

## 2. Code layout — every module at a glance

All Python lives at repo root (no `src/`). ~9,550 LOC across 30 modules.

### Runtime pipeline
| File | LOC | Role |
|---|---:|---|
| `takeoff_cli.py` | 1095 | Entry point. argparse, orchestrates parse→detect→infer→write. Flags: `--verify`, `--schedule-only`, `--pages`, `--all-pages`, `--conf`, `--model`, `--ocr-engine {easyocr,paddleocr_hvac}`, `--output-dir`. |
| `schedule_parser.py` | 870 | pdfplumber table extraction → `TagVariable` list. Handles tag ranges (`CU-1 thru CU-6`), compound cells, horizontal-table auto-transpose, status prefixes `(E)/(R)/(N)`. |
| `tag_inference.py` | 870 | The 3-level tag assignment. Holds `TAG_PREFIX_CLASS` map (EF→EXHAUST FAN, …) and `YOLO_CLASS_ALIASES` (e.g., `SPLIT SYSTEM`↔`CONDENSING UNIT`, list-form aliases for the AD-GRD family). |
| `tag_matcher.py` | 570 | OCR layer. `set_ocr_engine()`, `get_ocr_reader()` (EasyOCR), `get_paddle_rec()` (PP-OCRv4 fine-tune), `ocr_near_detection()`, `match_valid_tags()`. Reads env var `HVAC_OCR_ENGINE`. |
| `ocr_preprocess.py` | 59 | **Critical for train/inference parity.** `preprocess_bubble_crop(crop, upscale=3.0)` — 3× upscale + Otsu binarize. Used by **both** `prepare_ocr_finetune.py` and `tag_matcher.py`. Do not diverge. |
| `tag_extractor.py` | 240 | Legacy text-layer tag finder. Still used for `summarize_detections_by_tag()`. |
| `class_aliases.py` | 220 | Merges annotation-typo class names during YOLO dataset prep. |

### Data prep & training
| File | Role |
|---|---|
| `prepare_ocr_finetune.py` | Build PP-OCRv4 rec-head fine-tune dataset from `tag_bubble_labels.jsonl` + `tag_dataset/`. Per-project split (default held-out: `Sola_Salons`, `01_Flex_200_Corridors`). **Commit 4f0ecab** fixed the `bubble_rect_in_crop` coord bug — used to divide by 3 thinking the rect was in upscaled space, but `label_tag_bubbles_ocr.py:301-303` already undoes the upscale before writing. The bug produced 43% blank training crops. |
| `label_tag_bubbles_ocr.py` | EasyOCR-based auto-labeler for tag bubbles in 320×320 crops. Edit-distance ≤1 + OCR-confusion table (I↔1, O↔0, S↔5, etc.). Crash-safe (line-buffered + `--resume`). Hit rate on full set: **9,908 / 26,722 = 37.1%** (`reason=hit`); rest are `no_match`/`no_text`. |
| `build_tag_dataset.py` | Builds 320×320 crops + `labels.jsonl` from `ground_truth.jsonl`. |
| `build_yolo_tag_dataset.py` | Builds YOLO-format 2-class (symbol, tag_bubble) dataset from `labels.jsonl` + `tag_bubble_labels.jsonl`. |
| `train_yolo.py` | End-to-end YOLO training (Colab/Kaggle T4). 455 LOC. |

### Verification & benchmarking
| File | Role |
|---|---|
| `benchmark_samples.py` | Runs full CLI on each project in a sample corpus, scores per-product recall/precision against team's truth xlsx. Flags: `--projects`, `--cache`, `--root`, `--out`, `--layout {sample_files,projects}`, `--scanner` pass-through. Outputs `benchmark_results.csv`, `benchmark_per_product.csv`, `benchmark_summary.md`. |
| `smoke_test_paddle.py` | Loads `models/rec_ppocr_v4_hvac/` and runs it on 100 random val crops; reports exact-match + char-level accuracy. **This is the script that caught the 4f0ecab coord bug** (model showed 20% exact / 18% char before the fix). |

### Label Studio review loop
| File | Role |
|---|---|
| `export_to_label_studio.py` | `detections.json` → LS tasks (200 DPI base64 PNG, predictions pre-filled, JWT refresh-token auth). |
| `import_from_label_studio.py` | Pull verified annotations → `ls_ground_truth.json` + `ls_discrepancy_report.csv` + `ls_summary.txt`. |

---

## 3. Models — `models/` directory

| File | Size | What | Loaded by |
|---|---:|---|---|
| `hvac_yolov8s_v10.pt` | 22 MB | 35-class equipment detector. **Production default.** Trained April 29 on 124 projects, ~25K tiles. | `takeoff_cli.py` via `--model` (default) |
| `hvac_yolov8s_v9.pt` | 22 MB | Legacy v9. Kept for `--model` override. | manual |
| `hvac_tag_detector_v1.pt` | 22 MB | 2-class bubble detector (symbol vs tag_bubble). Trained April 27. Feeds L2b pre-crop. | `tag_inference.py` Level 2b |
| `rec_ppocr_v4_hvac/` | ~13 MB | PP-OCRv4 rec-head fine-tuned on 8,998 verified bubble crops. **Re-trained 2026-05-21** on corrected dataset (post-4f0ecab). | `tag_matcher.get_paddle_rec()` when `--ocr-engine paddleocr_hvac` |

`.gitignore` whitelists: `!models/hvac*.pt`, `!models/rec_ppocr_v4_hvac/**`.

> **Status note for rec_ppocr_v4_hvac:** Two training runs happened. The first (May 20) was on broken data — discard. The current weights came from a re-train on corrected coords. As of 2026-05-22 the user downloaded `final_model/` containing only the *training checkpoint* (`best_accuracy.pdparams` 68 MB) — the inference export still needs to run on Kaggle (`tools/export_model.py`) before the weights can be loaded by `PaddleOCR(rec_model_dir=...)`. Expected exported size: 10–15 MB total.

---

## 4. Datasets — where everything lives

| Artifact | Location | Committed? | Notes |
|---|---|---|---|
| `ground_truth.jsonl` | repo root | ✅ (19.8 MB) | 26,844 LS-verified detections from 6 projects (Erewhon, Bungalow, BMO Santee, Saint Mary's, Sola, Anaheim 82). Schema: per-detection page+rect+tag+class+properties. |
| `tag_bubble_labels.jsonl` | repo root | ✅ (3.7 MB) | 26,722 rows, one per crop. `reason ∈ {hit, no_match, no_text}`. Only `hit` rows carry `bubble_rect_in_crop` (in 320×320 source-crop space). |
| `labels.jsonl` | repo root | ❌ untracked | 26,722 source annotations from `build_tag_dataset.py`. Regenerate via build script. |
| `tag_dataset/` | gitignored | ❌ | 26,722 PNG crops (320×320), ~595 MB. Pull from GH release `datasets-2026-05-11`. |
| `ocr_finetune/` | gitignored | ❌ | Regenerated locally from `prepare_ocr_finetune.py`. Current manifest: 8,998 train / 363 val / 87 projects / vocab 37 / max_text_length 12 / preprocess "3x upscale + Otsu". Held-out val projects in current build: `Copper_Ranch`, `Goodwin_House`. |
| `ocr_finetune.zip` | repo root | ❌ untracked (8.4 MB) | Packaged for Kaggle upload. Up to date with the 4f0ecab fix. |
| `images/` | repo root | ❌ untracked | Source tag-detection crops parallel to `tag_dataset/`. |

### Re-source on a fresh PC

```bash
git clone https://github.com/triunesolutions/hvac-takeoff-tool
cd hvac-takeoff-tool

# Dataset (596 MB)
gh release download datasets-2026-05-11 --repo triunesolutions/hvac-takeoff-tool --pattern "tag_dataset.zip"
python -c "import zipfile; zipfile.ZipFile('tag_dataset.zip').extractall('.')"

# Benchmark PDFs — NOT in repo. Re-source from team Drive:
#   "SAMPLE FILES 27.04.26"  →  C:\Users\JFL\Downloads\SAMPLE FILES 27.04.26\
```

---

## 5. Environments

**Main env** (Python 3.12+; this PC uses 3.14): `pip install PyMuPDF Pillow opencv-python-headless pandas openpyxl easyocr ultralytics pdfplumber`.

**Paddle env** — separate, because paddlepaddle has no wheels for 3.14: `.venv-paddle/` (Python 3.11.15, created with `uv`).
- `paddlepaddle==2.6.*` (CPU, from `https://www.paddlepaddle.org.cn/packages/stable/cpu/`)
- `paddleocr==2.7.3`
- `numpy<2` (paddleocr deps drag in numpy 2.x; pin it down for ABI compatibility)
- Activate: `.venv-paddle/Scripts/python.exe …`

**No `requirements.txt` / `pyproject.toml`.** Install commands live in `CLAUDE.md` "New-PC bootstrap".

---

## 6. Training playbook — what runs where

### YOLO equipment detector (v10 production)
- **Notebook:** `kaggle_train_v10.ipynb`
- **Dataset:** Kaggle dataset `hvac-yolo-dataset-v10` (split into 2GB parts; reassemble via `cat *.part-*`)
- **Hardware:** T4 ×2 or P100, ~3–4h
- **Output:** `hvac_yolov8s_v10.pt` (22 MB) → drop into `models/`

### Tag-bubble detector
- **Notebook:** `kaggle_train_tag_detector.ipynb`
- **Dataset:** `yolo_tag_dataset/` (23K train / 3.4K val, 2-class)
- **Hardware:** T4 ×2, ~2–3h
- **Output:** `hvac_tag_detector_v1.pt` (22 MB)

### PP-OCRv4 rec-head fine-tune
- **Notebook:** `kaggle_train_ocr_rec.ipynb`
- **Dataset:** upload `ocr_finetune.zip` as Kaggle dataset (auto-mounts to `/kaggle/input/<slug>/ocr_finetune/`)
- **Hardware:** T4 single GPU. **Actual runtime: ~2.5–4h on T4** (the v4 MultiHead with NRTR transformer head is much heavier than v3; the original "~60-min" estimate in earlier docs was a v3 number — ignore it).
- **Key config:** lr=5e-4, batch=128/card, 80 epochs, cosine LR + 5-epoch warmup, char dict = 37 (A-Z, 0-9, `-`), max_text_length=12.
- **Output:** training checkpoint at `output/rec_ppocr_v4_hvac/best_accuracy.{pdparams,pdopt}`. **Must then run `tools/export_model.py`** to produce `inference/rec_ppocr_v4_hvac/{inference.pdmodel,inference.pdiparams}` — this is the only form PaddleOCR can load at inference. Final zip is ~13 MB.

#### Gotchas observed in the current train run (2026-05-21)
- `numpy<2` pin produces a wall of red "incompatible" warnings from Kaggle's base image packages (jax, cupy, rasterio, shap, etc.). **They are warnings, not errors** — those packages aren't used by training. Ignore.
- Missing `lmdb` module: PaddleOCR imports it unconditionally even though we use `SimpleDataSet`. `pip install lmdb` upfront.
- First few epochs report `ips: ~50 samples/s` and a scary ETA — that's real T4 throughput for v4 MultiHead, not CPU-bound (verify via `paddle.device.get_device()` → `gpu:0`). ETA decays as cuDNN autotune + dataloader caches warm.
- The training checkpoint is 68 MB (`pdparams`) + 121 MB optimizer state. Inference export drops the NRTR head and BN running stats, lands at ~10–12 MB.

---

## 7. Benchmark — the ship gate

**Command:** `python benchmark_samples.py --layout projects --out benchmark_output_v10`

**Baseline (v10, 2026-05-18):**
- 7 projects discovered, 6 scored (Aritzia Americana times out >10 min — known issue, large PDF)
- **Median product recall: 81%** · **Max: 95%** (Yucaipa Meadows BldgB) · **Min: 68%** (Flex 220)
- All 6 scored projects ≥ 50% recall

**Ship criteria for any new slice (PP-OCRv4 swap, v11, …):**
1. Median product recall ≥ **79%** (no more than 2-point drop from 81%)
2. Median product precision **strictly higher** than v10 (current ~74% median across scored projects)
3. No project regresses by more than 5 points on either axis
4. Yucaipa Meadows BldgB must stay above 90% recall / 80% precision

If gate fails: fall back via `--ocr-engine easyocr`, drill into per-project `_detections.json` diffs.

---

## 8. Open work / handover queue

### Immediately pending (PP-OCRv4 retrain in flight)
1. **Export inference model** — re-launch the Kaggle session, run step 6 of `kaggle_train_ocr_rec.ipynb` (`tools/export_model.py` against the saved `best_accuracy` checkpoint). Download `rec_ppocr_v4_hvac.zip` (~10–13 MB).
2. **Drop into repo** — extract to `models/rec_ppocr_v4_hvac/`, verify `inference.pdmodel` + `inference.pdiparams` + `dict.txt` are present.
3. **Re-run `smoke_test_paddle.py`** — expect ≥95% char-level accuracy (previous broken-data run was 18%).
4. **Re-source benchmark PDFs** from team Drive (`SAMPLE FILES 27.04.26` is *not* in repo).
5. **Run `benchmark_samples.py` head-to-head** EasyOCR vs `paddleocr_hvac`. Apply ship-criteria gate.
6. **Commit the export-step results** + flip `takeoff_cli.py` default to `paddleocr_hvac` if the gate passes.

### Queued (from CLAUDE.md §10 & §17, none implemented)
- Skip LEGEND / SCHEDULE / DETAILS sheets before YOLO (would knock out ~80% of the 61 phantoms found in the May 5 LS review).
- Bump `level2b_bubble_detect.max_distance` from 350 → 600.
- Extend `TAG_PREFIX_CLASS` with `TA` (transfer air), `LD` (linear damper), `MD` (motorized damper).
- Page-level NMS in `takeoff_cli.py` (overlapping duplicate boxes observed in Erewhon, Bungalow).
- Schedule-page OCR fallback for raster schedules (Krispy Kreme).
- Title-block extractor: sheet-number heuristic, latest-date selection, engineer-initials parsing (`extract_project_info_spatial` already exists).
- Aritzia Americana benchmark timeout (>10 min, large PDF) — streaming pdfplumber fallback.

### Known model behaviour issues (do not "fix" without reading first)
- `SPLIT SYSTEM` vs `CONDENSING UNIT`: handled via `YOLO_CLASS_ALIASES` — extend, don't rename in training data.
- AD-GRD vs AD-T-BAR SUPPLY/RETURN: 89 LS relabels in one batch. List-form alias in place; bubble OCR disambiguates.
- EasyOCR < 15 px height: unreliable. Tag-bubble detector pre-crop is the mitigation.
- Bluebeam ground-truth calibration is on hold — team is still preparing the CSV export.

---

## 9. CI / deployment

**There is no CI.** No `.github/workflows/`. No automated tests. Every change is validated by running `benchmark_samples.py` manually.

**There is no deployment.** This is an internal CLI tool. Phase 4 (SaaS) is in `PRD.md` — not started.

---

## 10. Reference — release tags & external resources

| Resource | Where |
|---|---|
| Code repo | `https://github.com/triunesolutions/hvac-takeoff-tool` |
| Datasets release (zips) | GH release `datasets-2026-05-11` (multi-part 2GB chunks for `tag_dataset.zip`, `yolo_dataset_v10.zip`, `yolo_dataset_v11.zip`) |
| Benchmark PDFs | Team Drive — `SAMPLE FILES 27.04.26/` (not in repo) |
| Label Studio | Self-hosted; JWT refresh-token auth (see `export_to_label_studio.py`) |
| Kaggle datasets | `hvac-yolo-dataset-v10`, `hvac-yolo-dataset-v11`, `hvac-tag-yolo-v1`, `ocr-finetune-zip` (under `mandeeps1ngh`) |
| Kaggle notebooks | `kaggle_train_v10.ipynb`, `kaggle_train_v11.ipynb`, `kaggle_train_tag_detector.ipynb`, `kaggle_train_ocr_rec.ipynb` (all committed at repo root) |
| Engineering memory | `CLAUDE.md` (733 lines, last updated 2026-05-11) |
| Product/vision | `PRD.md`, `WHAT_WE_ARE_BUILDING.md` |
| v10 launch notes | `docs/V10_LAUNCH.md`, `docs/v10_vs_v9_2026-04-30.md` |

---

## 11. Where files I created in this session live

| File | Purpose | Committed? |
|---|---|---|
| `prepare_ocr_finetune.py` | Coord-fix landed in commit 4f0ecab | ✅ master |
| `ocr_finetune/` | Regenerated dataset (8,998 train / 363 val, zero blank crops) | ❌ gitignored |
| `ocr_finetune.zip` | Packaged for Kaggle upload | ❌ |
| `smoke_test_paddle.py` | Val-set accuracy probe | ❌ untracked — should be committed |
| `models/rec_ppocr_v4_hvac/` | The broken-data inference model (do not use) | ❌ — overwrite once new export lands |
| `.venv-paddle/` | Python 3.11 venv for PaddleOCR | local only |

---

## 12. Quick reference — useful one-liners

```bash
# Run a takeoff
python takeoff_cli.py "path/to/blueprint.pdf"

# Force PaddleOCR fine-tune for the bubble OCR step
python takeoff_cli.py "x.pdf" --ocr-engine paddleocr_hvac

# Re-build the OCR fine-tune dataset (after editing tag_bubble_labels.jsonl)
python prepare_ocr_finetune.py

# Custom held-out val projects
python prepare_ocr_finetune.py --val-projects Sola Erewhon

# Smoke-test the fine-tuned PaddleOCR on val crops
.venv-paddle/Scripts/python.exe smoke_test_paddle.py

# Run the full benchmark suite
python benchmark_samples.py --layout projects --out benchmark_output_v10

# Pull the ~600MB tag_dataset zip from GH release
gh release download datasets-2026-05-11 --repo triunesolutions/hvac-takeoff-tool --pattern "tag_dataset.zip"
```

---

## Verification

To confirm this handover is correct end-to-end on a fresh PC:
1. Clone the repo and follow §4 "Re-source on a fresh PC" — should produce `tag_dataset/` with 26,722 PNGs.
2. Run `python prepare_ocr_finetune.py` → should print `train: 8998   val: 363` with `skipped: missing=0 bad_crop=0`.
3. Re-source benchmark PDFs from team Drive.
4. Run `python takeoff_cli.py <one of the benchmark PDFs>` and confirm it produces an xlsx + annotated PDF without errors.
5. Run `python benchmark_samples.py --layout projects --out benchmark_output_v10_repro` and verify median recall stays at 81%.

If all five succeed, the handover is valid. If anything diverges, the most likely cause is a missing Python package (cross-reference §5) or a missing PDF in the benchmark corpus.

---

## Plan: what gets committed

On approval I will:
1. Write this exact content to `HANDOVER.md` at repo root.
2. Add `smoke_test_paddle.py` to the commit (it's currently untracked but useful — referenced from this doc).
3. Commit and push to `triunesolutions/hvac-takeoff-tool` master.

No other changes.
