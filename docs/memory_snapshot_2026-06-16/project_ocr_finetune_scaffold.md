---
name: project-ocr-finetune-scaffold
description: PP-OCRv4 rec-head fine-tune trained+installed but benchmark gate FAILED 2026-05-26 (identical to EasyOCR + slower; OCR not the bottleneck) — keep EasyOCR, do not swap; real blockers are schedule parsing + AD-GRD taxonomy on new corpus
metadata: 
  node_type: memory
  type: project
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

Code scaffold for fine-tuning a PP-OCRv4 recognition head on HVAC tag bubbles was pushed to `triunesolutions/hvac-takeoff-tool` master at commit `32f4cd8` on 2026-05-19.

**Why:** v10 hits 81% median recall / 95% max but Flex projects sit at 57-66% precision, driven largely by EasyOCR character confusions (O↔0, I↔1, S↔5, B↔8, dropped hyphens) on CAD-stamped tag bubbles. Canaveral (canaveral.ai, $1,668/seat/yr) is a live competitor; tag accuracy is the moat. User picked this Tier-3 slice over the cheaper schedule-snap option to attack accuracy at the source.

**How to apply:**
- Files: `ocr_preprocess.py` (shared 3× + Otsu helper), `prepare_ocr_finetune.py` (9,908 hits → train/val), `kaggle_train_ocr_rec.ipynb` (T4 fine-tune), `tag_matcher.py` (`--ocr-engine paddleocr_hvac` switch via `set_ocr_engine`), `takeoff_cli.py` (`--ocr-engine` flag, default `easyocr`), `CLAUDE.md` §20 (full procedure).
- Trained weights commit at `models/rec_ppocr_v4_hvac/` (whitelisted in `.gitignore`).
- Held-out val projects: `Sola_Salons`, `01_Flex_200_Corridors` — per-project split, not random.
- Ship criteria: median product recall ≥ 79%, median precision strictly above v10, no project regresses > 5pts, Yucaipa B stays ≥ 90% / 80%. Benchmark via `python benchmark_samples.py --layout projects --out benchmark_output_v10_paddleocr`.

**Train/inference parity is critical** — both sides go through `ocr_preprocess.preprocess_bubble_crop` (3× upscale + Otsu). CORRECTION (was wrong in earlier version of this memory): `bubble_rect_in_crop` in `tag_bubble_labels.jsonl` is in the original **320×320 source-crop space, NOT 3× space** — `label_tag_bubbles_ocr.py:301-303` undoes its own upscale before writing. `prepare_ocr_finetune.py` used to divide it by 3; that was the bug fixed in commit **4f0ecab** which had produced 43% blank train / 50% blank val crops (smoke test caught it at 18% char acc). Do not re-introduce the /3. Do not edit one side without re-training.

**Status as of 2026-05-26:** Trained on corrected data (Kaggle T4, full 80 epochs, ~5h not ~1h — v4 MultiHead with NRTR head is heavy). Best val metric epoch 67: **80.4% exact-match / 0.892 norm_edit_dis** on held-out projects Copper_Ranch + Goodwin_House (val-projects filter changed from Sola/Flex to these). Exported via `tools/export_model.py` → `inference.pdmodel`+`inference.pdiparams` (~7.4 MB), already installed in `models/rec_ppocr_v4_hvac/`. Local smoke test (`smoke_test_paddle.py`): **76% exact / 80.7% char**. Failures cluster on 2-char tags (CTC stutter `HP-IDF`→`HP-IDHP-IDF`, dropped leading char `DB`→`B`, no-context misses `LD`→`1`) — partly mitigable by constraining to schedule-valid tags in `match_valid_tags()`.

**BENCHMARK GATE RESULT (2026-05-26): FAILED — do NOT swap, keep EasyOCR.** Ran head-to-head on a 4-project subset of the new `SAMPLE FILES 26.05.2026` corpus (Crunch, Emanate, Five Below, AutoZone; v10 model; both engines in `.venv-paddle`). Paddle produced **byte-for-byte identical takeoff output** to EasyOCR (same product counts, same recall/precision to 6 decimals) on all 3 projects where both completed, while running **1.2–2.6× slower** (Emanate timed out >30min on paddle vs 12min easyocr). Reason: bubble OCR (Level 2b) barely fires here — `schedule_tags_found == 0` on 3 of 4 projects, so there's no valid-tag list to match and the OCR engine is never exercised. The fine-tune is technically fine (80% char acc) but OCR is NOT the binding constraint on this corpus.

**The actual bottlenecks on the new sample set (none OCR-related, this is where to work next):**
1. **Schedule parser returns 0 tags** on Crunch/AutoZone/Five Below (Dick's found 57 — parser works, these Haldeman schedule layouts defeat it). #1 blocker: kills tag inference entirely.
2. **YOLO class taxonomy:** diffusers come out as generic `AD-GRD` (93 on Crunch) instead of the team's `AD-T-BAR/AD-SURF SUPPLY/RETURN` subtypes → ~0 product match despite boxes existing. (Known issue, CLAUDE §19.5.)
3. **Product-name vocabulary:** team uses `NEW EXHAUST FAN`/`TOILET EXHAUST FAN`/`DESTRATIFICATION FAN`; we emit `FAN`.
4. **Detection misses** (Five Below ours=3 vs team 37).

This whole machine has **no GPU** (torch is `+cpu`, no nvidia-smi) — full pipeline is ~8–15 min/project on these large-format sheets; the historical "~1.5 min/proj" baseline came from other hardware. `benchmark_samples.py` now has `--ocr-engine` and `--timeout` pass-through flags (added 2026-05-26). New corpus extracted at `C:\Users\JFL\Downloads\SAMPLE FILES 26.05.2026 EXTRACT`. NOTE: `models/rec_ppocr_v4_hvac/` is installed locally but shows **untracked in git** — never committed.

**Open next steps:** attack schedule parsing on the new Haldeman layouts (biggest win) and the AD-GRD subtype taxonomy; revisit the OCR swap only after those unblock Level 2b.

Related: [[project-hvac-takeoff-tool]], [[project-laptop-handoff]], [[feedback-repo-sync]].
