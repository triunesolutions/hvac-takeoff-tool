---
name: reference_tag_verifier
description: Standalone tag-bubble review tool sent to the team — lives OUTSIDE the git repo
metadata: 
  node_type: memory
  type: reference
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

The **HVAC Tag Verifier** is a standalone Flask web tool we built and shipped to the takeoff team to turn the OCR labeler's guesses into clean ground truth. It lives at `C:\Users\JFL\Downloads\Triune\tag_verifier` — **NOT inside the git repo**, so the repo cleanup never touched it and grep over the repo won't find it.

Contents: `verifier.py` (single-file app + embedded HTML/JS UI), `labels.jsonl` (3.7 MB — the OCR labeler's `tag_bubble_labels.jsonl` candidates, shipped with it), `README.md`/`STEPS.md` (team instructions), `setup.bat`/`run.bat` (double-click install/launch), `requirements.txt` (just flask). The crops ship separately as `tag_dataset.zip` (578 MB, unzip to `dataset/`).

Reviewer opens localhost:5000, sees crop + candidate tag, hits Y (confirm) / N (no tag) / S (skip) / types correction. Output `verified.jsonl` (append-mode, resumable, per-reviewer via `--output`). Feeds the [[project_ocr_finetune_scaffold]] bubble-detector / v11 training path.

Status 2026-06-08: user said "keep it aside for now." It improves Level-2b tag *assignment*, which is downstream of the current bottlenecks (schedule parsing + detection recall on the retail corpus — see [[project_v10_newcorpus_benchmark]]). Collect any returned `verified.jsonl` as free ground truth, but don't spend new review cycles on it until schedules + detection work on retail plans.
