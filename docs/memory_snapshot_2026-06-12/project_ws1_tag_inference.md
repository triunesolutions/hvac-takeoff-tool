---
name: project_ws1_tag_inference
description: "Accuracy push after 2026-06-10 code review: WS1 tag-inference fixes shipped; instrumented batch paused 58/157 — class gap dominates untagged mass"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---

2026-06-10 accuracy push on `hvac-takeoff-tool` (all on `master`, pushed). Builds on [[project_datatrain_batch]].

**Two strategy docs now in repo** (merged to master): `docs/repo_review_2026-06-10.md` (findings, mostly actioned) and `docs/architecture_strategy_2026-06-10.md` — the 4-workstream accuracy roadmap. Verdict there: architecture is right; **tag inference is the gate** (not the model — v11 retrain already regressed, rolled back). Do NOT retrain until ~10× more verified tiles + reviewers add missed boxes.

**Shipped this session (commits on master):**
- Full code-review fixes: PDF handle leaks, brand/model split-at-first-digit, headerless bare-letter marks, `TA` prefix, LS title clamp, untracked `.claude/sessions/`, `requirements.txt` deps, `HVAC_*` env-var path overrides, `tests/test_normalize_tag.py`.
- **WS1.1** instrument `tag_inference`: every untagged detection labeled `no_candidate_tags` (class gap) vs `has_candidates_unmatched` (OCR/distance); funnel in `detections.json` + run summary.
- **WS1.2** `CLASS_FAMILIES` (FAN↔EXHAUST FAN, CU↔SPLIT SYSTEM, dampers) so name splits stop starving the tag lookup; conservative (auto-levels resolve a sibling only when exactly one has tags). Cy-Fair 6→7, Flex 230 held 29/29.
- **WS1.3** `--bubble-max-dist` flag (default 350 unchanged; benchmark before moving).
- **WS1.4** upscale+Otsu-binarize bubble crops before OCR (`tag_matcher.preprocess_crop_for_ocr`). No regression; minor lever per the batch.

**Instrumented batch PAUSED at 58/157** (out `batch_ws1_instrumented/`, gitignored). Resume:
`python -X utf8 scripts/batch_run_all.py --input "C:\Users\JFL\Downloads\Triune\data to train\projects" --out batch_ws1_instrumented --recursive --time-budget 240 --resume`

**CORRECTED full-batch verdict (157/157, 2026-06-12): the SCHEDULE PARSER dominates, NOT the class gap.** The 58/157 "class gap dominates" read was an artifact of a crude `sched_tags>0` gate that lumped parser-under-extraction into "class gap." Gating on schedule HEALTH (≥10 tags = a real parse) flips it. Of 8,408 untagged detections (16.4% tagged overall): **WS2 schedule parser = 56.4%** (zero-parse 25.2% + sparse-parse 31.2%), **OCR/distance = 31.0%**, **TRUE class-gap (sched≥10, still no candidate) = only 12.7%**. Priority is **WS2 first, then WS1.3/1.4 OCR, then WS1.2 families** — the REVERSE of the strategy doc's original order. Full writeup: `docs/tag_inference_disaggregation_2026-06-12.md` (+ preserved CSV/report in docs/). Genuine class-gap poster children (healthy schedule): Yucaipa BldgB (sched40/368tagged/209 residual), Aritzia Americana (sched50/0tagged → likely WS2.4 EXHAUST bug). Acushnet (sched133/91det/0tagged) is a real class-gap outlier but NOT representative of the mass.

**WS2.4 bug found (logged in strategy doc):** `_infer_yolo_class_from_service` lists `'EXHAUST'` in its generic-diffuser keyword set, so fan rows (EF/KEF) get filed under `AD-GRD` and no FAN/EXHAUST FAN detection can ever match them. Fix candidate: prefer the tag-prefix's specific class over the generic AD-GRD service fallback — but it's a core class-inference change, validate on benchmark_samples + the batch first.

**Status 2026-06-12:** batch DONE 157/157; disaggregation DONE (see corrected verdict above). **Next session TODO:** (1) **WS2 schedule parser is now the priority** — attack zero-parse (43 projects, sched=0: Roots Church, Residence Inn El Paso, Aaron Packaging, Texas Roadhouse, Erewhon, Total Wine) + sparse-parse (Yucaipa BldgA sched=4/det=620, Rice Moody, Air Products); (2) WS2.4 EXHAUST-keyword class bug (Aritzia poster child); (3) schedule-OCR cap leak still unfixed (#1 runtime debt — small plans hit 240s on OCR fallback). **NOTE: laptop wiped after 2026-06-12 session** — repo is the source of truth; memory snapshotted to `docs/memory_snapshot_2026-06-12/` and session handoff in `docs/SESSION_STATE_2026-06-12.md`.
