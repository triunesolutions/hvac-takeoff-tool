# Architecture & Accuracy Strategy — 2026-06-10

How to get from today's accuracy to "the team trusts the count." Grounded in the
repo's own evidence: the four benchmark docs, the post-fix 157-project batch report
(`batch_datatrain_local/batch_report.md`, 2026-06-09), six projects of human-verified
ground truth (625 detections), and the v11 retrain post-mortem (commit `cbdb4ee`).

---

## 1. Verdict on the architecture

**The architecture is right.** Schedule-first (tags + properties as source of truth)
→ YOLO detection on plan pages → tag inference → Excel BOM is the correct
decomposition. What keeps changing is the weakest link — and the evidence says the
current weakest link is **tag inference**, not the model, and no longer (primarily)
runtime.

The evidence chain:

- **Detection is good where the pipeline runs cleanly.** v10 hits 100% product
  recall on HM SSC Vein Clinic, 78% Erewhon, 72% BMO Santee, ~80% median precision
  on text plans (`docs/text_based_accuracy_2026-06-04.md`). Human review of 625
  detections across 6 projects: 76% accepted, 14.2% relabeled, 9.8% phantoms
  (`ground_truth/*/ls_summary.txt`).
- **Timeouts are fixed.** The June-9 batch ran 157/157 projects, 0 crashes, runtimes
  comfortably inside budget — the time-budget work (commits c2b2824…ed9336b) landed.
- **Tag inference is now the gate.** Post-fix, only **14.5% of 8,604 detections got
  a tag**; 68 of 157 projects sit at 0% tagging. The dominant new pattern is
  projects with healthy schedule tags AND healthy detections but **zero tagged**:
  Fort Totten (7 tags / 43 dets / 0 tagged), AEP Pompano (24/58/0), Eide Ford
  (12/112/0), SPD Renovation (17/50/0), Space Shuttle Exhibit (44/14/0).
- **Schedule parsing still loses a long tail** even with the OCR fallback: 0–1 tag
  projects remain (Copper Ranch, Hope Chapel, Anaheim 82, Skytron, Free People),
  plus the manufacturer-as-tag convention (677 Imperial, Krispy Kreme).
- **A third cluster is page selection, not the model:** schedule parses fine but
  detections ≈ 0 (Alliance Medical 123 tags / 3 dets, Acushnet 133/0, UCLA Health
  Oncology 55/0). `find_mechanical_pages()` likely picked the wrong pages.

## 2. Should we retrain the model now? **No.**

Three hard reasons:

1. **It was already tried and rolled back.** v11 (warm-start from v10 + the 315
   verified tiles, 1% of the dataset) regressed across the board: accepted
   475 → 289 (−39%), phantoms 3.2× (commit `cbdb4ee`, `scripts/benchmark_v11.py`
   kept for the retry).
2. **The review data can't teach recall.** Across all 6 reviewed projects there are
   **zero "added" boxes** — reviewers only judged v10's own outputs. The ground
   truth is a precision audit; it contains no information about symbols the model
   missed.
3. **The benchmarks all converge on the same conclusion:** detection counts are
   roughly right; recall is gated downstream ("in every v10 row, tagged% == recall"
   — `docs/benchmark_newcorpus_2026-06-03.md`). Retraining attacks the stage that
   is already working.

**When a v12 retrain becomes worthwhile** (not before all four):
- ~10× more verified tiles (≈30–50 reviewed projects through the Label Studio loop);
- reviewers instructed to **add missed boxes**, so false negatives enter the data;
- cold-start training (warm-start anchored v11 to v10's mistakes);
- legend/schedule/detail pages included as explicit negatives.

**The one model problem worth attacking sooner: AD-GRD.** It is both the confusion
magnet (88 of 89 relabels are AD-GRD → AD-T-BAR SUPPLY) and the phantom magnet
(29 of 61 deletions). Until v12, handle it in post-processing: class-family
aliasing for tag lookup (already partially built in `class_aliases.py` /
`YOLO_CLASS_ALIASES`) and possibly a per-class confidence threshold.

## 3. The plan — four workstreams, priority order

### WS1 — Tag inference (current gate; biggest expected gain)

1. **Instrument before fixing.** Record per-level outcome stats (L1 / 2a / 2b' / 2b
   tagged counts; for each untagged detection: nearest-valid-tag distance, and
   whether its class had ANY schedule tags) into `detections.json` and the run
   summary. The 68 zero-tag projects are currently a guess; this makes the dominant
   sub-cause measurable in one batch run.
2. **Generalize class equivalence.** Today `class_to_tags` effectively requires the
   YOLO class name to equal the schedule-inferred class. Mismatches (FAN vs EXHAUST
   FAN, AD-GRD vs AD-T-BAR SUPPLY, SPLIT SYSTEM vs CONDENSING UNIT) silently empty
   the candidate tag list, so no level can ever fire. Replace exact matching with
   symmetric class families (extend `YOLO_CLASS_ALIASES` in `tag_inference.py`).
   This is the leading hypothesis for the tags-and-dets-but-0-tagged cluster.
3. **`max_distance` 350 → 600** in `level2b_bubble_detect` (`tag_inference.py:459`)
   — expose as a CLI flag, validate with `benchmark_samples.py` before changing the
   default (a wider radius can mis-tag dense plans).
4. **Bubble-OCR crop preprocessing** (3× upscale + Otsu binarize before EasyOCR) —
   already proven in `label_tag_bubbles_ocr.py`; port it to `tag_matcher.py`.

### WS2 — Schedule parser long tail

1. **Manufacturer-as-tag schedules** (Imperial / Krispy Kreme): when no MARK column
   exists but MANUFACTURER/MODEL does, key tags off the manufacturer to match the
   team's own takeoff convention.
2. **Diagnose the remaining 0-tag projects** from the batch report (Hope Chapel,
   Anaheim 82, Skytron, Free People) — likely OCR-fallback token filters
   (`_ocr_tag_from_token`) being too strict, or fragmented/horizontal tables.
3. **Single-letter tags in the headerless fallback** (`schedule_parser.py:523-530`)
   — ✅ **DONE** (2026-06-10): the fallback now accepts bare `A`/`B` marks that
   `normalize_tag` accepts.
4. **Generic-GRD service inference steals fan/equipment tags** — discovered via
   WS1.1 instrumentation on Circle K Olmito (20 schedule tags, 8 detections, **all
   8 candidate-starved**). `_infer_yolo_class_from_service` lists `'EXHAUST'` in
   its generic-diffuser keyword set, so a fan-schedule row with SERVICE="EXHAUST"
   returns `AD-GRD` and pre-empts the correct `EF/KEF → EXHAUST FAN` tag-prefix
   logic. Fan tags land under `AD-GRD`, so no FAN/EXHAUST FAN detection can ever
   match them. Fix candidate: when service inference yields the *generic* `AD-GRD`
   but the row's tag prefix maps to a *specific* equipment class (FAN, EXHAUST
   FAN, RTU, …), prefer the tag prefix. This is a core class-inference priority
   change — validate on `benchmark_samples.py` + the batch before shipping (it
   can shift many projects' classifications). Likely a bigger lever than WS1.4.

### WS3 — Page selection (cheap; unblocks the tags-but-no-detections cluster)

Log selected pages + selection reason per run; diagnose `find_mechanical_pages()`
on Alliance Medical / Acushnet / UCLA-type projects against
`batch_datatrain_local/plan_selection.csv`. Add a fallback that scores pages by
drawing density when keyword matching finds nothing plan-like.

### WS4 — Measurement (stop flying blind)

1. The batch report's "tagged%" is not recall. **`ground_truth.jsonl` (26,844
   equipment records with bboxes across ~180 projects, parsed from the team's own
   Bluebeam takeoffs) is a recall-capable truth source.** Extend the benchmark to
   score detections against it per project — count-based first, IoU later. That
   turns all 157 batch projects into scored projects instead of ~35.
2. Make the per-stage funnel (schedule tags → detections → tagged → matched-truth)
   the standard report row, so every change is attributable to a stage.

### Deferred (explicitly not now)

v12 retrain (criteria in §2), larger detector architectures (RT-DETR / YOLO-m),
VLM-assisted detection, cross-page NMS, `normalize_tag` refactor, big-PDF
streaming (St Elizabeth class).

## 4. Expected impact

Rough sizing from the failure-mode ledger (fractions of projects in each bucket,
157-project batch + benchmark docs):

| Workstream | Failure bucket it attacks | Projects affected |
|---|---|---|
| WS1 class families + radius | tags + dets present, 0 tagged | ~40% of batch |
| WS2 parser long tail | 0–1 schedule tags | ~20% |
| WS3 page selection | tags fine, ~0 detections | ~10% |
| WS4 measurement | makes the other three provable | all |

The honest framing on "perfect count": the realistic near-term target is a
high-recall, high-precision *assisted* count — the tool gets the bulk right, the
estimator corrects the rest in review, and (via the Label Studio loop) every
correction compounds into v12. The four workstreams above are sequenced to move the
median, which is where the product lives or dies.

## 5. Validation protocol for every change

1. `python benchmark_samples.py` (sample corpus) — regression bar; Flex 230 must
   not move.
2. Re-run the 157-project batch (`scripts/` batch runner) — compare funnel stats
   per stage, not just the headline tagged%.
3. Ship each WS1/WS2 change as a separate commit with before/after numbers in the
   commit message (existing team convention).
