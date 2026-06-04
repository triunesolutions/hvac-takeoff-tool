# Text-Based Plan Accuracy — v10 (2026-06-04)

**Corpus:** `SAMPLE FILES 02.06.2026` (16 projects). Plan PDFs classified as text-layer vs vector by median words/page (text plans run 300–2900/page; vector ~0–90).
**Model:** `models/hvac_yolov8s_v10.pt` (production default). **Not** in v10's training set — genuine holdout (June files).
**Scored:** product recall/precision vs each project's `Completed Takeoff/Takeoff_*.xlsx`, same overlap metric as `benchmark_samples.py`.
**Scope:** 11 deduped text-based mechanical drawings (excluded vector plans, spec books, WORKING_ dupes). Just ran v10 + scored — not a tuned benchmark.

## Headline

**Median product recall: 33% · Max: 100% · Median precision: ~80%.**

This is materially higher than the 7% median on the May-26 mixed corpus — because that set included vector plans whose schedules don't parse. On text-based plans (where the schedule parser works natively), the pipeline does much better. Confirms the core thesis: **vector schedules are the bottleneck, not the model.**

## Per-plan

| Plan | Size | Recall | Precision | team / ours | dets / tagged |
|---|---:|---:|---:|---:|---:|
| HM SSC Vein Clinic | 24 MB | 100% | 81% | 34 / 42 | 42 / 0 |
| AGCO Jackson | 27 MB | 80% | 48% | 81 / 136 | 136 / 0 |
| Javier Marcos | 125 MB | 69% | 73% | 196 / 184 | 184 / 3 |
| Cy-Fair Tire | 0.5 MB | 48% | 83% | 31 / 18 | 18 / 6 |
| Living Spaces | 4 MB | 36% | 87% | 612 / 252 | 252 / 125 |
| Circle K Progresso | 6 MB | 31% | 42% | 100 / 73 | 73 / 32 |
| SPE Sinton | 3 MB | 22% | 82% | 192 / 51 | 51 / 34 |
| Kaalo Studio | 1 MB | 16% | 42% | 62 / 24 | 24 / 0 |
| Circle K Olmito | 2 MB | 3% | 88% | 226 / 8 | 8 / 0 |
| Flour Mill | 36 MB | 2% | 80% | 500 / 15 | 15 / 0 |
| Buffalo Manor | 3 MB | — | — | — / 0 | 0 / — |

## Reading it

- **Precision is solid (~80% median)** — when we emit a product it's usually right. We lose on **recall** (missed equipment), not false-positive flooding.
- **Tagging is NOT required for product recall here** — AGCO (80%) and HM SSC (100%) tagged 0 but still scored, because the product label comes from the YOLO class and matched the team's products directly. (Contrast the May-26 corpus where tagged% == recall.) The tag→product dependency is project-specific.
- **The large plans tagged ~0** (AGCO, Flour Mill, HM SSC, Javier): the 900s budget was consumed by detection across 45–103 pages, so tag-inference hit the deadline and was skipped. Detection counts are healthy; tagging just didn't get to run. **Per-stage budgeting (don't let detection starve tag inference) is a fixable pipeline issue.**
- **Bottom-half failures** are the path up, not the top performers:
  - **Flour Mill 2%** (15 of 500) and **Circle K Olmito 3%** (8 of 226) — severe under-detection. Flour Mill is a 73-page/36 MB plan → likely budget/page-cap truncation.
  - **Buffalo Manor 0 detections** — page filter selected page 4 and v10 found nothing. Genuine miss (wrong page or unknown symbols).

## Next levers (in priority order)
1. **Per-stage time budgeting** so large text plans (AGCO/Flour Mill/Javier) finish tag inference instead of truncating.
2. **Under-detection on Flour Mill / Olmito** — page selection + page cap on long plans.
3. **Buffalo Manor 0-detection** — diagnose page filter / symbol coverage.

Outputs: `text_based_v10_output/<project>/` (local). v9/v10 annotated comparison: `annotated_compare/`.
