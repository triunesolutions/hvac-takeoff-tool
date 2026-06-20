# Benchmark — "SAMPLE FILES 02.06.2026" corpus (run 2026-06-20)

**Model:** `hvac_yolov8s_v10.pt` (production default).
**Corpus:** team's `SAMPLE FILES 02.06.2026` (16 projects). 12 benchmarked.
**Why this run:** first look at a corpus that is **text-based schedules**, not the
CID-font / CAD-vector files (Cityvet, CCV) where the schedule text layer is dead.
Run with `benchmark_samples.py --root "<...>/SAMPLE FILES 02.06.2026"
--model models/hvac_yolov8s_v10.pt`.

## Headline

The text-based hypothesis holds: **0 of 16 plans have a dead/CID text layer.** The
schedule parser extracts real tags on 10/12 projects (5–35 tags each) and tag
inference actually fires — a different world from the CID files where tagging was
flat 0%. The ceiling on these files is now **detection recall** and **runtime
robustness**, not schedule parsing.

## Results (12 benchmarked)

| Project | status | team | ours | prod recall | prod prec | tag recall | sched tags |
|---|---|--:|--:|--:|--:|--:|--:|
| Cy-Fair Tire | ok | 31 | 18 | **48%** | 83% | 0% | 6 |
| Living Spaces - Austin TX | ok | 612 | 233 | **35%** | 92% | 34% | 35 |
| Edward Jones Permit Package | ok | 116 | 48 | 32% | 77% | 30% | 0 |
| Circle K Progresso | ok | 100 | 73 | 31% | 42% | 8% | 22 |
| SPE Sinton | ok | 192 | 51 | 22% | 82% | 19% | 35 |
| DXD La Cantera | ok | 254 | 54 | 21% | 98% | 19% | 5 |
| Kaalo Studio | ok | 62 | 24 | 16% | 42% | 0% | 10 |
| Circle K Olmito | ok | 226 | 8 | 3% | 88% | 2% | 20 |
| Buffalo Manor Apartments | no_detections | 273 | 0 | 0% | — | — | 68 |
| Javier Marcos (127-pg) | no_detections | 196 | 0 | 0% | — | — | 0 |
| Hulsey Storage | timeout | 220 | 0 | 0% | — | — | 3 |
| Joslin Med Spa | timeout | 192 | 0 | 0% | — | — | 1 |

**Scored (8): median 26%, mean 26%, max 48%. 4 projects ≥25%.**
Excluded from the 16: 3 plans >20 MB (AGCO Jackson, Flour Mill, HM SSC Vein Clinic
— dropped by the picker's `MAX_PLAN_MB`); 1 missed because its folder is
`Plans Specs` not `Plans_Specs` (B08 RFP) — `discover_projects` matches the exact
name.

## Wins

1. **Schedule parser works on text-based schedules.** 10/12 extract real tags
   (max 35). On Living Spaces it pulled 35 tags → 34% tag recall, 92% precision on
   a 612-item project. This is the capability that was flat-zero on the CID files.
2. **High precision where it scores** (82–98% on 5 of 8 scored). When the tool
   detects + tags, it's usually right — the gap is recall (under-detection), not
   false positives, on those projects.
3. **Best cases are demo-ready-ish:** Cy-Fair 48%, Living Spaces 35% @ 92% prec.

## Issues, in priority order

1. **Runtime failures dominate the bottom (4/12).** 2 timeouts + 2 no-detections,
   all 0%. Root cause is the same one seen on CCV: `parse_pdf_schedules` scans
   **every page** with `pdfplumber.extract_tables()` (~8–20 s/page, pathological
   pages up to ~145 s), which eats the time budget before detection runs. Big
   plans (Javier 127 pp) and slow schedule pages (Joslin, Hulsey) never reach a
   useful detection pass. **This is the single highest-leverage fix** — bound or
   skip the all-pages table scan (see workflow notes) and these 4 zeros likely
   become real scores. Median is dragged down by runtime, not model quality.
2. **Detection recall is the accuracy ceiling on the ones that do run.** High
   precision, low recall = YOLO v10 under-detects (e.g. Olmito 3% recall @ 88%
   prec; Cy-Fair 48% is the best the current model gives). Recall, not schedule
   parsing, is now the WS1 problem on text-based files.
3. **Phantom leakage onto non-plan sheets** on multi-sheet sets. Circle K
   Progresso put **37% of detections on schedule/detail pages** (→ 42% precision).
   `_is_non_plan_sheet()` catches some but leaks on these. Known §19.5 behavior,
   still live.

## Neck size / CFM — on-plan, not in the schedule (extraction opportunity)

The team's takeoff has NECK SIZE / MODULE SIZE / DUCT SIZE / CFM columns. **None of
these are in the schedule table** — the schedule gives `tag → type/model` only
(verified: schedule columns are SERVES, MODEL, CFM-on-equipment, voltage…). The
per-instance neck size is annotated **on the plan, next to each tag bubble**, as a
vertical stack:

```
(A)        <- tag bubble (single letter)
8"ø        <- NECK SIZE: round = N"ø, rectangular = N"xN"  (e.g. 14"x6", 6"x6")
170 CFM    <- airflow; sometimes a "TYP 2" qty multiplier follows
```

Real examples from SPE Sinton M2.11: `(A) 8"ø 170 CFM`, `(E) 6"ø 75 CFM`,
`(C) 6"x6" 50 CFM TYP 2`, `14"x6" 340 CFM`.

**This is a cheap win:** Level-2b bubble-OCR already crops and OCRs this exact
region to read the tag — it just discards the size/CFM tokens. Extending it to
regex out `\d+"?ø` / `\d+"?x\d+"?` (neck) and `\d+\s*CFM`, binding to the closest
token below the tag bubble, would populate the NECK SIZE + CFM columns with no new
OCR pass. Caveat: duct-run sizes (`18"x16"`) sit nearby — must bind to the nearest
token under the bubble, not any size in the crop. This is the unbuilt Phase-3
"value extraction" item, but cheaper than it reads because the OCR already happens.

## Workflow notes (carry forward)

- **`--time-budget 0` is harmful on multi-page real plans.** It disables the cap
  that bounds the all-pages schedule scan; the scan then runs unbounded (CCV: 8+
  min, one page = 145 s). Use a real budget. The benchmark's own 450 s is correct.
- **`--pages` bounds *detection* only, not schedule parsing.** Schedule parse always
  scans the whole document. To make multi-page plans fast you need both a page
  subset for detection *and* a bounded schedule scan.
- **`detections.json` page keys are 0-indexed `page_idx`** (`doc[page_idx]`); the
  Excel "Pages: N" column is `page_idx + 1`. (Noted because an off-page-by-one in
  an external audit script produced a false "detected the legend" alarm — there is
  **no wrong-page bug**; detection lands on the correct plan page. The real issue
  is non-plan phantom *leakage* per Issue 3.)

## Suggested next steps

1. Bound/skip the all-pages schedule table scan → recover the 4 runtime zeros.
2. Neck-size + CFM extraction off the existing bubble-OCR crop (cheap, high
   demo value — fills columns the team expects).
3. Tighten `_is_non_plan_sheet()` to cut Progresso-style phantom leakage.
4. Detection recall (WS1) remains the model-side ceiling; out of scope for a
   parser/workflow pass.
