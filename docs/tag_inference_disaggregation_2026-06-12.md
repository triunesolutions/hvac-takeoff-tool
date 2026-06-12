# Tag-Inference Failure Disaggregation — 2026-06-12

Full-batch verdict from the WS1-instrumented run (`batch_ws1_instrumented/`,
157/157 projects, "data to train/projects" corpus, `--time-budget 240`).
This **overturns the 58/157 interim read** that said the class-equivalence gap
dominates. With all 157 projects and a corrected gate, the schedule parser is
the bottleneck — not class families, and not (primarily) OCR.

## Headline

- **10,060 detections, 1,652 tagged = 16.4%.** 8,408 untagged.
- Only **40/157 projects produced any tag**; **43/157 parsed zero schedule tags.**

## Where the untagged mass goes (8,408 detections)

| Bucket | Count | % of untagged | Owner |
|---|---|---|---|
| **Schedule parser** (sparse 1–9 tags + zero tags) | **4,738** | **56.4%** | **WS2** |
| OCR / distance (`has_candidates_unmatched`) | 2,606 | 31.0% | WS1.3 / WS1.4 |
| **TRUE class-gap** (healthy schedule ≥10 tags, still no candidate) | **1,064** | **12.7%** | WS1.2 families / WS2.4 |

Sub-split of the 5,802 `no_candidate` detections, gated on schedule **health**:

| Sub-bucket | Count | % of untagged |
|---|---|---|
| zero parse (sched = 0) | 2,115 | 25.2% |
| sparse parse (sched 1–9) | 2,623 | 31.2% |
| true class-gap (sched ≥ 10) | 1,064 | 12.7% |

## Why the interim read was wrong

The 58/157 note used a crude `sched_tags > 0` gate and concluded ~65–75% of
untagged = class gap. That bucket was inflated by projects where the schedule
parser extracted almost nothing yet the plan had hundreds of detections —
e.g. **Yucaipa BldgA** (sched=4, det=620), **Rice Moody** (sched=1, det=246),
**Air Products** (sched=1, det=194). Those are parser *under-extraction*, not
class-equivalence failures. Gating on schedule health (≥10 tags = a real parse)
moves that mass to WS2 where it belongs and collapses the "class gap" to 12.7%.

## Corrected priority (reverses the strategy doc's order)

1. **WS2 — schedule parser (56%).** The median mover. Half is total misses
   (sched=0): Roots Church, Residence Inn El Paso, Aaron Packaging,
   Texas Roadhouse Bastrop, Erewhon, Total Wine #1149. Half is sparse parses
   (1–9 tags on plans with 100s of detections): Yucaipa BldgA, Rice Moody,
   Air Products, new_office, Hartwood at Windstone.
2. **WS1.3 / WS1.4 — OCR/distance (31%).** Bigger than class families by
   2.5×. The bubble-OCR preprocessing (WS1.4, shipped) and `--bubble-max-dist`
   (WS1.3) attack this; it is NOT the minor lever the interim note claimed.
3. **WS1.2 / WS2.4 — class-gap (12.7%).** Real but smallest. Genuine cases
   (healthy schedule, residual no_candidate): Yucaipa BldgB (sched=40, 368
   tagged, 209 residual), JFK ITB (sched=50, 168 residual), Aritzia Americana
   (sched=50, **0** tagged → likely the WS2.4 EXHAUST-keyword class-inference
   bug), Pelican Commerce, Fendi Fashion Valley.

## Caveat on the corpus

This batch ran the full "data to train" set, including files far above the
≤15–20 MB working ceiling (up to 167 MB). The largest files (≥100 MB) mostly
returned det=0/no_xlsx (page-selection or render limits), which depresses the
headline but does not change the *relative* bucket sizes — those are dominated
by the small/mid plans that detect cleanly.

## Reproduce

```
python -X utf8 scripts/batch_run_all.py \
  --input "<corpus>/projects" --out batch_ws1_instrumented \
  --recursive --time-budget 240 --resume
# then the disaggregation: see this doc's git history / session 2026-06-12
```
The per-project numbers are in `batch_ws1_instrumented/batch_results.csv`
(columns `sched_tags, detections, tagged, untag_no_cand, untag_unmatched`).
That CSV is gitignored (heavy-artifact rule); this doc is the durable record.
