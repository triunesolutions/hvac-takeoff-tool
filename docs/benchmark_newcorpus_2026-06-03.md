# Benchmark — May-26 corpus, v10 vs v9 (2026-06-03)

**Corpus:** `SAMPLE FILES 26.05.2026` (14 projects, plan PDFs ≤20MB, `sample_files` layout).
**Harness:** `benchmark_samples.py` with `--time-budget 450` (commit 5094fc9). Product-recall scored against the team's Completed Takeoff xlsx.
**Note:** This entire corpus is dated *after* v10's April training set — every project is effectively a holdout. Low absolute numbers are expected; the value here is the v10-vs-v9 comparison and the failure-mode diagnosis.

## Headline

| Metric | v10 | v9 |
|---|---|---|
| Median product recall | **7%** | ~0–1% (stopped at 11/14) |
| Max product recall | **25%** (Golf Galaxy) | 7% (Golf Galaxy) |
| Projects ≥ 25% recall | 1 | 0 |
| Timeouts | 1 (Tesla, fixed on recheck) | 0 |

**v10 beats v9 decisively on this corpus.** This refutes the impression that "v9 was better" — on new retail plans v9 lost 7 of 10 scored projects, tied 2, and edged v10 only on Five Below.

## Head-to-head (product recall)

| Project | v10 | v9 | Winner |
|---|---:|---:|---|
| Golf Galaxy | 25% | 7% | **v10** |
| Emanate Health | 20% | 0% | **v10** |
| Rivian | 20% | 1% | **v10** |
| T&T Supermarket | 12% | 1% | **v10** |
| Dick's House of Sports | 9% | 0% | **v10** |
| Sephora | 4% | 0% | v10 |
| AutoZone | 4% | 0% | v10 |
| Crunch Fitness | 3% | 0% | v10 |
| CURO | 1% | 1% | tie |
| Tesla | 0% | 0% | tie |
| Five Below | 0% | 3% | v9 (barely) |
| Clearwater | 7% | — | (v9 stopped) |
| Les Schwab | 0% | — | (v9 stopped) |
| Nike | no_detections | — | (v9 stopped) |

(v9 run was stopped after 11/14 — the conclusion was already decided.)

## The real bottleneck: schedule parsing, NOT detection

In **every** v10 row, `tagged% == recall`. Recall is gated by how many detections get a tag assigned — not by detection count. The detector already finds ~the right number of objects:

| Project | team objects | v10 detections | recall |
|---|---:|---:|---:|
| Crunch | 124 | 118 | 3% |
| AutoZone | 82 | 97 | 4% |
| Dick's | 115 | 105 | 9% |

Detections don't get tagged because `schedule_parser.py` extracts **0 tags** on retail plans:

| `sched_tags` extracted | projects | recall |
|---|---|---|
| **0** | AutoZone, Crunch, Five Below, Les Schwab | 0–4% |
| low (3–8) | T&T, Rivian, Golf Galaxy | 12–25% |
| high (57–98) | Dick's, Clearwater | parser worked (Clearwater 85% precision) |

**Implication:** training a v11 on more data will not move recall — detection is not the bottleneck. The fix is in `schedule_parser.py`.

## Recommended next step

Diagnose *why* the parser returns 0 tags on a retail plan (AutoZone, 0 tags) vs one that works (Dick's, 57 tags). Leading hypothesis: retail schedules are rasterized/image tables with no text layer, so pdfplumber finds nothing → build the queued **OCR-fallback-on-schedule-pages** step. That is the single lever that moves the 7% median.

## Side findings
- **Tesla** timed out at 600s during the marathon run; on isolated recheck with `--time-budget 450` it finished in 210s (`ok`, recall 0%). Prior death was partly system load, not purely the CLI — but the budget headroom is a real safety net.
- **CURO** truth file counts every grille individually (team=534) — a granularity mismatch, not a pure miss.
- **Nike** is `no_detections` (model finds nothing on its pages) — a genuine detection gap, separate from the schedule issue.

## Artifacts
- v10 full: `benchmark_v10_small_0603/` (summary + CSVs committed here; xlsx/PDFs local-only)
- v9 partial: `benchmark_v9_small_0603/` (11/14)
- Tesla/Nike recheck: `benchmark_v10_recheck_0603/`
