#!/usr/bin/env python
"""Run the full takeoff pipeline end-to-end on every plan in a folder,
sequentially (one heavy process at a time), and emit a consolidated report.

Metrics are read back from each run's output artifacts (variables.json,
detections.json, the xlsx) rather than scraped from stdout, so they're robust.

Usage:
    python scripts/batch_run_all.py --input text_based_plans --out batch_2026-06-08 --time-budget 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path (moved into scripts/)
# Windows console defaults to cp1252 — any non-ASCII in a print() (→, ·, ✅)
# crashes with UnicodeEncodeError (CLAUDE.md §19.3). Force UTF-8 on the streams.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_json(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _flatten_detections(data):
    """Return a flat list of detection dicts from detections.json.

    Real format written by takeoff_cli:
        {"pdf":..., "dpi":..., "pages": {"1": [det, ...], "2": [...]}}
    where each det is {cls, tag, tag_method, conf, x1,y1,x2,y2}. Also tolerates a
    few older/alt shapes."""
    dets = []
    if data is None:
        return dets
    if isinstance(data, dict) and "pages" in data:
        pages = data["pages"]
        page_iter = pages.values() if isinstance(pages, dict) else pages
        for pg in page_iter:
            if isinstance(pg, list):
                dets.extend(pg)
            elif isinstance(pg, dict) and "detections" in pg:
                dets.extend(pg["detections"])
        return [d for d in dets if isinstance(d, dict)]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and ("cls" in item or "class" in item):
                dets.append(item)              # already a detection
            elif isinstance(item, dict) and "detections" in item:
                dets.extend(item["detections"])  # {page, detections:[...]}
            elif isinstance(item, list):
                dets.extend(item)              # list-of-pages-of-lists
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                dets.extend(v)
    return [d for d in dets if isinstance(d, dict)]


def run_one(pdf: Path, out_root: Path, model: str, conf: float, time_budget, out_name=None):
    # out_name lets recursive (per-project) runs key the output folder by project
    # name, since plan filenames can collide across projects (MECHANICAL.pdf...).
    out_dir = out_root / (out_name or pdf.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_run.log"

    cmd = [sys.executable, str(REPO / "takeoff_cli.py"), str(pdf),
           "--model", model, "--conf", str(conf),
           "--time-budget", str(time_budget), "--output-dir", str(out_dir)]

    t0 = time.time()
    status = "ok"
    err_tail = ""
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  cwd=str(REPO), timeout=None)
        if proc.returncode != 0:
            status = f"exit_{proc.returncode}"
    except Exception as e:
        status = "crashed"
        err_tail = str(e)[:300]
    runtime = time.time() - t0
    return compute_metrics(pdf, out_dir, status, runtime, err_tail)


def compute_metrics(pdf: Path, out_dir: Path, status: str, runtime, err_tail=""):
    """Derive the per-plan report row from the run's output artifacts. Used both
    live (run_one) and in --report-only regeneration."""
    stem = pdf.stem
    variables = _load_json(out_dir / f"{stem}_variables.json") or []
    dets = _flatten_detections(_load_json(out_dir / f"{stem}_detections.json"))
    xlsx = out_dir / f"{stem}_takeoff.xlsx"
    annotated = out_dir / f"{stem}_annotated.pdf"

    n_sched = len(variables) if isinstance(variables, list) else 0
    n_det = len(dets)
    n_tagged = sum(1 for d in dets if d.get("tag"))
    tagged_pct = round(100.0 * n_tagged / n_det, 1) if n_det else 0.0
    distinct_tags = sorted({d.get("tag") for d in dets if d.get("tag")})

    if status == "ok" and not xlsx.exists():
        status = "no_xlsx"          # ran clean but produced no takeoff (0 detections)

    return {
        "plan": pdf.name,
        "size_mb": round(pdf.stat().st_size / 1048576, 1),
        "status": status,
        "sched_tags": n_sched,
        "detections": n_det,
        "tagged": n_tagged,
        "tagged_pct": tagged_pct,
        "distinct_tags": len(distinct_tags),
        "xlsx": xlsx.exists(),
        "annotated": annotated.exists(),
        "runtime_s": round(runtime, 1) if runtime is not None else None,
        "sample_tags": distinct_tags[:12],
        "error": err_tail,
    }


def regenerate(in_dir: Path, out_root: Path):
    """Rebuild the report from already-produced artifacts (no re-running). Pulls
    each plan's runtime from the existing batch_results.csv when present."""
    import csv
    runtimes = {}
    statuses = {}
    csv_path = out_root / "batch_results.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                runtimes[row["plan"]] = float(row["runtime_s"]) if row.get("runtime_s") else None
                statuses[row["plan"]] = row.get("status", "ok")
    pdfs = sorted(in_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
    results = []
    for pdf in pdfs:
        out_dir = out_root / pdf.stem
        if not out_dir.exists():
            continue
        st = statuses.get(pdf.name, "ok")
        # a recorded crash/exit stays; otherwise infer from artifacts in compute_metrics
        st = st if st not in ("ok", "no_xlsx") else "ok"
        results.append(compute_metrics(pdf, out_dir, st, runtimes.get(pdf.name)))
    total_rt = sum(r["runtime_s"] or 0 for r in results)
    write_report(results, out_root, time.time() - total_rt, len(results))
    print(f"Regenerated report for {len(results)} plan(s): {out_root / 'batch_report.md'}",
          flush=True)
    return results


# ─── Plan selection for the nested "data to train" project layout ──────────
# Each project folder holds the raw plan plus derived/auxiliary PDFs. We run the
# RAW plan and skip our/team outputs and side documents.
_PLAN_SKIP_SUBSTR = ("annotated", "specification", "_variables",
                     "_project_info", "_detections")
_PLAN_SKIP_PREFIX = ("takeoff", "schedule-", "schedule ", "schedule_")
_PLAN_PREFER_KW = ("mechanical", "submittal", "drawings", "hvac", "plan", "final")


def pick_plan_pdf(project_dir: Path):
    """Choose the single plan PDF to run for a project folder. Returns
    (chosen_path | None, [all_pdf_names]). Heuristic, so every pick is logged to
    plan_selection.csv for auditing."""
    all_pdfs = sorted(project_dir.rglob("*.pdf"))
    cands = []
    for p in all_pdfs:
        nl = p.name.lower()
        if any(s in nl for s in _PLAN_SKIP_SUBSTR):
            continue
        if any(nl.startswith(pre) for pre in _PLAN_SKIP_PREFIX):
            continue
        cands.append(p)
    if not cands:
        cands = list(all_pdfs)          # nothing survived filters → fall back
    if not cands:
        return None, []

    def score(p: Path):
        nl = p.name.lower()
        kw = any(k in nl for k in _PLAN_PREFER_KW)
        return (1 if kw else 0, p.stat().st_size)   # prefer keyword, then largest

    cands.sort(key=score, reverse=True)
    return cands[0], [p.name for p in all_pdfs]


def gather_project_plans(root: Path, out_root: Path):
    """Walk immediate subfolders of `root`, pick one plan PDF each, log the
    selection. Returns a list of (project_name, plan_path)."""
    import csv
    picks = []
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        plan, allnames = pick_plan_pdf(d)
        if plan is None:
            rows.append([d.name, "(no PDF found)", 0, 0, ""])
            continue
        picks.append((d.name, plan))
        rows.append([d.name, plan.name, round(plan.stat().st_size / 1048576, 1),
                     len(allnames), " | ".join(allnames)])
    with open(out_root / "plan_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["project", "picked_pdf", "size_mb", "n_pdfs_in_project", "all_pdfs"])
        w.writerows(rows)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="text_based_plans")
    ap.add_argument("--out", default="batch_run")
    ap.add_argument("--model", default="models/hvac_yolov8s_v10.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--time-budget", type=float, default=0)
    ap.add_argument("--report-only", action="store_true",
                    help="Rebuild the report from existing artifacts; do not re-run any takeoff.")
    ap.add_argument("--recursive", action="store_true",
                    help="Treat --input as a folder of PROJECT subfolders; pick one plan "
                         "PDF per project (logged to plan_selection.csv).")
    args = ap.parse_args()

    in_dir = (REPO / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input)
    out_root = (REPO / args.out).resolve() if not os.path.isabs(args.out) else Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        regenerate(in_dir, out_root)
        return

    # Build the work list: (label, plan_pdf). Recursive = one plan per project.
    if args.recursive:
        picks = gather_project_plans(in_dir, out_root)
        picks.sort(key=lambda t: t[1].stat().st_size)            # small first
        work = picks
        print(f"Recursive batch over {len(work)} project(s) from {in_dir}", flush=True)
        print(f"Plan selection logged → {out_root / 'plan_selection.csv'}", flush=True)
    else:
        pdfs = sorted(in_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
        work = [(p.stem, p) for p in pdfs]
        print(f"Batch over {len(work)} plan(s) from {in_dir}", flush=True)
    print(f"Output → {out_root}", flush=True)

    results = []
    batch_t0 = time.time()
    for i, (label, pdf) in enumerate(work, 1):
        print(f"\n[{i}/{len(work)}] {label}  ({pdf.name}, {pdf.stat().st_size/1048576:.1f} MB) ...",
              flush=True)
        r = run_one(pdf, out_root, args.model, args.conf, args.time_budget, out_name=label)
        r["project"] = label
        results.append(r)
        print(f"    -> {r['status']}  sched={r['sched_tags']}  det={r['detections']}  "
              f"tagged={r['tagged']} ({r['tagged_pct']}%)  xlsx={r['xlsx']}  "
              f"{r['runtime_s']}s", flush=True)
        # Write the report after EVERY plan so partial progress survives a crash.
        write_report(results, out_root, batch_t0, len(work))

    print(f"\nBatch done in {time.time()-batch_t0:.0f}s. Report: "
          f"{out_root / 'batch_report.md'}", flush=True)


def write_report(results, out_root, batch_t0, total):
    # CSV
    import csv
    has_proj = any(r.get("project") for r in results)
    cols = (["project"] if has_proj else []) + [
            "plan", "size_mb", "status", "sched_tags", "detections", "tagged",
            "tagged_pct", "distinct_tags", "xlsx", "annotated", "runtime_s"]
    with open(out_root / "batch_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Markdown
    done = len(results)
    ok = sum(1 for r in results if r["xlsx"])
    zero = sum(1 for r in results if r["status"] == "no_xlsx")
    crashed = sum(1 for r in results if r["status"] not in ("ok", "no_xlsx"))
    tot_det = sum(r["detections"] for r in results)
    tot_tag = sum(r["tagged"] for r in results)
    overall_pct = round(100.0 * tot_tag / tot_det, 1) if tot_det else 0.0
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# HVAC Takeoff — Batch Report")
    lines.append("")
    lines.append(f"Generated: {ts}  ·  Plans: {done}/{total}  ·  "
                 f"Elapsed: {time.time()-batch_t0:.0f}s")
    lines.append("")
    lines.append(f"- **Produced a takeoff (xlsx):** {ok}/{done}")
    lines.append(f"- **Ran but 0 detections (no xlsx):** {zero}")
    lines.append(f"- **Crashed / nonzero exit:** {crashed}")
    lines.append(f"- **Total detections:** {tot_det}  ·  **Tagged:** {tot_tag} "
                 f"({overall_pct}% overall)")
    lines.append("")
    label_hdr = "Project" if has_proj else "Plan"
    lines.append(f"| {label_hdr} | MB | Status | Sched tags | Detections | Tagged | % | xlsx | Runtime |")
    lines.append("|---|--:|---|--:|--:|--:|--:|:--:|--:|")
    for r in results:
        x = "✅" if r["xlsx"] else "—"
        label = r.get("project") or r["plan"]
        lines.append(f"| {label} | {r['size_mb']} | {r['status']} | "
                     f"{r['sched_tags']} | {r['detections']} | {r['tagged']} | "
                     f"{r['tagged_pct']} | {x} | {r['runtime_s']}s |")
    lines.append("")
    lines.append("## Per-plan sample tags")
    lines.append("")
    for r in results:
        st = ", ".join(str(t) for t in r["sample_tags"]) or "(none)"
        lines.append(f"- **{r.get('project') or r['plan']}** — {st}")
        if r["error"]:
            lines.append(f"    - error: `{r['error']}`")
    lines.append("")

    with open(out_root / "batch_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
