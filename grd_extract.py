"""
Standalone GRD (Grille / Register / Diffuser) extractor — production wiring.

Runs the pure text-layer diffuser extractor against the mechanical plan pages of
a single PDF and writes an honest BOM. Unlike the benchmark harness, this does
NOT read the answer-key takeoff to learn valid marks — it derives them the way
the real pipeline must: from the drawing's own schedules via
parse_pdf_schedules(). That keeps the numbers it prints meaningful outside of a
benchmark.

Pipeline
--------
  1. find_mechanical_pages(pdf)        → plan-page indices (fallback: all pages)
  2. parse_pdf_schedules(pdf)          → valid_marks + mark_details (BOM join)
  3. extract_diffuser_instances(...)   → per-instance mark / neck / cfm
  4. aggregate_diffuser_bom(...)       → grouped BOM, joined with schedule props

Outputs (next to the PDF, or under --out):
  <stem>_grd_instances.json   every extracted instance + page + bbox + method
  <stem>_grd_bom.csv          grouped BOM (mark, neck, qty, total_cfm, props)
  <stem>_grd_meta.json        run metadata: page counts, style, warnings, marks

Usage
-----
  python grd_extract.py "path/to/plan.pdf"
  python grd_extract.py "path/to/plan.pdf" --pages 4 5 6      # 1-indexed
  python grd_extract.py "path/to/plan.pdf" --ocr             # enable OCR supplement
  python grd_extract.py "path/to/plan.pdf" --no-schedule     # accept every regex hit
  python grd_extract.py "path/to/plan.pdf" --out grd_output/myproj
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from diffuser_extractor import (
    extract_diffuser_instances,
    aggregate_diffuser_bom,
    detect_drawing_style,
    display_neck_size,
)


def _resolve_plan_pages(pdf_path, explicit_pages):
    """Return a list of 0-indexed plan-page indices.

    Priority: explicit --pages (1-indexed → 0-indexed) > find_mechanical_pages()
    > every page in the document.
    """
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        total = doc.page_count

    if explicit_pages:
        return [p - 1 for p in explicit_pages if 1 <= p <= total], total

    try:
        from takeoff_cli import find_mechanical_pages
        plan = find_mechanical_pages(str(pdf_path))
    except Exception as e:                       # noqa: BLE001 — diagnostic only
        print(f"  ! find_mechanical_pages failed ({e}); scanning all pages")
        plan = []

    if not plan:
        plan = list(range(total))
    return plan, total


def _resolve_valid_marks(pdf_path, use_schedule):
    """Derive the valid-mark whitelist and mark_details from the PDF's schedules.

    Returns (valid_marks_or_None, mark_details). valid_marks=None means 'accept
    every regex hit' (debugging only — noisier).
    """
    if not use_schedule:
        return None, {}
    try:
        from schedule_parser import parse_pdf_schedules
        _sched, marks, mark_details, _legend, _summary, _vars = parse_pdf_schedules(str(pdf_path))
        valid = set(marks) if marks else None
        return valid, (mark_details or {})
    except Exception as e:                       # noqa: BLE001 — diagnostic only
        print(f"  ! parse_pdf_schedules failed ({e}); running with valid_marks=None")
        return None, {}


def run(pdf_path, out_dir=None, explicit_pages=None, use_schedule=True, enable_ocr=False):
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        print(f"ERROR: not a file: {pdf_path}", file=sys.stderr)
        return 2

    stem = pdf_path.stem
    out_dir = Path(out_dir) if out_dir else pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GRD extract: {pdf_path.name}")

    plan_pages, total_pages = _resolve_plan_pages(pdf_path, explicit_pages)
    print(f"  plan pages: {len(plan_pages)} / {total_pages} total")

    valid_marks, mark_details = _resolve_valid_marks(pdf_path, use_schedule)
    if valid_marks is not None:
        print(f"  schedule marks: {len(valid_marks)} ({', '.join(sorted(valid_marks)[:12])}"
              f"{' …' if len(valid_marks) > 12 else ''})")
    else:
        print("  schedule marks: none (valid_marks=None — accepting all regex hits)")

    instances, warnings = extract_diffuser_instances(
        str(pdf_path), plan_pages,
        valid_marks=valid_marks,
        enable_ocr_supplement=enable_ocr,
    )

    style = detect_drawing_style(instances, warnings)
    bom = aggregate_diffuser_bom(instances, mark_details or None)

    pages_with_grd = sorted({i['page'] for i in instances})
    cfm_found = sum(1 for i in instances if i.get('cfm') is not None)

    # ── Write instances JSON ──────────────────────────────────────────────────
    inst_path = out_dir / f"{stem}_grd_instances.json"
    inst_path.write_text(json.dumps(instances, indent=2), encoding="utf-8")

    # ── Write BOM CSV ─────────────────────────────────────────────────────────
    bom_path = out_dir / f"{stem}_grd_bom.csv"
    with bom_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MARK", "NECK SIZE", "QTY", "TOTAL CFM", "CFM MISSING",
                    "MANUFACTURER", "MODEL", "MODULE SIZE", "MOUNTING"])
        for row in bom:
            w.writerow([
                row["mark"],
                display_neck_size(row["neck_size_canon"]),
                row["qty"],
                row["total_cfm"],
                row["cfm_missing"],
                row["manufacturer"],
                row["model"],
                row["module_size"],
                row["mounting"],
            ])

    # ── Write run metadata ────────────────────────────────────────────────────
    meta = {
        "pdf": pdf_path.name,
        "total_pages": total_pages,
        "plan_pages": [p + 1 for p in plan_pages],   # 1-indexed for humans
        "pages_with_grd": pages_with_grd,
        "drawing_style": style,
        "schedule_marks": sorted(valid_marks) if valid_marks else [],
        "instances_total": len(instances),
        "bom_rows": len(bom),
        "cfm_found": cfm_found,
        "cfm_missing": len(instances) - cfm_found,
        "warnings": warnings,
    }
    meta_path = out_dir / f"{stem}_grd_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  drawing style:   {style}")
    print(f"  instances:       {len(instances)}  "
          f"(CFM found {cfm_found}, missing {len(instances) - cfm_found})")
    print(f"  BOM rows:        {len(bom)}")
    print(f"  pages with GRD:  {pages_with_grd}")
    if warnings:
        print(f"  warnings:        {len(warnings)}")
        for w in warnings[:8]:
            print(f"    - {w}")
        if len(warnings) > 8:
            print(f"    … and {len(warnings) - 8} more")
    print(f"  → {inst_path.name}, {bom_path.name}, {meta_path.name}  (in {out_dir})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Standalone GRD plan-label extractor")
    ap.add_argument("pdf", help="path to the plan PDF")
    ap.add_argument("--out", default=None, help="output directory (default: next to PDF)")
    ap.add_argument("--pages", type=int, nargs="+", default=None,
                    help="explicit 1-indexed plan pages (overrides auto-detect)")
    ap.add_argument("--no-schedule", action="store_true",
                    help="do not derive valid_marks from schedules (accept all regex hits)")
    ap.add_argument("--ocr", action="store_true",
                    help="enable OCR supplement on healthy pages (slow)")
    args = ap.parse_args(argv)

    return run(
        args.pdf,
        out_dir=args.out,
        explicit_pages=args.pages,
        use_schedule=not args.no_schedule,
        enable_ocr=args.ocr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
