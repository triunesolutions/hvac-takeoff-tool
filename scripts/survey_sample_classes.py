"""One-shot survey of polygon `subject` fields across all 36 sample takeoff PDFs.

Writes:
  - sample_class_counts.csv  (class, total_count, project_count)
  - sample_per_project.csv   (project, class, count)
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path (moved into scripts/)

import csv
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF

SAMPLE_ROOT = Path(r"C:\Users\JFL\Downloads\SAMPLE FILES 27.04.26\SAMPLE FILES 27.04.26")
OUT_DIR = Path(__file__).parent
SKIP = {"KNAPE FILE"}


def find_takeoff_pdf(project_dir: Path) -> Path | None:
    ct = project_dir / "Completed Takeoff"
    if not ct.is_dir():
        return None
    pdfs = sorted(ct.glob("Takeoff_*.pdf"))
    if not pdfs:
        pdfs = sorted(ct.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def extract_subjects(pdf_path: Path) -> Counter:
    counts: Counter = Counter()
    with fitz.open(pdf_path) as doc:
        for page in doc:
            for annot in page.annots() or ():
                if annot.type[0] != fitz.PDF_ANNOT_POLYGON:
                    continue
                info = annot.info or {}
                subj = (info.get("subject") or info.get("title") or "").strip()
                if subj:
                    counts[subj] += 1
    return counts


def main():
    per_project: dict[str, Counter] = {}
    global_counts: Counter = Counter()
    project_counts: defaultdict[str, int] = defaultdict(int)

    projects = sorted(p for p in SAMPLE_ROOT.iterdir() if p.is_dir() and p.name not in SKIP)
    print(f"Scanning {len(projects)} projects...")

    for proj in projects:
        pdf = find_takeoff_pdf(proj)
        if not pdf:
            print(f"  SKIP no_takeoff: {proj.name}")
            continue
        try:
            counts = extract_subjects(pdf)
        except Exception as e:
            print(f"  ERROR {proj.name}: {e}")
            continue
        per_project[proj.name] = counts
        total = sum(counts.values())
        print(f"  {total:>5}  {proj.name}")
        for cls, n in counts.items():
            global_counts[cls] += n
            project_counts[cls] += 1

    with (OUT_DIR / "sample_class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "total_count", "project_count"])
        for cls, n in global_counts.most_common():
            w.writerow([cls, n, project_counts[cls]])

    with (OUT_DIR / "sample_per_project.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["project", "class", "count"])
        for proj_name, counts in per_project.items():
            for cls, n in counts.most_common():
                w.writerow([proj_name, cls, n])

    print()
    print(f"Total polygons: {sum(global_counts.values())}")
    print(f"Distinct classes: {len(global_counts)}")
    print(f"Projects scanned: {len(per_project)}")
    print(f"Wrote sample_class_counts.csv and sample_per_project.csv")


if __name__ == "__main__":
    main()
