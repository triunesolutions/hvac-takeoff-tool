"""Stage the 36 sample projects into the train_yolo.py PROJECTS_DIR layout.

Source: SAMPLE FILES 27.04.26/<project>/{Plans_Specs,Completed Takeoff}/*.pdf
Target: data to train/projects/sample_<name>/{raw,labeled}/*.pdf

Skipped:
  - KNAPE FILE                                      (different folder structure)
  - HVAC Replacement At The Leadership Academy     (only 6 polygons — incomplete)
  - MBUSD Manhattan Beach MS                        (only 1 polygon — incomplete)
  - MBUSD Mira Costa HS                             (only 2 polygons — incomplete)
  - Holdout set (kept for eval-only):
      Sola Salons, 677 Imperial, Alliance Mass Stern, Krispy Kreme

Re-runnable: existing target dirs are reused; pdfs are copied only if missing.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SAMPLE_ROOT = Path(r"C:\Users\JFL\Downloads\SAMPLE FILES 27.04.26\SAMPLE FILES 27.04.26")
TARGET_ROOT = Path(r"C:\Users\JFL\Downloads\Triune\data to train\projects")

DROP_LOW_ANNOT = {
    "4.10.26 HVAC Replacement At The Leadership Academy - Lancaster",
    "4.18.26 MBUSD Manhattan Beach MS",
    "4.18.26 MBUSD Mira Costa HS",
}
HOLDOUT = {
    "4.15.26 Sola Salons",
    "4.21.26 677 Imperial Street - TI",
    "4.17.26 Alliance Mass Stern Remodel",
    "4.13.26 Krispy Kreme #574 - Valencia",
}
SKIP_FOLDERS = {"KNAPE FILE"}


def safe_name(name: str) -> str:
    # Kaggle dataset uploads reject filenames containing: & # ? * ' " ( ) [ ] { } ! @ $ % ^ = + , ;
    n = name.replace("&", "and").replace("#", "")
    n = re.sub(r"[\'\"\(\)\[\]\{\}!@\$%\^=+,;]", "", n)
    n = re.sub(r"[<>:|/\\?*]", "_", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def copy_pdfs(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted(src_dir.glob("*.pdf")):
        out = dst_dir / pdf.name
        if out.exists() and out.stat().st_size == pdf.stat().st_size:
            continue
        shutil.copy2(pdf, out)
        n += 1
    return n


def main() -> None:
    if not SAMPLE_ROOT.is_dir():
        raise SystemExit(f"Sample root not found: {SAMPLE_ROOT}")
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    staged = 0
    skipped_lowannot = 0
    skipped_holdout = 0
    skipped_other = 0
    holdout_log: list[str] = []

    for proj in sorted(SAMPLE_ROOT.iterdir()):
        if not proj.is_dir() or proj.name in SKIP_FOLDERS:
            continue
        if proj.name in DROP_LOW_ANNOT:
            print(f"  DROP (low annotations) : {proj.name}")
            skipped_lowannot += 1
            continue
        if proj.name in HOLDOUT:
            print(f"  HOLDOUT (eval-only)    : {proj.name}")
            holdout_log.append(proj.name)
            skipped_holdout += 1
            continue

        plans = proj / "Plans_Specs"
        takeoff = proj / "Completed Takeoff"
        if not plans.is_dir() or not takeoff.is_dir():
            print(f"  SKIP no plan/takeoff   : {proj.name}")
            skipped_other += 1
            continue

        target_name = "sample_" + safe_name(proj.name)
        target = TARGET_ROOT / target_name
        n_raw = copy_pdfs(plans, target / "raw")
        n_lab = copy_pdfs(takeoff, target / "labeled")
        print(f"  STAGED ({n_raw}r/{n_lab}l)        : {target_name}")
        staged += 1

    print()
    print(f"Staged       : {staged}")
    print(f"Drop (low)   : {skipped_lowannot}")
    print(f"Holdout      : {skipped_holdout}  ({', '.join(holdout_log) or '-'})")
    print(f"Skipped other: {skipped_other}")
    print(f"Target root  : {TARGET_ROOT}")


if __name__ == "__main__":
    main()
