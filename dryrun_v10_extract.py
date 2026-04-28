"""Dry-run extraction over staged sample projects.

For each sample_* project under PROJECTS_DIR, run extract_annotations() (which
applies normalize_class), and report:
  - normalized class counts
  - annotations missing tag content (would be silently dropped by train_yolo)
  - any class not yet in the v9 KNOWN_CLASSES list (potential new classes)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from train_yolo import PROJECTS_DIR, extract_annotations
from class_aliases import normalize_class


V9_CLASSES = {
    "AD-T-BAR SUPPLY", "AD-T-BAR RETURN", "AD-SURF SUPPLY", "AD-SURF RETURN",
    "AD-LINEAR SLOT DIFFUSER", "AD-LINEAR PLENUM", "AD-GRD",
    "EXHAUST FAN", "FAN", "SUPPLY FAN", "FAN COIL UNIT",
    "LOUVER", "LOUVERS",
    "FIRE SMOKE DAMPER", "FIRE DAMPER", "MOTORIZED DAMPER",
    "MANUAL VOLUME DAMPER", "DAMPER WITH TAP",
    "VAV", "VRF", "PTAC",
    "CONDENSING UNIT", "AIR HANDLING UNIT", "PACKAGED ROOFTOP UNIT",
    "HEATER", "DUCT HEATER", "ENERGY RECOVERY", "HEAT PUMP",
    "SPLIT SYSTEM", "SPLIT SYSTEM HEAT PUMP", "FURNACE",
    "VENT CAP", "HOOD", "AIR CURTAIN",
}


def main() -> None:
    root = Path(PROJECTS_DIR)
    sample_projs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("sample_"))
    print(f"Sample projects staged: {len(sample_projs)}")

    norm_counts: Counter = Counter()
    raw_counts: Counter = Counter()
    missing_content: defaultdict[str, int] = defaultdict(int)
    per_project: dict[str, int] = {}

    for proj in sample_projs:
        labeled = proj / "labeled"
        if not labeled.is_dir():
            continue
        n_for_proj = 0
        for pdf in labeled.glob("*.pdf"):
            try:
                pages, _ = extract_annotations(str(pdf))
            except Exception as e:
                print(f"  ERR {proj.name} / {pdf.name}: {e}")
                continue
            for anns in pages.values():
                for ann in anns:
                    norm_counts[ann["class"]] += 1
                    n_for_proj += 1
            import fitz
            with fitz.open(pdf) as doc:
                for page in doc:
                    for a in page.annots() or ():
                        if a.type[1] != "Polygon":
                            continue
                        subj = (a.info.get("subject") or "").strip()
                        cont = (a.info.get("content") or "").strip()
                        if subj:
                            raw_counts[subj] += 1
                        if subj and not cont:
                            missing_content[normalize_class(subj)] += 1
        per_project[proj.name] = n_for_proj

    new_classes = [c for c in norm_counts if c not in V9_CLASSES]

    print()
    print("Normalized class counts (after aliasing):")
    for cls, n in norm_counts.most_common():
        marker = "" if cls in V9_CLASSES else "  <- NEW CLASS"
        print(f"  {n:>5}  {cls}{marker}")

    print()
    print(f"Total normalized: {sum(norm_counts.values())}")
    print(f"Total raw (subject only): {sum(raw_counts.values())}")
    print(f"Distinct classes after aliasing: {len(norm_counts)}")
    print(f"Classes not in v9 KNOWN_CLASSES list: {len(new_classes)}")
    if new_classes:
        for c in new_classes:
            print(f"   NEW {c}: {norm_counts[c]}")

    if missing_content:
        print()
        print("WARNING — annotations with empty 'content' (will be DROPPED by train_yolo):")
        for cls, n in sorted(missing_content.items(), key=lambda x: -x[1]):
            print(f"  {n:>5}  {cls}")
    else:
        print()
        print("OK — every polygon has a tag content field.")

    print()
    print("Per-project totals:")
    for proj, n in sorted(per_project.items(), key=lambda x: -x[1]):
        print(f"  {n:>5}  {proj}")


if __name__ == "__main__":
    main()
