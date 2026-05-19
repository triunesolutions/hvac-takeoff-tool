"""Prepare a PaddleOCR-format rec-head fine-tune dataset from the verified
tag-bubble label set.

Input:
  tag_bubble_labels.jsonl  — committed; one row per crop in tag_dataset/
                             reason in {"hit", "no_match", "no_text"}
                             only reason=="hit" rows have bubble_rect_in_crop
  tag_dataset/images/<project>/<idx>.png — 26K source crops, fetched from
                             GH release datasets-2026-05-11 (gitignored)

Output (ocr_finetune/):
  images/<project>/<idx>.png  — bubble crops, preprocessed (3x upscale + Otsu)
  train.txt                   — "img_path\tlabel" per line
  val.txt                     — same, for held-out projects
  dict.txt                    — 37-char vocabulary (A-Z, 0-9, -)
  manifest.json               — split sizes + held-out project list

Per-project split (not random) so train and val don't share font / scanner /
title-block style. Random splits leak per-project idiosyncrasies and inflate
the val accuracy that's reported.

Usage:
  # default: 2 projects held out (Sola, 01_Flex_200_Corridors)
  python prepare_ocr_finetune.py

  # custom held-out
  python prepare_ocr_finetune.py --val-projects Sola Erewhon

  # custom paths
  python prepare_ocr_finetune.py --labels tag_bubble_labels.jsonl \
      --tag-dataset tag_dataset --out ocr_finetune
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from ocr_preprocess import preprocess_bubble_crop


VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
MAX_TEXT_LEN = 12  # longest realistic tag e.g. "VAV-23A" + headroom

DEFAULT_VAL = ["Sola_Salons", "01_Flex_200_Corridors"]


def _load_hits(labels_path: Path):
    """Yield (row, project_slug) for every reason==hit row."""
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("reason") != "hit":
                continue
            rect = row.get("bubble_rect_in_crop")
            if not rect or len(rect) != 4:
                continue
            tag = (row.get("tag") or "").strip().upper()
            if not tag or len(tag) > MAX_TEXT_LEN:
                continue
            # Drop any tags using characters outside the vocabulary
            if not all(c in VOCAB for c in tag):
                continue
            project = row["img"].split("/")[1]
            yield row, project


def _match_project(project_slug: str, val_keys):
    """Loose match: held-out keyword is a substring of project slug.
    Lets the user pass 'Sola' instead of the full slug."""
    lower = project_slug.lower()
    return any(k.lower() in lower for k in val_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="tag_bubble_labels.jsonl")
    ap.add_argument("--tag-dataset", default="tag_dataset")
    ap.add_argument("--out", default="ocr_finetune")
    ap.add_argument("--val-projects", nargs="+", default=DEFAULT_VAL,
                    help="Project name substrings to hold out as val set")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="Skip upscale+Otsu; save raw crops (debug only)")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    src_root = Path(args.tag_dataset)
    out_root = Path(args.out)
    img_out = out_root / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise SystemExit(f"labels file not found: {labels_path}")
    if not src_root.exists():
        raise SystemExit(
            f"tag_dataset not found at {src_root}. Pull from GH release:\n"
            f"  gh release download datasets-2026-05-11 "
            f"--repo triunesolutions/hvac-takeoff-tool --pattern tag_dataset.zip\n"
            f"  python -c \"import zipfile;zipfile.ZipFile('tag_dataset.zip').extractall('.')\""
        )

    train_lines, val_lines = [], []
    project_counts = Counter()
    skipped_missing = 0
    skipped_crop = 0

    for row, project in _load_hits(labels_path):
        src_img = src_root / row["img"]
        if not src_img.exists():
            skipped_missing += 1
            continue
        x1, y1, x2, y2 = [float(v) for v in row["bubble_rect_in_crop"]]
        if x2 - x1 < 4 or y2 - y1 < 4:
            skipped_crop += 1
            continue

        # bubble_rect_in_crop is in the *upscaled* labeling coord space
        # (3x; see label_tag_bubbles_ocr.preprocess_for_ocr). To recover the
        # crop from the original 320x320 PNG we divide by 3, then re-apply
        # the canonical preprocessing.
        scale = 3.0
        bx1, by1, bx2, by2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale

        try:
            pil = Image.open(src_img).convert("RGB")
        except Exception:
            skipped_missing += 1
            continue
        # Small bbox pad so the recognizer sees a little context on each side
        pad = 2
        cw, ch = pil.size
        cx1 = max(0, int(bx1 - pad))
        cy1 = max(0, int(by1 - pad))
        cx2 = min(cw, int(bx2 + pad))
        cy2 = min(ch, int(by2 + pad))
        bubble = pil.crop((cx1, cy1, cx2, cy2))
        if bubble.size[0] < 4 or bubble.size[1] < 4:
            skipped_crop += 1
            continue

        if args.no_preprocess:
            import numpy as np
            arr = np.array(bubble)
        else:
            arr = preprocess_bubble_crop(bubble)

        proj_out = img_out / project
        proj_out.mkdir(parents=True, exist_ok=True)
        rel = f"images/{project}/{project_counts[project]:06d}.png"
        Image.fromarray(arr).save(out_root / rel, format="PNG", optimize=True)
        project_counts[project] += 1

        line = f"{rel}\t{row['tag'].strip().upper()}\n"
        if _match_project(project, args.val_projects):
            val_lines.append(line)
        else:
            train_lines.append(line)

    (out_root / "train.txt").write_text("".join(train_lines), encoding="utf-8")
    (out_root / "val.txt").write_text("".join(val_lines), encoding="utf-8")
    (out_root / "dict.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")

    manifest = {
        "train_rows": len(train_lines),
        "val_rows": len(val_lines),
        "val_projects_filter": args.val_projects,
        "projects_total": len(project_counts),
        "skipped_missing": skipped_missing,
        "skipped_crop": skipped_crop,
        "vocab_size": len(VOCAB),
        "max_text_length": MAX_TEXT_LEN,
        "preprocess": "3x upscale + Otsu" if not args.no_preprocess else "raw",
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"train: {len(train_lines):>5}   val: {len(val_lines):>5}")
    print(f"projects: {len(project_counts)} (val filter: {args.val_projects})")
    print(f"skipped: missing={skipped_missing} bad_crop={skipped_crop}")
    print(f"wrote {out_root}/")


if __name__ == "__main__":
    main()
