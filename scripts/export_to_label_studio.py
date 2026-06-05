"""
Export v10 detections for one project into Label Studio as pre-annotations.

Workflow:
  1. Reads `<project>_detections.json` produced by takeoff_cli.py
  2. Renders each PDF page at 200 DPI → PNG (base64-embedded in the task)
  3. Creates a Label Studio project with our full class list
  4. Uploads one task per page, with v10 boxes pre-loaded as predictions
  5. Open the LS URL it prints → click ✓ / 🗑 / relabel on each box

Auth: reads the LS refresh JWT from ~/.label_studio_token and exchanges it for
a 5-min access token before each API call (no manual token management).

Usage:
  python export_to_label_studio.py "4.15.26 Sola Salons"
  python export_to_label_studio.py "4.15.26 Sola Salons" --ls-project-name "Sola Review"
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import fitz
import numpy as np
import requests

from class_aliases import CLASS_ALIASES

LS_BASE = "http://localhost:8080"
TOKEN_FILE = Path.home() / ".label_studio_token"


def get_access_token():
    refresh = TOKEN_FILE.read_text().strip()
    r = requests.post(f"{LS_BASE}/api/token/refresh/", json={"refresh": refresh}, timeout=15)
    r.raise_for_status()
    return r.json()["access"]


def auth_headers():
    return {"Authorization": f"Bearer {get_access_token()}"}


def render_page_png(pdf_path, page_idx, dpi=200):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    png_bytes = pix.tobytes("png")
    w, h = pix.width, pix.height
    doc.close()
    return png_bytes, w, h


def build_label_config(classes):
    """LS labeling-config XML. RectangleLabels with one Label per class."""
    palette = []
    for i, c in enumerate(classes):
        hue = (i * 47) % 360
        palette.append(f'    <Label value="{c}" background="hsl({hue},70%,50%)"/>')
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        + "\n".join(palette)
        + "\n  </RectangleLabels>\n"
        "</View>"
    )


def all_target_classes():
    """Every class we want available for relabeling — model classes + alias targets."""
    try:
        from ultralytics import YOLO

        model_path = Path(__file__).with_name("models") / "hvac_yolov8s_v10.pt"
        names = list(YOLO(str(model_path)).names.values())
    except Exception as e:
        print(f"  (couldn't read model class list: {e}; falling back to alias targets only)")
        names = []
    return sorted(set(names) | set(CLASS_ALIASES.values()))


def find_or_create_project(title, label_config):
    h = auth_headers()
    r = requests.get(f"{LS_BASE}/api/projects/", headers=h, params={"page_size": 200}, timeout=15)
    r.raise_for_status()
    for p in r.json().get("results", []):
        if p.get("title") == title:
            return p["id"], False
    r = requests.post(
        f"{LS_BASE}/api/projects/",
        headers=h,
        json={"title": title, "label_config": label_config},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["id"], True


def make_pred_result(det, img_w, img_h):
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    x1, x2 = sorted([max(0, x1), max(0, x2)])
    y1, y2 = sorted([max(0, y1), max(0, y2)])
    x2 = min(x2, img_w)
    y2 = min(y2, img_h)
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x1 / img_w,
            "y": 100.0 * y1 / img_h,
            "width": 100.0 * (x2 - x1) / img_w,
            "height": 100.0 * (y2 - y1) / img_h,
            "rotation": 0,
            "rectanglelabels": [det["cls"]],
        },
        "score": float(det.get("conf") or 0.0),
    }


def import_tasks(project_id, tasks):
    h = auth_headers()
    r = requests.post(
        f"{LS_BASE}/api/projects/{project_id}/import",
        headers=h,
        json=tasks,
        timeout=300,
    )
    if r.status_code >= 400:
        print(f"Import failed: {r.status_code} {r.text[:500]}", file=sys.stderr)
        r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", help="Either a full path or a folder name under benchmark_output/")
    ap.add_argument("--ls-project-name", help="Override Label Studio project title")
    ap.add_argument("--dry-run", action="store_true", help="Build payload but don't hit the API")
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    proj_dir = Path(args.project_dir)
    if not proj_dir.exists():
        proj_dir = repo_root / "benchmark_output" / args.project_dir
    if not proj_dir.exists():
        print(f"Project dir not found: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    det_files = list(proj_dir.glob("*_detections.json"))
    if not det_files:
        print(
            f"No *_detections.json in {proj_dir}.\n"
            "Re-run takeoff_cli.py on this PDF to produce the sidecar.",
            file=sys.stderr,
        )
        sys.exit(1)

    det_path = det_files[0]
    print(f"Reading: {det_path}")
    with open(det_path, encoding="utf-8") as f:
        det_data = json.load(f)
    pdf_path = det_data["pdf"]
    dpi = det_data.get("dpi", 200)

    classes = all_target_classes()
    print(f"Class palette: {len(classes)} classes")
    label_config = build_label_config(classes)

    title = args.ls_project_name or f"HVAC Review — {proj_dir.name}"
    if args.dry_run:
        print(f"[dry-run] Would create LS project: {title}")
        pid = None
    else:
        pid, created = find_or_create_project(title, label_config)
        print(f"LS project '{title}' (id={pid}, {'created' if created else 'reused'})")

    tasks = []
    total_preds = 0
    for page_idx_str, dets in det_data["pages"].items():
        page_idx = int(page_idx_str)
        print(f"  page {page_idx + 1}: rendering ({len(dets)} detections) ...", end=" ", flush=True)
        png_bytes, w, h = render_page_png(pdf_path, page_idx, dpi=dpi)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        results = [make_pred_result(d, w, h) for d in dets]
        total_preds += len(results)
        scores = [r["score"] for r in results]
        tasks.append(
            {
                "data": {
                    "image": data_url,
                    "page": page_idx + 1,
                    "project": proj_dir.name,
                    "pdf": Path(pdf_path).name,
                },
                "predictions": [
                    {
                        "model_version": "v10",
                        "result": results,
                        "score": float(np.mean(scores)) if scores else 0.0,
                    }
                ]
                if results
                else [],
            }
        )
        print(f"{len(png_bytes) // 1024} KB png")

    print(f"\nBuilt {len(tasks)} tasks, {total_preds} pre-annotations total.")
    if args.dry_run:
        print("[dry-run] skipping import.")
        return

    print("Uploading to Label Studio ...")
    res = import_tasks(pid, tasks)
    print(f"  imported: {res}")
    print(f"\nReview here: {LS_BASE}/projects/{pid}/data")


if __name__ == "__main__":
    main()
