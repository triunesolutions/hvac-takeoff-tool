"""
Pull verified annotations back from Label Studio after the team's review.

Reads every annotated task from the LS project, joins them against the v10
predictions saved in `<project>_detections.json`, and emits:

  1. `<project_dir>/ls_ground_truth.json` — verified bbox+class per page
  2. `<project_dir>/ls_discrepancy_report.csv` — one row per (page, det):
       status ∈ {accepted, deleted (phantom), relabeled, added (missed)}
       v10_class, truth_class, iou
  3. `<project_dir>/ls_summary.txt` — top-line counts: phantoms by class,
     class confusions (v10 → truth) sorted by frequency.

Usage:
  python import_from_label_studio.py "4.15.26 Sola Salons"
  python import_from_label_studio.py "4.15.26 Sola Salons" --ls-project-name "Sola Review"
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

LS_BASE = "http://localhost:8080"
TOKEN_FILE = Path.home() / ".label_studio_token"
IOU_MATCH = 0.4


def get_access_token():
    refresh = TOKEN_FILE.read_text().strip()
    r = requests.post(f"{LS_BASE}/api/token/refresh/", json={"refresh": refresh}, timeout=15)
    r.raise_for_status()
    return r.json()["access"]


def auth_headers():
    return {"Authorization": f"Bearer {get_access_token()}"}


def find_project_id(title):
    h = auth_headers()
    r = requests.get(f"{LS_BASE}/api/projects/", headers=h, params={"page_size": 200}, timeout=15)
    r.raise_for_status()
    for p in r.json().get("results", []):
        if p.get("title") == title:
            return p["id"]
    return None


def fetch_tasks(project_id):
    """Pull every task with its annotations."""
    h = auth_headers()
    out = []
    page = 1
    while True:
        r = requests.get(
            f"{LS_BASE}/api/tasks/",
            headers=h,
            params={"project": project_id, "page": page, "page_size": 100},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("tasks") or data.get("results") or []
        if not results:
            break
        out.extend(results)
        if len(results) < 100:
            break
        page += 1
    return out


def fetch_annotations(task_id):
    h = auth_headers()
    r = requests.get(f"{LS_BASE}/api/tasks/{task_id}/annotations/", headers=h, timeout=30)
    r.raise_for_status()
    return r.json()


def ls_box_to_xyxy(value, img_w, img_h):
    """LS rectangle pct → pixel xyxy."""
    x = value["x"] / 100.0 * img_w
    y = value["y"] / 100.0 * img_h
    w = value["width"] / 100.0 * img_w
    h = value["height"] / 100.0 * img_h
    return x, y, x + w, y + h


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--ls-project-name")
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    proj_dir = Path(args.project_dir)
    if not proj_dir.exists():
        proj_dir = repo_root / "benchmark_output" / args.project_dir
    if not proj_dir.exists():
        print(f"Not found: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    title = args.ls_project_name or f"HVAC Review — {proj_dir.name}"
    pid = find_project_id(title)
    if pid is None:
        print(f"No LS project named '{title}' — did you run export first?", file=sys.stderr)
        sys.exit(1)
    print(f"LS project '{title}' (id={pid})")

    det_files = list(proj_dir.glob("*_detections.json"))
    if not det_files:
        print("No *_detections.json next to the project — can't reconcile.", file=sys.stderr)
        sys.exit(1)
    with open(det_files[0], encoding="utf-8") as f:
        det_data = json.load(f)

    print("Fetching tasks ...")
    tasks = fetch_tasks(pid)
    print(f"  {len(tasks)} tasks")

    truth_by_page = {}  # page_idx → list of {bbox, cls}
    for t in tasks:
        page_idx = t.get("data", {}).get("page", 0) - 1
        anns = t.get("annotations") or []
        if not anns:
            anns = fetch_annotations(t["id"])
        if not anns:
            continue
        # use the latest non-cancelled annotation
        chosen = None
        for a in sorted(anns, key=lambda x: x.get("created_at", ""), reverse=True):
            if not a.get("was_cancelled"):
                chosen = a
                break
        if chosen is None:
            continue
        truths = []
        for r in chosen.get("result", []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            iw = r.get("original_width", 1)
            ih = r.get("original_height", 1)
            x1, y1, x2, y2 = ls_box_to_xyxy(v, iw, ih)
            cls = v["rectanglelabels"][0] if v.get("rectanglelabels") else None
            truths.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})
        truth_by_page[page_idx] = truths

    discrepancies = []
    phantom_classes = Counter()
    confusion = Counter()  # (v10_class, truth_class) → count
    accepted = 0

    for page_idx_str, dets in det_data["pages"].items():
        page_idx = int(page_idx_str)
        truths = truth_by_page.get(page_idx, [])
        if not truths and page_idx not in truth_by_page:
            print(f"  page {page_idx + 1}: not reviewed yet — skipping")
            continue
        truth_used = [False] * len(truths)
        for d in dets:
            d_box = (d["x1"], d["y1"], d["x2"], d["y2"])
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(truths):
                if truth_used[i]:
                    continue
                t_box = (t["x1"], t["y1"], t["x2"], t["y2"])
                u = iou(d_box, t_box)
                if u > best_iou:
                    best_iou, best_i = u, i
            if best_i >= 0 and best_iou >= IOU_MATCH:
                truth_used[best_i] = True
                t = truths[best_i]
                if t["cls"] == d["cls"]:
                    accepted += 1
                    discrepancies.append(
                        dict(page=page_idx + 1, status="accepted", v10_class=d["cls"],
                             truth_class=t["cls"], iou=round(best_iou, 3),
                             x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])
                    )
                else:
                    confusion[(d["cls"], t["cls"])] += 1
                    discrepancies.append(
                        dict(page=page_idx + 1, status="relabeled", v10_class=d["cls"],
                             truth_class=t["cls"], iou=round(best_iou, 3),
                             x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])
                    )
            else:
                phantom_classes[d["cls"]] += 1
                discrepancies.append(
                    dict(page=page_idx + 1, status="deleted", v10_class=d["cls"],
                         truth_class="", iou=0.0,
                         x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])
                )

        # Anything in truth that wasn't matched = a missed detection the user added
        for i, t in enumerate(truths):
            if truth_used[i]:
                continue
            discrepancies.append(
                dict(page=page_idx + 1, status="added", v10_class="",
                     truth_class=t["cls"], iou=0.0,
                     x1=t["x1"], y1=t["y1"], x2=t["x2"], y2=t["y2"])
            )

    # Write outputs
    csv_path = proj_dir / "ls_discrepancy_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["page", "status", "v10_class", "truth_class", "iou",
                        "x1", "y1", "x2", "y2"],
        )
        w.writeheader()
        for row in discrepancies:
            w.writerow(row)
    print(f"\nWrote {csv_path} ({len(discrepancies)} rows)")

    truth_path = proj_dir / "ls_ground_truth.json"
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in truth_by_page.items()}, f, indent=2)
    print(f"Wrote {truth_path}")

    # Summary
    deleted_n = sum(phantom_classes.values())
    relabeled_n = sum(confusion.values())
    added_n = sum(1 for d in discrepancies if d["status"] == "added")
    summary_lines = [
        f"# {proj_dir.name} — Label Studio review summary",
        "",
        f"Accepted (correct box + class):   {accepted}",
        f"Relabeled (right box, wrong class): {relabeled_n}",
        f"Deleted (phantom — no symbol):    {deleted_n}",
        f"Added (missed detection):         {added_n}",
        "",
        "## Phantom detections by class (top 15)",
    ]
    for cls, n in phantom_classes.most_common(15):
        summary_lines.append(f"  {n:>4}  {cls}")
    summary_lines.append("\n## Class confusions: v10 → truth (top 15)")
    for (a, b), n in confusion.most_common(15):
        summary_lines.append(f"  {n:>4}  {a:30s} → {b}")
    summary_path = proj_dir / "ls_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
