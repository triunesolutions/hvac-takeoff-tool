"""
benchmark_v11.py — score v11 against the May-5 ground_truth/, side-by-side with v10.

For each of the 6 LS-reviewed projects:
  1. Run YOLO inference with v11 on the same plan pages we reviewed
  2. Match v11 detections against ls_ground_truth.json (IoU >= 0.4)
  3. Bucket each detection: accepted / relabeled / phantom
  4. Bucket each ground-truth box not matched: missed
  5. Compare totals to the v10 numbers from ls_summary.txt

Run from repo root:
  python benchmark_v11.py
"""
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import fitz
import numpy as np

REPO = Path(__file__).parent
GT_DIR = REPO / 'ground_truth'
BENCH_DIR = REPO / 'benchmark_output'
V11_MODEL = REPO / 'models' / 'hvac_yolov8s_v11.pt'

DPI = 200
IOU_MATCH = 0.4
CONF = 0.4

# Tile config matches takeoff_cli.run_inference defaults so v11 sees the
# same crops it was trained on.
TILE_SIZE = 640
TILE_OVERLAP = 160


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def render_page(pdf_path, page_idx, dpi=DPI):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def infer_page(model, img):
    h, w = img.shape[:2]
    step = TILE_SIZE - TILE_OVERLAP
    boxes = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)
            xs = max(0, x2 - TILE_SIZE)
            ys = max(0, y2 - TILE_SIZE)
            tile = img[ys:y2, xs:x2]
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                pad = np.ones((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) * 255
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            results = model(tile, conf=CONF, verbose=False)
            for r in results:
                names = r.names
                for b in r.boxes:
                    x1, y1, bx2, by2 = b.xyxy[0].tolist()
                    cls_name = names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    boxes.append({
                        'x1': x1 + xs, 'y1': y1 + ys,
                        'x2': bx2 + xs, 'y2': by2 + ys,
                        'cls': cls_name, 'conf': conf,
                    })
    # NMS across tile seams
    return _nms(boxes, iou_thresh=0.5)


def _nms(boxes, iou_thresh=0.5):
    boxes = sorted(boxes, key=lambda b: -b['conf'])
    keep = []
    for b in boxes:
        bb = (b['x1'], b['y1'], b['x2'], b['y2'])
        if any(iou(bb, (k['x1'], k['y1'], k['x2'], k['y2'])) > iou_thresh for k in keep):
            continue
        keep.append(b)
    return keep


def parse_v10_summary(summary_path):
    """Return dict with v10 accepted/relabeled/deleted counts."""
    if not summary_path.exists():
        return None
    text = summary_path.read_text()
    out = {}
    for key, label in [('accepted', 'Accepted'), ('relabeled', 'Relabeled'),
                       ('deleted', 'Deleted'), ('added', 'Added')]:
        m = re.search(rf'{label}.*?:\s*(\d+)', text)
        if m:
            out[key] = int(m.group(1))
    return out


def benchmark_project(model, project_name):
    gt_path = GT_DIR / project_name / 'ls_ground_truth.json'
    summary_path = GT_DIR / project_name / 'ls_summary.txt'
    if not gt_path.exists():
        return None
    truth_by_page = json.loads(gt_path.read_text())

    bench_proj = BENCH_DIR / project_name
    det_files = list(bench_proj.glob('*_detections.json'))
    if not det_files:
        return None
    pdf_path = json.loads(det_files[0].read_text()).get('pdf')
    if not pdf_path or not Path(pdf_path).exists():
        return None

    accepted = relabeled = phantom = missed = 0
    confusion = Counter()
    phantom_classes = Counter()

    for page_str, truths in truth_by_page.items():
        page_idx = int(page_str)
        try:
            img = render_page(pdf_path, page_idx)
        except Exception as e:
            print(f"  ! page {page_idx + 1} render failed: {e}")
            continue
        v11_dets = infer_page(model, img)

        truth_used = [False] * len(truths)
        for d in v11_dets:
            d_box = (d['x1'], d['y1'], d['x2'], d['y2'])
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(truths):
                if truth_used[i]:
                    continue
                t_box = (t['x1'], t['y1'], t['x2'], t['y2'])
                u = iou(d_box, t_box)
                if u > best_iou:
                    best_iou, best_i = u, i
            if best_i >= 0 and best_iou >= IOU_MATCH:
                truth_used[best_i] = True
                t = truths[best_i]
                if t['cls'] == d['cls']:
                    accepted += 1
                else:
                    relabeled += 1
                    confusion[(d['cls'], t['cls'])] += 1
            else:
                phantom += 1
                phantom_classes[d['cls']] += 1
        missed += sum(1 for u in truth_used if not u)

    v10 = parse_v10_summary(summary_path) or {}
    return {
        'project': project_name,
        'v11': {'accepted': accepted, 'relabeled': relabeled,
                'phantom': phantom, 'missed': missed},
        'v10': v10,
        'confusion': confusion,
        'phantom_classes': phantom_classes,
    }


def main():
    if not V11_MODEL.exists():
        print(f"ERROR: {V11_MODEL} not found")
        sys.exit(1)
    print(f"Loading v11: {V11_MODEL}")
    from ultralytics import YOLO
    model = YOLO(str(V11_MODEL))
    print(f"  {len(model.names)} classes\n")

    projects = sorted(p.name for p in GT_DIR.iterdir() if p.is_dir())
    rows = []
    for p in projects:
        print(f"=== {p}")
        r = benchmark_project(model, p)
        if r is None:
            print(f"  ! skipped (missing inputs)")
            continue
        v11 = r['v11']
        v10 = r['v10']
        print(f"  v11: accepted={v11['accepted']}  relabeled={v11['relabeled']}  "
              f"phantom={v11['phantom']}  missed={v11['missed']}")
        print(f"  v10: accepted={v10.get('accepted','?')}  relabeled={v10.get('relabeled','?')}  "
              f"phantom={v10.get('deleted','?')}  missed=?")
        if r['phantom_classes']:
            top = ', '.join(f"{c}={n}" for c, n in r['phantom_classes'].most_common(3))
            print(f"  v11 phantom top: {top}")
        if r['confusion']:
            top = ', '.join(f"{a}->{b}={n}" for (a, b), n in r['confusion'].most_common(3))
            print(f"  v11 confusion top: {top}")
        rows.append(r)
        print()

    print("=" * 90)
    print(f"{'Project':<48} {'v11 ok':>7} {'rel':>5} {'phn':>5} {'mis':>5}  | {'v10 ok':>7} {'rel':>5} {'phn':>5}")
    print("-" * 90)
    tot11 = Counter()
    tot10 = Counter()
    for r in rows:
        v11 = r['v11']; v10 = r['v10']
        print(f"{r['project'][:48]:<48} "
              f"{v11['accepted']:>7} {v11['relabeled']:>5} {v11['phantom']:>5} {v11['missed']:>5}  | "
              f"{v10.get('accepted',0):>7} {v10.get('relabeled',0):>5} {v10.get('deleted',0):>5}")
        for k in ('accepted', 'relabeled', 'phantom', 'missed'):
            tot11[k] += v11[k]
        for k in ('accepted', 'relabeled', 'deleted'):
            tot10[k] += v10.get(k, 0)
    print("-" * 90)
    print(f"{'TOTAL':<48} "
          f"{tot11['accepted']:>7} {tot11['relabeled']:>5} {tot11['phantom']:>5} {tot11['missed']:>5}  | "
          f"{tot10['accepted']:>7} {tot10['relabeled']:>5} {tot10['deleted']:>5}")


if __name__ == '__main__':
    main()
