"""
Benchmark v6 model against any project folder.

Usage:
    python benchmark.py                                    # All projects in projects/ dir
    python benchmark.py --projects 26 28 33               # Specific projects
    python benchmark.py --model models/hvac_yolov8s_v4.pt # Use different model
    python benchmark.py --conf 0.5                         # Confidence threshold
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

import fitz
import cv2
import numpy as np
import os
import argparse
from collections import defaultdict
from pathlib import Path
from class_aliases import normalize_class

PROJECTS_DIR = r"C:\Users\JFL\Downloads\Triune\data to train\projects"
OUTPUT_DIR = r"C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\output\benchmark"
DPI = 200
TILE_SIZE = 640
TILE_OVERLAP = 100  # Smaller overlap = fewer tiles
NMS_DIST = 50
SKIP_EMPTY_TILES = True  # Skip pure-white tiles
MAX_PAGES_PER_PROJECT = 2  # Only test top 2 most-annotated pages


def annot_to_display(ax, ay, rot, mb_w, mb_h):
    if rot == 270: return ay, mb_w - ax
    elif rot == 90: return mb_h - ay, ax
    elif rot == 180: return mb_w - ax, mb_h - ay
    return ax, ay


def render_page(pdf_path, page_idx, dpi=DPI):
    doc = fitz.open(pdf_path)
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def get_ground_truth(labeled_path):
    """Returns dict: page_idx -> list of {cls, cx, cy} in display pixel coords."""
    doc = fitz.open(labeled_path)
    pages = {}
    for pi in range(doc.page_count):
        pg = doc[pi]
        rot, mb_w, mb_h = pg.rotation, pg.mediabox.width, pg.mediabox.height
        anns = []
        for a in pg.annots() or []:
            try:
                if a.type[1] != 'Polygon': continue
            except: continue
            r = a.rect
            acx, acy = (r.x0+r.x1)/2, (r.y0+r.y1)/2
            dcx, dcy = annot_to_display(acx, acy, rot, mb_w, mb_h)
            anns.append({'cls': normalize_class(a.info.get('subject','')),
                         'cx': dcx*DPI/72, 'cy': dcy*DPI/72})
        if anns:
            pages[pi] = anns
    doc.close()
    return pages


def find_raw_pdf(project_dir, labeled_page_count):
    """Find the raw (unlabeled) PDF that matches the labeled page count."""
    raw_dir = os.path.join(project_dir, 'raw')
    if not os.path.exists(raw_dir):
        return None
    raws = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
    if not raws:
        return None
    for r in raws:
        try:
            d = fitz.open(os.path.join(raw_dir, r))
            pc = d.page_count
            d.close()
            if pc == labeled_page_count:
                return os.path.join(raw_dir, r)
        except:
            continue
    return os.path.join(raw_dir, raws[0])


def run_inference(model, img, conf=0.4):
    h, w = img.shape[:2]
    step = TILE_SIZE - TILE_OVERLAP

    # Build list of tiles to process, skipping empty ones
    tiles_to_process = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            xe, ye = min(x+TILE_SIZE, w), min(y+TILE_SIZE, h)
            xs, ys = max(0, xe-TILE_SIZE), max(0, ye-TILE_SIZE)
            tile = img[ys:ye, xs:xe]

            # Skip empty/mostly-white tiles
            if SKIP_EMPTY_TILES:
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if len(tile.shape) == 3 else tile
                dark_pct = (gray < 200).mean()
                if dark_pct < 0.005:  # Less than 0.5% dark pixels = essentially blank
                    continue

            if tile.shape[0]<TILE_SIZE or tile.shape[1]<TILE_SIZE:
                p = np.ones((TILE_SIZE,TILE_SIZE,3),dtype=np.uint8)*255
                p[:tile.shape[0],:tile.shape[1]] = tile
                tile = p
            tiles_to_process.append((tile, xs, ys))

    # Batch inference
    dets = []
    BATCH = 8
    for i in range(0, len(tiles_to_process), BATCH):
        batch = tiles_to_process[i:i+BATCH]
        batch_imgs = [t[0] for t in batch]
        results = model.predict(batch_imgs, conf=conf, verbose=False)
        for (_, xs, ys), r in zip(batch, results):
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                dets.append({'cls':model.names[int(box.cls[0])], 'conf':float(box.conf[0]),
                    'cx':(x1+x2)/2+xs, 'cy':(y1+y2)/2+ys,
                    'x1':x1+xs,'y1':y1+ys,'x2':x2+xs,'y2':y2+ys})

    final = []
    for d in sorted(dets, key=lambda x: x['conf'], reverse=True):
        if not any(abs(d['cx']-f['cx'])<NMS_DIST and abs(d['cy']-f['cy'])<NMS_DIST for f in final):
            final.append(d)
    return final


def score(dets, gt, match_dist=80):
    """Returns position recall + full match recall + confusion pairs."""
    matched_g = set()
    matched_d_pos = set()
    matched_d_full = set()
    confusion_pairs = []  # list of (predicted_class, actual_class) for matched positions
    for di, d in enumerate(dets):
        for gi, g in enumerate(gt):
            if gi in matched_g: continue
            if ((d['cx']-g['cx'])**2+(d['cy']-g['cy'])**2)**0.5 < match_dist:
                matched_g.add(gi)
                confusion_pairs.append((d['cls'], g['cls']))
                matched_d_pos.add(di)
                if d['cls'] == g['cls']:
                    matched_d_full.add(di)
                break
    tp_pos = len(matched_d_pos)
    tp_full = len(matched_d_full)
    return {
        'tp_pos': tp_pos,
        'tp_full': tp_full,
        'fp': len(dets) - tp_pos,
        'fn': len(gt) - tp_pos,
        'pos_recall': tp_pos / max(len(gt), 1),
        'full_recall': tp_full / max(len(gt), 1),
        'pos_precision': tp_pos / max(len(dets), 1),
        'full_precision': tp_full / max(len(dets), 1),
        'confusion_pairs': confusion_pairs,
    }


def benchmark_project(model, project_dir, project_name, conf=0.4, save_viz=True):
    """Benchmark a single project. Returns aggregated stats across all annotated pages."""
    labeled_dir = os.path.join(project_dir, 'labeled')
    if not os.path.isdir(labeled_dir):
        return None

    labeled_pdfs = [f for f in os.listdir(labeled_dir) if f.endswith('.pdf')]
    if not labeled_pdfs:
        return None

    # Pick the labeled PDF with the most polygon annotations
    best_lbl = None
    best_count = 0
    for f in labeled_pdfs:
        path = os.path.join(labeled_dir, f)
        pages = get_ground_truth(path)
        total = sum(len(v) for v in pages.values())
        if total > best_count:
            best_count = total
            best_lbl = path

    if not best_lbl or best_count == 0:
        return None

    pages = get_ground_truth(best_lbl)
    doc = fitz.open(best_lbl)
    page_count = doc.page_count
    doc.close()

    # Only keep top N pages by annotation count
    sorted_pages = sorted(pages.items(), key=lambda x: -len(x[1]))[:MAX_PAGES_PER_PROJECT]
    pages = dict(sorted_pages)

    raw_pdf = find_raw_pdf(project_dir, page_count)
    render_pdf = raw_pdf if raw_pdf else best_lbl

    # Aggregate stats across all annotated pages
    total = {'tp_pos':0,'tp_full':0,'fp':0,'fn':0,'gt_count':0,'det_count':0}
    page_results = []
    det_class_counts = defaultdict(int)
    gt_class_counts = defaultdict(int)
    confusion_pairs = []  # All (predicted, actual) pairs across all pages

    for page_idx, gt in pages.items():
        try:
            img = render_page(render_pdf, page_idx)
        except Exception as e:
            print(f"  SKIP page {page_idx+1}: {e}")
            continue

        dets = run_inference(model, img, conf=conf)
        s = score(dets, gt)

        total['tp_pos'] += s['tp_pos']
        total['tp_full'] += s['tp_full']
        total['fp'] += s['fp']
        total['fn'] += s['fn']
        total['gt_count'] += len(gt)
        total['det_count'] += len(dets)
        confusion_pairs.extend(s['confusion_pairs'])

        for d in dets: det_class_counts[d['cls']] += 1
        for g in gt: gt_class_counts[g['cls']] += 1

        page_results.append({
            'page': page_idx+1, 'gt': len(gt), 'det': len(dets),
            'pos_r': s['pos_recall'], 'full_r': s['full_recall'],
        })

        # Save visualization
        if save_viz:
            COLORS = [(0,200,0),(200,0,0),(0,165,255),(0,0,200),(255,200,0),(255,0,200),(0,255,255),(180,180,0)]
            vis = img.copy()
            for d in dets:
                color = COLORS[hash(d['cls']) % len(COLORS)]
                cv2.rectangle(vis, (int(d['x1']),int(d['y1'])), (int(d['x2']),int(d['y2'])), color, 3)
                cv2.putText(vis, d['cls'][:12], (int(d['x1']),int(d['y1'])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            for g in gt:
                cv2.drawMarker(vis, (int(g['cx']),int(g['cy'])), (0,0,255),
                              cv2.MARKER_CROSS, 30, 2)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f'{project_name}_p{page_idx+1}.png')
            small = cv2.resize(vis, (vis.shape[1]//2, vis.shape[0]//2))
            cv2.imwrite(out_path, small)

    pos_r = total['tp_pos'] / max(total['gt_count'], 1)
    full_r = total['tp_full'] / max(total['gt_count'], 1)
    pos_p = total['tp_pos'] / max(total['det_count'], 1)
    full_p = total['tp_full'] / max(total['det_count'], 1)

    return {
        'project': project_name,
        'pages_tested': len(page_results),
        'gt_count': total['gt_count'],
        'det_count': total['det_count'],
        'pos_recall': pos_r,
        'full_recall': full_r,
        'pos_precision': pos_p,
        'full_precision': full_p,
        'tp_pos': total['tp_pos'],
        'tp_full': total['tp_full'],
        'fp': total['fp'],
        'fn': total['fn'],
        'gt_classes': dict(gt_class_counts),
        'det_classes': dict(det_class_counts),
        'confusion_pairs': confusion_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/hvac_yolov8s_v9.pt')
    parser.add_argument('--projects', nargs='+', help='Project ID prefixes')
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--save-confusion', default='output/confusion_data.json',
                        help='Path to save raw confusion pair data')
    args = parser.parse_args()

    from ultralytics import YOLO
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    all_projects = sorted(os.listdir(PROJECTS_DIR))
    if args.projects:
        projects = [p for p in all_projects if any(p.startswith(pid) for pid in args.projects)]
    else:
        projects = all_projects

    print(f"Benchmarking {len(projects)} projects with conf={args.conf}\n")

    print(f"{'Project':<35}{'Pages':<8}{'GT':<6}{'Det':<6}{'Pos R':<8}{'Full R':<8}{'Verdict'}")
    print('-' * 90)

    results = []
    for proj in projects:
        proj_dir = os.path.join(PROJECTS_DIR, proj)
        try:
            r = benchmark_project(model, proj_dir, proj, conf=args.conf, save_viz=not args.no_viz)
            if r is None:
                print(f"{proj:<35}{'-':<8}{'NO LABELED DATA'}")
                continue
            verdict = '✓ EXCELLENT' if r['full_recall'] >= 0.8 else \
                      '~ OK' if r['full_recall'] >= 0.6 else \
                      '✗ POOR' if r['full_recall'] >= 0.3 else '✗✗ FAILED'
            print(f"{proj[:34]:<35}{r['pages_tested']:<8}{r['gt_count']:<6}{r['det_count']:<6}{r['pos_recall']:<8.0%}{r['full_recall']:<8.0%}{verdict}")
            results.append(r)
        except Exception as e:
            print(f"{proj:<35}ERROR: {e}")

    # Summary
    if results:
        print('\n' + '=' * 90)
        print('OVERALL')
        print('=' * 90)
        total_gt = sum(r['gt_count'] for r in results)
        total_tp_pos = sum(r['tp_pos'] for r in results)
        total_tp_full = sum(r['tp_full'] for r in results)
        total_det = sum(r['det_count'] for r in results)
        print(f"  Projects benchmarked: {len(results)}")
        print(f"  Total ground truth: {total_gt}")
        print(f"  Total detections:   {total_det}")
        print(f"  Position recall:    {total_tp_pos/max(total_gt,1):.1%}")
        print(f"  Full match recall:  {total_tp_full/max(total_gt,1):.1%}")
        print(f"  Position precision: {total_tp_pos/max(total_det,1):.1%}")
        print(f"  Full match prec:    {total_tp_full/max(total_det,1):.1%}")

        # Buckets
        excellent = sum(1 for r in results if r['full_recall'] >= 0.8)
        ok = sum(1 for r in results if 0.6 <= r['full_recall'] < 0.8)
        poor = sum(1 for r in results if 0.3 <= r['full_recall'] < 0.6)
        failed = sum(1 for r in results if r['full_recall'] < 0.3)
        print(f"\n  Excellent (≥80%): {excellent}")
        print(f"  OK (60-80%):      {ok}")
        print(f"  Poor (30-60%):    {poor}")
        print(f"  Failed (<30%):    {failed}")

        if not args.no_viz:
            print(f"\n  Visualizations saved to: {OUTPUT_DIR}")

        # Save confusion data
        if args.save_confusion:
            import json
            os.makedirs(os.path.dirname(args.save_confusion), exist_ok=True)
            data = {
                'model': args.model,
                'conf_threshold': args.conf,
                'projects': []
            }
            for r in results:
                data['projects'].append({
                    'project': r['project'],
                    'gt_count': r['gt_count'],
                    'det_count': r['det_count'],
                    'pos_recall': r['pos_recall'],
                    'full_recall': r['full_recall'],
                    'confusion_pairs': r.get('confusion_pairs', []),
                })
            with open(args.save_confusion, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n  Confusion data saved to: {args.save_confusion}")

    return results


if __name__ == "__main__":
    main()
