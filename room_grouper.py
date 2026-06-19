"""
room_grouper.py — "Automatic Room Search" (Rebar parity)

Group equipment detections by room. For each detection, find the nearest
room label on the same page and assign it to that room.

V1 strategy (text-only, no wall detection):
  1. Extract every text span on the page.
  2. Filter to spans that look like room labels:
       - room numbers: \\b\\d{2,4}[A-Z]?\\b  (e.g. 101, 102A, 1234)
       - room names: CONFERENCE, OFFICE, LOBBY, STORAGE, RESTROOM, etc.
       - prefixed numbers: ROOM 101, RM 102
  3. For each detection, assign to nearest room label by Euclidean distance
     between their center points. PDF coords (points) used throughout.
  4. Optionally enforce a max distance; beyond that the detection is
     labeled UNASSIGNED.

Usage:
    python room_grouper.py --detections "<*_detections.json>" --pdf "<plan.pdf>"

Output:
    <pdf-stem>_rooms.json  — per detection: nearest room + distance
    <pdf-stem>_rooms.csv   — room-by-class count matrix
"""

import argparse
import json
import math
import re
from collections import defaultdict, Counter
from pathlib import Path

import fitz


# --- Room label patterns ---

# Common HVAC drawing room names (extend as needed)
ROOM_NAME_KEYWORDS = (
    'OFFICE', 'CONFERENCE', 'CONF', 'LOBBY', 'STORAGE', 'STOR', 'CORRIDOR',
    'CORR', 'STAIR', 'ELEC', 'ELECTRICAL', 'MECH', 'MECHANICAL', 'JANITOR',
    'RESTROOM', 'TOILET', 'KITCHEN', 'BREAK', 'BREAKROOM', 'LOUNGE',
    'CLASSROOM', 'CLASS', 'LIBRARY', 'GYMNASIUM', 'GYM', 'CAFE',
    'CAFETERIA', 'AUDITORIUM', 'RECEPTION', 'WAITING', 'EXAM', 'PROCEDURE',
    'OR', 'PRE-OP', 'POST-OP', 'IT', 'SERVER', 'WORK', 'WORKROOM',
    'LOCKER', 'SHOWER', 'CHANGING', 'LOBBY', 'VESTIBULE', 'ENTRY',
    'COMMUNITY', 'STUDIO', 'CLEAN', 'SOILED', 'LAB',
)

ROOM_NAME_RE = re.compile(
    r'\b(?:' + '|'.join(ROOM_NAME_KEYWORDS) + r')\b',
    re.IGNORECASE,
)

# Room number patterns (avoid matching tag-like things)
ROOM_NUMBER_RE = re.compile(r'^\d{2,4}[A-Z]?$')
ROOM_NUMBER_PREFIXED_RE = re.compile(r'\b(?:ROOM|RM\.?)\s*[#:.]?\s*(\d{2,4}[A-Z]?)\b', re.IGNORECASE)

# Tag patterns to EXCLUDE (these look like room numbers but aren't)
TAG_LIKE_RE = re.compile(r'^[A-Z]{1,4}-?\d{1,3}[A-Z]?$')


def extract_room_labels(page):
    """Return list of {text, type, cx, cy, bbox} candidates."""
    candidates = []
    d = page.get_text('dict')
    for block in d.get('blocks', []):
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text = span.get('text', '').strip()
                if not text:
                    continue
                bbox = span.get('bbox')
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Skip obvious tags like CU-1, A1, EF-2
                if TAG_LIKE_RE.match(text):
                    continue

                # Prefixed room number wins (highest confidence)
                m = ROOM_NUMBER_PREFIXED_RE.search(text)
                if m:
                    candidates.append({
                        'text': f'ROOM {m.group(1)}',
                        'type': 'room_number_prefixed',
                        'cx': cx, 'cy': cy, 'bbox': bbox,
                        'confidence': 0.95,
                    })
                    continue

                # Plain room number (e.g. "101", "102A", "1234")
                # Heuristic: standalone span that is just a 2-4 digit number
                if ROOM_NUMBER_RE.match(text):
                    # Be conservative: only count as room number if at least 3 digits
                    # (avoids confusing with figure numbers or scale denominators)
                    if len(text.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) >= 3:
                        candidates.append({
                            'text': text,
                            'type': 'room_number',
                            'cx': cx, 'cy': cy, 'bbox': bbox,
                            'confidence': 0.7,
                        })
                    continue

                # Room name keyword
                m = ROOM_NAME_RE.search(text)
                if m and len(text) < 50:
                    candidates.append({
                        'text': text,
                        'type': 'room_name',
                        'cx': cx, 'cy': cy, 'bbox': bbox,
                        'confidence': 0.8,
                    })

    return candidates


# --- Assignment ---

def assign_detections(detections, rooms, max_distance=None):
    """Assign each detection to nearest room label.

    detections: list of dicts with 'cx','cy' (or x1,y1,x2,y2) in PDF coords
    rooms: list of room labels as produced by extract_room_labels
    """
    assignments = []
    for det in detections:
        # Derive center if not given
        if 'cx' not in det:
            dcx = (det['x1'] + det['x2']) / 2
            dcy = (det['y1'] + det['y2']) / 2
        else:
            dcx, dcy = det['cx'], det['cy']

        if not rooms:
            assignments.append({**det, 'room': None, 'room_distance': None})
            continue

        best_room, best_d = None, float('inf')
        for r in rooms:
            d = math.hypot(dcx - r['cx'], dcy - r['cy'])
            if d < best_d:
                best_d, best_room = d, r

        if max_distance is not None and best_d > max_distance:
            assignments.append({**det, 'room': 'UNASSIGNED', 'room_distance': round(best_d, 1)})
        else:
            assignments.append({
                **det,
                'room': best_room['text'],
                'room_type': best_room['type'],
                'room_distance': round(best_d, 1),
            })

    return assignments


# --- CLI ---

def detections_to_pdf_coords(det_records, dpi):
    """Convert image-space detections to PDF coords by inverse-scaling."""
    scale = 72.0 / dpi
    out = []
    for d in det_records:
        out.append({
            **d,
            'x1': d['x1'] * scale, 'y1': d['y1'] * scale,
            'x2': d['x2'] * scale, 'y2': d['y2'] * scale,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detections', required=True,
                    help='Path to *_detections.json sidecar from takeoff_cli or benchmark')
    ap.add_argument('--pdf', required=True,
                    help='Source PDF (needed to read room labels)')
    ap.add_argument('--dpi', type=int, default=200,
                    help='DPI the detections were produced at (default 200)')
    ap.add_argument('--max-distance-pt', type=float, default=None,
                    help='Max PDF-pt distance to assign; further -> UNASSIGNED')
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    det_path = Path(args.detections)
    out_dir = Path(args.output_dir) if args.output_dir else pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(det_path.read_text(encoding='utf-8'))

    # Normalize the various detections-sidecar formats into
    # by_page: dict[int -> list[detection]]
    by_page = defaultdict(list)

    def _norm(rec):
        # takeoff_cli uses 'cls'; benchmark uses 'class'/'norm_class'
        out = dict(rec)
        if 'class' not in out and 'cls' in out:
            out['class'] = out['cls']
        return out

    if isinstance(raw, dict) and 'pages' in raw:
        # takeoff_cli.py format: {pdf, dpi, pages: {"12": [det, det, ...]}}
        # Use the dpi embedded in the file if present
        dpi = raw.get('dpi', args.dpi)
        for page_str, dets in raw['pages'].items():
            pno = int(page_str)
            for d in dets:
                by_page[pno].append(_norm(d))
    elif isinstance(raw, dict) and 'detections' in raw:
        dpi = raw.get('dpi', args.dpi)
        for d in raw['detections']:
            by_page[d.get('page', 1)].append(_norm(d))
    elif isinstance(raw, list):
        dpi = args.dpi
        for d in raw:
            by_page[d.get('page', 1)].append(_norm(d))
    else:
        raise SystemExit(f'Unrecognized detections.json structure: keys={list(raw)[:5] if isinstance(raw, dict) else type(raw).__name__}')

    # Convert image-space coords -> PDF points to match get_text bboxes
    for pno in list(by_page):
        by_page[pno] = detections_to_pdf_coords(by_page[pno], dpi)

    doc = fitz.open(str(pdf))
    out_records = []
    summary_room_class = defaultdict(Counter)

    for pno in sorted(by_page):
        page = doc[pno - 1]
        rooms = extract_room_labels(page)
        page_assigned = assign_detections(by_page[pno], rooms,
                                          max_distance=args.max_distance_pt)
        out_records.extend(page_assigned)
        for a in page_assigned:
            room = a.get('room') or 'UNASSIGNED'
            cls = a.get('class') or a.get('norm_class') or '?'
            summary_room_class[(pno, room)][cls] += 1
        print(f'page {pno}: {len(rooms)} room labels, {len(by_page[pno])} detections')

    doc.close()

    # Write per-detection JSON
    out_json = out_dir / f'{pdf.stem}_rooms.json'
    out_json.write_text(json.dumps(out_records, indent=2), encoding='utf-8')

    # Write room×class CSV
    out_csv = out_dir / f'{pdf.stem}_rooms.csv'
    all_classes = sorted({cls for c in summary_room_class.values() for cls in c})
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('page,room,' + ','.join(f'"{c}"' for c in all_classes) + ',TOTAL\n')
        for (pno, room) in sorted(summary_room_class):
            cnts = summary_room_class[(pno, room)]
            total = sum(cnts.values())
            row = [str(pno), f'"{room}"'] + [str(cnts.get(c, 0)) for c in all_classes] + [str(total)]
            f.write(','.join(row) + '\n')

    print()
    print(f'Wrote {out_json.name} ({len(out_records)} detections assigned)')
    print(f'Wrote {out_csv.name} ({len(summary_room_class)} unique (page, room) rows)')

    # Console summary: top rooms
    room_totals = Counter()
    for (pno, room), cnts in summary_room_class.items():
        room_totals[(pno, room)] = sum(cnts.values())
    print()
    print('=== Top 10 (page, room) by equipment count ===')
    for (pno, room), n in room_totals.most_common(10):
        cls_breakdown = ', '.join(f'{c}={n2}' for c, n2 in summary_room_class[(pno, room)].most_common(4))
        print(f'  p{pno} {room[:30]:30s} ({n:>3d})  {cls_breakdown}')


if __name__ == '__main__':
    main()
