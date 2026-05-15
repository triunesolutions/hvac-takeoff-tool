"""
Tag-label verification tool.

Run:
    python verify_tags.py --dataset path/to/tag_dataset --labels tag_bubble_labels.jsonl

Then open http://localhost:5000 in a browser.

Each entry in tag_bubble_labels.jsonl has:
    {"img": "images/Project_Name/000123.png", "tag": "VAV-23", "reason": "hit"|"no_match"|"no_text", ...}

The verifier shows one crop at a time. The reviewer either:
  - Confirms the expected tag (Y / Enter)        → saved as confirmed
  - Marks "no visible tag" (N)                   → saved as no_tag
  - Skips because unclear (S)                    → not saved, can revisit
  - Types the correct tag and presses Enter      → saved as corrected

Output: tag_bubble_labels_verified.jsonl (append-mode, resumable). One row
per decision:
    {"img": "...", "tag": "VAV-23", "decision": "confirm"|"correct"|"no_tag",
     "original_tag": "VAV-23", "reviewer_tag": "VAV-23"|null, "ts": "..."}

To resume, just relaunch — already-verified images are skipped automatically.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_from_directory, abort
except ImportError:
    print("ERROR: Flask not installed. Run: pip install Flask", file=sys.stderr)
    sys.exit(1)


app = Flask(__name__)


CONFIG = {
    'dataset_root': None,   # base dir containing 'images/'
    'labels_path': None,
    'output_path': None,
    'entries': [],          # all entries from labels file
    'verified': {},         # img → decision dict (loaded from output_path)
    'show_all': False,      # show every entry, even already-verified
}


HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HVAC Tag Verifier</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
         max-width: 900px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  .progress { color: #666; font-size: 14px; margin-bottom: 16px; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px;
          background: #fafafa; }
  .crop { background: white; border: 1px solid #eee; padding: 8px;
          display: inline-block; margin-bottom: 12px; }
  .crop img { display: block; max-width: 100%; image-rendering: pixelated; }
  .meta { font-size: 14px; color: #555; margin-bottom: 12px; }
  .meta b { color: #222; }
  .tag-display { font-size: 28px; font-weight: 600; letter-spacing: 1px;
                 background: #fff7c0; padding: 6px 14px; border-radius: 4px;
                 border: 1px solid #e6d76b; display: inline-block; }
  .reason-hit    { color: #1a7f37; }
  .reason-no_match { color: #c41a1a; }
  .reason-no_text  { color: #888; }
  .buttons { margin: 18px 0; }
  button { font-size: 15px; padding: 8px 16px; margin-right: 8px;
           border: 1px solid #888; background: #fff; border-radius: 4px;
           cursor: pointer; }
  button:hover { background: #f0f0f0; }
  button.primary { background: #1a7f37; color: white; border-color: #1a7f37; }
  button.warn    { background: #c41a1a; color: white; border-color: #c41a1a; }
  button.skip    { background: #888; color: white; border-color: #888; }
  .correction { margin-top: 12px; }
  .correction input { font-size: 17px; padding: 6px 10px; width: 220px;
                      border: 1px solid #888; border-radius: 4px; }
  .hint { color: #888; font-size: 13px; margin-top: 12px; }
  .hint kbd { background: #eee; padding: 1px 6px; border-radius: 3px;
              border: 1px solid #ccc; font-family: monospace; font-size: 12px; }
  .done { color: #1a7f37; font-weight: 600; }
</style>
</head>
<body>
<h1>HVAC Tag Verifier</h1>
<div class="progress" id="progress">loading...</div>
<div class="card" id="card">
  <div class="crop"><img id="cropImg" src="" alt="crop"></div>
  <div class="meta">
    Project: <b id="project">—</b><br>
    Reason: <span id="reason">—</span> &nbsp;|&nbsp; Image: <span id="imgPath">—</span>
  </div>
  <div>
    Expected tag: <span class="tag-display" id="expectedTag">—</span>
  </div>
  <div class="buttons">
    <button class="primary" onclick="decide('confirm')">✓ Correct (Y / Enter)</button>
    <button class="warn"    onclick="decide('no_tag')">✗ No tag visible (N)</button>
    <button class="skip"    onclick="skipEntry()">↪ Skip (S)</button>
  </div>
  <div class="correction">
    Or type the correct tag and press Enter:
    <input id="correctInput" placeholder="e.g. VAV-23"
           autocomplete="off" spellcheck="false">
    <button onclick="submitCorrection()">Save correction</button>
  </div>
  <div class="hint">
    Keys: <kbd>Y</kbd> confirm · <kbd>N</kbd> no tag visible · <kbd>S</kbd> skip ·
    Type then <kbd>Enter</kbd> to correct · <kbd>←</kbd> previous
  </div>
</div>
<script>
let state = { current: null, total: 0, done: 0 };

async function fetchNext() {
  const r = await fetch('/api/next');
  const d = await r.json();
  if (d.finished === true) {
    const skipNote = d.skipped_count > 0
      ? ' (' + d.skipped_count + ' skipped — restart to revisit)' : '';
    document.getElementById('card').innerHTML =
      '<div class="done">No more entries to verify. ' +
      d.verified_count + ' of ' + d.total + ' actually decided' + skipNote +
      '. Output: ' + d.output + '</div>';
    document.getElementById('progress').textContent = '';
    return;
  }
  state.current = d.entry;
  state.total = d.total;
  state.done = d.verified_count;
  document.getElementById('progress').innerHTML =
    '<b>' + d.verified_count + '</b> of ' + d.total + ' verified (' +
    (100*d.verified_count/d.total).toFixed(1) + '%) · ' +
    d.remaining + ' remaining';
  document.getElementById('cropImg').src = '/img/' + encodeURIComponent(d.entry.img);
  document.getElementById('project').textContent = d.entry.img.split('/').slice(-2,-1)[0];
  document.getElementById('imgPath').textContent = d.entry.img;
  const reason = d.entry.reason || 'unknown';
  const rEl = document.getElementById('reason');
  rEl.textContent = reason;
  rEl.className = 'reason-' + reason;
  document.getElementById('expectedTag').textContent = d.entry.tag || '(none)';
  document.getElementById('correctInput').value = '';
  document.getElementById('correctInput').blur();
}

let busy = false;  // prevent double-submit from Enter + button click
async function decide(decision, reviewer_tag) {
  if (!state.current || busy) return;
  busy = true;
  const payload = { img: state.current.img, decision: decision };
  if (reviewer_tag) payload.reviewer_tag = reviewer_tag;
  try {
    await fetch('/api/decide', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    await fetchNext();
  } finally {
    busy = false;
  }
}

function submitCorrection() {
  const v = document.getElementById('correctInput').value.trim();
  if (!v) { alert('Type the correct tag first'); return; }
  decide('correct', v);
}

function skipEntry() {
  if (!state.current) return;
  fetch('/api/skip', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ img: state.current.img })
  }).then(fetchNext);
}

async function goBack() {
  await fetch('/api/back', { method: 'POST' });
  fetchNext();
}

document.addEventListener('keydown', (e) => {
  if (document.activeElement && document.activeElement.id === 'correctInput') {
    if (e.key === 'Enter') { e.preventDefault(); submitCorrection(); }
    if (e.key === 'Escape') document.getElementById('correctInput').blur();
    return;
  }
  if (e.key === 'y' || e.key === 'Y' || e.key === 'Enter') {
    e.preventDefault(); decide('confirm');
  } else if (e.key === 'n' || e.key === 'N') {
    e.preventDefault(); decide('no_tag');
  } else if (e.key === 's' || e.key === 'S') {
    e.preventDefault(); skipEntry();
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault(); goBack();
  } else if (/^[a-zA-Z0-9]$/.test(e.key)) {
    // Start typing → focus the input
    const inp = document.getElementById('correctInput');
    inp.focus();
    inp.value = e.key;
    e.preventDefault();
  }
});

fetchNext();
</script>
</body>
</html>
"""


# ─── Persistence ──────────────────────────────────────────────────────────


def _load_verified(output_path):
    """Read previously-saved decisions so we can resume."""
    verified = {}
    if not output_path.exists():
        return verified
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                verified[row['img']] = row
            except Exception:
                continue
    return verified


def _append_verified(output_path, row):
    with open(output_path, 'a', encoding='utf-8', buffering=1) as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


# ─── Routes ───────────────────────────────────────────────────────────────


@app.route('/')
def home():
    return HTML


@app.route('/img/<path:rel>')
def serve_image(rel):
    root = Path(CONFIG['dataset_root']).resolve()
    abspath = (root / rel).resolve()
    if not str(abspath).startswith(str(root)):
        abort(404)
    if not abspath.exists():
        abort(404)
    return send_from_directory(str(abspath.parent), abspath.name)


@app.route('/api/next', methods=['GET'])
def api_next():
    entries = CONFIG['entries']
    verified = CONFIG['verified']
    skipped = CONFIG.get('skipped_this_session', set())
    total = len(entries)
    done = len(verified)
    skipped_n = len(skipped)
    remaining = total - done - skipped_n
    print(f"[api_next] total={total} done={done} skipped_session={skipped_n} remaining={remaining}",
          flush=True)
    for entry in entries:
        if entry['img'] in verified:
            continue
        if entry['img'] in skipped:
            continue
        return jsonify({
            'finished': False,
            'entry': entry,
            'total': total,
            'verified_count': done,
            'skipped_count': skipped_n,
            'remaining': remaining,
            'output': str(CONFIG['output_path']),
        })
    return jsonify({
        'finished': True,
        'total': total,
        'verified_count': done,
        'skipped_count': skipped_n,
        'output': str(CONFIG['output_path']),
    })


@app.route('/api/decide', methods=['POST'])
def api_decide():
    data = request.get_json(force=True)
    img = data.get('img')
    decision = data.get('decision')
    print(f"[api_decide] img={img!r} decision={decision!r}", flush=True)
    if not img or decision not in ('confirm', 'no_tag', 'correct'):
        print(f"  → 400 bad request", flush=True)
        return jsonify({'error': 'bad request'}), 400

    entry = next((e for e in CONFIG['entries'] if e['img'] == img), None)
    if entry is None:
        print(f"  → 404 unknown img (entries has {len(CONFIG['entries'])} items)", flush=True)
        return jsonify({'error': 'unknown img'}), 404

    reviewer_tag = data.get('reviewer_tag')
    row = {
        'img': img,
        'original_tag': entry.get('tag'),
        'decision': decision,
        'reviewer_tag': reviewer_tag if decision == 'correct' else (entry.get('tag') if decision == 'confirm' else None),
        'reason': entry.get('reason'),
        'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    CONFIG['verified'][img] = row
    _append_verified(CONFIG['output_path'], row)
    # Re-add to recent stack so user can go back
    CONFIG.setdefault('recent', []).append(img)
    return jsonify({'ok': True})


@app.route('/api/skip', methods=['POST'])
def api_skip():
    data = request.get_json(force=True)
    img = data.get('img')
    if not img:
        return jsonify({'error': 'bad request'}), 400
    CONFIG.setdefault('skipped_this_session', set()).add(img)
    return jsonify({'ok': True})


@app.route('/api/back', methods=['POST'])
def api_back():
    recent = CONFIG.get('recent', [])
    if not recent:
        return jsonify({'ok': False, 'msg': 'no history'})
    last = recent.pop()
    CONFIG['verified'].pop(last, None)
    # We can't easily un-append from the jsonl, so we just remove from
    # in-memory state. The output file will have a duplicate decision —
    # downstream consumers can dedupe by taking the LAST entry per img.
    return jsonify({'ok': True})


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    help='Path to the unzipped tag_dataset directory '
                         '(the one containing images/<project>/...)')
    ap.add_argument('--labels', default='tag_bubble_labels.jsonl',
                    help='Path to the auto-labels JSONL')
    ap.add_argument('--output', default='tag_bubble_labels_verified.jsonl',
                    help='Where to append verified rows (resumable)')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--only', choices=['hit', 'no_match', 'no_text'],
                    help="Only review entries with this reason "
                         "(e.g. --only hit to verify EasyOCR's positive labels first)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not (dataset_root / 'images').is_dir() and not any(
            (dataset_root / d).is_dir() for d in os.listdir(dataset_root) if (dataset_root / d).is_dir()):
        print(f"ERROR: --dataset {dataset_root} doesn't look right "
              f"(no 'images/' subdir). Expected the unzipped tag_dataset/.",
              file=sys.stderr)
        sys.exit(1)

    labels_path = Path(args.labels).resolve()
    if not labels_path.exists():
        print(f"ERROR: labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if args.only and row.get('reason') != args.only:
                continue
            entries.append(row)

    output_path = Path(args.output).resolve()
    verified = _load_verified(output_path)

    CONFIG['dataset_root'] = str(dataset_root)
    CONFIG['labels_path'] = str(labels_path)
    CONFIG['output_path'] = output_path
    CONFIG['entries'] = entries
    CONFIG['verified'] = verified
    CONFIG['skipped_this_session'] = set()
    CONFIG['recent'] = []

    print(f"\nHVAC Tag Verifier")
    print(f"  dataset:  {dataset_root}")
    print(f"  labels:   {labels_path} ({len(entries)} entries"
          f"{' filtered to reason=' + args.only if args.only else ''})")
    print(f"  output:   {output_path} ({len(verified)} already verified)")
    print(f"\n  Open http://localhost:{args.port} in your browser.\n")
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=False)


if __name__ == '__main__':
    main()
