"""Quick sanity check: load the fine-tuned rec head and run it on val crops.
Reports char-level + exact-match accuracy on a 100-sample slice."""
import sys, random
from pathlib import Path
import cv2

from paddleocr import PaddleOCR

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'models' / 'rec_ppocr_v4_hvac'

ocr = PaddleOCR(
    rec_model_dir=str(MODEL_DIR),
    rec_char_dict_path=str(MODEL_DIR / 'dict.txt'),
    rec_image_shape='3, 48, 320',
    rec_algorithm='SVTR_LCNet',
    use_angle_cls=False,
    lang='en',
    use_gpu=False,
    show_log=False,
)

val_lines = (ROOT / 'ocr_finetune' / 'val.txt').read_text().strip().splitlines()
random.seed(0)
sample = random.sample(val_lines, min(100, len(val_lines)))

exact = 0
char_correct = 0
char_total = 0
mismatches = []
for line in sample:
    rel, truth = line.split('\t')
    img = cv2.imread(str(ROOT / 'ocr_finetune' / rel))
    res = ocr.ocr(img, det=False, cls=False, rec=True)
    # paddleocr returns [[(text, conf)]] for rec-only
    pred = res[0][0][0] if res and res[0] else ''
    conf = res[0][0][1] if res and res[0] else 0.0
    if pred == truth:
        exact += 1
    else:
        mismatches.append((truth, pred, conf))
    # char-level
    m = min(len(pred), len(truth))
    char_correct += sum(1 for i in range(m) if pred[i] == truth[i])
    char_total += max(len(pred), len(truth))

print(f"Exact match: {exact}/{len(sample)} = {100*exact/len(sample):.1f}%")
print(f"Char accuracy: {char_correct}/{char_total} = {100*char_correct/char_total:.1f}%")
print(f"\nFirst 10 mismatches (truth -> pred [conf]):")
for t, p, c in mismatches[:10]:
    print(f"  {t!r:12s} -> {p!r:12s} [{c:.2f}]")
