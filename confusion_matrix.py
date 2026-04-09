"""
Confusion Matrix Analyzer

Reads confusion data from benchmark.py output (output/confusion_data.json)
and produces:
1. Per-class precision/recall table
2. Top confusion pairs (what gets mislabeled as what)
3. CSV export of the full matrix
4. Recommendations for class_aliases.py tuning

Usage:
    python confusion_matrix.py                       # Use default JSON
    python confusion_matrix.py path/to/data.json     # Custom JSON
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import argparse
from collections import defaultdict, Counter


def analyze(json_path, output_csv='output/confusion_matrix.csv'):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Aggregate all confusion pairs across all projects
    all_pairs = []  # list of (predicted, actual)
    per_project = defaultdict(list)
    for proj in data['projects']:
        pairs = proj.get('confusion_pairs', [])
        all_pairs.extend(pairs)
        per_project[proj['project']] = pairs

    print(f"Model: {data['model']}")
    print(f"Conf threshold: {data['conf_threshold']}")
    print(f"Projects: {len(data['projects'])}")
    print(f"Total matched detections: {len(all_pairs)}")
    print()

    # Build confusion matrix: predicted -> actual -> count
    matrix = defaultdict(lambda: defaultdict(int))
    actual_counts = Counter()
    pred_counts = Counter()

    for predicted, actual in all_pairs:
        matrix[predicted][actual] += 1
        actual_counts[actual] += 1
        pred_counts[predicted] += 1

    all_classes = sorted(set(list(actual_counts.keys()) + list(pred_counts.keys())))

    # Per-class precision and recall
    print("=" * 100)
    print("PER-CLASS ACCURACY")
    print("=" * 100)
    print(f"{'Class':<40}{'Total GT':<12}{'Correct':<12}{'Recall':<12}{'Precision':<12}")
    print('-' * 100)

    class_stats = []
    for cls in all_classes:
        gt_total = actual_counts.get(cls, 0)
        if gt_total == 0:
            continue
        # Recall: of all GT instances of this class, how many did we correctly predict?
        correct = matrix[cls].get(cls, 0)
        recall = correct / gt_total if gt_total > 0 else 0
        # Precision: of all predictions of this class, how many were right?
        pred_total = pred_counts.get(cls, 0)
        precision = correct / pred_total if pred_total > 0 else 0
        class_stats.append({
            'class': cls,
            'gt': gt_total,
            'correct': correct,
            'recall': recall,
            'precision': precision,
            'pred_total': pred_total,
        })

    # Sort by GT count descending (most common first)
    class_stats.sort(key=lambda x: -x['gt'])
    for s in class_stats:
        marker = ' ✓' if s['recall'] >= 0.7 and s['precision'] >= 0.7 else \
                 ' ~' if s['recall'] >= 0.4 or s['precision'] >= 0.4 else ' ✗'
        print(f"{s['class'][:39]:<40}{s['gt']:<12}{s['correct']:<12}{s['recall']:<12.0%}{s['precision']:<12.0%}{marker}")

    # Top confusion pairs (where the model is wrong)
    print()
    print("=" * 100)
    print("TOP CONFUSION PAIRS — when the model is WRONG")
    print("=" * 100)
    print(f"{'Predicted as':<35}{'Actually was':<35}{'Count':<8}{'% of actual class'}")
    print('-' * 100)

    confusion_errors = []
    for predicted, actuals in matrix.items():
        for actual, count in actuals.items():
            if predicted != actual:  # Only errors
                pct_of_actual = count / actual_counts[actual] if actual_counts[actual] > 0 else 0
                confusion_errors.append({
                    'predicted': predicted,
                    'actual': actual,
                    'count': count,
                    'pct': pct_of_actual,
                })

    confusion_errors.sort(key=lambda x: -x['count'])
    for e in confusion_errors[:30]:
        print(f"{e['predicted'][:34]:<35}{e['actual'][:34]:<35}{e['count']:<8}{e['pct']:.0%}")

    # Recommendations
    print()
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    # Pattern 1: Classes that get heavily confused with each other
    paired_confusion = defaultdict(int)
    for e in confusion_errors:
        key = tuple(sorted([e['predicted'], e['actual']]))
        paired_confusion[key] += e['count']

    print("\n1. Classes that are systematically confused (consider merging if visually equivalent):")
    for (a, b), count in sorted(paired_confusion.items(), key=lambda x: -x[1])[:10]:
        if count >= 5:
            print(f"   {a} <-> {b}: {count} confusions")

    # Pattern 2: Classes that are over-predicted (low precision)
    print("\n2. Classes the model OVER-predicts (low precision = many false positives):")
    over_predicted = sorted(class_stats, key=lambda x: x['precision'])[:5]
    for s in over_predicted:
        if s['precision'] < 0.5 and s['pred_total'] > 5:
            print(f"   {s['class']}: {s['precision']:.0%} precision ({s['pred_total']} predictions, {s['correct']} correct)")

    # Pattern 3: Classes that are under-detected (low recall)
    print("\n3. Classes the model MISSES (low recall = should detect but doesn't):")
    under_detected = sorted(class_stats, key=lambda x: x['recall'])[:5]
    for s in under_detected:
        if s['recall'] < 0.5 and s['gt'] > 5:
            print(f"   {s['class']}: {s['recall']:.0%} recall ({s['correct']} of {s['gt']} found)")

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w') as f:
        # Header
        f.write('predicted_class,actual_class,count\n')
        for predicted in sorted(matrix.keys()):
            for actual in sorted(matrix[predicted].keys()):
                f.write(f'{predicted},{actual},{matrix[predicted][actual]}\n')
    print(f"\nFull confusion matrix CSV: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', nargs='?', default='output/confusion_data.json')
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"ERROR: {args.json_path} not found.")
        print("Run benchmark.py first to generate confusion data:")
        print("  python benchmark.py --projects 04 23 28 30 36 --save-confusion output/confusion_data.json")
        sys.exit(1)

    analyze(args.json_path)
