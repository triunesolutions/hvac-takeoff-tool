"""Quick comparison: pdfplumber vs text-layer vs OCR fallback per project."""
import sys
import time
from collections import Counter
from pathlib import Path

from schedule_parser import parse_pdf_schedules

if len(sys.argv) < 2:
    print("Usage: python _compare_parsers.py <pdf>")
    sys.exit(1)

pdf = Path(sys.argv[1])
print(f"=== {pdf.parent.parent.name} ===")
t0 = time.time()
schedules, marks, details, legend, summary, variables = parse_pdf_schedules(
    str(pdf), ocr_fallback=False)
elapsed = time.time() - t0
via = Counter(v.get('extracted_via', 'pdfplumber') for v in variables)
print(f"  variables: {len(variables)} total  ({elapsed:.1f}s)")
for src, n in via.most_common():
    print(f"    {src}: {n}")
