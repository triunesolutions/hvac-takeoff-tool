"""Unit tests for schedule_parser.normalize_tag — the single source of truth for
"is this string a valid equipment tag".

normalize_tag is ~100 lines with ~10 return paths and historically had no tests,
so it was risky to extend. These lock in the known-good / known-bad cases pulled
from real schedules and the failure modes documented in CLAUDE.md §10.

Run standalone (no pytest needed):
    python tests/test_normalize_tag.py
Or under pytest:
    pytest tests/test_normalize_tag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schedule_parser import normalize_tag


# (input, expected normalized output) — accepted tags
GOOD = [
    ("A", "A"),
    ("b", "B"),
    ("C", "C"),
    ("D", "D"),
    ("A1", "A1"),
    ("cu-1", "CU-1"),
    ("FCU-10", "FCU-10"),
    ("VAV-N1", "VAV-N1"),
    ("LD-1", "LD-1"),
    ("GR-2", "GR-2"),
    ("EF-1", "EF-1"),
    ("RTU-2", "RTU-2"),
    ("AHU-1", "AHU-1"),
    ("CD", "CD"),          # bare air-device mark
    ("RG", "RG"),
    ("(E) CU-1", "CU-1"),  # equipment-status prefix stripped
]

# Inputs that must be rejected (normalize_tag returns None)
BAD = [
    "",
    "   ",
    "123",            # pure number
    "R-410A",         # refrigerant
    "R-32",           # refrigerant
    "M101",           # drawing sheet number
    "E202",
    "P301",
    "RKF12AXVJU",     # model number (long, no hyphen, many letters)
    "NOTES-1",        # banned column-leak prefix
    "REV-2",
    "VAV",            # prefix only, no number (junk)
    "U-C",            # letter-suffix with non-HVAC prefix
]


def test_good_tags():
    for raw, expected in GOOD:
        got = normalize_tag(raw)
        assert got == expected, f"normalize_tag({raw!r}) = {got!r}, expected {expected!r}"


def test_bad_tags():
    for raw in BAD:
        got = normalize_tag(raw)
        assert got is None, f"normalize_tag({raw!r}) = {got!r}, expected None"


if __name__ == "__main__":
    test_good_tags()
    test_bad_tags()
    print(f"OK — {len(GOOD)} good + {len(BAD)} bad cases passed")
