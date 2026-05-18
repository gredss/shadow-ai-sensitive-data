# -*- coding: utf-8 -*-
"""
Utility functions for DLP Benchmark
Common helpers and formatting functions
"""

import re
from typing import List


def print_banner(title: str):
    """Print a formatted banner for section headers."""
    print("\n" + "═" * 75)
    print(f"  {title}")
    print("═" * 75)


def normalize_val(v: str) -> str:
    """Strip spaces, dashes, leading + and 0 for comparison."""
    return re.sub(r"[\s\-]", "", str(v)).lstrip("+").lstrip("0")


def is_hit(gt_val: str, detected_list: list) -> bool:
    """Return True if gt_val is found in detected_list under normalized exact match.
    
    Partial prefix match (≥10 digits) is also accepted for LLM outputs
    that may truncate leading-zero sequences.
    """
    gt_n = normalize_val(gt_val)
    for d in detected_list:
        d_n = normalize_val(str(d))
        if gt_n == d_n:
            return True
        if len(gt_n) >= 10 and len(d_n) >= 10 and gt_n[:10] == d_n[:10]:
            return True
    return False