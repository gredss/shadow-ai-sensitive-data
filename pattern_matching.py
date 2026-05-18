# -*- coding: utf-8 -*-
"""
Pattern-matching (Regex) detection for DLP Benchmark
Rule-based PII extraction using regular expressions
"""

import re
import time
import pandas as pd
from tqdm import tqdm

NIK_BROAD_PATTERN = re.compile(r"(?<!\d)\d{16}(?!\d)")

PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)
    (?:\+62|62|0)(?:8[1-9][0-9])[0-9]{6,9}
    (?!\d)
    """, re.VERBOSE,
)


def regex_detect(text: str) -> dict:
    """Paradigm A (CPU): keyword-disambiguated regex extraction."""
    text_lower = text.lower()
    potential_16 = [m.group() for m in NIK_BROAD_PATTERN.finditer(text)]
    niks, ccs = [], []
    for val in potential_16:
        if any(k in text_lower for k in ["nik", "ktp", "identitas"]):
            niks.append(val)
        elif any(k in text_lower for k in ["cc", "credit", "kredit", "kartu"]):
            ccs.append(val)
        else:
            niks.append(val)
    phones = [m.group() for m in PHONE_PATTERN.finditer(text)]
    banks = [
        m.group() for m in re.finditer(r"(?<!\d)\d{8,15}(?!\d)", text)
        if any(k in text_lower for k in ["rek", "rekening", "bank", "akun"])
    ]
    emails = list(set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)))
    
    return {
        "detected_niks": list(set(niks)),
        "detected_phones": list(set(phones)),
        "detected_ccs": list(set(ccs)),
        "detected_banks": list(set(banks)),
        "detected_emails": emails,
    }


def run_regex_benchmark(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Run Regex on the shared eval sample."""
    results = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Regex [CPU]"):
        t0 = time.perf_counter()
        det = regex_detect(row["prompt"])
        latency = (time.perf_counter() - t0) * 1000
        results.append({
            **det,
            "identity_id": row["identity_id"],
            "style": row["style"],
            "latency_ms": latency,
            "ground_truth_nik": row["ground_truth_nik"],
            "ground_truth_phone": row["ground_truth_phone"],
            "ground_truth_cc": row["ground_truth_cc"],
            "ground_truth_bank": row["ground_truth_bank"],
            "ground_truth_email": row["ground_truth_email"],
            "has_nik": row["has_nik"],
            "has_phone": row["has_phone"],
            "has_cc": row["has_cc"],
            "has_bank": row["has_bank"],
            "has_email": row["has_email"],
        })
    return pd.DataFrame(results)