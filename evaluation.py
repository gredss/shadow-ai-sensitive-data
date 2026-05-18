# -*- coding: utf-8 -*-
"""
Evaluation metrics for DLP Benchmark
Statistical analysis including bootstrap CI and permutation tests
"""

import numpy as np
import pandas as pd
from typing import Dict

from config import SEED, N_BOOTSTRAP, CI_ALPHA
from utils import is_hit


def compute_detection_metrics(df_results: pd.DataFrame, pii_type: str) -> dict:
    """Binary TP/FN/FP per sample → Precision, Recall, F1 (point estimates)."""
    gt_col = f"ground_truth_{pii_type}"
    det_col = f"detected_{pii_type}s"
    has_col = f"has_{pii_type}"
    df_ev = df_results[df_results[has_col] == True].copy()
    if df_ev.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "support": 0, "tp": 0, "fp": 0, "fn": 0}
    
    tp = fn = fp = 0
    for _, row in df_ev.iterrows():
        gt_val = str(row[gt_col]).strip() if pd.notna(row[gt_col]) else ""
        det = row[det_col] if isinstance(row[det_col], list) else []
        
        if is_hit(gt_val, det):
            tp += 1
        else:
            fn += 1
        
        if det:
            matched = is_hit(gt_val, det)
            if not matched:
                fp += len(det)
            elif len(det) > 1:
                fp += len(det) - 1
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": len(df_ev),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def bootstrap_recall_ci(
    df_results: pd.DataFrame,
    pii_type: str,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = SEED,
) -> dict:
    """Percentile bootstrap 95% CI for Recall on a given PII type."""
    gt_col = f"ground_truth_{pii_type}"
    det_col = f"detected_{pii_type}s"
    has_col = f"has_{pii_type}"
    df_ev = df_results[df_results[has_col] == True].reset_index(drop=True)
    if df_ev.empty:
        return {"recall_point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    hits = np.array([
        int(is_hit(
            str(row[gt_col]).strip() if pd.notna(row[gt_col]) else "",
            row[det_col] if isinstance(row[det_col], list) else []
        ))
        for _, row in df_ev.iterrows()
    ])

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(hits, size=len(hits), replace=True)
        boots.append(sample.mean())

    lo = np.percentile(boots, (1 - alpha) / 2 * 100)
    hi = np.percentile(boots, (1 + alpha) / 2 * 100)
    return {
        "recall_point": round(hits.mean(), 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
    }


def paired_permutation_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    pii_type: str,
    n_perm: int = 10_000,
    seed: int = SEED,
) -> dict:
    """Paired permutation test: H0 = Recall(A) == Recall(B) on identical samples."""
    gt_col = f"ground_truth_{pii_type}"
    det_col = f"detected_{pii_type}s"
    has_col = f"has_{pii_type}"

    key_cols = ["identity_id", "style", gt_col, has_col]
    merged = df_a[[*key_cols, det_col]].rename(columns={det_col: "det_a"}).merge(
        df_b[[*key_cols, det_col]].rename(columns={det_col: "det_b"}),
        on=["identity_id", "style", gt_col, has_col], how="inner"
    )
    merged = merged[merged[has_col] == True].reset_index(drop=True)
    if merged.empty:
        return {"delta_recall": 0.0, "p_value": 1.0, "significant": False}

    hits_a = np.array([
        int(is_hit(str(r[gt_col]).strip() if pd.notna(r[gt_col]) else "", r["det_a"] if isinstance(r["det_a"], list) else []))
        for _, r in merged.iterrows()
    ])
    hits_b = np.array([
        int(is_hit(str(r[gt_col]).strip() if pd.notna(r[gt_col]) else "", r["det_b"] if isinstance(r["det_b"], list) else []))
        for _, r in merged.iterrows()
    ])

    obs_delta = hits_a.mean() - hits_b.mean()
    rng = np.random.default_rng(seed)
    perm_deltas = []
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=len(hits_a)).astype(bool)
        perm_a = np.where(swap, hits_b, hits_a)
        perm_b = np.where(swap, hits_a, hits_b)
        perm_deltas.append(perm_a.mean() - perm_b.mean())

    perm_arr = np.array(perm_deltas)
    p_value = np.mean(np.abs(perm_arr) >= np.abs(obs_delta))
    return {
        "delta_recall": round(obs_delta, 4),
        "p_value": round(p_value, 4),
        "significant": bool(p_value < 0.05),
    }


def compute_robustness_delta(df_results: pd.DataFrame, pii_type: str) -> dict:
    """Recall drop across Formal → Code-Mixed → Slang, with CIs per scenario."""
    rows = {}
    for style in ["formal", "code_mixed", "slang"]:
        ci = bootstrap_recall_ci(df_results[df_results["style"] == style], pii_type)
        rows[style] = ci
    r_f = rows["formal"]["recall_point"]
    r_cm = rows["code_mixed"]["recall_point"]
    r_sl = rows["slang"]["recall_point"]
    return {
        "pii_type": pii_type.upper(),
        "recall_formal": r_f,
        "ci_formal": f"[{rows['formal']['ci_lower']:.3f}, {rows['formal']['ci_upper']:.3f}]",
        "recall_code_mixed": r_cm,
        "ci_code_mixed": f"[{rows['code_mixed']['ci_lower']:.3f}, {rows['code_mixed']['ci_upper']:.3f}]",
        "recall_slang": r_sl,
        "ci_slang": f"[{rows['slang']['ci_lower']:.3f}, {rows['slang']['ci_upper']:.3f}]",
        "delta_formal→cm": round(r_f - r_cm, 4),
        "delta_cm→slang": round(r_cm - r_sl, 4),
        "delta_formal→slang": round(r_f - r_sl, 4),
    }


def compute_latency_summary(df_results: pd.DataFrame, paradigm: str, hardware: str) -> dict:
    """Compute latency statistics."""
    lat = df_results["latency_ms"].values
    return {
        "paradigm": paradigm,
        "hardware": hardware,
        "mean_ms": round(np.mean(lat), 3),
        "median_ms": round(np.median(lat), 3),
        "p95_ms": round(np.percentile(lat, 95), 3),
        "p99_ms": round(np.percentile(lat, 99), 3),
        "throughput_per_sec": round(1000 / np.mean(lat), 1),
    }