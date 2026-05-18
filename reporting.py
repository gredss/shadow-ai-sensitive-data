# -*- coding: utf-8 -*-
"""
Reporting functions for DLP Benchmark
Generate and display comprehensive evaluation reports
"""

import numpy as np
import pandas as pd

from evaluation import (
    compute_detection_metrics,
    bootstrap_recall_ci,
    paired_permutation_test,
    compute_robustness_delta,
    compute_latency_summary
)
from utils import print_banner


def generate_full_report(df_regex, df_bert, df_llm) -> pd.DataFrame:
    """Master table: P/R/F1 + 95% bootstrap CI on Recall, per paradigm × PII × style."""
    approaches = [
        ("Pattern-Matching (Regex)", df_regex, "CPU"),
        ("Discriminative (IndoBERT-NER)", df_bert, "GPU (A100)"),
        ("Reasoning (Llama-3-8B)", df_llm, "GPU (A100)"),
    ]
    rows = []
    for name, df, hw in approaches:
        for pii in ["nik", "phone", "cc", "bank", "email"]:
            for style in ["formal", "code_mixed", "slang"]:
                df_s = df[df["style"] == style]
                m = compute_detection_metrics(df_s, pii)
                ci = bootstrap_recall_ci(df_s, pii)
                rows.append({
                    "paradigm": name,
                    "hardware": hw,
                    "pii_type": pii.upper(),
                    "style": style,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "recall_ci": f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]",
                    "support": m["support"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                })
    return pd.DataFrame(rows)


def generate_significance_report(df_regex, df_bert, df_llm) -> pd.DataFrame:
    """Paired permutation tests for every paradigm pair × PII type."""
    pairs = [
        ("Pattern-Matching (Regex)", "Discriminative (IndoBERT-NER)", df_regex, df_bert),
        ("Pattern-Matching (Regex)", "Reasoning (Llama-3-8B)", df_regex, df_llm),
        ("Discriminative (IndoBERT-NER)", "Reasoning (Llama-3-8B)", df_bert, df_llm),
    ]
    rows = []
    for name_a, name_b, df_a, df_b in pairs:
        for pii in ["nik", "phone", "cc", "bank", "email"]:
            res = paired_permutation_test(df_a, df_b, pii)
            rows.append({
                "paradigm_A": name_a,
                "paradigm_B": name_b,
                "pii_type": pii.upper(),
                "delta_recall_A_minus_B": res["delta_recall"],
                "p_value": res["p_value"],
                "significant_p005": res["significant"],
            })
    return pd.DataFrame(rows)


def generate_robustness_report(df_regex, df_bert, df_llm) -> pd.DataFrame:
    """Robustness delta with CI per scenario for all paradigms."""
    approaches = [
        ("Pattern-Matching (Regex)", df_regex, "CPU"),
        ("Discriminative (IndoBERT-NER)", df_bert, "GPU (A100)"),
        ("Reasoning (Llama-3-8B)", df_llm, "GPU (A100)"),
    ]
    rows = []
    for name, df, hw in approaches:
        for pii in ["nik", "phone", "cc", "bank", "email"]:
            d = compute_robustness_delta(df, pii)
            rows.append({"paradigm": name, "hardware": hw, **d})
    return pd.DataFrame(rows)


def generate_latency_report(df_regex, df_bert, df_llm) -> pd.DataFrame:
    """Latency table with explicit hardware column."""
    rows = [
        compute_latency_summary(df_regex, "Pattern-Matching (Regex)", "CPU"),
        compute_latency_summary(df_bert, "Discriminative (IndoBERT-NER)", "GPU (A100)"),
        compute_latency_summary(df_llm, "Reasoning (Llama-3-8B)", "GPU (A100)"),
    ]
    df_lat = pd.DataFrame(rows)
    llm_mean = df_lat[df_lat["paradigm"] == "Reasoning (Llama-3-8B)"]["mean_ms"].values[0]
    df_lat["gpu_speedup_vs_llm"] = (
        df_lat.apply(
            lambda r: round(llm_mean / r["mean_ms"], 1) if r["hardware"] == "GPU (A100)" else float("nan"),
            axis=1
        )
    )
    return df_lat


def display_results(df_report, df_significance, df_robustness, df_latency):
    """Display comprehensive benchmark results."""
    print_banner("FULL METRICS REPORT — Precision, Recall, F1 with 95% Bootstrap CI")
    print(df_report[[
        "paradigm", "hardware", "pii_type", "style",
        "precision", "recall", "f1", "recall_ci", "tp", "fp", "fn", "support"
    ]].to_string(index=False))

    print_banner("SIGNIFICANCE TESTS — Paired Permutation (H0: Recall_A = Recall_B)")
    print(df_significance.to_string(index=False))
    sig = df_significance[df_significance["significant_p005"]]
    print(f"\n  Significant differences (p<0.05): {len(sig)} / {len(df_significance)} comparisons")

    print_banner("ROBUSTNESS DELTA — Recall drop with CI per scenario")
    print(df_robustness[[
        "paradigm", "pii_type",
        "recall_formal", "ci_formal",
        "recall_code_mixed", "ci_code_mixed",
        "recall_slang", "ci_slang",
        "delta_formal→slang"
    ]].to_string(index=False))

    print_banner("LATENCY — ⚠ Regex=CPU; BERT/LLM=GPU(A100). Cross-HW comparison informational only.")
    print(df_latency[[
        "paradigm", "hardware", "mean_ms", "median_ms", "p95_ms",
        "throughput_per_sec", "gpu_speedup_vs_llm"
    ]].to_string(index=False))
    print("  gpu_speedup_vs_llm: GPU-only metric (NaN = CPU). Regex latency shown for deployment reference only.")

    print_banner("KEY FINDINGS")
    
    for pii_type in ["NIK", "PHONE", "CC", "BANK", "EMAIL"]:
        pii_data = df_report[df_report["pii_type"] == pii_type]
        if not pii_data.empty:
            best_recall = pii_data.groupby("paradigm")["recall"].mean().sort_values(ascending=False)
            best_precision = pii_data.groupby("paradigm")["precision"].mean().sort_values(ascending=False)
            best_f1 = pii_data.groupby("paradigm")["f1"].mean().sort_values(ascending=False)
            
            print(f"\n  {pii_type} Detection Performance (mean across styles):")
            print(f"    Best Recall:    {best_recall.index[0]} ({best_recall.iloc[0]:.3f})")
            print(f"    Best Precision: {best_precision.index[0]} ({best_precision.iloc[0]:.3f})")
            print(f"    Best F1-score:  {best_f1.index[0]} ({best_f1.iloc[0]:.3f})")
    
    worst_delta = df_robustness.nlargest(5, "delta_formal→slang")[["paradigm", "pii_type", "delta_formal→slang"]]
    print(f"\n  Highest robustness delta (most fragile):\n{worst_delta.to_string(index=False)}")