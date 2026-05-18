# -*- coding: utf-8 -*-
"""
Main orchestration script for DLP Benchmark
Coordinates the complete benchmark workflow
"""

import os
import time
import random
import logging
import warnings
import numpy as np
import torch

from config import SEED, NUM_IDENTITIES, EVAL_SAMPLE_SIZE, DEVICE
from data_generation import build_ground_truth_dataframe, build_prompt_dataset, get_eval_sample
from pattern_matching import run_regex_benchmark
from bert_detection import load_indobert_ner, run_bert_benchmark
from llm_detection import load_llama3_4bit, run_llm_benchmark
from reporting import generate_full_report, generate_significance_report, generate_robustness_report, generate_latency_report, display_results
from utils import print_banner

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logger.info(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def main(run_bert: bool = True, run_llm: bool = True, save_outputs: bool = True):
    """Main benchmarking pipeline."""
    start = time.time()
    print_banner("EVALUATION OF DLP PARADIGMS — Indonesian Shadow AI Traffic")
    print(f"  Paradigms : Pattern-Matching (CPU) | Discriminative (GPU) | Reasoning (GPU)")
    print(f"  Scenarios : Formal | Code-Mixed | Slang/Noisy")
    print(f"  Eval n    : {EVAL_SAMPLE_SIZE}/style (identical sample, all paradigms)")
    print(f"  Bootstrap : 1,000 resamples | Permutation: 10,000 shuffles")

    print_banner("STAGE 1 — Synthetic Data Factory")
    df_truth = build_ground_truth_dataframe(NUM_IDENTITIES)

    print_banner("STAGE 2 — Contextual Wrapping")
    df_prompts = build_prompt_dataset(df_truth)

    print_banner("STAGE 2B — Drawing Shared Stratified Eval Sample")
    df_eval = get_eval_sample(df_prompts, n=EVAL_SAMPLE_SIZE, seed=SEED)
    print(f"  Distribution:\n{df_eval['style'].value_counts()}")

    print_banner("STAGE 3A — Paradigm A: Pattern-Matching (Regex, CPU)")
    df_regex = run_regex_benchmark(df_eval)

    df_bert = None
    if run_bert:
        print_banner("STAGE 3B — Paradigm B: Discriminative (IndoBERT-NER, GPU)")
        ner_pipe = load_indobert_ner()
        df_bert = run_bert_benchmark(df_eval, ner_pipe)
        del ner_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        df_bert = df_regex.copy()

    df_llm = None
    if run_llm:
        print_banner("STAGE 3C — Paradigm C: Reasoning (Llama-3 8B, GPU)")
        tok, mdl = load_llama3_4bit()
        df_llm = run_llm_benchmark(df_eval, tok, mdl)
        del tok, mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        df_llm = df_regex.copy()

    print_banner("STAGE 4 — Metrics, Statistical Rigor & Robustness")
    logger.info("Computing Precision, Recall, F1-score, TP, FP, FN for all paradigms...")
    df_report = generate_full_report(df_regex, df_bert, df_llm)
    df_significance = generate_significance_report(df_regex, df_bert, df_llm)
    df_robustness = generate_robustness_report(df_regex, df_bert, df_llm)
    df_latency = generate_latency_report(df_regex, df_bert, df_llm)
    logger.info("Metrics computation complete with enhanced precision and FP tracking.")

    display_results(df_report, df_significance, df_robustness, df_latency)

    if save_outputs:
        print_banner("Saving Outputs")
        df_truth.to_csv("ground_truth.csv", index=False)
        df_prompts.to_csv("prompt_dataset.csv", index=False)
        df_eval.to_csv("eval_sample.csv", index=False)
        df_regex.to_csv("results_regex.csv", index=False)
        df_bert.to_csv("results_bert.csv", index=False)
        df_llm.to_csv("results_llm.csv", index=False)
        df_report.to_csv("benchmark_report.csv", index=False)
        df_significance.to_csv("significance_report.csv", index=False)
        df_robustness.to_csv("robustness_report.csv", index=False)
        df_latency.to_csv("latency_report.csv", index=False)
        logger.info("All outputs saved with enhanced metrics (Precision, Recall, F1, TP, FP, FN).")

    elapsed = time.time() - start
    print(f"\nBenchmark complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return dict(
        df_truth=df_truth, df_prompts=df_prompts, df_eval=df_eval,
        df_regex=df_regex, df_bert=df_bert, df_llm=df_llm,
        df_report=df_report, df_significance=df_significance,
        df_robustness=df_robustness, df_latency=df_latency
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DLP Benchmark for Indonesian PII Detection")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT evaluation")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM evaluation")
    parser.add_argument("--no-save", action="store_true", help="Don't save output files")
    args = parser.parse_args()
    
    results = main(
        run_bert=not args.skip_bert,
        run_llm=not args.skip_llm,
        save_outputs=not args.no_save
    )