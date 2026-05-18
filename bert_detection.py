# -*- coding: utf-8 -*-
"""
BERT-based NER detection for DLP Benchmark.

This module implements the Discriminative paradigm using a fine-tuned Indonesian BERT
model for Named Entity Recognition (NER) of PII. The model uses BIO (Beginning-Inside-Outside)
tagging to identify entity boundaries.

Key Features:
- Fine-tuned on Indonesian PII (NIK, phone, credit card, bank account, email)
- Token-level classification with proper entity boundary detection
- GPU-accelerated inference
- Handles multi-token entities

Functions:
    load_indobert_ner: Load fine-tuned BERT model
    bert_detect: Extract PII entities from text
    run_bert_benchmark: Run complete BERT evaluation
"""

import time
import logging
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

from config import INDOBERT_NER_MODEL, DEVICE, HF_TOKEN

logger = logging.getLogger(__name__)


def load_indobert_ner():
    """
    Load fine-tuned Indonesian BERT model for PII detection.
    
    Loads a token classification model fine-tuned specifically for Indonesian PII
    detection. The model uses BIO tagging to identify entity boundaries.
    
    Returns:
        pipeline: HuggingFace NER pipeline configured for PII detection
        
    Note:
        Requires HF_TOKEN environment variable for model downloads from HuggingFace Hub.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    logger.info(f"Loading Fine-tuned IndoBERT PII Model: {INDOBERT_NER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_NER_MODEL, token=HF_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(INDOBERT_NER_MODEL, token=HF_TOKEN)
    model = model.to(DEVICE)
    ner_pipe = pipeline(
        "ner", model=model, tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if DEVICE == "cuda" else -1,
    )
    logger.info("Fine-tuned IndoBERT PII model loaded successfully.")
    return ner_pipe


def bert_detect(text: str, ner_pipe) -> Dict[str, Any]:
    """
    Extract PII entities from text using fine-tuned BERT NER.
    
    Paradigm B (GPU): Discriminative approach using token classification.
    The model identifies entity boundaries using BIO tagging and aggregates
    multi-token entities.
    
    Args:
        text: Input text to analyze (truncated to 512 tokens)
        ner_pipe: HuggingFace NER pipeline from load_indobert_ner()
        
    Returns:
        Dict containing:
            - detected_persons: List of detected person names
            - detected_niks: List of detected NIK numbers
            - detected_phones: List of detected phone numbers
            - detected_ccs: List of detected credit card numbers
            - detected_banks: List of detected bank account numbers
            - detected_emails: List of detected email addresses
            - raw_entities: Raw NER output with scores and positions
            
    Note:
        Text is truncated to 512 tokens due to BERT's maximum sequence length.
    """
    entities = ner_pipe(text[:512])
    
    persons = []
    niks = []
    phones = []
    ccs = []
    banks = []
    emails = []
    
    # Map BIO tags to entity types
    for ent in entities:
        entity_group = ent.get("entity_group", "")
        word = ent.get("word", "")
        
        if entity_group in ("PERSON", "PER", "B-PERSON", "I-PERSON"):
            persons.append(word)
        elif entity_group in ("NIK", "B-NIK", "I-NIK"):
            niks.append(word)
        elif entity_group in ("PHONE", "B-PHONE", "I-PHONE"):
            phones.append(word)
        elif entity_group in ("CC", "CREDIT_CARD", "B-CC", "I-CC"):
            ccs.append(word)
        elif entity_group in ("BANK", "BANK_ACCOUNT", "B-BANK", "I-BANK"):
            banks.append(word)
        elif entity_group in ("EMAIL", "B-EMAIL", "I-EMAIL"):
            emails.append(word)
    
    return {
        "detected_persons": list(set([p.strip() for p in persons if p.strip()])),
        "detected_niks": list(set([n.strip() for n in niks if n.strip()])),
        "detected_phones": list(set([p.strip() for p in phones if p.strip()])),
        "detected_ccs": list(set([c.strip() for c in ccs if c.strip()])),
        "detected_banks": list(set([b.strip() for b in banks if b.strip()])),
        "detected_emails": list(set([e.strip() for e in emails if e.strip()])),
        "raw_entities": entities,
    }


def run_bert_benchmark(df_eval: pd.DataFrame, ner_pipe) -> pd.DataFrame:
    """
    Run BERT NER evaluation on the complete evaluation dataset.
    
    Processes each sample in the evaluation set, extracting PII entities and
    measuring inference latency. Results include both detected entities and
    ground truth for metrics calculation.
    
    Args:
        df_eval: Evaluation DataFrame with columns:
            - prompt: Text to analyze
            - identity_id: Unique identifier
            - style: Writing style (formal/code_mixed/slang)
            - ground_truth_*: Ground truth PII values
            - has_*: Boolean flags for PII presence
        ner_pipe: HuggingFace NER pipeline from load_indobert_ner()
        
    Returns:
        DataFrame with detection results including:
            - detected_*: Lists of detected PII values
            - latency_ms: Inference time in milliseconds
            - ground_truth_*: Original ground truth values
            - has_*: Original PII presence flags
            
    Note:
        Uses tqdm progress bar for monitoring long-running evaluations.
    """
    logger.info("Paradigm B: Discriminative (IndoBERT-NER, GPU)...")
    results = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="BERT [GPU]"):
        t0 = time.perf_counter()
        det = bert_detect(row["prompt"], ner_pipe)
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

# Made with Bob
