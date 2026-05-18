# -*- coding: utf-8 -*-
"""
LLM-based detection for DLP Benchmark.

This module implements the Reasoning paradigm using Llama-3-8B for PII detection.
The LLM uses structured prompts to extract PII with contextual understanding and
reasoning capabilities.

Key Features:
- Llama-3-8B with 4-bit NF4 quantization for memory efficiency
- Structured JSON output for reliable parsing
- Contextual reasoning for noisy/masked PII
- Handles multiple writing styles and noise levels

Functions:
    load_llama3_4bit: Load quantized Llama-3 model
    llm_detect: Extract PII using LLM reasoning
    run_llm_benchmark: Run complete LLM evaluation
"""

import re
import json
import time
import logging
from typing import Dict, Any, Tuple
import torch
import pandas as pd
from tqdm import tqdm

from config import LLAMA_MODEL_ID, DLP_SYSTEM_PROMPT, DEVICE, HF_TOKEN

logger = logging.getLogger(__name__)


def load_llama3_4bit() -> Tuple[Any, Any]:
    """
    Load Llama-3-8B with 4-bit NF4 quantization.
    
    Uses bitsandbytes for 4-bit quantization to reduce memory footprint from
    ~16GB to ~4GB while maintaining performance. Enables inference on consumer GPUs.
    
    Returns:
        Tuple of (tokenizer, model):
            - tokenizer: HuggingFace tokenizer for Llama-3
            - model: Quantized Llama-3-8B model ready for inference
            
    Note:
        Requires HF_TOKEN environment variable for model downloads from HuggingFace Hub.
        Model is loaded with device_map="auto" for automatic GPU placement.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    logger.info(f"Loading {LLAMA_MODEL_ID} with 4-bit NF4 quantization")
    
    # Configure 4-bit quantization
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, token=HF_TOKEN)
    tok.pad_token = tok.eos_token
    
    # Load model with quantization
    mdl = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    mdl.eval()
    logger.info("Llama-3-8B loaded successfully with 4-bit quantization")
    return tok, mdl


def llm_detect(text: str, tokenizer: Any, model: Any) -> Dict[str, Any]:
    """
    Extract PII from text using LLM reasoning with structured output.
    
    Paradigm C (GPU): Reasoning approach using large language model.
    The LLM analyzes text contextually and returns structured JSON with
    detected PII entities and confidence levels.
    
    Args:
        text: Input text to analyze
        tokenizer: HuggingFace tokenizer from load_llama3_4bit()
        model: Quantized Llama-3 model from load_llama3_4bit()
        
    Returns:
        Dict containing:
            - found_pii: Boolean indicating if any PII was found
            - findings: List of detected PII with type, value, and confidence
            - parse_error: Boolean indicating JSON parsing failure (if applicable)
            - raw: Raw LLM response (if parsing failed)
            
    Example:
        {
            "found_pii": true,
            "findings": [
                {"type": "NIK", "value": "3171011234567890", "confidence": "HIGH"},
                {"type": "PHONE", "value": "081234567890", "confidence": "MEDIUM"}
            ]
        }
        
    Note:
        Uses greedy decoding (do_sample=False) for deterministic output.
        JSON is extracted using regex pattern matching.
    """
    # Construct chat messages with system prompt
    messages = [
        {"role": "system", "content": DLP_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze for PII:\n\n{text}"},
    ]
    
    # Apply chat template and tokenize
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)
    
    # Generate response with mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding for determinism
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (skip input tokens)
    resp = tokenizer.decode(
        out[0][enc["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    # Extract and parse JSON
    try:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if m:
            return json.loads(m.group())
        else:
            return {"found_pii": False, "findings": [], "parse_error": True}
    except json.JSONDecodeError:
        return {
            "found_pii": False,
            "findings": [],
            "parse_error": True,
            "raw": resp[:200]
        }


def run_llm_benchmark(df_eval: pd.DataFrame, tokenizer: Any, model: Any) -> pd.DataFrame:
    """
    Run LLM evaluation on the complete evaluation dataset.
    
    Processes each sample in the evaluation set, extracting PII using LLM reasoning
    and measuring inference latency. Results include both detected entities and
    ground truth for metrics calculation.
    
    Args:
        df_eval: Evaluation DataFrame with columns:
            - prompt: Text to analyze
            - identity_id: Unique identifier
            - style: Writing style (formal/code_mixed/slang)
            - ground_truth_*: Ground truth PII values
            - has_*: Boolean flags for PII presence
        tokenizer: HuggingFace tokenizer from load_llama3_4bit()
        model: Quantized Llama-3 model from load_llama3_4bit()
        
    Returns:
        DataFrame with detection results including:
            - detected_*: Lists of detected PII values
            - llm_found_pii: Boolean indicating if LLM found any PII
            - llm_num_findings: Number of PII entities found
            - latency_ms: Inference time in milliseconds
            - ground_truth_*: Original ground truth values
            - has_*: Original PII presence flags
            
    Note:
        Uses tqdm progress bar for monitoring long-running evaluations.
        LLM inference is significantly slower than BERT (~10x).
    """
    logger.info(f"Paradigm C: Reasoning (Llama-3, GPU) — {len(df_eval)} total rows...")
    results = []
    
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="LLM [GPU]"):
        t0 = time.perf_counter()
        res = llm_detect(row["prompt"], tokenizer, model)
        latency = (time.perf_counter() - t0) * 1000
        
        findings = res.get("findings", [])
        results.append({
            "identity_id": row["identity_id"],
            "style": row["style"],
            "latency_ms": latency,
            "detected_niks": [f["value"] for f in findings if f.get("type") == "NIK"],
            "detected_phones": [f["value"] for f in findings if f.get("type") == "PHONE"],
            "detected_ccs": [f["value"] for f in findings if f.get("type") == "CREDIT_CARD"],
            "detected_banks": [f["value"] for f in findings if f.get("type") == "BANK_ACCOUNT"],
            "detected_emails": [f["value"] for f in findings if f.get("type") == "EMAIL"],
            "llm_found_pii": res.get("found_pii", False),
            "llm_num_findings": len(findings),
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
