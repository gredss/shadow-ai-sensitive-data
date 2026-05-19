# -*- coding: utf-8 -*-
"""
Configuration module for DLP Benchmark.

This module centralizes all configuration parameters including:
- Model names and paths
- Hardware specifications
- Prompt templates
- Regex patterns
- Evaluation parameters
- Environment variables (HuggingFace token)

All HuggingFace API calls use the token loaded from the HF_TOKEN environment variable.
"""

import os
import torch
from typing import Optional, Dict, List

def get_hf_token() -> Optional[str]:
    """
    Load HuggingFace API token from environment variable or Google Colab userdata.
    
    Supports multiple token sources:
    1. Environment variable HF_TOKEN (local/server deployment)
    2. Google Colab userdata (for Colab notebooks)
    3. Manual token passing (fallback)
    
    Returns:
        Optional[str]: HuggingFace token if available, None otherwise
        
    Raises:
        ValueError: If HF_TOKEN is not available from any source
        
    Example (Google Colab):
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        from huggingface_hub import login
        login(token=hf_token)
    """
    # Try environment variable first
    token = os.getenv("HF_TOKEN")
    
    # Try Google Colab userdata if in Colab environment
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except (ImportError, Exception):
            pass
    
    if token is None:
        raise ValueError(
            "HF_TOKEN not found. Please set it using one of these methods:\n"
            "1. Environment variable: export HF_TOKEN='your_token_here'\n"
            "2. Google Colab: Store HF_TOKEN in Colab secrets\n"
            "3. Manual: Pass token directly to model loading functions"
        )
    return token

# Load HuggingFace token (required for model downloads)
# Will work in both local environments and Google Colab
HF_TOKEN: Optional[str] = get_hf_token()


SEED: int = 42
"""Random seed for reproducibility across all random operations."""
NUM_IDENTITIES: int = 5_000
"""Number of synthetic identities to generate for the dataset."""
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
"""Device for model inference (cuda if available, otherwise cpu)."""

LLAMA_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"
"""HuggingFace model ID for Llama-3 8B Instruct (requires HF_TOKEN)."""

INDOBERT_NER_MODEL: str = "./bert_pii_model"
"""Path to fine-tuned Indonesian BERT model for PII detection."""

EVAL_SAMPLE_SIZE: int = 500
"""Number of samples per writing style for evaluation (stratified sampling)."""

N_BOOTSTRAP: int = 1_000
"""Number of bootstrap resamples for confidence interval estimation."""

CI_ALPHA: float = 0.95
"""Confidence level for bootstrap confidence intervals (95%)."""

HARDWARE: Dict[str, str] = {
    "Pattern-Matching (Regex)": "CPU",
    "Discriminative (IndoBERT-NER)": "GPU (A100)",
    "Reasoning (Llama-3-8B)": "GPU (A100)",
}
"""Hardware requirements for each DLP paradigm (informational only)."""

AREA_CODES: List[str] = [
    "317101", "317201", "317301",  # Surabaya
    "317401", "327101", "327201",  # East Java
    "327301", "317501", "317601",  # East Java suburbs
    "310101", "310201", "310301",  # Jakarta
    "310401", "310501", "327401",  # DKI Jakarta
    "320101", "320201", "320301",  # West Java
    "320401", "330101", "330201",  # Central Java
    "330301", "340101", "340201",  # Yogyakarta
    "350101", "350201", "360101",  # East Java / Banten
    "360201", "370101", "510101",  # NTB / Bali
    "510201", "510301", "610101",  # Bali / West Kalimantan
    "710101", "710201", "730101",  # Sulawesi
]

"""Indonesian area codes for NIK (National ID) generation."""

PHONE_PREFIXES: List[str] = [
    "0811", "0812", "0813", "0821", "0822", "0823", "0851", "0852", "0853",
    "0814", "0815", "0816", "0855", "0856", "0857", "0858",
    "0817", "0818", "0819", "0859", "0877", "0878",
    "0831", "0832", "0833", "0838",
    "0895", "0896", "0897", "0898", "0899",
    "0881", "0882", "0883", "0884", "0885",
]

"""Indonesian mobile phone number prefixes (Telkomsel, XL, Indosat, etc.)."""

EMAIL_DOMAINS: List[str] = ["gmail.com", "yahoo.co.id", "hotmail.com", "outlook.com", "ymail.com"]
"""Common email domains for synthetic email generation."""

CC_PREFIXES: List[str] = ["4", "51", "52", "53", "54", "55", "6011"]
"""Credit card number prefixes (Visa, Mastercard, Discover)."""

FORMAL_TEMPLATES: List[str] = [
    "Mohon bantuannya untuk memproses data nasabah berikut ini. Nama: {name}, NIK: {nik}, Nomor Telepon: {phone}, Nomor Rekening: {bank_account}, Kartu Kredit: {credit_card}, Email: {email}. Harap segera ditindaklanjuti.",
    "Dengan hormat, kami mengajukan permohonan verifikasi identitas atas nama {name} dengan NIK {nik}. Kartu kredit yang terdaftar: {credit_card}. Nomor HP: {phone}. Rekening bank: {bank_account}. Email: {email}.",
    "Kepada Tim Data Center, mohon lakukan pembaruan data pelanggan. Detail: {name}, NIK {nik}, telepon {phone}, rekening {bank_account}, kartu kredit {credit_card}, email {email}.",
    "Bapak/Ibu Tim Compliance, terlampir data untuk proses KYC: NIK {nik} atas nama {name}, kontak {phone}, nomor CC {credit_card}, rekening {bank_account}, dan email {email}.",
    "Sesuai SOP perusahaan, kami menyampaikan data berikut untuk keperluan audit: Nama {name}, NIK: {nik}, No. HP: {phone}, No. Rek: {bank_account}, CC: {credit_card}, Email: {email}.",
]

"""Formal Indonesian prompt templates for professional contexts."""

CODE_MIXED_TEMPLATES: List[str] = [
    "Guys, please help me check this user info ASAP ya! NIKnya: {nik}, dia punya HP {phone}, CC {credit_card}, rekening {bank_account}, email {email}. Nama lengkapnya {name}. Thanks banget yaa!",
    "Halo team, tolong dong update database buat user ini. CC number: {credit_card}, NIK {nik}, phone {phone}, rekening {bank_account}, email {email}. Urgent nih karena deadlinenya today!",
    "FYI everyone, ada data mismatch di system. User {name} dengan NIK {nik} nomernya {phone}, CC {credit_card}, tolong didouble check ya rekening {bank_account} sama email {email}.",
    "Hey, bisa tolong recheck gak? Katanya NIK {nik} sama CC {credit_card} invalid tapi menurut gw udah bener. Contact {name} di {phone}, rekening {bank_account}, email {email}.",
    "Tolong bantu verify data nih ASAP. {name} | NIK: {nik} | Phone: {phone} | CC: {credit_card} | Rek: {bank_account} | Email: {email}. Waiting for your response!",
]

"""Code-mixed Indonesian-English prompt templates for casual workplace communication."""

SLANG_TEMPLATES: List[str] = [
    "Weh gengs tlg bgt rapiin data KTP ni {nik} dr si {name} biar gampang diinput, nopeny {phone}, CC {credit_card}, reknya {bank_account}, emailny {email} klo mau follow up abis maksi yaw.",
    "Bre tolongin dong wkwkww cek NIK {nik} valid g siehh? Punya {name}, nomernya {phone}, CC {credit_card}, reknya {bank_account}, email {email}. Tenks yh,, biar gx kna audit.",
    "Ges, ada prob nih sm data user: NIKnya.. either {nik} ato {decoy_id}, CC {credit_card}, no hp {phone}, rek {bank_account}, email {email}. Tolng gaskeun asap yaah sblm si b0s marah eawwokwoekwk.",
    "Hlooo.. ni hrsny udh aman sijhh, tlg proses data inieh dungs: nik {nik} nm {name} hp {phone} CC {credit_card} rek {bank_account} email {email}, gece yee. Tq.",
    "Mksh sblmx mas mba, minta tolong verif NIK {nik} smaa CC {credit_card} dong, urgent bgtt sumpaa, kontakx {phone}, reknya {bank_account}, emailny {email}. Plus jan ktuker ama resi {decoy_id}, maaciw.",
]

"""Slang Indonesian prompt templates with typos and informal language."""

ALL_TEMPLATES: Dict[str, List[str]] = {
    "formal": FORMAL_TEMPLATES,
    "code_mixed": CODE_MIXED_TEMPLATES,
    "slang": SLANG_TEMPLATES
}

"""All prompt templates organized by writing style."""


NOISE_LEVELS: Dict[str, float] = {
    "formal": 0.0,
    "code_mixed": 0.1,
    "slang": 0.3
}

"""Noise injection probability by writing style (0.0 = no noise, 1.0 = maximum noise)."""


DLP_SYSTEM_PROMPT: str = """You are a Data Loss Prevention (DLP) filter for Indonesian PII.

Analyze the text and return ONLY a valid JSON object:
{
  "found_pii": true,
  "findings": [
    {"type": "NIK"|"PHONE"|"CREDIT_CARD"|"BANK_ACCOUNT"|"EMAIL"|"NAME",
     "value": "<exact value from text>",
     "confidence": "HIGH"|"MEDIUM"|"LOW"}
  ]
}

Indonesian PII rules:
- NIK: 16-digit national ID (Nomor Induk Kependudukan)
- PHONE: numbers starting with 08xx or +62
- CREDIT_CARD: 13-19 digit payment card numbers
- BANK_ACCOUNT: 8-15 digit bank account numbers
- EMAIL: email addresses

If no PII: {"found_pii": false, "findings": []}
Return ONLY the JSON."""
"""System prompt for LLM-based PII detection with structured JSON output."""