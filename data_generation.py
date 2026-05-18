# -*- coding: utf-8 -*-
"""
Synthetic data generation for DLP Benchmark
Generates Indonesian PII data with various obfuscation styles
"""

import random
from datetime import datetime
from typing import Optional
import pandas as pd
from faker import Faker
from tqdm import tqdm

from config import (
    SEED, AREA_CODES, PHONE_PREFIXES, EMAIL_DOMAINS, CC_PREFIXES,
    ALL_TEMPLATES, NOISE_LEVELS
)

random.seed(SEED)
fake_id = Faker("id_ID")
Faker.seed(SEED)


def generate_nik(gender: str = "M", dob: Optional[datetime] = None) -> str:
    """Generate Indonesian National ID (NIK) with proper encoding."""
    area_code = random.choice(AREA_CODES)
    if dob is None:
        dob = fake_id.date_of_birth(minimum_age=17, maximum_age=70)
    day = dob.day + (40 if gender == "F" else 0)
    dob_str = f"{day:02d}{dob.month:02d}{str(dob.year)[-2:]}"
    sequence = str(random.randint(1, 9999)).zfill(4)
    return f"{area_code}{dob_str}{sequence}"


def generate_indonesian_phone() -> str:
    """Generate Indonesian phone number."""
    prefix = random.choice(PHONE_PREFIXES)
    suffix_len = random.choice([7, 8])
    suffix = "".join([str(random.randint(0, 9)) for _ in range(suffix_len)])
    return f"+62{prefix[1:]}{suffix}" if random.random() < 0.3 else f"{prefix}{suffix}"


def generate_bank_account() -> str:
    """Generate bank account number."""
    length = random.randint(8, 15)
    return str(random.randint(1, 9)) + "".join([str(random.randint(0, 9)) for _ in range(length - 1)])


def generate_credit_card() -> str:
    """Generate Luhn-validated credit card number."""
    prefix = random.choice(CC_PREFIXES)
    partial = prefix + "".join([str(random.randint(0, 9)) for _ in range(16 - len(prefix) - 1)])
    digits = [int(d) for d in partial]
    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    checksum = (10 - (sum(digits) % 10)) % 10
    return partial + str(checksum)


def generate_email_indonesian(name: str) -> str:
    """Generate Indonesian email address."""
    clean_name = name.lower().replace(" ", random.choice([".", "_", ""]))
    suffix = random.choice(["", str(random.randint(1, 999))])
    return f"{clean_name}{suffix}@{random.choice(EMAIL_DOMAINS)}"


def generate_decoy_id():
    """16-digit string with invalid NIK prefix - used as a distractor."""
    return f"99{random.randint(10000000000000, 99999999999999)}"


def inject_noise(text, noise_level=0.2):
    """OCR typos (0→O, 1→l), dash-spacing, or partial masking (e.g. 1234XXXX)."""
    if random.random() > noise_level:
        return text
    noise_type = random.choice(["typo", "space", "mask"])
    if noise_type == "typo":
        text = text.replace("0", "O").replace("1", "l")
    elif noise_type == "space":
        parts = [text[i:i+4] for i in range(0, len(text), 4)]
        text = "-".join(parts)
    elif noise_type == "mask":
        text = text[:12] + "XXXX"
    return text


def wrap_pii_in_prompt(row: pd.Series, style: str) -> dict:
    """Wrap PII data in contextual prompt with specified style."""
    template = random.choice(ALL_TEMPLATES[style])
    noise_level = NOISE_LEVELS[style]

    nik_val = inject_noise(str(row["nik"]), noise_level)
    phone_val = inject_noise(str(row["phone"]), noise_level)
    cc_val = inject_noise(str(row["credit_card"]), noise_level)
    bank_val = inject_noise(str(row["bank_account"]), noise_level)

    prompt = template.format(
        name=row["name"], nik=nik_val, phone=phone_val,
        credit_card=cc_val, bank_account=bank_val,
        email=row["email"], decoy_id=row["decoy_id"],
    )
    
    fields_used = {
        "nik": "{nik}" in template,
        "phone": "{phone}" in template,
        "credit_card": "{credit_card}" in template,
        "bank_account": "{bank_account}" in template,
        "email": "{email}" in template,
    }
    
    return {
        "identity_id": row["id"],
        "style": style,
        "prompt": prompt,
        "ground_truth_nik": row["nik"] if fields_used["nik"] else None,
        "ground_truth_phone": row["phone"] if fields_used["phone"] else None,
        "ground_truth_cc": row["credit_card"] if fields_used["credit_card"] else None,
        "ground_truth_bank": row["bank_account"] if fields_used["bank_account"] else None,
        "ground_truth_email": row["email"] if fields_used["email"] else None,
        "has_nik": fields_used["nik"],
        "has_phone": fields_used["phone"],
        "has_cc": fields_used["credit_card"],
        "has_bank": fields_used["bank_account"],
        "has_email": fields_used["email"],
    }


def build_ground_truth_dataframe(n: int) -> pd.DataFrame:
    """Generate n synthetic identities with PII."""
    records = []
    for i in tqdm(range(n), desc="Generating identities"):
        gender = random.choice(["M", "F"])
        dob = fake_id.date_of_birth(minimum_age=17, maximum_age=70)
        name = fake_id.name_male() if gender == "M" else fake_id.name_female()
        records.append({
            "id": i, "name": name, "gender": gender, "dob": dob,
            "nik": generate_nik(gender, dob),
            "phone": generate_indonesian_phone(),
            "bank_account": generate_bank_account(),
            "credit_card": generate_credit_card(),
            "email": generate_email_indonesian(name),
            "address": fake_id.address().replace("\n", ", "),
            "decoy_id": generate_decoy_id(),
        })
    return pd.DataFrame(records)


def build_prompt_dataset(df_truth: pd.DataFrame) -> pd.DataFrame:
    """Build full prompt set with all styles."""
    all_wrapped = []
    for style in ["formal", "code_mixed", "slang"]:
        for _, row in tqdm(df_truth.iterrows(), total=len(df_truth), desc=style):
            all_wrapped.append(wrap_pii_in_prompt(row, style))
    return pd.DataFrame(all_wrapped)


def get_eval_sample(df_prompts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a stratified sample of n rows per style.
    
    All three paradigms (Regex, BERT, LLM) are evaluated on this identical
    sample so that Precision/Recall/F1 values are directly comparable.
    """
    frames = []
    for style in ["formal", "code_mixed", "slang"]:
        subset = df_prompts[df_prompts["style"] == style]
        frames.append(subset.sample(n=min(n, len(subset)), random_state=seed))
    return pd.concat(frames, ignore_index=True)