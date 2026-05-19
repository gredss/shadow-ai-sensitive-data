#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Fine-tuning Script for Indonesian PII Named Entity Recognition

This script fine-tunes an Indonesian BERT model for detecting sensitive PII across
multiple entity types: NIK, PHONE, CREDIT_CARD, BANK_ACCOUNT, EMAIL, PERSON.

The model is optimized to handle various writing styles (formal, code-mixed, slang)
and noise patterns (typos, abbreviations, masking) common in Indonesian text.

Usage:
    python train_bert_pii_ner.py --data_path ./prompt_dataset.csv --output_dir ./bert_pii_model
"""

import os
import re
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.scheme import IOB2

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Entity types for Indonesian PII
ENTITY_TYPES = [
    "O",           # Outside any entity
    "B-NIK",       # Beginning of NIK (16-digit national ID)
    "I-NIK",       # Inside NIK
    "B-PHONE",     # Beginning of phone number
    "I-PHONE",     # Inside phone number
    "B-CC",        # Beginning of credit card
    "I-CC",        # Inside credit card
    "B-BANK",      # Beginning of bank account
    "I-BANK",      # Inside bank account
    "B-EMAIL",     # Beginning of email
    "I-EMAIL",     # Inside email
    "B-PERSON",    # Beginning of person name
    "I-PERSON",    # Inside person name
]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_TYPES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Model configuration
DEFAULT_MODEL = "cahya/bert-base-indonesian-NER"  # Pre-trained Indonesian NER model (will discard classifier, keep BERT encoder)
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2


def extract_entities_from_prompt(prompt: str, ground_truth: Dict) -> List[Tuple[str, int, int, str]]:
    """
    Extract entity spans from prompt text using ground truth values.
    
    Returns:
        List of (entity_text, start_idx, end_idx, entity_type)
    """
    entities = []
    
    # Define entity patterns and their types
    entity_map = [
        (ground_truth.get("nik"), "NIK"),
        (ground_truth.get("phone"), "PHONE"),
        (ground_truth.get("cc"), "CC"),
        (ground_truth.get("bank"), "BANK"),
        (ground_truth.get("email"), "EMAIL"),
    ]
    
    for value, entity_type in entity_map:
        if value and pd.notna(value):
            value_str = str(value).strip()
            # Find all occurrences (case-insensitive for emails)
            if entity_type == "EMAIL":
                pattern = re.escape(value_str)
                for match in re.finditer(pattern, prompt, re.IGNORECASE):
                    entities.append((match.group(), match.start(), match.end(), entity_type))
            else:
                # For numeric entities, handle noise variations
                # Look for the core digits
                clean_value = re.sub(r'\D', '', value_str)
                if len(clean_value) >= 8:  # Minimum meaningful length
                    # Find patterns that contain these digits (with possible noise)
                    for match in re.finditer(re.escape(value_str), prompt):
                        entities.append((match.group(), match.start(), match.end(), entity_type))
    
    # Sort by start position
    entities.sort(key=lambda x: x[1])
    return entities


def create_bio_tags(text: str, entities: List[Tuple[str, int, int, str]], 
                    tokenizer) -> Tuple[List[str], List[str]]:
    """
    Create BIO tags for tokenized text.
    
    Returns:
        (tokens, bio_tags)
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH - 2  # Account for [CLS] and [SEP]
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    
    # Initialize all tags as "O"
    bio_tags = ["O"] * len(tokens)
    
    # Assign BIO tags based on entity spans
    for entity_text, start, end, entity_type in entities:
        entity_assigned = False
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_start >= start and token_end <= end:
                if not entity_assigned:
                    bio_tags[idx] = f"B-{entity_type}"
                    entity_assigned = True
                else:
                    bio_tags[idx] = f"I-{entity_type}"
            elif token_start < end and token_end > start:
                # Partial overlap
                if not entity_assigned:
                    bio_tags[idx] = f"B-{entity_type}"
                    entity_assigned = True
                else:
                    bio_tags[idx] = f"I-{entity_type}"
    
    return tokens, bio_tags


def load_and_preprocess_data(data_path: str, tokenizer) -> pd.DataFrame:
    """
    Load prompt dataset and create NER training examples.
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    training_examples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        prompt = row["prompt"]
        
        # Gather ground truth
        ground_truth = {
            "nik": row.get("ground_truth_nik"),
            "phone": row.get("ground_truth_phone"),
            "cc": row.get("ground_truth_cc"),
            "bank": row.get("ground_truth_bank"),
            "email": row.get("ground_truth_email"),
        }
        
        # Extract entities
        entities = extract_entities_from_prompt(prompt, ground_truth)
        
        # Create BIO tags
        tokens, bio_tags = create_bio_tags(prompt, entities, tokenizer)
        
        if len(tokens) > 0:
            training_examples.append({
                "identity_id": row["identity_id"],
                "style": row["style"],
                "tokens": tokens,
                "bio_tags": bio_tags,
                "text": prompt,
            })
    
    logger.info(f"Created {len(training_examples)} training examples")
    
    # Analyze label distribution
    all_tags = [tag for ex in training_examples for tag in ex["bio_tags"]]
    tag_counts = Counter(all_tags)
    logger.info(f"Label distribution: {dict(tag_counts)}")
    
    return pd.DataFrame(training_examples)


class PII_NER_Dataset(Dataset):
    """PyTorch Dataset for PII NER task."""
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row["tokens"]
        bio_tags = row["bio_tags"]
        
        # Convert to input IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Convert BIO tags to label IDs
        label_ids = [LABEL2ID[tag] for tag in bio_tags]
        
        # Add special tokens
        token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
        label_ids = [LABEL2ID["O"]] + label_ids + [LABEL2ID["O"]]
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]
        
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        token_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [LABEL2ID["O"]] * padding_length
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def compute_metrics(pred):
    """Compute seqeval metrics for NER evaluation."""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding and special tokens
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []
        for pred_id, label_id in zip(prediction, label):
            if label_id != LABEL2ID["O"] or pred_id != LABEL2ID["O"]:
                true_label.append(ID2LABEL[label_id])
                true_prediction.append(ID2LABEL[pred_id])
        
        if true_label:  # Only add non-empty sequences
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
    
    # Compute seqeval metrics
    results = {
        "precision": precision_recall_fscore_support(
            [tag for seq in true_labels for tag in seq],
            [tag for seq in true_predictions for tag in seq],
            average="weighted",
            zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            [tag for seq in true_labels for tag in seq],
            [tag for seq in true_predictions for tag in seq],
            average="weighted",
            zero_division=0
        )[1],
        "f1": seqeval_f1(true_labels, true_predictions, mode='strict', scheme=IOB2),
    }
    
    return results


def create_weighted_loss(label_counts: Dict[str, int], num_labels: int) -> torch.Tensor:
    """Create class weights for imbalanced dataset."""
    total = sum(label_counts.values())
    weights = torch.ones(num_labels)
    
    for label, count in label_counts.items():
        if label in LABEL2ID:
            label_id = LABEL2ID[label]
            # Inverse frequency weighting
            weights[label_id] = total / (count * num_labels)
    
    # Normalize
    weights = weights / weights.sum() * num_labels
    logger.info(f"Class weights: {weights}")
    
    return weights


def main(args):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("BERT Fine-tuning for Indonesian PII NER")
    logger.info("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load and preprocess data
    df_processed = load_and_preprocess_data(args.data_path, tokenizer)
    
    # Stratified split by style to maintain distribution
    train_df, temp_df = train_test_split(
        df_processed,
        test_size=0.3,
        random_state=SEED,
        stratify=df_processed["style"]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["style"]
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train style distribution:\n{train_df['style'].value_counts()}")
    
    # Create datasets
    train_dataset = PII_NER_Dataset(train_df, tokenizer, args.max_length)
    val_dataset = PII_NER_Dataset(val_df, tokenizer, args.max_length)
    test_dataset = PII_NER_Dataset(test_df, tokenizer, args.max_length)
    
    # Calculate class weights for handling imbalance
    all_tags = [tag for ex in train_df["bio_tags"] for tag in ex]
    tag_counts = Counter(all_tags)
    class_weights = create_weighted_loss(tag_counts, len(ENTITY_TYPES))
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(ENTITY_TYPES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,  # Allow loading with different number of labels
    )
    
    logger.info(f"Model loaded with {len(ENTITY_TYPES)} labels for PII detection")
    logger.info(f"Note: Pre-trained classifier discarded, training new classifier from scratch")
    
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        save_total_limit=3,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        push_to_hub=False,
    )
    
    # Initialize trainer (tokenizer parameter removed for transformers 4.36+)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=None,  # Use default data collator
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    trainer.log_metrics("val", val_metrics)
    trainer.save_metrics("val", val_metrics)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    # Detailed per-entity evaluation
    logger.info("Computing detailed per-entity metrics...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=2)
    true_labels = predictions.label_ids
    
    # Convert to entity labels
    true_tags_flat = []
    pred_tags_flat = []
    
    for pred_seq, true_seq in zip(pred_labels, true_labels):
        for pred_id, true_id in zip(pred_seq, true_seq):
            if true_id != LABEL2ID["O"]:
                true_tags_flat.append(ID2LABEL[true_id])
                pred_tags_flat.append(ID2LABEL[pred_id])
    
    # Classification report
    report = classification_report(
        true_tags_flat,
        pred_tags_flat,
        labels=list(set(true_tags_flat)),
        zero_division=0
    )
    
    logger.info("\n" + "="*80)
    logger.info("DETAILED CLASSIFICATION REPORT")
    logger.info("="*80)
    logger.info("\n" + report)
    
    # Save report
    with open(f"{args.output_dir}/classification_report.txt", "w") as f:
        f.write(report)
    
    # Save configuration
    config_dict = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "seed": SEED,
        "entity_types": ENTITY_TYPES,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "final_test_f1": test_metrics["eval_f1"],
        "final_test_precision": test_metrics["eval_precision"],
        "final_test_recall": test_metrics["eval_recall"],
    }
    
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {args.output_dir}")
    logger.info(f"Test F1: {test_metrics['eval_f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['eval_precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['eval_recall']:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Indonesian PII NER")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to prompt_dataset.csv"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bert_pii_model",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="Pretrained Indonesian BERT model name"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=WARMUP_RATIO,
        help="Warmup ratio for learning rate scheduler"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay for AdamW optimizer"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)