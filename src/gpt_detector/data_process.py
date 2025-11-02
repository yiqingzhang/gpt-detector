"""
Data preprocessing pipeline for AI text detection training.

This module handles loading, transforming, and preparing datasets for model training.
"""

import os

from datasets import load_dataset
from transformers import RobertaTokenizer

from gpt_detector.utils import parse_args


def load_huggingface_dataset(data_id="yelp_review_full"):
    """Load dataset from Hugging Face Hub.
    
    Args:
        data_id: Hugging Face dataset identifier. Default is "yelp_review_full".
    
    Returns:
        Tuple of (train_dataset, test_dataset, validation_dataset).
    """
    dataset_train = load_dataset(data_id, split="train")
    dataset_test = load_dataset(data_id, split="test").shard(num_shards=2, index=0)
    dataset_val = load_dataset(data_id, split="test").shard(num_shards=2, index=1)

    return dataset_train, dataset_test, dataset_val


def make_fake_label(example):
    """Convert multi-class labels to binary classification.
    
    Maps original labels to binary: low scores (< 2) as human-generated (1),
    high scores (>= 2) as machine-generated (0). This is a simulation approach
    for demonstration purposes.
    
    Args:
        example: Dataset example with a 'label' field.
    
    Returns:
        Modified example with binary label.
    """
    if example["label"] < 2:
        example["label"] = 1  # human generated
    else:
        example["label"] = 0  # machine generated
    return example


def dataset_label_change(dataset_train, dataset_test, dataset_val):
    """Apply label transformation to all dataset splits.
    
    Args:
        dataset_train: Training dataset.
        dataset_test: Test dataset.
        dataset_val: Validation dataset.
    
    Returns:
        Tuple of transformed datasets (train, test, val).
    """
    dataset_train = dataset_train.map(make_fake_label)
    dataset_test = dataset_test.map(make_fake_label)
    dataset_val = dataset_val.map(make_fake_label)
    return dataset_train, dataset_test, dataset_val


def create_small_dataset(args, dataset_train, dataset_test, dataset_val):
    """Create smaller dataset subsets for faster experimentation.
    
    Args:
        args: Configuration arguments containing seed.
        dataset_train: Training dataset.
        dataset_test: Test dataset.
        dataset_val: Validation dataset.
    
    Returns:
        Tuple of smaller datasets (train: 128, test: 32, val: 32 samples).
    """
    dataset_train = dataset_train.shuffle(seed=args.seed).select(range(128))
    dataset_test = dataset_test.shuffle(seed=args.seed).select(range(32))
    dataset_val = dataset_val.shuffle(seed=args.seed).select(range(32))
    return dataset_train, dataset_test, dataset_val


def tokenize_func(args, dataset_example):
    """Tokenize text examples using RoBERTa tokenizer.
    
    Args:
        args: Configuration arguments containing saved_model_dir.
        dataset_example: Dataset example(s) with 'text' field.
    
    Returns:
        Tokenized examples with input_ids and attention_mask.
    """
    tokenizer = RobertaTokenizer.from_pretrained(args.saved_model_dir)
    return tokenizer(
        dataset_example["text"], padding="max_length", truncation=True, max_length=256
    )


def dataset_tokenization(dataset_train, dataset_test, dataset_val):
    """Apply tokenization to all dataset splits.
    
    Args:
        dataset_train: Training dataset.
        dataset_test: Test dataset.
        dataset_val: Validation dataset.
    
    Returns:
        Tuple of tokenized datasets (train, test, val).
    """
    dataset_train = dataset_train.map(tokenize_func, batched=True)
    dataset_test = dataset_test.map(tokenize_func, batched=True)
    dataset_val = dataset_val.map(tokenize_func, batched=True)
    return dataset_train, dataset_test, dataset_val


def save_dataset(args, dataset_train, dataset_test, dataset_val):
    """Save processed datasets to disk.
    
    Args:
        args: Configuration arguments containing datafolder path.
        dataset_train: Training dataset to save.
        dataset_test: Test dataset to save.
        dataset_val: Validation dataset to save.
    """
    dataset_train.save_to_disk(os.path.join(args.datafolder, "train"))
    dataset_test.save_to_disk(os.path.join(args.datafolder, "test"))
    dataset_val.save_to_disk(os.path.join(args.datafolder, "val"))


def run_data_preprocess(args):
    """Execute the complete data preprocessing pipeline.
    
    This function orchestrates the entire data preparation process:
    1. Load dataset from Hugging Face
    2. Transform labels to binary classification
    3. Create smaller subsets for faster training
    4. Tokenize all text data
    5. Save processed datasets to disk
    
    Args:
        args: Configuration arguments containing all necessary parameters.
    """
    dataset_train, dataset_test, dataset_val = load_huggingface_dataset()
    dataset_train, dataset_test, dataset_val = dataset_label_change(
        dataset_train, dataset_test, dataset_val
    )
    dataset_train, dataset_test, dataset_val = create_small_dataset(
        args, dataset_train, dataset_test, dataset_val
    )
    dataset_train, dataset_test, dataset_val = dataset_tokenization(
        args, dataset_train, dataset_test, dataset_val
    )
    save_dataset(args, dataset_train, dataset_test, dataset_val)


if __name__ == "__main__":
    args = parse_args()
    run_data_preprocess(args)
