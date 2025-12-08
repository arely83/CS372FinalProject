from typing import List, Tuple

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

from .data_processing import load_example
from .utils import build_input

from datasets import Dataset



def build_raw_dataset(train_examples: List[Tuple[str, str, str]]):
    """
    train_examples: list of (notes_pdf, topics_pdf, cheatsheet_pdf) paths.
    Returns a HuggingFace Dataset with 'input_text' and 'target_text'.
    """
    data = []
    for (notes_pdf, topics_pdf, cheat_pdf) in train_examples:
        notes_text, topics_text, cheat_text = load_example(
            notes_pdf, topics_pdf, cheat_pdf
        )
        input_text = build_input(notes_text, topics_text)
        data.append({"input_text": input_text, "target_text": cheat_text})
    return Dataset.from_list(data)


def split_dataset(full_dataset: Dataset, seed: int = 42):
    """
    Split into train/val/test.

    For small datasets:
      - n == 1: use the same example for train/val/test
      - n == 2: 1 train, 1 val, and reuse val as test
      - n >= 3: use a 60/20/20 split (approx)
    """
    n = len(full_dataset)

    if n == 1:
        # Degenerate case: one example used everywhere
        train_dataset = full_dataset
        val_dataset = full_dataset
        test_dataset = full_dataset
        return train_dataset, val_dataset, test_dataset

    if n == 2:
        # 50/50 split, reuse val as test
        temp = full_dataset.train_test_split(test_size=0.5, seed=seed)
        train_dataset = temp["train"]
        val_dataset = temp["test"]
        test_dataset = temp["test"]
        return train_dataset, val_dataset, test_dataset

    # Default: n >= 3 -> 60/20/20 (approx)
    temp = full_dataset.train_test_split(test_size=0.4, seed=seed)
    test_valid = temp["test"].train_test_split(test_size=0.5, seed=seed)
    train_dataset = temp["train"]          # ~60%
    val_dataset = test_valid["train"]      # ~20%
    test_dataset = test_valid["test"]      # ~20%
    return train_dataset, val_dataset, test_dataset


def tokenize_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer,
    max_input_length: int = 2048,
    max_target_length: int = 512,
):
    def preprocess(example):
        model_inputs = tokenizer(
            example["input_text"],
            max_length=max_input_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["target_text"],
                max_length=max_target_length,
                truncation=True,
            )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train = train_dataset.map(preprocess, batched=False)
    tokenized_val = val_dataset.map(preprocess, batched=False)
    tokenized_test = test_dataset.map(preprocess, batched=False)
    return tokenized_train, tokenized_val, tokenized_test


def create_trainer(
    tokenized_train,
    tokenized_val,
    model,
    tokenizer,
    output_dir: str,
):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=15,
        weight_decay=0.01,  # L2 regularization
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    early_stop = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stop],
    )
    return trainer


def train_model(
    train_examples: List[Tuple[str, str, str]],
    model_name: str = "google/flan-t5-base",
    output_dir: str = "models/cheatsheet_model",
):
    """
    End-to-end training:
      - build dataset from PDFs
      - split into train/val/test
      - tokenize
      - create trainer
      - train and save model

    Returns (trainer, tokenized_test)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Dataset
    full_dataset = build_raw_dataset(train_examples)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

    # 2) Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    tokenized_train, tokenized_val, tokenized_test = tokenize_datasets(
        train_dataset, val_dataset, test_dataset, tokenizer
    )

    # 3) Trainer
    trainer = create_trainer(
        tokenized_train, tokenized_val, model, tokenizer, output_dir
    )

    # 4) Train
    trainer.train()

    # 5) Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer, tokenized_test


def get_training_log_df(trainer: Seq2SeqTrainer) -> pd.DataFrame:
    """Utility: convert trainer log history to a pandas DataFrame for plotting."""
    log_history = trainer.state.log_history
    return pd.DataFrame(log_history)