from typing import Tuple, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .utils import (
    build_input,
    apply_abbreviations,
    normalize_bold_markers,
    estimated_char_capacity,
)


def load_model(model_dir: str):
    """
    Load tokenizer + model from a local directory and put model on the right device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    return tokenizer, model, device


def generate_sheet_for_pages(
    notes_text: str,
    exam_topics_text: str,
    tokenizer,
    model,
    device,
    num_pages: int,
    vocab_terms: Optional[List[str]] = None,
    max_input_length: int = 2048,
    max_new_tokens: int = 768,
) -> str:
    """
    Generate a cheatsheet text for the given notes + topics,
    constrained by an estimated character capacity (num_pages),
    optionally nudged by a list of vocab_terms.
    """
    # 1) Estimate how much text we can fit
    char_capacity = estimated_char_capacity(
        num_pages=num_pages,
        font_size=8,
        columns=2,
    )

    # 2) Build the instruction-style prompt
    input_text = build_input(
        notes_text=notes_text,
        exam_topics_text=exam_topics_text,
        vocab_terms=vocab_terms,
    )

    # 3) Tokenize and generate
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 4) Apply abbreviations and simple bold normalization
    generated = apply_abbreviations(generated)
    generated = normalize_bold_markers(generated)

    # 5) Truncate to approximate page capacity
    if len(generated) > char_capacity:
        generated = generated[:char_capacity]

    return generated
