import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .utils import (
    build_input,
    apply_abbreviations,
    estimated_char_capacity,
)


def load_model(model_dir: str = "models/cheatsheet_model"):
    """Load fine-tuned model + tokenizer from disk."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    return tokenizer, model, device


def generate_cheatsheet_text(
    notes_text: str,
    exam_topics_text: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate cheat sheet text from notes + topics."""
    prompt = build_input(notes_text, exam_topics_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    text = apply_abbreviations(text)
    return text


def generate_sheet_for_pages(
    notes_text: str,
    exam_topics_text: str,
    tokenizer,
    model,
    device: str,
    num_pages: int,
) -> str:
    """Generate cheatsheet text but truncated to fit N pages."""
    cap = estimated_char_capacity(num_pages)
    max_new_tokens = max(128, min(1024, cap // 4))

    sheet = generate_cheatsheet_text(
        notes_text,
        exam_topics_text,
        tokenizer,
        model,
        device,
        max_new_tokens=max_new_tokens,
    )

    if len(sheet) > cap:
        sheet = sheet[:cap]
    return sheet
