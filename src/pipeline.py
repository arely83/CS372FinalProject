import os

from .data_processing import extract_text_and_images, concat_page_text
from .infer import load_model, generate_sheet_for_pages
from .utils import (
    build_input,
    apply_abbreviations,
    pick_relevant_images,
    estimated_char_capacity,
    extract_vocab_terms_from_notes,
    chunk_text,
    render_reference_pdf,
)


def generate_reference_sheet_pipeline(
    notes_pdf_path: str,
    exam_topics_pdf_path: str,
    num_pages: int,
    img_dir: str = "data/runtime",
    model_dir: str = "models/cheatsheet_model",
    out_pdf: str = "data/runtime/generated_reference.pdf",
):
    """
    End-to-end pipeline:
      - load & parse PDFs
      - CHUNK the notes and summarize each chunk
      - MERGE chunk summaries into a final cheat sheet
      - select relevant diagrams
      - render PDF with two-column layout and page constraint
    """
    # 1) Extract text + images from PDFs
    notes_pages, notes_images = extract_text_and_images(
        notes_pdf_path,
        os.path.join(img_dir, "notes_imgs"),
    )
    topics_pages, _ = extract_text_and_images(
        exam_topics_pdf_path,
        os.path.join(img_dir, "topics_imgs"),
    )

    notes_text = concat_page_text(notes_pages)
    exam_topics_text = concat_page_text(topics_pages)

    # 1b) Extract vocabulary terms from FULL notes
    global_vocab_terms = extract_vocab_terms_from_notes(notes_text)

    # 2) Load model
    tokenizer, model, device = load_model(model_dir)

    # 3) Chunk the notes and summarize each chunk separately
    note_chunks = chunk_text(notes_text, max_tokens=350)
    chunk_summaries = []

    for chunk_text_str in note_chunks:
        # For each chunk, extract local vocab (helps coverage)
        chunk_vocab = extract_vocab_terms_from_notes(chunk_text_str, max_terms=30)

        mini_summary = generate_sheet_for_pages(
            notes_text=chunk_text_str,
            exam_topics_text=exam_topics_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_pages=1,          # each chunk -> ~1 page worth of content
            vocab_terms=chunk_vocab,
        )
        chunk_summaries.append(mini_summary)

    # Combine mini summaries into one intermediate "notes summary"
    notes_summary = "\n".join(chunk_summaries)

    # 4) Final condensation step: summarize the combined notes summary
    # into the requested number of pages, using the global vocab list.
    final_sheet_text = generate_sheet_for_pages(
        notes_text=notes_summary,
        exam_topics_text=exam_topics_text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_pages=num_pages,
        vocab_terms=global_vocab_terms,
    )

    # 5) Select images
    image_paths = pick_relevant_images(
        exam_topics_text, notes_pages, notes_images, max_images=3
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(out_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 6) Render final PDF (two-column)
    render_reference_pdf(
        final_sheet_text,
        image_paths,
        out_pdf,
        num_pages=num_pages,
    )

    return final_sheet_text, image_paths, out_pdf
