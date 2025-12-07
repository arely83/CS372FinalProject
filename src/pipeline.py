import os

from .data_processing import extract_text_and_images, concat_page_text
from .infer import load_model, generate_sheet_for_pages
from .utils import pick_relevant_images, render_reference_pdf


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
      - run model to generate cheatsheet text
      - select relevant diagrams
      - render PDF with page constraint
    """
    # 1) Extract text + images
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

    # 2) Load model
    tokenizer, model, device = load_model(model_dir)

    # 3) Generate text (page constrained)
    sheet_text = generate_sheet_for_pages(
        notes_text,
        exam_topics_text,
        tokenizer,
        model,
        device,
        num_pages=num_pages,
    )

    # 4) Select images
    image_paths = pick_relevant_images(
        exam_topics_text, notes_pages, notes_images, max_images=3
    )

    # 5) Render final PDF
    render_reference_pdf(
        sheet_text,
        image_paths,
        out_pdf,
        num_pages=num_pages,
    )

    return sheet_text, image_paths, out_pdf