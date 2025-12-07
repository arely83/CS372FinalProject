import pathlib
import fitz  # PyMuPDF

def extract_text_and_images(pdf_path, image_dir=None):
    """
    Extract text and (optionally) images from a PDF.

    Args:
        pdf_path: str or Path to PDF.
        image_dir: directory to save extracted images, or None for no images.

    Returns:
        pages: list of { 'page_num': int, 'text': str }
        images: list of { 'page_num': int, 'img_path': str }
    """
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    images = []

    if image_dir is not None:
        image_dir = pathlib.Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_num": page_num, "text": text})

        if image_dir is not None:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_path = image_dir / f"p{page_num}_img{img_index}.png"
                pix.save(str(img_path))
                images.append({"page_num": page_num, "img_path": str(img_path)})

    return pages, images


def concat_page_text(pages):
    """Concatenate page text list into a single string."""
    return "\n".join(p["text"] for p in pages)


def load_example(notes_pdf_path, topics_pdf_path, cheat_pdf_path):
    """
    Load one training triple: (notes, exam topics, gold cheatsheet) from PDFs.
    Returns three strings.
    """
    notes_pages, _ = extract_text_and_images(notes_pdf_path)
    topics_pages, _ = extract_text_and_images(topics_pdf_path)
    cheat_pages, _ = extract_text_and_images(cheat_pdf_path)

    notes_text = concat_page_text(notes_pages)
    topics_text = concat_page_text(topics_pages)
    cheat_text = concat_page_text(cheat_pages)

    return notes_text, topics_text, cheat_text