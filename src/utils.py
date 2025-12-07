import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def build_input(notes_text, exam_topics_text):
    """Constructs the instruction-style input for the model."""
    return f"""
You are generating a dense exam reference sheet.

Exam topics:
{exam_topics_text}

Class notes:
{notes_text}

Write a bullet-point style cheat sheet covering ONLY the exam topics.
Use abbreviations where possible (bc, w/, &, w/o).
Use compact phrases instead of full sentences.
Output ONLY the cheat sheet text, with clear headings and bullet points.
""".strip()


ABBREV_MAP = {
    "because": "bc",
    "with": "w/",
    "without": "w/o",
    "and": "&",
    "between": "btwn",
    "example": "ex.",
    "approximately": "~",
}


def apply_abbreviations(text, mapping=ABBREV_MAP):
    """Apply simple word-level abbreviations to the text."""
    pattern = r"\b\w+\b"

    def repl(match):
        word = match.group(0)
        lower = word.lower()
        if lower in mapping:
            abbr = mapping[lower]
            if word.isupper():
                return abbr.upper()
            return abbr
        return word

    return re.sub(pattern, repl, text)


def estimated_char_capacity(
    num_pages,
    font_size=8,
    page_size=letter,
    margin=0.5 * inch,
):
    """Rough estimate of how many characters fit into N PDF pages."""
    width, height = page_size
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin

    line_height = font_size * 1.2
    lines_per_page = int(usable_height // line_height)
    chars_per_line = int(usable_width / (font_size * 0.5))

    total_chars = num_pages * lines_per_page * chars_per_line
    return int(total_chars * 0.9)


def simple_keyword_score(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def pick_relevant_images(exam_topics_text, notes_pages, notes_images, max_images=4):
    """
    Very simple heuristic: pick images from pages whose text overlaps exam topics.
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", exam_topics_text.lower())
    keywords = list(set(words))

    page_scores = {
        p["page_num"]: simple_keyword_score(p["text"], keywords)
        for p in notes_pages
    }

    best_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)

    selected_imgs = []
    for page_num, score in best_pages:
        if score == 0:
            break
        for img in notes_images:
            if img["page_num"] == page_num:
                selected_imgs.append(img["img_path"])
                if len(selected_imgs) >= max_images:
                    return selected_imgs
    return selected_imgs


def render_reference_pdf(
    sheet_text,
    image_paths,
    out_path,
    num_pages,
    font_name="Helvetica",
    font_size=8,
    margin=0.5 * inch,
):
    """Render the cheatsheet text + images into a multi-page PDF."""
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter

    line_height = font_size * 1.2
    usable_width = width - 2 * margin

    def new_page():
        c.showPage()
        c.setFont(font_name, font_size)
        return height - margin

    c.setFont(font_name, font_size)
    x = margin
    y = height - margin

    words = sheet_text.split()
    line = ""
    pages_used = 1

    for w in words:
        test = (line + " " + w).strip()
        if len(test) * font_size * 0.5 > usable_width:
            if y < margin + line_height:
                if pages_used >= num_pages:
                    c.save()
                    return
                pages_used += 1
                y = new_page()
            c.drawString(x, y, line)
            y -= line_height
            line = w
        else:
            line = test

    if line:
        if y < margin + line_height and pages_used < num_pages:
            pages_used += 1
            y = new_page()
        if y >= margin + line_height:
            c.drawString(x, y, line)
            y -= line_height

    for img_path in image_paths:
        if pages_used > num_pages:
            break
        img_width = 2.5 * inch
        img_height = 2.0 * inch
        if y < margin + img_height:
            if pages_used >= num_pages:
                break
            pages_used += 1
            y = new_page()
        c.drawImage(
            img_path,
            x,
            y - img_height,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= img_height + 0.2 * inch

    c.save()