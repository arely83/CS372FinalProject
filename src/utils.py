import re
from typing import List

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas  # can stay, even if not used
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Image,
)


def extract_vocab_terms_from_notes(notes_text: str, max_terms: int = 50) -> List[str]:
    """
    Heuristic extraction of vocabulary terms from the notes.

    - Looks for patterns like "Term - definition" or "Term: definition".
    - Treats the left-hand side as the vocab term.
    - Also picks up ALL-CAPS words that look like key concepts.
    """
    vocab_terms = set()

    # Pattern-based extraction from "Term - definition" style lines
    for line in notes_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # patterns: "Term - def", "Term – def", "Term: def"
        match = re.match(r"^([A-Za-z0-9 /()\[\]_]+)\s*[-–:]\s+.+$", line)
        if match:
            term = match.group(1).strip()
            # Avoid super short or obviously junk tokens
            if len(term) > 2:
                vocab_terms.add(term)

    # ALL-CAPS “concept” words of length >= 3
    for token in re.findall(r"\b[A-Z]{3,}\b", notes_text):
        if len(token) >= 3:
            vocab_terms.add(token)

    # Limit to a reasonable number to avoid blowing up the prompt
    return sorted(vocab_terms)[:max_terms]


def build_input(notes_text, exam_topics_text, vocab_terms=None):
    """
    Construct a strong instruction-style input for the model.

    - PRIORITIZE compressing the student's notes.
    - Use exam topics as a coverage checklist.
    - Include vocabulary terms + short definitions when possible.
    """
    vocab_block = ""
    if vocab_terms:
        vocab_list = "\n".join(f"- {t}" for t in vocab_terms)
        vocab_block = f"""
Key vocabulary terms that MUST appear with short definitions.
When you first introduce each term, surround it with **double asterisks** to mark it as important:
{vocab_list}
"""

    prompt = f"""You are generating a VERY COMPACT exam reference sheet.

Your job:
- Create a SUPER CONDENSED, efficient version of the STUDENT'S NOTES.
- PRIORITIZE the NOTES over the topic list.
- COVER EVERY EXAM TOPIC at least briefly.
- INCLUDE vocabulary terms and short DEFINITIONS.
- Use dense bullet points and short phrases, not full sentences.
- Organize content by topic in a logical order.
- Assume the cheat sheet will be rendered in two columns, but DO NOT mention columns in the text.

Exam topics (use these as a checklist, but do NOT just repeat them verbatim):
{exam_topics_text}

Student notes (this is the main source of content – compress this aggressively):
{notes_text}
{vocab_block}
Now write the final cheat sheet below. Start directly with the content:
"""
    return prompt.strip()


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
    sheet_text: str,
    image_paths: List[str],
    out_path: str,
    num_pages: int,
    font_name: str = "Helvetica",
    font_size: int = 8,
    margin: float = 0.5 * inch,
):
    """
    Render the cheatsheet text + images into a TWO-COLUMN PDF using ReportLab Platypus.

    We assume the text has already been truncated to roughly fit `num_pages`
    via `estimated_char_capacity` earlier in the pipeline.
    """
    # --- Document + layout setup ---
    doc = BaseDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    # Two columns per page with a small gap
    column_gap = 0.2 * inch
    frame_width = (doc.width - column_gap) / 2.0
    frame_height = doc.height

    frame1 = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        frame_width,
        frame_height,
        id="col1",
    )
    frame2 = Frame(
        doc.leftMargin + frame_width + column_gap,
        doc.bottomMargin,
        frame_width,
        frame_height,
        id="col2",
    )

    template = PageTemplate(id="TwoCol", frames=[frame1, frame2])
    doc.addPageTemplates([template])

    # --- Styles ---
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.fontName = font_name
    body_style.fontSize = font_size
    body_style.leading = font_size * 1.2

    # --- Build story from text + images ---
    story = []

    # Split text into paragraphs by line (you can tweak this if needed)
    lines = [line.strip() for line in sheet_text.split("\n") if line.strip()]

    for line in lines:
        # Basic replacement so bullet characters behave OK in Paragraph
        # (ReportLab understands a subset of HTML; here we just treat text literally)
        para_text = line.replace("•", "-")
        story.append(Paragraph(para_text, body_style))
        story.append(Spacer(1, 1.5))

    # Append images at the end (or you can insert them earlier in the loop)
    for img_path in image_paths or []:
        try:
            story.append(Image(img_path, width=1.8 * inch, height=1.3 * inch))
            story.append(Spacer(1, 4))
        except Exception:
            # If an image can't be loaded, just skip it
            continue

    # NOTE: We rely on earlier truncation (estimated_char_capacity) to
    # keep the story short enough to fit in `num_pages`. BaseDocTemplate
    # doesn't have a simple "max pages" cap, so the page limit is enforced
    # indirectly by how much text we feed in.
    doc.build(story)