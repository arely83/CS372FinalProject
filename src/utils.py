import re
from typing import List, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Image,
)


# -------------------------
# Vocab extraction utilities
# -------------------------

def extract_heading_bullet_groups(
    notes_text: str,
    max_groups: int = 50,
    max_bullets_per_group: int = 6,
) -> List[dict]:
    """
    Detect simple patterns where a HEADING is followed by one or more bullet points.
    Returns a list of {"term": heading, "bullets": [bullet1, bullet2, ...]}.

    This is heuristic, but works well for notes where:
      Term or concept name
      - first defining bullet
      - more details / examples
    """
    lines = notes_text.splitlines()
    groups: List[dict] = []

    current_term: str | None = None
    current_bullets: List[str] = []

    # Heuristic: a heading is a short-ish line starting with a capital letter,
    # without a leading bullet symbol. Adjust as needed for your notes style.
    heading_pattern = re.compile(r"^[A-Z][A-Za-z0-9 /&(),'\"\-]{2,}$")
    bullet_pattern = re.compile(r"^\s*[-•●–]\s+")

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        # Bullet line
        if bullet_pattern.match(stripped):
            bullet_text = bullet_pattern.sub("", stripped).strip()
            if current_term is not None and bullet_text:
                if len(current_bullets) < max_bullets_per_group:
                    current_bullets.append(bullet_text)
            continue

        # Heading line (only treat it as a heading if it's not a bullet)
        if heading_pattern.match(stripped) and not bullet_pattern.match(stripped):
            # If we were already in a group, close it out
            if current_term and current_bullets:
                groups.append({"term": current_term, "bullets": current_bullets})
                if len(groups) >= max_groups:
                    return groups
            # Start a new group
            current_term = stripped
            current_bullets = []
            continue

        # Other lines (non-bullets, non-headings) are ignored here. They’ll still
        # be present in the raw notes text that the model can see.

    # Final group at end of file
    if current_term and current_bullets and len(groups) < max_groups:
        groups.append({"term": current_term, "bullets": current_bullets})

    return groups

def extract_vocab_terms_from_notes(notes_text: str, max_terms: int = 50) -> List[str]:
    """
    Heuristic extraction of vocabulary terms from the notes.

    - Looks for patterns like "Term - definition" or "Term: definition".
    - Treats the left-hand side as the vocab term.
    - Also picks up ALL-CAPS words that might be key concepts.
    """
    vocab_terms = set()

    # 1) Pattern-based extraction from "Term - definition" style lines
    for line in notes_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # patterns: "Term - def", "Term – def", "Term: def"
        match = re.match(r"^([A-Za-z0-9 /()\[\]_]+)\s*[-–:]\s+.+$", line)
        if match:
            term = match.group(1).strip()
            if len(term) > 2:
                vocab_terms.add(term)

    # 2) ALL-CAPS “concept” words of length >= 3
    for token in re.findall(r"\b[A-Z]{3,}\b", notes_text):
        if len(token) >= 3:
            vocab_terms.add(token)

    return sorted(vocab_terms)[:max_terms]


# -------------------------
# Prompt construction
# -------------------------

def build_input(notes_text, exam_topics_text, vocab_terms=None):
    """
    Prompt for cheat sheet generation.
    """

    vocab_block = ""
    if vocab_terms:
        vocab_list = "\n".join(f"- {t}" for t in vocab_terms)
        vocab_block = f"""
Key terms to define briefly (one line each):
{vocab_list}
"""

    prompt = f"""
You are generating a compact, highly structured exam reference sheet.

### TASK
Create a dense, two-column-friendly cheat sheet that summarizes the NOTES.
Prioritize the NOTES over the exam topics.
Cover every exam topic at least briefly.

### STYLE RULES
- Use short bullet points (no full sentences).
- Use abbreviations when possible (bc, w/, w/o, &).
- Put **bold** markers around key terms when they are introduced.
- Keep definitions short and highlight relationships.
- Avoid repetition.
- Group related ideas under clear section headers.
- Assume the output will be printed, so use clean formatting.

### NOTES (primary source to compress):
{notes_text}

### EXAM TOPICS (ensure these appear somewhere):
{exam_topics_text}

{vocab_block}

### OUTPUT REQUIREMENTS
Write a clear, concise cheat sheet.
Start immediately with the first section header.
Do NOT mention formatting instructions.
    """
    return prompt.strip()


# -------------------------
# Abbreviation handling
# -------------------------

ABBREV_MAP = {
    "because": "bc",
    "with": "w/",
    "without": "w/o",
    "and": "&",
    "between": "btwn",
    "example": "ex.",
    "approximately": "~",
}


def apply_abbreviations(text: str, mapping: dict = ABBREV_MAP) -> str:
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


def normalize_bold_markers(text: str) -> str:
    """
    Very simple formatting helper:
    Turn **TERM** into TERM in ALL CAPS so vocabulary terms visually stand out,
    even if we don't do true bold rendering.
    """
    def repl(match):
        inner = match.group(1)
        return inner.upper()

    return re.sub(r"\*\*(.+?)\*\*", repl, text)


# -------------------------
# Capacity & image heuristics
# -------------------------

def estimated_char_capacity(
    num_pages: int,
    font_size: int = 8,
    page_size=letter,
    margin: float = 0.5 * inch,
    columns: int = 2,
    column_gap: float = 0.2 * inch,
    safety_factor: float = 0.95,
) -> int:
    """
    Rough estimate of how many characters fit into N PDF pages
    *for the current TWO-COLUMN layout*.

    We:
      - compute lines per column from vertical space and line height
      - estimate characters per line from column width and font size
      - multiply by number of columns and pages
      - apply a safety_factor so we don't overflow.
    """
    page_width, page_height = page_size
    usable_width = page_width - 2 * margin
    usable_height = page_height - 2 * margin

    # vertical capacity
    line_height = font_size * 1.2
    lines_per_column = int(usable_height // line_height)

    # horizontal capacity: width of a single column
    total_gap = column_gap * (columns - 1)
    col_width = (usable_width - total_gap) / columns

    # empirically, ~0.55 * font_size per character is a decent approximation
    chars_per_line = int(col_width / (font_size * 0.55))

    total_chars = num_pages * columns * lines_per_column * chars_per_line
    return int(total_chars * safety_factor)


def simple_keyword_score(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def pick_relevant_images(
    exam_topics_text: str,
    notes_pages: List[dict],
    notes_images: List[dict],
    max_images: int = 4,
) -> List[str]:
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


# -------------------------
# Chunking for long notes
# -------------------------

def chunk_text(text: str, max_tokens: int = 350) -> List[str]:
    """
    Splits long text into chunks small enough for T5 to summarize.
    Uses a simple "words per chunk" heuristic.
    """
    words = text.split()
    chunks: List[str] = []
    cur: List[str] = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# -------------------------
# Two-column PDF renderer
# -------------------------

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

    # Split text into paragraphs by line
    lines = [line.strip() for line in sheet_text.split("\n") if line.strip()]

    if not lines:
        lines = [sheet_text.strip()]

    for line in lines:
        para_text = line.replace("•", "-")
        story.append(Paragraph(para_text, body_style))
        story.append(Spacer(1, 1.5))

    # Append images at the end (or you can insert them earlier in the loop)
    for img_path in image_paths or []:
        try:
            story.append(Image(img_path, width=1.8 * inch, height=1.3 * inch))
            story.append(Spacer(1, 4))
        except Exception:
            continue

    # NOTE: We rely on earlier truncation (estimated_char_capacity) to
    # keep the story short enough. BaseDocTemplate doesn't easily expose
    # a hard "max pages" cap.
    doc.build(story)
