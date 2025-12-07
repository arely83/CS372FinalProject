# CS372FinalProject
# Exam Reference Sheet Generator

This project automatically generates compact, page-limited exam reference sheets from class notes and instructor-provided exam topics using a fine-tuned FLAN-T5 model. It extracts text and diagrams from PDFs, summarizes key concepts, and produces a polished study sheet PDF optimized for space and readability.

---

## What It Does

This system takes two inputs (your class notes PDF and an exam topics PDF) and generates a highly compressed reference sheet tailored to your exam. It extracts relevant diagrams, applies abbreviations, and fits the result within a user-specified page limit. Internally, the pipeline uses PDF parsing, diagram retrieval, a fine-tuned T5 model for summarization, and custom PDF rendering. The system is trained on example pairs of notes, exam topics, and human-written cheat sheets to learn how to compress, prioritize, and structure content effectively.

---

## Quick Start

1. **Run the training notebook:**

3. **Generate a cheat sheet using the inference notebook**

---

## Video Links

Demo Video: placeholder – link coming soon

Technical Walkthrough: placeholder – link coming soon

---

## Evaluation
Quantitative Results

Training Loss Curve:
(Insert screenshot)
placeholder – training curves image will go here (docs/training_curves.png)

Validation Loss Curve:
placeholder

Test Set Performance:

Final test loss: placeholder

Observed improvements: placeholder

Qualitative Outcomes

Model produces structured cheat sheets using headings + bullets

Automatically selects relevant diagrams based on keyword overlap

Consistently fits output within target page limit

Demonstrated effective abbreviation usage (bc, w/, w/o, &)

Example output (text excerpt):