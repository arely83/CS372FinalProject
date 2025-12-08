# CS372FinalProject
# Exam Reference Sheet Generator

This project automatically generates compact, page-limited exam reference sheets from class notes and instructor-provided exam topics using a fine-tuned FLAN-T5 model. It extracts text and diagrams from PDFs, summarizes key concepts, and produces a polished study sheet PDF optimized for space and readability.

---

## What It Does

This system takes two inputs (your class notes PDF and an exam topics PDF) and generates a highly compressed reference sheet tailored to your exam. It extracts relevant diagrams, applies abbreviations, and fits the result within a user-specified page limit. Internally, the pipeline uses PDF parsing, diagram retrieval, a fine-tuned T5 model for summarization, and custom PDF rendering. The system is trained on example pairs of notes, exam topics, and human-written cheat sheets to learn how to compress, prioritize, and structure content effectively.

---

## Quick Start 

Generate a cheat sheet using the [inference notebook](https://colab.research.google.com/drive/1JcPd02BrtB2o1V6OIxhLVjCzQASN2KBf#scrollTo=YMl1WkuWBH2V):
* Upload PDFs of your notes and a list of exam topics.
* Modify NOTES_PDF, TOPICS_PDF, and NUM_PAGES (notes PDF name, exam topics PDF name, number of pages allotted for reference sheet)
* Follow instructions in the notebook to retrieve the generated PDF.

The model should already be trained via the [training notebook](https://colab.research.google.com/drive/1cARDH59mQB4LZpH5EnlBDk3MEbqb7Cem#scrollTo=xBovSokrAufr).

---

## Video Links

Demo Video: BLAH

Technical Walkthrough: BLAH

---

## Evaluation
Quantitative Results

Training Curves:
![Training and Validation Loss](https://github.com/arely83/CS372FinalProject/blob/main/docs/training_curves.png?raw=true)


Qualitative Outcomes

* Model produces structured cheat sheets using headings + bullets

* Automatically selects relevant diagrams based on keyword overlap

* Consistently fits output within target page limit

* Demonstrated effective abbreviation usage (bc, w/, w/o, &)
