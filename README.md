# CS372FinalProject
# Exam Reference Sheet Generator

This project automatically generates compact, page-limited exam reference sheets from class notes and instructor-provided exam topics using a fine-tuned FLAN-T5 model. It extracts text and diagrams from PDFs, summarizes key concepts, and produces a polished study sheet PDF optimized for space and readability.

---

## What It Does

This system takes two inputs (your class notes PDF and an exam topics PDF) and generates a highly compressed reference sheet tailored to your exam. It extracts relevant diagrams, applies abbreviations, and fits the result within a user-specified page limit. Internally, the pipeline uses PDF parsing, diagram retrieval, a fine-tuned T5 model for summarization, and custom PDF rendering. The system is trained on example pairs of notes, exam topics, and human-written cheat sheets to learn how to compress, prioritize, and structure content effectively.

I intended to download the trained model to my local machine and then upload it to the models folder, but my laptop kept running out of system memory and crashing, which is why the folder is empty.

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

Because I only had a very small number of training examples (on the order of a few samples), the training and validation curves are extremely noisy and not very informative. With a batch size of 1, gradient accumulation of 2, and only a couple of optimization steps per epoch, the recorded training loss reflects the idiosyncrasies of individual examples rather than a stable trend. The validation set is also just a single example in the small-data regime, so the validation loss appears almost flat across epochs. As a result, the loss curves look unstable and “bad” visually, but this is primarily a consequence of the tiny dataset and fine-grained logging rather than a bug in the training loop.

Qualitative Outcomes

* Model cheat sheet using content from notes

* Automatically selects relevant diagrams

* Fits output within target page limit

Because this project is an early prototype, the generated PDFs reflect the limitations of the current pipeline rather than the overall potential of the system. Text extraction, diagram detection, and summarization are still handled with relatively simple heuristics, which leads to incomplete or low-fidelity output. As a result, the final PDF is functional enough to demonstrate the workflow, but not polished or representative of what a more advanced implementation could achieve.

Creating a clean, well-organized “reference sheet” from heterogeneous inputs (slides, notes, diagrams, topics) is an inherently hard problem that usually requires deep semantic understanding and layout reasoning. Because the system relies on lightweight models and rule-based formatting, it struggles with relevance ranking, space prioritization, and visual placement, which leads to lower-quality PDFs.

The quality of the generated PDF is heavily dependent on the clarity and structure of the input documents. Notes and scraped diagrams vary widely in formatting and readability, which introduces noise into the RAG pipeline and reduces coherence in the final PDF.
