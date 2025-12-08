# SETUP

This document explains how to set up, run, train, and test the Exam Reference Sheet Generator using **Google Colab** and your **GitHub repository**.  
All core code lives inside the `src/` directory, and Colab notebooks import that code directly after cloning the repo.

---

# 1. Prerequisites

You will need:

- Access to GitHub (here)
- Access to Google Colab
- PDFs for inference:
  - `notes_current.pdf`
  - `exam_topics_current.pdf`

No local Python installation is required unless you prefer to run everything locally.

---

# 2. Training the Model

All model training is done inside **Google Colab** using the project’s code from GitHub.
The notebook is in notebooks/training.ipynb.

See this notebook: https://colab.research.google.com/drive/1cARDH59mQB4LZpH5EnlBDk3MEbqb7Cem#scrollTo=LvydW67s2oVO

It should already have been run. It also contains training curves and an option to save the model.


---

# 3. Running Inference (Generating a New Cheat Sheet)

Inference is handled here: https://colab.research.google.com/drive/1JcPd02BrtB2o1V6OIxhLVjCzQASN2KBf#scrollTo=YMl1WkuWBH2V

* Upload PDFs of your notes and a list of exam topics.
* Modify NOTES_PDF, TOPICS_PDF, and NUM_PAGES (notes PDF name, exam topics PDF name, number of pages allotted for reference sheet)
* Follow instructions in the notebook to retrieve the generated PDF.
