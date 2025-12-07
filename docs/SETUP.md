# SETUP

This document explains how to set up, run, train, and test the Exam Reference Sheet Generator using **Google Colab** and your **GitHub repository**.  
All core code lives inside the `src/` directory, and Colab notebooks import that code directly after cloning the repo.

---

# 1. Prerequisites

You will need:

- A GitHub repository containing this project’s directory structure
- A Google account
- Access to Google Colab
- Your training PDFs:
  - Two sets of:
    - `notes_i.pdf`
    - `exam_topics_i.pdf`
    - `cheatsheet_i.pdf` (your human-created reference sheet)
- PDFs for inference:
  - `notes_current.pdf`
  - `exam_topics_current.pdf`

No local Python installation is required unless you prefer to run everything locally.

---

# 2. Directory Structure

Your repository should follow this structure:
project-root/
│
├─ src/
│ ├─ data_processing.py
│ ├─ utils.py
│ ├─ train.py
│ ├─ infer.py
│ └─ pipeline.py
│
├─ data/
│ ├─ train/ # your training PDFs
│ ├─ samples/ # inference PDFs (user-provided)
│ └─ runtime/ # temporary working files
│
├─ models/ # saved fine-tuned model (after training)
│ └─ cheatsheet_model/
│
├─ notebooks/
│ ├─ training.ipynb
│ └─ inference.ipynb
│
├─ videos/
│
├─ docs/
│ ├─ SETUP.md
│ └─ training_curves.png
│
├─ environment.yml (or requirements.txt)
└─ README.md


---

# 3. Running the Project in Google Colab

All training and inference should be run through Google Colab so you can access GPUs and run HF models easily.

---

## 3.1. Open a Colab notebook

You may either:

### Option A:  
Open your repo notebook directly in Colab:

1. Go to **File → Open notebook → GitHub tab**  
2. Paste your GitHub repo URL  
3. Choose `notebooks/training.ipynb` or `notebooks/inference.ipynb`

### Option B:  
Start a blank notebook and clone your repo:
!git clone https://github.com/arely83/CS372FinalProject
%cd CS372FinalProject


---

## 3.2. Install dependencies inside Colab

Run this at the **top of every notebook**:

```python
! pip install transformers datasets accelerate sentencepiece
! pip install pymupdf reportlab matplotlib pandas


---

# 4. Training the Model

All model training is done inside **Google Colab** using the project’s code from GitHub.
The notebook is in notebooks/training.ipynb.

Run this notebook: https://colab.research.google.com/drive/1cARDH59mQB4LZpH5EnlBDk3MEbqb7Cem#scrollTo=LvydW67s2oVO

It will also visualize training curves and save the model.


---

# 5. Running Inference (Generating a New Cheat Sheet)

Inference is handled by notebooks/inference.ipynb using the pipeline in src/.
