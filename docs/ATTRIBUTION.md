The model is trained on my own class notes and resources from classes I've taken at Duke.

**ML Libraries**
* Hugging Face Transformers: https://github.com/huggingface/transformers - Used for loading pretrained sequence-to-sequence models (FLAN-T5), tokenizers, the Seq2SeqTrainer, and training utilities
* Hugging Face Datasets: https://github.com/huggingface/datasets - Used to construct a text dataset from PDF extractions, map preprocessing functions, and create train/validation/test splits.
* PyTorch: https://pytorch.org/

**PDF and Text Processing**
* PyMuPDF (fitz): https://github.com/pymupdf/PyMuPDF - Used for parsing and extracting text and images from input PDFs (notes, exam topics, diagrams).
* ReportLab: https://www.reportlab.com/dev/docs/ - Used for rendering the final cheat sheet PDF with page-limit constraints, images, and custom layout.

**Supporting Libraries**
* Pandas
* Matplotlib

**Pretrained Models**
* FLAN-T5 (google/flan-t5-base): https://huggingface.co/google/flan-t5-base - Used as the base seq2seq language model for summarization and compression of class notes into cheat sheets. Model weights and tokenizer provided by Google and hosted by Hugging Face.


I used the LLM ChatGPT for the following:
* Prompted the LLM to give me advice on meeting 15 elements of the rubric, and it advised me to mention batching and shuffling in the report (since HF already does it).
* Prompted the LLM for help on structuring my project, specifically how to integrate Google CoLab with the directory structure I have locally/on Git.
* Prompted the LLM for help in writing code for google/flan-t5-base.
* Prompted the LLM to help write the method generate_cheatsheet_text in infer.py.
* Prompted the LLM for what heuristic to use for image selection and how to implement it in pick_relevant_images
* Prompted the LLM to write estimated_char_capacity in utils.py.
* Prompted the LLM to help me debug the following:
    * split_dataset in train.py in the case of smaller datasets
    * Version issue with transformers
    * Resolving issue with saving model to local machine due to memory limitations
    * Debugging saving trained model to Google Drive and retrieving it for inference (training and inference notebooks)