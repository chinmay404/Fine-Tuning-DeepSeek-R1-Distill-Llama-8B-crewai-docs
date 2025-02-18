# LLM Fine-Tuning with LoRA on DeepSeek-R1 Model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TH_nP2vMqMSoSOL4UR9U_AUH7O-r9QNh?usp=sharing)]
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Face-%23FFD21E?style=flat&logo=huggingface)](https://huggingface.co/chinmay1718/FineTuned-DeepSeek-R1-Distill-Llama-8-CrewAi-Docs-unsloth)

This project provides an end-to-end pipeline for the Fine-Tuning-DeepSeek-R1-Distill-Llama-8B LLM on CrewAI Framework Docs using Low-Rank Adaptation (LoRA).
The pipeline consists of:

- **Document Scraping:** Retrieving and processing documentation to create a corpus of text data.
- **Synthetic Question Generation:** Automatically generating questions from the scraped documents using prompt-based techniques.
- **Synthetic Answer Generation:** Employing a retrieval-augmented generation (RAG) pipeline to produce answers from the generated questions.
- **Fine-Tuning with LoRA:** Applying low-rank adaptation to fine-tune the language models effectively.


## Strategies & Implementation

This project is structured around multiple key components to efficiently generate synthetic data and fine-tune LLMs. Below are the main strategies used in the implementation:

### Document Scraping

**Goal:** Extract useful textual content from structured online documentation to serve as the foundation for synthetic data generation.

**Implementation Details:**
- Uses **BeautifulSoup** to scrape and parse web pages.
- Filters out unnecessary elements and extracts meaningful text from `<p>` and `<pre>` tags.
- Saves each document as a `.txt` file inside the `scrapped_docs` directory.
- Implements URL filtering to avoid irrelevant or duplicate pages.

**Script:** `scrapper.py`  

---

### Synthetic Question Generation

**Goal:** Automatically generate high-quality questions from scraped documents using a prompt-based approach.

**Implementation Details:**
- Loads prompt templates from `prompts.yaml` (key: `question_prompt`).
- Uses **LangChain** and **OllamaLLM** to generate questions.
- Splits large documents into smaller batches (configured via `batch_size`).
- Implements **checkpointing** to ensure previously processed batches are not re-generated.
- Extracts generated questions from model responses using **regular expressions**.

**Script:** `question_gen.py`  

---

### Synthetic Answer Generation

**Goal:** Generate contextually relevant answers for the synthetic questions using a **Retrieval-Augmented Generation (RAG)** approach.

**Implementation Details:**
- Reads `questions.csv`, which contains generated questions.
- Uses both **ChatGroq** and **ChatDeepSeek** to answer questions.
- Retrieves relevant document context before generating responses.
- Parses model outputs using **custom regex-based extraction**.
- Saves generated answers into `final_data.csv`.

**Script:** `answergen.py`  

---

### Fine-Tuning with LoRA

**Goal:** Efficiently fine-tune large language models by adapting them to domain-specific knowledge.

**Implementation Details:**
- Uses **LoRA (Low-Rank Adaptation)** to fine-tune models without excessive computational overhead.
- Incorporates **pre-trained LLMs** as the base model.
- Optimizes key layers while keeping most of the model parameters frozen.
- Ensures **efficient training with minimal GPU requirements**.

---

This section provides a high-level overview of how each component is implemented. Let me know if you want to add more details before moving to the next section!



### 🔗WanDB
![fine_tune_graph](https://github.com/user-attachments/assets/8cbc65a4-78e9-44b3-9272-3b4c2ef42eb2)

### 🔗FineTuned Model Response 
![Finetuned Results](https://github.com/user-attachments/assets/967754f0-bee3-45ac-9f27-2aa2990f359e)
