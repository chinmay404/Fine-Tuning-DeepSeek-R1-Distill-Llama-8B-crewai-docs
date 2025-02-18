# Fine-Tuning DeepSeek-R1-Distill-Llama-8B Using LoRA (Low-Rank Adaptation) and Unsloth

This repository contains code for fine-tuning **DeepSeek-R1-Distill-Llama-8B** using **LoRA** and integrating retrieval-augmented generation (RAG) for enhanced performance. It also includes synthetic data generation code.

![Project Banner](wandb_screenshot.png) <!-- Replace with actual image link -->

## 🚀 Features

- **Fine-tuning with LoRA**: Efficient adaptation of large models.
- **Synthetic Data Generation**: Automated dataset creation from scrapping documentation.
- **Logging & Monitoring**: Integrated with **Weights & Biases (W&B)** for tracking.
- **Hugging Face Model**: Available on [Hugging Face Hub](#hugging-face-models).

---

## 📂 Project Structure

📦 Fine_Tune_LLM_Using_LORA ├── 📜 Fine_Tune_LLM_Using_LORA.ipynb # Jupyter notebook for fine-tuning ├── 📜 answers_gen.py # Synthetic data generation & RAG pipeline ├── 📜 prompts.yaml # Prompt templates ├── 📜 questions.csv # Input questions for model ├── 📜 final_data.csv # Generated answers ├── 📜 requirements.txt # Dependencies └── 📜 README.md # Documentation

yaml
Copy

---

## 📌 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/Fine_Tune_LLM_LoRA.git
   cd Fine_Tune_LLM_LoRA
Create a virtual environment (optional but recommended)

bash
Copy
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
Install dependencies

bash
Copy
pip install -r requirements.txt
🛠️ Usage
1️⃣ Fine-Tuning the Model
Run the Jupyter notebook:

bash
Copy
jupyter notebook Fine_Tune_LLM_Using_LORA.ipynb
2️⃣ Generating Synthetic Data
Execute:

bash
Copy
python answers_gen.py
3️⃣ Monitoring with W&B
Track model performance using Weights & Biases.
Login:
bash
Copy
wandb login
Start tracking:
python
Copy
import wandb
wandb.init(project="fine-tune-lora")
🤖 Hugging Face Models
The fine-tuned models are available on Hugging Face:

Model Name	Link
LoRA Fine-Tuned Model	Hugging Face Model
Synthetic Dataset	Dataset
📸 Screenshots
W&B Training Metrics	Model Predictions
📜 License
This project is licensed under the MIT License.

