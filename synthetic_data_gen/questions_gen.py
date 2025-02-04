import os
import re
import glob
import csv
import yaml
import time
from ollama import chat, ChatResponse
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser  # (if needed)


class RAGChainProcessor:
    def __init__(
        self,
        model_name: str = "deepseek-r1:1.5b",
        temperature: float = 0.8,
        prompt_file: str = "prompts.yaml",
        batch_size: int = 200,
        output_csv: str = "questions.csv",
        checkpoint_file: str = "checkpoint.csv",
        max_retries: int = 3,
    ):
        self.batch_size = batch_size
        self.output_csv = output_csv
        self.checkpoint_file = checkpoint_file
        self.max_retries = max_retries
        self.llm = OllamaLLM(model=model_name, temperature=temperature)
        self.prompt_template = self._load_prompt(prompt_file)
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["document", "number_of_questions"],
        )
        self.chain = prompt | self.llm
        self._initialize_csv()
        self._initialize_checkpoint()

    def _initialize_csv(self):
        """Creates the output CSV if it does not exist."""
        if not os.path.exists(self.output_csv) or os.stat(self.output_csv).st_size == 0:
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["File", "Batch_Number", "Question"])

    def _initialize_checkpoint(self):
        """Creates the checkpoint file if it does not exist."""
        if not os.path.exists(self.checkpoint_file) or os.stat(self.checkpoint_file).st_size == 0:
            with open(self.checkpoint_file, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["File", "Batch_Number"])  # Track processed batches

    def _load_prompt(self, prompt_file: str) -> str:
        """Loads the prompt template from a YAML file."""
        with open(prompt_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        prompt = data.get("question_prompt")
        if not prompt:
            raise ValueError("The prompt 'question_prompt' was not found in the YAML file.")
        return prompt

    def _custom_parsing(self, data: str) -> list:
        """Extracts questions from the LLM response using regex."""
        data = data.strip()
        pattern = r'"Question \d+":\s*"([^"]+)"'
        questions = re.findall(pattern, data)
        if questions:
            for question in questions:
                print(f"Extracted Question: {question}")
        else:
            print("No questions extracted from the response.")
        return questions

    def _append_to_csv(self, file_name: str, batch_number: int, questions: list):
        """Appends extracted questions to the output CSV."""
        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for question in questions:
                writer.writerow([file_name, batch_number, question])

    def _update_checkpoint(self, file_name: str, batch_number: int):
        """Logs processed batches to a checkpoint file."""
        with open(self.checkpoint_file, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([file_name, batch_number])

    def _get_completed_batches(self):
        """Loads processed batches from the checkpoint file to avoid re-processing."""
        completed_batches = set()
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, mode="r", newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                for row in reader:
                    completed_batches.add((row[0], int(row[1])))  # (file_name, batch_number)
        return completed_batches

    def invoke_chain(self, document: str, num_questions: int = 5) -> list:
        """Invokes the chain with retry logic."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.chain.invoke(
                    {"document": document, "number_of_questions": str(num_questions)}
                )
                questions = self._custom_parsing(response)
                if not questions:
                    raise ValueError("No questions extracted from response.")
                return questions
            except Exception as e:
                attempt += 1
                print(f"Error processing batch (attempt {attempt}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(1)
                else:
                    print("Max retries reached. Skipping this batch.")
                    return []
        return []

    def process_file(self, file_path: str, num_questions: int = 5, from_checkpoint: bool = False):
        """Processes a single file, with optional checkpoint resumption."""
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file {file_name}: {str(e)}")
            return

        completed_batches = self._get_completed_batches() if from_checkpoint else set()

        batch_number = 1
        for i in range(0, len(content), self.batch_size):
            if (file_name, batch_number) in completed_batches:
                print(f"Skipping already processed batch {batch_number} of file {file_name}")
            else:
                batch = content[i: i + self.batch_size]
                questions = self.invoke_chain(batch, num_questions=num_questions)
                if questions:
                    self._append_to_csv(file_name, batch_number, questions)
                    self._update_checkpoint(file_name, batch_number)
                else:
                    print(f"No valid questions found in batch {batch_number} of file {file_name}.")
            batch_number += 1

    def process_directory(self, directory: str, file_pattern: str = "*.txt", num_questions: int = 5, from_checkpoint: bool = False):
        """Processes all files in a directory, with optional checkpoint resumption."""
        file_paths = glob.glob(os.path.join(directory, file_pattern))
        for file_path in file_paths:
            try:
                self.process_file(file_path, num_questions=num_questions, from_checkpoint=from_checkpoint)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue


if __name__ == "__main__":
    directory_path = "G:/Fine_Tune_LLM/scrapper/scrapped_docs/"
    processor = RAGChainProcessor(
        model_name="deepseek-r1:1.5b",
        temperature=1.0,
        prompt_file="prompts.yaml",
        batch_size=200,
        output_csv="questions.csv",
        checkpoint_file="checkpoint.csv",
        max_retries=3,
    )
    processor.process_directory(directory_path, file_pattern="*.txt", num_questions=5, from_checkpoint=True)
