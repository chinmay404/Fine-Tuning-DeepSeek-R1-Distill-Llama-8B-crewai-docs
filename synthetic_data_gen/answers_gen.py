import os
import glob
import csv , re
import yaml
import time
import json
from ollama import chat, ChatResponse
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser  # In case you wish to use it later

class RAGChainProcessor:
    def __init__(
        self,
        model_name: str = "deepseek-r1:1.5b",
        temperature: float = 0.8,
        prompt_file: str = "prompts.yaml",
        output_csv: str = "final_data.csv",
        max_retries: int = 3,
    ):
        self.output_csv = output_csv
        self.max_retries = max_retries
        self.llm = OllamaLLM(model=model_name, temperature=temperature)
        self.prompt_template = self._load_prompt(prompt_file)
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["document", "question"],
        )
        self.chain = prompt | self.llm
        self._initialize_csv()

    def _initialize_csv(self):
        """Creates the output CSV if it does not exist or is empty."""
        if not os.path.exists(self.output_csv) or os.stat(self.output_csv).st_size == 0:
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["File", "Batch_Number", "Question", "Answer"])

    def _load_prompt(self, prompt_file: str) -> str:
        """Loads the prompt template from a YAML file."""
        with open(prompt_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        prompt = data.get("answer_prompt")
        if not prompt:
            raise ValueError("The prompt 'answer_prompt' was not found in the YAML file.")
        return prompt

    def _custom_parsing(self, data: str) -> dict:
        try:
            pattern = r"<answer>\s*(.*?)\s*</endanswer>"
            match = re.search(pattern, data, re.DOTALL)
            if match:
                answer_text = match.group(1)
                return {"answer": answer_text}
            else:
                raise ValueError("Parsed response missing <answer> tags.")
        except Exception as e:
            print(f"Error parsing response: {e}")
            raise ValueError("Failed to parse chain response.")

    def _add_to_csv(self, file_name: str, batch_number, question_text: str, answer_data: dict):
        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                file_name,
                batch_number,
                question_text,
                answer_data.get("answer", ""),
            ])

    def invoke_chain(self, document: str, question: str) -> dict:
        """Invokes the chain with retry logic."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.chain.invoke({"document": document, "question": str(question)})
                answer_data = self._custom_parsing(response)
                if not answer_data:
                    raise ValueError("No answer extracted from response.")
                return answer_data
            except Exception as e:
                attempt += 1
                print(f"Error processing question (attempt {attempt}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(1)
                else:
                    print("Max retries reached. Skipping this question.")
                    return {}
        return {}

    def process_file(self, file_path: str):
        """Reads and returns the content of a single file."""
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {file_name}: {str(e)}")
            return None

    def process_directory(self, directory: str, file_pattern: str = "*.txt"):
        """Processes all files in a directory."""
        file_paths = glob.glob(os.path.join(directory, file_pattern))
        for file_path in file_paths:
            try:
                self.process_file(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue

    def process_questions(self, questions_csv: str = "questions.csv", documents_directory: str = "."):
        try:
            with open(questions_csv, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                total_questions = sum(1 for row in reader) - 1 

            # Now, process each row
            with open(questions_csv, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for index, row in enumerate(reader, start=1):
                    print(f"Processing {index} out of {total_questions} questions...")

                    file_name = row.get("File")
                    batch_number = row.get("Batch_Number")
                    question_text = row.get("Question")

                    if not file_name or not question_text:
                        print("Invalid row in CSV, skipping:", row)
                        continue

                    file_path = os.path.join(documents_directory, file_name)
                    document = self.process_file(file_path)
                    if document is None:
                        print(f"Skipping file {file_name} as it could not be read.")
                        continue

                    answer_data = self.invoke_chain(document, question_text)
                    if not answer_data:
                        print(f"No answer for question: {question_text}")
                        continue

                    self._add_to_csv(file_name, batch_number, question_text, answer_data)
        except FileNotFoundError:
            print(f"Error: The file {questions_csv} was not found.")
        except Exception as e:
            print(f"An error occurred while processing questions: {e}")



if __name__ == "__main__":
    # Directory where the document text files are stored.
    documents_directory = "G:\Fine_tune_LLM_FOR_crew_AI-\scrapper\scrapped_docs"

    # Create an instance of the RAGChainProcessor.
    processor = RAGChainProcessor(
        model_name="deepseek-r1:1.5b",
        temperature=0.9,
        prompt_file="prompts.yaml",
        output_csv="crew_ai_docs_synthetic_data.csv",
        max_retries=3,
    )

    # (Optional) Process all files in the directory if needed.
    # processor.process_directory(documents_directory, file_pattern="*.txt")

    # Process questions from questions.csv (located in the current folder),
    # using the documents stored in documents_directory.
    processor.process_questions(questions_csv="questions.csv", documents_directory=documents_directory)
