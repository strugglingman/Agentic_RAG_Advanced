"""
Dataset utilities for Ragas evaluation.

Handles loading/creating evaluation datasets in Ragas-compatible format.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from datasets import Dataset as HFDataset
import pandas as pd
from ragas.testset import TestsetGenerator
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config.settings import Config


@dataclass
class EvalRow:
    """Single evaluation row."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    metadata: dict = field(default_factory=dict)


class EvalDataset:
    """Manages evaluation datasets for Ragas."""

    def __init__(self):
        self.rows: list[EvalRow] = []

    def add_row(self, row: EvalRow):
        """Add a single evaluation row."""
        self.rows.append(row)

    def load_from_jsonl(self, path: str | Path):
        # Load rows from JSONL file
        if not Path(path).is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if Path(path).suffix != ".jsonl":
            raise ValueError("File must be a .jsonl file")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                one_dict = json.loads(line)
                row = EvalRow(**one_dict)
                self.add_row(row)

    def load_from_csv(self, path: str | Path):
        """Load rows from CSV file."""
        if not Path(path).is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if Path(path).suffix != ".csv":
            raise ValueError("File must be a .csv file")

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            eval_row = EvalRow(
                question=row["question"],
                answer=row["answer"],
                contexts=json.loads(row["contexts"]),
                ground_truth=row["ground_truth"],
                metadata=json.loads(row["metadata"]) if "metadata" in row else {},
            )
            self.add_row(eval_row)

    def to_ragas_dataset(self) -> HFDataset:
        """Convert to Hugging Face Dataset for Ragas."""
        data = {
            "question": [row.question for row in self.rows],
            "answer": [row.answer for row in self.rows],
            "contexts": [row.contexts for row in self.rows],
            "ground_truth": [row.ground_truth for row in self.rows],
        }

        return HFDataset.from_dict(data)

    def generate_from_documents(self, documents: list[str], num_rows: int = 10):
        """
        Use Ragas TestsetGenerator to create synthetic evaluation data.

        Args:
            documents: Source documents (strings) to generate questions from
            num_rows: Number of QA pairs to generate
        """
        # Convert strings to LangChain Document objects
        docs = [Document(page_content=text) for text in documents]

        # Create generator with LLM and embeddings
        generator = TestsetGenerator.from_langchain(
            llm=ChatOpenAI(model=Config.OPENAI_MODEL, api_key=Config.OPENAI_KEY),
            embedding_model=OpenAIEmbeddings(
                model=Config.OPENAI_EMBEDDING_MODEL, api_key=Config.OPENAI_KEY
            ),
        )

        # Generate test set
        testset = generator.generate_with_langchain_docs(docs, testset_size=num_rows)

        # Convert to EvalRows
        for item in testset.to_pandas().to_dict("records"):
            row = EvalRow(
                question=item["question"],
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth", ""),
            )
            self.add_row(row)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]
