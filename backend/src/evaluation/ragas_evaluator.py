"""
Ragas Evaluator - Offline evaluation for RAG pipelines.

Metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved docs relevant and well-ordered?
- Context Recall: Does context contain info needed for ground truth?
"""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path
from src.config.settings import Config

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# TODO: Import LangChain wrappers for OpenAI
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class EvalConfig:
    """Configuration for Ragas evaluation."""
    llm_model: str = Config.OPENAI_MODEL
    embedding_model: str = Config.EMBEDDING_MODEL_NAME
    batch_size: int = 10


class RagasEvaluator:
    """Wrapper for Ragas evaluation framework."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self._metrics = None
        self._llm = None
        self._embeddings = None

    def setup_metrics(self):
        """
        Initialize Ragas metrics with configured LLM.

        TODO Steps:
        1. Import: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        2. Create LLM wrapper:
           self._llm = ChatOpenAI(model=self.config.llm_model)
        3. Create embeddings wrapper:
           self._embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        4. Store metrics list (you can get from RAGMetrics.get_ragas_metrics() or use defaults)
           self._metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        NOTE: Ragas metrics will automatically use these LangChain wrappers during evaluation
        """
        raise NotImplementedError

    def evaluate_dataset(self, dataset):
        """
        Run Ragas evaluation on a dataset.

        Args:
            dataset: HuggingFace Dataset or EvalDataset.to_ragas_dataset()
                    Must have columns: ['question', 'answer', 'contexts', 'ground_truth']

        Returns:
            dict with metric scores (faithfulness, answer_relevancy, etc.)

        TODO Steps:
        1. Call self.setup_metrics() if not already done
        2. Convert dataset to HuggingFace Dataset format if needed
           (Use dataset.to_ragas_dataset() if it's an EvalDataset)
        3. Call ragas.evaluate():
           result = evaluate(
               dataset=hf_dataset,
               metrics=self._metrics,
               llm=self._llm,
               embeddings=self._embeddings
           )
        4. Return result as dict (result.to_pandas().to_dict() or similar)

        IMPORTANT: Dataset must have exact column names Ragas expects!
        """
        raise NotImplementedError

    def evaluate_single(self, question: str, answer: str, contexts: list[str], ground_truth: str):
        """
        Evaluate a single RAG response.

        Args:
            question: User question
            answer: Generated answer
            contexts: List of context strings used
            ground_truth: Reference/expected answer

        Returns:
            dict with metric scores for this single sample

        TODO Steps:
        1. Import: from datasets import Dataset
        2. Create single-row HuggingFace Dataset:
           data = {
               'question': [question],
               'answer': [answer],
               'contexts': [contexts],  # Must be list of lists!
               'ground_truth': [ground_truth]
           }
           single_dataset = Dataset.from_dict(data)
        3. Call self.evaluate_dataset(single_dataset)
        4. Return the scores dict

        NOTE: contexts must be [["doc1", "doc2"]] not ["doc1", "doc2"]!
        """
        raise NotImplementedError

    def export_results(self, results, output_path: str):
        """
        Export evaluation results to file.

        Args:
            results: Dict or DataFrame with evaluation results
            output_path: Path to save results (supports .json, .csv)

        TODO Steps:
        1. Create output directory if it doesn't exist: Path(output_path).parent.mkdir()
        2. Check file extension:
           - .json: Save as JSON with json.dump()
           - .csv: Convert to pandas DataFrame and use df.to_csv()
        3. Include metadata: timestamp, config settings, metric averages
        4. Print confirmation message

        Example JSON structure:
        {
            "config": {"llm_model": "gpt-4o-mini", ...},
            "timestamp": "2024-01-01T12:00:00",
            "results": [...],
            "summary": {"avg_faithfulness": 0.85, ...}
        }
        """
        raise NotImplementedError
