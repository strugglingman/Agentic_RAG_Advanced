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
from datetime import datetime
import json
from pathlib import Path
from src.config.settings import Config
from src.evaluation.metrics import RAGMetrics
from ragas import evaluate
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset as HFDataset
from langsmith import Client

# LangSmith integration works automatically via environment variables:
# LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
# No special import needed - ragas.evaluate() auto-logs to LangSmith


@dataclass
class EvalConfig:
    """Configuration for Ragas evaluation."""

    llm_model: str = Config.OPENAI_MODEL
    embedding_model: str = Config.OPENAI_EMBEDDING_MODEL
    batch_size: int = 10


class RagasEvaluator:
    """Wrapper for Ragas evaluation framework."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self._metrics: list = None
        self._llm = None
        self._embeddings = None
        self._langsmith_client = None
        if Config.LANGCHAIN_API_KEY:
            self._langsmith_client = Client(api_key=Config.LANGCHAIN_API_KEY)

    def setup_metrics(self):
        """Initialize Ragas metrics with configured LLM and embeddings."""
        self._llm = ChatOpenAI(
            model=self.config.llm_model, temperature=0, api_key=Config.OPENAI_KEY
        )
        self._embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model, api_key=Config.OPENAI_KEY
        )
        self._metrics = RAGMetrics().get_ragas_metrics()

    def evaluate_dataset(self, dataset):
        """
        Run Ragas evaluation on a dataset.

        Args:
            dataset: HuggingFace Dataset or EvalDataset.to_ragas_dataset()
                    Must have columns: ['question', 'answer', 'contexts', 'ground_truth']

        Returns:
            dict with metric scores (faithfulness, answer_relevancy, etc.)
        """
        if self._metrics is None or self._llm is None or self._embeddings is None:
            self.setup_metrics()
        # If dataset is not a HuggingFace Dataset, convert it
        try:
            if hasattr(dataset, "to_ragas_dataset"):
                hf_dataset = dataset.to_ragas_dataset()
            else:
                hf_dataset = dataset

            result = evaluate(
                dataset=hf_dataset,
                metrics=self._metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            return result
        except Exception as e:
            raise ValueError(f"Error during evaluation: {str(e)}")

    def evaluate_single(
        self, question: str, answer: str, contexts: list[str], ground_truth: str
    ):
        """
        Evaluate a single RAG response.

        Args:
            question: User question
            answer: Generated answer
            contexts: List of context strings used
            ground_truth: Reference/expected answer

        Returns:
            dict with metric scores for this single sample
        """
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        single_dataset = HFDataset.from_dict(data)

        return self.evaluate_dataset(single_dataset)

    def export_results(self, results, output_path: str):
        """
        Export evaluation results to file.

        Args:
            results: Dict or DataFrame with evaluation results
            output_path: Path to save results (supports .json, .csv)
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if hasattr(results, "to_pandas"):
                df = results.to_pandas()
                results_list = df.to_dict(orient="records")
                results_dict = df.to_dict()
            else:
                raise ValueError(
                    "Results must be a pandas DataFrame or have to_pandas() method"
                )
            data = {
                "config": {
                    "llm_model": self.config.llm_model,
                    "embedding_model": self.config.embedding_model,
                    "batch_size": self.config.batch_size,
                },
                "results": results_dict,
                "timestamp": datetime.now().isoformat(),
                "averages": (
                    RAGMetrics().aggregate_results(results_list)
                    if isinstance(results_list, list)
                    else results_list
                ),
            }
            if output_path.endswith(".json"):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            elif output_path.endswith(".csv"):
                df.to_csv(output_path, index=False)

            print(f"Results exported to {output_path}")
        except Exception as e:
            raise ValueError(f"Error exporting results: {str(e)}")

    def push_to_langsmith(self, results, dataset_name: str, description: str = ""):
        """
        Push existing evaluation results to LangSmith as a dataset.

        Args:
            results: Evaluation results (DataFrame or dict)
            dataset_name: Name for the dataset in LangSmith
            description: Optional description for the dataset

        Example:
            results = evaluator.evaluate_dataset(dataset)
            evaluator.push_to_langsmith(
                results,
                dataset_name="rag-eval-results-2024-12",
                description="December evaluation results"
            )
        """
        if not self._langsmith_client:
            raise ValueError(
                "LangSmith client not initialized. Set LANGSMITH_API_KEY in .env"
            )

        # Convert results to DataFrame if needed
        if hasattr(results, "to_pandas"):
            df = results.to_pandas()
        elif isinstance(results, dict):
            df = pd.DataFrame([results])
        else:
            df = results

        # Create dataset in LangSmith
        print(f"Uploading {len(df)} results to LangSmith dataset: {dataset_name}")

        # Convert DataFrame to LangSmith examples format
        examples = []
        for _, row in df.iterrows():
            example = {
                "inputs": {
                    "question": row.get("question", ""),
                    "contexts": row.get("contexts", []),
                },
                "outputs": {
                    "answer": row.get("answer", ""),
                },
                "metadata": {
                    "faithfulness": row.get("faithfulness", None),
                    "answer_relevancy": row.get("answer_relevancy", None),
                    "context_precision": row.get("context_precision", None),
                    "context_recall": row.get("context_recall", None),
                },
            }
            examples.append(example)

        # Create or update dataset
        self._langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description=description
            or f"Ragas evaluation results - {datetime.now().isoformat()}",
        )

        for example in examples:
            self._langsmith_client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
                dataset_name=dataset_name,
            )

        print(f"✅ Uploaded {len(examples)} examples to LangSmith")
        print(f"📊 View dataset at: https://smith.langchain.com/datasets")
