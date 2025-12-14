"""
CLI script to run Ragas evaluation.

Usage:
    python -m src.evaluation.run_eval --data eval_data.jsonl --output results/
"""

import argparse
from pathlib import Path

from .ragas_evaluator import RagasEvaluator, EvalConfig
from .dataset import EvalDataset

# from .metrics import RAGMetrics, MetricType


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on RAG pipeline")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results", help="Output directory"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5.2", help="LLM for evaluation"
    )
    parser.add_argument(
        "--metrics", type=str, default="all", help="Comma-separated metrics or 'all'"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = EvalConfig(llm_model=args.model)
    evaluator = RagasEvaluator(config)
    dataset = EvalDataset()
    # metrics = RAGMetrics()

    dataset.load_from_jsonl(args.data)
    evaluator.setup_metrics()
    results = evaluator.evaluate_dataset(dataset)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "metrics_summary.json"
    evaluator.export_results(results, str(file_path))


if __name__ == "__main__":
    main()
