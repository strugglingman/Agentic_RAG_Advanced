"""
Convert HuggingFace RAG datasets to evaluation format.

Usage:
    python -m src.evaluation.convert_hf_dataset --dataset financial --output eval_data/
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def convert_financial_dataset(output_dir: Path, num_samples: int = 100):
    """
    Convert philschmid/finanical-rag-embedding-dataset to JSONL format.

    This dataset has question-context pairs from financial documents (NVIDIA, etc.)
    """
    print("Loading philschmid/finanical-rag-embedding-dataset...")
    ds = load_dataset("philschmid/finanical-rag-embedding-dataset")

    # Get unique contexts for document corpus
    contexts = set()
    qa_pairs = []

    for item in ds["train"]:
        question = item["question"]
        context = item["context"]

        if context and len(context) > 50:  # Filter very short contexts
            contexts.add(context)
            qa_pairs.append({
                "question": question,
                "context": context
            })

    print(f"Found {len(contexts)} unique contexts")
    print(f"Found {len(qa_pairs)} QA pairs")

    # Sample QA pairs for evaluation
    import random
    random.seed(42)
    sampled_qa = random.sample(qa_pairs, min(num_samples, len(qa_pairs)))

    # Create JSONL for evaluation
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write test data JSONL
    jsonl_path = output_dir / "financial_test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sampled_qa:
            record = {
                "question": item["question"],
                "answer": "",  # To be filled by RAG
                "contexts": [],  # To be filled by retrieval
                "ground_truth": item["context"]  # Use context as ground truth
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled_qa)} QA pairs to {jsonl_path}")

    # Write document corpus as text files (for ingestion)
    docs_dir = output_dir / "financial_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Group contexts into larger documents (simulate company docs)
    context_list = list(contexts)
    docs_per_file = 50  # Group contexts into files

    for i in range(0, len(context_list), docs_per_file):
        chunk = context_list[i:i + docs_per_file]
        doc_content = "\n\n".join(chunk)
        doc_path = docs_dir / f"financial_doc_{i // docs_per_file + 1}.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_content)

    num_docs = (len(context_list) + docs_per_file - 1) // docs_per_file
    print(f"Wrote {num_docs} document files to {docs_dir}")

    return jsonl_path, docs_dir


def convert_wikipedia_dataset(output_dir: Path, num_samples: int = 100):
    """
    Convert rag-datasets/rag-mini-wikipedia to JSONL format.

    This dataset has passages and simple QA pairs.
    """
    print("Loading rag-datasets/rag-mini-wikipedia...")

    # Load both configs
    text_corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    qa_data = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

    passages = {item["id"]: item["passage"] for item in text_corpus["passages"]}

    print(f"Found {len(passages)} passages")
    print(f"Found {len(qa_data['test'])} QA pairs")

    # Sample QA pairs
    import random
    random.seed(42)

    qa_list = list(qa_data["test"])
    sampled_qa = random.sample(qa_list, min(num_samples, len(qa_list)))

    # Create JSONL for evaluation
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "wikipedia_test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sampled_qa:
            record = {
                "question": item["question"],
                "answer": "",
                "contexts": [],
                "ground_truth": item["answer"]  # Note: These are short answers
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled_qa)} QA pairs to {jsonl_path}")

    # Write passages as documents
    docs_dir = output_dir / "wikipedia_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    passage_list = list(passages.values())
    docs_per_file = 100

    for i in range(0, len(passage_list), docs_per_file):
        chunk = passage_list[i:i + docs_per_file]
        doc_content = "\n\n".join(chunk)
        doc_path = docs_dir / f"wikipedia_doc_{i // docs_per_file + 1}.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_content)

    num_docs = (len(passage_list) + docs_per_file - 1) // docs_per_file
    print(f"Wrote {num_docs} document files to {docs_dir}")

    return jsonl_path, docs_dir


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace datasets for RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["financial", "wikipedia", "both"],
        default="financial",
        help="Dataset to convert"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/evaluation/eval_data",
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of QA samples to extract"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.dataset in ["financial", "both"]:
        jsonl_path, docs_dir = convert_financial_dataset(output_dir, args.num_samples)
        print(f"\n=== Financial Dataset ===")
        print(f"Test data: {jsonl_path}")
        print(f"Documents: {docs_dir}")
        print(f"\nTo use:")
        print(f"1. Upload documents from {docs_dir} to your RAG system")
        print(f"2. Run: python -m src.evaluation.run_ragas_eval --data {jsonl_path}")

    if args.dataset in ["wikipedia", "both"]:
        jsonl_path, docs_dir = convert_wikipedia_dataset(output_dir, args.num_samples)
        print(f"\n=== Wikipedia Dataset ===")
        print(f"Test data: {jsonl_path}")
        print(f"Documents: {docs_dir}")
        print(f"\nTo use:")
        print(f"1. Upload documents from {docs_dir} to your RAG system")
        print(f"2. Run: python -m src.evaluation.run_ragas_eval --data {jsonl_path}")


if __name__ == "__main__":
    main()
