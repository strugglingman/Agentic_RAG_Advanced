"""
RAGAS Evaluation Runner - Uses actual RAG pipeline to generate answers and contexts.

This script:
1. Loads test data from JSONL (questions + ground_truth)
2. Runs each question through the RAG pipeline (retrieve + generate)
3. Fills answer and contexts fields
4. Runs RAGAS evaluation on the filled data
5. Exports results

Usage:
    python -m src.evaluation.run_ragas_eval --data eval_data/test.jsonl --output eval_results/

    # With options:
    python -m src.evaluation.run_ragas_eval --data eval_data/test.jsonl --top-k 5 --hybrid --rerank
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from src.config.settings import Config
from src.services.vector_db import VectorDB
from src.services.retrieval import retrieve
from src.services.retrieval_decomposition import retrieve_with_decomposition
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.dataset import EvalDataset, EvalRow


def optimize_query_for_retrieval(query: str, client: OpenAI) -> str:
    """
    Optimize query for retrieval by expanding abbreviations and removing filler.

    This mirrors the logic in langgraph_nodes._optimize_step_query() for "retrieve".
    Used to test whether query expansion improves RAGAS metrics.

    Args:
        query: Original user query
        client: OpenAI client

    Returns:
        Optimized query string
    """
    if not query or not client:
        return query

    try:
        prompt = f"""Optimize this query for document retrieval.

Input: {query}

Tasks (do ALL):
1. EXPAND common abbreviations and acronyms:
   - PTO → PTO paid time off
   - Q1/Q2/Q3/Q4 → Q1 first quarter (keep both forms)
   - YoY → YoY year over year
   - HR → HR human resources
   - ROI → ROI return on investment
   - KPI → KPI key performance indicator
   - OKR → OKR objectives key results
   - Rev → revenue
   - FY → FY fiscal year

2. REMOVE filler phrases (if present):
   - "Can you help me find..."
   - "I would like to know..."
   - "Search for information about..."
   - "Tell me about..."

3. KEEP important terms:
   - Names, dates, numbers, percentages
   - Domain-specific keywords
   - Key entities

Output a clear query (under 120 chars) with abbreviations expanded.
Output ONLY the optimized query, nothing else."""

        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )

        optimized = response.choices[0].message.content.strip().strip('"').strip("'")
        if len(optimized) >= 5:
            return optimized
        return query

    except Exception as e:
        print(f"[QUERY_OPT] Failed: {e}")
        return query


def load_test_data(path: str) -> list[dict]:
    """Load test data from JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_answer(client: OpenAI, query: str, contexts: list[dict]) -> str:
    """
    Generate answer using OpenAI with retrieved contexts.
    Mimics the agent's answer generation logic.
    Groups contexts by sub-query if decomposition was used.
    """
    # Check if contexts have sub_query labels (indicates decomposition was used)
    sub_queries = set(c.get("sub_query") for c in contexts if c.get("sub_query"))
    has_decomposition = len(sub_queries) > 1

    if has_decomposition:
        # Group contexts by sub-query for clearer presentation to LLM
        context_str = f"Original Query: \"{query}\"\n"
        context_str += f"Decomposed into {len(sub_queries)} sub-queries for better retrieval:\n\n"

        context_idx = 1
        for sq in sub_queries:
            sq_contexts = [c for c in contexts if c.get("sub_query") == sq]
            context_str += f"=== Sub-query: \"{sq}\" ({len(sq_contexts)} results) ===\n\n"

            for c in sq_contexts:
                context_str += (
                    f"Context {context_idx} (Source: {c.get('source', 'unknown')}"
                    + (f", Page: {c['page']}" if c.get("page", 0) > 0 else "")
                    + f"):\n{c.get('chunk', '')}\n\n"
                )
                context_idx += 1
    else:
        # Original flat format (no decomposition or single sub-query)
        context_str = "\n\n".join(
            f"Context {i+1} (Source: {c.get('source', 'unknown')}"
            + (f", Page: {c['page']}" if c.get("page", 0) > 0 else "")
            + f"):\n{c.get('chunk', '')}"
            for i, c in enumerate(contexts)
        )

    # Build system prompt based on whether decomposition was used
    if has_decomposition:
        system_prompt = (
            "You are a helpful assistant answering questions based on provided context. "
            "The original query was decomposed into sub-queries for better retrieval. "
            "Contexts are grouped by sub-query. Use information from ALL groups to fully answer the ORIGINAL query. "
            "When comparing entities, ensure you include data from each relevant sub-query group. "
            "Use ONLY the information from the contexts to answer. "
            "If the context doesn't contain enough information, say so."
        )
    else:
        system_prompt = (
            "You are a helpful assistant answering questions based on provided context. "
            "Use ONLY the information from the contexts to answer. "
            "If the context doesn't contain enough information, say so."
        )

    user_prompt = f"""Based on the following contexts, answer the question.

{context_str}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=2000,
    )

    return response.choices[0].message.content.strip()


def run_rag_pipeline(
    client: OpenAI,
    vector_db: VectorDB,
    question: str,
    dept_id: str,
    user_id: str,
    top_k: int = 8,
    use_hybrid: bool = False,
    use_reranker: bool = False,
    optimize_query: bool = False,
    use_decomposition: bool = False,
) -> tuple[str, list[str], list[dict]]:
    """
    Run the full RAG pipeline: retrieve + generate.

    Args:
        optimize_query: If True, expand abbreviations before retrieval
        use_decomposition: If True, use query decomposition for multi-hop queries

    Returns:
        Tuple of (answer, contexts_as_strings, raw_contexts)
    """
    # Optionally optimize query (expand abbreviations, remove filler)
    retrieval_query = question
    if optimize_query:
        retrieval_query = optimize_query_for_retrieval(question, client)
        if retrieval_query != question:
            print(f"         Optimized: {retrieval_query[:60]}...")

    # Retrieve contexts - with or without decomposition
    if use_decomposition:
        ctx_list, error = retrieve_with_decomposition(
            vector_db=vector_db,
            openai_client=client,
            query=retrieval_query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )
    else:
        ctx_list, error = retrieve(
            vector_db=vector_db,
            query=retrieval_query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

    if error or not ctx_list:
        return "No relevant documents found.", [], []

    # Generate answer
    answer = generate_answer(client, question, ctx_list)

    # Extract context strings for RAGAS (just the chunk text)
    context_strings = [c.get("chunk", "") for c in ctx_list]

    return answer, context_strings, ctx_list


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation using actual RAG pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="src/evaluation/eval_data/test.jsonl",
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/evaluation/eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid search (BM25 + semantic)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use reranker",
    )
    parser.add_argument(
        "--dept-id",
        type=str,
        default="EVAL|test|evaluation",
        help="Department ID for retrieval filtering",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="eval@test.com",
        help="User ID for retrieval filtering",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions to evaluate (0 = all)",
    )
    parser.add_argument(
        "--optimize-query",
        action="store_true",
        help="Optimize queries before retrieval (expand abbreviations, remove filler)",
    )
    parser.add_argument(
        "--decomposition",
        action="store_true",
        help="Use query decomposition for multi-hop queries (e.g., 'Compare A and B')",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize clients
    print("Initializing OpenAI client and VectorDB...")
    client = OpenAI(api_key=Config.OPENAI_KEY)
    vector_db = VectorDB(path="chroma_db", embedding_provider="openai")

    # Load test data
    print(f"Loading test data from {args.data}...")
    test_data = load_test_data(args.data)
    print(f"Loaded {len(test_data)} test cases")

    if args.limit > 0:
        test_data = test_data[: args.limit]
        print(f"Limited to {len(test_data)} test cases")

    # Run RAG pipeline for each question
    print("\nRunning RAG pipeline...")
    print(f"  Top-K: {args.top_k}")
    print(f"  Hybrid: {args.hybrid}")
    print(f"  Rerank: {args.rerank}")
    print(f"  Optimize Query: {args.optimize_query}")
    print(f"  Decomposition: {args.decomposition}")
    print("-" * 60)

    eval_dataset = EvalDataset()
    filled_data = []

    for i, item in enumerate(test_data, 1):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        print(f"[{i}/{len(test_data)}] Processing: {question[:50]}...")

        # Run RAG pipeline
        answer, contexts, raw_contexts = run_rag_pipeline(
            client=client,
            vector_db=vector_db,
            question=question,
            dept_id=args.dept_id,
            user_id=args.user_id,
            top_k=args.top_k,
            use_hybrid=args.hybrid,
            use_reranker=args.rerank,
            optimize_query=args.optimize_query,
            use_decomposition=args.decomposition,
        )

        print(f"         Answer: {answer[:80]}...")
        print(f"         Contexts: {len(contexts)} chunks")

        # Show sub-query distribution if decomposition was used
        if args.decomposition and raw_contexts:
            sub_query_counts = {}
            for ctx in raw_contexts:
                sq = ctx.get("sub_query", "original")
                sub_query_counts[sq] = sub_query_counts.get(sq, 0) + 1
            if len(sub_query_counts) > 1:
                print(f"         Sub-queries: {sub_query_counts}")

        # Add to eval dataset
        eval_row = EvalRow(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        eval_dataset.add_row(eval_row)

        # Also save filled data for reference
        filled_data.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    # Save filled data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filled_path = os.path.join(args.output, f"filled_data_{timestamp}.jsonl")
    with open(filled_path, "w", encoding="utf-8") as f:
        for item in filled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved filled data to: {filled_path}")

    # Run RAGAS evaluation
    print("\nRunning RAGAS evaluation...")
    evaluator = RagasEvaluator()

    try:
        results = evaluator.evaluate_dataset(eval_dataset)

        # Export results with runtime configuration
        results_path = os.path.join(args.output, f"ragas_results_{timestamp}.json")
        runtime_config = {
            "top_k": args.top_k,
            "hybrid_search": args.hybrid,
            "reranker": args.rerank,
            "optimize_query": args.optimize_query,
            "decomposition": args.decomposition,
            "dept_id": args.dept_id,
            "user_id": args.user_id,
        }
        evaluator.export_results(results, results_path, runtime_config=runtime_config)

        # Print summary
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 60)

        df = results.to_pandas()
        metric_cols = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        for metric in metric_cols:
            if metric in df.columns:
                avg_score = df[metric].mean()
                print(f"  {metric}: {avg_score:.4f}")

        print("=" * 60)
        print(f"Results saved to: {results_path}")

    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
