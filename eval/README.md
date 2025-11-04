Evaluation & Benchmarking

Overview
- This folder contains a simple, dependency-light evaluation harness for the `robot_chroma.py` RAG app.
- It measures retrieval quality (Hit@K, MRR, Recall), optional QA quality (Exact, Contains, token F1), latency, and throughput.

Dataset Format (JSONL)
- One JSON object per line with fields:
  - `id` (optional): unique identifier (number or string).
  - `query` (required): user question.
  - `gold_sources` (optional): list of expected source filenames (use basenames like `mydoc.pdf`).
  - `gold_answers` (optional): list of acceptable reference answers (strings).
  - `tags` (optional): list of tag filters to apply to retrieved chunks.

Example: see `eval/sample_eval.jsonl`.

Usage
1) Ensure you have ingested your documents:
   - Start the app and click "Build collection"; or in Python: `import robot_chroma as rc; rc.ingest(rc.UPLOAD_DIR)`

2) Run retrieval-only evaluation:
```
python eval/eval_benchmark.py --dataset eval/sample_eval.jsonl --top-k 5
```

3) Add QA generation (requires `OPENAI_API_KEY` set, model used by the app):
```
python eval/eval_benchmark.py --dataset eval/sample_eval.jsonl --top-k 5 --generate
```

Useful Flags
- `--reranker`       Use reranker if enabled in env (falls back if unavailable)
- `--no-hybrid`      Disable BM25+semantic hybrid; use semantic-only
- `--ext .pdf`       Restrict retrieval to given extensions (repeatable)
- `--limit 50`       Evaluate only the first N samples
- `--out-dir path`   Where to write results (JSON)

Outputs
- `eval/results/results_*.json`: per-sample details (retrieved ranks, times, answer, scores)
- `eval/results/summary_*.json`: aggregate metrics (Hit@K, MRR, Recall, F1, latency, QPS)

Notes
- Filename matching for gold sources uses basenames (e.g., `file.pdf`), regardless of full path.
- QA scoring is lightweight (string/word overlap) to avoid extra deps; for stricter evaluation, expand this script as needed.

