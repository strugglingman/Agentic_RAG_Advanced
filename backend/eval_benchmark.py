import argparse
import json
import os
import math
import time
from openai import OpenAI
import robot as rc
from typing import Optional

# Initialize OpenAI client from environment variable instead of hard-coding secret.
# This prevents leaking the API key and satisfies GitHub secret scanning.
_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable; set it before running eval_benchmark.py")
openAI_Client = OpenAI(api_key=_openai_api_key)

def load_data(path: str) -> list[dict[str, any]]:
    data = []
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data.append(json.loads(line))

    return data

def answer_score(answer_bot: str, gold_answer: str) -> float:
    set_bot, set_gold = set(answer_bot.split()), set(gold_answer.split())
    if not set_bot and not set_gold:
        return 1.0
    if not set_bot or not set_gold:
        return 0.0
    
    inter = len(set_bot & set_gold)
    prec = inter / (len(set_bot) or 1)
    rec = inter / (len(set_gold) or 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def best_answer_score(norm_answer_bot: str, norm_gold_answers: list[str], score) -> float:
    return max((score(norm_answer_bot, norm_gold_answer) for norm_gold_answer in norm_gold_answers), default=0.0)

def mean(xs: list[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and not (x != x)]
    return sum(xs) / len(xs) if xs else None

def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    # deduplicated retrieved documents
    seen = set()
    retrieved = [doc for doc in retrieved if not (doc in seen or seen.add(doc))]

    for i, doc in enumerate(retrieved[:k]):
        rel = 1 if doc in gold else 0
        dcg += rel / (math.log2(i + 2))  # i+2 because log2(1) for rank 0
    # Ideal DCG
    ideal_rels = [1] * min(len(gold), k)
    idcg = sum(rel / (math.log2(i + 2)) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def eval(d: dict[str, any], k_list: list[int], mode: str):
    row_id = d.get("id", "")
    query = d.get("query", "")
    use_hybrid = mode in ["hybrid", "hybrid_rerank"]
    use_rerank = mode in ["rerank", "hybrid_rerank"]
    t0 = time.time()
    top_k = max(k_list)
    ctx_bot, err = rc.retrieve(query, top_k, use_reranker=use_rerank, use_hybrid=use_hybrid)
    time_retrieval = time.time() - t0

    if err:
        return {}, err

    system, user = rc.build_prompt(query, ctx_bot)
    t1 = time.time()
    resp = openAI_Client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.2,
            max_tokens=500
    )
    time_answer = time.time() - t1
    answer_bot = resp.choices[0].message.content.strip()

    # region Evaluation of answer
    gold_answers = d.get("gold_answers", [])
    norm_gold_answers = [" ".join(gold_answer.strip().lower().split()) for gold_answer in gold_answers if gold_answer]
    norm_answer_bot = " ".join(answer_bot.strip().lower().split())

    exact = 0
    partial = 0
    f1 = 0.0
    if any(norm_answer_bot ==  norm_gold_answer for norm_gold_answer in norm_gold_answers):
        exact = 1
    elif any(norm_answer_bot in norm_gold_answer or norm_gold_answer in norm_answer_bot for norm_gold_answer in norm_gold_answers):
        partial = 1
    else:
        f1 = best_answer_score(norm_answer_bot, norm_gold_answers, answer_score)
    #endregion

    #region Evaluation of sources
    r_metrics_k = {}
    gold_sources = d.get("gold_sources", [])
    gold_basenames = [os.path.basename(f) for f in gold_sources]
    if gold_basenames:
        hit = 0
        rr = 0.0
        recall = float("nan")

        for k in k_list:
            sources_bot = [ctx.get("source", "") for ctx in ctx_bot[:k]]
            bot_basenames = [os.path.basename(f) for f in sources_bot]
            # Hit@K
            hit = int(any(r in gold_basenames for r in bot_basenames))
            # First relevant rank in gold basenames
            first_rank = None
            for idx, r in enumerate(bot_basenames, start=1):
                if r in gold_basenames:
                    first_rank = idx
                    break
            rr = 1.0 / first_rank if first_rank else 0.0
            inter = len(set(bot_basenames) & set(gold_basenames))
            recall = inter / max(1, len(set(gold_basenames)))
            ndcg = ndcg_at_k(bot_basenames, set(gold_basenames), k)

            r_metrics_k.update({f"hit@{k}": hit, f"mrr@{k}": rr, f"recall@{k}": recall, f"ndcg@{k}": ndcg})
    #endregion

    time_total = time.time() - t0
    result = {
        "id": row_id,
        "query": query,
        #"ctx_bot": ctx_bot,
        "answer_bot": answer_bot,
        "gold_answers": gold_answers,
        "exact": exact,
        "partial": partial,
        "f1": f1,
        "time_retrieval": round(time_retrieval, 2),
        "time_answer": round(time_answer, 2),
        "time_total": round(time_total, 2)
    }
    if r_metrics_k:
        result["retrieval_metrics"] = r_metrics_k

    return result, None


def main():
    p = argparse.ArgumentParser(description="Evaluate retrieval and QA over a dataset")
    p.add_argument("--data", type=str, default="", help="Path to the dataset or using default data, with fields: query, gold_sources[], gold_answers[] (optional)")
    p.add_argument("--list-k", type=str, default="", help="List of K results to retrieve and evaluate hit@K")
    p.add_argument("--mode", choices=["sem", "hybrid", "rerank", "hybrid_rerank"], default="sem", help="Retrieval mode")
    p.add_argument("--output", type=str, default="", help="Path to save the detailed results")
    args = p.parse_args()
    path_data = args.data
    list_k = [int(k) for k in args.list_k.strip().split(",") if k.isdigit()] if args.list_k else [1, 3, 5, 10]
    mode = args.mode
    data = load_data(path_data)
    outdir = args.output if args.output else "report"

    results = []
    for d in data:
        print(f"list_k: {list_k}")
        result, err = eval(d, list_k, mode)
        if err:
            print(f"Error: {err}")
            continue

        results.append(result)

    #region Aggregate results and output
    now = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(outdir, exist_ok=True)
    file_summary = os.path.join(outdir, f"summary_{mode}_{now}.json")
    with open(file_summary, "w", encoding="utf-8") as fsum:
        # aggregation for hit@K
        retrieval_metrics = {
            **{f"hit@{k}": mean(r.get("retrieval_metrics", {}).get(f"hit@{k}", 0) for r in results) for k in list_k},
            **{f"mrr@{k}": mean(r.get("retrieval_metrics", {}).get(f"mrr@{k}", 0.0) for r in results) for k in list_k},
            **{f"recall@{k}": mean(r.get("retrieval_metrics", {}).get(f"recall@{k}", 0.0) for r in results) for k in list_k},
            **{f"ndcg@{k}": mean(r.get("retrieval_metrics", {}).get(f"ndcg@{k}", 0.0) for r in results) for k in list_k}
        }
        rm = [r.get("retrieval_metrics", {}) for r in results if r.get("retrieval_metrics", {})]
        print(rm)

        aggregated_metrics = {
            "num_queries": len(results),
            "exact": mean(r.get("exact", 0) for r in results),
            "partial": mean(r.get("partial", 0) for r in results),
            "f1": mean(r.get("f1", 0.0) for r in results),
            "time_retrieval": mean(r.get("time_retrieval", 0.0) for r in results),
            "time_answer": mean(r.get("time_answer", 0.0) for r in results),
            "time_total": mean(r.get("time_total", 0.0) for r in results),
            **retrieval_metrics
        }

        json.dump(aggregated_metrics, fsum, indent=2)
    
    file_details = os.path.join(outdir, f"details_{mode}_{now}.json")
    with open(file_details, "w", encoding="utf-8") as fdet:
        json.dump(results, fdet, indent=2)
    #endregion

if __name__ == "__main__":
    main()