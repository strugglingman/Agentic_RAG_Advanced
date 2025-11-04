# RAG Chatbot

A full‑stack Retrieval-Augmented Generation (RAG) chatbot combining a **Next.js (App Router)** frontend with a **Python Flask** backend and **ChromaDB** vector store. It ingests PDF/text documents, chunks them with page awareness, retrieves semantically and hybrid-ranked contexts, and streams grounded answers with source citations.

## Table of Contents
1. Overview
2. Features
3. Architecture
4. Folder Structure
5. Prerequisites
6. Environment Variables
7. Quick Start
8. Ingestion Workflow
9. Retrieval Logic & Scoring
10. Streaming Protocol
11. Evaluation Benchmark
12. Development Tips
13. Security Practices
14. Roadmap
15. License / Contributing

## 1. Overview
This project demonstrates a pragmatic RAG pipeline:
- Page-aware PDF ingestion
- Overlapping sentence chunks to preserve context across boundaries
- Hybrid retrieval (semantic + BM25 fusion) with optional reranking
- Source + page citations embedded in the final answer
- Streaming responses to the UI plus a structured context payload

## 2. Features
- ✅ Next.js App Router with protected route group `(protected)` for authenticated pages (`/chat`, `/upload`)
- ✅ User authentication via NextAuth (JWT strategy)
- ✅ Flask backend `/chat` streaming endpoint
- ✅ Page number included in chunk IDs to avoid duplicate hash collisions
- ✅ Automatic context injection & citation formatting
- ✅ Hybrid retrieval thresholds (semantic similarity & fused score gating)
- ✅ Duplicate chunk detection logging
- ✅ Evaluation script for retrieval + answer quality (`eval_benchmark.py`)
- 🚧 Selective context display (metadata filtering) – planned
- 🚧 Skip trivial watermark-only chunks – planned

## 3. Architecture
```
+--------------------+        stream        +-----------------------+
|  Next.js Frontend  | <------------------ |  Flask API Backend    |
|  - /chat UI        |  answer tokens +    |  /chat (LLM + RAG)    |
|  - Auth (NextAuth) |  __CONTEXT__ JSON   |  Retrieval + Prompt   |
+---------+----------+                     +----------+------------+
          |  fetch() streaming                        |
          v                                           v
+--------------------+                     +-----------------------+
|  Browser Client    |                     | ChromaDB Vector Store |
|  React State       |                     | Page-aware chunks     |
+--------------------+                     +-----------------------+
                                                 ^
                                                 |
                                          Ingestion Pipeline
```

## 4. Folder Structure
```
backend/        Flask app, retrieval, ingestion, evaluation
frontend/       Next.js application (App Router)
materials/      Example PDF documents
eval/           Sample evaluation data + README
report/         Generated evaluation outputs (JSON)
```
(See source tree for full detail.)

## 5. Prerequisites
- Node.js 18+
- Python 3.10+
- Docker (optional but recommended)
- OpenAI API key

## 6. Environment Variables
Create a `.env` (NOT committed):
```
OPENAI_API_KEY=sk-...your key...
NEXTAUTH_SECRET=your_generated_secret
NEXTAUTH_URL=http://localhost:3000
DATABASE_URL=postgres://... (if using Prisma/Postgres)
```
Load for Python (backend) using `python-dotenv` or set in shell:
Windows PowerShell:
```powershell
$Env:OPENAI_API_KEY="sk-..."
```

## 7. Quick Start
### Option A: Docker Compose
```powershell
# From repository root
docker compose up --build
```
Frontend: http://localhost:3000  
Backend: http://localhost:5001

### Option B: Manual Dev
Backend:
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$Env:OPENAI_API_KEY="sk-..."
python app.py
```
Frontend:
```powershell
cd frontend
npm install
npm run dev
```

## 8. Ingestion Workflow (High Level)
1. Upload / place PDF or text files in target directory
2. Backend reads PDFs page-by-page (preserving page numbers)
3. Chunking: Sentence grouping + overlap to reduce boundary loss
4. Chunk IDs: `dept|user(optional)|filename|p{page_num}|{md5(chunk)}`
5. Store embeddings + metadata in ChromaDB
6. Duplicate detection: If a chunk hash repeats (e.g., watermark page), page number distinguishes ID

## 9. Retrieval Logic & Scoring
- Semantic similarity via SentenceTransformers MiniLM (normalized embeddings)
- Optional BM25 candidate pass; fusion with `FUSE_ALPHA`
- Threshold gating: Minimum semantic similarity & fused score
- Optional reranking via CrossEncoder (when enabled)
- Returned contexts include: source filename, page number, score(s)

## 10. Streaming Protocol
Backend `/chat` endpoint streams:
- Assistant answer tokens progressively
- Final contexts marker:
```
__CONTEXT__:{"contexts":[{...},{...}]}  // JSON payload
```
Frontend logic:
- Accumulates answer text
- Parses final context marker to attach sources to last assistant message
- Planned: Metadata directive `[METADATA]contexts:1,3|sources:...` for selective display

## 11. Evaluation Benchmark
`backend/eval_benchmark.py` supports measuring:
- Retrieval metrics: hit@K, mrr@K, recall@K, ndcg@K
- Answer quality: exact / partial / F1
Usage:
```powershell
python backend/eval_benchmark.py --data eval/sample_eval.jsonl --list-k 1,3,5,10 --mode hybrid
```
Outputs summary + detailed JSON into `report/`.

## 12. Development Tips
- Use functional React state updates for streaming text (`setMessages(prev => [...prev, newMsg])`)
- Use a ref mirror for latest state during async streaming (`messagesRef.current`)
- Keep chunks moderately sized (tunable: `CHUNK_SIZE`, `CHUNK_OVERLAP`)
- Adjust retrieval thresholds to control hallucination vs recall

## 13. Security Practices
- Never commit API keys (secret scanning blocks pushes)
- Store secrets in `.env` and add `.env*` to `.gitignore`
- Consider GitHub branch protection (require PR + CI pass)
- Rotate keys if accidentally exposed; purge from git history

## 14. Roadmap
- [ ] Selective context metadata line parsing
- [ ] Collapsible source panel in chat UI
- [ ] Filter trivial watermark-only chunks (< N chars)
- [ ] Add unit tests for retrieval normalization
- [ ] GitHub Actions CI: lint + type + minimal eval sample
- [ ] Add optional local embedding model swap layer

## 15. License / Contributing
Currently no explicit license; add one (MIT / Apache 2) before external contributions.
Contributions: open an issue or PR with a clear description, reproducible steps, and screenshots/logs when relevant.

## FAQ
**Why page numbers in chunk IDs?** Prevents duplicate hash collisions for very short identical pages (e.g., watermark-only pages).  
**Why overlap in chunks?** Preserves semantics across sentence boundaries, improving retrieval fidelity.  
**Why a final JSON marker?** Separates human-visible answer text from machine-readable structured context.  

---
Feel free to open issues for enhancements or clarifications.
