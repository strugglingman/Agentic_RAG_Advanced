# CLAUDE.md — Agentic RAG Project Intelligence

## Project Overview

Enterprise agentic RAG (Retrieval-Augmented Generation) system with:
- **Backend:** Python 3.12, FastAPI, LangGraph, ChromaDB, OpenAI, Prisma ORM
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Architecture:** DDD/CQRS with clean architecture layers (domain → application → infrastructure → presentation)
- **Integrations:** Slack adapter, web UI, file management, email (HITL)

### Key Paths
- Backend entry: `backend/src/fastapi_app.py`
- DI container: `backend/src/setup/ioc/container.py`
- Agent orchestration: `backend/src/services/query_supervisor.py` → `langgraph_builder.py`
- Ingestion pipeline: `backend/src/services/ingestion.py` → `document_processor.py` → `langchain_processor.py`
- Chat command: `backend/src/application/commands/chat/send_message.py`
- Config: `backend/src/config/settings.py`
- Frontend chat: `frontend/components/ChatUI.tsx`

---

## Evaluation Summary (2026 Benchmark)

### Overall Project Score: 5.78/10
Scored against 2025-2026 production RAG systems (LangChain/LlamaIndex ecosystem, enterprise standards).

### Ingestion Pipeline Score: 3.2/10

---

## Ingestion Pipeline — Detailed Findings

### Finding 1 (CRITICAL): Per-Page Chunking Destroys Cross-Page Context

**Files:** `backend/src/services/langchain_processor.py:116-122`

The chunker operates on each page independently:
```python
for page_num, text in pages_text:
    chunks = self._splitter.split_text(text)  # Each page chunked INDEPENDENTLY
```
A topic spanning pages 3-4 is never seen as one unit. The splitter creates a truncated chunk at end of page 3 and a separate chunk at start of page 4.

**2026 best practice:** Concatenate all pages into full document text (tracking page boundaries via character offsets), then chunk the entire document as one unit. Assign each chunk the page number(s) it originated from.

**Impact:** Up to 9% recall gap between page-level and document-level chunking (Weaviate research).

**Fix priority:** 1 — Low effort (1-2 hrs), high impact.

---

### Finding 2 (HIGH): PDF Extraction Uses Outdated Libraries

**Files:** `backend/src/services/document_processor.py:36-180`

Current fallback chain: PyMuPDF → pdfplumber → pypdf

| Current Tool | 2026 State of the Art | Gap |
|---|---|---|
| PyMuPDF `get_text("text")` | **pymupdf4llm** — Markdown output with headers, bold, tables | Loses all document structure |
| pdfplumber (rule-based tables) | **Docling** (IBM, 37k GitHub stars) — AI layout model + TableFormer | No layout understanding |
| pypdf (deprecated API) | **MinerU** (53k stars) — 90.67% on OmniDocBench benchmark | Lowest quality extraction |

PyMuPDF's `get_text("text")` returns a flat character stream with no heading hierarchy, no bold/italic, no structural awareness. The chunker cannot split by document sections because it doesn't know where sections are.

**2026 Tier-1 PDF parsers:**
- **Docling** (IBM) — 37k stars, layout-aware, table structure (TableFormer), built-in OCR, exports Markdown/JSON. Integrates with LangChain and LlamaIndex.
- **MinerU** (OpenDataLab) — 53k stars, highest benchmark scores (90.67 OmniDocBench), 109-language OCR, 1.2B model outperforms 72B VLMs.
- **Marker** — 20k+ stars, fast batch processing (~25 pages/sec on H100).
- **pymupdf4llm** — 2.5k stars, easy drop-in upgrade from PyMuPDF, Markdown output preserving headers via font-size detection.

**Fix priority:** 2 — Low effort for pymupdf4llm, medium effort for Docling.

---

### Finding 3 (HIGH): No OCR Fallback for Scanned PDFs

**Files:** `backend/src/services/document_processor.py:36-57`

If a PDF contains scanned images (no text layer), all three extractors return empty text. The pipeline silently produces zero chunks.

**2026 best practice:** Detect empty/near-empty text extraction per page, then run OCR.
- **PaddleOCR** — 48k stars, 80+ languages, layout-aware
- **Surya** — 15k stars, layout analysis + table recognition + reading order
- Docling has built-in OCR (EasyOCR or Tesseract internally)
- Vision LLM approaches (GPT-4o, Claude) for complex layouts — ~$0.01-0.03/image

**Fix priority:** 7 — Medium effort (3-4 hrs).

---

### Finding 4 (HIGH): No Structure-Aware Chunking

**Files:** `backend/src/services/langchain_processor.py:80-103`

Two strategies available: `RecursiveCharacterTextSplitter` and `SemanticChunker`. Neither uses document structure (headings, sections).

**2026 recommended chunking stack:**
1. **Primary:** Parse to Markdown (via pymupdf4llm or Docling), then `MarkdownHeaderTextSplitter` to chunk by section headers
2. **Fallback:** `RecursiveCharacterTextSplitter` for unstructured content (current default is fine — 600 chars, 120 overlap)
3. **Premium:** **Contextual Retrieval** (Anthropic pattern) — prepend each chunk with LLM-generated context summary. Reduces retrieval errors by up to 67%.
4. **Alternative:** **Late Chunking** (Jina AI) — processes entire document through long-context embedding model first, then chunks. No LLM cost per chunk.

**Optimal chunk parameters (2026 consensus):** 256-512 tokens with 10-20% overlap.

**Fix priority:** 3 — Low effort for MarkdownHeaderTextSplitter (2-3 hrs).

---

### Finding 5 (MEDIUM): DOCX Extraction Misses Heading Hierarchy

**Files:** `backend/src/services/document_processor.py:204-231`

Code iterates `doc.element.body` preserving paragraph/table order (good), but does not detect heading levels. `para.style.name` gives `'Heading 1'`, `'Heading 2'`, etc. — this heading hierarchy is critical for structure-aware chunking.

Additionally, the nested loop (`for para in doc.paragraphs` inside `for element in doc.element.body`) is O(n^2) — slow for large documents. Should build a lookup dict first.

**Fix priority:** 4 — Low effort (1 hr).

---

### Finding 6 (MEDIUM): Excel/CSV Loses Structure

**Files:** `backend/src/services/document_processor.py:190-194` (CSV), `backend/src/utils/file_content_extractor.py:155-180` (Excel)

CSV joins rows with commas into one big string. Excel uses tab-separated rows with no column headers repeated, no Markdown tables.

**2026 best practice:**
- Convert to Markdown tables via `pandas.DataFrame.to_markdown()`
- Repeat column headers every N rows when chunking
- Store sheet name as metadata
- For analytical use: load into SQLite + LLM SQL agent

**Fix priority:** 5 — Low effort (1 hr).

---

### Finding 7 (MEDIUM): Duplicate Extraction Logic

Two separate extraction systems exist:
1. `backend/src/services/document_processor.py:read_text()` — for ingestion (PyMuPDF/pdfplumber)
2. `backend/src/utils/file_content_extractor.py:extract_file_content()` — for chat attachments (uses deprecated PyPDF2, simpler docx/excel)

The chat extractor uses **PyPDF2** which is deprecated. It also doesn't extract tables from DOCX (just `paragraph.text`). These should share the same extraction logic.

**Fix priority:** 9 — Medium effort (3-4 hrs).

---

### Finding 8 (MEDIUM): Missing Metadata Enrichment

**Files:** `backend/src/services/ingestion.py:156-171`

Current metadata is source-level only:
```python
{"dept_id", "user_id", "file_for_user", "chunk_id", "source", "ext",
 "file_id", "size_kb", "tags", "upload_at", "uploaded_at_ts", "page"}
```

**Missing 2026-standard metadata:**
- Section heading breadcrumb (e.g., `"heading": "Introduction > Background"`)
- Element type (paragraph, table, list, code)
- Document-level summary (generated once per doc, stored on all chunks for contextual grounding)
- Language detection per chunk

**Fix priority:** 8 — Medium effort (2-3 hrs).

---

### Finding 9 (LOW): No PowerPoint/HTML/Email Support

`read_text()` handles: PDF, CSV, JSON, DOCX, plaintext. Missing:
- **PPTX** (presentations — common in enterprise)
- **HTML** (web-saved content)
- **EML/MSG** (emails with attachments)

Docling handles all of these natively in a single unified API.

**Fix priority:** 6 (comes with Docling adoption).

---

## Ingestion Upgrade Path (Priority Order)

| # | Upgrade | Effort | Impact | Files to Change |
|---|---------|--------|--------|-----------------|
| 1 | Chunk full document instead of per-page | Low (1-2 hrs) | High | `langchain_processor.py` |
| 2 | Replace PyMuPDF with pymupdf4llm for Markdown output | Low (1-2 hrs) | High | `document_processor.py`, `requirements` |
| 3 | Add MarkdownHeaderTextSplitter for structured docs | Low (2-3 hrs) | High | `langchain_processor.py` |
| 4 | Add heading detection to DOCX extraction | Low (1 hr) | Medium | `document_processor.py` |
| 5 | Excel/CSV to Markdown tables with header context | Low (1 hr) | Medium | `document_processor.py` |
| 6 | Add Docling as primary parser (PDF, DOCX, PPTX, HTML) | Medium (4-6 hrs) | Very High | `document_processor.py`, `requirements` |
| 7 | Add OCR fallback for scanned PDFs | Medium (3-4 hrs) | High | `document_processor.py` |
| 8 | Add heading breadcrumb + element type metadata | Medium (2-3 hrs) | High | `ingestion.py`, `document_processor.py` |
| 9 | Unify extraction logic (merge file_content_extractor) | Medium (3-4 hrs) | Medium | Both extraction files |
| 10 | Contextual Retrieval (Anthropic pattern — LLM context per chunk) | High (6-8 hrs) | Very High | `ingestion.py`, new module |

**Quick wins (1-5):** ~6 hours total, moves ingestion score from 3.2 to ~6/10.
**Full upgrade (1-10):** ~25-35 hours total, moves ingestion score to ~8/10.

---

## 2026 Reference Technologies

### PDF Parsing
- Docling: https://github.com/docling-project/docling
- MinerU: https://github.com/opendatalab/MinerU
- pymupdf4llm: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
- Marker: https://github.com/VikParuchuri/marker

### OCR
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- Surya: https://github.com/datalab-to/surya

### Chunking
- Contextual Retrieval (Anthropic): https://www.anthropic.com/news/contextual-retrieval
- Late Chunking (Jina): https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- LangChain MarkdownHeaderTextSplitter: https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/

### Benchmarks
- OmniDocBench (CVPR 2025): https://github.com/opendatalab/OmniDocBench
- Applied AI PDF Parsing Benchmark: https://www.applied-ai.com/briefings/pdf-parsing-benchmark/
- Weaviate Chunking Strategies: https://weaviate.io/blog/chunking-strategies-for-rag

---

## Implemented Improvements

### Contextual Retrieval (Upgrade #10 — DONE)

**Status:** Service created, pending integration into `ingestion.py`

**What it does:** Prepends an LLM-generated context preamble to each chunk before embedding. A chunk like `"revenue increased 15%"` becomes:
```
This chunk is from the Q3 2024 APAC Marketing Report, section on financial performance.

revenue increased 15%
```
The contextualized text is what gets **embedded and BM25-indexed**. The original chunk text is stored in `metadata.original_text` for display.

**Expected impact (Anthropic benchmarks):**
| Technique Stack | Top-20 Retrieval Failure Reduction |
|---|---|
| Contextual Embeddings only | 35% |
| + Contextual BM25 (our stack has BM25) | 49% |
| + Reranking (our stack has BGE v2 m3) | **67%** |

**Files:**

| File | Status | Role |
|---|---|---|
| `backend/src/config/settings.py:207-218` | Done | 3 env vars: `CONTEXTUAL_RETRIEVAL_ENABLED`, `_MODEL`, `_MAX_WORKERS` |
| `backend/src/services/contextual_retrieval.py` | Done | Core service — LLM calls, ThreadPoolExecutor parallelism, disk cache |
| `backend/src/services/ingestion.py` | **TODO** | Wire `contextualize_chunks()` between chunking and ChromaDB upsert |

**Config (env vars):**
```env
CONTEXTUAL_RETRIEVAL_ENABLED=true      # Feature flag (default: false)
CONTEXTUAL_RETRIEVAL_MODEL=gpt-4o-mini  # LLM for context generation
CONTEXTUAL_RETRIEVAL_MAX_WORKERS=8      # Parallel LLM calls per document
```

**Architecture decisions:**
- **Sync OpenAI client** — `ingest_file()` runs in `run_in_executor` thread pool, so sync is correct
- **ThreadPoolExecutor** for parallelism — 8 concurrent LLM calls per document (configurable)
- **Disk cache** in `{CHROMA_PATH}/.contextual_cache/` — SHA-256 of `"{model}|{full_text}"` as key. Re-ingesting same document costs $0
- **Graceful degradation** — LLM failure per chunk falls back to `"From {filename}."`. Feature disabled = zero behavior change
- **Chunk IDs use original text** (not contextualized) so re-indexing doesn't create duplicates

**Cost estimate:**
| Document Size | Chunks | First Ingest | Re-ingest (cached) |
|---|---|---|---|
| 5 pages | ~25 | ~$0.01 | $0 |
| 10 pages | ~50 | ~$0.06 | $0 |
| 50 pages | ~250 | ~$1.50 | $0 |

**Remaining integration step for `ingestion.py`:**
After chunking (line ~131), before building IDs/docs/metas (line ~134):
1. If enabled, call `contextualize_chunks(full_text, chunks_with_pages, filename)`
2. Loop changes from `(page_num, chunk)` to `(page_num, contextualized_chunk, original_chunk)`
3. `docs.append(contextualized_chunk)` for embedding, `seed` uses `original_chunk` for stable IDs
4. Add `"original_text": original_chunk` to metadata dict

---

## Build & Run

```bash
# Backend
cd backend
pip install -r requirements.linux.txt
python run_fastapi.py

# Frontend
cd frontend
npm install
npm run dev
```

## Testing

```bash
cd backend
python -m pytest tests/
```
