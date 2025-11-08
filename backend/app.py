import os, json, magic
from datetime import datetime
import numpy as np
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pathlib import Path
import logging
import csv
import re
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from flask import Flask, request, jsonify, Response, g
from werkzeug.utils import secure_filename
import chromadb
from chromadb.utils.embedding_functions import sentence_transformer_embedding_function
import hashlib
from rank_bm25 import BM25Okapi
from collections import defaultdict, deque
from typing import Optional
import jwt
from functools import wraps
from werkzeug.exceptions import RequestEntityTooLarge, TooManyRequests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, ~22MB
INDEX_FILE = "index.pkl"
SENT_TARGET = 400
SENT_OVERLAP = 90
CHAT_MAX_TOKENS = 200
TOP_K = 5
TEXT_MAX = 400000
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_HYBRID = os.getenv("USE_HYBRID", "false").lower() in {"1", "true", "yes", "on"}
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes", "on"}
ORG_STRUCTURE_FILE = os.getenv("ORG_STRUCTURE_FILE", "org_structure.json")
CANDIDATES = 20
FUSE_ALPHA = 0.5 # weight for BM25 in hybrid search
MIN_HYBRID = 0.1 # confidence gate for hybrid
MIN_SEM_SIM = 0.35 # confidence gate for semantic-only
_bm25 = None
_bm25_ids = []
_bm25_docs = []
_bm25_metas = []
MAX_HISTORY = 3  # max history to keep in session
FOLDER_SHARED = "shared"
DEPT_SPLIT = "|"
dept_previous = ""
user_previous = ""
SERVICE_AUTH_SECRET = os.getenv("SERVICE_AUTH_SECRET", "")
SERVICE_AUTH_ISSUER = os.getenv("SERVICE_AUTH_ISSUER", "your_service_name")
SERVICE_AUTH_AUDIENCE = os.getenv("SERVICE_AUTH_AUDIENCE", "your_service_audience")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "uploads")
ALLOWED_EXTENSIONS = ""
MIME_TYPES = ""
RATE_LIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

embedding_fun = sentence_transformer_embedding_function.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_fun
)
openAI_Client = OpenAI(api_key=OPENAI_KEY)
_reranker = None
SESSIONS: dict[str, deque] = defaultdict(lambda: deque(maxlen=2 * MAX_HISTORY))

def parse_env_param_to_list(param: str) -> list[str]:
    try:
        param_list = json.loads(param)
        if isinstance(param_list, list) and all(isinstance(p, str) for p in param_list):
            return param_list
    except:
        pass

    return []

def require_identity(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        identity = getattr(g, 'identity', None)
        if not identity:
            return jsonify({"error": "Unauthorized"}), 401
        else:
            user_id = identity.get('user_id', '')
            dept_id = identity.get('dept_id', '')
            sid = identity.get('sid', '')
            if not user_id or not dept_id or not sid:
                return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

app = Flask(__name__)
# Set maximum upload size
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    # This is for all content of the uploaded files + other http content
    return jsonify({"error": f"Error: {str(e)}, Maximum upload size is {MAX_UPLOAD_MB} MB."}), e.code

@app.before_request
def load_identity():
    g.identity = None
    if not SERVICE_AUTH_SECRET:
        return
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return
    token = auth_header.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(
            token,
            SERVICE_AUTH_SECRET,
            algorithms=["HS256"],
            audience=SERVICE_AUTH_AUDIENCE,
            issuer=SERVICE_AUTH_ISSUER,
            options={"require": ["exp", "iat", "aud", "iss"]}
        )
    except:
        return

    email = claims.get("email", "")
    dept = claims.get("dept", "")
    sid = claims.get("sid", "")
    if not email or not dept or not sid:
        return

    g.identity = {
        "user_id": email,
        "dept_id": dept,
        "sid": sid
    }

# Initialize the Limiter for Rate limit
def get_limiter_key():
    if not g.identity:
        return get_remote_address()

    return f"{g.identity.get('dept_id','')}-{g.identity.get('user_id','')}"

limiter = Limiter(
    #key_func=get_remote_address,  # Use the client's IP address for rate limiting
    key_func=get_limiter_key, # stronger, use user credential as key if available and use ip address for non-authenticated
    storage_uri=RATE_LIMIT_STORAGE_URI, # save the rate limit counters, in-memory by default
    app=app,
    default_limits=["50 per day", "10 per minute"]  # Default limits
)
# Handle rate limiting errors
@app.errorhandler(TooManyRequests)
def ratelimit_error(e):
    return jsonify({"error": "Too many requests. Please try again later."}), 429

def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def norm(xs):
    if not xs:
        return []
    mn, mx = min(xs), max(xs)
    if mx - mn < 1e-9:
        return [0.5 for _ in xs]
    
    return [(x - mn) / (mx - mn) for x in xs]

def unique_snippet(ctx, prefix=150):
    seen = set()
    out = []
    for it in ctx:
        key = it["source"] + it["chunk"][0:prefix]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    return out

def canonical_path(base: Path, *sub_paths: str) -> Path:
    base = base.resolve()
    upload_path = (base / Path(*sub_paths)).resolve()
    try:
        upload_path.relative_to(base)
    except ValueError:
        raise ValueError("Attempted directory traversal in upload path")

    return upload_path

def validate_filename(f) -> str:
    filename = secure_filename(f.filename)
    # check file extension
    if not allowed_file(filename):
        return ""

    # check mime type
    head = f.stream.read(8192)
    f.stream.seek(0)
    mime = (magic.from_buffer(head, mime=True) or "").lower()
    global MIME_TYPES
    if not MIME_TYPES:
        mime_types_str = os.getenv("MIME_TYPES", "[]")
        MIME_TYPES = parse_env_param_to_list(mime_types_str)

    mime_ok = any(mime.startswith(x) for x in MIME_TYPES)
    if not mime_ok:
        return ""
    f.stream.seek(0)
    return filename

def create_upload_dir(dept_id: str, user_id: str) -> str:
    try:
        folders = dept_id.split(DEPT_SPLIT)
        upload_dir = Path(UPLOAD_BASE)
        upload_dir = canonical_path(upload_dir, *folders, user_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir
    except:
        return ""

def get_upload_dir(dept_id: str, user_id: str) -> str:
    folders = dept_id.split(DEPT_SPLIT)
    try:
        upload_dir = Path(UPLOAD_BASE)
        upload_dir = canonical_path(upload_dir, *folders, user_id)
        return str(upload_dir) if os.path.exists(str(upload_dir)) else ""
    except:
        return ""

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as exc:
            logging.warning("Failed to load reranker %s: %s", RERANKER_MODEL_NAME, exc)
            return None

    return _reranker

def build_prompt(query, ctx, use_ctx=False):
    if use_ctx:
        system = (
            "You are a helpful assistant, Use ONLY the provided context to answer the question. "
            "Say I don't know if the answer can not be found in the context. "
            "Cite sources using file names exactly as shown, not 'Context #' numbers."
        )
        if not ctx:
            user = f"Question: {query}\n\nAnswer: I don't know."
            return system, user
        
        context_str = "\n\n".join(
            (f"Context {i+1} (Source: {os.path.basename(hit['source'])}" +
            (f", Page: {hit['page']}" if hit.get('page', 0) > 0 else "") +
            f"):\n{hit['chunk']}\n" +
            (f"Hybrid score: {hit['hybrid']:.2f}" if hit['hybrid'] is not None else "") +
            (f", Rerank score: {hit['rerank']:.2f}" if hit['rerank'] is not None else ""))
            for i, hit in enumerate(ctx))
        # unique_files = set(os.path.basename(hit["source"]) for hit in ctx)
        user = (
            f"Question: {query}\n\nContext:\n{context_str}\n\n"
            f"Instructions: Answer the question concisely by synthesizing information from the contexts above. "
            f"At the end of your answer, cite the sources you used. For each source file, list the specific page numbers "
            f"from the contexts you referenced (look at the 'Page:' information in each context header). "
            f"Format: 'Sources: filename1.pdf (pages 15, 23), filename2.pdf (page 7)'"
        )
    else:
        system = (
            "You are a helpful assistant, answer the question to the best of your ability. "
            "If you don't know the answer, say I don't know."
        )
        user = f"Question: {query}\n\nAnswer:"

    return system, user

def get_session_history(sid, n=20):
    if sid not in SESSIONS:
        return []
    
    return [{"role": h["role"], "content": h["content"]} for h in list(SESSIONS[sid])[-n:]]

def allowed_file(file_name: str) -> bool:
    global ALLOWED_EXTENSIONS
    if not ALLOWED_EXTENSIONS:
        allowed_extensions_str = os.getenv("ALLOWED_EXTENSIONS", "[]")
        ALLOWED_EXTENSIONS = parse_env_param_to_list(allowed_extensions_str)

    return "." in file_name and file_name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        # Return list of (page_num, text) tuples
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text:
                pages_text.append((page_num, text))
        return pages_text
    
    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            all_rows = [",".join(row) for row in reader]
            return [(0, "\n".join(all_rows)[:TEXT_MAX])]

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return [(0, json.dumps(data, indent=2)[:TEXT_MAX])]
            except:
                return [(0, f.read()[:TEXT_MAX])]

    if ext == ".docx":
        doc = Document(file_path)
        return [(0, "\n".join([p.text for p in doc.paragraphs])[:TEXT_MAX])]
    
    with open(file_path, "r", encoding="utf-8") as f:
        return [(0, f.read()[:TEXT_MAX])]
    
def sentence_split(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    return [p.strip() for p in parts if p and p.strip()]

def make_chunks(pages_text: list, target=SENT_TARGET, overlap=SENT_OVERLAP) -> list[tuple]:
    all_chunks = []
    
    for page_num, text in pages_text:
        chunks, buff, size = [], [], 0
        sentences = sentence_split(text)
        for s in sentences:
            buff.append(s)
            if size + len(s) <= target:
                size += len(s) + 1
            else:
                if buff:
                    chunks.append((page_num, ' '.join(buff)))

                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(buff):
                    if overlap_size + len(sent) + (1 if overlap_sentences else 0) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent) + (1 if overlap_size > 0 else 0)
                    else:
                        break

                buff = overlap_sentences
                size = sum(len(s) for s in buff) + max(0, len(buff) - 1)

        # Append remaining buffer
        if buff:
            chunks.append((page_num, ' '.join(buff)))
        
        all_chunks.extend(chunks)

    return all_chunks

def build_bm25(dept_id: str, user_id: str):
    global _bm25, _bm25_ids, _bm25_docs, _bm25_metas
    try:
        res = collection.get(include=["documents", "metadatas"])
        docs = res["documents"] if res and "documents" in res else []
        metas = res["metadatas"] if res and "metadatas" in res else []
        ids = res.get("ids", []) or []
        docs = docs[0] if docs and isinstance(docs[0], list) else docs
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        metas = metas[0] if metas and isinstance(metas[0], list) else metas
        # filter by user_id and dept_id
        filtered_ids, filtered_docs, filtered_metas = [], [], []
        for i, meta in enumerate(metas):
            if meta.get("dept_id", "") == dept_id and \
            ((meta.get("user_id", "") == user_id or (not meta.get("file_for_user", False)))):
                filtered_ids.append(ids[i])
                filtered_docs.append(docs[i])
                filtered_metas.append(meta)

        tokenized = [d.split() for d in filtered_docs]
        _bm25 = BM25Okapi(tokenized)
        _bm25_ids = filtered_ids
        _bm25_docs = filtered_docs
        _bm25_metas = filtered_metas
    except:
        _bm25 = None
        _bm25_ids = []
        _bm25_docs = []
        _bm25_metas = []

def ingest_one(info: Optional[dict], app_user_id: str, app_dept_id: str) -> Optional[str]:
    if not info:
        return None
    dept_id = info.get("dept_id", "")
    user_id = info.get("user_id", "")
    if not dept_id or not user_id:
        return None
    file_for_user = info.get("file_for_user", False)
    if file_for_user and (dept_id != app_dept_id or user_id != app_user_id):
        return None
    ingested = info.get("ingested", False)
    if ingested:
        return None
    file_path = info.get("file_path", "")
    if not os.path.exists(file_path):
        return None
    pages_text = read_text(file_path)
    if not pages_text:
        return None

    # chunking - now returns list of (page_num, chunk_text) tuples
    chunks_with_pages = make_chunks(pages_text)
    filename = info.get("filename", os.path.basename(file_path))
    # upsert to chroma
    ids, docs, metas = [], [], []
    seen = set()
    for page_num, chunk in chunks_with_pages:
        # Incorporate page number into the ID seed to avoid collisions when two pages have identical extracted text
        if file_for_user:
            seed = f"{dept_id}|{user_id}|{filename}|p{page_num}|{chunk}"
        else:
            seed = f"{dept_id}|{filename}|p{page_num}|{chunk}"
        chunk_id = make_id(seed)
        ids.append(chunk_id)
        docs.append(chunk)
        metas.append({
            "dept_id": info.get("dept_id", ""),
            "user_id": info.get("user_id", ""),
            "file_for_user": file_for_user,
            "chunk_id": chunk_id,
            "source": filename,
            "ext": filename.split(".")[-1].lower(),
            "file_id": info.get("file_id", ""),
            "size_kb": info.get("size_kb", 0),
            "tags": info.get("tags", "").lower(),
            "upload_at": info.get("upload_at", ""),
            "uploaded_at_ts": info.get("uploaded_at_ts", 0),
            "page": page_num  # Add page number to metadata
        })

        if chunk_id in seen:
            print(f"Duplicate chunk detected even with page in ID: {chunk_id}, file: {filename}, page: {page_num}, first 80 chars: {chunk[:80]}")
        else:
            seen.add(chunk_id)

    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)

    # set ingested flag
    with open(file_path + ".meta.json", "w", encoding="utf-8") as info_f:
        info["ingested"] = True
        json.dump(info, info_f, indent=2)

    return info.get("file_id", "") if docs else None

def build_where(dept_id, user_id):
    if not dept_id:
        raise ValueError("No organization ID provided in headers")
    if not user_id:
        raise ValueError("No user ID provided in headers")

    payload = request.get_json(force=True)
    filters = payload.get('filters', [])
    exts = next((f.get("exts") for f in filters if "exts" in f and isinstance(f.get("exts"), list)), None)

    where_clauses = []
    # build exts clause
    if exts:
        if len(exts) == 1:
            where_clauses.append({"ext": exts[0]})
        elif len(exts) > 1:
            where_clauses.append({"$or": [{"ext": ext} for ext in exts]})

    # build dept_id clause
    where_clauses.append({"dept_id": dept_id})
    # build user_id clause if file_for_user is specified
    where_clauses.append({ "$or": [
        {"file_for_user": False},
        {"user_id": user_id}
    ]})

    if len(where_clauses) > 1:
        return {"$and": where_clauses}
    elif len(where_clauses) == 1:
        return where_clauses[0]
    else:
        return None

def retrieve(query, dept_id="", user_id="", top_k=TOP_K, where: dict | None = None, use_hybrid=False, use_reranker=False):
    try:
        res = collection.query(
            query_texts=[query],
            n_results=max(CANDIDATES, TOP_K),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else []
        if not docs:
            return [], "No relevant documents found"

        # transform cosine distance -> similarity (1 - distance), normalize within semantic top-N
        sims_raw = [max(0, 1 - d) for d in dists]
        sims_norm = norm(sims_raw)  # Normalize semantic scores BEFORE union
        ctx_original = [{
            "dept_id": meta.get("dept_id", "") if meta else "",
            "user_id": meta.get("user_id", "") if meta else "",
            "file_for_user": meta.get("file_for_user", False) if meta else False,
            "chunk_id": meta.get("chunk_id", "") if meta else "",
            "chunk": d,
            "file_id": meta.get("file_id", "") if meta else "",
            "source": meta.get("source", "") if meta else "",
            "ext": meta.get("ext", "") if meta else "",
            "tags": meta.get("tags", "") if meta else "",
            "size_kb": meta.get("size_kb", 0) if meta else 0,
            "upload_at": meta.get("upload_at", "") if meta else "",
            "uploaded_at_ts": meta.get("uploaded_at_ts", 0) if meta else 0,
            "page": meta.get("page", 0) if meta else 0,  # Add page number
            "sem_sim": sim_norm,  # Already normalized within semantic top-N
            "bm25": 0.0,
            "hybrid": 0.0,
            "rerank": 0.0
        } for d, meta, sim_norm in zip(docs, metas, sims_norm)]
        ctx_original = unique_snippet(ctx_original, prefix=150)

        ctx_candidates = []
        # run bm25 and combine semantic + bm25 scores if hybrid
        if use_hybrid:
            global dept_previous, user_previous
            if not _bm25 or user_id != user_previous or dept_id != dept_previous:
                print("BM25 index not built, building now...")
                build_bm25(dept_id, user_id)
                dept_previous = dept_id
                user_previous = user_id
            
            if _bm25 and _bm25_docs:
                _bm25_scores = _bm25.get_scores(query.split())
                count = max(CANDIDATES, TOP_K)
                top_indexes = np.argsort(_bm25_scores)[::-1][:count]
                # Normalize BM25 scores BEFORE union (within BM25 top-N)
                bm25_norm = norm([_bm25_scores[i] for i in top_indexes])
                ctx_bm25 = [{
                    "dept_id": _bm25_metas[idx].get("dept_id", "") if _bm25_metas else "",
                    "user_id": _bm25_metas[idx].get("user_id", "") if _bm25_metas else "",
                    "file_for_user": _bm25_metas[idx].get("file_for_user", False) if _bm25_metas else False,
                    "chunk_id": _bm25_metas[idx].get("chunk_id", "") if _bm25_metas else "",
                    "chunk": _bm25_docs[idx],
                    "file_id": _bm25_metas[idx].get("file_id", "") if _bm25_metas else "",
                    "source": _bm25_metas[idx].get("source", "") if _bm25_metas else "",
                    "ext": _bm25_metas[idx].get("ext", "") if _bm25_metas else "",
                    "tags": _bm25_metas[idx].get("tags", "") if _bm25_metas else "",
                    "size_kb": _bm25_metas[idx].get("size_kb", 0) if _bm25_metas else 0,
                    "upload_at": _bm25_metas[idx].get("upload_at", "") if _bm25_metas else "",
                    "uploaded_at_ts": _bm25_metas[idx].get("uploaded_at_ts", 0) if _bm25_metas else 0,
                    "page": _bm25_metas[idx].get("page", 0) if _bm25_metas else 0,  # Add page number
                    "sem_sim": 0.0,
                    "bm25": float(score),  # Already normalized within BM25 top-N
                    "hybrid": 0.0,
                    "rerank": 0.0
                } for idx, score in zip(top_indexes, bm25_norm)]
                ctx_bm25 = unique_snippet(ctx_bm25, prefix=150)

                # Union both result sets
                ctx_unioned = {}
                for bm25_item in ctx_bm25:
                    key = bm25_item["chunk_id"]
                    ctx_unioned[key] = bm25_item
                
                for sem_item in ctx_original:
                    key = sem_item["chunk_id"]
                    if key in ctx_unioned:
                        # Merge: overlapping chunks get both normalized scores
                        ctx_unioned[key] = {** ctx_unioned[key], **sem_item}
                    else:
                        # Semantic-only chunks: sem_sim is normalized, bm25=0
                        ctx_unioned[key] = sem_item

                ctx_candidates = list(ctx_unioned.values())
                
                # Calculate hybrid with normalized scores (both already in [0,1])
                for item in ctx_candidates:
                    item["hybrid"] = FUSE_ALPHA * item.get("bm25", 0.0) + (1 - FUSE_ALPHA) * item.get("sem_sim", 0.0)

                # confidence gate on hybrid
                max_hybrid = max(item.get("hybrid", 0) for item in ctx_candidates) if ctx_candidates else 0
                if max_hybrid < MIN_HYBRID:
                    return [], "No relevant documents found after applying hybrid confidence threshold."

                ctx_candidates = sorted(ctx_candidates, key=lambda x: x.get("hybrid", 0), reverse=True)
        else:
            # confidence gate on semantic-only (already normalized in ctx_original)
            ctx_candidates = [item for item in ctx_original]
            if max(sims_raw) < MIN_SEM_SIM:
                return [], "No relevant documents found after applying semantic confidence threshold."
            
            ctx_candidates = sorted(ctx_candidates, key=lambda x: x.get("sem_sim", 0), reverse=True)
        
        # rerank top candidates if reranker is available
        if use_reranker:
            reranker = get_reranker()
            if not reranker:
                return [], "Rerank failed."
            if not ctx_candidates:
                return [], "No candidates to rerank."
            
            try:
                count = min(len(ctx_candidates), max(top_k*3, 12))
                ctx_for_rerank = ctx_candidates[:count]
                rerank_inputs = [(query, item["chunk"]) for item in ctx_for_rerank]
                rerank_scores = reranker.predict(rerank_inputs)
                ranked_pair = sorted(zip(rerank_scores, ctx_for_rerank), key=lambda pair: pair[0], reverse=True)
                ctx_candidates = [{**item, "rerank": float(score)} for score, item in ranked_pair]
            except:
                return [], "Rerank failed."

        return ctx_candidates[:top_k], None
    except Exception as e:
        return [], str(e)

@app.post('/chat')
@require_identity
@limiter.limit("30 per minute; 500 per day")
def chat():
    dept_id = g.identity.get('dept_id', '')
    user_id = g.identity.get('user_id', '')
    if not dept_id or not user_id:
        return jsonify({"error": "No organization ID or user ID provided in headers"}), 400

    payload = request.get_json(force=True)
    msgs = payload.get('messages', [])

    sid = g.identity.get('sid', '')
    if sid not in SESSIONS:
        SESSIONS[sid] = deque(maxlen=2 * MAX_HISTORY)
    latest_user_msg = None
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user":
        latest_user_msg = msgs[-1]

    if not latest_user_msg:
        return jsonify({"error": "No user message found"}), 400
    if not latest_user_msg.get("content").strip():
        return jsonify({"error": "Empty user message"}), 400

    query = latest_user_msg.get("content").strip()
    try:
        # build where
        where = build_where(dept_id, user_id)
        print(f"Where clause: {where}")
        ctx, err = retrieve(query, dept_id=dept_id, user_id=user_id, top_k=TOP_K, where=where, use_hybrid=USE_HYBRID, use_reranker=USE_RERANKER)
        if err:
            print(f"Retrieval error: {err}")
            return jsonify({"error": err}), 500
        if not ctx:
            return jsonify({"error": "No relevant documents found"}), 404

        # filter tags
        filters = payload.get('filters', [])
        tags_filter = next((f.get("tags") for f in filters if "tags" in f and isinstance(f.get("tags"), list)), None)
        if tags_filter:
            ctx = [c for c in ctx if any(tag in c.get("tags", "").lower().split(",") for tag in tags_filter)]

        system, user = build_prompt(query, ctx, use_ctx=True)
        history = get_session_history(sid, MAX_HISTORY)
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]
        messages = [m for m in messages if m.get("content") and m.get("role") in {"system", "user", "assistant"}]
        def generate():
            answer = []
            try:
                resp = openAI_Client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=CHAT_MAX_TOKENS,
                    stream=True
                )

                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                        answer.append(delta)
            except Exception as e:
                print(f"Error: {e}")
                yield f"\n[upstream_error] {type(e).__name__}: {e}"
            finally:
                # update session history with latest query and assistant answer
                if latest_user_msg:
                    SESSIONS[sid].append({"role": latest_user_msg.get("role"), "content": latest_user_msg.get("content")})
                if answer:
                    SESSIONS[sid].append({"role": "assistant", "content": ''.join(answer)})

                yield f"\n__CONTEXT__:{json.dumps(ctx)}"

        return Response(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post('/upload')
@require_identity
def upload():
    if not request.files and not request.files.get("file"):
        return jsonify({"error": "No file part in the request"}), 400

    user_id = g.identity.get("user_id", "")
    dept_id = g.identity.get("dept_id", "")
    if not user_id or not dept_id:
        return jsonify({"error": "No user ID or organization ID provided"}), 400

    f = request.files['file']
    filename = validate_filename(f)
    if not filename:
        return jsonify({"error": "File is not valid mime type or extension"}), 400

    file_for_user = request.form.get("file_for_user", "0")
    upload_dir = create_upload_dir(dept_id, FOLDER_SHARED)
    if file_for_user == "1":
        upload_dir = create_upload_dir(dept_id, user_id)
    if not upload_dir:
        return jsonify({"error": "Failed to create upload directory"}), 500
    file_path = canonical_path(upload_dir, filename)

    # check same file exits
    if os.path.exists(file_path):
        return jsonify({"error": "File with the same name already exists"}), 400
    
    f.save(file_path)

    # save file meta info into file for further ingestion
    tags = request.form.get("tags", "")
    tags_raw = json.loads(tags) if tags else []
    tags_str = ""
    if tags_raw:
        tags_str = ",".join(tags_raw) if tags_raw else ""

    file_info = {
            "file_id": make_id(filename),
            "file_path": str(file_path),
            "filename": filename,
            "source": filename,
            "ext": filename.rsplit(".", 1)[1].lower(),
            "size_kb": round(os.path.getsize(str(file_path)) / 1024, 1),
            "tags": tags_str,
            "upload_at": datetime.now().isoformat(),
            "uploaded_at_ts": datetime.now().timestamp(),
            "user_id": user_id,
            "dept_id": dept_id,
            "file_for_user": True if file_for_user == "1" else False,
            "ingested": False
    }
    fileinfo_path = canonical_path(upload_dir, f"{filename}.meta.json")
    with open(fileinfo_path, "w", encoding="utf-8") as info_f:
        json.dump(file_info, info_f, indent=2)

    return jsonify({"msg": "File uploaded successfully"}), 200

@app.post('/ingest')
@require_identity
def ingest():
    body = request.get_json(force=True)
    file_id = body.get("file_id", "") if body else ""
    file_path = body.get("file_path", "") if body else ""
    file_path = os.path.join(UPLOAD_BASE, file_path) if file_path and file_path != "ALL" else file_path
    if not file_id:
        return jsonify({"message": "No correct file specified"}), 400
    if file_path != "ALL" and not os.path.exists(file_path):
        return jsonify({"message": "No correct file path specified"}), 400
    dept_id = g.identity.get('dept_id', '')
    if not dept_id:
        return jsonify({"error": "No organization ID provided"}), 400
    user_id = g.identity.get('user_id', '')
    if not user_id:
        return jsonify({"error": "No user ID provided"}), 400

    meta_data_all = []
    meta_data_files = []
    dir_user = get_upload_dir(dept_id, user_id)
    if dir_user:
        meta_data_files = [f for f in os.listdir(dir_user) if f.endswith(".meta.json")]
        for mf in meta_data_files:
            with open(os.path.join(dir_user, mf), "r", encoding="utf-8") as info_f:
                info = json.load(info_f)
                meta_data_all.append(info)

    dir_shared = get_upload_dir(dept_id, FOLDER_SHARED)
    if dir_shared:
        meta_data_files = [f for f in os.listdir(dir_shared) if f.endswith(".meta.json")]
        for mf in meta_data_files:
            with open(os.path.join(dir_shared, mf), "r", encoding="utf-8") as info_f:
                info = json.load(info_f)
                meta_data_all.append(info)

    ingested_info = ""
    if file_id == "ALL":
        for info in meta_data_all:
            fid = ingest_one(info, app_user_id=user_id, app_dept_id=dept_id)
            if fid:
                ingested_info += f"{fid}\n"
    else:
        info = next((m for m in meta_data_all if m.get("file_id") == file_id), None)
        fid = ingest_one(info, app_user_id=user_id, app_dept_id=dept_id)
        if fid:
            ingested_info = f"{fid}\n"

    # rebuild BM25 index
    global dept_previous, user_previous
    dept_previous = dept_id
    user_previous = user_id
    build_bm25(dept_id, user_id)

    docs = collection.get(include=["documents"])["documents"]
    count = len(docs) if docs else 0
    ingested_info = f"{ingested_info}\n and the count of chunks is: {count}"

    msg = f"Ingestion completed for file_ids:\n {ingested_info}" if ingested_info else "No new content ingested."
    return jsonify({"message": msg}), 200

@app.get('/files')
@require_identity
def list_files():
    dept_id = g.identity.get('dept_id', '')
    if not dept_id:
        return jsonify({"error": "No organization ID provided"}), 400
    user_id = g.identity.get('user_id', '')
    if not user_id:
        return jsonify({"error": "No user ID provided"}), 400

    files_info = []
    # List this user's files first
    dir_user = get_upload_dir(dept_id, user_id)
    if dir_user:
        files = os.listdir(dir_user)
        for f in files:
            if f.endswith("meta.json"):
                with open(os.path.join(dir_user, f), "r", encoding="utf-8") as info_f:
                    info = json.load(info_f)
                    if info.get("dept_id", "") == dept_id and ((not info.get("file_for_user", False)) or info.get("user_id") == user_id):
                        # Include a sanitized or relative file path
                        info["file_path"] = os.path.relpath(info.get("file_path", ""), UPLOAD_BASE)
                        files_info.append(info)

    # List shared files next
    dir_shared = get_upload_dir(dept_id, FOLDER_SHARED)
    if dir_shared:
        files = os.listdir(dir_shared)
        for f in files:
            if f.endswith("meta.json"):
                with open(os.path.join(dir_shared, f), "r", encoding="utf-8") as info_f:
                    info = json.load(info_f)
                    if info.get("dept_id", "") == dept_id and ((not info.get("file_for_user", False)) or info.get("user_id") == user_id):
                        # Include a sanitized or relative file path
                        info["file_path"] = os.path.relpath(info.get("file_path", ""), UPLOAD_BASE)
                        files_info.append(info)

    return jsonify({"files": files_info}), 200

@app.get('/org-structure')
@limiter.limit("1 per minute; 10 per day", key_func=get_remote_address)
def org_structure():
    try:
        with open(ORG_STRUCTURE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return jsonify(data), 200
    except Exception as ex:
        return jsonify({"error": f"Failed to read organization structure: {str(ex)}"}), 500

@app.get('/')
@limiter.exempt
def root():
    return jsonify({ "message": "Server is running." }), 200

@app.get('/health')
@limiter.exempt # Exempt health check from rate limiting
def health():
    return jsonify({ "status": "healthy" }), 200

@app.get('/func-test')
@limiter.exempt
def func_test():
    return list_files()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)