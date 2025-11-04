import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pathlib import Path
import logging
import csv, json
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
from werkzeug.utils import secure_filename
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import hashlib
from rank_bm25 import BM25Okapi
from collections import defaultdict, deque
import uuid, time

load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, ~22MB
INDEX_FILE = "index.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
TEXT_MAX = 100000
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_HYBRID = os.getenv("USE_HYBRID", "false").lower() in {"1", "true", "yes", "on"}
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes", "on"}
CANDIDATES = 20
FUSE_ALPHA = 0.5 # weight for BM25 in hybrid search
MIN_HYBRID = 0.1 # confidence gate for hybrid
MIN_SEM_SIM = 0.35 # confidence gate for semantic-only
_bm25 = None
_bm25_ids = []
_bm25_docs = []
_bm25_metas = []
MAX_HISTORY = 10  # max history to keep in session

os.makedirs(UPLOAD_DIR, exist_ok=True)
embedding_fun = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_fun
)
openAI_Client = OpenAI(api_key=OPENAI_KEY)
_reranker = None

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "supersecretkey")
SESSIONS: dict[str, deque] = defaultdict(lambda: deque(maxlen=2 * MAX_HISTORY))

def allowed(file_name):
    return file_name.lower().endswith((".txt", ".md", ".pdf", ".csv", ".json", ".docx"))

def get_reranker():
    global _reranker
    if not USE_RERANKER:
        return None
    if _reranker is None:
        try:
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as exc:
            logging.warning("Failed to load reranker %s: %s", RERANKER_MODEL_NAME, exc)
            return None
    return _reranker

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

def add_session_history(sid, role, content):
    if sid not in SESSIONS:
        SESSIONS[sid] = deque(maxlen=2 * MAX_HISTORY)

    SESSIONS[sid].append({"role": role, "content": content, "ts": time.time()})

def get_session_history(sid, n=20):
    if sid not in SESSIONS:
        return []
    
    return [{"role": h["role"], "content": h["content"]} for h in list(SESSIONS[sid])[-n:]]

def build_bm25():
    global _bm25, _bm25_ids, _bm25_docs, _bm25_metas
    try:
        res = collection.get(include=["documents", "metadatas"])
        docs = res["documents"] if res and "documents" in res else []
        metas = res["metadatas"] if res and "metadatas" in res else []
        ids = res.get("ids", []) or []
        docs = docs[0] if docs and isinstance(docs[0], list) else docs
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        metas = metas[0] if metas and isinstance(metas[0], list) else metas
        tokenized = [d.split() for d in docs]
        _bm25 = BM25Okapi(tokenized)
        _bm25_ids = ids
        _bm25_docs = docs
        _bm25_metas = metas
    except:
        _bm25 = None
        _bm25_ids = []
        _bm25_docs = []
        _bm25_metas = []

def make_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

#def ef(texts):
#    return embedding_fun.embed_documents(texts, normalize=True).tolist()

def read_text(file_path):
    filename = os.path.splitext(file_path)
    ext = filename[1].lower()
    full_text = ""
    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            full_text += text + "\n"

        return full_text[:TEXT_MAX]
    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [",".join(row) for row in reader]
            return "\n".join(rows)[:TEXT_MAX]
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return json.dumps(data, indent=2)[:TEXT_MAX]
            except:
                return f.read()[:TEXT_MAX]
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])[:TEXT_MAX]

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks

def ingest(folder):
    files = [f for f in Path(folder).rglob("*") if f.is_file() and allowed(f.name)]
    for f in files:
        text = read_text(f)
        if not text or not text.strip():
            continue

        chunks = chunk_text(text)
        #read tags
        tags_str = ""
        path_tags = f"{f}.tags"
        if os.path.exists(path_tags):
            with open(path_tags, "r", encoding="utf-8") as tf:
                tags_str = tf.read().strip()

        ids, docs, metas = [], [], []
        for chunk in chunks:
            ids.append(make_id(f"{f}|{chunk}"))
            docs.append(chunk)
            metas.append({"source": os.fspath(f),
                          "ext": f.suffix.lower(),
                          "filename": f.name,
                          "uploaded_at": datetime.now().isoformat(),
                          "uploaded_at_ts": datetime.now().timestamp(),
                          "size_kb": f.stat().st_size // 1024,
                          "tags": tags_str if tags_str else ""})

        if docs:
          collection.upsert(ids=ids, documents=docs, metadatas=metas)
    try:
        build_bm25()
    except:
        pass

    return collection, None if collection.count() > 0 else "No documents in collection after ingestion."

def retrieve(query, top_k=TOP_K, where: dict | None = None, use_reranker=True, use_hybrid=True):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=max(CANDIDATES, top_k),
            where=where,
            include=["documents", "metadatas", "distances"]
            )
        
        ids = results["ids"][0] if results.get("ids") else []
        docs  = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        dists = results["distances"][0] if results.get("distances") else []
        if not docs:
            return [], "No relevant documents found."

        # transform cosine distance -> similarity (1 - distance), then normalize (for fusion)
        sims_raw = [max(0, 1 - d) for d in dists]
        sims_norm = norm(sims_raw)

        sem_ctx = [{"id": i,
                "chunk": d,
                "source": m.get("source", "unknown"),
                "uploaded_at": m.get("uploaded_at", "unknown"),
                "ext": m.get("ext", "unknown"),
                "tags": m.get("tags", "unknown"),
                "sem_sim": float(s)}
            for i, d, m, s in zip(ids, docs, metas, sims_norm)]
        sem_ctx = unique_snippet(sem_ctx, prefix=150)
        
        candidate_ctx = []
        if use_hybrid:
            if _bm25 is None:
                print("BM25 index not built, building now...")
                build_bm25()

            bm25_ctx = []
            _bm25_scores = _bm25.get_scores(query.split())
            count = max(CANDIDATES, top_k)
            top_id_indexes = np.argsort(_bm25_scores)[::-1][:count]
            _bm25_norm  = norm([_bm25_scores[i] for i in top_id_indexes])
            for idx, s in zip(top_id_indexes, _bm25_norm):
                bm25_ctx.append({
                    "id": _bm25_ids[idx],
                    "chunk": _bm25_docs[idx],
                    "source": _bm25_metas[idx].get("source", "unknown"),
                    "uploaded_at": _bm25_metas[idx].get("uploaded_at", "unknown"),
                    "ext": _bm25_metas[idx].get("ext", "unknown"),
                    "tags": _bm25_metas[idx].get("tags", "unknown"),
                    "bm25": float(s),
                    "sem_sim": 0.0
                })

            # union bm25 ctx with semantic ctx (collection.query)
            unioned_ctx = {}
            for bm25_item in bm25_ctx:
                unioned_ctx[bm25_item["id"]] = {**bm25_item}
            
            for sem_item in sem_ctx:
                if sem_item.get("id", "") in unioned_ctx:
                    unioned_ctx[sem_item["id"]] = {**unioned_ctx[sem_item["id"]], **sem_item}
                else:
                    unioned_ctx[sem_item["id"]] = {**sem_item, "bm25": 0.0}
            candidate_ctx = list(unioned_ctx.values())

            # compute hybrid score and sort
            for i, item in enumerate(candidate_ctx):
                item["hybrid"] = FUSE_ALPHA * item["bm25"] + (1 - FUSE_ALPHA) * item["sem_sim"]
            
            # confidence gate on hybrid
            candidate_ctx = [item for item in candidate_ctx if item.get("hybrid", 0) >= MIN_HYBRID]
            if not candidate_ctx:
                return [], "No relevant documents found after applying hybrid confidence threshold."

            candidate_ctx.sort(key=lambda x: x["hybrid"], reverse=True)
        else:
            candidate_ctx = [{**item, "bm25": None, "hybrid": None} for item in sem_ctx]
            # confidence gate on sem_sim
            if max(sims_raw) < MIN_SEM_SIM:
                return [], "No relevant documents found after applying sim confidence threshold."

        # take top candidates for reranking if used
        if use_reranker:
            reranker = get_reranker()
            if reranker and candidate_ctx:
                try:
                    max_count = min(len(candidate_ctx), max(top_k*3, 12))
                    top_for_rerank = candidate_ctx[:max_count]
                    pairs = [(query, item["chunk"]) for item in top_for_rerank]
                    reranker_scores = reranker.predict(pairs)
                    ranked = sorted(zip(top_for_rerank, reranker_scores), key=lambda pair: pair[1], reverse=True)
                    candidate_ctx = [{**item, "rerank": float(score)} for item, score in ranked]
                except Exception as exc:
                    candidate_ctx = [{**item, "rerank": None} for item in candidate_ctx]
                    logging.warning("Reranker failed for query %s: %s", query, exc)
        else:
            candidate_ctx = [{**item, "rerank": None} for item in candidate_ctx]

        return candidate_ctx[:top_k], None
    except Exception as e:
        return [], str(e)

def build_prompt(query, ctx):
    system = (
        "You are a helpful assistant, Use ONLY the provided context to answer the question. "
        "Say I don't know if the answer can not be found in the context. "
        "Cite sources using file names exactly as shown, not 'Context #' numbers."
    )
    context_str = "\n\n".join(
        (f"Context {i+1}:\n{hit['chunk']}\n" +
        (f", with hybrid score {hit['hybrid']:.2f}" if hit['hybrid'] is not None else "" + " \n") +
        (f", with rerank score {hit['rerank']:.2f}" if hit['rerank'] is not None else ""))
        for i, hit in enumerate(ctx))
    unique_files = set(os.path.basename(hit["source"]) for hit in ctx)
    user = (
        f"Question: {query}\n\n, And Context:\n {context_str}\n\n"
        f"Answer concisely based on context, and cite your sources: {', '.join(unique_files)}"
    )

    return system, user

def get_ext_options():
    try:
        res = collection.get(include=["metadatas"])
        metas = res["metadatas"] if res and "metadatas" in res else []
        if metas and isinstance(metas[0], list):
            metas = metas[0]
        return sorted({m.get("ext") for m in metas if m and m.get("ext")})
    except:
        return []
    
def build_where(form):
    where_clauses = []
    # filter by extensions
    exts = form.getlist("ext")
    if exts:
        if len(exts) > 1:
            #where = {"$or": [{"ext": ext} for ext in exts]}
            where = {"ext": {"$in": exts}}
        else:
            where = {"ext": exts[0]}
        
        where_clauses.append(where)

    # filter by upload date range
    uploaded_from = form.get("uploaded_from", "").strip() or ""
    uploaded_to = form.get("uploaded_to", "").strip() or ""
    if uploaded_from:
        uploaded_from_ts = datetime.fromisoformat(uploaded_from).timestamp()
        where_clauses.append({"uploaded_at_ts": {"$gte": uploaded_from_ts}})
    if uploaded_to:
        uploaded_to_ts = datetime.fromisoformat(uploaded_to).timestamp()
        where_clauses.append({"uploaded_at_ts": {"$lte": uploaded_to_ts}})

    if where_clauses:
        if len(where_clauses) > 1:
            return {"$and": where_clauses}
        else:
            return where_clauses[0]
    else:
        return None

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Mini RAG – Chat with Your Files</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 2rem; max-width: 900px; }
    header { margin-bottom: 1rem; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; }
    label { font-weight: 600; display:block; margin-bottom: .5rem; }
    input[type="text"], textarea { width: 100%; padding: .6rem; border: 1px solid #ccc; border-radius: 8px; }
    button { padding: .6rem 1rem; border: 0; border-radius: 8px; background: #0d6efd; color: #fff; cursor: pointer; }
    button.secondary { background: #6c757d; }
    .muted { color: #666; font-size: .9rem; }
    .sources { color:#444; font-size: .95rem; }
    .status { padding:.5rem .75rem; border-radius:8px; background:#f8f9fa; border:1px solid #eee; }
    .success { border-color:#c7f3d0; background:#e9fbea; }
    .error { border-color:#f5c2c7; background:#fcebea; }
    code { background:#f6f8fa; padding:.15rem .3rem; border-radius:6px; }
  </style>
</head>
<body>
<header>
  <h1>Mini RAG – Chat with Your Files</h1>
  <p class="muted">Upload .txt / .md / .pdf, build an index, then ask questions grounded in those files.</p>
</header>

<div class="card">
  <h2>1) Upload files</h2>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <label>Select files (.txt, .md, .pdf)</label>
    <input type="file" name="files" multiple accept=".txt,.md,.pdf,.csv,.json,.docx" />
    <br/>
    <div style="margin:.5rem 0;">
    <label>Tags for this upload (comma separated):</label>
    <input type="text" name="tags" placeholder="policy, hr, finance">
    </div>
    <p class="muted">Files are saved to <code>./uploads</code>.</p>
    <button type="submit">Upload</button>
  </form>
</div>

<div class="card">
  <h2>2) Build / Rebuild collection</h2>
  <form method="post" action="/ingest">
    <button type="submit">Build collection from uploads</button>
    <span class="muted">Embeddings: {{ model_name }}. Collection: <code>{{ index_file }}</code></span>
  </form>
</div>

<div class="card">
  <h2>Danger Zone</h2>
  <form method="post" action="/clear_collection" onsubmit="return confirm('Are you sure you want to clear the entire collection and database?');">
    <button type="submit" style="background:#d9534f;">Clear Collection &amp; ChromaDB</button>
    <span class="muted">This will delete all indexed data!</span>
  </form>
</div>

<div class="card">
  <h2>4) Ask a question</h2>
  <form method="post" action="/chat">
    <div style="margin-bottom:.75rem;">
      <label>File types</label><br/>
      {% if ext_options %}
        {% for ext in ext_options %}
          <label style="display:inline-block;margin-right:12px;">
            <input type="checkbox" name="ext" value="{{ ext }}"
                   {% if selected_ext and ext in selected_ext %}checked{% endif %}>
            {{ ext }}
          </label>
        {% endfor %}
      {% else %}
        <em>No file type metadata yet</em>
      {% endif %}
    </div>
    <div style="margin-bottom:.75rem;">
      <label>Uploaded between</label><br/>
        <label>From: <input type="date" name="uploaded_from" value="{{ uploaded_from or '' }}" /></label>
        <label>To: <input type="date" name="uploaded_to" value="{{ uploaded_to or '' }}" /></label>
    </div>
    <div style="margin-bottom:.75rem;">
        <label for="tags"><b>Tags</b> <small>(comma separated)</small></label><br/>
        <input id="tags" name="tags_q" value="{{ tags_csv or '' }}" placeholder="policy, hr, finance" style="width:60%;">
        {% if tag_options and not tags_csv %}
            <div style="margin-top:.25rem;"><small>Available: {{ ', '.join(tag_options) }}</small></div>
        {% endif %}
    </div>
  
    <label>Your question</label>
    <input name="q" type="text" placeholder="e.g., What is our refund policy?" />
    <button type="submit">Ask</button>
  </form>
</div>

<div class="card">
  <h2>Reset Chat</h2>
  <form method="post" action="/clear_history" onsubmit="return confirm('Clear your chat history?');">
    <button type="submit" class="secondary">Clear Chat History</button>
    <span class="muted">This will remove your conversation history for this session.</span>
  </form>
</div>

{% if msg %}
  <div class="card status {{ 'success' if ok else 'error' }}">
    <strong>{{ 'OK' if ok else 'Error' }}:</strong> {{ msg }}
  </div>
{% endif %}

{% if answer %}
  <div class="card">
    <h3>Answer</h3>
    <div>{{ answer|safe }}</div>
  </div>
    {% if sources %}
    <div style="margin-top:1rem;">
        <b>Sources</b>
        <ul>
        {% for s in sources %}
            <li>
                <details>
                    <summary>
                        {{ s.filename }}
                        {% if s.uploaded %} <small>({{ s.uploaded }}){% endif %}</small>
                        {% if s.ext %} <small>({{ s.ext }}){% endif %}</small>
                        {% if s.tags %} <small>({{ s.tags }}){% endif %}</small>
                        {% if s.hybrid is not none %} <small>(hybrid: {{ '%.2f'|format(s.hybrid) }}){% endif %}</small>
                        {% if s.rerank is not none %} <small>(rerank: {{ '%.2f'|format(s.rerank) }}){% endif %}</small>
                    </summary>
                    <pre style="white-space:pre-wrap; background:#f7f7f7; padding:.5rem; border-radius:6px;">{{ s.snippet }}</pre>
                </details>
            </li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
{% endif %}

<footer class="muted">
  <p>Built with Flask + SentenceTransformers + scikit-learn + OpenAI. Overlap keeps context across chunk boundaries.</p>
</footer>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    ext_options = get_ext_options()
    try:
        res = collection.get()
        total = len(res.get("ids", []))
        files = set(m["source"] for m in res.get("metadatas", []))
        msg = f"Collection has {total} chunks from {len(files)} files. Files are {', '.join(files) if files else 'None'}."
        ok = True
    except Exception as ex:
        msg, ok = f"Error accessing collection: {ex}", False

    return render_template_string(HTML,
                                  msg=msg,
                                  ok=ok,
                                  model_name=EMBED_MODEL_NAME,
                                  answer=None,
                                  sources=None,
                                  index_file="docs",
                                  ext_options=ext_options,
                                  selected_ext=None
                                  )

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    tags = request.form.get("tags", "").strip()
    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    saved = []
    for f in files:
        if not f or f.filename == "":
            continue
        file_name = secure_filename(f.filename)
        if not allowed(file_name):
            continue
        path = os.path.join(UPLOAD_DIR, file_name)
        f.save(path)
        #save tags
        if tags_list:
            with open(f"{path}.tags", "w", encoding="utf-8") as tf:
                tf.write(",".join(tags_list))
        saved.append(file_name)
    msg = f"Uploaded {len(saved)} files: {','.join(saved) or 'None'}"

    return render_template_string(HTML, msg=msg, ok=True, answer=None, sources=None,
                                  model_name=EMBED_MODEL_NAME, index_file="docs")

@app.route("/ingest", methods=["POST"])
def ingest_route():
    val = ingest(UPLOAD_DIR)
    _, err = val
    if err:
        return render_template_string(HTML, msg=err, ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs")
    else:
        return render_template_string(HTML, msg="Ingestion complete.", ok=True, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs")

@app.route("/chat", methods=["POST"])
def chat_route():
    query = request.form.get('q', '').strip()
    selected_exts = request.form.getlist("ext")
    ext_options = get_ext_options()

    if not query:
        return render_template_string(HTML,
                                      msg="Please enter a question.",
                                      ok=False,answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs",
                                      ext_options=ext_options,
                                      selected_ext=selected_exts
                                      )

    where = build_where(request.form)

    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        SESSIONS[sid] = deque(maxlen=2 * MAX_HISTORY)
    else:
        if sid not in SESSIONS:
            SESSIONS[sid] = deque(maxlen=2 * MAX_HISTORY)

    add_session_history(sid, "user", query)

    ctx, err = retrieve(query, TOP_K, where=where, use_reranker=USE_RERANKER, use_hybrid=USE_HYBRID)
    if err:
        add_session_history(sid, "assistant", f"Error retrieving context: {err}")
        return render_template_string(HTML,
                                      msg=err, ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs",
                                      ext_options=ext_options,
                                      selected_ext=selected_exts
                                      )
    elif not ctx:
        add_session_history(sid, "assistant", "No relevant context found.")
        return render_template_string(HTML,
                                      msg="No answer could be found since no relevant context was retrieved.", ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs",
                                      ext_options=ext_options,
                                      selected_ext=selected_exts
                                      )
    
    # filter by tags
    tags_csv = request.form.get("tags_q", "").strip().lower() or ""
    tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
    if tags:
        ctx = [item for item in ctx if item.get("tags") and
               any(tag in item.get("tags", "").lower() for tag in tags)]

    # Build prompt also include history from session if any
    system, user = build_prompt(query, ctx)
    if not system or not user:
        add_session_history(sid, "assistant", "Error building prompt.")
        return render_template_string(HTML, msg="Error building prompt.", ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs",
                                      ext_options=ext_options,
                                      selected_ext=selected_exts
                                      )
    # add history to messages in llm
    recent_history = get_session_history(sid, n=MAX_HISTORY)
    messages = [{"role": "system", "content": system}] + recent_history + [{"role": "user", "content": user}]
    print("--- Messages to LLM ---")
    print("SID:", sid)
    for h in recent_history:
        print(f"role: {h['role']}, content: {h['content'][:300]}")
    print("--------------------------")

    try:
        resp = openAI_Client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )

        answer = resp.choices[0].message.content
    except Exception as e:
        add_session_history(sid, "assistant", f"OpenAI API error: {e}")
        return render_template_string(HTML, msg=f"OpenAI API error: {e}", ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs",
                                      ext_options=ext_options,
                                      selected_ext=selected_exts
                                      )

    add_session_history(sid, "assistant", answer)

    sources = [
        {"filename": os.path.basename(hit["source"]),
         "uploaded": hit.get("uploaded_at", "unknown"),
         "ext": hit.get("ext", ""),
         "tags": hit.get("tags", ""),
         "hybrid": hit.get("hybrid", ""),
         "rerank": hit.get("rerank", ""),
         "snippet": hit["chunk"] if len(hit["chunk"]) < 300 else hit["chunk"][:300] + "..."}
         for hit in ctx]
    
    return render_template_string(HTML, msg=None, ok=True, answer=answer,
                                  sources=sources,
                                  model_name=EMBED_MODEL_NAME, index_file="docs",
                                  ext_options=ext_options,
                                  selected_ext=selected_exts
                                  )

@app.route("/clear_collection", methods=["POST"])
def clear_collection():
    try:
        global collection, chroma_client
        collection.delete(where={"source": {"$ne": ""}})
        import shutil
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            os.makedirs(CHROMA_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fun
        )
        msg = "Collection and ChromaDB cleared."
        ok = True
    except Exception as e:
        return render_template_string(HTML, msg=f"Error clearing collection: {e}", ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file="docs")
    
    ext_options = get_ext_options()
    return render_template_string(HTML, msg=msg, ok=ok, answer=None, sources=None,
                                  model_name=EMBED_MODEL_NAME, ext_options=ext_options, index_file="docs")

@app.route("/files")
def list_files():
    files = []
    for fname in os.listdir(UPLOAD_DIR):
        if allowed(fname):
            path = os.path.join(UPLOAD_DIR, fname)
            tags = ""
            tag_path = path + ".tags"
            if os.path.exists(tag_path):
                with open(tag_path, "r", encoding="utf-8") as tf:
                    tags = tf.read().strip()
            files.append({"name": fname, "size": os.path.getsize(path), "tags": tags})
    return render_template_string("""
                                  <h2>Uploaded Files</h2>
                                  <ul>
                                  {% for f in files %}
                                  <li>{{ f.name }} ({{ f.size }} bytes) [tags: {{ f.tags }}]
                                  <a href="/delete_file?file={{ f.name }}">Delete</a>
                                  </li>
                                  {% endfor %}
                                  </ul>
                                  """,
                                  files=files)

@app.route("/delete_file")
def delete_file():
    fname = request.args.get("file")
    if not fname or not allowed(fname):
        return redirect(url_for("list_files"))
    
    path = os.path.join(UPLOAD_DIR, fname)
    Path(path).unlink(missing_ok=True)
    tag_path = path + ".tags"
    Path(tag_path).unlink(missing_ok=True)

    return redirect(url_for("list_files"))

@app.route("/health")
def health():
    try:
        data = collection.get(include=["documents", "metadatas", "embeddings"])
        info_full = [{"id": id, "document": doc[:200], "metadata": meta}
                     for id, doc, meta in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", []))] if data else []
        return jsonify({"status": "ok", "content": info_full}), 200
    except Exception as e:
        return jsonify({"status": "error", "err": str(e)}), 500
    
@app.route("/clear_history", methods=["POST"])
def clear_history():
    sid = session.get("sid", "")
    if sid and sid in SESSIONS:
        SESSIONS[sid].clear()

    return redirect(url_for("home"))

# set session to be permanent (14 days)
app.permanent_session_lifetime = timedelta(days=14)
@app.before_request
def _make_session_permanent():
    session.permanent = True

if __name__ == "__main__":
    app.run(debug=True, port=5001)